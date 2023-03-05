import os
import logging

import numpy as np
import torch
from tqdm import tqdm
import shutil

from preprocess import set_seed, reshape_patch, reshape_patch_back

from model import Model
import dataset_loader
from recorder import Recorder
from metrics import metric
from schedule import reserve_schedule_sampling


def print_log(message):
    print(message)
    logging.info(message)


def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for key, value in configs.items():
        message += '\n' + key + ': \t' + str(value) + '\t'
    return message


def check_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Trainer:
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args

        self._preparation()
        print_log(output_namespace(self.args))

        self._get_data()
        self._select_optimizer()
        self._select_criterion()

    def _preparation(self):
        is_training = self.args.is_training
        pretrained_model = self.args.pretrained_model

        seed = self.args.seed

        if is_training:
            set_seed(seed)
            if pretrained_model == 0:
                check_path(self.args.save_dir)
                check_path(self.args.best_dir)
        else:
            check_path(self.args.res_dir)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename='log.log', filemode='a', format='%(asctime)s - %(message)s')

        self._build_model()

    def _build_model(self):
        args = self.args
        self.model = Model(tuple(args.in_shape), args.num_layers, args.num_hidden, args.filter_size, args.stride,
                           args.layer_norm, args.patch_size).to(args.device)
        print('Model parameters: ', count_parameters(self.model))

    def _get_data(self):
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = dataset_loader.load_data(
            **config)
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        return self.optimizer

    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()

    def _save(self, epoch):
        stats = self.model.state_dict()
        checkpoint_path = os.path.join(self.args.save_dir, str(epoch + 1) + '_model.pth')
        torch.save(stats, checkpoint_path)
        print('Save Common Model to {}'.format(checkpoint_path))

    def train_on_batch(self, input_tensor, target_tensor, epoch):
        args = self.args
        self.optimizer.zero_grad()
        input_length = input_tensor.shape[1]
        target_length = target_tensor.shape[1]
        loss = 0
        decouple_loss = []

        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []

        B, L, C, H, W = input_tensor.shape

        for i in range(args.num_layers):
            zeros = torch.zeros([args.batch_size, args.num_hidden, H, W]).to(args.device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros([args.batch_size, args.num_hidden, H, W]).to(args.device)

        real_input_flag = reserve_schedule_sampling(args.in_shape, args.batch_size, args.input_length, args.total_length,
                                                 args.r_sampling_step_1, args.r_sampling_step_2, args.r_exp_alpha, epoch,
                                                 args.patch_size)

        real_input_flag = torch.FloatTensor(real_input_flag).to(args.device)

        first_input = input_tensor[:, 0, :, :, :]
        for ei in range(input_length - 1):
            out_dec, delta_c_list, delta_m_list, memory, h_t, c_t = self.model(first_input, delta_c_list, delta_c_list,
                                                                               memory, h_t, c_t)
            loss += self.criterion(out_dec, input_tensor[:, ei+1, :, :, :])
            for i in range(args.num_layers):
                decouple_loss.append(
                    torch.mean(torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2)))
                )
            first_input = real_input_flag[:, ei] * input_tensor[:, ei + 1] + (1 - real_input_flag[:, ei]) * out_dec

        next_input = input_tensor[:, -1, :, :, :]

        for di in range(target_length):
            out_dec, delta_c_list, delta_m_list, memory, h_t, c_t = self.model(next_input, delta_c_list, delta_m_list,
                                                                               memory, h_t,  c_t)
            target = target_tensor[:, di, :, :, :]
            loss += self.criterion(out_dec, target)
            for i in range(args.num_layers):
                decouple_loss.append(
                    torch.mean(torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2)))
                )
            if di + (input_length -1)  < target_length - 2 - 1:
                next_input = real_input_flag[:, di+(input_length-1)] * target_tensor[:, di] + (1 - real_input_flag[:, di+(input_length-1)]) * out_dec

        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        loss = loss + decouple_loss * args.decouple_beta

        loss.backward()
        self.optimizer.step()
        return loss

    def trainiters(self):
        args = self.args
        record = Recorder(verbose=True)

        for epoch in range(args.epochs):
            train_loss = []

            self.model.train()
            train_pbar = tqdm(self.train_loader)

            for input_tensor, target_tensor in train_pbar:
                input_tensor = reshape_patch(input_tensor, args.patch_size)
                target_tensor = reshape_patch(target_tensor,args.patch_size)
                input_tensor = input_tensor.to(self.args.device)
                target_tensor = target_tensor.to(self.args.device)

                loss = self.train_on_batch(input_tensor=input_tensor,
                                           target_tensor=target_tensor,
                                           epoch=epoch)
                train_loss.append(loss.item())
                train_pbar.set_description('Train Loss {:.4f}'.format(loss.item()))

            train_loss = np.average(train_loss)

            if (epoch + 1) % args.log_step == 0:
                with torch.no_grad():
                    vali_loss = self.evaluate(self.vali_loader)
                    if (epoch + 1) % (args.log_step * 200) == 0:
                        self._save(epoch=epoch)
                print_log(
                    'Epoch: {} | Train Loss: {:.4f} Vali Loss: {:.4f}\n'.format((epoch + 1), train_loss, vali_loss))
                record(val_loss=vali_loss, model=self.model, path=self.args.best_dir)

        best_model_path = self.args.best_dir + '/' + 'best_checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def evaluate(self, vali_loader):
        args = self.args
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []

        vali_pbar = tqdm(vali_loader)
        for input_tensor, target_tensor in vali_pbar:
            input_tensor = reshape_patch(input_tensor, args.patch_size)
            target_tensor = reshape_patch(target_tensor, args.patch_size)
            input_tensor = input_tensor.to(args.device)
            target_tensor = target_tensor.to(args.device)
            input_length = input_tensor.shape[1]
            target_length = target_tensor.shape[1]

            h_t = []
            c_t = []
            delta_c_list = []
            delta_m_list = []

            B, L, C, H, W = input_tensor.shape

            for i in range(args.num_layers):
                zeros = torch.zeros([args.batch_size, args.num_hidden, H, W]).to(args.device)
                h_t.append(zeros)
                c_t.append(zeros)
                delta_c_list.append(zeros)
                delta_m_list.append(zeros)


            memory = torch.zeros([args.batch_size, args.num_hidden, H, W]).to(args.device)

            for ei in range(input_length - 1):  # frame1 ~ frame9
                out_dec, delta_c_list, delta_m_list, memory, h_t, c_t = self.model(input_tensor[:, ei, :, :, :],
                                                                                   delta_c_list, delta_m_list, memory,
                                                                                   h_t, c_t)

            next_input = input_tensor[:, -1, :, :, :]
            pred_batch = []

            loss = 0

            for di in range(target_length):
                out_dec, delta_c_list, delta_m_list, memory, h_t, c_t = self.model(next_input, delta_c_list, delta_c_list,
                                                                                   memory, h_t, c_t)
                target = target_tensor[:, di, :, :, :]

                loss += self.criterion(out_dec, target)
                next_input = out_dec

                pred_batch.append(out_dec.detach().cpu().numpy())

            pred_batch = np.stack(pred_batch)  # 10, batch_size, 1, 64, 64
            pred_batch = pred_batch.swapaxes(0, 1)  # batch_size, 10, 1, 64, 64
            target = target_tensor.cpu().numpy()

            pred_batch = reshape_patch_back(pred_batch, args.patch_size)
            target = reshape_patch_back(target, args.patch_size)

            preds_lst.append(pred_batch)
            trues_lst.append(target)

            vali_pbar.set_description('Vali Loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        mse, mae, ssim, psnr = metric(pred=preds, true=trues,
                                      mean=self.vali_loader.dataset.mean,
                                      std=self.vali_loader.dataset.std,
                                      return_ssim_psnr=True)
        print_log('Vali MSE:{:.4f}, MAE:{:.4f}, SSIM:{:.4f}, PSNR:{:.4f}'.format(mse, mae, ssim, psnr))
        self.model.train()
        return total_loss

    def test(self, args):
        best_checkpoint = os.path.join(args.best_dir, 'best_checkpoint.pth')
        stats = torch.load(best_checkpoint)
        self.model.load_state_dict(stats)

        self.model.eval()
        preds_lst, trues_lst = [], []

        test_pbar = tqdm(self.test_loader)
        for input_tensor, target_tensor in test_pbar:
            input_tensor = reshape_patch(input_tensor, args.patch_size)
            target_tensor = reshape_patch(target_tensor, args.patch_size)
            input_tensor = input_tensor.to(args.device)
            target_tensor = target_tensor.to(args.device)
            input_length = input_tensor.shape[1]
            target_length = target_tensor.shape[1]

            h_t = []
            c_t = []
            delta_c_list = []
            delta_m_list = []

            B, L, C, H, W = input_tensor.shape

            for i in range(args.num_layers):
                zeros = torch.zeros([args.batch_size, args.num_hidden, H, W]).to(args.device)
                h_t.append(zeros)
                c_t.append(zeros)
                delta_c_list.append(zeros)
                delta_m_list.append(zeros)

            memory = torch.zeros([args.batch_size, args.num_hidden, H, W]).to(args.device)

            for ei in range(input_length - 1):  # frame1 ~ frame9
                out_dec, delta_c_list, delta_m_list, memory, h_t, c_t = self.model(input_tensor[:, ei, :, :, :],
                                                                                   delta_c_list, delta_m_list, memory,
                                                                                   h_t, c_t)

            next_input = input_tensor[:, -1, :, :, :]
            pred_batch = []

            for di in range(target_length):
                out_dec, delta_c_list, delta_m_list, memory, h_t, c_t = self.model(next_input, delta_c_list,
                                                                                   delta_c_list,
                                                                                   memory, h_t, c_t)
                target = target_tensor[:, di, :, :, :]

                next_input = out_dec

                pred_batch.append(out_dec.detach().cpu().numpy())

            pred_batch = np.stack(pred_batch)  # 10, batch_size, 1, 64, 64
            pred_batch = pred_batch.swapaxes(0, 1)  # batch_size, 10, 1, 64, 64
            target = target_tensor.cpu().numpy()

            pred_batch = reshape_patch_back(pred_batch, args.patch_size)
            target = reshape_patch_back(target, args.patch_size)

            preds_lst.append(pred_batch)
            trues_lst.append(target)

        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        mse, mae, ssim, psnr = metric(pred=preds, true=trues,
                                      mean=self.vali_loader.dataset.mean,
                                      std=self.vali_loader.dataset.std,
                                      return_ssim_psnr=True)
        print_log('Test MSE:{:.4f}, MAE:{:.4f}, SSIM:{:.4f}, PSNR:{:.4f}'.format(mse, mae, ssim, psnr))

