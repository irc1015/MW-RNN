import os
import gzip
import random
import numpy as np
import torch
import torch.utils.data as data

def load_dataset_npy(root, filename):
    path = os.path.join(root, filename)
    dataset = np.load(path)
    return dataset


class Radar(data.Dataset):
    def __init__(self, root, is_train=True, n_frames_input=5, n_frames_output=5):
        super(Radar, self).__init__()

        self.dataset = None
        if is_train:
            filename = 'radar_train.npy'
            self.dataset = load_dataset_npy(root, filename)
        else:
            filename = 'radar_test.npy'
            self.dataset = load_dataset_npy(root, filename)

        self.length = int(1e4) if self.dataset is None else self.dataset.shape[0]

        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output

        self.mean = 0
        self.std = 1

    def __getitem__(self, idx):
        images = self.dataset[idx, :, :, :, :]  # [length, height, width, channel]
        images = images.transpose(0, 3, 1, 2)  # [length, channel, height, width]

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:self.n_frames_total]
        else:
            output = []

        input = torch.from_numpy(input).contiguous().float()
        output = torch.from_numpy(output).contiguous().float()

        return input, output

    def __len__(self):
        return self.length


def load_data(batch_size, data_root, num_workers):
    train_set = Radar(root=data_root, is_train=True, n_frames_input=5, n_frames_output=5)
    test_set = Radar(root=data_root, is_train=False, n_frames_input=5, n_frames_output=5)

    dataloader_train = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=num_workers,
                                                   drop_last=True)
    dataloader_validation = torch.utils.data.DataLoader(test_set,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        pin_memory=True,
                                                        num_workers=num_workers,
                                                        drop_last=True)
    dataloader_test = torch.utils.data.DataLoader(test_set,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=num_workers,
                                                  drop_last=True)

    mean, std = 0, 1
    return dataloader_train, dataloader_validation, dataloader_test, mean, std