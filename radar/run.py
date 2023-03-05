import argparse
from trainer import Trainer

def create_parse():
    parser = argparse.ArgumentParser(description='beta_radar')

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--dataname', type=str, default='radar')
    parser.add_argument('--save_dir', type=str, default='checkpoints')  # save common checkpoint
    parser.add_argument('--res_dir', type=str, default='results')  # save result frames
    parser.add_argument('--best_dir', type=str, default='best')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--batch_size', default=10, type=int, help='Batch size')
    parser.add_argument('--input_length', default=5, type=int)
    parser.add_argument('--total_length', default=10, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--pretrained_model', type=int, default=0)

    parser.add_argument('--in_shape', default=[5, 1, 128, 128], type=int, nargs='*')
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--num_hidden', default=256, type=int)
    parser.add_argument('--filter_size', default=5, type=int)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--layer_norm', type=int, default=1)
    parser.add_argument('--decouple_beta', type=float, default=0.1)

    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')

    # reverse scheduled_sampling
    parser.add_argument('--r_sampling_step_1', type=float, default=100)
    parser.add_argument('--r_sampling_step_2', type=int, default=300)
    parser.add_argument('--r_exp_alpha', type=int, default=40)

    return parser


if __name__ == '__main__':
    args = create_parse().parse_args()

    experiment = Trainer(args)

    if args.is_training:
        experiment.trainiters()
    experiment.test(args)
