import argparse
from utils import *
from dataset import *
from model import *
import os


if __name__ == '__main__':
    current_dir = os.getcwd()  # Get the current directory
    parent_dir = os.path.dirname(current_dir)  # Get the upper-level directory
    parser = argparse.ArgumentParser(description='Cmapss Dataset With Pytorch')
    # To evaluate the trained models on different sub-datasets,
    # please change the following two options
    parser.add_argument('--sub-dataset', type=str, default='FD003', help='FD001/2/3/4')
    parser.add_argument('--smooth-rate', type=int, default=30)
    # Below is the default settings
    parser.add_argument('--use-exponential-smoothing', default=True)
    parser.add_argument('--sequence-len', type=int, default=30)
    parser.add_argument('--feature-num', type=int, default=14)
    parser.add_argument('--dataset-root', type=str,
                        default=parent_dir + '/CMAPSSData/',
                        help='The dir of CMAPSS dataset1')
    parser.add_argument('--max-rul', type=int, default=125, help='piece-wise RUL')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--step-size', type=int, default=10, help='interval of learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='ratio of learning rate scheduler')
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=8, help='Early Stop Patience')
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--log-path', type=str, default='..\\_trials\\', help='The dir of logging path')
    args = parser.parse_args()

    model = torch.load(parent_dir + '/trials/' + 'model_FD003.pkl')
    model_type = type(model).__name__
    model.to(torch.device('cuda'))
    train_loader, valid_loader, test_loader, test_loader_last, \
        num_test_windows, train_visualize, engine_id = get_dataloader(
            dir_path=args.dataset_root,
            sub_dataset=args.sub_dataset,
            max_rul=args.max_rul,
            seq_length=args.sequence_len,
            batch_size=args.batch_size,
            use_exponential_smoothing=args.use_exponential_smoothing,
            smooth_rate=args.smooth_rate)

    rmse_final, score = evaluate(
            model, num_test_windows, test_loader, args.max_rul,
            device=torch.device('cuda') if not args.no_cuda else torch.device('cpu'))

    print('rmse_final:{}, score:{}'.format(rmse_final, score))
