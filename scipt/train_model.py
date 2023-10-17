import datetime
import argparse
import os
from dataset import *
from model import *
from utils import *


if __name__ == '__main__':
    current_dir = os.getcwd()  # Get the current directory
    parent_dir = os.path.dirname(current_dir)  # Get the upper-level directory
    parser = argparse.ArgumentParser(description='Cmapss Dataset With Pytorch')

    parser.add_argument('--sequence-len', type=int, default=30)
    parser.add_argument('--feature-num', type=int, default=14)
    parser.add_argument('--dataset-root', type=str,
                        default=parent_dir + '/CMAPSSData/',
                        help='The dir of CMAPSS dataset1')
    parser.add_argument('--sub-dataset', type=str, default='FD004', help='FD001/2/3/4')
    parser.add_argument('--max-rul', type=int, default=125, help='piece-wise RUL')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--step-size', type=int, default=10, help='interval of learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='ratio of learning rate scheduler')
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=8, help='Early Stop Patience')
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--use-exponential-smoothing', default=True)
    parser.add_argument('--smooth-rate', type=int, default=40)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--save-model', type=str, default=False, help='save trained models')
    args = parser.parse_args()

    torch.manual_seed(28)

    train_loader, valid_loader, test_loader, test_loader_last, \
        num_test_windows, train_visualize, engine_id = get_dataloader(
            dir_path=args.dataset_root,
            sub_dataset=args.sub_dataset,
            max_rul=args.max_rul,
            seq_length=args.sequence_len,
            batch_size=args.batch_size,
            use_exponential_smoothing=args.use_exponential_smoothing,
            smooth_rate=args.smooth_rate)

    encoder = Seq2SeqEncoder(input_size=14, num_layers=2, num_hiddem=8)
    decoder = Seq2SeqDecoder(input_size=14, num_layers=2, num_hidden=8,
                             seq_len=30, attention_size=28)
    model = EncoderDecoder(encoder=encoder, decoder=decoder,
                           feature_attention_size=4)

    model_type = type(model).__name__

    criterion_train = torch.nn.MSELoss()
    criterion_eval = RMSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)

    train(
        model, train_loader, valid_loader,
        test_loader, args.max_epochs, optimizer,
        scheduler, criterion_train, criterion_eval,
        lines_list=[], patience=args.patience, max_rul=args.max_rul, num_test_windows=num_test_windows,
        device=torch.device('cuda') if not args.no_cuda else torch.device('cpu'))

    if args.save_model:
        torch.save(model, parent_dir+'/trials/'+'model.pkl')
