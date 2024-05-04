import argparse
import sys


def parse_training_args():
    ### Argument and global variables
    parser = argparse.ArgumentParser('Interface for TAGON experiments on link predictions')
    parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
    parser.add_argument('--bs', type=int, default=200, help='batch_size')
    parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
    parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs')
    parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
    parser.add_argument('--reg', type=float, default=0.1, help='regularization')
    parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
    parser.add_argument('--node_dim', type=int, default=20, help='Dimentions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=20, help='Dimentions of the time embedding')
    parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
    parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod', help='use dot product attention or mapping based')
    parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty', 'disentangle'], help='how to use time information', default='time')
    parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
    parser.add_argument('--new_node', action='store_true', help='model new node')
    parser.add_argument('--samplerate', type=float, default=1, help='samplerate for each user')
    parser.add_argument('--popnegsample', action='store_true', help='use popularity based negative sampling')
    parser.add_argument('--timepopnegsample', action='store_true', help='use timely popularity based negative sampling')
    parser.add_argument('--negsampleeval', type=int, default=1000, help='number of negative sampling evaluation, -1 for all')
    parser.add_argument('--disencomponents', type=int, default=1, help='number of various time encoding')
    parser.add_argument('--test_mode', action='store_true', help='test mode for debugging', default=None)
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')

    try:
        args = parser.parse_args()
        return args
    except:
        parser.print_help()
        sys.exit(0)
        

def parse_inference_args():
    ### Argument and global variables
    parser = argparse.ArgumentParser('Interface for TAGON experiments on link predictions')
    parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
    parser.add_argument('--bs', type=int, default=200, help='batch_size')
    parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
    parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs')
    parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
    parser.add_argument('--reg', type=float, default=0.1, help='regularization')
    parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
    parser.add_argument('--node_dim', type=int, default=20, help='Dimentions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=20, help='Dimentions of the time embedding')
    parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
    parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod', help='use dot product attention or mapping based')
    parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty', 'disentangle'], help='how to use time information', default='time')
    parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
    parser.add_argument('--new_node', action='store_true', help='model new node')
    parser.add_argument('--samplerate', type=float, default=1, help='samplerate for each user')
    parser.add_argument('--popnegsample', action='store_true', help='use popularity based negative sampling')
    parser.add_argument('--timepopnegsample', action='store_true', help='use timely popularity based negative sampling')
    parser.add_argument('--negsampleeval', type=int, default=1000, help='number of negative sampling evaluation, -1 for all')
    parser.add_argument('--disencomponents', type=int, default=1, help='number of various time encoding')

    try:
        args = parser.parse_args()
        return args
    except:
        parser.print_help()
        sys.exit(0)

