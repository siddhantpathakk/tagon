import argparse
import sys

def node_classify_parser():
    ### Argument and global variables
    parser = argparse.ArgumentParser('Interface for TGAT experiments on node classification')
    parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
    parser.add_argument('--bs', type=int, default=30, help='batch_size')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--n_degree', type=int, default=50, help='number of neighbors to sample')
    parser.add_argument('--n_neg', type=int, default=1)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--n_epoch', type=int, default=15, help='number of epochs')
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--tune', action='store_true', help='parameters tunning mode, use train-test split on training data only.')
    parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
    parser.add_argument('--node_dim', type=int, default=None, help='Dimentions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=None, help='Dimentions of the time embedding')
    parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
    parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod')
    parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')

    parser.add_argument('--new_node', action='store_true', help='model new node')
    parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')

    try:
        args = parser.parse_args()
        return args
    except:
        parser.print_help()
        sys.exit(0)
        

def link_predict_parser():
    ### Argument and global variables
    parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
    parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
    parser.add_argument('--bs', type=int, default=200, help='batch_size')
    parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
    parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
    parser.add_argument('--node_dim', type=int, default=100, help='Dimentions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=100, help='Dimentions of the time embedding')
    parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
    parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod', help='use dot product attention or mapping based')
    parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')
    parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')

    try:
        args = parser.parse_args()
        return args
    except:
        parser.print_help()
        sys.exit(0)
        