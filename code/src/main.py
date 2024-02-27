import json
import argparse
import os
import torch
from .dataloader.collector import DataCollector
from .trainer import Trainer
from .utils.metric import seed_everything

def str2bool(v):
    """
    Convert string to boolean

    Args:
        v (str): string to be converted to boolean

    Raises:
        argparse.ArgumentTypeError: Boolean value expected.

    Returns:
        bool: boolean value
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_opt():
    """
    Parse command line arguments

    Returns:
        argparse.Namespace: command line arguments
    """
    parser = argparse.ArgumentParser(description='Trainer for FYP GNN')
    
    # directory based parameters
    parser.add_argument('--dataset', type=str, default='ml100k', help='ml100k, ml1m')
    parser.add_argument('--out_path', type=str, default='/home/FYP/siddhant005/fyp/code/src/logs/tmp/', help='output path')
    parser.add_argument('--log_path', type=str, default='/home/FYP/siddhant005/fyp/code/src/logs/', help='log path')
    parser.add_argument('--processed', type=str2bool, nargs='?', const=True, default=True, help='whether to use processed data or not')
    
    # data based parameters
    parser.add_argument('--L', type=int, default=11, help='length of sequence')
    parser.add_argument('--H', type=int, default=3, help='length of history')
    parser.add_argument('--topk', type=int, default=20, help='top k items to recommend')
    
    # model training based parameters
    parser.add_argument('--epoch_num', type=int, default=500, help='number of epochs') # approx 300-500
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate') # {1e-3, 1e-4}
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization') # {1e-1 ... 1e-5}
    
    # negative sampling parameters
    parser.add_argument('--neg_samples', type=int, default=2, help='number of negative samples')
    parser.add_argument('--sets_of_neg_samples', type=int, default=50, help='number of sets of negative samples')
    
    # GCN based parameters
    parser.add_argument('--dim', type=int, default=32, help='dimension of hidden layers')
    parser.add_argument('--conv_layer_num', type=int, default=2, help='number of long term GCN layers')
    parser.add_argument('--short_term_conv_layer_num', type=int, default=3, help='number of short term GCN layers')
    parser.add_argument('--num_bases', type=int, default=3, help='number of bases')
    parser.add_argument('--adj_dropout', type=int, default=0, help='adjacency dropout')
    parser.add_argument('--lambda_val', type=float, default=0.5, help='lambda value for combination of embeddings')

    # TSAL based parameters
    parser.add_argument('--TSAL_head_num', type=int, default=2, help='number of heads for Temporal Sequential Attn Layer')
    parser.add_argument('--TSAL_attn_drop', type=float, default=0.0, help='attention dropout for Temporal Sequential Attn Layer')
    
    # Cross attention based parameters
    parser.add_argument('--cross_attn_head_num', type=int, default=2, help = 'number of heads for Cross Attn Layer')
    parser.add_argument('--cross_attn_drop', type=float, default=0.0, help='attention dropout for Cross Attn Layer')
    
    # other parameters
    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    parser.add_argument('--verbose', type=str2bool, nargs='?', const=True, default=False, help='whether to print verbose logs or not')
    parser.add_argument('--debug', type=str2bool, nargs='?', const=True, default=False, help='whether to run in debug mode or not')
    parser.add_argument('--plot', type=str2bool, nargs='?', const=True, default=True, help='whether to plot metrics or not')
    parser.add_argument('--block_backprop', type=str2bool, nargs='?', const=True, default=False, help='whether to block backpropagation or not')
    return parser.parse_args()

if __name__ == '__main__':

    config = parse_opt()
    cwd = os.getcwd()
    
    if config.dataset == 'ml100k':
        config.batch_size = 128
        if config.processed:
            config.file_path = cwd + '/data/processed_timesteps/ml-100k/'
        else:
            config.file_path = cwd + '/data/processed/ml-100k/'
    
    elif config.dataset == 'ml1m':
        config.batch_size = 1024
        if config.processed:
            config.file_path = cwd + '/data/processed_timesteps/ml-1m/'
        else:
            config.file_path = cwd + '/data/processed/ml-1m/'
    else:
        raise NotImplementedError(f'[ERROR] Dataset {config.dataset} not implemented')

    config.log_path = config.log_path + "train/" + config.dataset + '/'
    seed_everything(config.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device: {device} ')
    
    if config.debug:
        print(f'[INFO] Running in debug mode')
        if config.dataset == 'ml100k':
            config.epoch_num = 30
        elif config.dataset == 'ml1m':
            config.epoch_num = 10
        
        import pprint
        pprint.pprint(config.__dict__)
        
    with open(config.out_path+'commandline_args.txt', 'w', encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2)
    
    datacollector = DataCollector(config)
    train_part, test_part, edges, node_num, relation_num = datacollector.main()
    
    if config.verbose and config.debug is False:
        print(f'[STATUS] Running train.py with arguments: {config}')
        print('=='*100)
        print('=='*100)
        print('[STATUS] Data preprocessing completed')
        print(f'[INFO] node_num: {node_num} ')
        print(f'[INFO] relation_num: {relation_num} ')
        
    print('=='*100)
    print(f'[STATUS] Commencing training for {config.epoch_num} epochs\n')
    
    trainer = Trainer(config=config,
                      node_num=node_num,
                      relation_num=relation_num, 
                      u2v=edges[0], u2vc=edges[1], v2u=edges[2], v2vc=edges[3],
                      device=device)

    trainer.train(train_part,test_part)