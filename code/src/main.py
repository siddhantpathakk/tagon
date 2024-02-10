import argparse
import torch
import os
from dataloader.dataloader import DataCollector
from trainer import Trainer
import json
import numpy as np
import random
from utils.metric import seed_everything

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_opt():
    parser = argparse.ArgumentParser(description='Trainer for FYP GNN')
    
    # directory based parameters
    parser.add_argument('--dataset', type=str, default='ml100k')
    parser.add_argument('--out_path', type=str, default='/home/FYP/siddhant005/fyp/code/src/logs/tmp/')
    parser.add_argument('--processed', type=str2bool, nargs='?', const=True, default=False )
    # data based parameters
    parser.add_argument('--L', type=int, default=11)
    parser.add_argument('--H', type=int, default=3)
    parser.add_argument('--topk', type=int, default=20)
    
    # model training based parameters
    parser.add_argument('--epoch_num', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--l2', type=float, default=1e-4)
    
    # negative sampling parameters
    parser.add_argument('--neg_samples', type=int, default=2)
    parser.add_argument('--sets_of_neg_samples', type=int, default=50)
    
    # GCN based parameters
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--conv_layer_num', type=int, default=2)
    parser.add_argument('--short_term_conv_layer_num', type=int, default=3)
    parser.add_argument('--num_bases', type=int, default=3)
    parser.add_argument('--adj_dropout', type=int, default=0)
    parser.add_argument('--lambda_val', type=float, default=0.5)

    # TSAL based parameters
    parser.add_argument('--TSAL_head_num', type=int, default=2)
    parser.add_argument('--TSAL_attn_drop', type=float, default=0.0)
    
    # Cross attention based parameters
    parser.add_argument('--cross_attn_head_num', type=int, default=2)
    parser.add_argument('--cross_attn_drop', type=float, default=0.0)
    
    # other parameters
    parser.add_argument('--verbose', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--debug', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--seed', type=int, default=123456)
    return parser.parse_args()

if __name__ == '__main__':

    config = parse_opt()
    
    if config.dataset == 'ml100k':
        config.batch_size = 128
        if config.processed:
            config.file_path = '/home/FYP/siddhant005/fyp/code/data/processed_timesteps/ml-100k/'
        else:
            config.file_path = '/home/FYP/siddhant005/fyp/code/data/processed/ml-100k/'
    elif config.dataset == 'ml1m':
        config.file_path = '/home/FYP/siddhant005/fyp/code/data/processed/ml-1m/'
        config.batch_size = 512
    else:
        raise NotImplementedError(f'[ERROR] Dataset {config.dataset} not implemented')

    seed_everything(config.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if config.debug:
        config.epoch_num = 10
        # config.verbose = True
        print(config.__dict__)
        
    with open(config.out_path+'commandline_args.txt', 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    datacollector = DataCollector(config)
    uid2locid_time,locid2detail,node_num,relation_num,uid_list_,user_np,seq_train,seq_test,test_set,u2v,u2vc,v2u,v2vc = datacollector.main(save1=True,save2=True)
    
    if config.verbose:
        print(f'[STATUS] Running train.py with arguments: {config}')
        print('=='*100)
        print(f'[INFO] device: {device} ')
        print('=='*100)
        print(f'[STATUS] Data preprocessing completed')
        print(f'[INFO] uid2locid_time: {len(uid2locid_time)} ')
        print(f'[INFO] locid2detail: {len(locid2detail)} ')
        print(f'[INFO] node_num: {node_num} ')
        print(f'[INFO] relation_num: {relation_num} ')
        print(f'[INFO] user_num: {len(uid_list_)} ')
        print(f'[INFO] train_num: {user_np.shape[0]} ')
        print(f'[INFO] test_num: {seq_test.shape[0]} ')
        print(f'[INFO] train_set: {seq_train.shape} ')
        print(f'[INFO] test_set: {seq_test.shape} ')
        print(f'[INFO] uid_list_: {len(uid_list_)} ')
        print(f'[INFO] u2v: {len(u2v)} ')
        print(f'[INFO] u2vc: {len(u2vc)} ')
        print(f'[INFO] v2u: {len(v2u)} ')
        print(f'[INFO] v2vc: {len(v2vc)} ')
        print('=='*100)
        print(f'[STATUS] Commencing training for {config.epoch_num} epochs\n')
    
    train_part = [user_np,seq_train]
    test_part = [seq_test,test_set,uid_list_,uid2locid_time]
    
    trainer = Trainer(config,node_num,relation_num,u2v,u2vc,v2u,v2vc,device)
    
    trainer.train(train_part,test_part)