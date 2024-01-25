import argparse
import torch
import os
from dataloader.dataloader import DataCollector
from trainer import Trainer

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='/home/FYP/siddhant005/fyp/code/data/processed/ml-100k/')
    parser.add_argument('--out_path', type=str, default='/home/FYP/siddhant005/fyp/code/src/logs/tmp/')
    parser.add_argument('--L', type=int, default=11)
    parser.add_argument('--H', type=int, default=2)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lambda_val', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=4e-3)
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--sets_of_neg_samples', type=int, default=50)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--conv_layer_num', type=int, default=3)
    parser.add_argument('--adj_dropout', type=int, default=0)
    parser.add_argument('--num_bases', type=int, default=2)
    parser.add_argument('--verbose', type=bool, default=True)
    return parser.parse_args()

if __name__ == '__main__':

    config = parse_opt()
    print(f'[STATUS] Running train.py with arguments: {config}')
    print('=='*55)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device: {device} ')
    print('=='*55)
    
    datacollector = DataCollector(config)
    uid2locid_time,locid2detail,node_num,relation_num,uid_list_,user_np,seq_train,seq_test,test_set,u2v,u2vc,v2u,v2vc = datacollector.main(save1=True,save2=True)
    
    print(f'[STATUS] Data preprocessing completed')
    if config.verbose:
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
    print('=='*55)
    
    modeling = Trainer(config,node_num,relation_num,u2v,u2vc,v2u,v2vc,device)
    
    train_part = [user_np,seq_train]
    test_part = [seq_test,test_set,uid_list_,uid2locid_time]
    
    print(f'[STATUS] Commencing training for {config.epoch_num} epochs\n')
    modeling.train(train_part,test_part)
    print('=='*55)
    print('[STATUS] Training completed successfully')
    
