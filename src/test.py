import math
import logging
import time
import random
import sys
import argparse
import os
import json
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from model import TGRec
from data import Data
from evaluation.evaluation import *
from utils.utils import EarlyStopMonitor




# argument and global variables
parser = argparse.ArgumentParser('Interface for TGSRec experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=120, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
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
parser.add_argument('--timepopnegsample', action='store_false', help='use timely popularity based negative sampling')
parser.add_argument('--negsampleeval', type=int, default=1000, help='number of negative sampling evaluation, -1 for all')
parser.add_argument('--disencomponents', type=int, default=1, help='number of various time encoding')

args = parser.parse_args()

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = 0
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATASET = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim



# set up paths

best_checkpoint = {
    'ml-100k': 5,
    'Baby': 7,
    'Digital_Music': 10,
    'Toys_and_Games': 5,
    'Tools_and_Home_Improvement': 5,
}


# set up paths
pretrain_path = f'./saved_checkpoints/{args.data}/{args.data}-{str(best_checkpoint[args.data])}_TARGON.pt'
datetime_str = time.strftime("%Y%m%d-%H%M%S")

log_file_path = f'./log/test/{args.data}_TARGON_{datetime_str}.log'
log_format = f'%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s'
result_file_path = f'./log/test/results/TARGON_{args.data}_results.csv'
result_log_format = f'%(message)s'


# set random seed
SEED = 2024
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# set up logger
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('root')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_file_path)
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter(log_format))
logger.addHandler(fh)

result_logger = logging.getLogger('result')
result_logger.setLevel(logging.DEBUG)
rfh = logging.FileHandler(result_file_path)
rfh.setFormatter(logging.Formatter(result_log_format))
result_logger.addHandler(rfh)

logger.info(args)



# loss function
def bpr_loss(pos_score, neg_score):
    loss = -((pos_score - neg_score).sigmoid().log().mean())
    return loss



# load data
data = Data(DATASET, args)



# model initialize and setup
device = torch.device('cuda:{}'.format(GPU))

n_nodes = data.max_idx
logger.info('number of nodes: {}'.format(n_nodes))

n_edges = data.num_total_edges
logger.info('number of nodes: {}'.format(n_nodes))

tgan = TGRec(data.train_ngh_finder, n_nodes+1, args,
            num_layers=NUM_LAYER, use_time=USE_TIME, 
            agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, 
            n_head=NUM_HEADS, drop_out=DROP_OUT, 
            node_dim=NODE_DIM, time_dim=TIME_DIM)

checkpoint = torch.load(pretrain_path)
tgan.load_state_dict(checkpoint['model_state_dict'])
logger.info(f'loaded pretrain model from {pretrain_path}')

tgan = tgan.to(device)

optimizer = torch.optim.AdamW(tgan.parameters(), 
                             lr = LEARNING_RATE, 
                             weight_decay=1e-2)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# prepare for training
num_instance = len(data.train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)

logger.info('num of training instances: {}'.format(num_instance))
logger.info('num of batches per epoch: {}'.format(num_batch))

idx_list = np.arange(num_instance)
np.random.shuffle(idx_list) 

early_stopper = EarlyStopMonitor(max_round=5)

start_time = time.time()
epoch=0
m_loss = []
best_ap = 0
best_epoch = 0

logger.info('commencing final evaluation and test')
tgan.ngh_finder = data.full_ngh_finder
test_result, test_pred_output = eval_users(tgan, 
                                           data.test_src_l, data.test_dst_l, data.test_ts_l, 
                                           data.train_src_l, data.train_dst_l, 
                                           args)
logger.info(f'test: {test_result}')
        
endtime = time.time()
logger.info(f"need time: {endtime - start_time:.3f} seconds")
