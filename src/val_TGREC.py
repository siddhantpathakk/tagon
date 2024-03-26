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
from model import TGRec
from data import Data
from evaluation import *


# Argument and global variables
parser = argparse.ArgumentParser('Interface for TGSRec experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use', default='ml-100k')
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
parser.add_argument('--model', type=str, default='TARGON',choices=['TARGON', 'TGSREC'], help='model to use')
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

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

# set up logger - for logging info and results separately
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/eval/{}_{}.log'.format(args.data, args.model))
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)
logger.info(args)

result_logger = logging.getLogger('result')
result_logger.setLevel(logging.DEBUG)
rfh = logging.FileHandler('log/eval/results/{}_{}.log'.format(args.data, args.model))
rfh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
result_logger.addHandler(rfh)


# set up directories
RANK_RESULTS_DIR = f"./rank_results/{args.data}"
if not os.path.isdir(RANK_RESULTS_DIR):
    os.mkdir(RANK_RESULTS_DIR)
RANK_RESULTS_FILE = RANK_RESULTS_DIR + f"/{args.data}_{args.model}"
test_pred_output_file = RANK_RESULTS_FILE + "_test.json"
valid_pred_output_file = RANK_RESULTS_FILE + '_valid.json'


# set random seed
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# load data
data = Data(DATASET, args)


# model initialize and load from checkpoint
device = torch.device('cuda:{}'.format(GPU))
n_nodes = data.max_idx
print(n_nodes, "n nodes")
n_edges = data.num_total_edges
print(n_edges, "n edges")

tgan = TGRec(data.train_ngh_finder, n_nodes+1, args,
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)

checkpoint_path = f"./saved_models/{DATASET}/{DATASET}_{args.model}.pt"
checkpoint = torch.load(checkpoint_path)

tgan.load_state_dict(checkpoint['model_state_dict'])
logger.info('loaded pretrain model from {}'.format(checkpoint_path))

tgan = tgan.to(device)

logger.info('{} model loaded to {}'.format(args.model, device))

start_time = time.time()
logger.info('commence final evaluation and test')


# validation
tgan.ngh_finder = data.test_train_ngh_finder
valid_result, valid_pred_output = eval_users(tgan, data.val_src_l, data.val_dst_l, data.val_ts_l, data.train_src_l, data.train_dst_l, args)
result_logger.info(f'validation: {valid_result}')


# test
tgan.ngh_finder = data.full_ngh_finder
test_result, test_pred_output = eval_users(tgan, data.test_src_l, data.test_dst_l, data.test_ts_l, data.train_src_l, data.train_dst_l, args)
result_logger.info(f'test: {test_result}')


# save the results
with open(valid_pred_output_file, 'w') as f:
    for eachoutput in valid_pred_output:
        f.write(json.dumps(eachoutput) + "\n")
        
with open(test_pred_output_file, 'w') as f:
    for eachoutput in test_pred_output:
        f.write(json.dumps(eachoutput) + "\n")
        
endtime = time.time()
result_logger.info(f"need time: {endtime - start_time:.3f} seconds")
