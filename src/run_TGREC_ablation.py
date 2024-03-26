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
from evaluation import *
from utils import EarlyStopMonitor




# argument and global variables
parser = argparse.ArgumentParser('Interface for TGSRec experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=20, help='number of epochs')
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
NUM_EPOCH = args.n_epoch
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

# args.agg_method = 'lstm'

best_checkpoint = {
    'ml-100k': 5,
    'Baby': 7,
    'Digital_Music': 10,
    'Toys_and_Games': 5,
    'Tools_and_Home_Improvement': 5,
}


# set up paths
# pretrain_path = f'./saved_checkpoints/{args.data}/{args.data}-{str(best_checkpoint[args.data])}_TARGON.pt'
datetime_str = time.strftime("%Y%m%d-%H%M%S")
# model_save_path = f'./saved_models/{args.data}/{args.data}_TARGON.pt'
# get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.data}/{args.data}-{epoch}_TARGON.pt'

log_file_path = f'./log/ablation_empty/{args.data}_TARGON_{datetime_str}.log'
log_format = f'%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s'

result_file_path = f'./log/ablation_empty/results/TARGON_{args.data}_results.csv'
result_log_format = f'%(message)s'

# RANK_RESULTS_FILE = f"./rank_results/{args.data}/{args.data}_TARGON_{datetime_str}"
# test_pred_output_file = RANK_RESULTS_FILE + "_test.json"

# SAVE_MODEL_DIR = f"./saved_models/{args.data}"
# if not os.path.isdir(SAVE_MODEL_DIR):
#     os.mkdir(SAVE_MODEL_DIR)




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

# checkpoint = torch.load(pretrain_path)
# tgan.load_state_dict(checkpoint['model_state_dict'])
# logger.info(f'Loaded pretrain model from {pretrain_path}')

tgan = tgan.to(device)

optimizer = torch.optim.AdamW(tgan.parameters(), 
                             lr = LEARNING_RATE, 
                             weight_decay=1e-2)
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
criterion = torch.nn.BCELoss()

# result_log_str = lambda epoch, tr_loss, tr_ap, tr_f1, val_ap, val_f1, test_ap, test_f1, test_auc: \
#     f'Epoch [{epoch+1}/{NUM_EPOCH}]: Train Loss: {tr_loss:.4f} | Train AP: {tr_ap:.4f} | Train F1: {tr_f1:.4f} | Val AP: {val_ap:.4f} | Val F1: {val_f1:.4f} | Test AP: {test_ap:.4f} | Test F1: {test_f1:.4f} | Test AUC: {test_auc:.4f}'


result_logger.info('Epoch, Train Loss, Train AP, Train F1, Val AP, Val F1, Test AP, Test F1, Test AUC')

for epoch in range(NUM_EPOCH):
    # Training
    
    tgan.train()
    optimizer.zero_grad()
    
    tgan.ngh_finder = data.train_ngh_finder
    
    ap, f1 = [], []
    
    logger.info('Running epoch {}'.format(epoch+1))
    
    np.random.shuffle(idx_list)
    
    for k in range(num_batch):
        tgan.ngh_finder = data.train_ngh_finder
        
        s_idx = k * BATCH_SIZE
        e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
        
        src_l_cut, dst_l_cut = data.train_src_l[s_idx:e_idx], data.train_dst_l[s_idx:e_idx]
        ts_l_cut = data.train_ts_l[s_idx:e_idx]
        label_l_cut = data.train_label_l[s_idx:e_idx]
        
        size = len(src_l_cut)
        
        dst_l_fake = data.train_rand_sampler.sample_neg(src_l_cut)
        
        with torch.no_grad():
            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)

        optimizer.zero_grad()
        # pos_score, neg_score = tgan.contrast_nosigmoid(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)
        pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)
        
        loss = criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)
        
        # loss = bpr_loss(pos_score, neg_score)
        l2_reg = 0
        
        # for name, p in tgan.named_parameters():
        #     if "node_hist_embed" in name:
        #         l2_reg += p.norm(2)
        # loss = loss + (args.reg * l2_reg)
        
        loss.backward()
        optimizer.step()
        
    #     with torch.no_grad():
    #         tgan.eval()
    #         pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
    #         # pred_score = np.concatenate([(pos_score).cpu().detach().numpy(), (neg_score).cpu().detach().numpy()])
    #         # scaler = MinMaxScaler()
    #         pred_label = pred_score > 0.5
    #         # preds = np.transpose(scaler.fit_transform(np.transpose([pred_score])))[0]
            
    #         # pred_label = preds > 0.5
    #         true_label = np.concatenate([np.ones(size), np.zeros(size)])
            
    #         ap.append(average_precision_score(true_label, pred_score))
    #         f1.append(f1_score(true_label, pred_label))
    #         m_loss.append(loss.item())
        
    #     batch_iter_printer = num_batch // 5
    #     if k % batch_iter_printer == 0:
    #         logger.info(f'\tBatch: {k+1}/{num_batch} done')
            
    lr_scheduler.step()
    
    # validation within each epoch
    # tgan.ngh_finder = data.test_train_ngh_finder
    # _, val_ap, val_f1, _ = eval_one_epoch('validation during training', 
    #                                     tgan, data.val_rand_sampler,
    #                                     data.val_src_l, data.val_dst_l, data.val_ts_l, '')
    
    # tgan.ngh_finder = data.full_ngh_finder
    # _, test_ap, test_f1, test_auc = eval_one_epoch('testing during training', 
    #                                     tgan, data.test_rand_sampler,
    #                                     data.test_src_l, data.test_dst_l, data.test_dst_l, '')
    
    # result_logger.info(f'{epoch+1}, {np.mean(m_loss):.4f}, {np.mean(ap):.4f}, {np.mean(f1):.4f}, {val_ap:.4f}, {val_f1:.4f}, {test_ap:.4f}, {test_f1:.4f}, {test_auc:.4f}')
    
    # best_ap = max(best_ap, test_ap)
    # best_epoch = epoch if best_ap == val_ap else best_epoch
    
    # # if best_epoch == epoch:
    #     torch.save(
    #         {
    #             'model_state_dict': tgan.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': np.mean(m_loss),
    #             }, get_checkpoint_path(epoch+1)
    #         )
    
    
    # if early_stopper.early_stop_check(val_ap) and epoch > 25:
    #     logger.info('early stop triggered, break')
    #     break
    

# logger.info(f'best epoch: {best_epoch+1}, best test ap: {best_ap:.4f}')


# if args.n_epoch > 0:
#     logger.info('saving the last model')
#     torch.save(
#             {
#                 'model_state_dict': tgan.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': np.mean(m_loss),
#                 }, model_save_path
#     )
#     logger.info("model saved")


logger.info('commencing final test')
tgan.ngh_finder = data.full_ngh_finder
test_result, test_pred_output = eval_users(tgan, 
                                           data.test_src_l, data.test_dst_l, data.test_ts_l, 
                                           data.train_src_l, data.train_dst_l, 
                                           args)
logger.info(f'test: {test_result}')


# with open(test_pred_output_file, 'w') as f:
#     for eachoutput in test_pred_output:
#         f.write(json.dumps(eachoutput) + "\n")
        
endtime = time.time()
logger.info(f"need time: {endtime - start_time:.3f} seconds")
