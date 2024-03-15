import argparse
import os
import random

import numpy as np
import torch
import logging

import time
import sys

def parse_opt():

    ### Argument and global variables
    parser = argparse.ArgumentParser('Interface for TGSRec experiments on link predictions')
    
    # model hparam based arguments
    parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer') # [1,2,4]
    parser.add_argument('--n_layer', type=int, default=2, help='number of network layers') # [1,2,3]
    parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
    parser.add_argument('--reg', type=float, default=0.1, help='regularization')
    
    # model dimension based arguments
    parser.add_argument('--node_dim', type=int, default=32, help='Dimentions of the node embedding') # [8,16,32,64]
    parser.add_argument('--time_dim', type=int, default=32, help='Dimentions of the time embedding') # [8,16,32,64]
    parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
    parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='map', help='use dot product attention or mapping based')
    parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
    parser.add_argument('--new_node', action='store_true', help='model new node')
    parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty', 'disentangle'], help='how to use time information', default='disentangle')
    parser.add_argument('--disencomponents', type=int, default=10, help='number of various time encoding')
    
    # data based arguments
    parser.add_argument('-d', '--data', type=str, help='data sources to use', default='ml-100k')
    parser.add_argument('--train_test_val', type=str, default='80-10-10', help='train-test-validation split')
    parser.add_argument('--samplerate', type=float, default=1.0, help='sample rate for each user')
    parser.add_argument('--popnegsample', action='store_true', help='use popularity based negative sampling')
    parser.add_argument('--timepopnegsample', action='store_true', help='use timely popularity based negative sampling')
    parser.add_argument('--negsampleeval', type=int, default=-1, help='number of negative sampling evaluation, -1 for all')
    
    # training based arguments
    parser.add_argument('--bs', type=int, default=256, help='batch_size')
    parser.add_argument('--n_epoch', type=int, default=500, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate') # [1e-2, 1e-3, 1e-4]
    parser.add_argument('--prefix', type=str, default='ml100k', help='prefix to name the checkpoints')
    parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization') # [5e-1, 1e-1, 1e-2, 1e-3]
    parser.add_argument('--ckpt_epoch', type=int, default=10, help='save model per k epochs')

    
    try:
        args = parser.parse_args()
        return args
    except:
        parser.print_help()
        sys.exit(0)


def set_up_logger():
    logging_format =  f"%(levelname)s:\t%(module)s:\t%(message)s"
    
    logging.basicConfig(level=logging.INFO, format=logging_format)
    
    logging.addLevelName(logging.WARNING, 'WARN')
    
    logger = logging.getLogger()
    
    fh = logging.FileHandler(f'/home/FYP/siddhant005/fyp/log/{time.strftime("%d%m%y-%H%M%S")}.log')

    formatter = logging.Formatter(logging_format)
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)

    return logger


def seed_everything(seed=2020):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  
    
def get_model_save_path(args):
    MODEL_SAVE_PATH = f'/home/FYP/siddhant005/fyp/log/saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}.pth'
    get_checkpoint_path = lambda epoch: f'/home/FYP/siddhant005/fyp/log/saved_checkpoints/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'
    SAVE_MODEL_DIR = f"./log/saved_models/{args.data}"
    
    if not os.path.isdir(SAVE_MODEL_DIR):
        os.mkdir(SAVE_MODEL_DIR)
        
    SAVE_MODEL_PATH = SAVE_MODEL_DIR + f"/checkpoint.{args.bs}_{args.n_degree}_{args.n_epoch}_{args.n_head}_{args.drop_out}_{args.time}_{args.n_layer}_{args.n_degree}_{args.node_dim}_{args.time_dim}_{args.lr}.pth.tar"


    return SAVE_MODEL_DIR, SAVE_MODEL_PATH, MODEL_SAVE_PATH, get_checkpoint_path


def get_rank_results_paths(args):
    RANK_RESULTS_DIR = f"/home/FYP/siddhant005/fyp/log/rank_results/{args.data}"
    
    if not os.path.isdir(RANK_RESULTS_DIR):
        os.mkdir(RANK_RESULTS_DIR)
        
    RANK_RESULTS_FILE = RANK_RESULTS_DIR + f"/{args.bs}_{args.n_degree}_{args.n_epoch}_{args.n_head}_{args.drop_out}_{args.time}_{args.n_layer}_{args.n_degree}_{args.node_dim}_{args.time_dim}_{args.lr}"
    
    return RANK_RESULTS_FILE, RANK_RESULTS_DIR

