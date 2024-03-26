import logging
import numpy as np
import torch

from model.TGSRec import TGRec
import pytorch_warmup as warmup


def get_new_history():
    return {
            'train_loss': [],
            
            'train_acc':[],
            'train_ap':[],
            'train_f1':[],
            'train_auc':[],
            
            'val_loss': [],
            'val_acc':[],
            'val_ap':[],
            'val_f1':[],
            'val_auc':[],
            
            'val_recall10': [],
            'val_recall20': [],
            'val_mrr': [],
            
            'test_recall10': [],
            'test_recall20': [],
            'test_mrr':[]
        }


def bpr_loss(pos_score, neg_score):
    loss = -((pos_score - neg_score).sigmoid().log().mean())
    return loss


def build_model(args, data, logger):
    device = torch.device('cuda:{}'.format(args.gpu))
    
    n_nodes = data.max_idx
    n_edges = data.num_total_edges
    
    model = TGRec(data.train_ngh_finder, n_nodes+1, args,
                    num_layers= args.n_layer, 
                    use_time=args.time, agg_method=args.agg_method, attn_mode=args.attn_mode,
                    seq_len=args.n_degree, n_head=args.n_head, 
                    drop_out=args.drop_out, 
                    node_dim=args.node_dim, time_dim=args.time_dim)
    
    if args.pretrain:
        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info(f"Pretrained model loaded from {args.pretrain}")
    
    optimizer = torch.optim.Adam(model.parameters(), 
                                        lr=args.lr,
                                        weight_decay=args.l2,)
    
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    
    model = model.to(device)
    
    warmup_period = 15
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=warmup_period)
    
    logger.info("Model built successfully")
    logger.info(model)
    logger.info(f'Number of parameters: {sum(p.numel() for p in model.parameters())}'
                f' (trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)})')
    logger.info(f'Loss function: BPR + L2 regularization')
    logger.info(f'Optimizer: {optimizer.__class__.__name__} with lr: {args.lr} and l2: {args.l2}')
    logger.info(f'Learning rate scheduler: {lr_scheduler.__class__.__name__}')
    logger.info(f'Warmup scheduler: {warmup_scheduler.__class__.__name__} with warmup period: {warmup_period}')
    logger.info(f'Device: {device}')
    logger.info(f'Number of nodes: {n_nodes}')
    logger.info(f'Number of edges: {n_edges}')
    
    return model, optimizer, lr_scheduler, warmup_scheduler, device
    

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        logger = logging.getLogger()
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        
        logger.info(f'Early stopping callback initialized with patience: {patience} and min_delta: {min_delta}')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    