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
    # seed_everything(self.args.seed)
    device = torch.device('cuda:{}'.format(args.gpu))
    
    n_nodes = data.max_idx
    n_edges = data.num_total_edges
    
    model = TGRec(data.train_ngh_finder, n_nodes+1, args,
                    num_layers= args.n_layer, 
                    use_time=args.time, agg_method=args.agg_method, attn_mode=args.attn_mode,
                    seq_len=args.n_degree, n_head=args.n_head, 
                    drop_out=args.drop_out, 
                    node_dim=args.node_dim, time_dim=args.time_dim)
    
    optimizer = torch.optim.Adam(model.parameters(), 
                                        lr=args.lr,
                                        weight_decay=args.l2,)
    
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    
    model = model.to(device)
    
    warmup_period = 15
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=warmup_period)
    
    # logger.info("Model built successfully")
    # logger.info(model)
    # logger.info(f'Optimizer: {optimizer.__class__.__name__} with lr: {args.lr} and l2: {args.l2}')
    # logger.info(f'Learning rate scheduler: {lr_scheduler.__class__.__name__}')
    # logger.info(f'Warmup scheduler: {warmup_scheduler.__class__.__name__}')
    # logger.info(f'Device: {device}')
    # logger.info(f'Number of nodes: {n_nodes}')
    # logger.info(f'Number of edges: {n_edges}')
    
    return model, optimizer, lr_scheduler, warmup_scheduler, device
    

class EarlyStopMonitor(object):
    def __init__(self, max_round=10, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance
        
        self.logger = logging.getLogger(__name__)
        
        # self.logger.info(f"Early stopping monitor: max_round={max_round}, higher_better={higher_better}, tolerance={tolerance}")


    def early_stop_check(self, curr_val):
        self.epoch_count += 1
        
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        return self.num_round >= self.max_round
    
    