from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import logging
import time
import pytorch_warmup as warmup

from model.TGSRec import TGRec
from model.components.losses import bpr_loss
from utils.callbacks import EarlyStopMonitor
from utils.evaluation import eval_one_epoch, eval_users
from utils.seed import seed_everything

class Trainer:
    def __init__(self, args, data, SAVE_MODEL_PATH):
        self.logger = logging.getLogger()
        
        self.data = data
        self.args = args
        
        self.build_model()
        
        self.num_instance = self.data.get_num_instances()
        self.idx_list = np.arange(self.num_instance)
        
        self.num_batch = self.data.get_num_batches(self.args)
        
        self.early_stopper = EarlyStopMonitor()
        
        self.SAVE_MODEL_PATH = SAVE_MODEL_PATH
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_ap': [],
            'train_f1': [],
            'train_auc': [],
            
            'val_acc': [],
            'val_ap': [],
            'val_f1': [],
            'val_auc': [],
            
            'test_acc': [],
            'test_ap': [],
            'test_f1': [],
            'test_auc': []
        }
        
        self.Ks = [1, 5, 10, 20, 40, 50, 60, 70, 80, 90, 100]

        self.NUM_NEIGHBORS = self.args.n_degree
        
    def build_model(self):
        seed_everything(self.args.seed)
        device = torch.device('cuda:{}'.format(self.args.gpu))
        
        n_nodes = self.data.max_idx
        n_edges = self.data.num_total_edges
        
        self.model = TGRec(self.data.train_ngh_finder, n_nodes+1, self.args,
                        num_layers= self.args.n_layer, 
                        use_time=self.args.time, agg_method=self.args.agg_method, attn_mode=self.args.attn_mode,
                        seq_len=self.args.n_degree, n_head=self.args.n_head, 
                        drop_out=self.args.drop_out, 
                        node_dim=self.args.node_dim, time_dim=self.args.time_dim)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                            lr=self.args.lr,
                                            weight_decay=self.args.l2,)
        
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer)
        
        self.model = self.model.to(device)
        
        self.warmup_period = 15
        self.warmup_scheduler = warmup.LinearWarmup(self.optimizer, warmup_period=self.warmup_period)
        
        self.logger.info("Model built successfully")
        self.logger.info(self.model)
        self.logger.info(f'Optimizer: {self.optimizer.__class__.__name__} with lr: {self.args.lr} and l2: {self.args.l2}')
        self.logger.info(f'Learning rate scheduler: {self.lr_scheduler.__class__.__name__}')
        self.logger.info(f'Warmup scheduler: {self.warmup_scheduler.__class__.__name__}')
        self.logger.info(f'Device: {device}')
        self.logger.info(f'Number of nodes: {n_nodes}')
        self.logger.info(f'Number of edges: {n_edges}')
        
    
    def train_for_one_epoch(self):
        self.model.ngh_finder = self.data.train_ngh_finder
        np.random.shuffle(self.idx_list)
        acc, ap, f1, auc, m_loss = [], [], [], [], []

        for k in range(self.num_batch):
            
            start_index = k * self.args.bs
            end_index = min(self.num_instance - 1, start_index + self.args.bs)
            
            src_l_cut = self.data.train_src_l[start_index:end_index]
            dst_l_cut = self.data.train_dst_l[start_index:end_index]
            ts_l_cut = self.data.train_ts_l[start_index:end_index]
            label_l_cut = self.data.train_label_l[start_index:end_index]
            
            size = len(src_l_cut)
            
            if self.args.popnegsample:
                dst_l_fake = self.data.train_rand_sampler.popularity_based_sample_neg(src_l_cut)
            elif self.args.timepopnegsample:
                dst_l_fake = self.data.train_rand_sampler.timelypopularity_based_sample_neg(src_l_cut, ts_l_cut)
            else:
                dst_l_fake = self.data.train_rand_sampler.sample_neg(src_l_cut)
                
            self.optimizer.zero_grad()
            self.model.train()
            
            pos_score, neg_score = self.model.contrast_nosigmoid(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, self.args.n_degree)

            loss = self.calc_loss(pos_score, neg_score)
            
            loss.backward()
            self.optimizer.step()
            
            with self.warmup_scheduler.dampening():
                if self.warmup_scheduler.last_step + 1 >= self.warmup_period:
                    self.lr_scheduler.step()
            
            with torch.no_grad():
                self.model.eval()
                pred_score = np.concatenate([(pos_score).cpu().detach().numpy(), (neg_score).cpu().detach().numpy()])
                scaler = MinMaxScaler() # pred_label = pred_score > 0.5
                preds = np.transpose(scaler.fit_transform(np.transpose([pred_score])))[0]
                pred_label = preds > 0.5
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                
                
                acc_, ap_, f1_, auc_ = self.calc_metrics(true_label, pred_label, pred_score)
                acc.append(acc_)
                ap.append(ap_)
                f1.append(f1_)
                auc.append(auc_)
                m_loss.append(loss.item())
        
        return np.mean(m_loss), np.mean(acc), np.mean(ap), np.mean(f1), np.mean(auc)
    

    def calc_metrics(self, true_label, pred_label, pred_score):
        acc = (pred_label == true_label).mean()
        ap = average_precision_score(true_label, pred_score)
        f1 = f1_score(true_label, pred_label)
        auc = roc_auc_score(true_label, pred_score)
        
        return acc, ap, f1, auc

    def calc_loss(self, pos_score, neg_score):
        loss = bpr_loss(pos_score, neg_score)
        l2_reg = 0
        
        for name, p in self.model.named_parameters():
            if "node_hist_embed" in name:
                l2_reg += p.norm(2)
        
        loss = loss + self.args.reg*l2_reg
        
        return loss
    
    def train(self):
        
        for epoch in range(self.args.n_epoch):
            # Train for one epoch
            m_loss, acc, ap, f1, auc = self.train_for_one_epoch()
            
            self.history['train_loss'].append(m_loss)
            self.history['train_acc'].append(acc)
            self.history['train_ap'].append(ap)
            self.history['train_f1'].append(f1)
            self.history['train_auc'].append(auc)
                    
            if np.mean(acc) == 0.5 and np.mean(auc) == 0.5 and np.mean(f1) == 0:
                break
            
            if ((epoch+1) % 20 == 0 and (epoch+1) >= 200) or (epoch+1) == self.args.n_epoch:
                model_state = {
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': np.mean(m_loss),
                            } 
                torch.save(model_state, self.SAVE_MODEL_PATH)
                
                
            # Validation
            self.model.ngh_finder = self.data.full_ngh_finder
            
            val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for old nodes', self.model, self.data.val_rand_sampler, self.data.val_src_l, self.data.val_dst_l, self.data.val_ts_l, self.NUM_NEIGHBORS, self.data.val_label_l)
            # valid_result, valid_pred_output = eval_users(self.model, self.data.val_src_l, self.data.val_dst_l, self.data.val_ts_l, self.data.train_src_l, self.data.train_dst_l, self.args)
            
            if epoch+1 > self.warmup_period:
                if self.early_stopper.early_stop_check(val_auc):
                    self.logger.info(f'Early stopping at epoch {epoch+1}')
                    break
            
            # test_result, test_pred_output = eval_users(self.model, self.data.test_src_l, self.data.test_dst_l, self.data.test_ts_l, self.data.train_src_l, self.data.train_dst_l, self.args)
            test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for old nodes', self.model, self.data.test_rand_sampler, self.data.test_src_l, self.data.test_dst_l, self.data.test_ts_l, self.NUM_NEIGHBORS,  self.data.test_label_l)
        
            self.logger.info(f'Epoch [{epoch+1}/{self.args.n_epoch}]:\tTrain Loss: {m_loss:.4f}\tTrain Acc: {acc:.4f}\tTrain AP: {ap:.4f}\tTrain F1: {f1:.4f}\tTrain AUC: {auc:.4f}\tVal Acc: {val_acc:.4f}\tVal AP: {val_ap:.4f}\tVal F1: {val_f1:.4f}\tVal AUC: {val_auc:.4f}\tTest Acc: {test_acc:.4f}\tTest AP: {test_ap:.4f}\tTest F1: {test_f1:.4f}\tTest AUC: {test_auc:.4f}')
            self.history['val_acc'].append(val_acc)
            self.history['val_ap'].append(val_ap)
            self.history['val_f1'].append(val_f1)
            self.history['val_auc'].append(val_auc)
            
            self.history['test_acc'].append(test_acc)
            self.history['test_ap'].append(test_ap)
            self.history['test_f1'].append(test_f1)
            self.history['test_auc'].append(test_auc)
            
                    
    def load_model(self, ckpt):
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()
        self.logger.info(f'Model loaded from {ckpt}')
        return self.model
    
    def export_history(self):
        import json
        
        with open(f'/home/FYP/siddhant005/fyp/log/history/{time.strftime("%d%m%y-%H%M%S")}_history.json', 'w') as f:
            json.dump(self.history, f)
        
        self.logger.info(f'History exported to /home/FYP/siddhant005/fyp/log/{time.strftime("%d%m%y-%H%M%S")}_history.json')
    
    def plot_history(self):
        
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 5, 1)
        plt.plot(self.history['train_loss'], label='train_loss')
        plt.title('Train Loss')
        plt.legend()
        
        plt.subplot(1, 5, 2)
        plt.plot(self.history['train_acc'], label='train_acc')
        plt.plot(self.history['val_acc'], label='val_acc')
        plt.plot(self.history['test_acc'], label='test_acc')
        plt.title('Train Acc vs Val Acc vs Test Acc')
        plt.legend()
        
        plt.subplot(1, 5, 3)
        plt.plot(self.history['train_ap'], label='train_ap')
        plt.plot(self.history['val_ap'], label='val_ap')
        plt.plot(self.history['test_ap'], label='test_ap')
        plt.title('Train AP vs Val AP vs Test AP')
        plt.legend()
        
        plt.subplot(1, 5, 4)
        plt.plot(self.history['train_f1'], label='train_f1')
        plt.plot(self.history['val_f1'], label='val_f1')
        plt.plot(self.history['test_f1'], label='test_f1')
        plt.title('Train F1 vs Val F1 vs Test F1')
        plt.legend()
        
        plt.subplot(1, 5, 5)
        plt.plot(self.history['train_auc'], label='train_auc')
        plt.plot(self.history['val_auc'], label='val_auc')
        plt.plot(self.history['test_auc'], label='test_auc')
        plt.title('Train AUC vs Val AUC vs Test AUC')
        plt.legend()
        
        plt.savefig(f'/home/FYP/siddhant005/fyp/log/history/{time.strftime("%d%m%y-%H%M%S")}_history.png')
        
        