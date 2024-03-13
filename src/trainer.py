from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import logging
import time
import torch.multiprocessing as mp

from utils import metrics
from model.TGSRec import TGRec
from model.components.losses import bpr_loss
from utils.callbacks import EarlyStopMonitor
from utils.evaluation import eval_users

class Trainer:
    def __init__(self, args, data, SAVE_MODEL_PATH):
        self.logger = logging.getLogger()
        
        self.data = data
        self.args = args
        
        self.model, self.optimizer, self.device, self.n_nodes, self.n_edges = self.build_model()
        
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
            'val_loss': [],
            'val_acc': [],
            'val_ap': [],
            'val_f1': [],
            'val_auc': []
        }
        
        self.Ks = [1, 5, 10, 20, 40, 50, 60, 70, 80, 90, 100]

        
    def build_model(self):
        device = torch.device('cuda:{}'.format(self.args.gpu))
        
        n_nodes = self.data.max_idx
        n_edges = self.data.num_total_edges
        
        model = TGRec(self.data.train_ngh_finder, n_nodes+1, self.args,
                        num_layers= self.args.n_layer, 
                        use_time=self.args.time, agg_method=self.args.agg_method, attn_mode=self.args.attn_mode,
                        seq_len=self.args.n_degree, n_head=self.args.n_head, 
                        drop_out=self.args.drop_out, 
                        node_dim=self.args.node_dim, time_dim=self.args.time_dim)
        
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=self.args.lr)
        
        model = model.to(device)
        
        self.logger.info("Model built successfully")
        self.logger.info(model)
        self.logger.info(f'Optimizer: {optimizer.__class__.__name__}')
        self.logger.info(f'Device: {device}')
        self.logger.info(f'Number of nodes: {n_nodes}')
        self.logger.info(f'Number of edges: {n_edges}')
        
        return model, optimizer, device, n_nodes, n_edges
    
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
        start_time = time.time()
        
        for epoch in range(1, self.args.n_epoch+1):
            # Train for one epoch
            m_loss, acc, ap, f1, auc = self.train_for_one_epoch()
            
            self.history['train_loss'].append(m_loss)
            self.history['train_acc'].append(acc)
            self.history['train_ap'].append(ap)
            self.history['train_f1'].append(f1)
            self.history['train_auc'].append(auc)
            
            self.logger.info(f'Epoch {epoch}:\tTrain Loss: {m_loss}\tTrain Acc: {acc}\tTrain AP: {ap}\tTrain F1: {f1}\tTrain AUC: {auc}')
        
            if np.mean(acc) == 0.5 and np.mean(auc) == 0.5 and np.mean(f1) == 0:
                break
            
            if ((epoch+1) % 20 == 0 and (epoch+1) >= 200) or (epoch+1) == self.args.n_epoch:
                model_state = {
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': np.mean(m_loss),
                            } 
                torch.save(model_state, self.SAVE_MODEL_PATH)
                
                # self.logger.info(f'Model checkpoint created at {self.SAVE_MODEL_PATH}')
                self.model.ngh_finder = self.data.full_ngh_finder
                #val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for old nodes', self.model, val_rand_sampler, val_src_l, val_dst_l, val_ts_l, val_label_l)
                valid_result, valid_pred_output = eval_users(self.model, self.data.val_src_l, self.data.val_dst_l, self.data.val_ts_l, self.data.train_src_l, self.data.train_dst_l, self.args)
                print('validation: ', valid_result)
                
                #test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for old nodes', self.model, test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l)
                test_result, test_pred_output = eval_users(self.model, self.data.test_src_l, self.data.test_dst_l, self.data.test_ts_l, self.data.train_src_l, self.data.train_dst_l, self.args)
                print('test: ', test_result)

        
        end_time = time.time()
        self.logger.info(f'Training time: {end_time - start_time}')
        
    def load_model(self, ckpt):
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()
        self.logger.info(f'Model loaded from {ckpt}')
        return self.model
    
    def plot_history(self):
        
        # for each metric, plot train and validation history in one graph together
        
        fig, axs = plt.subplots(2, 4, figsize=(15, 10))
        
        axs[0, 0].plot(self.history['train_loss'], label='Train')
        axs[0, 0].set_title('Train Loss')
        
        axs[0, 1].plot(self.history['train_acc'], label='Train')
        axs[0, 1].set_title('Train Accuracy')
        
        axs[0, 2].plot(self.history['train_ap'], label='Train')
        axs[0, 2].set_title('Train AP')
        
        axs[0, 3].plot(self.history['train_f1'], label='Train')
        axs[0, 3].set_title('Train F1')
        
        axs[1, 0].plot(self.history['train_auc'], label='Train')
        axs[1, 0].set_title('Train AUC')
        
        axs[1, 1].plot(self.history['val_loss'], label='Validation')
        axs[1, 1].set_title('Validation Loss')
        
        axs[1, 2].plot(self.history['val_acc'], label='Validation')
        axs[1, 2].set_title('Validation Accuracy')
        
        axs[1, 3].plot(self.history['val_ap'], label='Validation')
        axs[1, 3].set_title('Validation AP')
        
        plt.savefig(f'./log/{time.strftime("%d%m%y-%H%M%S")}_history.png')