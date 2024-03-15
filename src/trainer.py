import time
import torch
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from utils.evaluation_utils import eval_users
from utils.trainer_utils import EarlyStopMonitor, bpr_loss, build_model, get_new_history


class Trainer:
    def __init__(self, args, data, SAVE_MODEL_PATH):
        self.logger = logging.getLogger()
        
        self.args = args
        self.data = data
        
        self.model, self.optimizer, self.lr_scheduler, self.warmup_scheduler, self.device = build_model(self.args, self.data, self.logger)
        
        self.num_instance = self.data.get_num_instances()
        self.idx_list = np.arange(self.num_instance)
        
        self.num_batch = self.data.get_num_batches(self.args)
        
        self.early_stopper = EarlyStopMonitor()
        
        self.SAVE_MODEL_PATH = SAVE_MODEL_PATH
        self.NUM_NEIGHBORS = self.args.n_degree
        
        self.history = get_new_history()
   
    
    def train_for_one_epoch(self, epoch):
        self.logger.info(f'Commencing training for epoch # {epoch+1}')   
        self.model.train()
        self.model.ngh_finder = self.data.train_ngh_finder
        np.random.shuffle(self.idx_list)
        m_loss, acc_arr, ap_arr, f1_arr, auc_arr = [], [], [], [], []

        for k in range(self.num_batch):
            # self.logger.info(f'Epoch {epoch+1}, batch {k+1}/{self.num_batch}')
            start_index = k * self.args.bs
            end_index = min(self.num_instance - 1, start_index + self.args.bs)
            
            src_l_cut = self.data.train_src_l[start_index:end_index]
            dst_l_cut = self.data.train_dst_l[start_index:end_index]
            ts_l_cut = self.data.train_ts_l[start_index:end_index]
            
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
            
            # with self.warmup_scheduler.dampening():
            #     if self.warmup_scheduler.last_step + 1 >= self.warmup_period:
            #         self.lr_scheduler.step()
            
            with torch.no_grad():
                self.model.eval()
                pred_score = np.concatenate([(pos_score).cpu().detach().numpy(), (neg_score).cpu().detach().numpy()])
                scaler = MinMaxScaler() # pred_label = pred_score > 0.5
                preds = np.transpose(scaler.fit_transform(np.transpose([pred_score])))[0]
                pred_label = preds > 0.5
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                accuracy, ap, f1, auc = self.calc_training_performance(true_label, pred_label)
                
                acc_arr.append(accuracy)
                ap_arr.append(ap)
                f1_arr.append(f1)
                auc_arr.append(auc)
                m_loss.append(loss.item())
        

        return np.mean(m_loss), np.mean(acc_arr), np.mean(ap_arr), np.mean(f1_arr), np.mean(auc_arr)
   

    def train(self):
        for epoch in range(self.args.n_epoch):
            tr_time = time.time()
            mean_loss, mean_acc, mean_ap, mean_f1, mean_auc = self.train_for_one_epoch(epoch)
                    
            self.history['train_loss'].append(mean_loss)
            self.history['train_acc'].append(mean_acc)
            self.history['train_ap'].append(mean_ap)
            self.history['train_f1'].append(mean_f1)
            self.history['train_auc'].append(mean_auc)
            
            self.logger.info(f'Epoch [{epoch+1}/{self.args.n_epoch}]\t[Training]:\tTime: {time.time() - tr_time:.2f}sec\tLoss: {mean_loss:.4f}\tAcc: {mean_acc:.4f}\tAP: {mean_ap:.4f}\tF1: {mean_f1:.4f}\tAUC: {mean_auc:.4f}')
            
            if ((epoch+1) % self.args.ckpt_epoch == 0 and (epoch+1) >= 200) or (epoch+1) == self.args.n_epoch:
                self.export_model_optim_state(np.mean(mean_loss))
            
            val_time = time.time()
            valid_result = self.evaluate('validate')
            val_r10, val_r20, val_mrr = valid_result[0]['recall'][0], valid_result[0]['recall'][1], valid_result[0]['mrr']
            self.logger.info(f'Epoch [{epoch+1}/{self.args.n_epoch}]\t[Validation]:\tTime: {time.time() - val_time:.2f}sec\tR@10: {val_r10:.4f}\tR@20: {val_r20:.4f}\tMRR: {val_mrr:.4f}')
        
        test_time = time.time()
        test_results = self.evaluate('test')
        test_r10, test_r20, test_mrr = test_results[0]['recall'][0], test_results[0]['recall'][1], test_results[0]['mrr']
        self.logger.info('\n')
        self.logger.info(f'Final Test:\tTime: {time.time() - test_time:.2f}sec\tR@10: {test_r10:.4f}\tR@20: {test_r20:.4f}\tMRR: {test_mrr:.4f}')
        
        self.export_history()
        self.export_model_optim_state(mean_loss)
        self.logger.info('Training completed, model and optimizer state saved locally.')
    

    def evaluate(self, type='validate'):
        # self.logger.info(f'Commencing evaluation for {type}')
        self.model.ngh_finder = self.data.full_ngh_finder
        if type == 'validate':
            return eval_users(self.model, 
                            self.data.val_src_l, self.data.val_dst_l, self.data.val_ts_l, 
                            self.data.train_src_l, self.data.train_dst_l, 
                            self.args)
        elif type == 'test':
            return eval_users(self.model, 
                            self.data.test_src_l, self.data.test_dst_l, self.data.test_ts_l, 
                            self.data.train_src_l, self.data.train_dst_l, 
                            self.args)
        else:
            raise ValueError('Invalid type, use val or test')

    def calc_training_performance(self, true_label, pred_label):
        accuracy = (true_label == pred_label).mean()
        ap = average_precision_score(true_label, pred_label)
        f1 = f1_score(true_label, pred_label)
        auc = roc_auc_score(true_label, pred_label)
        
        return accuracy, ap, f1, auc

    def calc_loss(self, pos_score, neg_score):
        loss = bpr_loss(pos_score, neg_score)
        l2_reg = 0
        
        for name, p in self.model.named_parameters():
            if "node_hist_embed" in name:
                l2_reg += p.norm(2)
        
        loss = loss + self.args.reg * l2_reg
        
        return loss
    
    
    # export functions (to checkpoint model and history)
    def export_model_optim_state(self, m_loss):
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': np.mean(m_loss),
        } 
        torch.save(model_state, self.SAVE_MODEL_PATH)
        # self.logger.info(f'Model saved at {self.SAVE_MODEL_PATH}')
    
    
    def export_history(self):
        import json
        
        with open(f'/home/FYP/siddhant005/fyp/log/history/{time.strftime("%d%m%y-%H%M%S")}_history.json', 'w') as f:
            json.dump(self.history, f)
        
        self.logger.info(f'History exported to /home/FYP/siddhant005/fyp/log/{time.strftime("%d%m%y-%H%M%S")}_history.json')
