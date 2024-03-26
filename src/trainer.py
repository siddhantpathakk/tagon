import math
import time
import torch
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from utils.evaluation_utils import eval_users
from utils.trainer_utils import EarlyStopper, bpr_loss, build_model, get_new_history


class Trainer:
    def __init__(self, args, data, SAVE_MODEL_PATH, SAVE_MODEL_DIR):
        self.logger = logging.getLogger()
        
        self.args = args
        self.data = data
        
        self.model, self.optimizer, self.lr_scheduler, self.warmup_scheduler, self.device = build_model(self.args, self.data, self.logger)
        
        self.num_instance = self.data.get_num_instances()
        self.idx_list = np.arange(self.num_instance)
        
        self.num_batch = self.data.get_num_batches(self.args)
        
        self.early_stopper = EarlyStopper(patience=5)
        
        self.SAVE_MODEL_PATH = SAVE_MODEL_PATH
        self.SAVE_MODEL_DIR = SAVE_MODEL_DIR
        self.NUM_NEIGHBORS = self.args.n_degree
        
        self.history = get_new_history()
        
        self.warmup_period = 15
   
    
    def train_for_one_epoch(self, epoch):
        # self.logger.info(f'Commencing training for epoch # {epoch+1}')   
        self.model.train()
        self.model.ngh_finder = self.data.train_ngh_finder
        np.random.shuffle(self.idx_list)
        m_loss, acc_arr, ap_arr, f1_arr, auc_arr = [], [], [], [], []
        for k in range(self.num_batch):
            # print(f'Epoch {epoch+1}, batch {k+1}/{self.num_batch}')
            start_index = k * self.args.bs
            end_index = min(self.num_instance - 1, start_index + self.args.bs)
            
            src_l_cut = self.data.train_src_l[start_index:end_index]
            dst_l_cut = self.data.train_dst_l[start_index:end_index]
            ts_l_cut = self.data.train_ts_l[start_index:end_index]
            
            size = len(src_l_cut)
            
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
                accuracy, ap, f1, auc = self.calc_training_performance(true_label, pred_label)
                
                acc_arr.append(accuracy)
                ap_arr.append(ap)
                f1_arr.append(f1)
                auc_arr.append(auc)
                m_loss.append(loss.item())

        return np.mean(m_loss), np.mean(acc_arr), np.mean(ap_arr), np.mean(f1_arr), np.mean(auc_arr)

    def val_for_one_epoch(self, hint, sampler, src, dst, ts, label):
        val_acc, val_ap, val_f1, val_auc, val_loss = [], [], [], [], []
        with torch.no_grad():
            self.model.eval()
            TEST_BATCH_SIZE=1024
            num_test_instance = len(src)
            num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
            for k in range(num_test_batch):
                print(f'Validation batch {k+1}/{num_test_batch}')
                s_idx = k * TEST_BATCH_SIZE
                e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
                src_l_cut = src[s_idx:e_idx]
                dst_l_cut = dst[s_idx:e_idx]
                ts_l_cut = ts[s_idx:e_idx]

                size = len(src_l_cut)
                dst_l_fake = sampler.sample_neg(src_l_cut)
                
                pos_score, neg_score = self.model.contrast_nosigmoid(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, self.args.n_degree)

                loss = self.calc_loss(pos_score, neg_score)
                
                pred_score = np.concatenate([(pos_score).cpu().detach().numpy(), (neg_score).cpu().detach().numpy()])
                scaler = MinMaxScaler() # pred_label = pred_score > 0.5
                preds = np.transpose(scaler.fit_transform(np.transpose([pred_score])))[0]
                pred_label = preds > 0.5
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                accuracy, ap, f1, auc = self.calc_training_performance(true_label, pred_label)
                    
                val_acc.append(accuracy)
                val_ap.append(ap)
                val_f1.append(f1)
                val_auc.append(auc)
                val_loss.append(loss.item())
                
        return np.mean(val_loss), np.mean(val_acc), np.mean(val_ap), np.mean(val_f1), np.mean(val_auc)   


    def train(self):
        for epoch in range(self.args.n_epoch):
            
            start = time.time()
            mean_loss, mean_acc, mean_ap, mean_f1, mean_auc = self.train_for_one_epoch(epoch)
            tr = time.time() - start
            
            self.history['train_loss'].append(mean_loss)
            self.history['train_acc'].append(mean_acc)
            self.history['train_ap'].append(mean_ap)
            self.history['train_f1'].append(mean_f1)
            self.history['train_auc'].append(mean_auc)
            
            self.logger.info(f'Epoch [{epoch+1}/{self.args.n_epoch}]\t{tr:.2f}s\t[TRAIN]:\tLoss: {mean_loss:.3f}\tAcc: {mean_acc:.3f}\tAP: {mean_ap:.3f}\tF1: {mean_f1:.3f}\tAUC: {mean_auc:.3f}')
            
            start = time.time()
            valid_result = self.evaluate('validate')
            val_t = time.time() - start
            loss, acc, ap, f1, auc = valid_result[0], valid_result[1], valid_result[2], valid_result[3], valid_result[4]
            self.logger.info(f'Epoch [{epoch+1}/{self.args.n_epoch}]\t{val_t:.2f}s\t[VALID]:\tLoss: {loss:.3f}\tAcc: {acc:.3f}\tAP: {ap:.3f}\tF1: {f1:.3f}\tAUC: {auc:.3f}')
            
            self.history['val_loss'].append(loss)
            self.history['val_acc'].append(acc)
            self.history['val_ap'].append(ap)
            self.history['val_f1'].append(f1)
            self.history['val_auc'].append(auc)
            
            self.export_model_optim_state(np.mean(mean_loss), epoch+1)
                
            if not self.args.pretrain and (epoch+1 > self.warmup_period) and self.early_stopper.early_stop(loss):      
                self.logger.info(f'Early stopping at epoch {epoch+1} due to validation loss {loss:.3f}')       
                break
            
            if (epoch+1) == self.args.n_epoch:
                start = time.time()
                valid_result_last = self.evaluate('validate_last')
                val_t = time.time() - start
                val_r10, val_r20, val_mrr = valid_result_last['recall'][0], valid_result_last['recall'][1], valid_result_last['mrr']
                self.logger.info(f'Epoch [{epoch+1}/{self.args.n_epoch}]\t{val_t:.2f}s\t[VALID**]:\tR@10: {val_r10:.3f}\tR@20: {val_r20:.3f}\tMRR: {val_mrr:.3f}')
                self.history['val_recall10'].append(val_r10)
                self.history['val_recall20'].append(val_r20)
                self.history['val_mrr'].append(val_mrr)
                
        test_time = time.time()
        test_results = self.evaluate('test')
        test_r10, test_r20, test_mrr = test_results['recall'][0], test_results['recall'][1], test_results['mrr']
        self.logger.info(f'Final TEST SET:\tTime: {time.time() - test_time:.2f}sec\tR@10: {test_r10:.3f}\tR@20: {test_r20:.3f}\tMRR: {test_mrr:.3f}')
        
        self.history['test_recall10'].append(test_r10)
        self.history['test_recall20'].append(test_r20)
        self.history['test_mrr'].append(test_mrr)
        
        self.export_history()
        self.export_model_optim_state(mean_loss, epoch+1)
        self.logger.info('Training completed, model and optimizer state saved locally.')
    

    def evaluate(self, type='validate'):
        
        if type == "validate_last":
            self.logger.info(f'Commencing evaluation for {type}')
            self.model.ngh_finder = self.data.test_train_ngh_finder
            return eval_users(self.model, 
                            self.data.val_src_l, self.data.val_dst_l, self.data.val_ts_l, 
                            self.data.train_src_l, self.data.train_dst_l, 
                            self.args)
            
            
        if type == 'validate':
            self.model.ngh_finder = self.data.test_train_ngh_finder
            return self.val_for_one_epoch('val for old nodes', self.data.val_rand_sampler,
                                          self.data.val_src_l, self.data.val_dst_l, self.data.val_ts_l, self.data.val_label_l)
            
                                  
        elif type == 'test':
            self.logger.info(f'Commencing evaluation for {type}')
            self.model.ngh_finder = self.data.full_ngh_finder
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
        
        loss = loss + (self.args.reg * l2_reg)
        return loss
    
    
    # export functions (to checkpoint model and history)
    def export_model_optim_state(self, m_loss, epoch):
        model_name = f'{self.SAVE_MODEL_DIR}/TARGON_{self.args.data}_ckpt_epoch{epoch}.pt'
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': np.mean(m_loss),
        } 
        torch.save(model_state, model_name)
        # self.logger.info(f'Model saved at {model_name}')
    
    def export_history(self):
        import json
        
        with open(f'/home/FYP/siddhant005/fyp/log/history/{time.strftime("%d%m%y-%H%M%S")}_history.json', 'w') as f:
            json.dump(self.history, f)
        
        self.logger.info(f'History exported to /home/FYP/siddhant005/fyp/log/{time.strftime("%d%m%y-%H%M%S")}_history.json')
