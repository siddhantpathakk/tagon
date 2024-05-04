import math
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import torch
from trainer.evaluation.evaluation import eval_one_epoch, eval_users
from trainer.trainer_utils import bpr_loss

class Trainer:
    def __init__(self, data, model, optimizer, early_stopper, NUM_EPOCH, BATCH_SIZE, args):
        self.model = model
        self.optimizer = optimizer
        self.early_stopper = early_stopper
        self.NUM_EPOCH = NUM_EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.data = data
        
        self.args = args
        
        self.num_instance = len(data.train_src_l)
        self.num_batch = math.ceil(self.num_instance / BATCH_SIZE)
        self.idx_list = np.arange(self.num_instance)



    def train_one_epoch(self):
        acc, ap, f1, auc, m_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        np.random.shuffle(self.idx_list)
        
        for k in range(self.num_batch):
            self.model.ngh_finder = self.data.train_ngh_finder
            s_idx = k * self.BATCH_SIZE
            e_idx = min(self.num_instance - 1, s_idx + self.BATCH_SIZE)
            src_l_cut, dst_l_cut = self.data.train_src_l[s_idx:e_idx], self.data.train_dst_l[s_idx:e_idx]
            ts_l_cut = self.data.train_ts_l[s_idx:e_idx]
            label_l_cut = self.data.train_label_l[s_idx:e_idx]
            size = len(src_l_cut)
            
            if self.args.popnegsample:
                dst_l_fake = self.data.train_rand_sampler.popularity_based_sample_neg(src_l_cut)
            elif self.args.timepopnegsample:
                dst_l_fake = self.data.train_rand_sampler.timelypopularity_based_sample_neg(src_l_cut, ts_l_cut)
            
            else:
                dst_l_fake = self.data.train_rand_sampler.sample_neg(src_l_cut)
                      
        self.model.train()
        self.optimizer.zero_grad()
        pos_score, neg_score = self.model.contrast_nosigmoid(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, self.args.n_degree)
        loss = bpr_loss(pos_score, neg_score)
        l2_reg = 0
        for name, p in self.model.named_parameters():
            if "node_hist_embed" in name:
                l2_reg += p.norm(2)
                
        loss = loss + (self.args.reg*l2_reg)
        
        loss.backward()
        self.optimizer.step()
        
         # get training results
        with torch.no_grad():
            self.model.eval()
            pred_score = np.concatenate([(pos_score).cpu().detach().numpy(), 
                                         (neg_score).cpu().detach().numpy()])
            scaler = MinMaxScaler()
            preds = np.transpose(scaler.fit_transform(np.transpose([pred_score])))[0]
            pred_label = preds > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            acc = (pred_label == true_label).mean()
            ap = average_precision_score(true_label, pred_score)
            f1 = f1_score(true_label, pred_label)
            m_loss = loss.item()
            auc = roc_auc_score(true_label, pred_score)
        
        return acc, ap, f1, auc, m_loss
    
    
    def val_one_epoch(self):
        self.model.ngh_finder = self.data.test_train_ngh_finder
        val_acc, val_ap, val_f1, val_auc = eval_one_epoch('validation during training', 
                                                    self.model, self.data.val_rand_sampler,
                                                    self.data.val_src_l, self.data.val_dst_l, self.data.val_ts_l, '')

        return val_acc, val_ap, val_f1, val_auc
    
    
    def train(self):
        accs, aps, f1s, aucs, m_losses = [], [], [], [], []
        val_accs, val_aps, val_f1s, val_aucs = [], [], [], []
        
        print('Commence training phase for TAGON for {} epochs'.format(self.NUM_EPOCH))
        for epoch in range(self.NUM_EPOCH):
            acc, ap, f1, auc, m_loss = self.train_one_epoch()
            val_acc, val_ap, val_f1, val_auc = self.val_one_epoch()
            
            print(f'Epoch {epoch+1}/{self.NUM_EPOCH} acc: {acc:.4f} ap: {ap:.4f} f1: {f1:.4f} auc: {auc:.4f} loss: {m_loss:.4f} val_acc: {val_acc:.4f} val_ap: {val_ap:.4f} val_f1: {val_f1:.4f} val_auc: {val_auc:.4f}')
            
            if self.early_stopper.early_stop_check(val_auc):
                print('Early stopping at epoch: ', epoch)
                break
            
            accs.append(acc)
            aps.append(ap)
            f1s.append(f1)
            aucs.append(auc)
            m_losses.append(m_loss)
            
            val_accs.append(val_acc)
            val_aps.append(val_ap)
            val_f1s.append(val_f1)
            val_aucs.append(val_auc)
            
            if (epoch+1)%5 == 0:
                self.save_model(self.args.save_model_path, epoch+1)
            
        print('Training completed')
    
    def test(self):
        self.model.ngh_finder = self.data.test_train_ngh_finder
        print('Commence final test phase')
        valid_result, valid_pred_output = eval_users(self.model, self.data.val_src_l, self.data.val_dst_l, self.data.val_ts_l, self.data.train_src_l, self.data.train_dst_l, self.args)
        print("Final test phase completed")
        return valid_result, valid_pred_output
    
    def save_model(self, save_model_path, ckpt=None):
        model_name = self.args.data + '_TAGON.pt' if not ckpt else self.args.data + '-' + str(ckpt) + '_TAGON.pt'
        model_save_path = save_model_path + model_name
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, model_save_path
        )
        print("Model saved successfully at: ", model_save_path)
