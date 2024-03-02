import matplotlib.pyplot as plt
import time
import numpy as np
import torch
torch.cuda.empty_cache()
import tqdm

from .utils.eval_utils import precision_at_k, recall_at_k, mapk, ndcg_k, hit_ratio_at_k
from .utils.trainer_utils import NegativeSampler,SubGraphExtractor
from .model import CAGSRec
from .utils.seed import seed_everything

def build_model(config):
    seed_everything(config.seed)
    
    model = CAGSRec(config)
    
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=config.lr, 
                                     weight_decay=config.l2)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=config.lr, 
                                    momentum=0.9, 
                                    weight_decay=config.l2)
    elif config.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), 
                                        lr=config.lr, 
                                        momentum=0.9, 
                                        weight_decay=config.l2)
    else:
        raise Exception('Unknown optimizer {}'.format(config.optimizer))
        
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    
    if config.verbose:
        print(model)
        print(f'Optimizer:\t{optimizer.__class__.__name__} with initial lr = {config.lr}, l2 = {config.l2}')
        print(f'LR Scheduler:\t{lr_scheduler.__class__.__name__}')
    
    return model, optimizer, lr_scheduler

def load_model_from_ckpt(config, model_ckpt):
    model = CAGSRec(config)
    model.load_state_dict(torch.load(model_ckpt))
    return model

class Trainer:
    def __init__(self, config, info, edges, resume=False, model_ckpt=None, eval_mode=False):
        
        node_num, relation_num = info['node_num'], info['relation_num']
        u2v, u2vc, v2u, v2vc = edges
        self.arg = config
        self.device = self.device
        
        self.item_indexes = torch.tensor(v_list_).to(self.device)
        
        self.u2v, self.u2vc, self.v2u, self.v2vc = u2v, u2vc, v2u, v2vc
        self.item_set = set(self.v2u.keys())
        
        v_list_ = list(self.u2v.values())
        unique_values = set()
        for sublist in v_list_:
            for item in sublist:
                unique_values.add(item)
        v_list_ = sorted(list(unique_values))        
        item_num = len(v_list_)
        
        self.model, self.optimizer, self.lr_scheduler = build_model(config, node_num, relation_num, item_num, self.device)
        
        if resume:
            assert model_ckpt is not None, 'model_ckpt must be provided if resume is True'
            self.model = load_model_from_ckpt(self.arg, model_ckpt)
        
        if eval_mode:
            assert model_ckpt is not None, 'model_ckpt must be provided if eval_mode is True'
            self.model = load_model_from_ckpt(self.arg, model_ckpt)
        
        self.node_num = node_num
        self.short_term_window_num = 3
        
        self.negative_sampler = NegativeSampler(self.item_set, self.node_num, self.short_term_window_num, self.arg.negative_num)
        self.subgraph_extractor = SubGraphExtractor(self.u2v, self.u2vc, self.v2u, self.v2vc, self.node_num, self.short_term_window_num)
    
        self.topk_list = [10, 20]
        
    def train(self, train_part, test_part):
        
        loss_list = []
        p, r, map_, ndcg, hr = [], [], [], [], []
        p2, r2, map2, ndcg2, hr2 = [], [], [], [], []
        
        users_np, sequences_np_train = train_part[0], train_part[1]
        sequences_np_test, test_set, uid_list_ = test_part[0], test_part[1], test_part[2]
        users_np_test = np.array(uid_list_)

        short_term_window_size = int(self.arg.L / self.short_term_window_num)
        short_term_window = [0] + [i + short_term_window_size for i in range(self.short_term_window_num-1)] + [-1]
        
        train_num = users_np.shape[0]
        record_indexes = np.arange(train_num)

        epoch_loss = 0.0
        batch_num = int(train_num/self.arg.batch_size) + 1

        self.model.train()
        
        for epoch_ in tqdm(range(self.arg.epoch_num), desc='Epoch Progress'):
            user_emd_batch_list,item_emd_batch_list = list(), list()
            start = time.time()
            for batch_ in tqdm(range(batch_num), desc=f'Batch Progress in Epoch {epoch_+1}', leave=False):
                start_index, end_index = batch_ * self.arg.batch_size, (batch_+1) * self.arg.batch_size
                batch_record_index = record_indexes[start_index: end_index]

                batch_users = users_np[batch_record_index]
                batch_neg = self.negative_sampler.sample(self.u2v, batch_users, self.item_set)

                batch_sequences_train = sequences_np_train[batch_record_index]
                batch_sequences, batch_targets = batch_sequences_train[:,
                                                                       :self.arg.L], batch_sequences_train[:, self.arg.L:]

                # Extracting SUBGraph (long term)
                batch_users, batch_sequences, edge_index, edge_type, node_no, node2ids = self.subgraph_extractor.extract(batch_users, batch_sequences, sub_seq_no=None)

                # Extracting SUBGraph (short term)
                short_term_part = []
                for i in range(len(short_term_window)):
                    if i != len(short_term_window)-1:
                        sub_seq_no = batch_sequences[:, short_term_window[i]:short_term_window[i+1]]
                        _, _, edge_index, edge_type, _, _ = self.subgraph_extractor.extract(batch_users, batch_sequences, sub_seq_no=sub_seq_no, node2ids=node2ids)
                        short_term_part.append((edge_index, edge_type))

                batch_users = torch.tensor(batch_users).to(self.device)
                batch_sequences = torch.from_numpy(batch_sequences).type(torch.LongTensor).to(self.device)
                batch_targets = torch.from_numpy(batch_targets).type(torch.LongTensor).to(self.device)
                batch_negatives = torch.from_numpy(batch_neg).type(torch.LongTensor).to(self.device)

                items_to_predict = torch.cat((batch_targets, batch_negatives), 1)

                X_user_item = [batch_users, batch_sequences, items_to_predict]
                X_graph_base = [edge_index, edge_type,
                                node_no, short_term_part]

                pred_score, user_emb, item_embs_conv = self.model(X_user_item, X_graph_base, for_pred=False)

                user_emd_batch_list.append(user_emb)
                item_emd_batch_list.append(item_embs_conv)

                (targets_pred, negatives_pred) = torch.split(pred_score,
                                                             [batch_targets.size(1), batch_negatives.size(1)], 
                                                             dim=1)

                # RAGCN loss (long term)
                gcn_loss = 0
                for gconv in self.model.conv_modulelist:
                    w = torch.matmul(gconv.att_r, gconv.basis.view(gconv.num_bases, -1)).view(gconv.num_relations, gconv.in_channels, gconv.out_channels)
                    gcn_loss = gcn_loss + torch.sum((w[1:, :, :] - w[:-1, :, :])**2)
                gcn_loss = gcn_loss/len(self.model.conv_modulelist)

                # RAGCN loss (short term)
                short_gcn_loss = 0
                for gconv in self.model.short_conv_modulelist:
                    w = torch.matmul(gconv.att_r, gconv.basis.view(gconv.num_bases, -1)).view(gconv.num_relations, gconv.in_channels, gconv.out_channels)
                    short_gcn_loss = short_gcn_loss + torch.sum((w[1:, :, :] - w[:-1, :, :])**2)
                short_gcn_loss = short_gcn_loss / len(self.model.short_conv_modulelist)

                # Combine GCN short term and long term loss
                gcn_loss = gcn_loss + short_gcn_loss

                # BPR loss
                loss = - torch.log(torch.sigmoid(targets_pred - negatives_pred) + 1e-8)
                loss = torch.mean(loss)

                # Total loss = BPR loss + RAGCN loss
                loss = loss + gcn_loss

                if self.arg.block_backprop:
                    loss = loss * 0  # needed in case to block backpropagation

                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                
                epoch_loss = epoch_loss + (loss.item() * (end_index - start_index))

            epoch_loss = epoch_loss / train_num
            time_ = time.time()
            _, rem = divmod(time_-start, 3600)
            minutes, seconds = divmod(rem, 60)
            time_str = "{:0>2}m{:0>2}s".format(int(minutes), int(seconds))

            precision, recall, MAP, ndcg, hr = self.evaluate(users_np_test, sequences_np_test, test_set, return_metrics=True)
            
            loss_list, p, r, map_, ndcg, hr, p2, r2, map2, ndcg2, hr2 = \
                (lst + [value] for lst, value in zip(
                    (loss_list, p, r, map_, ndcg, hr, p2, r2, map2, ndcg2, hr2), 
                    (epoch_loss, precision[0], recall[0], MAP[0], ndcg[0], hr[0], precision[1], recall[1], MAP[1], ndcg[1], hr[1])
                ))

            
            tqdm.write(f'Epoch {epoch_+1}/{self.arg.epoch_num} - Loss: {epoch_loss:.4f} - Precision@10: {precision[0]:.4f} - Recall@10: {recall[0]:.4f} - MAP@10: {MAP[0]:.4f} - NDCG@10: {ndcg[0]:.4f} - HR@10: {hr[0]:.4f} - Time: {time_str}')

        for metric, metric_name in zip([loss_list,p, r, map_, ndcg, hr, p2, r2, map2, ndcg2, hr2], 
                                       ['Loss', 'Precision@10', 'Recall@10', 'MAP@10', 'NDCG@10', 'HR@10', 'Precision@20', 'Recall@20', 'MAP@20', 'NDCG@20', 'HR@20']):
            self.plot_metric(metric, metric_name)
        
    def evaluate(self, users_np_test, sequences_np_test, test_set=None, return_metrics=False):  # TODO: recheck this function
        
        assert not return_metrics or test_set is not None, 'test_set must be provided if return_metrics is True'
        self.model.eval()
        short_term_window_size = int(self.arg.L / self.short_term_window_num)
        short_term_window = [0] + [i+short_term_window_size for i in range(self.short_term_window_num-1)] + [-1]
        batch_num = int(users_np_test.shape[0]/self.arg.batch_size) + 1
        data_index = np.arange(users_np_test.shape[0])
        self.pred_list = None
        with torch.no_grad():
            for batch_ in range(batch_num):
                start_index, end_index = batch_ * self.arg.batch_size, (batch_+1)*self.arg.batch_size
                batch_record_index = data_index[start_index: end_index]

                batch_users = users_np_test[batch_record_index]
                batch_sequences = sequences_np_test[batch_record_index]

                batch_users, batch_sequences, edge_index, edge_type, node_no, node2ids = self.subgraph_extractor.extract(batch_users, batch_sequences, sub_seq_no=None)

                short_term_part = []
                for i in range(len(short_term_window)):
                    if i != len(short_term_window)-1:
                        sub_seq_no = batch_sequences[:,short_term_window[i]:short_term_window[i+1]]
                        edge_index, edge_type= self.subgraph_extractor.extract(batch_users, batch_sequences, sub_seq_no=sub_seq_no, node2ids=node2ids, short_term=True)
                        short_term_part.append((edge_index, edge_type))

                batch_users = torch.tensor(batch_users).to(self.device)
                batch_sequences = torch.from_numpy(batch_sequences).type(torch.LongTensor).to(self.device)

                X_user_item = [batch_users, batch_sequences, self.item_indexes]
                X_graph_base = [edge_index, edge_type, node_no, short_term_part]

                rating_pred = self.model(X_user_item, X_graph_base, for_pred=True)

                rating_pred = rating_pred.cpu().data.numpy().copy()

                ind = np.argpartition(rating_pred, -self.arg.topk)
                ind = ind[:, -self.arg.topk:]
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                if batch_ == 0:
                    self.pred_list = batch_pred_list
                else:
                    self.pred_list = np.append(self.pred_list, batch_pred_list, axis=0)

        if return_metrics:
            precision, recall, MAP, ndcg, hr = [], [], [], [], []
            for k in [10, 20]:
                precision.append(precision_at_k(test_set, self.pred_list, k))
                recall.append(recall_at_k(test_set, self.pred_list, k))
                MAP.append(mapk(test_set, self.pred_list, k))
                ndcg.append(ndcg_k(test_set, self.pred_list, k))
                hr.append(hit_ratio_at_k(test_set, self.pred_list, k))
            return precision, recall, MAP, ndcg, hr
        
        return self.pred_list
        
    def save_model(self, model_ckpt):
        torch.save(self.model.state_dict(), model_ckpt)
    
    def plot_metric(self, metric, metric_name):
        plt.style.use('ggplot')
        plt.plot(metric)
        plt.xlabel('Epochs')
        plt.ylabel(metric_name)
        plt.savefig(self.arg.plot_dir + '/' + metric_name + '.png')
