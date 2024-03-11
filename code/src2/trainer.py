import numpy as np
import torch
torch.cuda.empty_cache()

from utils.eval_utils import precision_at_k, recall_at_k, mapk, ndcg_k, hit_ratio_at_k
from utils.callbacks import EarlyStoppingCallback
from utils.trainer_utils import build_model, Negative_Sampling, Extract_SUBGraph, plot_metric


class Trainer:
    """
    Trainer class for CAGSRec model
    """
    def __init__(self, config, info, edges, device, logger):
        
        node_num, relation_num = info['node_num'], info['relation_num']
        u2v, u2vc, v2u, v2vc = edges
        
        self.u2v, self.u2vc, self.v2u, self.v2vc = u2v, u2vc, v2u, v2vc
        
        v_list_ = list(self.u2v.values())
        v_list_ = sorted(set([v_list_[i][j] for i in range(len(v_list_)) for j in range(len(v_list_[i]))]))
        
        self.item_indexes = torch.tensor(v_list_).to(device)
        item_num = len(v_list_)
        self.item_set = set(self.v2u.keys())
        
        self.gnn_sr_model, self.optimizer, self.lr_scheduler = build_model(config, item_num, node_num, relation_num, logger)
  
        self.device = device
        self.arg = config
        
        self.node_num = node_num
        self.short_term_window_num = 3
        self.pred_list = None
        self.K_eval_list = [10]
        
        self.logger = logger

        
    def train_for_epoch(self, users_np, sequences_np_train, record_indexes, train_num, short_term_window, batch_num):
        self.optimizer.zero_grad()
        self.gnn_sr_model.train()

        epoch_loss = 0.0
        for batch_ in range(batch_num):
            start_index, end_index = batch_ * self.arg.batch_size, (batch_+1) * self.arg.batch_size
            batch_record_index = record_indexes[start_index: end_index]

            batch_users = users_np[batch_record_index]
            batch_neg = Negative_Sampling(self.arg.H, self.u2v, batch_users, self.item_set)

            batch_sequences_train = sequences_np_train[batch_record_index]
            batch_sequences, batch_targets = batch_sequences_train[:, :self.arg.L], batch_sequences_train[:, self.arg.L:]

            # Extracting SUBGraph (long term)
            batch_users, batch_sequences, edge_index, edge_type, node_no, node2ids = Extract_SUBGraph(self.v2vc, self.u2v, self.v2u, self.arg.device, batch_users, batch_sequences, sub_seq_no=None)

            # Extracting SUBGraph (short term)
            short_term_part = []
            for i in range(len(short_term_window)):
                if i != len(short_term_window)-1:
                    sub_seq_no = batch_sequences[:, short_term_window[i]:short_term_window[i+1]]
                    _, _, edge_index, edge_type, _, _ = Extract_SUBGraph(self.v2vc, self.u2v, self.v2u, self.arg.device ,batch_users, batch_sequences, sub_seq_no=sub_seq_no, node2ids=node2ids)
                    short_term_part.append((edge_index, edge_type))

            batch_users = torch.tensor(batch_users).to(self.device)
            batch_sequences = torch.from_numpy(batch_sequences).type(torch.LongTensor).to(self.device)
            batch_targets = torch.from_numpy(batch_targets).type(torch.LongTensor).to(self.device)
            batch_negatives = torch.from_numpy(batch_neg).type(torch.LongTensor).to(self.device)

            items_to_predict = torch.cat((batch_targets, batch_negatives), 1)

            X_user_item = [batch_users, batch_sequences, items_to_predict]
            X_graph_base = [edge_index, edge_type, node_no, short_term_part]

            pred_score, _, _ = self.gnn_sr_model(X_user_item, X_graph_base, for_pred=False)

            (targets_pred, negatives_pred) = torch.split(pred_score, [batch_targets.size(1), batch_negatives.size(1)], dim=1)

            # Total loss = BPR loss + RAGCN loss
            loss = self.calc_bpr_loss(targets_pred, negatives_pred) + self.calc_gnn_loss()

            if self.arg.block_backprop:
                loss = loss * 0  # needed in case to block backpropagation

            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            epoch_loss = epoch_loss + (loss.item() * (end_index - start_index))

        epoch_loss /= train_num
        return epoch_loss


    def train(self, train_part, test_part):

        earlystopper = EarlyStoppingCallback(patience=10, min_delta=0.0)

        loss_list = list()
        precision_list, recall_list, MAP_list, ndcg_list, hr_list = list(), list(), list(), list(), list()

        users_np, sequences_np_train = train_part[0], train_part[1]
        sequences_np_test, test_set, users_np_test = test_part[0], test_part[1], np.array(test_part[2])

        train_num = users_np.shape[0]
        record_indexes = np.arange(train_num)

        epoch_loss = 0.0
        batch_num = int(train_num/self.arg.batch_size) + 1

        # short term part
        short_term_window_size = int(self.arg.L / self.short_term_window_num)
        short_term_window = [0] + [i + short_term_window_size for i in range(self.short_term_window_num-1)] + [-1]

        for epoch_ in range(self.arg.epoch_num):

                # train
                epoch_loss = self.train_for_epoch(users_np, sequences_np_train, record_indexes, train_num, short_term_window, batch_num)

                with torch.no_grad():
                    self.gnn_sr_model.eval()
                    precision, recall, MAP, ndcg, hr = self.evaluate(users_np_test, sequences_np_test, test_set)
                
                self.logger.info(f'Epoch {epoch_+1}/{self.arg.epoch_num}:\tLoss: {epoch_loss:.4f}\tP@10: {precision[0]:.4f}\tR@10: {recall[0]:.4f}\tMAP@10: {MAP[0]:.4f}\tNDCG@10: {ndcg[0]:.4f}\tHR@10: {hr[0]:.4f}')

                loss_list.append(epoch_loss)
                precision_list.append(precision[0])
                recall_list.append(recall[0])
                MAP_list.append(MAP[0])
                ndcg_list.append(ndcg[0])
                hr_list.append(hr[0])
                
                if earlystopper.call(epoch_loss):
                    self.logger.info(f'Early stopping at epoch {epoch_+1}')
                    break
            
        metric_list = [loss_list, precision_list, recall_list, MAP_list, ndcg_list, hr_list]
        metric_names = ['Loss', 'Precision@10', 'Recall@10', 'MAP@10', 'NDCG@10', 'HR@10']
        for metric, metric_name in zip(metric_list, metric_names):
            plot_metric(self.arg, metric, metric_name)


    def calc_bpr_loss(self, targets_pred, negatives_pred):
        loss = - torch.log(torch.sigmoid(targets_pred - negatives_pred) + 1e-8)
        loss = torch.mean(loss)
        return loss


    def calc_gnn_loss(self):
        return self.calc_long_term_loss() + self.calc_short_term_loss()

    
    def calc_long_term_loss(self):
        # RAGCN loss (long term)
        gcn_loss = 0
        for gconv in self.gnn_sr_model.long_term_gnn.conv_modulelist:
            w = torch.matmul(gconv.att_r, gconv.basis.view(gconv.num_bases, -1)).view(gconv.num_relations, gconv.in_channels, gconv.out_channels)
            gcn_loss = gcn_loss + torch.sum((w[1:, :, :] - w[:-1, :, :])**2)
        gcn_loss = gcn_loss/len(self.gnn_sr_model.long_term_gnn.conv_modulelist)
        return gcn_loss


    def calc_short_term_loss(self):
        # RAGCN loss (short term)
        short_gcn_loss = 0
        for gconv in self.gnn_sr_model.short_term_gnn.conv_modulelist:
            w = torch.matmul(gconv.att_r, gconv.basis.view(gconv.num_bases, -1)).view(gconv.num_relations, gconv.in_channels, gconv.out_channels)
            short_gcn_loss = short_gcn_loss +  torch.sum((w[1:, :, :] - w[:-1, :, :])**2)
        short_gcn_loss = short_gcn_loss / len(self.gnn_sr_model.short_term_gnn.conv_modulelist)
        return short_gcn_loss


    def predict(self, user_id, sequences_np, item_indexes):
        short_term_window_size = int(self.arg.L / self.short_term_window_num)
        short_term_window = [0] + [i+short_term_window_size for i in range(self.short_term_window_num-1)] + [-1]
        batch_num = int(sequences_np.shape[0]/self.arg.batch_size)+1
        data_index = np.arange(sequences_np.shape[0])
        
        for batch_ in range(batch_num):
            
            start_index, end_index = batch_ * self.arg.batch_size, (batch_+1) * self.arg.batch_size
            batch_record_index = data_index[start_index: end_index]

            batch_users = np.array([user_id] * len(batch_record_index))
            batch_sequences = sequences_np[batch_record_index]

            # Extracting SUBGraph (long term)
            batch_users, batch_sequences, edge_index, edge_type, node_no, node2ids = Extract_SUBGraph(self.v2vc, self.u2v, self.v2u, self.arg.device, user_no=batch_users, seq_no=batch_sequences, sub_seq_no=None)

            # Extracting SUBGraph (short term)
            short_term_part = []
            for i in range(len(short_term_window)):
                if i != len(short_term_window)-1:
                    sub_seq_no = batch_sequences[:, short_term_window[i]:short_term_window[i+1]]
                    _, _, edge_index, edge_type, _, _ = Extract_SUBGraph(self.v2vc, self.u2v, self.v2u, self.arg.device , user_no=batch_users, seq_no=batch_sequences, sub_seq_no=sub_seq_no, node2ids=node2ids)
                    short_term_part.append((edge_index, edge_type))

            batch_users = torch.tensor(batch_users).to(self.device)
            batch_sequences = torch.from_numpy(batch_sequences).type(torch.LongTensor).to(self.device)

            X_user_item = [batch_users, batch_sequences, item_indexes]
            X_graph_base = [edge_index, edge_type, node_no, short_term_part]

            rating_pred = self.gnn_sr_model(X_user_item, X_graph_base, for_pred=True)

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
                
        return self.pred_list


    def evaluate(self, users_np_test, sequences_np_test, test_set):

        short_term_window_size = int(self.arg.L / self.short_term_window_num)
        short_term_window = [0] + [i+short_term_window_size for i in range(self.short_term_window_num-1)] + [-1]
        batch_num = int(users_np_test.shape[0]/self.arg.batch_size)+1 # number of batches for validation
        data_index = np.arange(users_np_test.shape[0])
        
        for batch_ in range(batch_num):
            start_index, end_index = batch_ * self.arg.batch_size, (batch_+1) * self.arg.batch_size
            batch_record_index = data_index[start_index: end_index]

            batch_users = users_np_test[batch_record_index]
            batch_sequences = sequences_np_test[batch_record_index]

            # Extracting SUBGraph (long term)
            batch_users, batch_sequences, edge_index, edge_type, node_no, node2ids = Extract_SUBGraph(self.v2vc, self.u2v, self.v2u, self.arg.device, user_no=batch_users, seq_no=batch_sequences, sub_seq_no=None)

            # Extracting SUBGraph (short term)
            short_term_part = []
            for i in range(len(short_term_window)):
                if i != len(short_term_window)-1:
                    sub_seq_no = batch_sequences[:, short_term_window[i]:short_term_window[i+1]]
                    _, _, edge_index, edge_type, _, _ = Extract_SUBGraph(self.v2vc, self.u2v, self.v2u, self.arg.device , user_no=batch_users, seq_no=batch_sequences, sub_seq_no=sub_seq_no, node2ids=node2ids)
                    short_term_part.append((edge_index, edge_type))

            batch_users = torch.tensor(batch_users).to(self.device)
            batch_sequences = torch.from_numpy(batch_sequences).type(torch.LongTensor).to(self.device)

            X_user_item = [batch_users, batch_sequences, self.item_indexes]
            X_graph_base = [edge_index, edge_type, node_no, short_term_part]

            rating_pred = self.gnn_sr_model(X_user_item, X_graph_base, for_pred=True)

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

        precision, recall, MAP, ndcg, hr = [], [], [], [], []
        for k in self.K_eval_list:
            precision.append(precision_at_k(test_set, self.pred_list, k))
            recall.append(recall_at_k(test_set, self.pred_list, k))
            MAP.append(mapk(test_set, self.pred_list, k))
            ndcg.append(ndcg_k(test_set, self.pred_list, k))
            hr.append(hit_ratio_at_k(test_set, self.pred_list, k))
        return precision, recall, MAP, ndcg, hr
    
          
    def save_model(self, path_name):
        torch.save(self.gnn_sr_model.state_dict(), path_name)