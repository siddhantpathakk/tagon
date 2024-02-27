import numpy as np
import random
import time
import matplotlib.pyplot as plt
import torch
from .models.model import GNN_SR_Net
from .models.GCN import GCN
from .utils.metric import precision_at_k, recall_at_k, mapk, ndcg_k, hit_ratio_at_k


class EarlyStoppingCallback:
    """
    Early stopping callback
    """

    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best = float('inf')

    def call(self, loss):
        """
        Call the early stopping callback

        Args:
            loss (float): loss value

        Returns:
            bool: whether to stop or not
        """
        if loss < self.best - self.min_delta:
            self.best = loss
            self.wait = 0
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
            return False


class Trainer:
    """
    Trainer class
    """
    def __init__(self, config, node_num, relation_num, u2v, u2vc, v2u, v2vc, device):
        self.u2v, self.u2vc, self.v2u, self.v2vc = u2v, u2vc, v2u, v2vc
        v_list_ = list(self.u2v.values())
        v_list_ = sorted(set([v_list_[i][j] for i in range(
            len(v_list_)) for j in range(len(v_list_[i]))]))
        self.item_indexes = torch.tensor(v_list_).to(device)
        item_num = len(v_list_)
        self.item_set = set(self.v2u.keys())
        self.gnn_sr_model = GNN_SR_Net(
            config, item_num, node_num, relation_num, gcn=GCN, device=device)
        self.device = device
        self.arg = config
        self.node_num = node_num
        self.short_term_window_num = 3
        self.pred_list = None

        self.optimizer = torch.optim.Adam(self.gnn_sr_model.parameters(),
                                          lr=config.learning_rate,
                                          weight_decay=config.l2)

        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer)

    def Negative_Sampling(self, user2item, batch_users, item_set):
        """
        Negative sampling

        Args:
            user2item (dict): user to item mapping
            batch_users (np.array): batch of users
            item_set (set): set of items

        Returns:
            np.array: negative samples
        """
        negatives_np = list()
        for i in range(batch_users.shape[0]):
            user_item_set = set(user2item[batch_users[i]])
            difference_set = item_set - user_item_set
            negtive_list_ = random.sample(sorted(difference_set), self.arg.H)
            negatives_np.append(negtive_list_)
        negatives_np = np.array(negatives_np)
        return negatives_np

    def Extract_SUBGraph(self, user_no, seq_no, sub_seq_no=None, node2ids=None, hop=2):
        '''
         vc (O)     vc(X)       vc(O)    
        u{v1,.. => {u1,... => {v5,v6,... 
        _______    _______    _______ ....
         0-hop      1-hop      2-hop

         edge type: uv 0 ; vu 1 ; vvc 2 ;  vcv 3
        '''
        if sub_seq_no is not None:
            origin_seq_no = seq_no
            seq_no = sub_seq_no

        if node2ids is None:
            node2ids = dict()

        edge_index, edge_type = list(), list()
        update_set, index, memory_set = list(), 0, list()
        for i in range(hop):
            if i == 0:
                # uv
                for j in range(user_no.shape[0]):
                    if user_no[j] not in node2ids:
                        node2ids[user_no[j]] = index
                        index += 1
                    user_node_ids = node2ids[user_no[j]]
                    for k in range(seq_no[j, :].shape[0]):
                        if seq_no[j, k] not in node2ids:
                            node2ids[seq_no[j, k]] = index
                            index += 1
                        item_node_ids = node2ids[seq_no[j, k]]
                        edge_index.append([user_node_ids, item_node_ids])
                        edge_type.append(0)
                        edge_index.append([item_node_ids, user_node_ids])
                        edge_type.append(1)
                        # vc
                        vc = self.v2vc[seq_no[j, k]]
                        if vc not in node2ids:
                            node2ids[vc] = index
                            index += 1
                        vc_node_ids = node2ids[vc]
                        edge_index.append([item_node_ids, vc_node_ids])
                        edge_type.append(2)
                        edge_index.append([vc_node_ids, item_node_ids])
                        edge_type.append(3)
                        # update
                        update_set.append(seq_no[j, k])
                        # memory
                        memory_set.append(user_no[j])

                update_set = list(set(update_set))  # v
                memory_set = set(memory_set)  # u

                if sub_seq_no is not None:
                    for j in range(user_no.shape[0]):
                        user_node_ids = node2ids[user_no[j]]
                        for k in range(origin_seq_no[j, :].shape[0]):
                            if origin_seq_no[j, k] not in node2ids:
                                node2ids[origin_seq_no[j, k]] = index
                                index += 1
                            if origin_seq_no[j, k] not in set(update_set):
                                item_node_ids = node2ids[origin_seq_no[j, k]]
                                edge_index.append(
                                    [user_node_ids, item_node_ids])
                                edge_type.append(0)
                                edge_index.append(
                                    [item_node_ids, user_node_ids])
                                edge_type.append(1)
                                # vc
                                vc = self.v2vc[origin_seq_no[j, k]]
                                if vc not in node2ids:
                                    node2ids[vc] = index
                                    index += 1
                                vc_node_ids = node2ids[vc]
                                edge_index.append([item_node_ids, vc_node_ids])
                                edge_type.append(2)
                                edge_index.append([vc_node_ids, item_node_ids])
                                edge_type.append(3)

            elif i != 0 and i % 2 != 0:
                # vu
                new_update_set = list()
                for j in range(len(update_set)):
                    item_node_ids = node2ids[update_set[j]]
                    u_list = self.v2u[update_set[j]]
                    for k in range(len(u_list)):
                        if u_list[k] not in node2ids:
                            node2ids[u_list[k]] = index
                            index += 1
                        if u_list[k] not in memory_set:
                            user_node_ids = node2ids[u_list[k]]
                            edge_index.append([item_node_ids, user_node_ids])
                            edge_type.append(1)
                            edge_index.append([user_node_ids, item_node_ids])
                            edge_type.append(0)
                            new_update_set.append(u_list[k])
                memory_set = set(update_set)  # v
                update_set = new_update_set  # u
            elif i != 0 and i % 2 == 0:
                # uv
                for j in range(len(update_set)):
                    user_node_ids = node2ids[update_set[j]]
                    v_list = self.u2v[update_set[j]]
                    for k in range(len(v_list)):
                        if v_list[k] not in node2ids:
                            node2ids[v_list[k]] = index
                            index += 1
                        if v_list[k] not in memory_set:
                            item_node_ids = node2ids[v_list[k]]
                            edge_index.append([item_node_ids, user_node_ids])
                            edge_type.append(1)
                            edge_index.append([user_node_ids, item_node_ids])
                            edge_type.append(0)
                            new_update_set.append(v_list[k])
                memory_set = set(update_set)  # u
                update_set = new_update_set  # v

        edge_index = torch.t(torch.tensor(edge_index).to(self.device))
        edge_type = torch.tensor(edge_type).to(self.device)
        node_no = torch.tensor(sorted(list(node2ids.values()))).to(self.device)

        # new_user_no,new_seq_no
        new_user_no, new_seq_no = list(), list()
        for i in range(user_no.shape[0]):
            new_user_no.append(node2ids[user_no[i]])
        for i in range(seq_no.shape[0]):
            new_seq_no.append([node2ids[seq_no[i, j]]
                              for j in range(seq_no[i, :].shape[0])])
        new_user_no, new_seq_no = np.array(new_user_no), np.array(new_seq_no)

        return new_user_no, new_seq_no, edge_index, edge_type, node_no, node2ids

    def Evaluation(self, users_np_test, sequences_np_test, test_set):
        # TODO: recheck this function

        # short term part
        short_term_window_size = int(self.arg.L / self.short_term_window_num)
        short_term_window = [
            0] + [i+short_term_window_size for i in range(self.short_term_window_num-1)] + [-1]
        batch_num = int(users_np_test.shape[0]/self.arg.batch_size)+1
        data_index = np.arange(users_np_test.shape[0])
        for batch_ in range(batch_num):
            start_index, end_index = batch_ * \
                self.arg.batch_size, (batch_+1)*self.arg.batch_size
            batch_record_index = data_index[start_index: end_index]

            batch_users = users_np_test[batch_record_index]
            batch_sequences = sequences_np_test[batch_record_index]

            # Extracting SUBGraph (long term)
            batch_users, batch_sequences, edge_index, edge_type, node_no, node2ids = self.Extract_SUBGraph(
                batch_users, batch_sequences, sub_seq_no=None)

            # Extracting SUBGraph (short term)
            short_term_part = []
            for i in range(len(short_term_window)):
                if i != len(short_term_window)-1:
                    sub_seq_no = batch_sequences[:,
                                                 short_term_window[i]:short_term_window[i+1]]
                    _, _, edge_index, edge_type, _, _ = self.Extract_SUBGraph(
                        batch_users, batch_sequences, sub_seq_no=sub_seq_no, node2ids=node2ids)
                    short_term_part.append((edge_index, edge_type))

            batch_users = torch.tensor(batch_users).to(self.device)
            batch_sequences = torch.from_numpy(batch_sequences).type(
                torch.LongTensor).to(self.device)

            X_user_item = [batch_users, batch_sequences, self.item_indexes]
            X_graph_base = [edge_index, edge_type, node_no, short_term_part]

            rating_pred = self.gnn_sr_model(
                X_user_item, X_graph_base, for_pred=True)

            attn_weight_list = self.gnn_sr_model.attn_weight_list
            self._Eval_Draw_Graph_(batch_users, test_set, rating_pred,
                                   edge_index, edge_type, attn_weight_list, node2ids, topk=10)

            rating_pred = rating_pred.cpu().data.numpy().copy()

            ind = np.argpartition(rating_pred, -self.arg.topk)
            ind = ind[:, -self.arg.topk:]
            arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
            arr_ind_argsort = np.argsort(
                arr_ind)[np.arange(len(rating_pred)), ::-1]
            batch_pred_list = ind[np.arange(len(rating_pred))[
                :, None], arr_ind_argsort]

            if batch_ == 0:
                self.pred_list = batch_pred_list
            else:
                self.pred_list = np.append(
                    self.pred_list, batch_pred_list, axis=0)

        precision, recall, MAP, ndcg, hr = [], [], [], [], []
        for k in [10, 20]:
            precision.append(precision_at_k(test_set, self.pred_list, k))
            recall.append(recall_at_k(test_set, self.pred_list, k))
            MAP.append(mapk(test_set, self.pred_list, k))
            ndcg.append(ndcg_k(test_set, self.pred_list, k))
            hr.append(hit_ratio_at_k(test_set, self.pred_list, k))
        return precision, recall, MAP, ndcg, hr

    def save_model(self, path_name):
        torch.save(self.gnn_sr_model.state_dict(), path_name)

    def train(self, train_part, test_part):

        earlystopper = EarlyStoppingCallback(patience=10, min_delta=0.0)

        if self.arg.plot or self.arg.debug:
            loss_list = list()
            precision_list, recall_list, MAP_list, ndcg_list, hr_list = list(), list(), list(), list(), list()
            precision_list_20, recall_list_20, MAP_list_20, ndcg_list_20, hr_list_20 = list(), list(), list(), list(), list()

        users_np, sequences_np_train = train_part[0], train_part[1]
        sequences_np_test, test_set, uid_list_ = test_part[0], test_part[1], test_part[2]
        _ = test_part[-1]
        users_np_test = np.array(uid_list_)

        train_num = users_np.shape[0]
        record_indexes = np.arange(train_num)

        epoch_loss = 0.0
        batch_num = int(train_num/self.arg.batch_size) + 1

        # short term part
        short_term_window_size = int(self.arg.L / self.short_term_window_num)
        short_term_window = [
            0] + [i + short_term_window_size for i in range(self.short_term_window_num-1)] + [-1]

        print('Epoch\tTime\tLoss\t\tP@10\tR@10\tMAP@10\tNDCG@10\tHR@10\t\tP@20\tR@20\tMAP@20\tNDCG@20\tHR@20')

        for epoch_ in range(self.arg.epoch_num):
            self.optimizer.zero_grad()
            self.gnn_sr_model.train()

            start = time.time()

            user_emd_batch_list, item_emd_batch_list = list(), list()

            for batch_ in range(batch_num):
                start_index, end_index = batch_ * \
                    self.arg.batch_size, (batch_+1) * self.arg.batch_size
                batch_record_index = record_indexes[start_index: end_index]

                batch_users = users_np[batch_record_index]
                batch_neg = self.Negative_Sampling(
                    self.u2v, batch_users, self.item_set)

                batch_sequences_train = sequences_np_train[batch_record_index]
                batch_sequences, batch_targets = batch_sequences_train[:,
                                                                       :self.arg.L], batch_sequences_train[:, self.arg.L:]

                # Extracting SUBGraph (long term)
                batch_users, batch_sequences, edge_index, edge_type, node_no, node2ids = self.Extract_SUBGraph(
                    batch_users, batch_sequences, sub_seq_no=None)

                # Extracting SUBGraph (short term)
                short_term_part = []
                for i in range(len(short_term_window)):
                    if i != len(short_term_window)-1:
                        sub_seq_no = batch_sequences[:,
                                                     short_term_window[i]:short_term_window[i+1]]
                        _, _, edge_index, edge_type, _, _ = self.Extract_SUBGraph(
                            batch_users, batch_sequences, sub_seq_no=sub_seq_no, node2ids=node2ids)
                        short_term_part.append((edge_index, edge_type))

                batch_users = torch.tensor(batch_users).to(self.device)
                batch_sequences = torch.from_numpy(batch_sequences).type(torch.LongTensor).to(self.device)
                batch_targets = torch.from_numpy(batch_targets).type(torch.LongTensor).to(self.device)
                batch_negatives = torch.from_numpy(batch_neg).type(torch.LongTensor).to(self.device)

                items_to_predict = torch.cat((batch_targets, batch_negatives), 1)

                X_user_item = [batch_users, batch_sequences, items_to_predict]
                X_graph_base = [edge_index, edge_type, node_no, short_term_part]

                pred_score, user_emb, item_embs_conv = self.gnn_sr_model(X_user_item, X_graph_base, for_pred=False)

                user_emd_batch_list.append(user_emb)
                item_emd_batch_list.append(item_embs_conv)

                (targets_pred, negatives_pred) = torch.split(pred_score, [batch_targets.size(1), batch_negatives.size(1)], dim=1)

                # RAGCN loss (long term)
                gcn_loss = 0
                for gconv in self.gnn_sr_model.conv_modulelist:
                    w = torch.matmul(gconv.att_r, gconv.basis.view(
                        gconv.num_bases, -1)).view(gconv.num_relations, gconv.in_channels, gconv.out_channels)
                    gcn_loss = gcn_loss + \
                        torch.sum((w[1:, :, :] - w[:-1, :, :])**2)
                gcn_loss = gcn_loss/len(self.gnn_sr_model.conv_modulelist)

                # RAGCN loss (short term)
                short_gcn_loss = 0
                for gconv in self.gnn_sr_model.short_conv_modulelist:
                    w = torch.matmul(gconv.att_r, gconv.basis.view(
                        gconv.num_bases, -1)).view(gconv.num_relations, gconv.in_channels, gconv.out_channels)
                    short_gcn_loss = short_gcn_loss + \
                        torch.sum((w[1:, :, :] - w[:-1, :, :])**2)
                short_gcn_loss = short_gcn_loss / \
                    len(self.gnn_sr_model.short_conv_modulelist)

                # Combine GCN short term and long term loss
                gcn_loss = gcn_loss + short_gcn_loss

                # BPR loss
                loss = - torch.log(torch.sigmoid(targets_pred - negatives_pred) + 1e-8)
                loss = torch.mean(loss)  # loss = torch.mean(torch.sum(loss))

                # Total loss = BPR loss + RAGCN loss
                loss = loss + gcn_loss

                if self.arg.block_backprop:
                    loss = loss * 0  # needed in case to block backpropagation

                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                epoch_loss = epoch_loss + (loss.item() * (end_index - start_index))

            epoch_loss /= train_num
            self.epoch_ = epoch_

            _, rem = divmod(time.time()-start, 3600)
            minutes, seconds = divmod(rem, 60)
            time_str = "{:0>2}m{:0>2}s".format(int(minutes), int(seconds))

            if (epoch_ + 1) % 1 == 0:
                with torch.no_grad():
                    self.gnn_sr_model.eval()
                    precision, recall, MAP, ndcg, hr = self.Evaluation(users_np_test, sequences_np_test, test_set)

                    if self.arg.plot or self.arg.debug:
                        loss_list.append(epoch_loss)
                        precision_list.append(precision[0])
                        recall_list.append(recall[0])
                        MAP_list.append(MAP[0])
                        ndcg_list.append(ndcg[0])
                        hr_list.append(hr[0])

                        precision_list_20.append(precision[1])
                        recall_list_20.append(recall[1])
                        MAP_list_20.append(MAP[1])
                        ndcg_list_20.append(ndcg[1])
                        hr_list_20.append(hr[1])

                    print(
                        f'{epoch_+1}\t{time_str}\t{epoch_loss:.3f}\t\t{precision[0]:.3f}\t{recall[0]:.3f}\t{MAP[0]:.3f}\t{ndcg[0]:.3f}\t{hr[0]:.3f}\t\t{precision[1]:.3f}\t{recall[1]:.3f}\t{MAP[1]:.3f}\t{ndcg[1]:.3f}\t{hr[1]:.3f}')

            if earlystopper.call(epoch_loss):
                print(f'\n\nEarly stopping at epoch {epoch_+1}\n\n')
                break

        if not self.arg.debug:
            self.save_model(self.arg.out_path + 'saved_model.pt')

        if self.arg.plot or self.arg.debug:
            # plot all and save them in the out_path
            self._plot(loss_list, 'loss')
            self._plot(precision_list, 'precision@10')
            self._plot(recall_list, 'recall@10')
            self._plot(MAP_list, 'MAP@10')
            self._plot(ndcg_list, 'ndcg@10')
            self._plot(hr_list, 'hr@10')
            self._plot(precision_list_20, 'precision@20')
            self._plot(recall_list_20, 'recall@20')
            self._plot(MAP_list_20, 'MAP@20')
            self._plot(ndcg_list_20, 'ndcg@20')
            self._plot(hr_list_20, 'hr@20')

    def _plot(self, values, title):
        plt.figure()
        plt.plot(values)
        plt.title(title)
        plt.savefig(self.arg.log_path + title + '.png')