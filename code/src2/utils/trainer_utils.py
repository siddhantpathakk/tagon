import random
import numpy as np
import torch

class NegativeSampler:
    def __init__(self, config, H):
        self.config = config
        self.H = H

    def sample(self, user2item, batch_users, item_set):
        negatives_np = list()
        for i in range(batch_users.shape[0]):
            user_item_set = set(user2item[batch_users[i]])
            difference_set = item_set - user_item_set
            negtive_list_ = random.sample(sorted(difference_set), self.arg.H)
            negatives_np.append(negtive_list_)
        negatives_np = np.array(negatives_np)
        return negatives_np


class SubGraphExtractor:
    def __init__(self, config, v2vc, v2u, u2v, hop=2):
        self.config = config
        self.device = self.config.device
        
        self.v2vc = v2vc
        self.v2u = v2u
        self.u2v = u2v

        self.hop = hop
        
    def extract(self, user_no, seq_no, sub_seq_no=None, node2ids=None, short_term=False):
        if sub_seq_no is not None:
            origin_seq_no = seq_no
            seq_no = sub_seq_no

        if node2ids is None:
            node2ids = dict()

        edge_index, edge_type = list(), list()
        update_set, index, memory_set = list(), 0, list()
        for i in range(self.hop):
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
                new_update_set, new_memory_set = list(), list()
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

        if short_term:
            return edge_index, edge_type

        return new_user_no, new_seq_no, edge_index, edge_type, node_no, node2ids
