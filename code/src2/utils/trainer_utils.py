import random
from matplotlib import pyplot as plt
import numpy as np
import torch
from model.CAGSRec import CAGSRec
from utils.seed import seed_everything

def build_model(config, item_num, node_num, relation_num, logger):
    seed_everything(config.seed)
    model = CAGSRec(config, item_num, node_num, relation_num).to(config.device)
    
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=config.learning_rate, 
                                     weight_decay=config.l2)
    elif config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), 
                                      lr=config.learning_rate, 
                                      weight_decay=config.l2)
    else:
        raise Exception('Unknown optimizer {}'.format(config.optimizer))
        
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    
    if config.verbose:
<<<<<<< HEAD
        logger.info(f'Model:\t{model.__class__.__name__} with {config.model_variant} variant')
=======
        logger.info(f'Model:\t{model.__class__.__name__} with variant {config.model_variant}')
>>>>>>> 5f39a5b309fbd7211589578d1f40ba9a89413c2a
        logger.info(model)
        logger.info(f'Optimizer:\t{optimizer.__class__.__name__} with initial lr = {config.learning_rate}, l2 = {config.l2}')
        logger.info(f'LR Scheduler:\t{lr_scheduler.__class__.__name__}')
    
    return model, optimizer, lr_scheduler


def load_model_from_ckpt(config, model_ckpt):
    model = CAGSRec(config)
    model.load_state_dict(torch.load(model_ckpt))
    return model


def Negative_Sampling(H, user2item, batch_users, item_set):
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
        negtive_list_ = random.sample(sorted(difference_set), H)
        negatives_np.append(negtive_list_)
    negatives_np = np.array(negatives_np)
    return negatives_np


def Extract_SUBGraph(v2vc, u2v, v2u, device, user_no, seq_no, sub_seq_no=None, node2ids=None, hop=2):
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
                    vc = v2vc[seq_no[j, k]]
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
                            vc = v2vc[origin_seq_no[j, k]]
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
                u_list = v2u[update_set[j]]
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
                v_list = u2v[update_set[j]]
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

    edge_index = torch.t(torch.tensor(edge_index).to(device))
    edge_type = torch.tensor(edge_type).to(device)
    node_no = torch.tensor(sorted(list(node2ids.values()))).to(device)

    # new_user_no,new_seq_no
    new_user_no, new_seq_no = list(), list()
    for i in range(user_no.shape[0]):
        new_user_no.append(node2ids[user_no[i]])
    for i in range(seq_no.shape[0]):
        new_seq_no.append([node2ids[seq_no[i, j]]
                            for j in range(seq_no[i, :].shape[0])])
    new_user_no, new_seq_no = np.array(new_user_no), np.array(new_seq_no)

    return new_user_no, new_seq_no, edge_index, edge_type, node_no, node2ids


def plot_metric(config, metric, metric_name):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(metric, label=metric_name)
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.savefig(config.plot_dir + '/' + metric_name + '.png')
