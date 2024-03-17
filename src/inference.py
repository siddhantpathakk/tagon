import multiprocessing
from matplotlib import pyplot as plt
import torch
import logging
import time, datetime
import numpy as np
from src.utils.data_utils import Data
from src.utils.main_utils import parse_opt
import json
import pandas as pd
import networkx as nx

def eval_user(tgrec, src, dst, ts, train_src, train_dst, args):
    cores = multiprocessing.cpu_count() // 2
    train_itemset = set(train_dst)
    pos_edges = {}
    for u, i, t in zip(src, dst, ts):
        if i not in train_itemset:
            continue
        if u in pos_edges:
            pos_edges[u].add((i, t))
        else:
            pos_edges[u] = set([(i, t)])
    train_pos_edges = {}
    for u, i in zip(train_src, train_dst):
        if u in train_pos_edges:
            train_pos_edges[u].add(i)
        else:
            train_pos_edges[u] = set([i])

    pool = multiprocessing.Pool(cores)
    batch_users = 5

    preds_list = []
    preds_len_preditems = []
    preds_uit = []
    preds_rec_items = []
    preds_sampled_neg = []
    preds_num_candidates = []

    test_outputs = []

    num_interactions = 0
    num_test_instances = 0
    with torch.no_grad():
        tgrec = tgrec.eval()
        batch_src_l,batch_test_items, batch_ts, batch_i= [], [], [], 0
        for u, i, t in zip(src, dst, ts):
            num_test_instances += 1
            if u not in train_src or i not in train_itemset or u not in pos_edges:
                continue
            num_interactions += 1
            batch_i += 1

            pos_items = [i]
            pos_ts = [t]
            src_l = [u for _ in range(len(pos_items))]
            pos_label = np.ones(len(pos_items))

            interacted_dst = train_pos_edges[u]

            neg_candidates = list(train_itemset - set(pos_items) - interacted_dst)
            if args.negsampleeval == -1:
                neg_items = neg_candidates
            else:
                neg_items = list(np.random.choice(neg_candidates, size=args.negsampleeval, replace=False))
            neg_ts = [t for _ in range(len(neg_items))]
            neg_src_l = [u for _ in range(len(neg_items))]

            batch_src_l += src_l + neg_src_l
            batch_test_items += pos_items + neg_items
            batch_ts += pos_ts + neg_ts

            test_items = np.array(batch_test_items)
            test_ts = np.array(batch_ts)
            test_src_l = np.array(batch_src_l)

            pred_scores = tgrec(test_src_l, test_items, test_ts, args.n_degree)
            preds = pred_scores.cpu().numpy()
            preds_list.append(preds)
            preds_len_preditems.append(len(src_l+neg_src_l))
            preds_uit.append((u,i,t))
            rec_items = []
            rec_items += pos_items + neg_items
            preds_rec_items.append(rec_items)
            preds_sampled_neg.append(args.negsampleeval)
            preds_num_candidates.append(len(pos_items+neg_candidates))
            batch_src_l = []
            batch_test_items = []
            batch_ts = []

            if len(preds_list) % batch_users == 0 or num_test_instances == len(ts):

                batchset_predictions = zip(preds_list, preds_len_preditems, preds_uit, preds_rec_items, preds_sampled_neg, preds_num_candidates)
                batch_preds = pool.map(eval_one_user, batchset_predictions)
                for oneresult in batch_preds:
                    pred_rank_list, uit, rec_items = oneresult[1], oneresult[2], oneresult[3]

                    one_pred_result = {"u_ind": int(uit[0]), "u_pos_gd": int(uit[1]), "timestamp": float(uit[2])}
                    one_pred_result["predicted"] = [int(rec_items[int(rec_ind)]) for rec_ind in pred_rank_list]
                    test_outputs.append(one_pred_result)


                preds_list = []
                preds_len_preditems = []
                preds_uit = []
                preds_rec_items = []
                preds_sampled_neg = []
                preds_num_candidates = []
                batch_src_l = []
                batch_test_items = []
                batch_ts = []

    return test_outputs

def eval_one_user(x, Ks=[10]):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
            'hit_ratio': np.zeros(len(Ks)), 'auc': 0., 'mrr': 0.}
    preds = np.transpose(x[0])
    uit = x[2]
    rec_items = x[3]
    rankeditems = np.argsort(-preds)[:max(Ks)]
    return (rankeditems[:max(Ks)], uit, rec_items)

def rank_corrected(r, m, n):
    pos_ranks = np.argwhere(r==1)[:,0]
    corrected_r = np.zeros_like(r)
    for each_sample_rank in list(pos_ranks):
        corrected_rank = int(np.floor(((n-1)*each_sample_rank)/m))
        if corrected_rank >= len(corrected_r) - 1:
            continue
        corrected_r[corrected_rank] = 1
    assert sum(corrected_r) <= 1
    return corrected_r

class InferenceEngine:
    def __init__(self, args, data):
        self.logger = logging.getLogger()
        
        self.args = args
        self.data = data
        # self.model, _, _, _, self.device = build_model(self.args, self.data, self.logger)
        self.NUM_NEIGHBORS = self.args.n_degree
        
        # self.history = get_new_history()
        
        self.i_map = json.load(open('/home/FYP/siddhant005/fyp/processed/ml-100k_i_map.json'))
        self.u_map = json.load(open('/home/FYP/siddhant005/fyp/processed/ml-100k_u_map.json'))
        self.u_i_csv = pd.read_csv('/home/FYP/siddhant005/fyp/processed/ml_ml-100k.csv')
    
    def get_mapped_user_id(self, user_id):
        return self.u_map[str(user_id)]
     
    def run(self, user_id):
        # self.model.ngh_finder = self.data.full_ngh_finder
        
        real_uid = user_id
        mapped_uid = self.get_key_by_value(real_uid, self.u_map)
        
        print(f'The mapped user id {mapped_uid} is for {real_uid} real user id')
        
        return self.data.get_user_data(mapped_uid)
        # print(f'User {real_uid} has {len(src)} interactions')

        # print(src)
        # print(dst)
        # for it, ts_ in zip(dst, ts):
        #     print(f'\t({self.i_map[str(it)]["item_id"]})\t{self.i_map[str(it)]["title"]} at {datetime.datetime.fromtimestamp(ts_)}')
        # self.get_graph(src, dst, ts)
        # return eval_user(self.model, src, dst, ts, 
        #                  self.data.train_src_l, self.data.train_dst_l, self.args)
    
    def get_key_by_value(self, real_userid, dict_map):
        for key, value in dict_map.items():
            if value == real_userid:
                return key
            
    

        
if __name__ == "__main__":
    
    args = parse_opt()
    
    data = Data(args.data, args, split=False)
    inference = InferenceEngine(args, data)
    
    user_id = 259
    print(f'Running inference for user {user_id}...')
    print(inference.run(user_id))
