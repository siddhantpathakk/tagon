import multiprocessing
import torch
import logging
import numpy as np
from utils.data_utils import Data
from utils.main_utils import parse_opt
from utils.trainer_utils import  build_model, get_new_history

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
    batch_users = 1

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
            #neg_items = list(train_itemset - set(pos_items))
            neg_ts = [t for _ in range(len(neg_items))]
            neg_src_l = [u for _ in range(len(neg_items))]

            batch_src_l += src_l + neg_src_l
            batch_test_items += pos_items + neg_items
            batch_ts += pos_ts + neg_ts
            #batch_len.append(len(src_l+neg_src_l))

            test_items = np.array(batch_test_items)
            test_ts = np.array(batch_ts)
            test_src_l = np.array(batch_src_l)

            pred_scores = tgrec(test_src_l, test_items, test_ts, args.n_degree)
            preds = pred_scores.cpu().numpy()
            #start_ind = 0
            #for i_len in batch_len:
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
                #batch_len = []

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
        self.model, _, _, _, self.device = build_model(self.args, self.data, self.logger)
        self.NUM_NEIGHBORS = self.args.n_degree
        
        self.history = get_new_history()
        
    
    def run(self, user_id):
        self.model.ngh_finder = self.data.full_ngh_finder
        src, dst, ts = self.get_src_dst_ts(user_id)
        return eval_user(self.model, src, dst, ts, 
                         self.data.train_src, self.data.train_dst, self.args)

    def get_src_dst_ts(self, user_id):
        src = self.data.test_src[self.data.test_user_id == user_id]
        dst = self.data.test_dst[self.data.test_user_id == user_id]
        ts = self.data.test_ts[self.data.test_user_id == user_id]
        return src, dst, ts
    
if __name__ == "__main__":
    
    args = parse_opt()
    data = Data(args.data, args)
    inference = InferenceEngine(args, data)
    
    
    user_id = str(input("Enter user id: "))
    print(inference.run(user_id))