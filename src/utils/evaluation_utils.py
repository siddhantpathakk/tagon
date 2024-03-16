import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import multiprocessing


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num

def mean_reciprocal_rank(r):
    r = np.array(r)
    if np.sum(r) == 0:
        return 0.
    return np.reciprocal(np.where(r==1)[0]+1, dtype=np.float64)[0]


Ks = [10, 20]

def eval_one_user(x):    
    result = {
              'recall': np.zeros(len(Ks)), 
              'ndcg': np.zeros(len(Ks)),
                'mrr': 0.}
    
    preds = np.transpose(x[0])
    num_preditems = x[1]

    num_neg_sample_items = x[2]
    num_candidate_items = x[3]

    labels = np.zeros(num_preditems)
    labels[0] = 1
    r = []
    rankeditems = np.argsort(-preds)[:max(Ks)]
    for i in rankeditems:
        if i == 0:
            r.append(1)
        else:
            r.append(0)
    if num_neg_sample_items != -1:
        r = rank_corrected(np.array(r), num_preditems, num_candidate_items)

    recall, ndcg = [], []
    for K in Ks:
        recall.append(recall_at_k(r, K, 1))
        ndcg.append(ndcg_at_k(r, K))
    mrr = mean_reciprocal_rank(r)


    result['recall'] += recall
    result['ndcg'] += ndcg
    result['mrr'] += mrr
    return result


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


def eval_users(tgrec, src, dst, ts, train_src, train_dst, args):
    result = {
              'recall': np.zeros(len(Ks)), 
              'ndcg': np.zeros(len(Ks)),
                'mrr': 0.}
    
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
    batch_users = 1000

    preds_list, preds_len_preditems, preds_sampled_neg, preds_num_candidates = [], [], [], []


    num_interactions,num_test_instances = 0, 0

    with torch.no_grad():
        tgrec = tgrec.eval()
        batch_src_l = []
        batch_test_items = []
        batch_ts = []
        batch_i = 0
        
        for u, i, t in zip(src, dst, ts):
            
            num_test_instances += 1
            if u not in train_src or i not in train_itemset or u not in pos_edges:
                continue
            num_interactions += 1
            batch_i += 1

            pos_items = [i]
            pos_ts = [t]
            src_l = [u for _ in range(len(pos_items))]

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
            preds_sampled_neg.append(args.negsampleeval)
            preds_num_candidates.append(len(pos_items+neg_candidates))
            batch_src_l = []
            batch_test_items = []
            batch_ts = []

            if len(preds_list) % batch_users == 0 or num_test_instances == len(ts):

                batchset_predictions = zip(preds_list, preds_len_preditems, preds_sampled_neg, preds_num_candidates)
                batch_preds = pool.map(eval_one_user, batchset_predictions)
                for oneresult in batch_preds:
                    result['recall'] += oneresult['recall']
                    result['ndcg'] += oneresult['ndcg']
                    result['mrr'] += oneresult['mrr']
                    # print(result)

                preds_list, preds_len_preditems, preds_sampled_neg, preds_num_candidates = [], [], [], []
                batch_src_l, batch_test_items,batch_ts = [], [], []

    # print(result)
    result['recall'] /= num_interactions
    result['ndcg'] /= num_interactions
    result['mrr'] /= num_interactions
    # print(result)
    return result