import torch
torch.cuda.empty_cache()
import numpy as np
import math

def Information_Entropy(prob_list):
    entropy_ = list()
    for i in range(len(prob_list)):
        if prob_list[i] == 0:
            prob_list_i = 0.0000000000000001
        else:
            prob_list_i = prob_list[i]
        entropy_i = prob_list_i*np.log(prob_list_i)
        entropy_.append(entropy_i)
    information_entropy = (-1)*sum(entropy_)
    return information_entropy


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def ndcg_k(actual, predicted, topk):
    k = min(topk, len(actual))
    idcg = idcg_k(k)
    res = 0
    for user_id in range(len(actual)):
        dcg_k = sum([int(predicted[user_id][j] in set(
            actual[user_id])) / math.log(j+2, 2) for j in range(k)])
        res += dcg_k / idcg
    return res / float(len(actual))


def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def hit_ratio_at_k(actual, predicted, topk):
    num_users = len(predicted)
    num_hits = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set & pred_set) != 0:
            num_hits += 1
    return num_hits / num_users


def mean_reciprocal_rank(actual, predicted):
    sum_mrr = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i])
        if len(act_set & pred_set) != 0:
            sum_mrr += 1 / (list(pred_set).index(list(act_set & pred_set)[0]) + 1)
    return sum_mrr / num_users
