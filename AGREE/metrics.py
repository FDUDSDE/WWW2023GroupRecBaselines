import math
import torch
import torch.nn as nn
import numpy as np


def eval_one_rating(model: nn.Module, test_ratings, test_negatives, device, k_list, type_m, idx):
    test_rating, test_items = test_ratings[idx], test_negatives[idx]
    test_items, test_user = [test_rating[1]] + test_items, test_rating[0]

    test_users = np.full(len(test_items), test_user)

    users_var = torch.from_numpy(test_users).long().to(device)
    items_var = torch.LongTensor(test_items).to(device)

    if type_m == 'group':
        predictions = model(users_var, None, items_var)
    else:
        predictions = model(None, users_var, items_var)

    pred_score = predictions.data.cpu().numpy().reshape(1, -1)
    return pred_score


def evaluate(model: nn.Module, test_ratings, test_negatives, device, k_list, type_m='group'):
    model.eval()
    pred_score = np.zeros((len(test_ratings), len(test_negatives[0]) + 1))
    for idx in range(len(test_ratings)):
        pred_score[idx, :] = eval_one_rating(model, test_ratings, test_negatives, device, k_list, type_m, idx)

    pred_rank = np.argsort(pred_score * -1, axis=1)
    hits, ndcgs = [], []

    for k in k_list:
        hits.append(get_hit_k(pred_rank, k))
        ndcgs.append(get_ndcg_k(pred_rank, k))

    return hits, ndcgs


def get_hit_k(pred_rank, k):
    pred_rank_k = pred_rank[:, :k]
    hit = np.count_nonzero(pred_rank_k == 0)
    hit = hit / pred_rank_k.shape[0]
    return round(hit, 5)


def get_ndcg_k(pred_rank, k):
    ndcgs = np.zeros(pred_rank.shape[0])
    for user in range(pred_rank.shape[0]):
        for j in range(k):
            if pred_rank[user][j] == 0:
                ndcgs[user] = math.log(2) / math.log(j + 2)
    return np.round(np.mean(ndcgs), decimals=5)
