import numpy as np
import math
import model
import torch


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
                ndcgs[user] = math.log(2) / math.log(j+2)
    return np.round(np.mean(ndcgs), decimals=5)


def model_leave_one_test(rec_model: model.CubeRec, test_ratings, test_negatives, device, k_list, mode='user'):
    rec_model.eval()

    test_hits, test_ndcgs = [], []
    pred_score = np.zeros((len(test_ratings), len(test_negatives[0]) + 1))

    if mode == 'user':
        users, items = rec_model.compute()
    else:
        users, items, all_centers, all_offsets = rec_model.compute_all()

    for idx in range(len(test_ratings)):
        rating = test_ratings[idx]

        test_user, test_items = rating[0], [rating[1]] + test_negatives[idx]

        if mode == 'user':
            test_user_emb = users[test_user].detach().cpu().numpy()
            test_items_emb = items[test_items].detach().cpu().numpy()
            # For query user, compute the distance with all candidate items
            score = np.sqrt(np.sum(np.asarray(test_user_emb - test_items_emb) ** 2, axis=1))
        else:
            test_centers = all_centers[[test_user]].detach().cpu().numpy()
            test_offsets = all_offsets[[test_user]].detach().cpu().numpy()
            test_centers = np.repeat(test_centers, len(test_items), axis=0)
            test_offsets = np.repeat(test_offsets, len(test_items), axis=0)
            # print(centers.shape, offsets.shape)
            score = rec_model.gi_scores(torch.FloatTensor(test_centers).to(device), torch.FloatTensor(test_offsets).to(device),
                                        torch.LongTensor(test_items).to(device), items)
            score = score.detach().cpu().numpy().reshape(1, -1)

        pred_score[idx, :] = score

    pred_rank = np.argsort(pred_score, axis=1)

    for k in k_list:
        test_hits.append(get_hit_k(pred_rank, k))
        test_ndcgs.append(get_ndcg_k(pred_rank, k))

    return test_hits, test_ndcgs
