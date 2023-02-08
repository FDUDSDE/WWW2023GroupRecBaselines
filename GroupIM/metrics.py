import math
import numpy as np
import model
import dataloader


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


def user_model_leave_one_test(rec_model: model.GroupIM, dataset: dataloader.GroupDataset, test_ratings, test_negatives, device, k_list=None):
    rec_model.eval()
    user_hits, user_ndcgs = [], []

    pred_score = np.zeros((len(test_ratings), len(test_negatives[0])+1))

    test_users = [rating[0] for rating in test_ratings]
    user_feat = dataset.user_pretrain_dataloader(batch_size=256, test_mode=True).to(device)

    test_user_feat = user_feat[test_users]
    test_logits, _ = rec_model.user_preference_encoder.pretrain_forward(test_user_feat)
    test_logits = test_logits.detach().cpu().numpy()

    for idx in range(len(test_negatives)):
        test_item = [test_ratings[idx][1]] + test_negatives[idx]
        pred_score[idx, :] = test_logits[idx, test_item]

    pred_rank = np.argsort(pred_score * -1, axis=1)
    for k in k_list:
        user_hits.append(get_hit_k(pred_rank, k))
        user_ndcgs.append(get_ndcg_k(pred_rank, k))
    return user_hits, user_ndcgs


def group_model_leave_one_test(rec_model: model.GroupIM, dataset: dataloader.GroupDataset, test_ratings, test_negatives, device, k_list=None):
    rec_model.eval()
    group_hits, group_ndcgs = [], []

    pred_score = np.zeros((len(test_ratings), len(test_negatives[0])+1))

    test_groups = [rating[0] for rating in test_ratings]

    _, all_group_mask, all_user_items, _ = dataset.group_dataloader(batch_size=256, test_mode=True)
    all_group_mask = all_group_mask.to(device)
    all_user_items = all_user_items.to(device)

    test_group_logits, _, _ = rec_model(all_group_mask[test_groups], all_user_items[test_groups])
    test_group_logits = test_group_logits.detach().cpu().numpy()

    for idx in range(len(test_negatives)):
        test_item = [test_ratings[idx][1]] + test_negatives[idx]
        pred_score[idx, :] = test_group_logits[idx, test_item]

    pred_rank = np.argsort(pred_score * -1, axis=1)
    for k in k_list:
        group_hits.append(get_hit_k(pred_rank, k))
        group_ndcgs.append(get_ndcg_k(pred_rank, k))
    return group_hits, group_ndcgs
