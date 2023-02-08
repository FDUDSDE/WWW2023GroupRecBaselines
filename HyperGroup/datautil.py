"""Helper functions for loading dataset"""
import scipy.sparse as sp
import numpy as np
from collections import defaultdict
import torch


def load_rating_file_to_list(filename):
    rating_list = []
    lines = open(filename, 'r').readlines()

    for line in lines:
        contents = line.split()
        rating_list.append([int(contents[0]), int(contents[1])])
    return rating_list


def load_rating_file_to_matrix(filename):
    num_users, num_items = 0, 0

    lines = open(filename, 'r').readlines()
    for line in lines:
        contents = line.split()
        u, i = int(contents[0]), int(contents[1])
        num_users = max(num_users, u)
        num_items = max(num_items, i)

    mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    for line in lines:
        contents = line.split()
        if len(contents) > 2:
            u, i, rating = int(contents[0]), int(contents[1]), int(contents[2])
            if rating > 0:
                mat[u, i] = 1.0
        else:
            u, i = int(contents[0]), int(contents[1])
            mat[u, i] = 1.0
    return mat


def load_negative_file(filename):
    negative_list = []

    lines = open(filename, 'r').readlines()

    for line in lines:
        negatives = line.split()[1:]
        negatives = [int(neg_item) for neg_item in negatives]
        negative_list.append(negatives)
    return negative_list


def load_group_member_to_dict(user_in_group_path):
    group_member_dict = defaultdict(list)
    lines = open(user_in_group_path, 'r').readlines()

    for line in lines:
        contents = line.split()
        group = int(contents[0])
        for member in contents[1].split(','):
            group_member_dict[group].append(int(member))
    return group_member_dict


def build_user_group_hyper_graph(member_dict, n_group, n_user):
    hg = np.zeros((n_group, n_user))

    for group_id, members in member_dict.items():
        hg[group_id, members] = 1.0
    return torch.FloatTensor(hg)


def build_user_group_feat(member_dict, n_group):
    max_len = max([len(members) for members in member_dict.values()])

    membership, member_mask = np.zeros((n_group, max_len)), np.zeros((n_group, max_len))
    for group_id, members in member_dict.items():
        membership[group_id, :len(members)] = members
        member_mask[group_id, len(members):] = -np.inf
    return torch.LongTensor(membership), torch.FloatTensor(member_mask)
