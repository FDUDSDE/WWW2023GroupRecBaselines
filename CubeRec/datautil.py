import numpy as np
from collections import defaultdict
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch


def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


def load_rating_file_to_csr_matrix(filename):
    """Build user-item interaction sparse matrix"""
    train_u, train_i = [], []

    lines = open(filename, 'r').readlines()

    for line in lines:
        content = line.split()
        if len(content) > 2:
            u, i, rating = int(content[0]), int(content[1]), int(content[2])
            if rating > 0:
                train_u.append(u)
                train_i.append(i)
        else:
            u, i = int(content[0]), int(content[1])
            train_u.append(u)
            train_i.append(i)

    num_user, num_item = max(train_u), max(train_i)
    ui_net = csr_matrix((np.ones(len(train_u)), (np.array(train_u), np.array(train_i))), shape=(num_user+1, num_item+1))

    return ui_net


def load_rating_file_to_list(filename):
    ratings = []

    lines = open(filename, 'r').readlines()

    for line in lines:
        content = line.split()
        ratings.append([int(content[0]), int(content[1])])
    return ratings


def load_rating_file_to_matrix(filename):
    num_users, num_items = 0, 0

    lines = open(filename, 'r').readlines()

    for line in lines:
        content = line.split()
        u, i = int(content[0]), int(content[1])
        num_users = max(num_users, u)
        num_items = max(num_items, i)

    mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)

    for line in lines:
        content = line.split()
        if len(content) > 2:
            u, i, rating = int(content[0]), int(content[1]), int(content[2])
            if rating > 0:
                mat[u, i] = 1.0
        else:
            u, i = int(content[0]), int(content[1])
            mat[u, i] = 1.0
    return mat


def load_negative_file(filename):
    negatives = []

    lines = open(filename, 'r').readlines()

    for line in lines:
        neg = line.split()[1:]
        neg = [int(item) for item in neg]
        negatives.append(neg)
    return negatives


def load_group_member_to_dict(user_in_group_path):
    group_member_dict = defaultdict(list)

    lines = open(user_in_group_path, 'r').readlines()

    for line in lines:
        content = line.split()
        group = int(content[0])
        for member in content[1].split(','):
            group_member_dict[group].append(int(member))

    return group_member_dict
