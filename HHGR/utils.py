"""Helper functions for loading dataset"""
import scipy.sparse as sp
import numpy as np
from collections import defaultdict
from scipy.sparse import coo_matrix


def load_rating_file_to_list(filename):
    """基于[user, item]交互文件构建List形式的交互记录"""
    rating_list = []
    lines = open(filename, 'r').readlines()

    for line in lines:
        contents = line.split()
        rating_list.append([int(contents[0]), int(contents[1])])
    return rating_list


def load_rating_file_to_matrix(filename):
    """基于交互记录，构建M*N的交互矩阵"""
    num_users, num_items = 0, 0
    group2item = defaultdict(list)

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
                group2item[u].append(i)
        else:
            u, i = int(contents[0]), int(contents[1])
            mat[u, i] = 1.0
            group2item[u].append(i)
    return mat, group2item


def load_negative_file(filename):
    """加载负样本文件"""
    negative_list = []

    lines = open(filename, 'r').readlines()

    for line in lines:
        negatives = line.split()[1:]
        negatives = [int(neg_item) for neg_item in negatives]
        negative_list.append(negatives)
    return negative_list


def build_group_member_hyper_graph(group_member_dict, num_user, num_group):
    """构建成员-群组超图"""
    row_users, col_groups = [], []

    for group_id, members in group_member_dict.items():
        col_groups.extend([group_id] * len(members))
        row_users.extend(members)

    user_hg = sp.csr_matrix((np.ones_like(row_users), (row_users, col_groups)), dtype='float32',
                            shape=(num_user, num_group))
    return user_hg


def load_group_member_to_dict(user_in_group_path):
    """读取Group-Member文件，并返回List词典格式(GroupId到成员List的映射)"""
    group_member_dict = defaultdict(list)
    lines = open(user_in_group_path, 'r').readlines()

    for line in lines:
        contents = line.split()
        group = int(contents[0])
        for member in contents[1].split(','):
            group_member_dict[group].append(int(member))
    return group_member_dict


def build_group_graph(group_member_dict, num_groups):
    """构建群组之间的超图"""
    row, col, entries = [], [], []

    for i in range(num_groups):
        g1 = set(group_member_dict[i])
        for j in range(num_groups):
            g2 = set(group_member_dict[j])

            if len(g1 & g2) > 0 and len(g1 ^ g2) > 0:
                row += [i]
                col += [j]
                entries += [1.0]

    data_gg = coo_matrix((entries, (row, col)), shape=(num_groups, num_groups), dtype=np.float32)
    data_gg = data_gg.tocsr()
    HH_gg = data_gg.dot(data_gg.transpose())
    H_gg = HH_gg.multiply(data_gg)

    return H_gg


def generate_G_from_H(H):
    H = np.array(H)
    dv = np.sum(H, axis=1) + 1e-5
    de = np.sum(H, axis=0) + 1e-5

    inv_de = np.mat(np.diag(np.power(de, -1)))
    dv2 = np.mat(np.diag(np.power(dv, -1)))

    H = np.mat(H)
    HT = H.T

    G = dv2 * H * inv_de * HT * dv2
    # 返回的是一个(n,n)的表示节点间关系的矩阵
    return G
