import random
from utils import load_rating_file_to_matrix, load_rating_file_to_list, load_negative_file, \
    load_group_member_to_dict, build_group_member_hyper_graph, build_group_graph
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import scipy.sparse as sp


class GroupDataset(object):
    def __init__(self, num_negatives, dataset="Mafengwo"):
        print(f"[{dataset.upper()} loading...]")

        self.num_negatives = num_negatives

        user_path, group_path = f"../data/{dataset}/userRating", f"../data/{dataset}/groupRating"

        # 用户(user)数据
        self.user_train_matrix, self.user_item_dict = load_rating_file_to_matrix(user_path + "Train.txt")
        self.user_test_ratings = load_rating_file_to_list(user_path + "Test.txt")
        self.user_test_negatives = load_negative_file(user_path + "Negative.txt")
        self.num_users, self.num_items = self.user_train_matrix.shape

        print(f"UserItem: {self.user_train_matrix.shape} with {len(self.user_train_matrix.keys())} "
              f"interactions, sparsity: {(1 - len(self.user_train_matrix.keys()) / self.num_users / self.num_items):.5f}")

        # 群组(group)数据
        self.group_train_matrix, self.group_item_dict = load_rating_file_to_matrix(group_path + "Train.txt")
        self.group_test_ratings = load_rating_file_to_list(group_path + "Test.txt")
        self.group_test_negatives = load_negative_file(group_path + "Negative.txt")
        self.num_groups, self.num_group_net_items = self.group_train_matrix.shape
        self.group_member_dict = load_group_member_to_dict(f"../data/{dataset}/groupMember.txt")

        print(f"GroupItem: {self.group_train_matrix.shape} with {len(self.group_train_matrix.keys())} interactions, spa"
              f"rsity: {(1 - len(self.group_train_matrix.keys()) / self.num_groups / self.group_train_matrix.shape[1]):.5f}")

        # 成员级别的经过不同粒度mask的超图
        self.data_gu_fine, self.data_gu_coarse = self.get_corrupt_user_hyper_graph()
        # 群组级别的超图
        self.H_gg = build_group_graph(self.group_member_dict, self.num_groups)
        print(f"{dataset.upper()} finish loading!\n")

    def get_train_instances(self, train):
        """生成训练数据实例，包括User列表，成对的正负商品列表"""
        users, pos_items, neg_items = [], [], []

        num_users, num_items = train.shape[0], train.shape[1]

        for (u, i) in train.keys():
            for _ in range(self.num_negatives):
                users.append(u)

                # Pos instances
                pos_items.append(i)

                # Neg instances
                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                neg_items.append(j)
        pos_neg_items = [[pos_item, neg_item] for pos_item, neg_item in zip(pos_items, neg_items)]
        return users, pos_neg_items

    def get_user_dataloader(self, batch_size):
        users, pos_neg_items = self.get_train_instances(self.user_train_matrix)
        train_data = TensorDataset(torch.LongTensor(users), torch.LongTensor(pos_neg_items))
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)

    def get_group_dataloader(self, batch_size):
        groups, pos_neg_items = self.get_train_instances(self.group_train_matrix)
        train_data = TensorDataset(torch.LongTensor(groups), torch.LongTensor(pos_neg_items))
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)

    def get_user_ssl_dataloader(self, batch_size):
        """准备Self-supervised的user数据"""
        users = [u for u in range(self.num_users)]
        neg_users = [random.randint(0, self.num_users - 1) for _ in range(self.num_users)]
        train_data = TensorDataset(torch.LongTensor(users), torch.LongTensor(neg_users))
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)

    def get_corrupt_user_hyper_graph(self):
        """生成两种粒度的user-group-hypergraph"""
        theta = np.zeros(self.num_users, dtype=int)
        rand = np.random.randint(0, len(theta), int(0.2 * len(theta)))
        if len(set(rand)) != len(rand):
            rand = np.random.randint(0, len(theta), int(0.2 * len(theta)))

        theta[rand] = 1

        # 群组-成员超图
        data_gu_csr = build_group_member_hyper_graph(self.group_member_dict, num_user=self.num_users,
                                                     num_group=self.num_groups)

        # 粗粒度
        data_gu_csr_coarse = data_gu_csr.toarray()
        data_gu_csr_coarse = data_gu_csr_coarse.transpose() * (np.array(theta.transpose()))
        data_gu_csr_coarse = data_gu_csr_coarse.transpose()
        data_gu_csr_coarse = sp.csr_matrix(data_gu_csr_coarse)

        # 细粒度
        beta = np.zeros(self.num_groups, dtype=int)
        data_gu_csr_fine = data_gu_csr.toarray()
        for i in range(len(data_gu_csr_fine)):
            rand = np.random.randint(0, len(beta), int(0.3 * len(beta)))
            beta[rand] = 1
            data_gu_csr_fine[i] = data_gu_csr_fine[i] * (np.array(beta.transpose()))
        data_gu_csr_fine = sp.csr_matrix(data_gu_csr_fine)

        return data_gu_csr_fine, data_gu_csr_coarse


if __name__ == "__main__":
    g = GroupDataset(num_negatives=2)
    print(g.H_gg)
    # print(g.data_gu_fine)
