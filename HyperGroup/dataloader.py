from datautil import load_rating_file_to_list, load_rating_file_to_matrix, load_negative_file, load_group_member_to_dict, \
    build_user_group_hyper_graph, build_user_group_feat
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class GroupDataset(object):
    def __init__(self, num_negatives, dataset="Mafengwo",):
        print(f"[{dataset.upper()}] loading...")

        self.num_negatives = num_negatives
        user_path, group_path = f"../data/{dataset}/userRating", f"../data/{dataset}/groupRating"

        # user
        self.user_train_matrix = load_rating_file_to_matrix(user_path + "Train.txt")
        self.user_test_ratings = load_rating_file_to_list(user_path + "Test.txt")
        self.user_test_negatives = load_negative_file(user_path + "Negative.txt")
        self.num_users, self.num_items = self.user_train_matrix.shape

        print(f"UserItem: {self.user_train_matrix.shape} with {len(self.user_train_matrix.keys())} interactions, "
              f"sparsity: {(1 - len(self.user_train_matrix.keys()) / self.num_users / self.num_items):.5f}")

        # group
        self.group_train_matrix = load_rating_file_to_matrix(group_path + "Train.txt")
        self.group_test_ratings = load_rating_file_to_list(group_path + "Test.txt")
        self.group_test_negatives = load_negative_file(group_path + "Negative.txt")
        self.num_groups, self.num_group_net_items = self.group_train_matrix.shape
        self.group_member_dict = load_group_member_to_dict(f"../data/{dataset}/groupMember.txt")

        print(f"GroupItem: {self.group_train_matrix.shape} with {len(self.group_train_matrix.keys())} interactions, spa"
              f"rsity: {(1 - len(self.group_train_matrix.keys()) / self.num_groups / self.group_train_matrix.shape[1]):.5f}")

        # TODO: Hyper-graph
        self.hyper_graph = build_user_group_hyper_graph(self.group_member_dict, self.num_groups, self.num_users)
        self.membership, self.member_mask = build_user_group_feat(self.group_member_dict, self.num_groups)
        # print(self.membership, self.member_mask)
        # print(torch.mm(self.hyper_graph, self.hyper_graph.t()))
        print(f"{dataset.upper()} finish loading!")

    def get_train_instances(self, train):
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


if __name__ == "__main__":
    d = GroupDataset(num_negatives=3, dataset="Mafengwo")
    # print(d.hyper_graph)
    # print(d.membership, d.member_mask)
