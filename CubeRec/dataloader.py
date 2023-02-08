import torch
from datautil import load_rating_file_to_list, load_rating_file_to_matrix, load_negative_file, load_group_member_to_dict, \
    load_rating_file_to_csr_matrix, convert_sp_mat_to_sp_tensor
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import scipy.sparse as sp


class GroupDataset(object):
    def __init__(self, num_negatives=6, dataset="Mafengwo"):
        print(f"[{dataset.upper()}] loading...")

        user_path = f"../data/{dataset}/userRating"
        group_path = f"../data/{dataset}/groupRating"
        self.num_negatives = num_negatives

        # User data
        self.user_train_matrix = load_rating_file_to_matrix(user_path+"Train.txt")
        self.user_test_ratings = load_rating_file_to_list(user_path+"Test.txt")
        self.user_test_negatives = load_negative_file(user_path+"Negative.txt")
        self.user_item_net = load_rating_file_to_csr_matrix(user_path+"Train.txt")
        self.num_users, self.num_items = self.user_train_matrix.shape

        print(f"UserItem {self.user_train_matrix.shape} with {len(self.user_train_matrix.keys())} "
              f"interactions, sparsity: {(1-(len(self.user_train_matrix.keys()) / self.num_users / self.num_items)):.5f}")

        # Group data
        self.group_train_matrix = load_rating_file_to_matrix(group_path+"Train.txt")
        self.group_test_ratings = load_rating_file_to_list(group_path+"Test.txt")
        self.group_test_negatives = load_negative_file(group_path+"Negative.txt")
        self.group_member_dict = load_group_member_to_dict(f"../data/{dataset}/groupMember.txt")
        self.num_groups, self.num_group_items = self.group_train_matrix.shape

        print(f"GroupItem {self.group_train_matrix.shape} with {len(self.group_train_matrix.keys())} interactions, "
              f"sparsity: {(1-(len(self.group_train_matrix.keys()) / self.num_groups / self.num_group_items)):.5f}")

        print(f"[{dataset.upper()}] finish loading! \n")

    def get_sparse_graph(self):
        """Build adj matrix based on user-item interactions (for LightGCN computation)"""
        adj_mat = sp.dok_matrix((self.num_users+self.num_items, self.num_users+self.num_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        R = self.user_item_net.tolil()
        adj_mat[:self.num_users, self.num_users:] = R
        adj_mat[self.num_users:, :self.num_users] = R.T
        adj_mat = adj_mat.todok()

        row_sum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(row_sum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocsr()
        graph = convert_sp_mat_to_sp_tensor(norm_adj)
        return graph.coalesce()

    def get_train_instances(self, train):
        """Generate training instances (for each positive pair, generate num_negatives negative samples)"""
        users, pos_items, neg_items = [], [], []

        n_users, n_items = train.shape

        for (u, i) in train.keys():
            users.extend([u]*self.num_negatives)
            pos_items.extend([i]*self.num_negatives)
            neg_items.extend(np.random.choice(n_items, self.num_negatives))

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
