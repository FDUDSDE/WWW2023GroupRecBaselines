import torch
from datautil import load_rating_file_to_list, load_rating_file_to_matrix, load_negative_file, load_group_member_to_dict
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict


class GroupDataset(object):
    def __init__(self, dataset="Mafengwo"):
        print(f"[{dataset.upper()}] loading...")

        user_path = f"../data/{dataset}/userRating"
        group_path = f"../data/{dataset}/groupRating"
        self.user_train_matrix = load_rating_file_to_matrix(user_path+"Train.txt")
        self.user_test_ratings = load_rating_file_to_list(user_path+"Test.txt")
        self.user_test_negatives = load_negative_file(user_path+"Negative.txt")
        self.num_users, self.num_items = self.user_train_matrix.shape

        print(f"UserItem {self.user_train_matrix.shape} with {len(self.user_train_matrix.keys())} interactions, "
              f"sparsity: {(1-(len(self.user_train_matrix.keys()) / self.num_users / self.num_items)):.5f}")

        self.group_train_matrix = load_rating_file_to_matrix(group_path+"Train.txt")
        self.group_test_ratings = load_rating_file_to_list(group_path+"Test.txt")
        self.group_test_negatives = load_negative_file(group_path+"Negative.txt")
        self.group_member_dict = load_group_member_to_dict(f"../data/{dataset}/groupMember.txt")
        self.num_groups, self.num_group_items = self.group_train_matrix.shape

        print(f"GroupItem {self.group_train_matrix.shape} with {len(self.group_train_matrix.keys())} interactions, "
              f"sparsity: {(1-(len(self.group_train_matrix.keys()) / self.num_groups / self.num_group_items)):.5f}")

        print(f"[{dataset.upper()}] finish loading! \n")

    def user_pretrain_dataloader(self, batch_size, test_mode=False):
        user2item = defaultdict(list)
        for (u, i) in self.user_train_matrix.keys():
            user2item[u].append(i)

        user_feat = np.zeros((self.num_users, self.num_items))
        for user_id in range(self.num_users):
            items = user2item[user_id]
            user_feat[user_id, items] = 1.0
        if test_mode:
            return torch.FloatTensor(user_feat)
        train_data = TensorDataset(torch.FloatTensor(user_feat))
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)

    def group_dataloader(self, batch_size, test_mode=False, num_corrupt=5):
        group2items = defaultdict(list)
        for (g, i) in self.group_train_matrix.keys():
            group2items[g].append(i)

        max_len = max([len(members) for members in self.group_member_dict.values()])
        users_feat = self.user_pretrain_dataloader(batch_size, test_mode=True)  # User features

        all_group_users, all_group_mask, all_user_items, all_corrupt_user_items = [], [], [], []
        group_feat = np.zeros((self.num_groups, self.num_items))
        for group_id in range(self.num_groups):
            members = self.group_member_dict[group_id]

            # Mask for labeling valid group members
            group_mask = torch.zeros((max_len,))
            # group_mask[:len(members)] = 1
            group_mask[len(members):] = -np.inf
            if len(members) < max_len:
                # print(np.array(members).shape, np.zeros((1, max_len-len(members))).shape)
                group_user = torch.hstack((torch.LongTensor(members), torch.zeros((max_len-len(members),)))).long()
            else:
                group_user = torch.LongTensor(members)

            corrupted_user = []
            for j in range(num_corrupt):
                random_u = np.random.randint(self.num_users)
                while random_u in members:
                    random_u = np.random.randint(self.num_users)
                corrupted_user.append(random_u)
            corrupted_user = np.array(corrupted_user)

            group_feat[group_id, group2items[group_id]] = 1.0

            all_group_users.append(group_user)
            all_group_mask.append(group_mask)
            all_user_items.append(users_feat[group_user])
            all_corrupt_user_items.append(users_feat[corrupted_user])

        if test_mode:
            return torch.stack(all_group_users), torch.stack(all_group_mask), torch.stack(all_user_items), \
                   torch.FloatTensor(group_feat)

        # print(torch.stack(all_group_users).shape,  torch.stack(all_group_mask).shape, torch.stack(all_user_items).shape,
        #       torch.stack(all_corrupt_user_items).shape)
        train_data = TensorDataset(torch.stack(all_group_users), torch.stack(all_group_mask),
                                   torch.stack(all_user_items), torch.stack(all_corrupt_user_items),
                                   torch.FloatTensor(group_feat))
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)
