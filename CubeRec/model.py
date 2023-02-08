import torch.nn.functional as F
import torch
import torch.nn as nn
import dataloader
import math


class CubeRec(nn.Module):
    def __init__(self, args, dataset: dataloader.GroupDataset, device):
        super(CubeRec, self).__init__()
        self.args = args
        self.dataset = dataset
        self.device = device
        self.__init_setting()

    def __init_setting(self):
        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items

        self.n_layers = self.args.n_layers
        self.emb_dim = self.args.emb_dim
        self.keep_prob = self.args.keep_prob

        self.embedding_user = nn.Embedding(self.num_users, self.emb_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.emb_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.f = nn.Sigmoid()

        # LightGCN propagation graph
        self.graph = self.dataset.get_sparse_graph().to(self.device)

        # Group aggregation strategy (geometric / attentive)
        if self.args.group_agg == "geometric":
            self.wc = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
            self.wo = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        else:
            self.wc = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
            self.wo = nn.Linear(self.emb_dim, self.emb_dim, bias=True)
            self.act_relu = nn.ReLU(inplace=True)
            self.query = nn.Linear(self.emb_dim, 1, bias=False)
            self.key = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
            self.value = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

    def __dropout_x(self, x):
        """Sparse matrix dropout"""
        size, index, values = x.size(), x.indices().t(), x.values()

        random_index = torch.rand(len(values)) + self.keep_prob
        random_index = random_index.int().bool()

        index, values = index[random_index], values[random_index] / self.keep_prob
        return torch.sparse.FloatTensor(index.t(), values, size)

    def __dropout(self):
        return self.__dropout_x(self.graph)

    def compute(self):
        """LightGCN forward propagation"""
        users_emb, items_emb = self.embedding_user.weight, self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        emb = [all_emb]

        g_dropped = self.__dropout() if self.training else self.graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_dropped, all_emb)
            emb.append(all_emb)

        emb = torch.mean(torch.stack(emb, dim=1), dim=1)

        users, items = torch.split(emb, [self.num_users, self.num_items])
        return users, items

    def get_embedding(self, users, pos_items, neg_items):
        all_users, all_items = self.compute()

        users_emb, pos_emb, neg_emb = all_users[users], all_items[pos_items], all_items[neg_items]
        users_emb_ego, pos_emb_ego, neg_emb_ego = self.embedding_user(users), self.embedding_item(pos_items), \
                                                  self.embedding_item(neg_items)

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        """Compute user-item loss based on distances"""
        users_emb, pos_emb, neg_emb, user_emb_0, pos_emb_0, neg_emb_0 = self.get_embedding(users, pos, neg)

        reg_loss = (1 / 2) * (user_emb_0.norm(2).pow(2) + pos_emb_0.norm(2).pow(2) + neg_emb_0.norm(2).pow(2)) / float(
            len(users))

        pos_dis = torch.sqrt(torch.sum((users_emb - pos_emb) ** 2, dim=-1))
        neg_dis = torch.sqrt(torch.sum((users_emb - neg_emb) ** 2, dim=-1))
        distance_loss = torch.mean(torch.max(pos_dis - neg_dis + 0.5, torch.zeros(pos_dis.shape).to(pos.device)))

        return distance_loss, reg_loss

    def geometric_group(self, embedding_member):
        """Geometric bounding and projection for group representation"""
        u_max = torch.max(embedding_member, dim=0).values
        u_min = torch.min(embedding_member, dim=0).values

        center = self.wc((u_max + u_min) / 2)
        offset = self.wo((u_max - u_min) / 2)
        return center, offset

    def attentive_group(self, embedding_member):
        """Attentive fusion and projection for group representation"""
        key_user = self.key(embedding_member)
        key_user_query = F.softmax(self.query(key_user) / math.sqrt(self.emb_dim), dim=-1)
        value_user = self.value(embedding_member)

        attn = torch.squeeze(torch.matmul(value_user.T, key_user_query))
        center = self.wc(attn)
        offset = self.act_relu(self.wo(attn))
        return center, offset

    def group_representations(self, members, all_users, device):
        centers = torch.empty(0).to(device)
        offsets = torch.empty(0).to(device)

        for member in members:
            embedding_member = torch.index_select(all_users, 0, member)
            if self.args.group_agg == "geometric":
                center, offset = self.geometric_group(embedding_member)
            else:
                center, offset = self.attentive_group(embedding_member)

            centers = torch.cat((centers, torch.unsqueeze(center, dim=0)), dim=0)
            offsets = torch.cat((offsets, torch.unsqueeze(offset, dim=0)), dim=0)

        return centers, offsets

    def gi_scores(self, centers, offsets, items, all_items):
        """Compute groups and items scores"""
        lower_left_s = centers - offsets
        upper_right_s = centers + offsets
        embedding_items = all_items[items]

        dis_out = torch.max(embedding_items - upper_right_s, torch.zeros(embedding_items.shape).to(self.device)) + \
                  torch.max(lower_left_s - embedding_items, torch.zeros(embedding_items.shape).to(self.device))
        dis_out = torch.sqrt(torch.sum(dis_out ** 2, dim=-1)).unsqueeze(-1)
        # print(dis_out.shape)

        dis_in = centers - torch.min(upper_right_s, torch.max(lower_left_s, embedding_items))
        dis_in = torch.sqrt(torch.sum(dis_in ** 2, dim=-1)).unsqueeze(-1)
        # gamma is set as 0.3
        return dis_out + 0.3 * dis_in

    def compute_all(self):
        """Forward propagation: Compute all users, items, and groups representations"""
        all_users, all_items = self.compute()
        num_groups = self.dataset.num_groups

        group_member_dict = self.dataset.group_member_dict

        members = [torch.LongTensor(group_member_dict[group_id]).to(self.device) for group_id in range(num_groups)]

        all_centers, all_offsets = self.group_representations(members, all_users, self.device)
        return all_users, all_items, all_centers, all_offsets
