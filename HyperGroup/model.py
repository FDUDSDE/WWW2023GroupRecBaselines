import torch.nn.functional as F
import torch.nn as nn
import torch


class HyperGroup(nn.Module):
    def __init__(self, n_users, n_items, n_groups, hyper_graph, membership, member_mask, member_dict, emb_dim=64, k=2):
        super(HyperGroup, self).__init__()
        self.n_users, self.n_items, self.n_groups = n_users, n_items, n_groups
        self.emb_dim = emb_dim
        self.hyper_graph = hyper_graph
        # 成员身份 (n_group, max_num_member), 成员身份掩码 (n_group, max_num_member), 群组到成员的映射词典
        self.membership, self.member_mask, self.member_dict = membership, member_mask, member_dict

        self.k = k
        self.weight = nn.ModuleList([nn.Linear(2*emb_dim, emb_dim) for _ in range(k)])

        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        self.predictor = nn.Sequential(nn.Linear(emb_dim, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

    def user_forward(self, users, items):
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        return self.predictor(user_emb*item_emb)

    def group_forward(self, groups, items):
        device = groups.device
        # Step 1, MeanPooling compute group embeddings
        group_mask = torch.exp(self.member_mask)
        group_users = self.user_embedding(self.membership)
        # all_groups_emb (n_groups, emb_dim)
        all_groups_emb = torch.sum(group_users*group_mask.unsqueeze(2), dim=1) / group_mask.sum(1).unsqueeze(1)

        # Step 2, hypergraph convolution - forward propagation
        group_adj = torch.mm(self.hyper_graph, self.hyper_graph.t())
        group_adj = group_adj - torch.diag_embed(torch.diag(group_adj))
        group_adj = group_adj.to(device)
        # group_adj_deg = torch.diag(1.0 / torch.sum(group_adj, dim=1).squeeze())  # (n_groups, n_groups)
        neigh_member_messages = torch.FloatTensor().to(device)
        #  Step 2.1 aggregate common members' features
        for idx in range(group_adj.shape[0]):
            # indices = list(group_adj[idx, :].nonzero().t().cpu().numpy())
            indices = group_adj[idx, :].nonzero()
            neigh_member_message = torch.zeros((1, self.emb_dim)).to(device)
            # if indices[0] is not None:
            #     indices = indices[0]
            for indice in indices:
                common_members = list(set(self.member_dict[idx]) & set(self.member_dict[indice.item()]))
                # print(self.user_embedding(torch.LongTensor(common_members)).shape,
                #      torch.mean(self.user_embedding(torch.LongTensor(common_members)), dim=0).unsqueeze(0).shape)
                neigh_member_message += group_adj[idx, indice] * torch.mean(self.user_embedding(torch.LongTensor(common_members).to(device)), dim=0).unsqueeze(0)
            neigh_member_messages = torch.cat((neigh_member_messages, neigh_member_message))

        # print(neigh_member_messages)
        for i in range(self.k):
            #  Step 2.2 aggregate neighboring groups' features
            neigh_group_messages = torch.mm(group_adj, all_groups_emb)  # (n_groups, emb_dim)
            messages = neigh_group_messages + neigh_member_messages
            #  Step 2.3 linear combination
            # print(all_groups_emb.device, messages.device)
            all_groups_emb = self.weight[i](torch.cat((all_groups_emb, messages), dim=1).to(device))
            all_groups_emb = F.normalize(all_groups_emb, dim=-1).to(device)

        groups_emb = all_groups_emb[groups]
        items_emb = self.item_embedding(items)
        return self.predictor(groups_emb*items_emb)

    def forward(self, group_inputs, user_inputs, item_inputs):
        if (group_inputs is not None) and (user_inputs is None):
            return self.group_forward(group_inputs, item_inputs)
        return self.user_forward(user_inputs, item_inputs)
