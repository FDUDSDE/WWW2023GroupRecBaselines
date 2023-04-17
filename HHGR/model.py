import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, emb_dim=64):
        super(Discriminator, self).__init__()

        self.lin_layer = nn.Linear(emb_dim, emb_dim, bias=True)
        nn.init.xavier_uniform_(self.lin_layer.weight)
        nn.init.zeros_(self.lin_layer.bias)

        self.bi_lin_layer = nn.Bilinear(emb_dim, emb_dim, 1)
        nn.init.zeros_(self.bi_lin_layer.weight)
        nn.init.zeros_(self.bi_lin_layer.bias)

        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pos, neg):
        """Bilinear discriminator"""
        pos = self.lin_layer(pos)
        neg = self.lin_layer(neg)

        return self.bi_lin_layer(pos, neg)

    def mi_loss(self, scores_group, scores_corrupted, device="cpu"):
        batch_size = scores_group.shape[0]
        pos_size, neg_size = scores_group.shape[1], scores_corrupted.shape[1]

        one_labels = torch.ones(batch_size, pos_size).to(device)
        zero_labels = torch.zeros(batch_size, neg_size).to(device)

        labels = torch.cat([one_labels, zero_labels], dim=1).to(device)
        logits = torch.cat([scores_group, scores_corrupted], dim=1).to(device)

        return self.bce_loss(logits, labels) * (batch_size * (pos_size + neg_size)) / (batch_size * neg_size)


class HGNNConv(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(HGNNConv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_dim, out_dim))

        if bias:
            self.bias = Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        std_v = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std_v, std_v)
        if self.bias is not None:
            self.bias.data.uniform_(-std_v, std_v)

    def forward(self, x: torch.Tensor, g: torch.LongTensor):
        x = x.matmul(self.weight)
        x = x.long()
        if self.bias is not None:
            x = x + self.bias
        x = g.matmul(x)
        return x


class HyperGCN(nn.Module):
    def __init__(self, emb_dim, layers, device, dropout=0.2):
        super(HyperGCN, self).__init__()

        self.dropout = dropout

        self.hgnn = [HGNNConv(emb_dim, emb_dim).to(device) for _ in range(layers)]
        self.device = device

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, g):
        x = F.normalize(x)

        for i in range(len(self.hgnn)):
            x = self.hgnn[i](x, g).to(self.device)
            if i == 0:
                x = F.dropout(x, self.dropout)
        return x


class SelfAttnPooling(nn.Module):
    def __init__(self, emb_dim, drop_ratio=0.2):
        super(SelfAttnPooling, self).__init__()
        self.score_layer = nn.Sequential(nn.Linear(emb_dim, 8), nn.ReLU(), nn.Dropout(drop_ratio), nn.Linear(8, 1))

    def forward(self, x, y):
        weight = self.score_layer(torch.cat((x, y), dim=1))
        weight = F.softmax(weight, dim=0)
        return (x*weight).sum(dim=0, keepdim=True)


class HHGR(nn.Module):
    def __init__(self, n_items, n_users, n_groups, group_member_dict, emb_dim, lambda_mi=0.1, drop_ratio=0.2,
                 device=torch.device("cuda:0")):
        super(HHGR, self).__init__()

        self.n_items, self.n_users, self.n_groups = n_items, n_users, n_groups
        self.group_member_dict = group_member_dict
        self.lambda_mi = lambda_mi
        self.drop = nn.Dropout(drop_ratio)
        self.device = device

        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        self.group_embedding = nn.Embedding(n_groups, emb_dim)

        self.hgcn_fine = HyperGCN(emb_dim, layers=2, device=device, dropout=drop_ratio)
        self.hgcn_coarse = HyperGCN(emb_dim, layers=2, device=device, dropout=drop_ratio)
        self.discriminator = Discriminator(emb_dim)

        self.hgcn_gl = HyperGCN(emb_dim, layers=1, device=device, dropout=drop_ratio)
        self.aggregator = SelfAttnPooling(2*emb_dim, drop_ratio)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def user_forward(self, user_inputs, item_inputs):
        users_embed = self.user_embedding(user_inputs)
        items_embed = self.item_embedding(item_inputs)
        element_embed = users_embed * items_embed
        return torch.sum(element_embed, dim=-1)

    def group_forward(self, group_inputs, item_inputs, all_user_embeds, all_group_embeds):
        groups_embed = torch.Tensor().to(self.device)
        items_embed = self.item_embedding(item_inputs)

        for i, j in zip(group_inputs, item_inputs):
            # 群组i的成员
            members = self.group_member_dict[i.item()]
            # print(f"群组{i}, 成员{members}")
            member_embed_part = self.user_embedding(torch.LongTensor(members).to(self.device)) + all_user_embeds[members, :]
            item_embed_part = self.item_embedding(torch.LongTensor([j]*len(members)).to(self.device))
            group_attn_emb = self.aggregator(member_embed_part, item_embed_part)
            group_pure_emb = self.group_embedding(torch.LongTensor([i]).to(self.device)) + all_group_embeds[[i], :]
            group_emb = group_attn_emb + group_pure_emb

            groups_embed = torch.cat((groups_embed, group_emb))

        element_embed = groups_embed * items_embed
        return torch.sum(element_embed, dim=-1)
