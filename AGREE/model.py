import torch
import torch.nn as nn


class SelfAttnPooling(nn.Module):
    def __init__(self, emb_dim, drop_ratio=0.):
        super().__init__()
        self.score_layer = nn.Sequential(nn.Linear(emb_dim, 16), nn.ReLU(), nn.Dropout(drop_ratio), nn.Linear(16, 1))

    def forward(self, x, y):
        weight = self.score_layer(torch.cat([x, y], dim=1))
        weight = torch.softmax(weight, dim=0)
        return (x*weight).sum(dim=0, keepdim=True)


class AGREE(nn.Module):
    def __init__(self, num_users, num_items, num_groups, emb_dim, group_member_dict, drop_ratio):
        super(AGREE, self).__init__()

        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        self.group_embedding = nn.Embedding(num_groups, emb_dim)
        self.group_member_dict = group_member_dict

        self.aggregator = SelfAttnPooling(emb_dim*2, drop_ratio)
        self.predictor = nn.Sequential(nn.Linear(emb_dim * 3, 16), nn.ReLU(), nn.Dropout(drop_ratio), nn.Linear(16, 1))

        # init parameters
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def user_forward(self, users, items):
        users_embed = self.user_embedding(users)
        items_embed = self.item_embedding(items)

        element_embed = torch.mul(users_embed, items_embed)
        return torch.sigmoid(self.predictor(torch.cat((element_embed, users_embed, items_embed), dim=1)))

    def group_forward(self, groups, items):
        device = items.device
        groups_emb = torch.Tensor().to(device)

        for i, j in zip(groups, items):
            members = self.group_member_dict[i.item()]
            members_embed = self.user_embedding(torch.LongTensor(members).to(device))
            items_embed = self.item_embedding(torch.LongTensor([j.item()] * len(members)).to(device))

            group_emb = self.aggregator(members_embed, items_embed)
            group_emb += self.group_embedding(torch.LongTensor([i]).to(device))
            groups_emb = torch.cat((groups_emb, group_emb))

        items_embed = self.item_embedding(items)
        element_embed = torch.mul(groups_emb, items_embed)
        return torch.sigmoid(self.predictor(torch.cat((element_embed, groups_emb, items_embed), dim=1)))

    def forward(self, groups, users, items):
        if groups is not None:
            return self.group_forward(groups, items)
        return self.user_forward(users, items)
