import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """User preference encoder"""
    def __init__(self, n_items, emb_dim, drop_ratio):
        super(Encoder, self).__init__()
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.drop = nn.Dropout(drop_ratio)

        # Input (B, n_items)的0/1值的特征, map into (B, emb_dim)
        self.user_preference_encoder = nn.Linear(self.n_items, self.emb_dim, bias=True)
        nn.init.xavier_uniform_(self.user_preference_encoder.weight)
        nn.init.zeros_(self.user_preference_encoder.bias)

        self.transform_layer = nn.Linear(self.emb_dim, self.emb_dim)
        nn.init.xavier_uniform_(self.transform_layer.weight)
        nn.init.zeros_(self.transform_layer.bias)

        self.user_predictor = nn.Linear(self.emb_dim, self.n_items, bias=False)
        nn.init.xavier_uniform_(self.user_predictor.weight)

    def pretrain_forward(self, user_items):
        user_items_norm = F.normalize(user_items)
        user_pref_emb = self.drop(user_items_norm)
        user_pref_emb = torch.tanh(self.user_preference_encoder(user_pref_emb))

        logits = self.user_predictor(user_pref_emb)
        return logits, user_pref_emb

    def forward(self, user_items):
        _, user_embeds = self.pretrain_forward(user_items)
        user_embeds = torch.tanh(self.transform_layer(user_embeds))
        return user_embeds


class AttentionAggregator(nn.Module):
    """User preferences aggregator (attention-based)"""
    def __init__(self, output_dim, drop_ratio=0.):
        super(AttentionAggregator, self).__init__()

        self.attention = nn.Linear(output_dim, 1)
        self.drop = nn.Dropout(drop_ratio)

    def forward(self, x, mask):
        attention_out = torch.tanh(self.attention(x))
        if mask is None:
            weight = torch.softmax(attention_out, dim=1)
        else:
            weight = torch.softmax(attention_out + mask.unsqueeze(2), dim=1)
        ret = torch.matmul(x.transpose(2, 1), weight).squeeze(2)
        return ret


class Discriminator(nn.Module):
    """Discriminator - bilinear layer outputs the user-group score"""
    def __init__(self, emb_dim=64):
        super(Discriminator, self).__init__()
        self.emb_dim = emb_dim

        self.fc_layer = nn.Linear(self.emb_dim, self.emb_dim, bias=True)
        nn.init.xavier_uniform_(self.fc_layer.weight)
        nn.init.zeros_(self.fc_layer.bias)

        self.bilinear_layer = nn.Bilinear(self.emb_dim, self.emb_dim, 1)
        nn.init.xavier_uniform_(self.bilinear_layer.weight)
        nn.init.zeros_(self.bilinear_layer.bias)

        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, group_inputs, user_inputs):
        group_emb = torch.tanh(self.fc_layer(group_inputs))

        user_emb = torch.tanh(self.fc_layer(user_inputs))

        return self.bilinear_layer(user_emb, group_emb.unsqueeze(1).repeat(1, user_inputs.shape[1], 1))

    def mi_loss(self, scores_group, group_mask, scores_corrupted, device="cpu"):
        batch_size = scores_group.shape[0]

        pos_size, neg_size = scores_group.shape[1], scores_corrupted.shape[1]

        one_labels = torch.ones(batch_size, pos_size).to(device)
        zero_labels = torch.zeros(batch_size, neg_size).to(device)

        labels = torch.cat((one_labels, zero_labels), 1)
        logits = torch.cat((scores_group, scores_corrupted), 1).squeeze(2)

        mask = torch.cat((torch.exp(group_mask), torch.ones([batch_size, neg_size]).to(device)), 1)

        mi_loss = self.bce_loss(logits * mask, labels * mask) * (batch_size * (pos_size + neg_size)) \
                  / (torch.exp(group_mask).sum() + batch_size * neg_size)

        return mi_loss


class GroupIM(nn.Module):
    def __init__(self, n_items, emb_dim, lambda_mi=0.1, drop_ratio=0.4):
        super(GroupIM, self).__init__()
        self.n_items = n_items
        self.lambda_mi = lambda_mi
        self.drop = nn.Dropout(drop_ratio)

        self.emb_dim = emb_dim

        self.user_preference_encoder = Encoder(self.n_items, self.emb_dim, drop_ratio)
        self.preference_aggregator = AttentionAggregator(self.emb_dim)
        self.group_predictor = nn.Linear(self.emb_dim, self.n_items, bias=False)
        nn.init.xavier_uniform_(self.group_predictor.weight)

        self.discriminator = Discriminator(emb_dim=self.emb_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, group_mask, user_items):
        user_pref_embeds = self.user_preference_encoder(user_items)
        group_embed = self.preference_aggregator(user_pref_embeds, group_mask)
        group_logit = self.group_predictor(group_embed)

        if self.train:
            obs_user_embeds = self.user_preference_encoder(user_items)
            scores_ug = self.discriminator(group_embed, obs_user_embeds).detach()
            return group_logit, group_embed, scores_ug
        else:
            return group_logit, group_embed

    def multinomial_loss(self, logits, items):
        return -torch.mean(torch.sum(F.log_softmax(logits, 1) * items, -1))

    def user_loss(self, user_logits, user_items):
        return self.multinomial_loss(user_logits, user_items)

    def info_max_group_loss(self, group_logits, group_embeds, scores_ug, group_mask, group_items, user_items,
                            corrupted_user_items, device="cpu"):
        group_user_embeds = self.user_preference_encoder(user_items)  # [B, G, D]
        corrupt_user_embeds = self.user_preference_encoder(corrupted_user_items)  # [B, N, D]

        scores_observed = self.discriminator(group_embeds, group_user_embeds)  # [B, G]
        scores_corrupted = self.discriminator(group_embeds, corrupt_user_embeds)  # [B, N]

        mi_loss = self.discriminator.mi_loss(scores_observed, group_mask, scores_corrupted, device=device)

        ui_sum = user_items.sum(2, keepdim=True)  # [B, G]
        user_items_norm = user_items / torch.max(torch.ones_like(ui_sum), ui_sum)  # [B, G, I]
        gi_sum = group_items.sum(1, keepdim=True)
        group_items_norm = group_items / torch.max(torch.ones_like(gi_sum), gi_sum)  # [B, I]
        assert scores_ug.requires_grad is False

        group_mask_zeros = torch.exp(group_mask).unsqueeze(2)  # [B, G, 1]
        scores_ug = torch.sigmoid(scores_ug)  # [B, G, 1]

        user_items_norm = torch.sum(user_items_norm * scores_ug * group_mask_zeros, dim=1) / group_mask_zeros.sum(1)
        user_group_loss = self.multinomial_loss(group_logits, user_items_norm)
        group_loss = self.multinomial_loss(group_logits, group_items_norm)

        return mi_loss, user_group_loss, group_loss

    def loss(self, group_logits, summary_embeds, scores_ug, group_mask, group_items, user_items, corrupted_user_items,
             device='cpu'):
        """ L_G + lambda L_UG + L_MI """
        mi_loss, user_group_loss, group_loss = self.info_max_group_loss(group_logits, summary_embeds, scores_ug,
                                                                        group_mask, group_items, user_items,
                                                                        corrupted_user_items, device)

        return group_loss + mi_loss + self.lambda_mi * user_group_loss
