import argparse
import time
import numpy as np
import torch
from datetime import datetime
import random
import os
import utils
import model
import dataloader
import metric

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Mafengwo')

parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--wd', type=float, default=0.00, help='weight decay')
parser.add_argument('--lambda_mi', type=float, default=1.0)
parser.add_argument('--drop_ratio', type=float, default=0.4, help='Dropout ratio')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epoch', type=int, default=5, help='maximum # training epochs')
parser.add_argument('--group_epoch', type=int, default=1)
parser.add_argument("--device", type=str, default="cuda:0")

# Model settings.
parser.add_argument('--emb_dim', type=int, default=64, help='layer size')
parser.add_argument('--num_negatives', type=int, default=10, help='# negative users sampled per group')
parser.add_argument('--seed', type=int, default=1111, help='random seed for reproducibility')

args = parser.parse_args()
set_seed(args.seed)

device = torch.device(args.device)

print('= ' * 20)
print('## Starting Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print(args)

dataset = dataloader.GroupDataset(num_negatives=args.num_negatives, dataset=args.dataset)
n_users, n_items, n_groups = dataset.num_users, dataset.num_items, dataset.num_groups
group_member_dict = dataset.group_member_dict

fine_data_ug, coarse_data_ug = dataset.data_gu_fine.toarray(), dataset.data_gu_coarse.toarray()
data_gg_csr = dataset.H_gg.toarray()

H_ul_coarse = utils.generate_G_from_H(coarse_data_ug)
H_ul_coarse = torch.Tensor(H_ul_coarse).to(device)
H_ul_fine = utils.generate_G_from_H(fine_data_ug)
H_ul_fine = torch.Tensor(H_ul_fine).to(device)
H_gl = torch.Tensor(data_gg_csr).to(device)

rec_model = model.HHGR(n_items, n_users, n_groups, group_member_dict, args.emb_dim, drop_ratio=args.drop_ratio,
                       lambda_mi=args.lambda_mi, device=device).to(device)
# torch.autograd.set_detect_anomaly(True)

# User-Item交互训练，优化的是User/Item的Embedding
print("Pre-Training on user-item interactions...")
optimizer_ui = torch.optim.Adam(rec_model.parameters(), lr=0.0005, weight_decay=args.wd)
for epoch in range(args.epoch):
    rec_model.train()

    losses = []
    start_time = time.time()

    for batch_idx, (u, pi_ni) in enumerate(dataset.get_user_dataloader(args.batch_size)):
        user_input = u.to(device)
        pos_input, neg_input = pi_ni[:, 0].to(device), pi_ni[:, 1].to(device)

        pos_pred = rec_model.user_forward(user_input, pos_input)
        neg_pred = rec_model.user_forward(user_input, neg_input)

        optimizer_ui.zero_grad()
        loss = torch.mean((pos_pred - neg_pred - 1) ** 2)
        losses.append(float(loss))
        loss.backward()
        optimizer_ui.step()

    elapsed = time.time() - start_time
    print(f"[Epoch {epoch}] time {elapsed:4.2f}s, loss {np.mean(losses):4.4f}")
print("Pre-Training finish!\n")

# User自监督训练，优化的是两个用户的HyperGCN参数
print("Training on user-level self-supervised...")
optimizer_ul = torch.optim.Adam(rec_model.parameters(), lr=0.0005, weight_decay=args.wd)
all_user_embedding = rec_model.user_embedding(torch.LongTensor([i for i in range(n_users)]).to(device)).to(
    device).detach()
for epoch in range(args.epoch):
    rec_model.train()
    start_time = time.time()

    user_level_loss = []

    all_user_embed_coarse = rec_model.hgcn_coarse(all_user_embedding, H_ul_coarse)
    all_user_embed_fine = rec_model.hgcn_fine(all_user_embedding, H_ul_fine)

    for batch_idx, data in enumerate(dataset.get_user_ssl_dataloader(args.batch_size)):
        uid, neg_id = data

        u_emb_coarse = all_user_embed_coarse[uid.to(device)]
        u_emb_fine = all_user_embed_fine[uid.to(device)]
        u_emb_neg = all_user_embed_coarse[neg_id.to(device)]

        score_pos = rec_model.discriminator(u_emb_coarse, u_emb_fine)
        score_neg = rec_model.discriminator(u_emb_coarse, u_emb_neg)
        mi_loss = rec_model.discriminator.mi_loss(score_pos, score_neg, device)

        optimizer_ul.zero_grad()
        # with torch.autograd.detect_anomaly():
        mi_loss.backward(retain_graph=True)
        user_level_loss.append(float(mi_loss))
        optimizer_ul.step()

    elapsed = time.time() - start_time
    print(f"[Epoch {epoch}] time {elapsed:4.2f}s, loss {np.mean(user_level_loss):4.4f}")
print("Training on user-level finish!\n")

all_user_embedding = rec_model.user_embedding(torch.LongTensor([i for i in range(n_users)]).to(device)).to(
    device).detach()

all_user_embed_coarse = rec_model.hgcn_coarse(all_user_embedding, H_ul_coarse).detach()
all_user_embed_fine = rec_model.hgcn_fine(all_user_embedding, H_ul_fine).detach()
user_embedding = all_user_embed_coarse + all_user_embed_fine
user_embedding = user_embedding.to(device)
print("Training on group-level hypergraph learning...")
optimizer_gr = torch.optim.Adam(rec_model.parameters(), lr=0.0001, weight_decay=args.wd)
for epoch in range(args.group_epoch):
    rec_model.train()
    start_time = time.time()

    group_level_loss = []

    group_embedding = rec_model.group_embedding(torch.LongTensor([g for g in range(n_groups)]).to(device)).to(device)
    group_embedding = rec_model.hgcn_gl(group_embedding, H_gl).to(device)

    for batch_idx, (g, pi_ni) in enumerate(dataset.get_group_dataloader(args.batch_size)):
        if batch_idx % 100 == 0:
            print(f"batch {batch_idx}, {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        group_input = g.to(device)
        pos_input, neg_input = pi_ni[:, 0].to(device), pi_ni[:, 1].to(device)

        pos_pred = rec_model.group_forward(group_input, pos_input, user_embedding, group_embedding)
        neg_pred = rec_model.group_forward(group_input, neg_input, user_embedding, group_embedding)

        optimizer_gr.zero_grad()
        loss = torch.mean((pos_pred - neg_pred - 1) ** 2)
        group_level_loss.append(float(loss))

        # with torch.autograd.detect_anomaly():
        loss.backward(retain_graph=True)
        optimizer_gr.step()
    elapsed = time.time() - start_time
    print(f"[Epoch {epoch}] time {elapsed:4.2f}s, loss {np.mean(group_level_loss):4.4f}")
    print("Training on group-level finish!\n")

###############################################################################
# Evaluate
###############################################################################
all_user_embedding = rec_model.user_embedding(torch.LongTensor([u for u in range(n_users)]).to(device)).to(
    device).detach()
all_user_embed_coarse = rec_model.hgcn_coarse(all_user_embedding, H_ul_coarse).to(device)
all_user_embed_fine = rec_model.hgcn_fine(all_user_embedding, H_ul_fine).to(device)
users_embed = (all_user_embed_coarse + all_user_embed_fine).to(device)

all_group_embedding = rec_model.group_embedding(torch.LongTensor([g for g in range(n_groups)]).to(device)).to(
    device).detach()
groups_embed = rec_model.hgcn_gl(all_group_embedding, H_gl).to(device)

hits, ndcgs = metric.evaluate(rec_model, dataset.user_test_ratings, dataset.user_test_negatives, device, [1, 5, 10],
                              None, None, 'user')
print(f"[Evaluate] User Hit@[1, 5, 10] {hits}, NDCG@[1, 5, 10] {ndcgs}")

hits, ndcgs = metric.evaluate(rec_model, dataset.group_test_ratings, dataset.group_test_negatives, device, [1, 5, 10],
                              users_embed, groups_embed, 'group')
print(f"[Evaluate] Group Hit@[1, 5, 10] {hits}, NDCG@[1, 5, 10] {ndcgs}")

print('## Finishing Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print('= ' * 20)
print("Done!")
