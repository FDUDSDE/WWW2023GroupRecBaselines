import dataloader
import model
import torch
import metrics
from torch import optim
import numpy as np
import argparse
import random
from datetime import datetime
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="[Mafengwo, CAMRa2011]", default="Mafengwo")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, help="[cuda:0, ..., cpu]", default="cuda:0")

# Hyper-parameters
parser.add_argument("--emb_dim", type=int, default=64)
parser.add_argument("--n_layers", type=int, default=3)
parser.add_argument("--keep_prob", type=float, default=0.8)

# Group training epochs
parser.add_argument("--epoch", type=int, default=50)
# User pre-train epochs
parser.add_argument("--pretrain_epoch", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--topK", type=list, default=[1, 5, 10])
parser.add_argument("--num_negatives", type=int, default=4)
parser.add_argument("--group_agg", type=str, help="[geometric, attentive]", default="geometric")

args = parser.parse_args()
set_seed(args.seed)
device = torch.device(args.device)

print('= ' * 20)
print('## Starting Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print(args)

dataset = dataloader.GroupDataset(num_negatives=args.num_negatives, dataset=args.dataset)
group_member_dict = dataset.group_member_dict
rec_model = model.CubeRec(args, dataset, device)
rec_model = rec_model.to(device)
opt = optim.Adam(rec_model.parameters(), lr=args.lr)

print("PRETRAIN on User-Item interactions...")
for epoch_id in range(args.pretrain_epoch):
    st_time = time.time()
    rec_model.train()

    ui_loader = dataset.get_user_dataloader(args.batch_size)

    losses = []
    for _, (u, pi_ni) in enumerate(ui_loader):
        ui_loss, reg_loss = rec_model.bpr_loss(u.to(device), pi_ni[:, 0].to(device), pi_ni[:, 1].to(device))

        user_rec_loss = ui_loss + 0.1 * reg_loss
        losses.append(user_rec_loss)

        opt.zero_grad()
        user_rec_loss.backward()
        opt.step()

    print(f"[Epoch {epoch_id}] UI loss {torch.mean(torch.stack(losses)):.4f}, Cost time {time.time()-st_time:.4f}s")
    hits, ndcgs = metrics.model_leave_one_test(rec_model, dataset.user_test_ratings, dataset.user_test_negatives, device, args.topK)
    print(f"[Epoch {epoch_id}] User, Hit@{args.topK}: {hits}, NDCG@{args.topK}: {ndcgs}")

print()

print("TRAIN on Group-Item interactions...")
for epoch_id in range(args.epoch):
    st_time = time.time()
    rec_model.train()

    gi_loader = dataset.get_group_dataloader(args.batch_size)

    losses = []
    for _, (g, pi_ni) in enumerate(gi_loader):
        all_users, all_items = rec_model.compute()

        members = [torch.LongTensor(group_member_dict[group_id]).to(device) for group_id in list(g.cpu().numpy())]

        centers, offsets = rec_model.group_representations(members, all_users, device)

        pos_scores = rec_model.gi_scores(centers, offsets, pi_ni[:, 0].to(device), all_items)
        neg_scores = rec_model.gi_scores(centers, offsets, pi_ni[:, 1].to(device), all_items)

        group_rec_loss = torch.mean(torch.max(pos_scores - neg_scores + 0.5, torch.zeros(pos_scores.shape).to(device)))
        losses.append(group_rec_loss)

        opt.zero_grad()
        group_rec_loss.backward()
        opt.step()

    print(f"[Epoch {epoch_id}] GI loss {torch.mean(torch.stack(losses)):.4f}, Cost time {time.time() - st_time:.4f}s")
    hits, ndcgs = metrics.model_leave_one_test(rec_model, dataset.user_test_ratings, dataset.user_test_negatives, device, args.topK)
    print(f"[Epoch {epoch_id}] User, Hit@{args.topK}: {hits}, NDCG@{args.topK}: {ndcgs}")

    hits, ndcgs = metrics.model_leave_one_test(rec_model, dataset.group_test_ratings, dataset.group_test_negatives, device, args.topK, mode='group')
    print(f"[Epoch {epoch_id}] Group, Hit@{args.topK}: {hits}, NDCG@{args.topK}: {ndcgs}")

print()
print('## Finishing Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print('= ' * 20)
print("Done!")
