import dataloader
import model
import torch
import utils
from torch import optim
import numpy as np
import argparse
import random
from datetime import datetime
from tensorboardX import SummaryWriter
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


def model_leave_one_test(test_ratings, test_negatives, device, k_list, mode='user'):
    rec_model.eval()

    test_hits, test_ndcgs = [], []
    pred_score = np.zeros((len(test_ratings), len(test_negatives[0]) + 1))

    if mode == 'user':
        users, items = rec_model.compute()
    else:
        users, items, all_centers, all_offsets = rec_model.compute_all()

    for idx in range(len(test_ratings)):
        rating = test_ratings[idx]

        test_user, test_items = rating[0], [rating[1]] + test_negatives[idx]

        if mode == 'user':
            test_user_emb = users[test_user].detach().cpu().numpy()
            test_items_emb = items[test_items].detach().cpu().numpy()
            # For query user, compute the distance with all candidate items
            score = np.sqrt(np.sum(np.asarray(test_user_emb - test_items_emb) ** 2, axis=1))
        else:
            test_centers = all_centers[[test_user]].detach().cpu().numpy()
            test_offsets = all_offsets[[test_user]].detach().cpu().numpy()
            test_centers = np.repeat(test_centers, len(test_items), axis=0)
            test_offsets = np.repeat(test_offsets, len(test_items), axis=0)
            # print(centers.shape, offsets.shape)
            score = rec_model.gi_scores(torch.FloatTensor(test_centers).to(device), torch.FloatTensor(test_offsets).to(device),
                                        torch.LongTensor(test_items).to(device), all_items)
            score = score.detach().cpu().numpy().reshape(1, -1)

        pred_score[idx, :] = score

    pred_rank = np.argsort(pred_score, axis=1)

    for k in k_list:
        test_hits.append(utils.get_hit_k(pred_rank, k))
        test_ndcgs.append(utils.get_ndcg_k(pred_rank, k))

    return test_hits, test_ndcgs


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="[Mafengwo, CAMRa2011]", default="Mafengwo")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, help="[cuda:0, ..., cpu]", default="cpu")

# Hyper-parameters
parser.add_argument("--emb_dim", type=int, default=64)
parser.add_argument("--n_layers", type=int, default=3)
parser.add_argument("--keep_prob", type=float, default=0.8)

parser.add_argument("--epoch", type=int, default=5)
parser.add_argument("--pretrain_epoch", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--topK", type=list, default=[1, 5, 10])
parser.add_argument("--num_negatives", type=int, default=4)
parser.add_argument("--group_agg", type=str, help="[geometric, attentive]", default="geometric")
# Self-supervised loss weight
parser.add_argument("--mu", type=float, default=0.5)

args = parser.parse_args()
set_seed(args.seed)
device = torch.device(args.device)

print('= ' * 20)
print('## Starting Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print(args)
writer_dir = f"ckpts/{args.dataset}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
writer = SummaryWriter(writer_dir)

dataset = dataloader.GroupDataset(num_negatives=args.num_negatives, dataset=args.dataset)
group_member_dict = dataset.group_member_dict
rec_model = model.CubeRec(args, dataset, device)
rec_model = rec_model.to(device)
opt = optim.Adam(rec_model.parameters(), lr=args.lr)

print("PRETRAIN on User-Item interactions...")
for epoch_id in range(args.pretrain_epoch):
    rec_model.train()

    ui_loader = dataset.get_user_dataloader(args.batch_size)

    losses = []
    for _, (u, pi_ni) in enumerate(ui_loader):
        ui_loss, reg_loss = rec_model.bpr_loss(u.to(device), pi_ni[:, 0].to(device), pi_ni[:, 1].to(device))

        # We set the coef of regulation term as 0.1
        user_rec_loss = ui_loss + 0.1 * reg_loss
        losses.append(user_rec_loss)

        opt.zero_grad()
        user_rec_loss.backward()
        opt.step()

    print(f"[Epoch {epoch_id}] UI loss {torch.mean(torch.stack(losses)):.4f}")
    hits, ndcgs = model_leave_one_test(dataset.user_test_ratings, dataset.user_test_negatives, device, args.topK)
    print(f"[Epoch {epoch_id}] User, Hit@{args.topK}: {hits}, NDCG@{args.topK}: {ndcgs}")

print()

print("TRAIN on Group-Item interactions...")
for epoch_id in range(args.epoch):
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

    print(f"[Epoch {epoch_id}] GI loss {torch.mean(torch.stack(losses)):.4f}")
    hits, ndcgs = model_leave_one_test(dataset.user_test_ratings, dataset.user_test_negatives, device, args.topK)
    writer.add_scalars(f'User/Hit@{args.topK}', {str(args.topK[i]): hits[i] for i in range(len(args.topK))}, epoch_id)
    writer.add_scalars(f'User/NDCG@{args.topK}', {str(args.topK[i]): ndcgs[i] for i in range(len(args.topK))},
                       epoch_id)

    print(f"[Epoch {epoch_id}] User, Hit@{args.topK}: {hits}, NDCG@{args.topK}: {ndcgs}")
    hits, ndcgs = model_leave_one_test(dataset.group_test_ratings, dataset.group_test_negatives, device, args.topK,
                                       mode='group')
    print(f"[Epoch {epoch_id}] Group, Hit@{args.topK}: {hits}, NDCG@{args.topK}: {ndcgs}")
    writer.add_scalars(f'Group/Hit@{args.topK}', {str(args.topK[i]): hits[i] for i in range(len(args.topK))}, epoch_id)
    writer.add_scalars(f'Group/NDCG@{args.topK}', {str(args.topK[i]): ndcgs[i] for i in range(len(args.topK))}, epoch_id)

print()
print('## Finishing Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print('= ' * 20)
print("Done!")
