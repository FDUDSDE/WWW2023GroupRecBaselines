import dataloader
import model
import argparse
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
import random
import os
import metrics
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


def train(train_loader, epoch_id, lr, type_m='user'):
    rec_model.train()
    # optimizer = optim.Adam(rec_model.parameters(), lr)
    optimizer = optim.RMSprop(rec_model.parameters(), lr)

    losses = []

    for _, (u, pi_ni) in enumerate(train_loader):
        users, pos, neg = u.to(device), pi_ni[:, 0].to(device), pi_ni[:, 1].to(device)
        if type_m == 'user':
            pos_prediction = rec_model(None, users, pos)
            neg_prediction = rec_model(None, users, neg)
        else:
            pos_prediction = rec_model(users, None, pos)
            neg_prediction = rec_model(users, None, neg)

        rec_model.zero_grad()
        loss = torch.mean((pos_prediction - neg_prediction - 1)**2)
        losses.append(loss)
        loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch_id}] {type_m} loss: {torch.mean(torch.stack(losses)):.5f}")


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="[Mafengwo, CAMRa2011]", default="Mafengwo")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, help="[cuda:0, ..., cpu]", default="cuda:0")

parser.add_argument("--emb_dim", type=int, default=32)
parser.add_argument("--epoch", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--drop_ratio", type=float, default=0.2)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--topK", type=list, default=[1, 5, 10])
parser.add_argument("--num_negatives", type=int, default=4)


args = parser.parse_args()
set_seed(args.seed)
device = torch.device(args.device)

print('= ' * 20)
print('## Starting Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print(args)

dataset = dataloader.GroupDataset(num_negatives=args.num_negatives, dataset=args.dataset)
group_member_dict = dataset.group_member_dict

rec_model = model.AGREE(dataset.num_users, dataset.num_items, dataset.num_groups, args.emb_dim, group_member_dict, args.drop_ratio)
rec_model.to(device)


for epoch in range(args.epoch):
    train(dataset.get_user_dataloader(args.batch_size), epoch, args.lr, type_m='user')
    train(dataset.get_group_dataloader(args.batch_size), epoch, args.lr, type_m='group')

    hits, ndcgs = metrics.evaluate(rec_model, dataset.user_test_ratings, dataset.user_test_negatives, device, args.topK, type_m='user')
    print(f"[Epoch {epoch}] User, Hit@{args.topK}: {hits}, NDCG@{args.topK}: {ndcgs}")

    hits, ndcgs = metrics.evaluate(rec_model, dataset.group_test_ratings, dataset.group_test_negatives, device, args.topK, type_m='group')
    print(f"[Epoch {epoch}] Group, Hit@{args.topK}: {hits}, NDCG@{args.topK}: {ndcgs}")
    print()

print()
print('## Finishing Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print('= ' * 20)
print("Done!")
