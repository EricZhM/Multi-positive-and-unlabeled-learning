
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI training script (single-run) for NTMP/MPU experiments.

Example:
  python experiment_.py \
    --dataset MNIST \
    --classes 4 \
    --mode mpau5 \
    --kprior 0.5 \
    --noise 1.0 \
    --epochs 50 \
    --batchsize 512 \
    --seed 42 \
    --save-dir runs

It will write metrics CSV to: runs/<dataset>_<mode>_<classes>_<kprior>_<seed>/metrics.csv
Optionally, add --save-margins to dump logits and labels from the final epoch.
"""

import os
import csv
import time
import random
from datetime import datetime
import argparse

import torch
from torch.utils.data import DataLoader
import numpy as np

from loss import mpan_loss
from custommodel import MLP, resnet20
from dataset import CustomDataset


SUPPORTED_DATASETS = ['MNIST', 'FashionMNIST', 'USPS', 'KMNIST', 'SVHN']


def set_seed(seed: int):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_net(dataset_name, ds_nb):
    if dataset_name in ['MNIST', 'FashionMNIST', 'USPS', 'KMNIST']:
        return MLP(dataset_name, ds_nb)
    elif dataset_name == 'SVHN':
        return resnet20(num_classes=ds_nb)
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset_name}")


def get_optimizer(dataset_name, model):
    if dataset_name in ['MNIST', 'FashionMNIST']:
        return torch.optim.Adam(model.parameters(), lr=1e-3)
    elif dataset_name in ['USPS', 'KMNIST']:
        return torch.optim.Adam(model.parameters(), lr=1e-4)
    elif dataset_name == 'SVHN':
        return torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset_name}")


@torch.no_grad()
def multiclass_metrics(pred, true, num_classes=None, average='macro'):
    pred_labels = torch.argmax(pred, dim=1)

    if num_classes is None:
        num_classes = pred.size(1)

    tp = torch.zeros(num_classes, dtype=torch.float, device=pred.device)
    fp = torch.zeros(num_classes, dtype=torch.float, device=pred.device)
    fn = torch.zeros(num_classes, dtype=torch.float, device=pred.device)

    for i in range(num_classes):
        pred_pos = (pred_labels == i)
        act_pos = (true == i)
        tp[i] = (pred_pos & act_pos).sum().float()
        fp[i] = (pred_pos & ~act_pos).sum().float()
        fn[i] = (~pred_pos & act_pos).sum().float()

    accuracy = (pred_labels == true).float().mean()

    precision_per_class = tp / (tp + fp + 1e-8)
    recall_per_class = tp / (tp + fn + 1e-8)
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + 1e-8)

    if average == 'macro':
        precision = precision_per_class.mean()
        recall = recall_per_class.mean()
        f1 = f1_per_class.mean()
    elif average == 'micro':
        total_tp = tp.sum()
        total_fp = fp.sum()
        total_fn = fn.sum()
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    else:
        raise ValueError("average must be 'macro' or 'micro'")

    return (accuracy.item(), precision.item(), recall.item(), f1.item())


def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / (batch + 1)


@torch.no_grad()
def evaluate(dataloader, model, loss_fn, device, num_classes, save_margins=False, margins_path=None):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_logits, all_labels = [], []

    logits = None
    y = None

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        logits = model(X)

        loss = loss_fn(logits, y)
        total_loss += loss.item()
        n_batches += 1

        if save_margins:
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())

    avg_loss = total_loss / max(1, n_batches)

    if save_margins and len(all_logits) > 0:
        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        acc, prec, rec, f1 = multiclass_metrics(logits_cat, labels_cat, num_classes=num_classes)
        if margins_path is not None:
            os.makedirs(margins_path, exist_ok=True)
            np.save(os.path.join(margins_path, "pred.npy"), logits_cat.numpy())
            np.save(os.path.join(margins_path, "y.npy"), labels_cat.numpy().astype('int64'))
    else:
        acc, prec, rec, f1 = multiclass_metrics(logits, y, num_classes=num_classes)

    return avg_loss, acc, prec, rec, f1


def main():
    parser = argparse.ArgumentParser(description="NTMP/MPU single-run trainer (CLI)")
    parser.add_argument("--dataset", type=str, choices=SUPPORTED_DATASETS, required=True,
                        help="Dataset name")
    parser.add_argument("--classes", type=int, required=True,
                        help="Number of classes (ds_nb)")
    parser.add_argument("--mode", type=str, default="ABS",
                        choices=["URE", "NN", "ABS"], help="Training mode / tuple scheme")
    parser.add_argument("--kprior", type=float, default=0.5,
                        help="Class prior for unlabeled set (e.g., 0.5 or 0.8)")
    parser.add_argument("--noise", type=float, default=1.0,
                        help="Noise multiplier for prior in loss")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batchsize", type=int, default=512, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="runs",
                        help="Directory to save CSV and (optional) margins")
    parser.add_argument("--save-margins", action="store_true",
                        help="Save logits/labels (pred.npy, y.npy) for the final evaluation")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using {device} device")

    ds = CustomDataset(dataset_name=args.dataset, train_flag=True,
                       dataset_number=args.classes, kprior=args.kprior, root='D:\\data')
    tra_ds, tes_ds = ds()

    train_loader = DataLoader(tra_ds, batch_size=args.batchsize, shuffle=True)
    test_loader = DataLoader(tes_ds, batch_size=len(tes_ds.targets), shuffle=False)

    model = get_net(args.dataset, args.classes).to(device)
    loss_fn = mpan_loss(ds.get_pri(), args.mode, args.noise)
    optimizer = get_optimizer(args.dataset, model)

    now_time = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
    print('time now: ' + now_time)
    print(f'{args.dataset}\tmode:{args.mode}\tclasses:{args.classes}\tkprior:{args.kprior}\tseed:{args.seed}')

    save_dir = os.path.join(args.save_dir, f"{args.dataset}_{args.mode}_{args.classes}_{args.kprior}_{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    train_losses = []
    eval_losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        t0 = time.time()

        tr_loss = train_one_epoch(train_loader, model, loss_fn, optimizer, device)
        ev_loss, acc, prec, rec, f1 = evaluate(test_loader, model, loss_fn, device,
                                               num_classes=args.classes,
                                               save_margins=(args.save_margins and (epoch == args.epochs - 1)),
                                               margins_path=save_dir)

        train_losses.append(tr_loss)
        eval_losses.append(ev_loss)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        print(f"train_loss:{tr_loss:.4f}  eval_loss:{ev_loss:.4f}  "
              f"acc:{acc*100:.2f}  prec:{prec*100:.2f}  rec:{rec*100:.2f}  f1:{f1*100:.2f}  "
              f"time:{time.time()-t0:.2f}s")

    csv_path = os.path.join(save_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_loss"] + train_losses)
        writer.writerow(["eval_loss"] + eval_losses)
        writer.writerow(["accuracy"] + accuracies)
        writer.writerow(["precision"] + precisions)
        writer.writerow(["recall"] + recalls)
        writer.writerow(["f1"] + f1s)

    print(f"Saved metrics to: {csv_path}")
    if args.save_margins:
        print(f"Saved margins (pred.npy, y.npy) under: {save_dir}")


if __name__ == "__main__":
    main()
