import sys
from pathlib import Path

import torch
from spikingjelly.clock_driven import functional
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from training_recipe import mixup_criterion, mixup_data

from utils.metrics import accuracy


def train_one_epoch(model, loader, criterion, optimizer, device, mixup_alpha=0.0):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Train", leave=True, mininterval=1.0)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        functional.reset_net(model)
        if mixup_alpha > 0:
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
            logits = model(images)
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            batch_acc = accuracy(logits, labels_a)
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            batch_acc = accuracy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += batch_acc
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.2f}%")

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Val  ", leave=True, mininterval=1.0)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        functional.reset_net(model)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_acc = accuracy(logits, labels)
        total_loss += loss.item()
        total_acc += batch_acc
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.2f}%")

    return total_loss / n_batches, total_acc / n_batches
