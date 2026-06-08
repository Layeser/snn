import torch
from spikingjelly.clock_driven import functional
from tqdm import tqdm

from utils.metrics import accuracy


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Train", leave=True, mininterval=1.0)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        functional.reset_net(model)
        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_acc = accuracy(logits, labels)
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
