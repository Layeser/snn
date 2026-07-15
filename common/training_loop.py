"""Boucle train/val partagée (AMP CUDA, mixup)."""

from __future__ import annotations

import torch
from spikingjelly.clock_driven import functional
from tqdm import tqdm

from training_recipe import mixup_criterion, mixup_data


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return 100.0 * (preds == labels).float().mean().item()


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    mixup_alpha: float = 0.0,
    use_amp: bool = False,
):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    pbar = tqdm(loader, desc="Train", leave=True, mininterval=1.0)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        functional.reset_net(model)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            if mixup_alpha > 0:
                images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
                logits = model(images)
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                batch_acc = _accuracy(logits, labels_a)
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                batch_acc = _accuracy(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_acc += batch_acc
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.2f}%")

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp: bool = False):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    amp_enabled = use_amp and device.type == "cuda"

    pbar = tqdm(loader, desc="Val  ", leave=True, mininterval=1.0)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        functional.reset_net(model)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_acc = _accuracy(logits, labels)
        total_loss += loss.item()
        total_acc += batch_acc
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.2f}%")

    return total_loss / n_batches, total_acc / n_batches
