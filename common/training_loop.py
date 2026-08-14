"""Boucle train/val partagée (AMP CUDA, mixup, TET)."""

from __future__ import annotations

import torch
from spikingjelly.clock_driven import functional
from tqdm import tqdm

from tet_loss import tet_loss
from training_recipe import mixup_criterion, mixup_data


def _model_logits(model, images, *, return_timesteps: bool):
    if return_timesteps:
        return model(images, return_timesteps=True)
    return model(images)


def _compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    criterion,
    *,
    use_tet: bool,
    tet_means: float,
    tet_lamb: float,
    labels_a=None,
    labels_b=None,
    lam: float = 1.0,
    mixup_active: bool = False,
):
    if use_tet:
        if mixup_active:
            loss_a = tet_loss(logits, labels_a, criterion, means=tet_means, lamb=tet_lamb)
            loss_b = tet_loss(logits, labels_b, criterion, means=tet_means, lamb=tet_lamb)
            return lam * loss_a + (1.0 - lam) * loss_b
        return tet_loss(logits, labels, criterion, means=tet_means, lamb=tet_lamb)

    if mixup_active:
        return mixup_criterion(criterion, logits, labels_a, labels_b, lam)
    return criterion(logits, labels)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return 100.0 * (preds == labels).float().mean().item()


def _batch_accuracy(logits: torch.Tensor, labels: torch.Tensor, *, use_tet: bool) -> float:
    if use_tet:
        logits = logits.mean(0)
    return _accuracy(logits, labels)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    mixup_alpha: float = 0.0,
    use_amp: bool = False,
    use_tet: bool = False,
    tet_means: float = 1.0,
    tet_lamb: float = 0.0,
    grad_clip_max_norm: float | None = None,
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
                logits = _model_logits(model, images, return_timesteps=use_tet)
                loss = _compute_loss(
                    logits,
                    labels,
                    criterion,
                    use_tet=use_tet,
                    tet_means=tet_means,
                    tet_lamb=tet_lamb,
                    labels_a=labels_a,
                    labels_b=labels_b,
                    lam=lam,
                    mixup_active=True,
                )
                batch_acc = _batch_accuracy(logits, labels_a, use_tet=use_tet)
            else:
                logits = _model_logits(model, images, return_timesteps=use_tet)
                loss = _compute_loss(
                    logits,
                    labels,
                    criterion,
                    use_tet=use_tet,
                    tet_means=tet_means,
                    tet_lamb=tet_lamb,
                )
                batch_acc = _batch_accuracy(logits, labels, use_tet=use_tet)

        scaler.scale(loss).backward()
        if grad_clip_max_norm is not None and grad_clip_max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_acc += batch_acc
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.2f}%")

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    use_amp: bool = False,
    use_tet: bool = False,
    tet_means: float = 1.0,
    tet_lamb: float = 0.0,
):
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
            logits = _model_logits(model, images, return_timesteps=use_tet)
            if use_tet:
                logits = logits.mean(0)
            loss = criterion(logits, labels)

        batch_acc = _accuracy(logits, labels)
        total_loss += loss.item()
        total_acc += batch_acc
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.2f}%")

    return total_loss / n_batches, total_acc / n_batches
