"""Helpers d'entraînement partagés (recette officielle STAtten / Spikformer)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def mixup_data(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup batch-wise (comme timm / repos officiels)."""
    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[index]
    return mixed, labels, labels[index], lam


def mixup_criterion(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay avec warmup linéaire (comme conf STAtten / Spikformer)."""

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        if not base_lrs:
            return 1.0
        min_factor = min_lr / base_lrs[0]
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
