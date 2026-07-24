"""TET — Temporal Efficient Training (STAtten / Spike-Driven Transformer)."""

from __future__ import annotations

import torch
import torch.nn as nn


def tet_loss(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    *,
    means: float = 1.0,
    lamb: float = 0.0,
) -> torch.Tensor:
    """Loss multi-pas : moyenne de la CE sur chaque timestep + régularisation MMD optionnelle.

    ``outputs`` : (T, B, num_classes) — logits par pas de temps (avant mean temporel).
    ``lamb`` : poids de la régularisation MSE vers ``means`` (0 = TET pur, défaut STAtten).
    """
    if outputs.dim() != 3:
        raise ValueError(f"tet_loss attend (T, B, C), reçu {tuple(outputs.shape)}")

    t_steps = outputs.size(0)
    loss_es = sum(criterion(outputs[t], labels) for t in range(t_steps)) / t_steps

    if lamb == 0:
        return loss_es

    mmd = nn.MSELoss()
    target = torch.zeros_like(outputs).fill_(means)
    return (1 - lamb) * loss_es + lamb * mmd(outputs, target)
