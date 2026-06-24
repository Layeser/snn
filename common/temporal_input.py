"""Conversion des tenseurs d'entrée vers le format temporel (T, B, C, H, W)."""

from __future__ import annotations

import torch


def to_temporal_input(x: torch.Tensor, T: int) -> torch.Tensor:
    """
    (B, C, H, W)  → répétition statique → (T, B, C, H, W)
    (B, T, C, H, W) → séquence DVS       → (T, B, C, H, W)
    """
    if x.dim() == 5:
        return x.permute(1, 0, 2, 3, 4).contiguous()
    if x.dim() == 4:
        return x.unsqueeze(0).expand(T, -1, -1, -1, -1)
    raise ValueError(f"Entrée attendue en 4D ou 5D, reçu {x.dim()}D (shape={tuple(x.shape)})")
