"""Scaling et stabilisation K pour hybrid A²OS²A dans HP-STAtten (STAtten factorisé).

Papier Guo et al. CVPR 2025 (§4.3) : pas de softmax ni scaling explicite avec
Q binaire, K ReLU float, V ternaire — contrairement à VSSA / Spikformer (1/H).
"""
from __future__ import annotations

import torch


def vssa_factorized_scaling(*, dvs: bool, spatial_h: int, chunk_size: int) -> float:
    """Scaling STAtten / Spikformer pour Q,K,V binaires."""
    if dvs:
        return 1.0 / (spatial_h * spatial_h * chunk_size)
    return 1.0 / spatial_h


def a2os2a_factorized_scaling() -> float:
    """A²OS²A : pas de facteur Spikformer sur KᵀV (papier eq. 28, forme factorisée)."""
    return 1.0


def resolve_factorized_scaling(
    *,
    hybrid_qkv: bool,
    dvs: bool,
    spatial_h: int,
    chunk_size: int,
) -> float:
    if hybrid_qkv:
        return a2os2a_factorized_scaling()
    return vssa_factorized_scaling(dvs=dvs, spatial_h=spatial_h, chunk_size=chunk_size)


def resolve_contrast_scaling(*, hybrid_qkv: bool, dvs: bool, spatial_h: int) -> float:
    if hybrid_qkv:
        return a2os2a_factorized_scaling()
    if dvs:
        return 1.0 / (spatial_h * spatial_h)
    return 1.0 / spatial_h


def stabilize_hybrid_keys(k: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """K ReLU post-BN : RMS par token/tête (contrast/SDT agrège sur N tokens)."""
    scale = k.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=eps)
    return k / scale
