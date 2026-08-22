#!/usr/bin/env python3
"""Tests scaling A²OS²A vs VSSA dans HP-STAtten."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))

from modules.a2os2a_scaling import (
    a2os2a_factorized_scaling,
    hybrid_factorized_dvs_token_norm,
    resolve_contrast_scaling,
    resolve_factorized_scaling,
    vssa_factorized_scaling,
)
from modules.hp_stattn import HPSTAtten
import torch
from spikingjelly.clock_driven import functional


def test_scaling_constants() -> None:
    assert a2os2a_factorized_scaling() == 1.0
    assert vssa_factorized_scaling(dvs=False, spatial_h=8, chunk_size=2) == 0.125
    assert resolve_factorized_scaling(
        hybrid_qkv=True, dvs=False, spatial_h=8, chunk_size=2
    ) == 1.0
    assert resolve_factorized_scaling(
        hybrid_qkv=True, dvs=True, spatial_h=16, chunk_size=4
    ) == hybrid_factorized_dvs_token_norm(spatial_h=16, chunk_size=4)
    assert resolve_factorized_scaling(
        hybrid_qkv=False, dvs=False, spatial_h=8, chunk_size=2
    ) == 0.125
    assert resolve_contrast_scaling(hybrid_qkv=True, dvs=False, spatial_h=8) == 1.0
    assert resolve_contrast_scaling(hybrid_qkv=False, dvs=False, spatial_h=8) == 0.125


def test_hp_statten_hybrid_no_vssa_scaling() -> None:
    hyb = HPSTAtten(dim=256, num_heads=8, chunk_size=2, hybrid_qkv=True, dvs=False)
    bin_ = HPSTAtten(dim=256, num_heads=8, chunk_size=2, hybrid_qkv=False, dvs=False)
    assert hyb._factorized_scaling(8) == 1.0
    assert bin_._factorized_scaling(8) == 0.125

    hyb_dvs = HPSTAtten(
        dim=256, num_heads=8, chunk_size=4, hybrid_qkv=True, dvs=True,
        attention_mode="factorized", lif_backend="torch",
    )
    assert hyb_dvs._factorized_scaling(16) == 1.0 / (16 * 16 * 4)


def test_hybrid_factorized_dvs_forward_finite() -> None:
    """Régression NaN/AMP — hybrid factorized DVS (grille v0)."""
    torch.backends.mkldnn.enabled = False
    attn = HPSTAtten(
        dim=512,
        num_heads=8,
        chunk_size=4,
        hybrid_qkv=True,
        dvs=True,
        attention_mode="factorized",
        lif_backend="torch",
    )
    functional.reset_net(attn)
    x = torch.randn(16, 2, 512, 16, 16)
    y = attn(x)
    assert torch.isfinite(y).all(), f"non-finite output: min={y.min()} max={y.max()}"
    functional.reset_net(attn)
    y.sum().backward()
    assert attn.v_conv.weight.grad is not None
    assert torch.isfinite(attn.v_conv.weight.grad).all()
    assert attn.v_conv.weight.grad.norm() > 0


def main() -> None:
    test_scaling_constants()
    test_hp_statten_hybrid_no_vssa_scaling()
    test_hybrid_factorized_dvs_forward_finite()
    print("OK — a2os2a scaling tests passed")


if __name__ == "__main__":
    main()
