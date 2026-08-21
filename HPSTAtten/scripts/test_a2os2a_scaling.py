#!/usr/bin/env python3
"""Tests scaling A²OS²A vs VSSA dans HP-STAtten."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))

from modules.a2os2a_scaling import (
    a2os2a_factorized_scaling,
    resolve_contrast_scaling,
    resolve_factorized_scaling,
    vssa_factorized_scaling,
)
from modules.hp_stattn import HPSTAtten


def test_scaling_constants() -> None:
    assert a2os2a_factorized_scaling() == 1.0
    assert vssa_factorized_scaling(dvs=False, spatial_h=8, chunk_size=2) == 0.125
    assert resolve_factorized_scaling(
        hybrid_qkv=True, dvs=False, spatial_h=8, chunk_size=2
    ) == 1.0
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


def main() -> None:
    test_scaling_constants()
    test_hp_statten_hybrid_no_vssa_scaling()
    print("OK — a2os2a scaling tests passed")


if __name__ == "__main__":
    main()
