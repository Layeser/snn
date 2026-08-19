#!/usr/bin/env python3
"""Smoke tests — gradients TernaryLIF + v_conv (hybrid vs binary)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional

torch.backends.mkldnn.enabled = False

from modules.hp_stattn import HPSTAtten
from modules.spike import make_lif
from modules.ternary_lif import MultiStepTernaryLIFNode


def _grad_norm(t: torch.Tensor | None) -> float:
    if t is None:
        return 0.0
    return float(t.norm().item())


def smoke_ternary_lif() -> None:
    print("=== 1. MultiStepTernaryLIFNode seul ===")
    node = MultiStepTernaryLIFNode(tau=2.0, v_threshold=1.0)
    x = torch.randn(4, 2, 8, requires_grad=True)
    node.reset()
    y = node(x)
    y.sum().backward()

    assert x.grad is not None, "x.grad is None"
    assert x.grad.norm() > 0, f"x.grad norm={x.grad.norm()}"
    uniq = torch.unique(y.detach()).tolist()
    assert set(uniq).issubset({-1.0, 0.0, 1.0}), f"spikes hors {{-1,0,1}}: {uniq}"
    print(f"  x.grad norm: {_grad_norm(x.grad):.6f}")
    print(f"  spike values: {sorted(set(round(v, 4) for v in uniq))}")
    print("  OK")


def _mini_v_conv_grad(*, hybrid: bool) -> torch.Tensor | None:
    """Chemin V minimal : v_conv → BN → (TernaryLIF | LIF)."""
    conv = nn.Conv2d(256, 256, 1, bias=False)
    bn = nn.BatchNorm2d(256)
    x = torch.randn(4, 2, 256, 8, 8)
    t, b, c, h, w = x.shape
    v = bn(conv(x.flatten(0, 1))).reshape(t, b, c, h, w)
    if hybrid:
        node = MultiStepTernaryLIFNode()
        node.reset()
        out = node(v)
    else:
        lif = make_lif("lif", lif_backend="torch")
        functional.reset_net(lif)
        out = lif(v)
    out.sum().backward()
    return conv.weight.grad


def smoke_v_conv_grad() -> None:
    print("\n=== 2. v_conv.weight.grad (mini module V-path) ===")
    g_hyb = _mini_v_conv_grad(hybrid=True)
    g_bin = _mini_v_conv_grad(hybrid=False)

    print(
        f"  hybrid_qkv=True  v_conv.grad norm: {_grad_norm(g_hyb):.6f}"
        f" ({'None' if g_hyb is None else 'OK'})"
    )
    print(
        f"  hybrid_qkv=False v_conv.grad norm: {_grad_norm(g_bin):.6f}"
        f" ({'None' if g_bin is None else 'OK'})"
    )

    assert g_hyb is not None and g_hyb.norm() > 0, "hybrid: v_conv grad missing"
    assert g_bin is not None and g_bin.norm() > 0, "binary: v_conv grad missing (regression)"
    print("  OK")


def smoke_hp_statten_v_conv() -> None:
    print("\n=== 2b. HPSTAtten end-to-end v_conv.weight.grad ===")
    for hybrid in (True, False):
        attn = HPSTAtten(
            dim=256,
            num_heads=8,
            chunk_size=2,
            hybrid_qkv=hybrid,
            attention_mode="factorized",
            lif_backend="torch",
        )
        functional.reset_net(attn)
        x = torch.randn(4, 2, 256, 8, 8)
        attn(x).sum().backward()
        g = attn.v_conv.weight.grad
        label = "hybrid" if hybrid else "binary"
        print(f"  {label}: v_conv.grad norm {_grad_norm(g):.6f} ({'None' if g is None else 'OK'})")
        assert g is not None, f"{label}: v_conv grad is None"
        if hybrid:
            assert g.norm() > 0, "hybrid: v_conv grad norm is zero"
    print("  OK")


def smoke_hybrid_spike_values() -> None:
    print("\n=== 3. Spikes hybrid ∈ {{-1,0,1}} via HPSTAtten ===")
    attn = HPSTAtten(dim=256, num_heads=8, chunk_size=2, hybrid_qkv=True, lif_backend="torch")
    functional.reset_net(attn)
    x = torch.randn(4, 2, 256, 8, 8)
    with torch.no_grad():
        t, b, c, h, w = x.shape
        v = attn.v_bn(attn.v_conv(x.flatten(0, 1))).reshape(t, b, c, h, w)
        spikes = attn.v_ternary(v)
    uniq = torch.unique(spikes).tolist()
    assert set(round(u, 4) for u in uniq).issubset({-1.0, 0.0, 1.0})
    print(f"  unique spike values: {sorted(set(round(v, 4) for v in uniq))}")
    print("  OK")


def main() -> None:
    smoke_ternary_lif()
    smoke_v_conv_grad()
    smoke_hp_statten_v_conv()
    smoke_hybrid_spike_values()
    print("\nAll TernaryLIF gradient smoke tests passed.")


if __name__ == "__main__":
    main()
