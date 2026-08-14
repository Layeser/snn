#!/usr/bin/env python3
"""Diagnostic stabilité contrast / contrast_sdt sur CIFAR-10-DVS (lr=0.01, hybrid)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src"))

from models import HPSTAttenTransformer  # noqa: E402


def run_steps(
    *,
    attention_mode: str,
    hybrid_qkv: bool,
    lr: float,
    steps: int,
    device: torch.device,
) -> None:
    model = HPSTAttenTransformer(
        img_size=64,
        in_channels=2,
        num_classes=10,
        embed_dim=256,
        depth=2,
        num_heads=8,
        pooling_stat="0011",
        chunk_size=4,
        hybrid_qkv=hybrid_qkv,
        dvs=True,
        T=16,
        attention_mode=attention_mode,
        vct_num=16,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(4, 16, 2, 64, 64, device=device)
    y = torch.randint(0, 10, (4,), device=device)

    print(f"\n=== {attention_mode} hybrid={hybrid_qkv} lr={lr} ===")
    for step in range(1, steps + 1):
        functional.reset_net(model)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        if not torch.isfinite(logits).all():
            print(f"  step {step}: logits NON-FINIS")
            break
        loss = crit(logits, y)
        if not torch.isfinite(loss):
            print(f"  step {step}: loss NON-FINIS")
            break
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        acc = (logits.argmax(1) == y).float().mean().item() * 100
        print(f"  step {step:3d} loss={loss.item():.4f} grad_norm={float(gn):.3f} acc={acc:.1f}%")
    else:
        print("  OK — toutes les étapes finies")


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA requis pour ce diagnostic")
        sys.exit(1)
    torch.backends.mkldnn.enabled = False
    device = torch.device("cuda")

    # Référence stable
    run_steps(attention_mode="factorized", hybrid_qkv=False, lr=0.01, steps=20, device=device)
    # Cible instable historique
    # Régression stabilité longue durée (50 steps @ lr=0.01, simule effondrement ~ep28)
    for mode in ("contrast", "contrast_sdt"):
        run_steps(attention_mode=mode, hybrid_qkv=True, lr=0.01, steps=50, device=device)


if __name__ == "__main__":
    main()
