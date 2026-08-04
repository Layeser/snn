#!/usr/bin/env python3
"""Vérifie le forward complet HP-STAtten (grille 3×2)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src"))

import torch
from spikingjelly.clock_driven import functional
from models import HPSTAttenTransformer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    modes = ("factorized", "sdt", "contrast")

    for attention_mode in modes:
        for hybrid in (True, False):
            model = HPSTAttenTransformer(
                img_size=32,
                in_channels=3,
                num_classes=num_classes,
                embed_dim=256,
                depth=2,
                num_heads=8,
                pooling_stat="0011",
                chunk_size=2,
                hybrid_qkv=hybrid,
                T=4,
                attention_mode=attention_mode,
                vct_num=16,
            ).to(device)
            label = f"mode={attention_mode}, hybrid_qkv={hybrid}"

            t_in = torch.randn(4, 2, 3, 32, 32, device=device)
            functional.reset_net(model)
            with torch.no_grad():
                feats = model.forward_features(t_in)
                logits = model(torch.randn(2, 3, 32, 32, device=device))

            assert feats.shape == (4, 2, 256), feats.shape
            assert logits.shape == (2, num_classes), logits.shape
            print(f"[{label}] features={tuple(feats.shape)} logits={tuple(logits.shape)} OK")

    print("OK — grille 3×2 forward (factorized/sdt/contrast × hybrid true/false)")


if __name__ == "__main__":
    main()
