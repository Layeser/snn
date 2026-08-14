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
    if torch.cuda.is_available():
        torch.backends.mkldnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    modes = ("factorized", "sdt", "contrast", "contrast_sdt")

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

    print("OK — grille 4×2 forward (factorized/sdt/contrast/contrast_sdt × hybrid true/false)")

    for attention_mode in ("factorized_hgr", "mk_hgr"):
        model = HPSTAttenTransformer(
            img_size=32,
            in_channels=3,
            num_classes=num_classes,
            embed_dim=256,
            depth=2,
            num_heads=8,
            pooling_stat="0011",
            chunk_size=2,
            hybrid_qkv=True,
            T=4,
            attention_mode=attention_mode,
            hgr_lambda=0.1,
            mk_dual_scale=True,
        ).to(device)
        functional.reset_net(model)
        with torch.no_grad():
            logits = model(torch.randn(2, 3, 32, 32, device=device))
        assert torch.isfinite(logits).all(), f"{attention_mode}: non-finite logits"
        print(f"[{attention_mode}] logits={tuple(logits.shape)} OK")

    print("OK — MK-HGR modes forward (factorized_hgr, mk_hgr)")

    complexity_cases = (
        ("baseline", dict(window_size=0, mix_rank=0, num_landmarks=0)),
        ("win4", dict(window_size=4, mix_rank=0, num_landmarks=0)),
        ("rank8", dict(window_size=0, mix_rank=8, num_landmarks=0)),
        ("nystrom16", dict(window_size=0, mix_rank=0, num_landmarks=16)),
        ("win4_rank8", dict(window_size=4, mix_rank=8, num_landmarks=0)),
    )
    for name, kw in complexity_cases:
        model = HPSTAttenTransformer(
            img_size=32,
            in_channels=3,
            num_classes=num_classes,
            embed_dim=256,
            depth=2,
            num_heads=8,
            pooling_stat="0011",
            chunk_size=2,
            hybrid_qkv=True,
            T=4,
            attention_mode="factorized",
            **kw,
        ).to(device)
        functional.reset_net(model)
        with torch.no_grad():
            logits = model(torch.randn(32, 3, 32, 32, device=device))
        assert logits.shape == (32, num_classes), logits.shape
        print(f"[complexity:{name}] batch=32 logits={tuple(logits.shape)} OK")

    print("OK — ablations complexité forward (baseline/win4/rank8/nystrom16/win4_rank8)")

    # DVS-shaped contrast (régression scaling / contrast_sdt kv)
    for mode in ("contrast", "contrast_sdt"):
        model = HPSTAttenTransformer(
            img_size=64,
            in_channels=2,
            num_classes=num_classes,
            embed_dim=256,
            depth=2,
            num_heads=8,
            pooling_stat="0011",
            chunk_size=4,
            hybrid_qkv=True,
            dvs=True,
            T=16,
            attention_mode=mode,
            vct_num=16,
        ).to(device)
        functional.reset_net(model)
        dvs_in = torch.randn(2, 16, 2, 64, 64, device=device)
        with torch.no_grad():
            logits = model(dvs_in)
        assert torch.isfinite(logits).all(), f"DVS {mode}: non-finite logits"
        assert logits.shape == (2, num_classes), logits.shape
        print(f"[DVS {mode}] logits={tuple(logits.shape)} finite OK")

    # Régression NaN DVS contrast + AMP (warmup lr=0.01)
    if device.type == "cuda":
        model = HPSTAttenTransformer(
            img_size=64,
            in_channels=2,
            num_classes=num_classes,
            embed_dim=256,
            depth=2,
            num_heads=8,
            pooling_stat="0011",
            chunk_size=4,
            hybrid_qkv=True,
            dvs=True,
            T=16,
            attention_mode="contrast",
            vct_num=16,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        dvs_in = torch.randn(4, 16, 2, 64, 64, device=device)
        labels = torch.randint(0, num_classes, (4,), device=device)
        model.train()
        for _ in range(3):
            functional.reset_net(model)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=True):
                logits = model(dvs_in)
                loss = torch.nn.functional.cross_entropy(logits, labels)
            assert torch.isfinite(loss), "DVS contrast+AMP: loss non-finite"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        print("[DVS contrast + AMP + lr=0.01] 3 steps finite OK")


if __name__ == "__main__":
    main()
