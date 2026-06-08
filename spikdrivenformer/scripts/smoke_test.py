#!/usr/bin/env python3
"""Vérifie le forward complet Spike-Driven Transformer."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src"))

import torch
from spikingjelly.clock_driven import functional
from models import SpikeDrivenTransformer
from loaders.cifar import get_cifar10_loaders


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    model = SpikeDrivenTransformer(
        img_size=32,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=256,
        depth=4,
        num_heads=8,
        pooling_stat="0011",
        T=4,
    ).to(device)

    x = torch.randn(2, 3, 32, 32, device=device)
    t_in = x.unsqueeze(0).repeat(model.T, 1, 1, 1, 1)
    with torch.no_grad():
        feats = model.forward_features(t_in)
    print("features:", feats.shape)  # attendu (T, B, 256)

    train_loader, _ = get_cifar10_loaders(batch_size=4, num_workers=0, download=True)
    images, labels = next(iter(train_loader))
    images = images.to(device)
    functional.reset_net(model)
    with torch.no_grad():
        logits = model(images)

    print("logits:", logits.shape)
    print("labels:", labels.shape)
    assert logits.shape == (images.shape[0], num_classes)
    print("OK — SpikeDrivenTransformer forward (B, num_classes)")


if __name__ == "__main__":
    main()
