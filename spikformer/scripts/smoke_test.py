#!/usr/bin/env python3
"""Vérifie le forward complet : SPS → blocks → head."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src"))

import torch
from spikingjelly.clock_driven import functional
from models import Spikformer
from loaders.cifar import get_cifar10_loaders


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    model = Spikformer(
        img_size=32,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=256,
        depth=4,
        num_heads=8,
        T=4,
    ).to(device)

    train_loader, _ = get_cifar10_loaders(batch_size=4, num_workers=0, download=True)
    images, labels = next(iter(train_loader))
    images = images.to(device)

    functional.reset_net(model)
    with torch.no_grad():
        logits = model(images)

    print("logits:", logits.shape)   # attendu (B, 10)
    print("labels:", labels.shape)   # attendu (B,)
    assert logits.shape == (images.shape[0], num_classes)
    print("OK — head produit bien (B, num_classes)")


if __name__ == "__main__":
    main()
