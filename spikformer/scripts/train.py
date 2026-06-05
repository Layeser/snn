#!/usr/bin/env python3
"""
Entraînement Spikformer sur CIFAR-10.

Train : 50 000 images (split officiel train=True)
Val   : 10 000 images (split officiel train=False, utilisé comme validation)
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src"))

from loaders.cifar import get_cifar10_loaders
from models import Spikformer


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item() * 100.0


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        functional.reset_net(model)
        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_acc = accuracy(logits, labels)
        total_loss += loss.item()
        total_acc += batch_acc
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.2f}%")

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Val  ", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        functional.reset_net(model)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_acc = accuracy(logits, labels)
        total_loss += loss.item()
        total_acc += batch_acc
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.2f}%")

    return total_loss / n_batches, total_acc / n_batches


def parse_args():
    p = argparse.ArgumentParser(description="Train Spikformer on CIFAR-10")
    p.add_argument("--epochs", type=int, default=10, help="nombre d'époques")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=6e-2)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--T", type=int, default=4, help="pas de temps SNN")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--save-dir", type=str, default=str(ROOT / "checkpoints"))
    p.add_argument("--device", type=str, default=None, help="cuda ou cpu (auto si omis)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    train_loader, val_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=True,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = Spikformer(
        img_size=32,
        in_channels=3,
        num_classes=10,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        T=args.T,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.2f}% | "
            f"val loss {val_loss:.4f} acc {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = save_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                ckpt,
            )
            print(f"  → meilleur modèle sauvegardé ({val_acc:.2f}%) → {ckpt}")

    print(f"Entraînement terminé. Meilleure val accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
