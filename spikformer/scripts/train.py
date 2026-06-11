#!/usr/bin/env python3
"""
Entraînement Spikformer sur CIFAR-10.

Hyperparamètres : config/train.yml (validé) + override CLI.
Tracking : MLflow (métriques, hyperparamètres, artefacts).
"""
import argparse
import sys
from pathlib import Path
from typing import Any

import mlflow
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "train.yml"
MLFLOW_EXPERIMENT = "Spikformer-CIFAR10"

sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src"))

from loaders.cifar import get_cifar10_loaders
from models import Spikformer
from modules.spike import resolve_lif_backend
from utils.config import parse_train_args
from utils.config_schema import SPIKFORMER_CONFIG_SCHEMA, validate_spikformer_config
from utils.device import resolve_device
from utils.mlflow_tracking import (
    log_artifacts,
    log_epoch_metrics,
    log_final_metrics,
    log_hyperparameters,
    setup_experiment,
)
from utils.training import train_one_epoch, validate


def build_parser(config: dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Spikformer on CIFAR-10")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--epochs", type=int, default=config["epochs"])
    p.add_argument("--batch-size", type=int, default=config["batch_size"])
    p.add_argument("--lr", type=float, default=config["learning_rate"])
    p.add_argument("--weight-decay", type=float, default=config["weight_decay"])
    p.add_argument("--embed-dim", type=int, default=config["emb_dim"])
    p.add_argument("--depth", type=int, default=config["depth"])
    p.add_argument("--num-heads", type=int, default=config["num_heads"])
    p.add_argument("--T", type=int, default=config["T"])
    p.add_argument("--num-workers", type=int, default=config["num_workers"])
    p.add_argument("--lif-backend", type=str, default=config["lif_backend"])
    p.add_argument("--data-dir", type=str, default=config["data_dir"])
    p.add_argument("--save-dir", type=str, default=config["save_dir"])
    p.add_argument("--device", type=str, default=config["device"])
    return p


def main():
    args, config = parse_train_args(
        DEFAULT_CONFIG_PATH,
        SPIKFORMER_CONFIG_SCHEMA,
        build_parser,
        extra_validators=[validate_spikformer_config],
    )
    print(f"Config validée: {args.config}")
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    lif_backend = resolve_lif_backend(args.lif_backend)
    print(f"Device: {device}")
    print(f"LIF backend: {lif_backend} (config lif_backend={args.lif_backend})")
    if lif_backend == "torch" and str(device) == "cuda":
        print(
            "Astuce: installe cupy pour accélérer les neurones LIF sur GPU "
            "(pip install cupy-cuda12x selon ta version CUDA)"
        )
    print(
        f"Hyperparamètres: epochs={args.epochs}, batch_size={args.batch_size}, "
        f"lr={args.lr}, embed_dim={args.embed_dim}, depth={args.depth}, T={args.T}"
    )

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
        lif_backend=args.lif_backend,
        T=args.T,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    setup_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run():
        log_hyperparameters(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "embed_dim": args.embed_dim,
                "depth": args.depth,
                "num_heads": args.num_heads,
                "T": args.T,
                "num_workers": args.num_workers,
                "lif_backend": lif_backend,
                "device": str(device),
            }
        )

        best_val_acc = 0.0
        best_epoch = 0
        ckpt = save_dir / "best.pt"

        for epoch in range(1, args.epochs + 1):
            print(f"\n--- Epoch {epoch}/{args.epochs} ---")
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            log_epoch_metrics(epoch, train_loss, train_acc, val_loss, val_acc)

            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.2f}% | "
                f"val loss {val_loss:.4f} acc {val_acc:.2f}%"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": val_acc,
                        "config": config,
                        "args": vars(args),
                    },
                    ckpt,
                )
                print(f"  → meilleur modèle sauvegardé ({val_acc:.2f}%) → {ckpt}")

        log_final_metrics(best_val_acc, best_epoch)
        log_artifacts(args.config, ckpt)

        print(f"Entraînement terminé. Meilleure val accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
        print(f"MLflow experiment: {MLFLOW_EXPERIMENT}")


if __name__ == "__main__":
    main()
