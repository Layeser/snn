#!/usr/bin/env python3
"""
Entraînement STAtten (Spatial-Temporal Attention) sur CIFAR-10.

Hyperparamètres : config/train.yml (validé) + override CLI.
Tracking : MLflow.
Reprise : save_dir/last.pt (--resume auto) ou --fresh pour repartir de zéro.
"""
import argparse
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "train.yml"
MLFLOW_EXPERIMENT = "STAtten-CIFAR10"

sys.path.insert(0, str(ROOT.parent / "common"))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src"))

from loaders.cifar import get_cifar10_loaders
from models import STAttenTransformer
from modules.spike import resolve_lif_backend
from train_cli import add_checkpoint_args
from train_integration import build_criterion, build_scheduler, loader_kwargs_from_config, recipe_hyperparams
from training_runner import run_training
from utils.config import parse_train_args
from utils.config_schema import STATTEN_CONFIG_SCHEMA, validate_stattn_config
from utils.device import resolve_device
from utils.training import train_one_epoch, validate


def build_parser(config: dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train STAtten on CIFAR-10")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--epochs", type=int, default=config["epochs"])
    p.add_argument("--batch-size", type=int, default=config["batch_size"])
    p.add_argument("--lr", type=float, default=config["learning_rate"])
    p.add_argument("--weight-decay", type=float, default=config["weight_decay"])
    p.add_argument("--embed-dim", type=int, default=config["emb_dim"])
    p.add_argument("--depth", type=int, default=config["depth"])
    p.add_argument("--num-heads", type=int, default=config["num_heads"])
    p.add_argument("--T", type=int, default=config["T"])
    p.add_argument("--chunk-size", type=int, default=config["chunk_size"])
    p.add_argument("--num-workers", type=int, default=config["num_workers"])
    p.add_argument("--pooling-stat", type=str, default=config["pooling_stat"])
    p.add_argument("--spike-mode", type=str, default=config["spike_mode"])
    p.add_argument("--lif-backend", type=str, default=config["lif_backend"])
    p.add_argument("--attention-mode", type=str, default=config["attention_mode"])
    p.add_argument("--data-dir", type=str, default=config["data_dir"])
    p.add_argument("--save-dir", type=str, default=config["save_dir"])
    p.add_argument("--device", type=str, default=config["device"])
    add_checkpoint_args(p)
    return p


def main():
    args, config = parse_train_args(
        DEFAULT_CONFIG_PATH,
        STATTEN_CONFIG_SCHEMA,
        build_parser,
        extra_validators=[validate_stattn_config],
    )
    print(f"Config validée: {args.config}")
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)

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
        f"lr={args.lr}, embed_dim={args.embed_dim}, depth={args.depth}, "
        f"T={args.T}, chunk_size={args.chunk_size}, attention_mode={args.attention_mode}"
    )

    train_loader, val_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=True,
        **loader_kwargs_from_config(config),
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = STAttenTransformer(
        img_size=32,
        in_channels=3,
        num_classes=10,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        pooling_stat=args.pooling_stat,
        spike_mode=args.spike_mode,
        lif_backend=args.lif_backend,
        attention_mode=args.attention_mode,
        chunk_size=args.chunk_size,
        T=args.T,
    ).to(device)

    criterion = build_criterion(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(optimizer, config, args.epochs)

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        args=args,
        save_dir=save_dir,
        experiment_name=MLFLOW_EXPERIMENT,
        hyperparams={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "embed_dim": args.embed_dim,
            "depth": args.depth,
            "num_heads": args.num_heads,
            "T": args.T,
            "chunk_size": args.chunk_size,
            "pooling_stat": args.pooling_stat,
            "spike_mode": args.spike_mode,
            "lif_backend": lif_backend,
            "attention_mode": args.attention_mode,
            "num_workers": args.num_workers,
            "device": str(device),
            **recipe_hyperparams(config),
        },
        train_one_epoch=train_one_epoch,
        validate=validate,
        mismatch_keys=("embed_dim", "depth", "num_heads", "T", "batch_size", "chunk_size"),
        run_name_prefix="stattn",
    )


if __name__ == "__main__":
    main()
