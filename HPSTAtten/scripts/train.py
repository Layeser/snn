#!/usr/bin/env python3
"""
Entraînement HP-STAtten (Proposition A) sur CIFAR-10.

Fusion STAtten + A²OS²A + backbone SDT.
Reprise : save_dir/last.pt (--resume auto) ou --fresh pour repartir de zéro.
Référence : Notes/proprosition.md
"""
import argparse
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG_BY_DATASET = {
    "cifar10": ROOT / "config" / "train_cifar10.yml",
    "cifar10-dvs": ROOT / "config" / "train_cifar10-dvs.yml",
}
DEFAULT_CONFIG_PATH = CONFIG_BY_DATASET["cifar10"]
MLFLOW_PROJECT_PREFIX = "HP-STAtten"

sys.path.insert(0, str(ROOT.parent / "common"))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src"))

from datasets import (
    dataset_hyperparams,
    dataset_split_params,
    get_dataset_loaders,
    get_dataset_profile,
    mlflow_experiment_name,
)
from models import HPSTAttenTransformer
from modules.spike import resolve_lif_backend
from train_cli import add_checkpoint_args
from train_integration import (
    apply_dataset_training_overrides,
    configure_cuda_runtime,
    resolve_learning_rate,
    resolve_num_workers,
    resolve_project_path,
    resolve_training_batch_size,
    build_criterion,
    build_scheduler,
    loader_kwargs_from_config,
    recipe_hyperparams,
)
from reproducibility import set_seed
from training_runner import run_training
from utils.config import parse_train_args
from utils.config_schema import HPSTATTEN_CONFIG_SCHEMA, validate_hpstattn_config
from utils.device import resolve_device
from utils.training import train_one_epoch, validate


def build_parser(config: dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train HP-STAtten on CIFAR-10")
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
    p.add_argument("--hybrid-qkv", type=str, default=config["hybrid_qkv"])
    p.add_argument(
        "--attention-mode",
        type=str,
        default=config["attention_mode"],
        choices=["factorized", "sdt"],
    )
    p.add_argument(
        "--membrane-block",
        type=str,
        default=config["membrane_block"],
        choices=["true", "false"],
    )
    p.add_argument(
        "--tet-loss",
        type=str,
        default=config["tet_loss"],
        choices=["true", "false"],
    )
    p.add_argument("--dataset", type=str, default=config["dataset"], choices=["cifar10", "cifar10-dvs"])
    p.add_argument("--data-dir", type=str, default=config["data_dir"])
    p.add_argument("--save-dir", type=str, default=config["save_dir"])
    p.add_argument("--device", type=str, default=config["device"])
    p.add_argument("--seed", type=int, default=42, help="Graine aléatoire (reproductibilité)")
    p.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Fraction stratifiée du train à utiliser (ex: 0.333 pour 1/3)",
    )
    add_checkpoint_args(p)
    return p


def main():
    args, config = parse_train_args(
        DEFAULT_CONFIG_PATH,
        HPSTATTEN_CONFIG_SCHEMA,
        build_parser,
        extra_validators=[validate_hpstattn_config],
    )
    if config["dataset"] != args.dataset:
        print(
            f"ATTENTION: config dataset={config['dataset']!r} ≠ --dataset {args.dataset!r}. "
            f"Utilisez --config {CONFIG_BY_DATASET.get(args.dataset, DEFAULT_CONFIG_PATH)}"
        )
    config = apply_dataset_training_overrides(config, args.dataset)
    args.data_dir = resolve_project_path(args.data_dir, ROOT)
    args.save_dir = resolve_project_path(args.save_dir, ROOT)
    profile = get_dataset_profile(args.dataset)
    batch_size = resolve_training_batch_size(args.batch_size, config, profile)
    learning_rate = resolve_learning_rate(args.lr, batch_size, config, profile)
    hybrid_qkv = args.hybrid_qkv == "true"
    membrane_block = args.membrane_block == "true"
    lif_backend = resolve_lif_backend(args.lif_backend)

    print(f"Config validée: {args.config}")
    print(f"Dataset: {profile.display_name} ({profile.name})")
    device = resolve_device(args.device)
    config["use_amp"] = configure_cuda_runtime(device)
    set_seed(args.seed)
    num_workers = resolve_num_workers(args.num_workers, profile)
    save_dir = Path(args.save_dir)

    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"AMP: {config['use_amp']}")
    print(f"DataLoader workers: {num_workers}")
    print(f"Seed: {args.seed} | train_fraction: {args.train_fraction}")
    print(f"LIF backend: {lif_backend} (config lif_backend={args.lif_backend})")
    if lif_backend == "torch" and str(device) == "cuda":
        print(
            "Astuce: installe cupy pour accélérer les neurones LIF sur GPU "
            "(pip install cupy-cuda12x selon ta version CUDA)"
        )
    print(
        f"Hyperparamètres: epochs={args.epochs}, batch_size={batch_size}, "
        f"lr={learning_rate}, embed_dim={args.embed_dim}, depth={args.depth}, "
        f"T={args.T}, chunk_size={args.chunk_size}, hybrid_qkv={hybrid_qkv}, "
        f"attention_mode={args.attention_mode}, membrane_block={membrane_block}, "
        f"tet_loss={config['tet_loss']}"
    )

    train_loader, val_loader = get_dataset_loaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        download=True,
        project_root=ROOT,
        T=args.T,
        train_fraction=args.train_fraction,
        seed=args.seed,
        **loader_kwargs_from_config(config, args.dataset),
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = HPSTAttenTransformer(
        img_size=profile.img_size,
        in_channels=profile.in_channels,
        num_classes=profile.num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        pooling_stat=args.pooling_stat,
        spike_mode=args.spike_mode,
        lif_backend=args.lif_backend,
        chunk_size=args.chunk_size,
        hybrid_qkv=hybrid_qkv,
        dvs=profile.dvs,
        T=args.T,
        attention_mode=args.attention_mode,
        membrane_block=membrane_block,
    ).to(device)

    criterion = build_criterion(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=args.weight_decay
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
        experiment_name=mlflow_experiment_name(MLFLOW_PROJECT_PREFIX, args.dataset),
        hyperparams={
            **dataset_hyperparams(args.dataset, args.data_dir),
            **dataset_split_params(train_loader, val_loader),
            "epochs": args.epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": args.weight_decay,
            "embed_dim": args.embed_dim,
            "depth": args.depth,
            "num_heads": args.num_heads,
            "T": args.T,
            "chunk_size": args.chunk_size,
            "pooling_stat": args.pooling_stat,
            "spike_mode": args.spike_mode,
            "lif_backend": lif_backend,
            "hybrid_qkv": hybrid_qkv,
            "attention_mode": args.attention_mode,
            "membrane_block": membrane_block,
            "tet_loss": config["tet_loss"],
            "tet_lamb": config["tet_lamb"],
            "tet_means": config["tet_means"],
            "dvs_augment": config.get("dvs_augment", "true"),
            "dvs_random_split": config.get("dvs_random_split", "false"),
            "num_workers": num_workers,
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "use_amp": config["use_amp"],
            "device": str(device),
            **recipe_hyperparams(config),
        },
        train_one_epoch=train_one_epoch,
        validate=validate,
        mismatch_keys=("dataset", "embed_dim", "depth", "num_heads", "T", "batch_size", "chunk_size", "attention_mode", "membrane_block"),
        run_name_prefix="hpstattn",
        project_root=ROOT,
    )


if __name__ == "__main__":
    main()
