"""Helpers pour brancher la recette d'entraînement dans scripts/train.py."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from datasets import effective_mixup, get_dataset_profile, loader_kwargs_for_dataset
from training_recipe import build_cosine_scheduler


def resolve_project_path(path: str | None, project_root: Any) -> str | None:
    """Rend un chemin portable inter-machines.

    - chemin absolu (ex: passé par le Makefile) -> conservé tel quel ;
    - chemin relatif (ex: config ``../data``) -> résolu contre la racine projet.
    Évite les chemins en dur type ``/home/<user>/...`` à changer par machine.
    """
    from pathlib import Path

    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((Path(project_root) / p).resolve())


def loader_kwargs_from_config(config: dict[str, Any], dataset: str | None = None) -> dict[str, Any]:
    profile = get_dataset_profile(dataset or config["dataset"])
    return loader_kwargs_for_dataset(config, profile)


def apply_dataset_training_overrides(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    profile = get_dataset_profile(dataset)
    updated = dict(config)
    updated["mixup"] = effective_mixup(config, profile)
    if profile.temporal_input:
        updated["augment_train"] = "false"
        updated["rand_augment"] = "false"
        updated["random_erasing"] = 0.0
        epochs = int(config.get("epochs", 50))
        updated["warmup_epochs"] = min(
            int(config["warmup_epochs"]),
            max(2, epochs // 10),
        )
    return updated


def resolve_learning_rate(
    learning_rate: float,
    batch_size: int,
    config: dict[str, Any],
    profile,
) -> float:
    """
    Ajuste le LR pour DVS : batch plus petit → gradients plus bruités.
    Règle : scale linéairement avec batch_size / batch_size de référence.
    """
    if not profile.temporal_input:
        return learning_rate
    if "learning_rate_dvs" in config:
        return float(config["learning_rate_dvs"])
    ref_batch = int(config.get("batch_size", 64))
    scaled = learning_rate * (batch_size / ref_batch)
    if abs(scaled - learning_rate) > 1e-12:
        print(
            f"LR ajusté {learning_rate:g} → {scaled:g} pour {profile.display_name} "
            f"(batch {batch_size}, ref {ref_batch})"
        )
    return scaled


def resolve_training_batch_size(
    batch_size: int,
    config: dict[str, Any],
    profile,
) -> int:
    """Réduit le batch pour CIFAR-10-DVS (128×128) afin d'éviter l'OOM GPU."""
    if not profile.temporal_input:
        return batch_size
    cap = int(config.get("batch_size_dvs", 16))
    if batch_size > cap:
        print(
            f"Batch size réduit {batch_size} → {cap} pour {profile.display_name} "
            f"({profile.img_size}×{profile.img_size}, mémoire GPU)"
        )
        return cap
    return batch_size


def resolve_num_workers(num_workers: int, profile) -> int:
    """DVS charge des .npz depuis le disque : workers par défaut si 0."""
    if num_workers > 0:
        return num_workers
    if profile.temporal_input:
        return 4
    return num_workers


def configure_cuda_runtime(device) -> bool:
    """Optimisations CUDA ; retourne True si AMP est recommandé."""
    if getattr(device, "type", None) != "cuda":
        return False
    import torch

    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    return True


def build_criterion(config: dict[str, Any]) -> nn.CrossEntropyLoss:
    return nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])


def build_scheduler(
    optimizer,
    config: dict[str, Any],
    epochs: int,
):
    if config["scheduler"] != "cosine":
        return None
    return build_cosine_scheduler(
        optimizer,
        epochs=epochs,
        warmup_epochs=config["warmup_epochs"],
        min_lr=config["min_lr"],
    )


def recipe_hyperparams(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "augment_train": config["augment_train"],
        "rand_augment": config["rand_augment"],
        "random_erasing": config["random_erasing"],
        "mixup": config["mixup"],
        "label_smoothing": config["label_smoothing"],
        "scheduler": config["scheduler"],
        "warmup_epochs": config["warmup_epochs"],
        "min_lr": config["min_lr"],
    }
