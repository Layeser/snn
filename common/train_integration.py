"""Helpers pour brancher la recette d'entraînement dans scripts/train.py."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from datasets import effective_mixup, get_dataset_profile, loader_kwargs_for_dataset
from training_recipe import build_cosine_scheduler


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
    return updated


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
