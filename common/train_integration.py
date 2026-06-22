"""Helpers pour brancher la recette d'entraînement dans scripts/train.py."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from training_recipe import build_cosine_scheduler


def loader_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "augment_train": config["augment_train"] == "true",
        "rand_augment": config["rand_augment"] == "true",
        "random_erasing": config["random_erasing"],
    }


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
