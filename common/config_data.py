"""Validation et métadonnées des jeux de données d'entraînement."""

from __future__ import annotations

from typing import Any

SUPPORTED_DATASETS = ("cifar10", "cifar10-dvs")

DATASET_CONFIG_SCHEMA: dict[str, tuple[type, str | None]] = {
    "dataset": (str, None),
    "data_dir": (str, None),
}


def validate_dataset_config(config: dict[str, Any]) -> None:
    dataset = config["dataset"]
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"dataset doit être l'un de {SUPPORTED_DATASETS} (reçu: {dataset!r})"
        )
