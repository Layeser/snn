"""Validation des hyperparamètres de recette d'entraînement CIFAR."""

from typing import Any

TRAINING_RECIPE_SCHEMA: dict[str, tuple[type, str | None]] = {
    "augment_train": (str, None),
    "rand_augment": (str, None),
    "random_erasing": (float, "non_negative"),
    "mixup": (float, "non_negative"),
    "label_smoothing": (float, "non_negative"),
    "scheduler": (str, None),
    "warmup_epochs": (int, "non_negative"),
    "min_lr": (float, "non_negative"),
}


def validate_training_recipe(config: dict[str, Any]) -> None:
    for key in ("augment_train", "rand_augment"):
        if config[key] not in ("true", "false"):
            raise ValueError(f"{key} doit être 'true' ou 'false' (reçu: {config[key]!r})")
    if config["scheduler"] not in ("none", "cosine"):
        raise ValueError(f"scheduler doit être 'none' ou 'cosine' (reçu: {config['scheduler']!r})")
    if config["random_erasing"] > 1.0:
        raise ValueError(f"random_erasing doit être <= 1.0 (reçu: {config['random_erasing']})")
    if config["label_smoothing"] >= 1.0:
        raise ValueError(f"label_smoothing doit être < 1.0 (reçu: {config['label_smoothing']})")
