from typing import Any

import sys
from pathlib import Path

from utils.config import Schema

sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent / "common"))
from config_training import TRAINING_RECIPE_SCHEMA, validate_training_recipe
from config_data import validate_dataset_config

A2OS2A_CONFIG_SCHEMA: Schema = {
    "epochs": (int, "positive"),
    "batch_size": (int, "positive"),
    "learning_rate": (float, "positive"),
    "weight_decay": (float, "non_negative"),
    "emb_dim": (int, "positive"),
    "depth": (int, "positive"),
    "num_heads": (int, "positive"),
    "T": (int, "positive"),
    "num_workers": (int, "non_negative"),
    "lif_backend": (str, None),
    "dataset": (str, None),
    "data_dir": (str, None),
    "save_dir": (str, None),
    "device": (str, None),
    **TRAINING_RECIPE_SCHEMA,
}


def validate_a2os2a_config(config: dict[str, Any]) -> None:
    validate_dataset_config(config)
    if config["emb_dim"] % config["num_heads"] != 0:
        raise ValueError(
            f"emb_dim ({config['emb_dim']}) doit être divisible par num_heads ({config['num_heads']})"
        )
    if config["lif_backend"] not in ("auto", "torch", "cupy"):
        raise ValueError(
            f"lif_backend doit être 'auto', 'torch' ou 'cupy' (reçu: {config['lif_backend']!r})"
        )
    if config["device"] is not None and config["device"] not in ("cuda", "cpu"):
        raise ValueError(f"'device' doit être 'cuda' ou 'cpu' (reçu: {config['device']!r})")
    validate_training_recipe(config)
