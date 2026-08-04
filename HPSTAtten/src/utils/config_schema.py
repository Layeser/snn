from typing import Any

import sys
from pathlib import Path

from utils.config import Schema

sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent / "common"))
from config_training import TRAINING_RECIPE_SCHEMA, validate_training_recipe
from config_data import validate_dataset_config

HPSTATTEN_CONFIG_SCHEMA: Schema = {
    "epochs": (int, "positive"),
    "batch_size": (int, "positive"),
    "learning_rate": (float, "positive"),
    "weight_decay": (float, "non_negative"),
    "emb_dim": (int, "positive"),
    "depth": (int, "positive"),
    "num_heads": (int, "positive"),
    "T": (int, "positive"),
    "chunk_size": (int, "positive"),
    "num_workers": (int, "non_negative"),
    "pooling_stat": (str, None),
    "spike_mode": (str, None),
    "lif_backend": (str, None),
    "hybrid_qkv": (str, None),
    "attention_mode": (str, None),
    "membrane_block": (str, None),
    "tet_loss": (str, None),
    "tet_lamb": (float, "non_negative"),
    "tet_means": (float, "non_negative"),
    "dvs_augment": (str, None),
    "dvs_random_split": (str, None),
    "dataset": (str, None),
    "data_dir": (str, None),
    "save_dir": (str, None),
    "device": (str, None),
    **TRAINING_RECIPE_SCHEMA,
    "scheduler_step_size": (int, "positive"),
    "scheduler_gamma": (float, "positive"),
    "scheduler_patience": (int, "positive"),
    "scheduler_factor": (float, "positive"),
    "scheduler_threshold": (float, "non_negative"),
}


def validate_hpstattn_config(config: dict[str, Any]) -> None:
    validate_dataset_config(config)
    if config["emb_dim"] % config["num_heads"] != 0:
        raise ValueError(
            f"emb_dim ({config['emb_dim']}) doit être divisible par num_heads ({config['num_heads']})"
        )
    if config["T"] % config["chunk_size"] != 0:
        raise ValueError(
            f"T ({config['T']}) doit être divisible par chunk_size ({config['chunk_size']})"
        )
    if len(config["pooling_stat"]) != 4 or any(c not in "01" for c in config["pooling_stat"]):
        raise ValueError(
            f"pooling_stat doit être une chaîne de 4 caractères '0' ou '1' "
            f"(reçu: {config['pooling_stat']!r})"
        )
    if config["spike_mode"] not in ("lif", "plif"):
        raise ValueError(f"spike_mode doit être 'lif' ou 'plif' (reçu: {config['spike_mode']!r})")
    if config["lif_backend"] not in ("auto", "torch", "cupy"):
        raise ValueError(
            f"lif_backend doit être 'auto', 'torch' ou 'cupy' (reçu: {config['lif_backend']!r})"
        )
    if config["hybrid_qkv"] not in ("true", "false"):
        raise ValueError(
            f"hybrid_qkv doit être 'true' ou 'false' (reçu: {config['hybrid_qkv']!r})"
        )
    if config["attention_mode"] not in ("factorized", "sdt", "contrast"):
        raise ValueError(
            f"attention_mode doit être 'factorized', 'sdt' ou 'contrast' "
            f"(reçu: {config['attention_mode']!r})"
        )
    if config["membrane_block"] not in ("true", "false"):
        raise ValueError(
            f"membrane_block doit être 'true' ou 'false' (reçu: {config['membrane_block']!r})"
        )
    if config["tet_loss"] not in ("true", "false"):
        raise ValueError(f"tet_loss doit être 'true' ou 'false' (reçu: {config['tet_loss']!r})")
    if config["tet_lamb"] > 1.0:
        raise ValueError(f"tet_lamb doit être <= 1.0 (reçu: {config['tet_lamb']})")
    if config["dvs_augment"] not in ("true", "false"):
        raise ValueError(
            f"dvs_augment doit être 'true' ou 'false' (reçu: {config['dvs_augment']!r})"
        )
    if config["dvs_random_split"] not in ("true", "false"):
        raise ValueError(
            f"dvs_random_split doit être 'true' ou 'false' (reçu: {config['dvs_random_split']!r})"
        )
    if config["device"] is not None and config["device"] not in ("cuda", "cpu"):
        raise ValueError(f"'device' doit être 'cuda' ou 'cpu' (reçu: {config['device']!r})")
    validate_training_recipe(config)
