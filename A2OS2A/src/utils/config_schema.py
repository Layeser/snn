from typing import Any

from utils.config import Schema

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
    "data_dir": (str, None),
    "save_dir": (str, None),
    "device": (str, None),
}


def validate_a2os2a_config(config: dict[str, Any]) -> None:
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
