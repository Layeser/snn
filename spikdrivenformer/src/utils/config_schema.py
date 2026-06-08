from typing import Any

from utils.config import Schema

SPIKDRIVEN_CONFIG_SCHEMA: Schema = {
    "epochs": (int, "positive"),
    "batch_size": (int, "positive"),
    "learning_rate": (float, "positive"),
    "weight_decay": (float, "non_negative"),
    "emb_dim": (int, "positive"),
    "depth": (int, "positive"),
    "num_heads": (int, "positive"),
    "T": (int, "positive"),
    "num_workers": (int, "non_negative"),
    "pooling_stat": (str, None),
    "spike_mode": (str, None),
    "data_dir": (str, None),
    "save_dir": (str, None),
    "device": (str, None),
}


def validate_spikdriven_config(config: dict[str, Any]) -> None:
    if config["emb_dim"] % config["num_heads"] != 0:
        raise ValueError(
            f"emb_dim ({config['emb_dim']}) doit être divisible par num_heads ({config['num_heads']})"
        )
    if len(config["pooling_stat"]) != 4 or any(c not in "01" for c in config["pooling_stat"]):
        raise ValueError(
            f"pooling_stat doit être une chaîne de 4 caractères '0' ou '1' "
            f"(reçu: {config['pooling_stat']!r})"
        )
    if config["spike_mode"] not in ("lif", "plif"):
        raise ValueError(f"spike_mode doit être 'lif' ou 'plif' (reçu: {config['spike_mode']!r})")
    if config["device"] is not None and config["device"] not in ("cuda", "cpu"):
        raise ValueError(f"'device' doit être 'cuda' ou 'cpu' (reçu: {config['device']!r})")
