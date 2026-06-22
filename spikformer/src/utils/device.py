from __future__ import annotations

import torch


def resolve_device(device_cfg: str | None) -> torch.device:
    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            print("Attention: device=cuda demandé mais CUDA indisponible → utilisation du CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    if device_cfg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
