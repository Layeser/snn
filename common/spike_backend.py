"""Résolution du backend LIF SpikingJelly (torch / cupy)."""

from __future__ import annotations

import torch


def spikingjelly_cupy_available() -> bool:
    """True seulement si SpikingJelly a chargé CuPy et ses kernels CUDA."""
    from spikingjelly.clock_driven import neuron as sj_neuron

    return sj_neuron.cupy is not None


def resolve_lif_backend(backend: str = "auto") -> str:
    if backend == "torch":
        return "torch"
    if backend not in ("auto", "cupy"):
        raise ValueError(f"lif_backend invalide: {backend!r} (attendu: auto, torch, cupy)")

    if torch.cuda.is_available() and spikingjelly_cupy_available():
        return "cupy"

    if backend == "cupy":
        print(
            "Attention: lif_backend=cupy demandé mais indisponible dans SpikingJelly "
            "(vérifier cupy + tensorboard) → utilisation de torch"
        )
    return "torch"
