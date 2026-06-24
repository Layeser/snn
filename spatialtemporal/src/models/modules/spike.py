from __future__ import annotations

import sys
from pathlib import Path

from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepParametricLIFNode

_COMMON = Path(__file__).resolve().parents[3].parent / "common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))
from spike_backend import resolve_lif_backend

__all__ = ["resolve_lif_backend", "make_lif"]


def make_lif(
    spike_mode: str,
    v_threshold: float | None = None,
    lif_backend: str = "auto",
):
    backend = resolve_lif_backend(lif_backend)
    kwargs = {"detach_reset": True, "backend": backend}
    if v_threshold is not None:
        kwargs["v_threshold"] = v_threshold

    if spike_mode == "lif":
        return MultiStepLIFNode(tau=2.0, **kwargs)
    if spike_mode == "plif":
        return MultiStepParametricLIFNode(init_tau=2.0, **kwargs)
    raise NotImplementedError(f"Unsupported spike mode: {spike_mode}")
