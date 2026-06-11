import torch
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepParametricLIFNode

__all__ = ["resolve_lif_backend", "make_lif"]


def resolve_lif_backend(backend: str = "auto") -> str:
    if backend in ("torch", "cupy"):
        return backend
    if backend != "auto":
        raise ValueError(f"lif_backend invalide: {backend!r} (attendu: auto, torch, cupy)")

    if torch.cuda.is_available():
        try:
            import cupy  # noqa: F401

            return "cupy"
        except ImportError:
            pass
    return "torch"


def make_lif(
    spike_mode: str = "lif",
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
