import math

import torch
import torch.nn as nn

__all__ = ["MultiStepTernaryLIFNode"]

# ATan surrogate (SpikingJelly clock_driven/surrogate.py, alpha=2.0).
_SURROGATE_ALPHA = 2.0


class _SurrogateHeaviside(torch.autograd.Function):
    """Forward: H(x) hard step. Backward: ATan surrogate gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return (x >= 0).to(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_x = None
        if ctx.needs_input_grad[0]:
            x = ctx.saved_tensors[0]
            alpha = ctx.alpha
            grad_x = (
                alpha
                / 2
                / (1 + (math.pi / 2 * alpha * x).pow(2))
                * grad_output
            )
        return grad_x, None


def _surrogate_heaviside(x: torch.Tensor, alpha: float = _SURROGATE_ALPHA) -> torch.Tensor:
    return _SurrogateHeaviside.apply(x, alpha)


class MultiStepTernaryLIFNode(nn.Module):
    """
    Neurone LIF ternaire — sorties dans {-1, 0, 1}.

    Entrée / sortie : (T, B, ...)
    Utilisé pour V dans A2OS2A (papier eq. 25–27, Guo et al. 2024).

    Seuils ±v_threshold : forward dur (spikes exacts), backward via surrogate ATan.
    """

    def __init__(self, tau=2.0, v_threshold=1.0, detach_reset=True):
        super().__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.detach_reset = detach_reset
        self.v = None

    def reset(self):
        self.v = None

    def forward(self, x):
        T = x.shape[0]
        if self.v is None:
            self.v = torch.zeros_like(x[0])

        outputs = []
        for t in range(T):
            self.v = self.v + (x[t] - self.v) / self.tau
            pos = _surrogate_heaviside(self.v - self.v_threshold)
            neg = _surrogate_heaviside(-self.v - self.v_threshold)
            spike = pos - neg
            reset = pos + neg
            if self.detach_reset:
                self.v = self.v.detach() * (1.0 - reset)
            else:
                self.v = self.v * (1.0 - reset)
            outputs.append(spike)

        return torch.stack(outputs, dim=0)
