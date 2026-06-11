import torch
import torch.nn as nn

__all__ = ["MultiStepTernaryLIFNode"]


class MultiStepTernaryLIFNode(nn.Module):
    """
    Neurone LIF ternaire — sorties dans {-1, 0, 1}.
    Utilisé pour V dans HP-STAtten (A²OS²A, Guo et al. CVPR 2025).
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
            pos = (self.v >= self.v_threshold).float()
            neg = (self.v <= -self.v_threshold).float()
            spike = pos - neg
            reset = pos + neg
            if self.detach_reset:
                self.v = self.v.detach() * (1.0 - reset)
            else:
                self.v = self.v * (1.0 - reset)
            outputs.append(spike)

        return torch.stack(outputs, dim=0)
