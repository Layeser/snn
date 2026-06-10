import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepParametricLIFNode

__all__ = ["ClassificationHead"]


class ClassificationHead(nn.Module):
    """
    Tête de classification STAtten.

    Entrée  : (T, B, D)              ex. (4, B, 256)
    Sortie  : (B, num_classes)       ex. (B, 10)

    Pipeline : head_lif → Linear → mean temporel
    """

    def __init__(self, embed_dim, num_classes, spike_mode="lif"):
        super().__init__()
        if spike_mode == "lif":
            self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        elif spike_mode == "plif":
            self.head_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True)
        else:
            raise NotImplementedError(f"Unsupported spike mode: {spike_mode}")
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.head_lif(x)         # (T, B, D)
        x = self.fc(x)               # (T, B, num_classes)
        x = x.mean(0)                # (B, num_classes)
        return x
