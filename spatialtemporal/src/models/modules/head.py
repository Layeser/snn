import torch.nn as nn

from modules.spike import make_lif

__all__ = ["ClassificationHead"]


class ClassificationHead(nn.Module):
    """
    Tête de classification STAtten.

    Entrée  : (T, B, D)              ex. (4, B, 256)
    Sortie  : (B, num_classes)       ex. (B, 10)

    Pipeline : head_lif → Linear → mean temporel
    """

    def __init__(self, embed_dim, num_classes, spike_mode="lif", lif_backend="auto"):
        super().__init__()
        self.head_lif = make_lif(spike_mode, lif_backend=lif_backend)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.head_lif(x)         # (T, B, D)
        x = self.fc(x)               # (T, B, num_classes)
        x = x.mean(0)                # (B, num_classes)
        return x
