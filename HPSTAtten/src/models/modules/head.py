import torch.nn as nn

from modules.spike import make_lif

__all__ = ["ClassificationHead"]


class ClassificationHead(nn.Module):
    """
    head_lif → Linear → mean(T)

    Entrée  : (T, B, D)
    Sortie  : (B, num_classes)
    """

    def __init__(self, embed_dim, num_classes, spike_mode="lif", lif_backend="auto"):
        super().__init__()
        self.head_lif = make_lif(spike_mode, lif_backend=lif_backend)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (T, B, D)
        x = self.head_lif(x)
        # x: (T, B, D)
        x = self.fc(x)
        # x: (T, B, num_classes)
        # temporal average -> (B, num_classes)
        return x.mean(0)
