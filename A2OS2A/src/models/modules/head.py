import torch.nn as nn

__all__ = ["ClassificationHead"]


class ClassificationHead(nn.Module):
    """
    CH(GAP(S_L)) — papier eq. 18.

    Entrée  : (T, B, N, D)
    Sortie  : (B, num_classes)
    """

    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, s):
        x = s.mean(2)          # GAP sur patches N → (T, B, D)
        x = self.fc(x)         # (T, B, num_classes)
        return x.mean(0)       # mean temporel → (B, num_classes)
