import torch.nn as nn

__all__ = ['ClassificationHead']


class ClassificationHead(nn.Module):
    """
    Tête de classification linéaire.

    Reçoit les features après agrégation spatiale (mean patches) et temporelle (mean T).
    Pas de neurone spiking ici : on classifie sur la représentation float agrégée,
    comme dans le repo officiel Spikformer.

    Entrée  : (B, D)              ex. (B, 256)
    Sortie  : (B, num_classes)    ex. (B, 10) pour CIFAR-10
    """

    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)  # (D) → (num_classes)

    def forward(self, x):
        x = self.fc(x)  # (B, D) → (B, num_classes)
        return x
