import torch.nn as nn

from modules.spike import make_lif

__all__ = ["MLP"]


class MLP(nn.Module):
    """
    MLP spiking (Linear + BN + LIF), sortie float avant SN du bloc.

    Entrée  : (T, B, N, D)  spikes
    Sortie  : (T, B, N, D)  float (contribution membrane)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, lif_backend="auto"):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.c_hidden = hidden_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = make_lif(lif_backend=lif_backend)

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        T, B, N, D = x.shape
        x = self.fc1(x.flatten(0, 1))
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden)
        x = self.fc1_lif(x)

        x = self.fc2(x.flatten(0, 1))
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, D)
        return x
