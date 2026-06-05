import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

__all__ = ['MLP']


class MLP(nn.Module):
    """
    Spiking MLP (2 couches linéaires + LIF).

    Entrée  : (T, B, N, D)
    Sortie  : (T, B, N, D)   (D = out_features = in_features par défaut)

    Notations :
      T  = pas de temps
      B  = batch size
      N  = nombre de patches
      D  = in_features / out_features (embed_dim)
      H  = hidden_features = D * mlp_ratio  (ex. 256×4 = 1024)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.c_hidden = hidden_features
        self.c_output = out_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

    def forward(self, x):
        T, B, N, D = x.shape  # ex. (4, B, 64, 256)
        H = self.c_hidden     # ex. 1024 si mlp_ratio=4

        # --- couche 1 ---
        x = x.flatten(0, 1)                          # (T*B, N, D)
        x = self.fc1(x)                              # (T*B, N, H)
        x = x.transpose(-1, -2)                    # (T*B, H, N)  pour BatchNorm1d sur H
        x = self.fc1_bn(x)                           # (T*B, H, N)
        x = x.transpose(-1, -2)                    # (T*B, N, H)
        x = x.reshape(T, B, N, H).contiguous()     # (T, B, N, H)
        x = self.fc1_lif(x)                          # (T, B, N, H)  spikes

        # --- couche 2 ---
        x = x.flatten(0, 1)                          # (T*B, N, H)
        x = self.fc2(x)                              # (T*B, N, D)
        x = x.transpose(-1, -2)                    # (T*B, D, N)
        x = self.fc2_bn(x)                           # (T*B, D, N)
        x = x.transpose(-1, -2)                    # (T*B, N, D)
        x = x.reshape(T, B, N, self.c_output).contiguous()  # (T, B, N, D)
        x = self.fc2_lif(x)                          # (T, B, N, D)  spikes
        return x
