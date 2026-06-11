import torch.nn as nn

from modules.spike import make_lif

__all__ = ["MLP"]


class MLP(nn.Module):
    """
    MS_MLP_Conv — MLP spiking avec conv 1×1.

    Entrée  : (T, B, D, H, W)
    Sortie  : (T, B, D, H, W)

    Ordre officiel : LIF → Conv → BN  (LIF avant la conv)
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        spike_mode="lif",
        lif_backend="auto",
        layer=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features
        self.c_hidden = hidden_features
        self.c_output = out_features
        self.layer = layer

        lif = lambda: make_lif(spike_mode, lif_backend=lif_backend)

        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = lif()
        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = lif()

    def forward(self, x):
        T, B, C, H, W = x.shape  # ex. (4, B, 256, 8, 8)
        identity = x

        x = self.fc1_lif(x)                                          # (T, B, D, H, W)  LIF d'abord
        x = self.fc1_conv(x.flatten(0, 1))                           # (T*B, H_hidden, H, W)
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        if self.res:
            x = identity + x
            identity = x

        x = self.fc2_lif(x)                                          # (T, B, H_hidden, H, W)
        x = self.fc2_conv(x.flatten(0, 1))                           # (T*B, D, H, W)
        x = self.fc2_bn(x).reshape(T, B, self.c_output, H, W).contiguous()

        x = x + identity                                             # (T, B, D, H, W)
        return x
