import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepParametricLIFNode

__all__ = ["MLP"]


def _make_lif(spike_mode: str):
    if spike_mode == "lif":
        return MultiStepLIFNode(tau=2.0, detach_reset=True)
    if spike_mode == "plif":
        return MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True)
    raise NotImplementedError(f"Unsupported spike mode: {spike_mode}")


class MLP(nn.Module):
    """
    MS_MLP_Conv — MLP spiking (LIF → Conv 1×1 → BN).

    Entrée  : (T, B, D, H, W)
    Sortie  : (T, B, D, H, W)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, spike_mode="lif", layer=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features
        self.c_hidden = hidden_features
        self.c_output = out_features
        self.layer = layer

        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = _make_lif(spike_mode)

        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = _make_lif(spike_mode)

    def forward(self, x):
        T, B, C, H, W = x.shape
        identity = x

        x = self.fc1_lif(x)                                          # (T, B, D, H, W)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        if self.res:
            x = identity + x
            identity = x

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, self.c_output, H, W).contiguous()
        x = x + identity
        return x
