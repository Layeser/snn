import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepParametricLIFNode

__all__ = ["SPS"]


def _make_lif(spike_mode: str):
    if spike_mode == "lif":
        return MultiStepLIFNode(tau=2.0, detach_reset=True)
    if spike_mode == "plif":
        return MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True)
    raise NotImplementedError(f"Unsupported spike mode: {spike_mode}")


class SPS(nn.Module):
    """
    MS_SPS — Spiking Patch Splitting.

    Entrée  : (T, B, C_in, H, W)   ex. CIFAR → (4, B, 3, 32, 32)
    Sortie  : (T, B, D, H', W')    ex. pooling_stat="0011" → (4, B, 256, 8, 8)
    """

    def __init__(
        self,
        img_size_h=32,
        img_size_w=32,
        patch_size=4,
        in_channels=3,
        embed_dims=256,
        pooling_stat="0011",
        spike_mode="lif",
    ):
        super().__init__()
        patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.patch_size = patch_size
        self.pooling_stat = pooling_stat

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = _make_lif(spike_mode)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = _make_lif(spike_mode)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = _make_lif(spike_mode)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = _make_lif(spike_mode)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)

    def forward(self, x):
        T, B, _, H, W = x.shape
        ratio = 1

        x = self.proj_conv(x.flatten(0, 1))                          # (T*B, D/8, H, W)
        x = self.proj_bn(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif(x)
        x = x.flatten(0, 1)
        if self.pooling_stat[0] == "1":
            x = self.maxpool(x)
            ratio *= 2

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif1(x)
        x = x.flatten(0, 1)
        if self.pooling_stat[1] == "1":
            x = self.maxpool1(x)
            ratio *= 2

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif2(x)
        x = x.flatten(0, 1)
        if self.pooling_stat[2] == "1":
            x = self.maxpool2(x)
            ratio *= 2

        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        if self.pooling_stat[3] == "1":
            x = self.maxpool3(x)
            ratio *= 2

        x_feat = x
        x = self.proj_lif3(x.reshape(T, B, -1, H // ratio, W // ratio).contiguous())
        x = x.flatten(0, 1)
        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        x = (x + x_feat).reshape(T, B, -1, H // ratio, W // ratio).contiguous()  # (T,B,D,H',W')
        return x
