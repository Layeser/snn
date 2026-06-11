import torch.nn as nn

from modules.spike import make_lif

__all__ = ["SPS"]


class SPS(nn.Module):
    """
    MS_SPS — Spiking Patch Splitting (SDT / STAtten).

    Entrée  : (T, B, C_in, H, W)
    Sortie  : (T, B, D, H', W')
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
        lif_backend="auto",
    ):
        super().__init__()
        patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.pooling_stat = pooling_stat
        lif = lambda: make_lif(spike_mode, lif_backend=lif_backend)

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, 3, 1, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = lif()
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, 3, 1, 1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = lif()
        self.maxpool1 = nn.MaxPool2d(3, 2, 1)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, 3, 1, 1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = lif()
        self.maxpool2 = nn.MaxPool2d(3, 2, 1)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, 3, 1, 1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = lif()
        self.maxpool3 = nn.MaxPool2d(3, 2, 1)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, 3, 1, 1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)

    def forward(self, x):
        T, B, _, H, W = x.shape
        ratio = 1

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif(x).flatten(0, 1)
        if self.pooling_stat[0] == "1":
            x = self.maxpool(x)
            ratio *= 2

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif1(x).flatten(0, 1)
        if self.pooling_stat[1] == "1":
            x = self.maxpool1(x)
            ratio *= 2

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif2(x).flatten(0, 1)
        if self.pooling_stat[2] == "1":
            x = self.maxpool2(x)
            ratio *= 2

        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        if self.pooling_stat[3] == "1":
            x = self.maxpool3(x)
            ratio *= 2

        x_feat = x
        x = self.proj_lif3(x.reshape(T, B, -1, H // ratio, W // ratio).contiguous()).flatten(0, 1)
        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        return (x + x_feat).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
