import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

__all__ = ["SPS"]


class SPS(nn.Module):
    """
    Spiking Patch Splitting (Spikformer / papier A2OS2A).

    Entrée  : (T, B, C_in, H, W)   ex. (4, B, 3, 32, 32)
    Sortie  : (T, B, N, D)          membrane U0 après RPE  ex. (4, B, 64, 256)
    """

    def __init__(self, img_size_h=32, img_size_w=32, patch_size=(4, 4), in_channels=3, embed_dims=256):
        super().__init__()
        self.image_size = (img_size_h, img_size_w)
        self.patch_size = patch_size
        self.H = img_size_h // patch_size[0]
        self.W = img_size_w // patch_size[1]

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, 3, 1, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, 3, 1, 1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, 3, 1, 1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.maxpool2 = nn.MaxPool2d(3, 2, 1)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, 3, 1, 1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.maxpool3 = nn.MaxPool2d(3, 2, 1)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, 3, 1, 1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x).flatten(0, 1)

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif1(x).flatten(0, 1)

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif2(x).flatten(0, 1)
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj_lif3(x).flatten(0, 1)
        x = self.maxpool3(x)

        x_feat = x.reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.rpe_lif(x)
        x = x + x_feat

        return x.flatten(-2).transpose(-1, -2)  # (T, B, N, D) = U0
