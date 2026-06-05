import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

__all__ = ['SPS']


class SPS(nn.Module):
    """
    Spiking Patch Splitting (SPS).

    Entrée  : (T, B, C_in, H, W)   ex. CIFAR → (4, B, 3, 32, 32)
    Sortie  : (T, B, N, D)          ex. CIFAR → (4, B, 64, 256)
              N = (H/4) * (W/4)     ex. 8×8 = 64 patches
              D = embed_dims

    Notations utilisées dans forward :
      T  = pas de temps
      B  = batch size
      C_in = in_channels (3 pour CIFAR)
      H, W = hauteur / largeur image
      D  = embed_dims (256 par défaut)
    """

    def __init__(self, img_size_h=32, img_size_w=32, patch_size=(4, 4), in_channels=3, embed_dims=256):
        super().__init__()
        self.image_size = (img_size_h, img_size_w)
        self.patch_size = patch_size
        self.C = in_channels
        self.D = embed_dims
        self.H = self.image_size[0] // self.patch_size[0]
        self.W = self.image_size[1] // self.patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

    def forward(self, x):
        T, B, C, H, W = x.shape  # ex. (4, B, 3, 32, 32)

        # --- bloc 1 : projection initiale ---
        x = x.flatten(0, 1)                          # (T*B, C_in, H, W)
        x = self.proj_conv(x)                        # (T*B, D/8, H, W)
        x = self.proj_bn(x)                          # (T*B, D/8, H, W)
        x = x.reshape(T, B, -1, H, W).contiguous()  # (T, B, D/8, H, W)
        x = self.proj_lif(x)                         # (T, B, D/8, H, W)  spikes binaires
        x = x.flatten(0, 1)                          # (T*B, D/8, H, W)

        # --- bloc 2 ---
        x = self.proj_conv1(x)                       # (T*B, D/4, H, W)
        x = self.proj_bn1(x)                         # (T*B, D/4, H, W)
        x = x.reshape(T, B, -1, H, W).contiguous()  # (T, B, D/4, H, W)
        x = self.proj_lif1(x)                        # (T, B, D/4, H, W)
        x = x.flatten(0, 1)                          # (T*B, D/4, H, W)

        # --- bloc 3 + 1er downsample ---
        x = self.proj_conv2(x)                       # (T*B, D/2, H, W)
        x = self.proj_bn2(x)                         # (T*B, D/2, H, W)
        x = x.reshape(T, B, -1, H, W).contiguous()  # (T, B, D/2, H, W)
        x = self.proj_lif2(x)                        # (T, B, D/2, H, W)
        x = x.flatten(0, 1)                          # (T*B, D/2, H, W)
        x = self.maxpool2(x)                         # (T*B, D/2, H/2, W/2)  ex. 16×16

        # --- bloc 4 + 2e downsample ---
        x = self.proj_conv3(x)                       # (T*B, D, H/2, W/2)
        x = self.proj_bn3(x)                         # (T*B, D, H/2, W/2)
        x = x.reshape(T, B, -1, H // 2, W // 2).contiguous()  # (T, B, D, H/2, W/2)
        x = self.proj_lif3(x)                        # (T, B, D, H/2, W/2)
        x = x.flatten(0, 1)                          # (T*B, D, H/2, W/2)
        x = self.maxpool3(x)                         # (T*B, D, H/4, W/4)  ex. 8×8

        # --- RPE (relative position embedding) + skip connection ---
        x_feat = x.reshape(T, B, -1, H // 4, W // 4).contiguous()  # (T, B, D, H/4, W/4)
        x = self.rpe_conv(x)                         # (T*B, D, H/4, W/4)
        x = self.rpe_bn(x)                           # (T*B, D, H/4, W/4)
        x = x.reshape(T, B, -1, H // 4, W // 4).contiguous()  # (T, B, D, H/4, W/4)
        x = self.rpe_lif(x)                          # (T, B, D, H/4, W/4)
        x = x + x_feat                               # (T, B, D, H/4, W/4)

        # --- aplatir la grille spatiale en séquence de patches ---
        x = x.flatten(-2).transpose(-1, -2)          # (T, B, N, D)  N=(H/4)*(W/4)
        return x
