import torch.nn as nn

from modules.spike import make_lif

__all__ = ["SDSA"]


class Erode(nn.Module):
    """Érosion temporelle-spatiale pour données DVS."""

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

    def forward(self, x):
        return self.pool(x)  # (T, B, Hh, N, Dh) ou (T, B, C, H, W)


class SDSA(nn.Module):
    """
    MS_SSA_Conv — Spike-Driven Self-Attention (conv 1×1, attention spike-driven).

    Entrée  : (T, B, D, H, W)
    Sortie  : (T, B, D, H, W)

    Attention : kv = sum(k * v), out = q * kv  (pas de softmax)
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        spike_mode="lif",
        lif_backend="auto",
        dvs=False,
        layer=0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads
        self.layer = layer
        if dvs:
            self.pool = Erode()

        def lif(v_threshold=None):
            return make_lif(spike_mode, v_threshold=v_threshold, lif_backend=lif_backend)

        self.shortcut_lif = lif()
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.q_bn = nn.BatchNorm2d(dim)
        self.q_lif = lif()
        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.k_bn = nn.BatchNorm2d(dim)
        self.k_lif = lif()
        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.v_bn = nn.BatchNorm2d(dim)
        self.v_lif = lif()
        self.talking_heads_lif = lif(v_threshold=0.5)
        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape  # ex. (4, B, 256, 8, 8)
        identity = x
        N = H * W
        Dh = C // self.num_heads

        x = self.shortcut_lif(x)                                     # (T, B, D, H, W)

        x_for_qkv = x.flatten(0, 1)                                  # (T*B, D, H, W)

        q = self.q_conv(x_for_qkv)                                   # (T*B, D, H, W)
        q = self.q_bn(q).reshape(T, B, C, H, W).contiguous()
        q = self.q_lif(q)
        q = (
            q.flatten(3).transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, Dh)
            .permute(0, 1, 3, 2, 4).contiguous()
        )                                                            # (T, B, Hh, N, Dh)

        k = self.k_conv(x_for_qkv)
        k = self.k_bn(k).reshape(T, B, C, H, W).contiguous()
        k = self.k_lif(k)
        if self.dvs:
            k = self.pool(k)
        k = (
            k.flatten(3).transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, Dh)
            .permute(0, 1, 3, 2, 4).contiguous()
        )                                                            # (T, B, Hh, N, Dh)

        v = self.v_conv(x_for_qkv)
        v = self.v_bn(v).reshape(T, B, C, H, W).contiguous()
        v = self.v_lif(v)
        if self.dvs:
            v = self.pool(v)
        v = (
            v.flatten(3).transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, Dh)
            .permute(0, 1, 3, 2, 4).contiguous()
        )                                                            # (T, B, Hh, N, Dh)

        kv = k.mul(v)                                                # (T, B, Hh, N, Dh)
        if self.dvs:
            kv = self.pool(kv)
        kv = kv.sum(dim=-2, keepdim=True)                            # (T, B, Hh, 1, Dh)
        kv = self.talking_heads_lif(kv)

        x = q.mul(kv)                                                # (T, B, Hh, N, Dh)
        if self.dvs:
            x = self.pool(x)
        x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()  # (T, B, D, H, W)

        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W).contiguous()
        x = x + identity                                             # (T, B, D, H, W)
        return x
