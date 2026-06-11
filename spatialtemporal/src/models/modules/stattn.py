import torch
import torch.nn as nn

from modules.spike import make_lif

__all__ = ["STAtten"]


class DvsPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

    def forward(self, x):
        return self.pool(x)


class STAtten(nn.Module):
    """
    MS_SSA_Conv avec Spatial-Temporal Attention (STAtten) ou SDT en fallback.

    Entrée  : (T, B, D, H, W)
    Sortie  : (T, B, D, H, W)

    STAtten : attention par chunks temporels (chunk_size) + matmul K^T V
    SDT     : q * sum(k * v)  (Spike-Driven Transformer)

    Référence : https://github.com/Intelligent-Computing-Lab-Panda/STAtten
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        spike_mode="lif",
        lif_backend="auto",
        dvs=False,
        layer=0,
        attention_mode="STAtten",
        chunk_size=2,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        assert attention_mode in ("STAtten", "SDT"), f"attention_mode invalide: {attention_mode}"

        self.dim = dim
        self.num_heads = num_heads
        self.dvs = dvs
        self.layer = layer
        self.attention_mode = attention_mode
        self.chunk_size = chunk_size

        if dvs:
            self.pool = DvsPooling()

        def lif(v_threshold=None):
            return make_lif(spike_mode, v_threshold=v_threshold, lif_backend=lif_backend)

        self.shortcut_lif = lif()
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        self.q_lif = lif()
        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)
        self.k_lif = lif()
        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        self.v_lif = lif()
        self.attn_lif = lif(v_threshold=0.5)
        self.talking_heads_lif = lif(v_threshold=0.5)

        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_bn = nn.BatchNorm2d(dim)

    def _qkv(self, x, T, B, C, H, W):
        N = H * W
        head_dim = C // self.num_heads
        x_for_qkv = x.flatten(0, 1)                                  # (T*B, D, H, W)

        q = self.q_conv(x_for_qkv)
        q = self.q_bn(q).reshape(T, B, C, H, W).contiguous()
        q = self.q_lif(q)
        if self.dvs:
            q = self.pool(q)
        q = (
            q.flatten(3).transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, head_dim)
            .permute(0, 1, 3, 2, 4).contiguous()
        )                                                            # (T, B, Hh, N, Dh)

        k = self.k_conv(x_for_qkv)
        k = self.k_bn(k).reshape(T, B, C, H, W).contiguous()
        k = self.k_lif(k)
        if self.dvs:
            k = self.pool(k)
        k = (
            k.flatten(3).transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, head_dim)
            .permute(0, 1, 3, 2, 4).contiguous()
        )

        v = self.v_conv(x_for_qkv)
        v = self.v_bn(v).reshape(T, B, C, H, W).contiguous()
        v = self.v_lif(v)
        if self.dvs:
            v = self.pool(v)
        v = (
            v.flatten(3).transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, head_dim)
            .permute(0, 1, 3, 2, 4).contiguous()
        )
        return q, k, v, N, head_dim

    def _stattn_attention(self, q, k, v, T, B, C, H, W, N, head_dim):
        if self.dvs:
            scaling_factor = 1.0 / (H * H * self.chunk_size)
        else:
            scaling_factor = 1.0 / H

        num_chunks = T // self.chunk_size
        # (num_chunks, B, Hh, chunk_size*N, Dh)
        q_chunks = q.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)
        k_chunks = k.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)
        v_chunks = v.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)

        q_chunks = q_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)
        k_chunks = k_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)
        v_chunks = v_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)

        attn = torch.matmul(k_chunks.transpose(-2, -1), v_chunks) * scaling_factor  # (nc,B,Hh,Dh,Dh)
        out = torch.matmul(q_chunks, attn)                                            # (nc,B,Hh,chunk*N,Dh)

        out = out.reshape(num_chunks, B, self.num_heads, self.chunk_size, N, head_dim).permute(0, 3, 1, 2, 4, 5)
        output = out.reshape(T, B, self.num_heads, N, head_dim)                       # (T, B, Hh, N, Dh)

        x = output.transpose(3, 4).reshape(T, B, C, N).contiguous()                # (T, B, D, N)
        x = self.attn_lif(x).reshape(T, B, C, H, W).contiguous()                    # (T, B, D, H, W)
        return x

    def _sdt_attention(self, q, k, v, T, B, C, H, W):
        kv = k.mul(v)                                                                 # (T, B, Hh, N, Dh)
        if self.dvs:
            kv = self.pool(kv)
        kv = kv.sum(dim=-2, keepdim=True)                                             # (T, B, Hh, 1, Dh)
        kv = self.talking_heads_lif(kv)
        x = q.mul(kv)                                                                 # (T, B, Hh, N, Dh)
        if self.dvs:
            x = self.pool(x)
        x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()
        return x

    def forward(self, x):
        T, B, C, H, W = x.shape
        identity = x

        x = self.shortcut_lif(x)                                                      # (T, B, D, H, W)
        x_pool = self.pool(x) if self.dvs else None

        q, k, v, N, head_dim = self._qkv(x, T, B, C, H, W)

        if self.attention_mode == "STAtten":
            x = self._stattn_attention(q, k, v, T, B, C, H, W, N, head_dim)
            if self.dvs:
                x = x.mul(x_pool) + x_pool
        else:
            x = self._sdt_attention(q, k, v, T, B, C, H, W)

        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W).contiguous()
        x = x + identity                                                              # (T, B, D, H, W)
        return x
