import torch
import torch.nn as nn

from modules.spike import make_lif

__all__ = ['SSA']


class SSA(nn.Module):
    """
    Spiking Self-Attention (SSA).

    Entrée  : (T, B, N, D)
    Sortie  : (T, B, N, D)

    Notations :
      T  = pas de temps
      B  = batch size
      N  = nombre de patches (64 pour CIFAR)
      D  = dimension d'embedding (embed_dim)
      Hh = num_heads
      Dh = D // Hh  (dimension par tête, ex. 256/8=32)
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        lif_backend="auto",
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125  # 1 / sqrt(Dh)  avec Dh=32 → 1/8 mais le papier utilise 0.125

        lif = lambda v_th=None: make_lif(v_threshold=v_th, lif_backend=lif_backend)

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = lif()

        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = lif()

        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = lif()

        self.attn_lif = lif(v_th=0.5)

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = lif()

    def forward(self, x):
        T, B, N, D = x.shape  # ex. (4, B, 64, 256)
        Dh = D // self.num_heads

        x_for_qkv = x.flatten(0, 1)                  # (T*B, N, D)

        # --- Query ---
        q = self.q_linear(x_for_qkv)                 # (T*B, N, D)
        q = q.transpose(-1, -2)                      # (T*B, D, N)  pour BatchNorm1d sur D
        q = self.q_bn(q)                             # (T*B, D, N)
        q = q.transpose(-1, -2)                      # (T*B, N, D)
        q = q.reshape(T, B, N, D).contiguous()     # (T, B, N, D)
        q = self.q_lif(q)                            # (T, B, N, D)  spikes
        q = q.reshape(T, B, N, self.num_heads, Dh)   # (T, B, N, Hh, Dh)
        q = q.permute(0, 1, 3, 2, 4).contiguous()  # (T, B, Hh, N, Dh)

        # --- Key ---
        k = self.k_linear(x_for_qkv)                 # (T*B, N, D)
        k = k.transpose(-1, -2)                      # (T*B, D, N)
        k = self.k_bn(k)                             # (T*B, D, N)
        k = k.transpose(-1, -2)                      # (T*B, N, D)
        k = k.reshape(T, B, N, D).contiguous()     # (T, B, N, D)
        k = self.k_lif(k)                            # (T, B, N, D)  spikes
        k = k.reshape(T, B, N, self.num_heads, Dh)   # (T, B, N, Hh, Dh)
        k = k.permute(0, 1, 3, 2, 4).contiguous()  # (T, B, Hh, N, Dh)

        # --- Value ---
        v = self.v_linear(x_for_qkv)                 # (T*B, N, D)
        v = v.transpose(-1, -2)                      # (T*B, D, N)
        v = self.v_bn(v)                             # (T*B, D, N)
        v = v.transpose(-1, -2)                      # (T*B, N, D)
        v = v.reshape(T, B, N, D).contiguous()     # (T, B, N, D)
        v = self.v_lif(v)                            # (T, B, N, D)  spikes
        v = v.reshape(T, B, N, self.num_heads, Dh)   # (T, B, N, Hh, Dh)
        v = v.permute(0, 1, 3, 2, 4).contiguous()  # (T, B, Hh, N, Dh)

        # --- Attention spike (pas de softmax) ---
        attn = q @ k.transpose(-2, -1)               # (T, B, Hh, N, N)  produit spike-spike
        attn = attn * self.scale                     # (T, B, Hh, N, N)
        x = attn @ v                                 # (T, B, Hh, N, Dh)
        x = x.transpose(2, 3)                        # (T, B, N, Hh, Dh)
        x = x.reshape(T, B, N, D).contiguous()     # (T, B, N, D)
        x = self.attn_lif(x)                         # (T, B, N, D)  spikes

        # --- Projection finale ---
        x = x.flatten(0, 1)                          # (T*B, N, D)
        x = self.proj_linear(x)                      # (T*B, N, D)
        x = x.transpose(-1, -2)                    # (T*B, D, N)
        x = self.proj_bn(x)                          # (T*B, D, N)
        x = x.transpose(-1, -2)                    # (T*B, N, D)
        x = x.reshape(T, B, N, D).contiguous()     # (T, B, N, D)
        x = self.proj_lif(x)                         # (T, B, N, D)
        return x
