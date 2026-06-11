import torch
import torch.nn as nn

from modules.spike import make_lif
from modules.ternary_lif import MultiStepTernaryLIFNode

__all__ = ["A2OS2A"]


class A2OS2A(nn.Module):
    """
    Accurate Addition-Only Spiking Self-Attention (A²OS²A).

    Papier : https://arxiv.org/abs/2503.00226
      Q → binaire LIF     {0, 1}
      K → ReLU            ℝ⁺  (pleine précision)
      V → ternaire LIF    {-1, 0, 1}
      out = SN(Q @ K^T @ V)   sans softmax ni scaling

    Entrée  : (T, B, N, D)  spikes S
    Sortie  : (T, B, N, D)  spikes après SN
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, lif_backend="auto"):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads

        lif = lambda: make_lif(lif_backend=lif_backend)

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = lif()                                                   # binaire

        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_act = nn.ReLU(inplace=True)                                   # float ℝ⁺

        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepTernaryLIFNode(tau=2.0, v_threshold=1.0)     # ternaire (custom)

        self.out_lif = lif()                                                 # SN final

    def _bn_linear(self, x, linear, bn):
        T, B, N, D = x.shape
        x = linear(x.flatten(0, 1))                                        # (T*B, N, D)
        x = bn(x.transpose(-1, -2)).transpose(-1, -2)                    # (T*B, N, D)
        return x.reshape(T, B, N, D).contiguous()

    def forward(self, x):
        T, B, N, D = x.shape
        Dh = D // self.num_heads

        q = self._bn_linear(x, self.q_linear, self.q_bn)
        q = self.q_lif(q)                                                  # (T,B,N,D) binaire
        q = q.reshape(T, B, N, self.num_heads, Dh).permute(0, 1, 3, 2, 4)  # (T,B,Hh,N,Dh)

        k = self._bn_linear(x, self.k_linear, self.k_bn)
        k = self.k_act(k)                                                  # (T,B,N,D) float
        k = k.reshape(T, B, N, self.num_heads, Dh).permute(0, 1, 3, 2, 4)

        v = self._bn_linear(x, self.v_linear, self.v_bn)
        v = self.v_lif(v)                                                  # (T,B,N,D) ternaire
        v = v.reshape(T, B, N, self.num_heads, Dh).permute(0, 1, 3, 2, 4)

        attn = q @ k.transpose(-2, -1)                                     # (T,B,Hh,N,N) pas de scale
        out = attn @ v                                                     # (T,B,Hh,N,Dh)
        out = out.transpose(2, 3).reshape(T, B, N, D).contiguous()
        out = self.out_lif(out)                                            # (T,B,N,D)
        return out
