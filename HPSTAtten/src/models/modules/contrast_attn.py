"""Visual-Contrast Attention adapté SNN (inspiré de VCA / LinearDiff).

Référence : Pu et al., Linear Differential Vision Transformer (NeurIPS 2025)
https://github.com/LeapLabTHU/LinearDiff

Adaptation HP-STAtten :
  - AvgPool spatial sur Q → n tokens contraste (n ≪ N)
  - dual embeddings e+ / e- (flux positif / négatif)
  - Stage I/II différentiels **sans Softmax** (associativité linéaire)
  - LIF à la place de RMSNorm (spike-friendly)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["lambda_init_fn", "ContrastTokenMixer"]


def lambda_init_fn(depth: int) -> float:
    """Initialisation λ du Differential / VCA (LinearDiff)."""
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class ContrastTokenMixer(nn.Module):
    """Mixage Q/K/V par contraste visuel linéaire (sans Softmax).

    Entrées q, k, v : (T, B, heads, N, head_dim) avec N = H·W.
    Sortie          : (T, B, heads, N, head_dim)
    """

    def __init__(self, num_heads: int, head_dim: int, vct_num: int = 16, layer: int = 0):
        super().__init__()
        if vct_num < 1:
            raise ValueError(f"vct_num doit être >= 1 (reçu: {vct_num})")
        pool_size = max(1, int(math.sqrt(vct_num)))
        if pool_size * pool_size != vct_num:
            raise ValueError(
                f"vct_num doit être un carré parfait (reçu: {vct_num}, "
                f"sqrt≈{math.sqrt(vct_num):.3f})"
            )

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.vct_num = vct_num
        self.pool_size = pool_size
        self.contrast_pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

        # e+, e- : embeddings positionnels des tokens contraste (par tête)
        self.e_pos = nn.Parameter(torch.randn(1, 1, num_heads, vct_num, head_dim) * 0.02)
        self.e_neg = nn.Parameter(torch.randn(1, 1, num_heads, vct_num, head_dim) * 0.02)

        # λ^(1), λ^(2) — même paramétrisation que LinearDiff VisualContrastAttention
        init = lambda_init_fn(layer)
        self.lambda_1_init = init
        self.lambda_2_init = init
        self.lambda_1_q1 = nn.Parameter(torch.zeros(head_dim).normal_(mean=0, std=0.1))
        self.lambda_1_k1 = nn.Parameter(torch.zeros(head_dim).normal_(mean=0, std=0.1))
        self.lambda_1_q2 = nn.Parameter(torch.zeros(head_dim).normal_(mean=0, std=0.1))
        self.lambda_1_k2 = nn.Parameter(torch.zeros(head_dim).normal_(mean=0, std=0.1))
        self.lambda_2_q1 = nn.Parameter(torch.zeros(head_dim).normal_(mean=0, std=0.1))
        self.lambda_2_k1 = nn.Parameter(torch.zeros(head_dim).normal_(mean=0, std=0.1))
        self.lambda_2_q2 = nn.Parameter(torch.zeros(head_dim).normal_(mean=0, std=0.1))
        self.lambda_2_k2 = nn.Parameter(torch.zeros(head_dim).normal_(mean=0, std=0.1))

    def _lambda(self, q1, k1, q2, k2, init: float, ref: torch.Tensor) -> torch.Tensor:
        term1 = torch.exp(torch.sum(q1 * k1, dim=-1).float()).type_as(ref)
        term2 = torch.exp(torch.sum(q2 * k2, dim=-1).float()).type_as(ref)
        return term1 - term2 + init

    def _pool_queries(self, q: torch.Tensor, T: int, B: int, H: int, W: int) -> torch.Tensor:
        """AvgPool spatial sur Q → tokens contraste (T, B, heads, n, d)."""
        head_dim = self.head_dim
        # (T, B, heads, N, d) -> (T*B*heads, d, H, W)
        q_map = (
            q.reshape(T * B, self.num_heads, H, W, head_dim)
            .permute(0, 1, 4, 2, 3)
            .reshape(T * B * self.num_heads, head_dim, H, W)
        )
        t_map = self.contrast_pool(q_map)  # (..., d, ph, pw)
        # -> (T, B, heads, n, d)
        return (
            t_map.reshape(T, B, self.num_heads, head_dim, self.vct_num)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        H: int,
        W: int,
        scaling_factor: float,
        stage_lif: nn.Module,
    ) -> torch.Tensor:
        T, B, _, _, head_dim = q.shape
        assert head_dim == self.head_dim

        t = self._pool_queries(q, T, B, H, W)
        t_pos = t + self.e_pos
        t_neg = t + self.e_neg

        # ---------- Stage I : contraste global (linéaire, sans Softmax) ----------
        # Softmax(t Kᵀ) V  →  t (Kᵀ V)   (associativité)
        kv = torch.matmul(k.transpose(-2, -1), v) * scaling_factor  # (..., d, d)
        v_hat_pos = torch.matmul(t_pos, kv)  # (..., n, d)
        v_hat_neg = torch.matmul(t_neg, kv)
        lambda_1 = self._lambda(
            self.lambda_1_q1, self.lambda_1_k1, self.lambda_1_q2, self.lambda_1_k2,
            self.lambda_1_init, q,
        )
        v_hat = (v_hat_pos - lambda_1 * v_hat_neg) * (1.0 - self.lambda_1_init)
        # LIF à la place de RMSNorm (équivalent spike du « talking heads »)
        v_hat = stage_lif(v_hat)

        # ---------- Stage II : lecture différentielle des patches ----------
        # Softmax(Q tᵀ) v_hat  →  Q (tᵀ v_hat)
        ctx_pos = torch.matmul(t_pos.transpose(-2, -1), v_hat)  # (..., d, d)
        ctx_neg = torch.matmul(t_neg.transpose(-2, -1), v_hat)
        lambda_2 = self._lambda(
            self.lambda_2_q1, self.lambda_2_k1, self.lambda_2_q2, self.lambda_2_k2,
            self.lambda_2_init, q,
        )
        ctx = (ctx_pos - lambda_2 * ctx_neg) * (1.0 - self.lambda_2_init)
        return torch.matmul(q, ctx)  # (T, B, heads, N, d)
