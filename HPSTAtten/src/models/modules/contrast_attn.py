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

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        vct_num: int = 16,
        layer: int = 0,
        *,
        linear_aggregation: bool = False,
    ):
        super().__init__()
        self.linear_aggregation = linear_aggregation
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
        # Paramètres λ appris (head_dim,) — pas les activations Q/K.
        # exp(s) avec s∈[-10,10] → ~22k : v_hat diverge sur DVS (lr=0.01, hybrid K=ReLU).
        s1 = torch.clamp(torch.sum(q1 * k1, dim=-1).float(), min=-4.0, max=4.0)
        s2 = torch.clamp(torch.sum(q2 * k2, dim=-1).float(), min=-4.0, max=4.0)
        lam = torch.exp(s1) - torch.exp(s2) + init
        return torch.clamp(lam, min=-1.0, max=4.0).type_as(ref)

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
        stage2_lif: nn.Module | None = None,
        kv_token_norm: float = 1.0,
    ) -> torch.Tensor:
        T, B, _, _, head_dim = q.shape
        assert head_dim == self.head_dim

        t = self._pool_queries(q, T, B, H, W)
        t_pos = t + self.e_pos
        t_neg = t + self.e_neg

        if self.linear_aggregation:
            return self._forward_sdt(
                q, t_pos, t_neg, k, v, scaling_factor, stage_lif, stage2_lif or stage_lif,
                kv_token_norm=kv_token_norm,
            )

        # Matmuls en FP32 sous AMP : évite overflow fp16 sur DVS (ctx ~ O(N·d²)).
        k_f = k.float()
        v_f = v.float()
        q_f = q.float()
        t_pos_f = t_pos.float()
        t_neg_f = t_neg.float()

        # ---------- Stage I : contraste global (linéaire factorisé KᵀV) ----------
        # Softmax(t Kᵀ) V  →  t (Kᵀ V)   (associativité)
        kv = torch.matmul(k_f.transpose(-2, -1), v_f) * scaling_factor  # (..., d, d)
        v_hat_pos = torch.matmul(t_pos_f, kv)
        v_hat_neg = torch.matmul(t_neg_f, kv)
        lambda_1 = self._lambda(
            self.lambda_1_q1, self.lambda_1_k1, self.lambda_1_q2, self.lambda_1_k2,
            self.lambda_1_init, q,
        )
        v_hat = (v_hat_pos - lambda_1 * v_hat_neg) * (1.0 - self.lambda_1_init)
        v_hat = v_hat.type_as(q)
        # LIF à la place de RMSNorm (équivalent spike du « talking heads »)
        v_hat = stage_lif(v_hat)

        # ---------- Stage II : lecture différentielle des patches ----------
        # Softmax(Q tᵀ) v_hat  →  Q (tᵀ v_hat)
        v_hat_f = v_hat.float()
        ctx_pos = torch.matmul(t_pos_f.transpose(-2, -1), v_hat_f)
        ctx_neg = torch.matmul(t_neg_f.transpose(-2, -1), v_hat_f)
        lambda_2 = self._lambda(
            self.lambda_2_q1, self.lambda_2_k1, self.lambda_2_q2, self.lambda_2_k2,
            self.lambda_2_init, q,
        )
        ctx = (ctx_pos - lambda_2 * ctx_neg) * (1.0 - self.lambda_2_init)
        ctx = ctx / math.sqrt(head_dim)
        return torch.matmul(q_f, ctx).type_as(q)

    def _forward_sdt(
        self,
        q: torch.Tensor,
        t_pos: torch.Tensor,
        t_neg: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scaling_factor: float,
        kv_lif: nn.Module,
        vhat_lif: nn.Module,
        *,
        kv_token_norm: float = 1.0,
    ) -> torch.Tensor:
        """VCA avec agrégation SDT (Σ K⊙V) — complexité O(N·Dh + n·Dh).

        Deux LIF distincts : kv (n=1, talking heads) et v_hat (n=vct_num).
        SpikingJelly garde un état interne dimensionné à la première forme vue.
        kv_token_norm : 1/N sur DVS (K ReLU hybride) pour stabiliser l'agrégation.
        """
        k_f = k.float()
        v_f = v.float()
        kv = (k_f * v_f).sum(dim=-2, keepdim=True)
        if kv_token_norm != 1.0:
            kv = kv * kv_token_norm
        kv = kv.type_as(k)
        kv = kv_lif(kv)

        v_hat_pos = t_pos * kv
        v_hat_neg = t_neg * kv
        lambda_1 = self._lambda(
            self.lambda_1_q1, self.lambda_1_k1, self.lambda_1_q2, self.lambda_1_k2,
            self.lambda_1_init, q,
        )
        v_hat = (v_hat_pos - lambda_1 * v_hat_neg) * (1.0 - self.lambda_1_init)
        v_hat = vhat_lif(v_hat)

        # Stage II : lecture différentielle Hadamard (comme SDSA)
        ctx_pos = (t_pos * v_hat).sum(dim=-2)
        ctx_neg = (t_neg * v_hat).sum(dim=-2)
        lambda_2 = self._lambda(
            self.lambda_2_q1, self.lambda_2_k1, self.lambda_2_q2, self.lambda_2_k2,
            self.lambda_2_init, q,
        )
        ctx = (ctx_pos - lambda_2 * ctx_neg) * (1.0 - self.lambda_2_init)
        ctx = ctx / math.sqrt(self.head_dim)
        return q * ctx.unsqueeze(-2)
