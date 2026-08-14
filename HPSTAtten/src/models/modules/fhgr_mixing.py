"""Fast-HGR linear attention mixing — O(N·d²) via Q̂ @ (K̂^T V).

Réf. : Notes/n_articles/mk_cavit_hpstatten_integration.md
"""
from __future__ import annotations

import torch


def _center_tokens(x: torch.Tensor) -> torch.Tensor:
    """Centre sur la dimension token N (avant-dernière pour ..., heads, N, d)."""
    return x - x.mean(dim=-2, keepdim=True)


def _normalize_tokens(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def fhgr_trace_gate(
    q: torch.Tensor,
    k: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Gate 1 + λ·tr(cov(Q) cov(K)), moyenné sur les têtes.

    q, k : (..., num_heads, N, head_dim)
    Retour : (..., 1, 1, 1) broadcastable sur out.
    """
    if lam <= 0.0:
        return torch.ones((), device=q.device, dtype=q.dtype)

    n_tokens = q.shape[-2]
    denom = max(n_tokens - 1, 1)
    q_c = _center_tokens(q)
    k_c = _center_tokens(k)
    cov_q = torch.matmul(q_c.transpose(-2, -1), q_c) / denom
    cov_k = torch.matmul(k_c.transpose(-2, -1), k_c) / denom
    prod = torch.matmul(cov_q, cov_k)
    tr = prod.diagonal(dim1=-2, dim2=-1).sum(-1)
    g = 1.0 + lam * tr.mean(dim=-1)
    return g


def mix_factorized_hgr(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling_factor: float,
    *,
    hgr_lambda: float = 0.1,
    hgr_diag_gate: bool = True,
    hgr_trace_gate: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """out = gate · Q̂ @ (K̂^T V) with optional diagonal cosine and trace gates.

    q, k, v : (..., num_heads, N, head_dim)
    """
    q_c = _center_tokens(q.float())
    k_c = _center_tokens(k.float())
    q_hat = _normalize_tokens(q_c, eps)
    k_hat = _normalize_tokens(k_c, eps)
    v_f = v.float()

    attn = torch.matmul(k_hat.transpose(-2, -1), v_f) * scaling_factor
    out = torch.matmul(q_hat, attn)

    if hgr_diag_gate:
        w_diag = (q_hat * k_hat).sum(dim=-1, keepdim=True)
        out = out * w_diag

    if hgr_trace_gate and hgr_lambda > 0.0:
        g = fhgr_trace_gate(q_c, k_c, hgr_lambda)
        while g.dim() < out.dim():
            g = g.unsqueeze(-1)
        out = out * g

    return out.to(dtype=q.dtype)
