"""Variantes de mixage factorisé STAtten (low-rank, Nyström, fenêtre locale).

Campagne « complexity ablation » — HP baseline (factorized + hybrid_qkv), hors grille 4×2.
Voir Notes/hpstatten_complexity_ablations.md
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from modules.fhgr_mixing import mix_factorized_hgr


def mix_factorized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling_factor: float,
) -> torch.Tensor:
    """out = Q @ (Kᵀ V) — baseline STAtten."""
    attn = torch.matmul(k.transpose(-2, -1), v) * scaling_factor
    return torch.matmul(q, attn)


def mix_lowrank(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling_factor: float,
    proj_k: torch.Tensor,
    proj_v: torch.Tensor,
) -> torch.Tensor:
    """Mixage rang-r : O(N·d·r) via projections apprises par tête.

    proj_k, proj_v : (num_heads, head_dim, rank)
    q, k, v       : (..., num_heads, N, head_dim)
    """
    *batch_shape, num_heads, n_tokens, head_dim = q.shape
    flat = math.prod(batch_shape) if batch_shape else 1
    q = q.reshape(flat, num_heads, n_tokens, head_dim)
    k = k.reshape(flat, num_heads, n_tokens, head_dim)
    v = v.reshape(flat, num_heads, n_tokens, head_dim)

    kr = torch.einsum("bhnd,hdr->bhnr", k, proj_k)
    vr = torch.einsum("bhnd,hdr->bhnr", v, proj_v)
    a_r = torch.einsum("bhnr,bhns->bhrs", kr, vr) * scaling_factor
    qp = torch.einsum("bhnd,hdr->bhnr", q, proj_k)
    out_r = torch.einsum("bhnr,bhrs->bhns", qp, a_r)
    out = torch.einsum("bhns,hds->bhnd", out_r, proj_v)
    return out.reshape(*batch_shape, num_heads, n_tokens, head_dim)


def mix_nystrom(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling_factor: float,
    landmark_proj: torch.Tensor,
) -> torch.Tensor:
    """Landmarks soft (Nyström) : A ≈ K̃ᵀ Ṽ avec r tokens pondérés.

    landmark_proj : (num_heads, head_dim, num_landmarks)
    """
    *batch_shape, num_heads, n_tokens, head_dim = k.shape
    flat = math.prod(batch_shape) if batch_shape else 1
    q = q.reshape(flat, num_heads, n_tokens, head_dim)
    k = k.reshape(flat, num_heads, n_tokens, head_dim)
    v = v.reshape(flat, num_heads, n_tokens, head_dim)

    scores = torch.einsum("bhnd,hdr->bhnr", k, landmark_proj)
    weights = F.softmax(scores, dim=-2)
    k_m = torch.einsum("bhnr,bhnd->bhrd", weights, k)
    v_m = torch.einsum("bhnr,bhnd->bhrd", weights, v)
    attn = torch.matmul(k_m.transpose(-2, -1), v_m) * scaling_factor
    out = torch.matmul(q, attn)
    return out.reshape(*batch_shape, num_heads, n_tokens, head_dim)


def _maybe_shift(t: torch.Tensor, H: int, W: int, window_size: int, shift: bool) -> torch.Tensor:
    if not shift:
        return t
    shift_size = window_size // 2
    t = t.view(*t.shape[:-2], H, W, t.shape[-1])
    t = torch.roll(t, shifts=(-shift_size, -shift_size), dims=(-3, -2))
    return t.flatten(-3, -2)


def _maybe_unshift(t: torch.Tensor, H: int, W: int, window_size: int, shift: bool) -> torch.Tensor:
    if not shift:
        return t
    shift_size = window_size // 2
    t = t.view(*t.shape[:-2], H, W, t.shape[-1])
    t = torch.roll(t, shifts=(shift_size, shift_size), dims=(-3, -2))
    return t.flatten(-3, -2)


def window_factorized_mix(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    H: int,
    W: int,
    window_size: int,
    scaling_factor: float,
    *,
    shift: bool = False,
    mix_rank: int = 0,
    proj_k: torch.Tensor | None = None,
    proj_v: torch.Tensor | None = None,
    num_landmarks: int = 0,
    landmark_proj: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attention factorisée dans des fenêtres w×w (par pas de temps, toutes têtes).

    q, k, v : (T, B, heads, N, d) avec N = H·W
    """
    ws = window_size
    if H % ws != 0 or W % ws != 0:
        raise ValueError(f"H={H} et W={W} doivent être divisibles par window_size={ws}")

    T, B, heads, N, d = q.shape
    n_h, n_w = H // ws, W // ws
    num_win = n_h * n_w
    m = ws * ws

    def _mix_fn(qw, kw, vw):
        if mix_rank > 0:
            assert proj_k is not None and proj_v is not None
            return mix_lowrank(qw, kw, vw, scaling_factor, proj_k, proj_v)
        if num_landmarks > 0:
            assert landmark_proj is not None
            return mix_nystrom(qw, kw, vw, scaling_factor, landmark_proj)
        return mix_factorized(qw, kw, vw, scaling_factor)

    q = _maybe_shift(q, H, W, ws, shift)
    k = _maybe_shift(k, H, W, ws, shift)
    v = _maybe_shift(v, H, W, ws, shift)

    q = q.view(T, B, heads, n_h, ws, n_w, ws, d)
    k = k.view(T, B, heads, n_h, ws, n_w, ws, d)
    v = v.view(T, B, heads, n_h, ws, n_w, ws, d)

    # (T, B, heads, n_h, n_w, M, d)
    q = q.permute(0, 1, 2, 3, 5, 4, 6, 7).contiguous().view(T, B, heads, num_win, m, d)
    k = k.permute(0, 1, 2, 3, 5, 4, 6, 7).contiguous().view(T, B, heads, num_win, m, d)
    v = v.permute(0, 1, 2, 3, 5, 4, 6, 7).contiguous().view(T, B, heads, num_win, m, d)

    q = q.reshape(T * B * num_win, heads, m, d)
    k = k.reshape(T * B * num_win, heads, m, d)
    v = v.reshape(T * B * num_win, heads, m, d)
    out = _mix_fn(q, k, v)
    out = out.reshape(T, B, heads, num_win, m, d)

    out = out.view(T, B, heads, n_h, n_w, ws, ws, d)
    out = out.permute(0, 1, 2, 3, 5, 4, 6, 7).contiguous().view(T, B, heads, N, d)
    return _maybe_unshift(out, H, W, ws, shift)


def chunked_factorized_mix(
    q_chunks: torch.Tensor,
    k_chunks: torch.Tensor,
    v_chunks: torch.Tensor,
    scaling_factor: float,
    *,
    mix_rank: int = 0,
    proj_k: torch.Tensor | None = None,
    proj_v: torch.Tensor | None = None,
    num_landmarks: int = 0,
    landmark_proj: torch.Tensor | None = None,
    use_hgr: bool = False,
    hgr_lambda: float = 0.1,
    hgr_diag_gate: bool = True,
    hgr_trace_gate: bool = True,
) -> torch.Tensor:
    """Mixage sur tokens cs·N fusionnés (chemin STAtten chunk temporel).

    q_chunks, k_chunks, v_chunks : (num_chunks, B, heads, cs*N, d)
    """
    if use_hgr:
        return mix_factorized_hgr(
            q_chunks,
            k_chunks,
            v_chunks,
            scaling_factor,
            hgr_lambda=hgr_lambda,
            hgr_diag_gate=hgr_diag_gate,
            hgr_trace_gate=hgr_trace_gate,
        )
    if mix_rank > 0:
        assert proj_k is not None and proj_v is not None
        return mix_lowrank(q_chunks, k_chunks, v_chunks, scaling_factor, proj_k, proj_v)
    if num_landmarks > 0:
        assert landmark_proj is not None
        return mix_nystrom(q_chunks, k_chunks, v_chunks, scaling_factor, landmark_proj)
    return mix_factorized(q_chunks, k_chunks, v_chunks, scaling_factor)
