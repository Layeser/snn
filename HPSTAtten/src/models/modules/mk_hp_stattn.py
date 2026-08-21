"""MK-HPSTAtten — pipeline Figure 1 (multi-scale + Fast-HGR + fusion).

Réf. : Notes/n_articles/mk_cavit_hpstatten_integration.md
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.a2os2a_scaling import resolve_factorized_scaling
from modules.fhgr_mixing import mix_factorized_hgr
from modules.spike import make_lif
from modules.ternary_lif import MultiStepTernaryLIFNode

__all__ = ["MKHPSTAtten"]


class DvsPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

    def forward(self, x):
        return self.pool(x)


class _ScaleBranch(nn.Module):
    """Conv multi-échelle sur la carte de features (T,B,C,H,W)."""

    def __init__(self, dim: int, kernel: int, stride: int, lif_factory):
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv2d(dim, dim, kernel, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.lif = lif_factory()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, b, c, h, w = x.shape
        y = self.conv(x.flatten(0, 1))
        y = self.bn(y).reshape(t, b, c, y.shape[-2], y.shape[-1]).contiguous()
        return self.lif(y)


def _interpolate_feat(feat: torch.Tensor, h_tgt: int, w_tgt: int) -> torch.Tensor:
    t, b, c, h, w = feat.shape
    if h == h_tgt and w == w_tgt:
        return feat
    y = F.interpolate(
        feat.flatten(0, 1),
        size=(h_tgt, w_tgt),
        mode="bilinear",
        align_corners=False,
    )
    return y.reshape(t, b, c, h_tgt, w_tgt).contiguous()


class MKHPSTAtten(nn.Module):
    """
    Multi-Kernel Fast-HGR attention (MK-CAViT adapté SNN).

    Pipeline : 2–3 branches conv → FHGR-attn → fusion local–mid → intégration globale
    → gates par tête → out_lif + proj + résidu.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        spike_mode="lif",
        lif_backend="auto",
        chunk_size=2,
        hybrid_qkv=True,
        dvs=False,
        layer=0,
        hgr_lambda=0.1,
        hgr_diag_gate=True,
        hgr_trace_gate=True,
        mk_dual_scale=True,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.hybrid_qkv = hybrid_qkv
        self.dvs = dvs
        self.layer = layer
        self.hgr_lambda = hgr_lambda
        self.hgr_diag_gate = hgr_diag_gate
        self.hgr_trace_gate = hgr_trace_gate
        self.mk_dual_scale = mk_dual_scale

        if dvs:
            self.pool = DvsPooling()

        def lif(v_threshold=None):
            return make_lif(spike_mode, v_threshold=v_threshold, lif_backend=lif_backend)

        self.shortcut_lif = lif()

        self.branch_small = _ScaleBranch(dim, kernel=3, stride=1, lif_factory=lif)
        self.branch_mid = _ScaleBranch(dim, kernel=7, stride=2, lif_factory=lif)
        if not mk_dual_scale:
            self.branch_large = _ScaleBranch(dim, kernel=15, stride=1, lif_factory=lif)

        self.q_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        self.q_lif = lif()

        self.k_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)
        self.k_relu = nn.ReLU(inplace=True)
        self.k_lif = lif()

        self.v_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        self.v_ternary = MultiStepTernaryLIFNode(tau=2.0, v_threshold=1.0)
        self.v_lif = lif()

        self.out_lif = lif()
        self.proj_conv = nn.Conv2d(dim, dim, 1)
        self.proj_bn = nn.BatchNorm2d(dim)

        # Fusion Eq. 12 — gates α, β (vecteurs appris, sigmoid)
        self.fusion_alpha = nn.Parameter(torch.zeros(dim))
        self.fusion_beta = nn.Parameter(torch.zeros(dim))
        # Eq. 13 — γ global context
        self.global_gamma = nn.Parameter(torch.zeros(()))
        # Adaptive multi-head
        self.head_gates = nn.Parameter(torch.zeros(num_heads))

    def _encode_qkv(self, x, t, b, c, h, w):
        n = h * w
        head_dim = c // self.num_heads
        x_flat = x.flatten(0, 1)

        q = self.q_conv(x_flat)
        q = self.q_bn(q).reshape(t, b, c, h, w).contiguous()
        q = self.q_lif(q)
        if self.dvs:
            q = self.pool(q)
        q = (
            q.flatten(3).transpose(-1, -2)
            .reshape(t, b, n, self.num_heads, head_dim)
            .permute(0, 1, 3, 2, 4).contiguous()
        )

        k = self.k_conv(x_flat)
        k = self.k_bn(k).reshape(t, b, c, h, w).contiguous()
        if self.hybrid_qkv:
            k = self.k_relu(k)
        else:
            k = self.k_lif(k)
        if self.dvs:
            k = self.pool(k)
        k = (
            k.flatten(3).transpose(-1, -2)
            .reshape(t, b, n, self.num_heads, head_dim)
            .permute(0, 1, 3, 2, 4).contiguous()
        )

        v = self.v_conv(x_flat)
        v = self.v_bn(v).reshape(t, b, c, h, w).contiguous()
        if self.hybrid_qkv:
            v = self.v_ternary(v)
        else:
            v = self.v_lif(v)
        if self.dvs:
            v = self.pool(v)
        v = (
            v.flatten(3).transpose(-1, -2)
            .reshape(t, b, n, self.num_heads, head_dim)
            .permute(0, 1, 3, 2, 4).contiguous()
        )
        return q, k, v, n, head_dim

    def _scaling(self, h: int) -> float:
        return resolve_factorized_scaling(
            hybrid_qkv=self.hybrid_qkv,
            dvs=self.dvs,
            spatial_h=h,
            chunk_size=self.chunk_size,
        )

    def _fhgr_mix(self, q, k, v, h: int, w: int) -> torch.Tensor:
        """FHGR mixing avec chunk temporel STAtten. Retour (T,B,heads,N,d)."""
        t = q.shape[0]
        scaling = self._scaling(h)
        num_chunks = t // self.chunk_size
        n = h * w

        q_chunks = q.view(num_chunks, self.chunk_size, *q.shape[1:]).permute(0, 2, 3, 1, 4, 5)
        k_chunks = k.view(num_chunks, self.chunk_size, *k.shape[1:]).permute(0, 2, 3, 1, 4, 5)
        v_chunks = v.view(num_chunks, self.chunk_size, *v.shape[1:]).permute(0, 2, 3, 1, 4, 5)

        q_chunks = q_chunks.reshape(num_chunks, q.shape[1], self.num_heads, self.chunk_size * n, q.shape[-1])
        k_chunks = k_chunks.reshape(num_chunks, k.shape[1], self.num_heads, self.chunk_size * n, k.shape[-1])
        v_chunks = v_chunks.reshape(num_chunks, v.shape[1], self.num_heads, self.chunk_size * n, v.shape[-1])

        out = mix_factorized_hgr(
            q_chunks,
            k_chunks,
            v_chunks,
            scaling,
            hgr_lambda=self.hgr_lambda,
            hgr_diag_gate=self.hgr_diag_gate,
            hgr_trace_gate=self.hgr_trace_gate,
        )

        out = out.reshape(num_chunks, q.shape[1], self.num_heads, self.chunk_size, n, q.shape[-1])
        out = out.permute(0, 3, 1, 2, 4, 5).reshape(t, q.shape[1], self.num_heads, n, q.shape[-1])
        return out

    def _tokens_to_map(self, out, t, b, c, h, w, n):
        return out.transpose(3, 4).reshape(t, b, c, n).reshape(t, b, c, h, w).contiguous()

    def _cross_fhgr(self, q_src, k_src, v_src, h_tgt, w_tgt):
        """FHGR-Attn avec Q,K de (h_tgt,w_tgt) et V éventuellement autre résolution."""
        t, b, c = q_src.shape[0], q_src.shape[1], self.dim
        v_al = _interpolate_feat(v_src, h_tgt, w_tgt)
        q, k, _, n, _ = self._encode_qkv(q_src, t, b, c, h_tgt, w_tgt)
        _, _, v, _, _ = self._encode_qkv(v_al, t, b, c, h_tgt, w_tgt)
        out = self._fhgr_mix(q, k, v, h_tgt, w_tgt)
        return self._tokens_to_map(out, t, b, c, h_tgt, w_tgt, n)

    def forward(self, x, return_contribution: bool = False):
        t, b, c, h, w = x.shape
        identity = x

        x = self.shortcut_lif(x)
        x_pool = self.pool(x) if self.dvs else None

        xs = self.branch_small(x)
        xm = self.branch_mid(x)
        hs, ws = xs.shape[-2], xs.shape[-1]
        hm, wm = xm.shape[-2], xm.shape[-1]

        # Fusion local–mid (Eq. 12), aligné sur small
        xm_up = _interpolate_feat(xm, hs, ws)
        cross_sm = self._cross_fhgr(xs, xs, xm_up, hs, ws)
        cross_ms = self._cross_fhgr(xm_up, xm_up, xs, hs, ws)

        alpha = torch.sigmoid(self.fusion_alpha).view(1, 1, c, 1, 1)
        beta = torch.sigmoid(self.fusion_beta).view(1, 1, c, 1, 1)
        asm = alpha * cross_sm + beta * cross_ms

        if self.mk_dual_scale:
            gamma = torch.sigmoid(self.global_gamma)
            asm_pool = asm.mean(dim=(-2, -1), keepdim=True).expand_as(asm)
            fused = gamma * asm + (1.0 - gamma) * asm_pool
        else:
            xl = self.branch_large(x)
            xl_up = _interpolate_feat(xl, hs, ws)
            cross_global = self._cross_fhgr(asm, asm, xl_up, hs, ws)
            gamma = torch.sigmoid(self.global_gamma)
            asm_pool = asm.mean(dim=(-2, -1), keepdim=True).expand_as(asm)
            fused = gamma * cross_global + (1.0 - gamma) * asm_pool

        # Adaptive multi-head (sur tokens re-encodés pour appliquer gates)
        q_f, k_f, v_f, n_f, _ = self._encode_qkv(fused, t, b, c, hs, ws)
        out_tok = self._fhgr_mix(q_f, k_f, v_f, hs, ws)
        head_g = torch.sigmoid(self.head_gates).view(1, 1, self.num_heads, 1, 1)
        out_tok = out_tok * head_g

        x = self._tokens_to_map(out_tok, t, b, c, hs, ws, n_f)
        if (hs, ws) != (h, w):
            x = _interpolate_feat(x, h, w)

        x = self.out_lif(x)

        if self.dvs:
            x = x.mul(x_pool) + x_pool

        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(t, b, c, h, w).contiguous()
        if return_contribution:
            return x
        return x + identity
