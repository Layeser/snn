import torch
import torch.nn as nn

from modules.contrast_attn import ContrastTokenMixer
from modules.spike import make_lif
from modules.ternary_lif import MultiStepTernaryLIFNode

__all__ = ["HPSTAtten"]

_ATTENTION_MODES = ("factorized", "sdt", "contrast")


class DvsPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

    def forward(self, x):
        return self.pool(x)


class HPSTAtten(nn.Module):
    """
    Hybrid-Precision Spatio-Temporal Attention (Proposition A).

    Fusion :
      - STAtten : chunks temporels + factorisation A = K^T V, out = Q @ A
      - A²OS²A  : Q binaire (LIF), K float (ReLU), V ternaire (TernaryLIF), SN final
      - Modes   : factorized | sdt | contrast (VCA-light sans Softmax)

    Entrée  : (T, B, D, H, W)
    Sortie  : (T, B, D, H, W)

    Référence : Notes/proprosition.md, Notes/vca_integration_attention.md
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
        attention_mode="factorized",
        vct_num=16,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        assert attention_mode in _ATTENTION_MODES, (
            f"attention_mode doit être l'un de {_ATTENTION_MODES} (reçu: {attention_mode!r})"
        )

        self.dim = dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.hybrid_qkv = hybrid_qkv
        self.dvs = dvs
        self.layer = layer
        self.attention_mode = attention_mode
        self.vct_num = vct_num

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
        self.k_relu = nn.ReLU(inplace=True)
        self.k_lif = lif()

        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        self.v_ternary = MultiStepTernaryLIFNode(tau=2.0, v_threshold=1.0)
        self.v_lif = lif()

        self.out_lif = lif()  # SN(Q @ A) — A²OS²A

        # Spike-Driven Self-Attention (SDSA) : neurone entre le produit K⊙V
        # sommé sur les tokens et le produit de Hadamard avec Q. Créé uniquement
        # en mode "sdt" pour préserver la compatibilité des checkpoints existants.
        if attention_mode == "sdt":
            self.talking_heads_lif = lif(v_threshold=0.5)

        # Visual-Contrast (VCA-inspired, sans Softmax). Modules créés uniquement
        # en mode "contrast" pour préserver la compatibilité des checkpoints.
        if attention_mode == "contrast":
            head_dim = dim // num_heads
            self.contrast_mixer = ContrastTokenMixer(
                num_heads=num_heads,
                head_dim=head_dim,
                vct_num=vct_num,
                layer=layer,
            )
            self.contrast_lif = lif(v_threshold=0.5)

        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_bn = nn.BatchNorm2d(dim)

    def _encode_qkv(self, x, T, B, C, H, W):
        """Q/K/V avec encodage hybride (A²OS²A) ou binaire (ablation)."""
        N = H * W
        head_dim = C // self.num_heads
        # x: (T, B, C, H, W) -> (T*B, C, H, W) for 2D convs
        x_for_qkv = x.flatten(0, 1)

        q = self.q_conv(x_for_qkv)
        # (T*B, C, H, W) -> (T, B, C, H, W) for multi-step spiking neuron
        q = self.q_bn(q).reshape(T, B, C, H, W).contiguous()
        q = self.q_lif(q)
        if self.dvs:
            q = self.pool(q)
        # Spatial tokens: (T, B, C, H, W) -> (T, B, heads, N, head_dim)
        q = (
            q.flatten(3).transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, head_dim)
            .permute(0, 1, 3, 2, 4).contiguous()
        )

        k = self.k_conv(x_for_qkv)
        k = self.k_bn(k).reshape(T, B, C, H, W).contiguous()
        if self.hybrid_qkv:
            k = self.k_relu(k)
        else:
            k = self.k_lif(k)
        if self.dvs:
            k = self.pool(k)
        # (T, B, C, H, W) -> (T, B, heads, N, head_dim)
        k = (
            k.flatten(3).transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, head_dim)
            .permute(0, 1, 3, 2, 4).contiguous()
        )

        v = self.v_conv(x_for_qkv)
        v = self.v_bn(v).reshape(T, B, C, H, W).contiguous()
        if self.hybrid_qkv:
            v = self.v_ternary(v)
        else:
            v = self.v_lif(v)
        if self.dvs:
            v = self.pool(v)
        # (T, B, C, H, W) -> (T, B, heads, N, head_dim)
        v = (
            v.flatten(3).transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, head_dim)
            .permute(0, 1, 3, 2, 4).contiguous()
        )
        return q, k, v, N, head_dim

    def _factorized_attention(self, q, k, v, T, B, C, H, W, N, head_dim):
        """STAtten : Q @ (K^T V) par chunk temporel."""
        if self.dvs:
            scaling_factor = 1.0 / (H * H * self.chunk_size)
        else:
            scaling_factor = 1.0 / H

        num_chunks = T // self.chunk_size
        # Group time into chunks:
        # q: (T, B, heads, N, head_dim)
        # -> (chunks, B, heads, cs, N, head_dim)
        q_chunks = q.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)
        k_chunks = k.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)
        v_chunks = v.view(num_chunks, self.chunk_size, B, self.num_heads, N, head_dim).permute(0, 2, 3, 1, 4, 5)

        # Merge chunk time and spatial tokens: cs*N tokens per chunk
        q_chunks = q_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)
        k_chunks = k_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)
        v_chunks = v_chunks.reshape(num_chunks, B, self.num_heads, self.chunk_size * N, head_dim)

        # Factorized attention (STAtten):
        # A = K^T V -> (chunks, B, heads, head_dim, head_dim)
        # out = Q A -> (chunks, B, heads, cs*N, head_dim)
        attn = torch.matmul(k_chunks.transpose(-2, -1), v_chunks) * scaling_factor
        out = torch.matmul(q_chunks, attn)

        # Restore original layout:
        # (chunks, B, heads, cs*N, head_dim) -> (T, B, heads, N, head_dim)
        out = out.reshape(num_chunks, B, self.num_heads, self.chunk_size, N, head_dim).permute(0, 3, 1, 2, 4, 5)
        output = out.reshape(T, B, self.num_heads, N, head_dim)

        # heads + head_dim -> channels
        # (T, B, heads, N, head_dim) -> (T, B, C, N) -> (T, B, C, H, W)
        x = output.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.out_lif(x).reshape(T, B, C, H, W).contiguous()
        return x

    def _sdt_attention(self, q, k, v, T, B, C, H, W, N, head_dim):
        """Spike-Driven Self-Attention : produit de Hadamard, complexité O(N·D).

        Contrairement à la factorisation STAtten (A = K^T V, matrice Dh×Dh),
        SDSA calcule un vecteur de contexte par tête en sommant K⊙V sur les
        tokens, puis le module par Q élément par élément.
        q, k, v : (T, B, heads, N, head_dim)
        """
        kv = k.mul(v)                       # (T, B, heads, N, head_dim)
        kv = kv.sum(dim=-2, keepdim=True)   # (T, B, heads, 1, head_dim)
        kv = self.talking_heads_lif(kv)
        x = q.mul(kv)                       # broadcast sur N -> (T, B, heads, N, head_dim)
        # heads + head_dim -> channels : (T, B, C, N) -> (T, B, C, H, W)
        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.out_lif(x).reshape(T, B, C, H, W).contiguous()
        return x

    def _contrast_attention(self, q, k, v, T, B, C, H, W, N, head_dim):
        """Visual-Contrast Attention linéaire (VCA sans Softmax).

        AvgPool(Q) → t±, Stage I/II différentiels via KᵀV / tᵀv̂ (associatif).
        Complexité O(N·Dh² + n·Dh²) avec n = vct_num ≪ N.
        q, k, v : (T, B, heads, N, head_dim)
        """
        if self.dvs:
            scaling_factor = 1.0 / (H * H)
        else:
            scaling_factor = 1.0 / H

        out = self.contrast_mixer(
            q, k, v, H=H, W=W,
            scaling_factor=scaling_factor,
            stage_lif=self.contrast_lif,
        )
        x = out.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.out_lif(x).reshape(T, B, C, H, W).contiguous()
        return x

    def forward(self, x, return_contribution: bool = False):
        # x: (T, B, C, H, W)
        T, B, C, H, W = x.shape
        identity = x

        x = self.shortcut_lif(x)
        x_pool = self.pool(x) if self.dvs else None

        q, k, v, N, head_dim = self._encode_qkv(x, T, B, C, H, W)
        if self.attention_mode == "sdt":
            x = self._sdt_attention(q, k, v, T, B, C, H, W, N, head_dim)
        elif self.attention_mode == "contrast":
            x = self._contrast_attention(q, k, v, T, B, C, H, W, N, head_dim)
        else:
            x = self._factorized_attention(q, k, v, T, B, C, H, W, N, head_dim)

        if self.dvs:
            # DVS gating: keep sparse activity aligned with pooled shortcut
            x = x.mul(x_pool) + x_pool

        # (T, B, C, H, W) -> (T*B, C, H, W) -> proj -> (T, B, C, H, W)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W).contiguous()
        if return_contribution:
            return x
        x = x + identity
        return x
