import torch.nn as nn

from modules.sps import SPS
from modules.spike import make_lif
from modules.a2os2a import A2OS2A
from modules.mlp import MLP
from modules.head import ClassificationHead

__all__ = ["Block", "A2OS2ATransformer"]


class Block(nn.Module):
    """
    Bloc A2OS2A avec résidus sur membrane (papier eq. 15–17).

    Entrée  : s (T,B,N,D) spikes, u (T,B,N,D) membrane
    Sortie  : s_out, u_out
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, qkv_bias=False, lif_backend="auto"):
        super().__init__()
        self.sn = make_lif(lif_backend=lif_backend)
        self.attn = A2OS2A(dim, num_heads=num_heads, qkv_bias=qkv_bias, lif_backend=lif_backend)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden, lif_backend=lif_backend)

    def forward(self, s, u):
        u_prime = self.attn(s) + u                                         # eq. 15
        s_prime = self.sn(u_prime)                                         # eq. 16
        s_out = self.sn(self.mlp(s_prime) + u_prime)                       # eq. 17
        return s_out, u_prime


class A2OS2ATransformer(nn.Module):
    """
    Spiking Transformer avec A²OS²A (Guo et al., CVPR 2025).

    Entrée forward  : (B, 3, 32, 32)
    Sortie forward  : (B, num_classes)

    Pipeline :
      repeat T → SPS → U0
      S0 = SN(U0)
      blocs (s, u) avec A2OS2A + MLP
      CH(GAP(S_L))

    Config CIFAR ablation : depth=2, D=256, T=4 (Table 1 du papier)
    Référence : https://arxiv.org/abs/2503.00226
    """

    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        depth=2,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        lif_backend="auto",
        T=4,
    ):
        super().__init__()
        self.T = T
        self.num_classes = num_classes

        self.patch_embed = SPS(
            img_size_h=img_size,
            img_size_w=img_size,
            patch_size=(patch_size, patch_size),
            in_channels=in_channels,
            embed_dims=embed_dim,
            lif_backend=lif_backend,
        )
        self.s0_sn = make_lif(lif_backend=lif_backend)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    lif_backend=lif_backend,
                )
                for _ in range(depth)
            ]
        )
        self.head = ClassificationHead(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        u = self.patch_embed(x)              # (T, B, N, D)  U0
        s = self.s0_sn(u)                    # (T, B, N, D)  S0  eq. 14
        for blk in self.blocks:
            s, u = blk(s, u)
        return s

    def forward(self, x):
        if x.dim() == 5:
            x = x.permute(1, 0, 2, 3, 4)
        elif x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        s = self.forward_features(x)
        return self.head(s)
