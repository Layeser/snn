from modules.spn import SPS
from modules.ssa import SSA
from modules.mlp import MLP
from modules.head import ClassificationHead
import torch
import torch.nn as nn


class Block(nn.Module):
    """
    Bloc Transformer spiking = SSA + MLP avec connexions résiduelles.

    Entrée  : (T, B, N, D)
    Sortie  : (T, B, N, D)
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        lif_backend="auto",
    ):
        super().__init__()
        # LayerNorm présent dans le repo officiel mais non utilisé dans forward — à brancher si besoin.
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SSA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            lif_backend=lif_backend,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, lif_backend=lif_backend)

    def forward(self, x):
        # Référence officielle CIFAR : résidu direct (sans norm). Décommenter pour pre-norm :
        # x = x + self.attn(self.norm1(x))   # (T, B, N, D)
        # x = x + self.mlp(self.norm2(x))    # (T, B, N, D)
        x = x + self.attn(x)                 # (T, B, N, D)  résidu + SSA
        x = x + self.mlp(x)                  # (T, B, N, D)  résidu + MLP
        return x


class Spikformer(nn.Module):
    """
    Spikformer pour CIFAR-10/100.

    Entrée forward  : (B, C_in, H, W)     ex. (B, 3, 32, 32)
    Sortie forward  : (B, num_classes)    ex. (B, 10) pour CIFAR-10

    Pipeline interne :
      (B, C_in, H, W)
        → repeat T pas
      (T, B, C_in, H, W)
        → SPS
      (T, B, N, D)
        → depth × Block
      (T, B, N, D)
        → mean sur N (patches)
      (T, B, D)
        → mean sur T (temps) + head
      (B, num_classes)
    """

    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        lif_backend="auto",
        T=4,
    ):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.depth = depth

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.patch_embed = SPS(
            img_size_h=img_size,
            img_size_w=img_size,
            patch_size=(patch_size, patch_size),
            in_channels=in_channels,
            embed_dims=embed_dim,
            lif_backend=lif_backend,
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    lif_backend=lif_backend,
                )
                for i in range(depth)
            ]
        )
        self.head = (
            ClassificationHead(embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)              # (T, B, N, D)
        for blk in self.blocks:
            x = blk(x)                       # (T, B, N, D)
        return x.mean(2)                     # (T, B, D)  mean sur dim patches N

    def forward(self, x):
        # Appeler functional.reset_net(self) avant chaque batch en entraînement.
        if x.dim() == 5:
            x = x.permute(1, 0, 2, 3, 4)
        else:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)         # (T, B, D)
        x = x.mean(0)                        # (B, D)  mean sur dim temps T
        x = self.head(x)                     # (B, num_classes)
        return x
