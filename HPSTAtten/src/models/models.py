import torch.nn as nn

from modules.sps import SPS
from modules.hp_stattn import HPSTAtten
from modules.mlp import MLP
from modules.head import ClassificationHead

__all__ = ["MS_Block_Conv", "HPSTAttenTransformer"]


class MS_Block_Conv(nn.Module):
    """Bloc = HP-STAtten + MLP SDT."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        spike_mode="lif",
        lif_backend="auto",
        chunk_size=2,
        hybrid_qkv=True,
        dvs=False,
        layer=0,
    ):
        super().__init__()
        self.attn = HPSTAtten(
            dim,
            num_heads=num_heads,
            spike_mode=spike_mode,
            lif_backend=lif_backend,
            chunk_size=chunk_size,
            hybrid_qkv=hybrid_qkv,
            dvs=dvs,
            layer=layer,
        )
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            spike_mode=spike_mode,
            lif_backend=lif_backend,
            layer=layer,
        )

    def forward(self, x):
        x = self.attn(x)
        return self.mlp(x)


class HPSTAttenTransformer(nn.Module):
    """
    HP-STAtten Transformer — Proposition A.

    Pipeline :
      (B,3,H,W) → repeat T → SPS → depth×(HP-STAtten+MLP)
      → GAP spatial mean(N) → (T,B,D)
      → head_lif → Linear → mean(T) → (B, C)

    Config CIFAR : depth=2, D=256, T=4, chunk_size=2, pooling_stat=0011
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
        pooling_stat="0011",
        spike_mode="lif",
        lif_backend="auto",
        chunk_size=2,
        hybrid_qkv=True,
        dvs=False,
        T=4,
    ):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.depth = depth

        if T % chunk_size != 0:
            raise ValueError(f"T ({T}) doit être divisible par chunk_size ({chunk_size})")

        self.patch_embed = SPS(
            img_size_h=img_size,
            img_size_w=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dim,
            pooling_stat=pooling_stat,
            spike_mode=spike_mode,
            lif_backend=lif_backend,
        )
        self.blocks = nn.ModuleList(
            [
                MS_Block_Conv(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    spike_mode=spike_mode,
                    lif_backend=lif_backend,
                    chunk_size=chunk_size,
                    hybrid_qkv=hybrid_qkv,
                    dvs=dvs,
                    layer=i,
                )
                for i in range(depth)
            ]
        )
        self.head = (
            ClassificationHead(embed_dim, num_classes, spike_mode=spike_mode, lif_backend=lif_backend)
            if num_classes > 0
            else nn.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        return x.flatten(3).mean(3)

    def forward(self, x):
        if x.dim() == 5:
            x = x.permute(1, 0, 2, 3, 4)
        elif x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        return self.head(self.forward_features(x))
