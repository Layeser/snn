import torch.nn as nn

from modules.sps import SPS
from modules.stattn import STAtten
from modules.mlp import MLP
from modules.head import ClassificationHead

__all__ = ["MS_Block_Conv", "STAttenTransformer"]


class MS_Block_Conv(nn.Module):
    """
    Bloc Transformer = STAtten + MLP.

    Entrée  : (T, B, D, H, W)
    Sortie  : (T, B, D, H, W)
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        spike_mode="lif",
        lif_backend="auto",
        dvs=False,
        layer=0,
        attention_mode="STAtten",
        chunk_size=2,
    ):
        super().__init__()
        self.attn = STAtten(
            dim,
            num_heads=num_heads,
            spike_mode=spike_mode,
            lif_backend=lif_backend,
            dvs=dvs,
            layer=layer,
            attention_mode=attention_mode,
            chunk_size=chunk_size,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            spike_mode=spike_mode,
            lif_backend=lif_backend,
            layer=layer,
        )

    def forward(self, x):
        x = self.attn(x)   # (T, B, D, H, W)
        x = self.mlp(x)    # (T, B, D, H, W)
        return x


class STAttenTransformer(nn.Module):
    """
    Spatial-Temporal Attention Transformer (STAtten) pour CIFAR-10.

    Entrée forward  : (B, C_in, H, W)     ex. (B, 3, 32, 32)
    Sortie forward  : (B, num_classes)    ex. (B, 10)

    Pipeline interne :
      (B, 3, 32, 32)
        → repeat T
      (T, B, 3, 32, 32)
        → SPS
      (T, B, D, H', W')     ex. (4, B, 256, 8, 8)
        → depth × Block (STAtten + MLP)
      (T, B, D, H', W')
        → mean spatial
      (T, B, D)
        → head
      (B, 10)

    Config CIFAR officielle : depth=2, dim=256, T=4, chunk_size=2, pooling_stat="0011"
    Référence : https://github.com/Intelligent-Computing-Lab-Panda/STAtten
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
        attention_mode="STAtten",
        chunk_size=2,
        dvs=False,
        T=4,
    ):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.depth = depth
        self.chunk_size = chunk_size

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
                    dvs=dvs,
                    layer=i,
                    attention_mode=attention_mode,
                    chunk_size=chunk_size,
                )
                for i in range(depth)
            ]
        )
        self.head = (
            ClassificationHead(
                embed_dim, num_classes, spike_mode=spike_mode, lif_backend=lif_backend
            )
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
        x = self.patch_embed(x)              # (T, B, D, H', W')
        for blk in self.blocks:
            x = blk(x)                       # (T, B, D, H', W')
        x = x.flatten(3).mean(3)             # (T, B, D)
        return x

    def forward(self, x):
        # Appeler functional.reset_net(self) avant chaque batch en entraînement.
        if x.dim() == 5:
            x = x.permute(1, 0, 2, 3, 4)
        elif x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # (T, B, C_in, H, W)
        x = self.forward_features(x)         # (T, B, D)
        return self.head(x)                  # (B, num_classes)
