import torch
import torch.nn as nn

from modules.sps import SPS
from modules.sdsa import SDSA
from modules.mlp import MLP
from modules.head import ClassificationHead

__all__ = ["MS_Block_Conv", "SpikeDrivenTransformer"]


class MS_Block_Conv(nn.Module):
    """
    Bloc Spike-Driven Transformer = SDSA + MLP.

    Entrée  : (T, B, D, H, W)
    Sortie  : (T, B, D, H, W)
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        spike_mode="lif",
        dvs=False,
        layer=0,
    ):
        super().__init__()
        self.attn = SDSA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            spike_mode=spike_mode,
            dvs=dvs,
            layer=layer,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            spike_mode=spike_mode,
            layer=layer,
        )

    def forward(self, x):
        x = self.attn(x)   # (T, B, D, H, W)
        x = self.mlp(x)    # (T, B, D, H, W)
        return x


class SpikeDrivenTransformer(nn.Module):
    """
    Spike-Driven Transformer (SDT) pour CIFAR-10.

    Entrée forward  : (B, C_in, H, W)     ex. (B, 3, 32, 32)
    Sortie forward  : (B, num_classes)    ex. (B, 10)

    Pipeline interne :
      (B, C_in, H, W)
        → repeat T pas
      (T, B, C_in, H, W)
        → SPS
      (T, B, D, H', W')
        → depth × Block
      (T, B, D, H', W')
        → mean spatial
      (T, B, D)
        → head (LIF + Linear + mean T)
      (B, num_classes)

    Référence : https://github.com/BICLab/Spike-Driven-Transformer
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
        drop_path_rate=0.0,
        pooling_stat="0011",
        spike_mode="lif",
        dvs=False,
        T=4,
    ):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.depth = depth

        self.patch_embed = SPS(
            img_size_h=img_size,
            img_size_w=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dim,
            pooling_stat=pooling_stat,
            spike_mode=spike_mode,
        )
        self.blocks = nn.ModuleList(
            [
                MS_Block_Conv(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    spike_mode=spike_mode,
                    dvs=dvs,
                    layer=i,
                )
                for i in range(depth)
            ]
        )
        self.head = (
            ClassificationHead(embed_dim, num_classes, spike_mode=spike_mode)
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
        x = x.flatten(3).mean(3)             # (T, B, D)  mean spatial H'*W'
        return x

    def forward(self, x):
        # Appeler functional.reset_net(self) avant chaque batch en entraînement.
        if x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # (T, B, C_in, H, W)
        x = self.forward_features(x)         # (T, B, D)
        return self.head(x)                  # (B, num_classes)
