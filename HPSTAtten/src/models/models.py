import torch
import torch.nn as nn

from modules.sps import SPS
from modules.hp_stattn import HPSTAtten
from modules.mlp import MLP
from modules.head import ClassificationHead
from modules.spike import make_lif

__all__ = ["MS_Block_Conv", "MS_Block_Membrane", "HPSTAttenTransformer"]


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
        attention_mode="factorized",
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
            attention_mode=attention_mode,
        )
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            spike_mode=spike_mode,
            lif_backend=lif_backend,
            layer=layer,
        )

    def forward(self, x):
        # x: (T, B, D, H, W)
        x = self.attn(x)
        # x: (T, B, D, H, W)
        return self.mlp(x)


class MS_Block_Membrane(nn.Module):
    """Bloc A²OS²A avec résidus sur membrane (eq. 15–17), adapté aux maps conv."""

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
        attention_mode="factorized",
    ):
        super().__init__()
        self.sn = make_lif(spike_mode, lif_backend=lif_backend)
        self.attn = HPSTAtten(
            dim,
            num_heads=num_heads,
            spike_mode=spike_mode,
            lif_backend=lif_backend,
            chunk_size=chunk_size,
            hybrid_qkv=hybrid_qkv,
            dvs=dvs,
            layer=layer,
            attention_mode=attention_mode,
        )
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            spike_mode=spike_mode,
            lif_backend=lif_backend,
            layer=layer,
        )

    def forward(self, s, u):
        u_prime = self.attn(s, return_contribution=True) + u
        s_prime = self.sn(u_prime)
        s_out = self.sn(self.mlp(s_prime, return_contribution=True) + u_prime)
        return s_out, u_prime


class HPSTAttenTransformer(nn.Module):
    """
    HP-STAtten Transformer — Proposition A.

    Pipeline (shapes) :
      - CIFAR-10       : (B, C, H, W)  with C=3, H=W=32
      - CIFAR-10-DVS   : (B, T, C, H, W) with C=2, H=W=128

      Forward:
        (B, C, H, W)          -> expand to (T, B, C, H, W)
        (B, T, C, H, W)       -> permute to (T, B, C, H, W)
        SPS                   -> (T, B, D, H', W')
        depth x (Attn + MLP)  -> (T, B, D, H', W')
        GAP over spatial tokens N=H'*W' -> (T, B, D)
        Head                  -> (B, num_classes)

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
        attention_mode="factorized",
        membrane_block=False,
    ):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.depth = depth
        self.membrane_block = membrane_block

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
        block_cls = MS_Block_Membrane if membrane_block else MS_Block_Conv
        self.blocks = nn.ModuleList(
            [
                block_cls(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    spike_mode=spike_mode,
                    lif_backend=lif_backend,
                    chunk_size=chunk_size,
                    hybrid_qkv=hybrid_qkv,
                    dvs=dvs,
                    layer=i,
                    attention_mode=attention_mode,
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
        # x: (T, B, C_in, H, W)
        x = self.patch_embed(x)
        # x: (T, B, D, H', W')
        if self.membrane_block:
            s = x
            u = torch.zeros_like(s)
            for blk in self.blocks:
                s, u = blk(s, u)
            x = s
        else:
            for blk in self.blocks:
                x = blk(x)
        # x: (T, B, D, H', W') -> flatten spatial -> (T, B, D, N) -> mean N -> (T, B, D)
        return x.flatten(3).mean(3)

    def forward(self, x, return_timesteps: bool = False):
        if x.dim() == 5:
            # DVS batch: (B, T, C, H, W) -> (T, B, C, H, W)
            x = x.permute(1, 0, 2, 3, 4)
        elif x.dim() == 4:
            # Image batch: (B, C, H, W) -> (T, B, C, H, W) by repeating in time
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        # features: (T, B, D) -> head -> (B, num_classes) or (T, B, num_classes)
        return self.head(self.forward_features(x), return_timesteps=return_timesteps)
