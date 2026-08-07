"""
Augmentation spécifique DVS (frames événementielles).

Reprend la recette Spikformer / STAtten pour CIFAR-10-DVS :
  - RandomHorizontalFlip(p=0.5)
  - SNNAugmentWide : une opération géométrique tirée au hasard parmi
    {Identity, ShearX, TranslateX, TranslateY, Rotate, Cutout}.

Contrairement à RandAugment/RandomErasing de torchvision (pensés pour des
images RGB uint8), ces transforms opèrent directement sur un tenseur de frames
float de forme (T, C, H, W). torchvision applique la même transformation
géométrique à toutes les frames T (cohérence temporelle), ce qui est le
comportement voulu pour des événements.

Référence : https://github.com/ZK-Zhou/spikformer (cifar10dvs/autoaugment.py)
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn.functional as nnf
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from torchvision.transforms.transforms import RandomErasing

__all__ = ["SNNAugmentWide", "DvsResize", "DvsCutout", "build_dvs_train_transform"]


def _apply_op(
    img: Tensor,
    op_name: str,
    magnitude: float,
    interpolation: InterpolationMode,
    fill: Optional[List[float]],
) -> Tensor:
    if op_name == "ShearX":
        img = F.affine(
            img, angle=0.0, translate=[0, 0], scale=1.0,
            shear=[math.degrees(magnitude), 0.0],
            interpolation=interpolation, fill=fill,
        )
    elif op_name == "ShearY":
        img = F.affine(
            img, angle=0.0, translate=[0, 0], scale=1.0,
            shear=[0.0, math.degrees(magnitude)],
            interpolation=interpolation, fill=fill,
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img, angle=0.0, translate=[int(magnitude), 0], scale=1.0,
            shear=[0.0, 0.0], interpolation=interpolation, fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img, angle=0.0, translate=[0, int(magnitude)], scale=1.0,
            shear=[0.0, 0.0], interpolation=interpolation, fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"Opérateur non reconnu: {op_name}")
    return img


class DvsResize(torch.nn.Module):
    """Redimensionne chaque frame (T, C, H, W) — recette STAtten/SDT (128 → 64)."""

    def __init__(self, size: int = 64) -> None:
        super().__init__()
        self.size = size

    def forward(self, img: Tensor) -> Tensor:
        if img.dim() != 4:
            raise ValueError(f"DvsResize attend (T, C, H, W), reçu {tuple(img.shape)}")
        h, w = img.size(-2), img.size(-1)
        if h == self.size and w == self.size:
            return img
        # (T, C, H, W) : chaque pas de temps est redimensionné indépendamment.
        return nnf.interpolate(img, size=(self.size, self.size), mode="nearest")


class DvsCutout(torch.nn.Module):
    """Cutout spatial identique sur tous les pas de temps (STAtten: n_holes=1, length=16)."""

    def __init__(self, n_holes: int = 1, length: int = 16) -> None:
        super().__init__()
        self.n_holes = n_holes
        self.length = length

    def forward(self, img: Tensor) -> Tensor:
        if img.dim() != 4:
            raise ValueError(f"DvsCutout attend (T, C, H, W), reçu {tuple(img.shape)}")
        h, w = img.size(-2), img.size(-1)
        out = img.clone()
        for _ in range(self.n_holes):
            y = int(torch.randint(0, h, (1,)).item())
            x = int(torch.randint(0, w, (1,)).item())
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            out[..., y1:y2, x1:x2] = 0.0
        return out


class SNNAugmentWide(torch.nn.Module):
    """
    TrivialAugment-Wide restreint aux opérations sûres pour des frames DVS.

    Une seule opération est tirée à chaque appel, avec une magnitude aléatoire.
    Entrée / sortie : tenseur float (T, C, H, W).
    """

    def __init__(
        self,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        # Cutout : efface une petite zone (identique sur toutes les frames T).
        self.cutout = RandomErasing(p=1.0, scale=(0.001, 0.11), ratio=(1.0, 1.0))

    def _augmentation_space(self, num_bins: int):
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(-0.3, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(-5.0, 5.0, num_bins), True),
            "TranslateY": (torch.linspace(-5.0, 5.0, num_bins), True),
            "Rotate": (torch.linspace(-30.0, 30.0, num_bins), True),
            "Cutout": (torch.linspace(1.0, 30.0, num_bins), True),
        }

    def forward(self, img: Tensor) -> Tensor:
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = (
            float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
            if magnitudes.ndim > 0
            else 0.0
        )
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        if op_name == "Cutout":
            return self.cutout(img)
        return _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_magnitude_bins={self.num_magnitude_bins}, "
            f"interpolation={self.interpolation}, fill={self.fill})"
        )


def build_dvs_train_transform(*, cutout: bool = True) -> transforms.Compose:
    """Transform train DVS : flip + Cutout (optionnel) + SNNAugmentWide."""
    steps: list[torch.nn.Module | transforms.RandomHorizontalFlip] = [
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if cutout:
        steps.append(DvsCutout(n_holes=1, length=16))
    steps.append(SNNAugmentWide())
    return transforms.Compose(steps)
