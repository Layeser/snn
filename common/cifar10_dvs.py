"""
Loader CIFAR-10-DVS (spikingjelly).

Chaque échantillon est une séquence de frames (T, 2, H, W) avec H=W=128.
Le split train/val utilise split_to_train_test_set (90/10 par classe).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import spikingjelly.datasets as sjds
import torch
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from torch.utils.data import DataLoader, Subset

from cifar10 import _resolve_data_dir
from cifar10_dvs_patch import apply_cifar10_dvs_numpy2_patch, prepare_cifar10_dvs_frames
from dvs_augment import build_dvs_train_transform

CIFAR10_DVS_NUM_CLASSES = 10
CIFAR10_DVS_TRAIN_RATIO = 0.9


class _AugmentedFrameDataset(torch.utils.data.Dataset):
    """Applique une transform (augmentation DVS) à la volée sur les frames.

    Chaque échantillon (T, C, H, W) est converti en tenseur float puis passé à
    la transform. Utilisé uniquement pour le split train.
    """

    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        frames, label = self.base[idx]
        frames = torch.as_tensor(frames, dtype=torch.float32)
        if self.transform is not None:
            frames = self.transform(frames)
        return frames, label


def _split_cache_path(
    dvs_root: Path,
    train_ratio: float,
    frames_number: int,
    *,
    random_split: bool,
    seed: int,
) -> Path:
    mode = "random" if random_split else "ordered"
    return dvs_root / f"split_{mode}_seed_{seed}_train_{train_ratio}_frames_{frames_number}.pt"


def _cached_train_val_split(
    origin_set,
    dvs_root: Path,
    train_ratio: float,
    frames_number: int,
    num_classes: int = CIFAR10_DVS_NUM_CLASSES,
    *,
    random_split: bool = True,
    seed: int = 0,
):
    cache_path = _split_cache_path(
        dvs_root,
        train_ratio,
        frames_number,
        random_split=random_split,
        seed=seed,
    )
    if cache_path.is_file():
        indices = torch.load(cache_path, weights_only=True)
        return (
            Subset(origin_set, indices["train"]),
            Subset(origin_set, indices["val"]),
        )

    if random_split:
        np.random.seed(seed)
    train_set, val_set = sjds.split_to_train_test_set(
        train_ratio,
        origin_set,
        num_classes=num_classes,
        random_split=random_split,
    )
    torch.save(
        {"train": train_set.indices, "val": val_set.indices},
        cache_path,
    )
    return train_set, val_set


def _dvs_collate(batch):
    frames, labels = zip(*batch)
    stacked = torch.stack([torch.as_tensor(f, dtype=torch.float32) for f in frames], dim=0)
    targets = torch.tensor(labels, dtype=torch.long)
    return stacked, targets


def get_cifar10_dvs_loaders(
    data_dir=None,
    batch_size=64,
    num_workers=4,
    download=True,
    project_root: Path | None = None,
    frames_number: int = 10,
    train_ratio: float = CIFAR10_DVS_TRAIN_RATIO,
    augment_train: bool = True,
    random_split: bool = False,
):
    apply_cifar10_dvs_numpy2_patch()

    data_dir = _resolve_data_dir(data_dir, project_root)
    dvs_root = data_dir / "CIFAR10DVS"
    dvs_root.mkdir(parents=True, exist_ok=True)

    if download:
        prepare_cifar10_dvs_frames(dvs_root, frames_number=frames_number, split_by="number")

    origin_set = CIFAR10DVS(
        root=str(dvs_root),
        data_type="frame",
        frames_number=frames_number,
        split_by="number",
    )

    train_set, val_set = _cached_train_val_split(
        origin_set,
        dvs_root,
        train_ratio,
        frames_number,
        random_split=random_split,
    )
    print(
        f"Split DVS: {'random (seed=0)' if random_split else 'ordonné (comme Spikformer)'} "
        f"— {int(train_ratio * 100)}/{int((1 - train_ratio) * 100)} stratifié par classe."
    )

    if augment_train:
        train_set = _AugmentedFrameDataset(train_set, build_dvs_train_transform())
        print("Augmentation DVS activée (flip + SNNAugmentWide) sur le train.")

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": _dvs_collate,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
