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
from data_subset import apply_train_fraction
from dvs_augment import DvsResize, build_dvs_train_transform
from reproducibility import dataloader_generator, seed_worker

CIFAR10_DVS_NUM_CLASSES = 10
CIFAR10_DVS_TRAIN_RATIO = 0.9


class _FrameTransformDataset(torch.utils.data.Dataset):
    """Applique resize (+ augmentation train optionnelle) sur les frames DVS."""

    def __init__(self, base, resize=None, transform=None):
        self.base = base
        self.resize = resize
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        frames, label = self.base[idx]
        frames = torch.as_tensor(frames, dtype=torch.float32)
        if self.resize is not None:
            frames = self.resize(frames)
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
    train_fraction: float = 1.0,
    seed: int | None = None,
    dvs_resize: int | None = 64,
    dvs_cutout: bool = True,
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

    if train_fraction < 1.0:
        if seed is None:
            raise ValueError("train_fraction < 1.0 requiert un seed pour un sous-échantillonnage reproductible.")
        full_train_size = len(train_set)
        train_set = apply_train_fraction(train_set, train_fraction, seed)
        print(
            f"Sous-échantillon train DVS: {len(train_set)}/{full_train_size} "
            f"({train_fraction:.1%}, stratifié, seed={seed})."
        )

    resize = DvsResize(dvs_resize) if dvs_resize is not None else None
    if dvs_resize is not None:
        print(f"Redimensionnement DVS activé : frames 128×128 → {dvs_resize}×{dvs_resize} (recette STAtten/SDT).")

    train_set = _FrameTransformDataset(
        train_set,
        resize=resize,
        transform=build_dvs_train_transform(cutout=dvs_cutout) if augment_train else None,
    )
    val_set = _FrameTransformDataset(val_set, resize=resize, transform=None)
    if augment_train:
        aug_msg = "flip + SNNAugmentWide"
        if dvs_cutout:
            aug_msg = "flip + Cutout + SNNAugmentWide"
        print(f"Augmentation DVS activée ({aug_msg}) sur le train.")

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": _dvs_collate,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    loader_seed = seed if seed is not None else None
    if loader_seed is not None:
        loader_kwargs["generator"] = dataloader_generator(loader_seed)
        if num_workers > 0:
            loader_kwargs["worker_init_fn"] = seed_worker

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
