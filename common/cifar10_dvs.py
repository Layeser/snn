"""
Loader CIFAR-10-DVS (spikingjelly).

Chaque échantillon est une séquence de frames (T, 2, H, W) avec H=W=128.
Le split train/val utilise split_to_train_test_set (90/10 par classe).
"""

from __future__ import annotations

from pathlib import Path

import spikingjelly.datasets as sjds
import torch
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from torch.utils.data import DataLoader

from cifar10 import _resolve_data_dir
from data_download import download_cifar10_dvs_archives

CIFAR10_DVS_NUM_CLASSES = 10
CIFAR10_DVS_TRAIN_RATIO = 0.9


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
):
    data_dir = _resolve_data_dir(data_dir, project_root)
    dvs_root = data_dir / "CIFAR10DVS"
    dvs_root.mkdir(parents=True, exist_ok=True)

    if download:
        download_cifar10_dvs_archives(dvs_root)

    origin_set = CIFAR10DVS(
        root=str(dvs_root),
        data_type="frame",
        frames_number=frames_number,
        split_by="number",
    )

    train_set, val_set = sjds.split_to_train_test_set(
        train_ratio,
        origin_set,
        num_classes=CIFAR10_DVS_NUM_CLASSES,
        random_split=False,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": _dvs_collate,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
