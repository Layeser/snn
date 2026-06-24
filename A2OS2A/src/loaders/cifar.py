import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT.parent / "common"))

from cifar10 import CIFAR10_MEAN, CIFAR10_STD
from datasets import get_dataset_loaders, get_dataset_profile

__all__ = [
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "get_cifar10_loaders",
    "get_dataset_loaders",
    "get_dataset_profile",
]


def get_cifar10_loaders(
    data_dir=None,
    batch_size=128,
    num_workers=4,
    download=True,
    augment_train=True,
    rand_augment=True,
    random_erasing=0.25,
):
    return get_dataset_loaders(
        dataset="cifar10",
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
        project_root=PROJECT_ROOT,
        augment_train=augment_train,
        rand_augment=rand_augment,
        random_erasing=random_erasing,
    )
