"""
Loader CIFAR-10 partagé — recette proche des repos STAtten / Spikformer.

Train : RandomCrop + Flip + (RandAugment) + (RandomErasing) + Normalize
Val   : ToTensor + Normalize
"""

from __future__ import annotations

from pathlib import Path

from data_download import ensure_cifar10
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _resolve_data_dir(data_dir=None, project_root: Path | None = None) -> Path:
    if data_dir is not None:
        return Path(data_dir)

    if project_root is None:
        project_root = Path.cwd()

    repo_data = project_root.parent / "data"
    local = project_root / "data"
    if repo_data.exists():
        return repo_data
    if local.exists():
        return local
    return repo_data


def build_cifar10_transforms(
    augment_train: bool = True,
    rand_augment: bool = True,
    random_erasing: float = 0.25,
):
    """
    Recette officielle (approximation sans timm) :
      - aa: rand-m9-n1-mstd0.4-inc1  → RandAugment magnitude ~9
      - reprob: 0.25                → RandomErasing p=0.25
    """
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)

    if not augment_train:
        basic = transforms.Compose([transforms.ToTensor(), normalize])
        return basic, basic

    train_steps = [
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
    ]
    if rand_augment:
        train_steps.append(transforms.RandAugment(num_ops=2, magnitude=9))
    train_steps.append(transforms.ToTensor())
    if random_erasing > 0:
        train_steps.append(transforms.RandomErasing(p=random_erasing, scale=(0.02, 0.33)))
    train_steps.append(normalize)

    val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    return transforms.Compose(train_steps), val_transform


def get_cifar10_loaders(
    data_dir=None,
    batch_size=128,
    num_workers=4,
    download=True,
    project_root: Path | None = None,
    augment_train: bool = True,
    rand_augment: bool = True,
    random_erasing: float = 0.25,
):
    data_dir = _resolve_data_dir(data_dir, project_root)
    data_dir.mkdir(parents=True, exist_ok=True)

    if download:
        ensure_cifar10(data_dir)

    train_tf, val_tf = build_cifar10_transforms(
        augment_train=augment_train,
        rand_augment=rand_augment,
        random_erasing=random_erasing,
    )

    train_set = datasets.CIFAR10(root=str(data_dir), train=True, download=False, transform=train_tf)
    val_set = datasets.CIFAR10(root=str(data_dir), train=False, download=False, transform=val_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
