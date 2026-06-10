from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
SHARED_DATA_DIR = PROJECT_ROOT.parent.parent / "data"


def get_cifar10_loaders(
    data_dir=None,
    batch_size=128,
    num_workers=4,
    download=True,
):
    """
    Retourne (train_loader, val_loader).

    - train_loader : CIFAR-10 train (50k), shuffle=True
    - val_loader   : CIFAR-10 test  (10k), shuffle=False
    """
    if data_dir is None:
        if (SHARED_DATA_DIR / "cifar-10-batches-py").exists():
            data_dir = SHARED_DATA_DIR
        else:
            data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_set = datasets.CIFAR10(root=str(data_dir), train=True, download=download, transform=transform)
    val_set = datasets.CIFAR10(root=str(data_dir), train=False, download=download, transform=transform)

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
