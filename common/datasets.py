"""
Sélection de jeu de données (CIFAR-10 / CIFAR-10-DVS) et profils associés.

Utilisé par les scripts d'entraînement et MLflow pour tracer le dataset utilisé.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cifar10 import get_cifar10_loaders
from cifar10_dvs import get_cifar10_dvs_loaders
from config_data import SUPPORTED_DATASETS


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    display_name: str
    img_size: int
    in_channels: int
    num_classes: int
    dvs: bool
    temporal_input: bool
    storage_subdir: str

    @property
    def mlflow_label(self) -> str:
        return {"cifar10": "CIFAR10", "cifar10-dvs": "CIFAR10-DVS"}[self.name]


DATASET_PROFILES: dict[str, DatasetProfile] = {
    "cifar10": DatasetProfile(
        name="cifar10",
        display_name="CIFAR-10",
        img_size=32,
        in_channels=3,
        num_classes=10,
        dvs=False,
        temporal_input=False,
        storage_subdir="cifar-10-batches-py",
    ),
    "cifar10-dvs": DatasetProfile(
        name="cifar10-dvs",
        display_name="CIFAR-10-DVS",
        img_size=128,
        in_channels=2,
        num_classes=10,
        dvs=True,
        temporal_input=True,
        storage_subdir="CIFAR10DVS",
    ),
}


def get_dataset_profile(dataset: str) -> DatasetProfile:
    if dataset not in DATASET_PROFILES:
        raise ValueError(f"Dataset inconnu: {dataset!r}. Choix: {SUPPORTED_DATASETS}")
    return DATASET_PROFILES[dataset]


def mlflow_experiment_name(project_prefix: str, dataset: str) -> str:
    profile = get_dataset_profile(dataset)
    return f"{project_prefix}-{profile.mlflow_label}"


def dataset_hyperparams(dataset: str, data_dir: str | Path) -> dict[str, Any]:
    profile = get_dataset_profile(dataset)
    resolved = Path(data_dir).resolve()
    return {
        "dataset": profile.name,
        "dataset_display_name": profile.display_name,
        "data_dir": str(resolved),
        "data_storage_path": str(resolved / profile.storage_subdir),
        "img_size": profile.img_size,
        "in_channels": profile.in_channels,
        "num_classes": profile.num_classes,
        "dvs": profile.dvs,
    }


def dataset_split_params(train_loader, val_loader) -> dict[str, int]:
    """Tailles train/val pour logging MLflow (échantillons et batches)."""
    return {
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
    }


def loader_kwargs_for_dataset(config: dict[str, Any], profile: DatasetProfile) -> dict[str, Any]:
    if profile.name == "cifar10":
        return {
            "augment_train": config["augment_train"] == "true",
            "rand_augment": config["rand_augment"] == "true",
            "random_erasing": config["random_erasing"],
        }
    if profile.name == "cifar10-dvs":
        return {
            "augment_train": config.get("dvs_augment", "true") == "true",
            "random_split": config.get("dvs_random_split", "false") == "true",
        }
    return {}


def effective_mixup(config: dict[str, Any], profile: DatasetProfile) -> float:
    return config["mixup"]


def get_dataset_loaders(
    dataset: str,
    data_dir=None,
    batch_size=128,
    num_workers=4,
    download=True,
    project_root: Path | None = None,
    T: int = 4,
    **loader_kwargs,
):
    profile = get_dataset_profile(dataset)
    if profile.name == "cifar10":
        return get_cifar10_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            download=download,
            project_root=project_root,
            **loader_kwargs,
        )
    if profile.name == "cifar10-dvs":
        return get_cifar10_dvs_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            download=download,
            project_root=project_root,
            frames_number=T,
            **loader_kwargs,
        )
    raise ValueError(f"Dataset non supporté: {dataset!r}")
