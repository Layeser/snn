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
        img_size=64,
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


def mlflow_experiment_name_option_a(project_prefix: str, dataset: str) -> str:
    """Expérience MLflow dédiée Option A / Optuna lite (hors CIFAR10 / CIFAR10-DVS grid)."""
    profile = get_dataset_profile(dataset)
    return f"{project_prefix}-OptionA-{profile.mlflow_label}"


def is_option_a_optuna_study(study_name: str) -> bool:
    """Détecte les études Optuna Option A (lite ou complète)."""
    return "-oa-" in study_name or "-sota-" in study_name


def option_a_variant_from_study(study_name: str) -> str | None:
    """Extrait la variante Option A depuis un study_name Optuna (ex. hpstattn-cifar10-oa-hp → hp)."""
    for marker in ("-oa-", "-sota-"):
        if marker in study_name:
            return study_name.split(marker, 1)[1]
    return None


def mlflow_experiment_name_for_study(
    project_prefix: str,
    dataset: str,
    study_name: str | None = None,
    *,
    mlflow_experiment: str | None = None,
) -> str:
    """Nom d'expérience MLflow ; Option A → dossier HP-STAtten-OptionA-<dataset>."""
    if mlflow_experiment:
        return mlflow_experiment
    if study_name and is_option_a_optuna_study(study_name):
        return mlflow_experiment_name_option_a(project_prefix, dataset)
    return mlflow_experiment_name(project_prefix, dataset)


def mlflow_experiment_name_from_option_a_config(
    project_prefix: str,
    dataset: str,
    config_path: str | Path,
) -> str | None:
    """Déduit l'expérience Option A depuis campaigns option_a/ ou sota_lite/."""
    path_str = str(config_path).replace("\\", "/")
    if "option_a" in path_str or "sota_lite" in path_str:
        return mlflow_experiment_name_option_a(project_prefix, dataset)
    stem = Path(config_path).stem
    prefix = "cifar10_"
    if stem.startswith(prefix) and (stem.endswith("_best") or stem.endswith("_transferred")):
        return mlflow_experiment_name_option_a(project_prefix, dataset)
    return None


def dataset_hyperparams(dataset: str, data_dir: str | Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = get_dataset_profile(dataset)
    resolved = Path(data_dir).resolve()
    img_size = resolve_model_img_size(config, profile) if config is not None else profile.img_size
    params = {
        "dataset": profile.name,
        "dataset_display_name": profile.display_name,
        "data_dir": str(resolved),
        "data_storage_path": str(resolved / profile.storage_subdir),
        "img_size": img_size,
        "in_channels": profile.in_channels,
        "num_classes": profile.num_classes,
        "dvs": profile.dvs,
    }
    if config is not None and profile.dvs:
        params["dvs_resize"] = config.get("dvs_resize", profile.img_size)
        params["dvs_cutout"] = config.get("dvs_cutout", "true")
    return params


def dataset_split_params(train_loader, val_loader) -> dict[str, int]:
    """Tailles train/val pour logging MLflow (échantillons et batches)."""
    return {
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
    }


def resolve_model_img_size(config: dict[str, Any], profile: DatasetProfile) -> int:
    """Taille spatiale effective passée au modèle (après resize DVS éventuel)."""
    if not profile.dvs:
        return profile.img_size
    if "dvs_resize" in config and config["dvs_resize"] is not None:
        return int(config["dvs_resize"])
    return profile.img_size


def loader_kwargs_for_dataset(config: dict[str, Any], profile: DatasetProfile) -> dict[str, Any]:
    if profile.name == "cifar10":
        return {
            "augment_train": config["augment_train"] == "true",
            "rand_augment": config["rand_augment"] == "true",
            "random_erasing": config["random_erasing"],
        }
    if profile.name == "cifar10-dvs":
        dvs_resize = config.get("dvs_resize", profile.img_size)
        if dvs_resize is not None:
            dvs_resize = int(dvs_resize)
        return {
            "augment_train": config.get("dvs_augment", "true") == "true",
            "random_split": config.get("dvs_random_split", "false") == "true",
            "dvs_resize": dvs_resize,
            "dvs_cutout": config.get("dvs_cutout", "true") == "true",
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
    train_fraction: float = 1.0,
    seed: int | None = None,
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
            train_fraction=train_fraction,
            seed=seed,
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
            train_fraction=train_fraction,
            seed=seed,
            **loader_kwargs,
        )
    raise ValueError(f"Dataset non supporté: {dataset!r}")
