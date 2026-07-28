"""Sous-échantillonnage stratifié du jeu d'entraînement."""

from __future__ import annotations

import numpy as np
from torch.utils.data import Subset


def stratified_subset_indices(labels, fraction: float, seed: int) -> list[int]:
    """Retourne des indices stratifiés couvrant ~``fraction`` de chaque classe."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"train_fraction doit être dans (0, 1), reçu: {fraction}")

    labels_arr = np.asarray(labels)
    rng = np.random.default_rng(seed)
    selected: list[int] = []

    for class_id in np.unique(labels_arr):
        class_indices = np.where(labels_arr == class_id)[0]
        n_keep = max(1, int(round(len(class_indices) * fraction)))
        picked = rng.choice(class_indices, size=n_keep, replace=False)
        selected.extend(picked.tolist())

    return sorted(selected)


def _resolve_labels_and_positions(dataset) -> tuple[list[int], list[int]]:
    """Retourne (labels, positions_dans_dataset) pour un Dataset ou Subset."""
    if isinstance(dataset, Subset):
        base = dataset.dataset
        positions = list(dataset.indices)
        if hasattr(base, "targets"):
            labels = [base.targets[idx] for idx in positions]
        else:
            labels = [base[idx][1] for idx in positions]
        return labels, positions

    if hasattr(dataset, "targets"):
        positions = list(range(len(dataset)))
        return list(dataset.targets), positions

    positions = list(range(len(dataset)))
    labels = [dataset[idx][1] for idx in positions]
    return labels, positions


def apply_train_fraction(dataset, fraction: float, seed: int):
    """Sous-échantillonne un dataset d'entraînement en conservant les proportions par classe."""
    if fraction >= 1.0:
        return dataset

    labels, positions = _resolve_labels_and_positions(dataset)
    local_indices = stratified_subset_indices(labels, fraction, seed)
    mapped = [positions[i] for i in local_indices]

    if isinstance(dataset, Subset):
        return Subset(dataset.dataset, mapped)
    return Subset(dataset, mapped)
