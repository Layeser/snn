"""Utilitaires Optuna partagés pour l'optimisation d'hyperparamètres.

Objectifs :
- Stockage SQLite versionnable (`optuna.db`) au même niveau que `mlflow.db`,
  pour synchroniser les études entre machines (SSH, local, serveurs) via git.
- Création/reprise d'une étude nommée (relançable et parallélisable).
- Persistance des meilleurs hyperparamètres dans un YAML réutilisable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def optuna_storage_url(project_root: str | Path, *, db_name: str = "optuna.db") -> str:
    """Retourne une URL SQLite absolue pour le stockage Optuna.

    Le fichier vit à la racine du projet (comme `mlflow.db`) afin d'être suivi
    par git et partagé entre environnements d'exécution.
    """
    root = Path(project_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{root / db_name}"


def create_study(
    *,
    study_name: str,
    storage: str,
    direction: str = "maximize",
    seed: int | None = None,
    use_pruner: bool = True,
    n_warmup_steps: int = 5,
):
    """Crée (ou reprend) une étude Optuna avec TPE + MedianPruner.

    `load_if_exists=True` permet de relancer la même commande pour ajouter des
    essais à une étude existante, ou de lancer plusieurs workers en parallèle
    sur le même storage.
    """
    import optuna

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = (
        optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps)
        if use_pruner
        else optuna.pruners.NopPruner()
    )
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def save_best_params(path: str | Path, study) -> Path:
    """Écrit les meilleurs hyperparamètres de l'étude dans un fichier YAML."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "study_name": study.study_name,
        "direction": study.direction.name.lower(),
        "best_value": float(study.best_value),
        "best_trial": int(study.best_trial.number),
        "n_trials": len(study.trials),
        "best_params": dict(study.best_params),
    }
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return out


def summarize_study(study) -> str:
    """Résumé texte lisible d'une étude terminée."""
    import optuna

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    lines = [
        f"Étude: {study.study_name}",
        f"Essais: {len(study.trials)} (complétés={len(completed)}, prunés={len(pruned)})",
        f"Meilleure valeur: {study.best_value:.4f} (trial #{study.best_trial.number})",
        "Meilleurs hyperparamètres:",
    ]
    for key, value in study.best_params.items():
        lines.append(f"  - {key}: {value}")
    return "\n".join(lines)
