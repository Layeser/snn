#!/usr/bin/env python3
"""
Optimisation d'hyperparamètres HP-STAtten avec Optuna.

Réutilise exactement la même boucle d'entraînement que scripts/train.py
(run_training), mais :
  - chaque essai (trial) a son propre save_dir ;
  - reprise automatique des essais interrompus (Ctrl+C / préemption OAR) ;
  - Optuna échantillonne les hyperparamètres (LR, weight_decay, mixup, ...) ;
  - un budget d'epochs réduit (--tune-epochs) accélère la recherche ;
  - le pruning (MedianPruner) coupe tôt les essais peu prometteurs ;
  - l'étude est stockée dans optuna.db (SQLite, versionnable comme mlflow.db).

Exemples :
  python -m scripts.tune --n-trials 30 --tune-epochs 30 --dataset cifar10-dvs \
      --data-dir ../data --batch-size 8 --T 8
  make tune DATASET=cifar10-dvs N_TRIALS=30
"""
import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "train_cifar10.yml"
CONFIG_BY_DATASET = {
    "cifar10": ROOT / "config" / "train_cifar10.yml",
    "cifar10-dvs": ROOT / "config" / "train_cifar10-dvs.yml",
}
MLFLOW_PROJECT_PREFIX = "HP-STAtten"

sys.path.insert(0, str(ROOT.parent / "common"))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src"))

import optuna

from datasets import (
    dataset_hyperparams,
    dataset_split_params,
    get_dataset_loaders,
    get_dataset_profile,
    mlflow_experiment_name,
)
from models import HPSTAttenTransformer
from modules.spike import resolve_lif_backend
from optuna_search import (
    create_study,
    optuna_storage_url,
    resume_interrupted_trials,
    save_best_params,
    summarize_study,
)
from reproducibility import set_seed
from train_integration import (
    apply_dataset_training_overrides,
    build_criterion,
    build_scheduler,
    configure_cuda_runtime,
    loader_kwargs_from_config,
    recipe_hyperparams,
    resolve_num_workers,
    resolve_project_path,
    resolve_training_batch_size,
)
from training_runner import run_training
from utils.config import load_and_validate_config
from utils.config_schema import HPSTATTEN_CONFIG_SCHEMA, validate_hpstattn_config
from utils.device import resolve_device
from utils.training import train_one_epoch, validate


def build_tune_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Optimisation Optuna pour HP-STAtten")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--n-trials", type=int, default=20, help="Nombre d'essais Optuna")
    p.add_argument("--tune-epochs", type=int, default=30, help="Epochs par essai (budget réduit)")
    p.add_argument("--timeout", type=int, default=None, help="Temps max en secondes (optionnel)")
    p.add_argument("--study-name", type=str, default=None, help="Nom de l'étude (reprise/parallèle)")
    p.add_argument("--dataset", type=str, default=None, choices=["cifar10", "cifar10-dvs"])
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--save-dir", type=str, default=None, help="Dossier racine des checkpoints d'essais")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None, help="Batch fixe pour l'étude")
    p.add_argument("--T", type=int, default=None, help="Timesteps fixes pour l'étude")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Fraction stratifiée du train à utiliser (ex: 0.333 pour 1/3)",
    )
    p.add_argument("--tune-arch", action="store_true", help="Inclure embed_dim/depth/num_heads dans la recherche")
    p.add_argument("--no-pruning", action="store_true", help="Désactiver le MedianPruner")
    p.add_argument("--no-resume-interrupted", action="store_true",
                   help="Ne pas reprendre les essais Optuna laissés en RUNNING")
    return p


def params_from_trial(
    trial_params: dict[str, Any],
    base_config: dict[str, Any],
    tune_arch: bool,
) -> dict[str, Any]:
    """Reconstruit le dict d'hyperparamètres à partir d'un essai Optuna stocké."""
    params = dict(trial_params)
    if not tune_arch:
        params.setdefault("embed_dim", int(base_config["emb_dim"]))
        params.setdefault("depth", int(base_config["depth"]))
        params.setdefault("num_heads", int(base_config["num_heads"]))
    if params.get("scheduler") != "cosine":
        params.setdefault("warmup_epochs", int(base_config["warmup_epochs"]))
    return params


def suggest_hyperparams(trial, base_config: dict[str, Any], tune_epochs: int, tune_arch: bool) -> dict[str, Any]:
    """Espace de recherche : optimisation + régularisation (+ archi si demandé)."""
    params: dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "mixup": trial.suggest_float("mixup", 0.0, 0.8, step=0.1),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.15, step=0.05),
        "scheduler": trial.suggest_categorical("scheduler", ["none", "cosine"]),
    }
    if params["scheduler"] == "cosine":
        params["warmup_epochs"] = trial.suggest_int("warmup_epochs", 0, max(1, tune_epochs // 5))
    else:
        params["warmup_epochs"] = int(base_config["warmup_epochs"])

    if tune_arch:
        params["embed_dim"] = trial.suggest_categorical("embed_dim", [128, 256])
        params["depth"] = trial.suggest_categorical("depth", [2, 4])
        params["num_heads"] = trial.suggest_categorical("num_heads", [4, 8])
    else:
        params["embed_dim"] = int(base_config["emb_dim"])
        params["depth"] = int(base_config["depth"])
        params["num_heads"] = int(base_config["num_heads"])
    return params


def main() -> None:
    tune_args = build_tune_parser().parse_args()

    config_path = Path(tune_args.config)
    base_config = load_and_validate_config(
        config_path,
        HPSTATTEN_CONFIG_SCHEMA,
        extra_validators=[validate_hpstattn_config],
    )

    dataset = tune_args.dataset or base_config["dataset"]
    data_dir = resolve_project_path(tune_args.data_dir or base_config["data_dir"], ROOT)
    profile = get_dataset_profile(dataset)
    device = resolve_device(tune_args.device or base_config["device"])
    use_amp = configure_cuda_runtime(device)
    set_seed(tune_args.seed)

    # Batch/T fixes pour toute l'étude → on construit les loaders une seule fois.
    batch_size = resolve_training_batch_size(
        tune_args.batch_size or int(base_config["batch_size"]), base_config, profile
    )
    T = tune_args.T or int(base_config["T"])
    num_workers = resolve_num_workers(
        tune_args.num_workers if tune_args.num_workers is not None else int(base_config["num_workers"]),
        profile,
    )

    loader_config = apply_dataset_training_overrides(base_config, dataset)
    train_loader, val_loader = get_dataset_loaders(
        dataset=dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        download=True,
        project_root=ROOT,
        T=T,
        train_fraction=tune_args.train_fraction,
        seed=tune_args.seed,
        **loader_kwargs_from_config(loader_config, dataset),
    )

    study_name = tune_args.study_name or f"hpstattn-{dataset}"
    storage = optuna_storage_url(ROOT)
    study = create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        seed=tune_args.seed,
        use_pruner=not tune_args.no_pruning,
    )
    base_save_dir = Path(resolve_project_path(tune_args.save_dir or "save/optuna", ROOT))
    experiment_name = mlflow_experiment_name(MLFLOW_PROJECT_PREFIX, dataset)

    print(f"Étude Optuna: {study_name}")
    print(f"Storage: {storage}")
    print(f"Dataset: {profile.display_name} | batch={batch_size} T={T} | device={device} AMP={use_amp}")
    print(f"Seed: {tune_args.seed} | train_fraction: {tune_args.train_fraction}")
    print(f"Budget: {tune_args.n_trials} essais × {tune_args.tune_epochs} epochs | pruning={not tune_args.no_pruning}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    lif_backend = resolve_lif_backend(base_config["lif_backend"])
    hybrid_qkv = base_config["hybrid_qkv"] == "true"
    attention_mode = base_config["attention_mode"]

    study_dir = base_save_dir / study_name

    def run_optuna_trial(trial, *, params: dict[str, Any], fresh: bool) -> float:
        trial_config = copy.deepcopy(base_config)
        trial_config["mixup"] = params["mixup"]
        trial_config["label_smoothing"] = params["label_smoothing"]
        trial_config["scheduler"] = params["scheduler"]
        trial_config["warmup_epochs"] = params["warmup_epochs"]
        trial_config = apply_dataset_training_overrides(trial_config, dataset)
        trial_config["use_amp"] = use_amp

        model = HPSTAttenTransformer(
            img_size=profile.img_size,
            in_channels=profile.in_channels,
            num_classes=profile.num_classes,
            embed_dim=params["embed_dim"],
            depth=params["depth"],
            num_heads=params["num_heads"],
            pooling_stat=base_config["pooling_stat"],
            spike_mode=base_config["spike_mode"],
            lif_backend=base_config["lif_backend"],
            chunk_size=base_config["chunk_size"],
            hybrid_qkv=hybrid_qkv,
            dvs=profile.dvs,
            T=T,
            attention_mode=attention_mode,
        ).to(device)

        criterion = build_criterion(trial_config)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )
        scheduler = build_scheduler(optimizer, trial_config, tune_args.tune_epochs)

        trial_args = argparse.Namespace(
            config=str(config_path),
            epochs=tune_args.tune_epochs,
            fresh=fresh,
            resume="auto" if not fresh else "none",
            resume_path=None,
            dataset=dataset,
            **params,
        )
        save_dir = study_dir / f"trial_{trial.number}"

        epoch_callback = None
        if fresh and not tune_args.no_pruning:
            def epoch_callback(epoch: int, metrics: dict[str, float]) -> None:
                trial.report(metrics["val_acc"], step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        best_val_acc, _ = run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=trial_config,
            args=trial_args,
            save_dir=save_dir,
            experiment_name=experiment_name,
            hyperparams={
                **dataset_hyperparams(dataset, data_dir),
                **dataset_split_params(train_loader, val_loader),
                "optuna_trial": trial.number,
                "optuna_study": study_name,
                "epochs": tune_args.tune_epochs,
                "batch_size": batch_size,
                "T": T,
                "chunk_size": base_config["chunk_size"],
                "pooling_stat": base_config["pooling_stat"],
                "spike_mode": base_config["spike_mode"],
                "lif_backend": lif_backend,
                "hybrid_qkv": hybrid_qkv,
                "attention_mode": attention_mode,
                "num_workers": num_workers,
                "seed": tune_args.seed,
                "train_fraction": tune_args.train_fraction,
                "use_amp": use_amp,
                "device": str(device),
                **params,
                **recipe_hyperparams(trial_config),
            },
            train_one_epoch=train_one_epoch,
            validate=validate,
            run_name_prefix=f"optuna-t{trial.number}",
            project_root=ROOT,
            epoch_callback=epoch_callback,
        )
        return best_val_acc

    if not tune_args.no_resume_interrupted:
        n_resumed = resume_interrupted_trials(
            study,
            study_dir=study_dir,
            tune_epochs=tune_args.tune_epochs,
            run_trial=lambda trial, fresh: run_optuna_trial(
                trial,
                params=params_from_trial(trial.params, base_config, tune_args.tune_arch),
                fresh=fresh,
            ),
        )
        if n_resumed:
            print(f"{n_resumed} essai(s) interrompu(s) repris avant les nouveaux essais.")

    def objective(trial: optuna.Trial) -> float:
        params = suggest_hyperparams(trial, base_config, tune_args.tune_epochs, tune_args.tune_arch)
        return run_optuna_trial(trial, params=params, fresh=True)

    study.optimize(
        objective,
        n_trials=tune_args.n_trials,
        timeout=tune_args.timeout,
        gc_after_trial=True,
    )

    print("\n" + summarize_study(study))
    best_yaml = save_best_params(study_dir / "best_params.yml", study)
    print(f"Meilleurs hyperparamètres écrits → {best_yaml}")


if __name__ == "__main__":
    main()
