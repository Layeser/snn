"""Boucle d'entraînement partagée avec reprise et sauvegarde robuste."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import mlflow
import torch

from checkpointing import (
    BEST_CHECKPOINT,
    LAST_CHECKPOINT,
    CheckpointPaths,
    PreemptionHandler,
    append_metrics_jsonl,
    atomic_torch_save,
    build_checkpoint_state,
    load_checkpoint,
    resolve_resume_path,
    restore_training_state,
    setup_train_logger,
    warn_checkpoint_mismatch,
)
from mlflow_tracking import (
    default_run_name,
    log_checkpoint_artifact,
    log_epoch_metrics,
    log_final_metrics,
    log_artifacts,
    log_hyperparameters,
    start_training_run,
)


TrainFn = Callable[..., tuple[float, float]]
ValidateFn = Callable[..., tuple[float, float]]


def run_training(
    *,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device: torch.device,
    config: dict[str, Any],
    args: argparse.Namespace,
    save_dir: Path,
    experiment_name: str,
    hyperparams: dict[str, Any],
    train_one_epoch: TrainFn,
    validate: ValidateFn,
    mismatch_keys: tuple[str, ...] = (),
    run_name_prefix: str = "train",
) -> tuple[float, int]:
    paths = CheckpointPaths.from_dir(save_dir)
    logger = setup_train_logger(paths.save_dir)

    resume_path = None
    if getattr(args, "resume_path", None):
        resume_path = Path(args.resume_path)
        if not resume_path.is_absolute() and (paths.save_dir / resume_path).exists():
            resume_path = paths.save_dir / resume_path
    elif not getattr(args, "fresh", False):
        resume_path = resolve_resume_path(
            paths.save_dir,
            getattr(args, "resume", "auto"),
            fresh=False,
        )

    start_epoch = 0
    best_val_acc = 0.0
    best_epoch = 0
    mlflow_run_id: str | None = None

    if resume_path is not None:
        print(f"Reprise depuis checkpoint: {resume_path}")
        logger.info("Reprise depuis checkpoint: %s", resume_path)
        checkpoint = load_checkpoint(resume_path, map_location=device)
        if mismatch_keys:
            warn_checkpoint_mismatch(checkpoint, args, mismatch_keys)
        start_epoch, best_val_acc, best_epoch, mlflow_run_id = restore_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        print(
            f"  → epoch {start_epoch}/{args.epochs} | "
            f"best val acc {best_val_acc:.2f}% (epoch {best_epoch})"
        )
        logger.info(
            "État restauré: epoch=%s best_val_acc=%.4f best_epoch=%s",
            start_epoch,
            best_val_acc,
            best_epoch,
        )
    else:
        print("Démarrage from scratch (aucun checkpoint utilisé).")
        logger.info("Démarrage from scratch.")

    if start_epoch >= args.epochs:
        print(
            f"Entraînement déjà terminé ({start_epoch}/{args.epochs} epochs). "
            f"Meilleure val accuracy: {best_val_acc:.2f}% (epoch {best_epoch})"
        )
        return best_val_acc, best_epoch

    resume_tag = "true" if resume_path is not None else "false"
    tags = {"resumed": resume_tag}

    latest_state_holder: dict[str, Any] = {"state": None}

    def emergency_save() -> None:
        state = latest_state_holder.get("state")
        if state is None:
            return
        atomic_torch_save(state, paths.last)
        logger.warning("Checkpoint d'urgence sauvegardé → %s", paths.last)
        try:
            log_checkpoint_artifact(paths.last, LAST_CHECKPOINT)
        except Exception:
            pass

    PreemptionHandler(save_fn=emergency_save)

    with start_training_run(
        experiment_name,
        run_id=mlflow_run_id,
        run_name=None if mlflow_run_id else default_run_name(run_name_prefix),
        tags=tags,
    ) as active_run_id:
        if start_epoch == 0:
            log_hyperparameters(
                {
                    **hyperparams,
                    "resume_mode": "fresh" if getattr(args, "fresh", False) else getattr(args, "resume", "auto"),
                }
            )
        else:
            mlflow.set_tag("resumed_from_epoch", str(start_epoch))

        for epoch in range(start_epoch + 1, args.epochs + 1):
            print(f"\n--- Epoch {epoch}/{args.epochs} ---")
            logger.info("Epoch %s/%s — début", epoch, args.epochs)

            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                mixup_alpha=config["mixup"],
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            if scheduler is not None:
                scheduler.step()

            log_epoch_metrics(epoch, train_loss, train_acc, val_loss, val_acc)

            summary = (
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.2f}% | "
                f"val loss {val_loss:.4f} acc {val_acc:.2f}%"
            )
            print(summary)
            logger.info(summary)

            append_metrics_jsonl(
                paths.save_dir,
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_acc": best_val_acc,
                    "best_epoch": best_epoch,
                },
            )

            state = build_checkpoint_state(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_acc=best_val_acc,
                best_epoch=best_epoch,
                val_acc=val_acc,
                val_loss=val_loss,
                train_loss=train_loss,
                train_acc=train_acc,
                config=config,
                args=args,
                mlflow_run_id=active_run_id,
            )
            latest_state_holder["state"] = state
            atomic_torch_save(state, paths.last)
            log_checkpoint_artifact(paths.last, LAST_CHECKPOINT)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                state["best_val_acc"] = best_val_acc
                state["best_epoch"] = best_epoch
                atomic_torch_save(state, paths.best)
                log_checkpoint_artifact(paths.best, BEST_CHECKPOINT)
                print(f"  → meilleur modèle sauvegardé ({val_acc:.2f}%) → {paths.best}")
                logger.info("Nouveau best checkpoint: %.4f%% epoch %s", val_acc, epoch)

        log_final_metrics(best_val_acc, best_epoch)
        log_artifacts(args.config, paths.best)

        print(
            f"Entraînement terminé. Meilleure val accuracy: {best_val_acc:.2f}% (epoch {best_epoch})"
        )
        print(f"Checkpoints: {paths.last} (dernier) | {paths.best} (meilleur)")
        print(f"MLflow experiment: {experiment_name}")
        logger.info(
            "Entraînement terminé. best_val_acc=%.4f best_epoch=%s",
            best_val_acc,
            best_epoch,
        )

    return best_val_acc, best_epoch
