"""Checkpointing robuste pour entraînements interruptibles (ex. Grid5000 besteffort)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

LAST_CHECKPOINT = "last.pt"
BEST_CHECKPOINT = "best.pt"


def atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _as_torch_byte_tensor(raw: Any) -> torch.Tensor:
    """Convertit un état RNG sérialisé en ByteTensor CPU (requis par torch.set_rng_state)."""
    if isinstance(raw, torch.Tensor):
        return raw.detach().cpu().to(torch.uint8).clone()
    return torch.as_tensor(raw, dtype=torch.uint8)


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(_as_torch_byte_tensor(state["torch"]))
        if "torch_cuda" in state and torch.cuda.is_available():
            cuda_states = state["torch_cuda"]
            if isinstance(cuda_states, (list, tuple)):
                cuda_states = [_as_torch_byte_tensor(s) for s in cuda_states]
            else:
                cuda_states = _as_torch_byte_tensor(cuda_states)
            torch.cuda.set_rng_state_all(cuda_states)
    except Exception as exc:
        print(f"  ⚠ état RNG non restauré (reprise continue): {exc}")


def build_checkpoint_state(
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    best_val_acc: float,
    best_epoch: int,
    val_acc: float,
    val_loss: float,
    train_loss: float,
    train_acc: float,
    config: dict[str, Any],
    args: argparse.Namespace | dict[str, Any],
    mlflow_run_id: str | None = None,
) -> dict[str, Any]:
    args_dict = vars(args) if hasattr(args, "__dict__") else dict(args)
    state: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "config": config,
        "args": args_dict,
        "rng_state": capture_rng_state(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if mlflow_run_id is not None:
        state["mlflow_run_id"] = mlflow_run_id
    return state


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def restore_training_state(
    checkpoint: dict[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    device: torch.device,
) -> tuple[int, float, int, str | None]:
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        else:
            start_epoch = int(checkpoint.get("epoch", 0))
            for _ in range(start_epoch):
                scheduler.step()

    restore_rng_state(checkpoint.get("rng_state"))

    start_epoch = int(checkpoint.get("epoch", 0))
    best_val_acc = float(checkpoint.get("best_val_acc", checkpoint.get("val_acc", 0.0)))
    best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0)))
    mlflow_run_id = checkpoint.get("mlflow_run_id")
    return start_epoch, best_val_acc, best_epoch, mlflow_run_id


def resolve_resume_path(save_dir: Path, resume: str, fresh: bool) -> Path | None:
    if fresh or resume == "none":
        return None
    if resume == "auto":
        last_path = save_dir / LAST_CHECKPOINT
        return last_path if last_path.exists() else None
    path = Path(resume)
    if not path.is_absolute():
        candidate = save_dir / path
        if candidate.exists():
            path = candidate
    return path


def warn_checkpoint_mismatch(
    checkpoint: dict[str, Any],
    current_args: argparse.Namespace,
    keys: tuple[str, ...],
) -> None:
    saved_args = checkpoint.get("args") or {}
    for key in keys:
        saved = saved_args.get(key)
        current = getattr(current_args, key, None)
        if saved is not None and current is not None and saved != current:
            print(
                f"  ⚠ checkpoint/CLI mismatch pour '{key}': "
                f"checkpoint={saved!r}, actuel={current!r}"
            )


@dataclass
class CheckpointPaths:
    save_dir: Path
    last: Path
    best: Path

    @classmethod
    def from_dir(cls, save_dir: Path) -> CheckpointPaths:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        return cls(save_dir=save_dir, last=save_dir / LAST_CHECKPOINT, best=save_dir / BEST_CHECKPOINT)


class PreemptionHandler:
    """Sauvegarde last.pt sur SIGTERM/SIGINT (oarsub / Ctrl+C)."""

    def __init__(self, save_fn: Callable[[], None]) -> None:
        self._save_fn = save_fn
        self._triggered = False
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum: int, _frame: Any) -> None:
        if self._triggered:
            sys.exit(128 + signum)
        self._triggered = True
        sig_name = signal.Signals(signum).name
        print(f"\n[{sig_name}] Interruption détectée — sauvegarde du checkpoint en cours...")
        try:
            self._save_fn()
            print(f"Checkpoint sauvegardé. Relance avec --resume auto dans {LAST_CHECKPOINT}.")
        except Exception as exc:
            print(f"Échec de la sauvegarde d'urgence: {exc}", file=sys.stderr)
        sys.exit(128 + signum)


def setup_train_logger(save_dir: Path) -> logging.Logger:
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"snn.train.{save_dir.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(save_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def append_metrics_jsonl(save_dir: Path, record: dict[str, Any]) -> None:
    path = save_dir / "metrics.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
