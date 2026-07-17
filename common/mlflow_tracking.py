"""Tracking MLflow partagé avec reprise de run."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import mlflow

logger = logging.getLogger(__name__)

_ARTIFACT_LOCATION: str | None = None
_ARTIFACT_ROOT: Path | None = None

_MLRUNS_MARKER = "/mlruns"


def _rewrite_mlruns_path(path: str, artifact_root: Path) -> str:
    """Réécrit un chemin d'artefact MLflow vers ``artifact_root``."""
    if not path or _MLRUNS_MARKER not in path:
        return path

    file_prefix = ""
    body = path
    if body.startswith("file://"):
        file_prefix = "file://"
        body = body[len("file://") :]

    idx = body.find(_MLRUNS_MARKER)
    suffix = body[idx + len(_MLRUNS_MARKER) :]
    rewritten = f"{artifact_root.resolve()}{suffix}"
    return f"{file_prefix}{rewritten}"


def _migrate_stale_artifact_locations(db_path: Path, artifact_root: Path) -> int:
    """Corrige les chemins d'artefacts hérités d'un autre utilisateur ou machine."""
    if not db_path.exists():
        return 0

    artifact_root = artifact_root.resolve()
    updated = 0
    conn = sqlite3.connect(db_path)
    try:
        for table, column in (
            ("experiments", "artifact_location"),
            ("runs", "artifact_uri"),
        ):
            rows = conn.execute(f"SELECT rowid, {column} FROM {table}").fetchall()
            for rowid, location in rows:
                if not location:
                    continue
                rewritten = _rewrite_mlruns_path(location, artifact_root)
                if rewritten != location:
                    conn.execute(
                        f"UPDATE {table} SET {column} = ? WHERE rowid = ?",
                        (rewritten, rowid),
                    )
                    updated += 1
        if updated:
            conn.commit()
    finally:
        conn.close()
    return updated


def _rewrite_artifact_path(
    original: str | None, artifact_dir: Path, artifact_dirname: str
) -> str | None:
    """Réécrit un chemin d'artefact d'une autre machine vers le mlruns local.

    Gère les chemins bruts (``/home/x/.../mlruns/...``) et les URI
    (``file:///home/x/.../mlruns/...``). Retourne None si rien à changer.
    """
    if not original:
        return None
    local_plain = str(artifact_dir)
    local_uri = artifact_dir.as_uri()
    if original.startswith(local_plain) or original.startswith(local_uri):
        return None
    marker = f"/{artifact_dirname}"
    idx = original.find(marker)
    if idx == -1:
        return None
    tail = original[idx + len(marker):]  # ex: "/2/<run>/artifacts" ou ""
    base = local_uri if original.startswith("file:") else local_plain
    return base + tail


def _migrate_artifact_paths(
    db_path: Path, artifact_dir: Path, artifact_dirname: str
) -> int:
    """Rend le store MLflow portable : réécrit tous les chemins d'artefact
    (expériences + runs) vers le dossier ``mlruns`` local de la machine.

    Idempotent et silencieux en cas d'échec (ex: schéma inattendu).
    """
    if not db_path.exists():
        return 0
    changed = 0
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        for table, id_col, path_col in (
            ("experiments", "experiment_id", "artifact_location"),
            ("runs", "run_uuid", "artifact_uri"),
        ):
            try:
                rows = cur.execute(f"SELECT {id_col}, {path_col} FROM {table}").fetchall()
            except sqlite3.OperationalError:
                continue
            for rid, path in rows:
                new = _rewrite_artifact_path(path, artifact_dir, artifact_dirname)
                if new is not None and new != path:
                    cur.execute(
                        f"UPDATE {table} SET {path_col}=? WHERE {id_col}=?", (new, rid)
                    )
                    changed += 1
        con.commit()
        con.close()
    except sqlite3.Error:
        return changed
    return changed


def configure_tracking(
    project_root: str | Path,
    *,
    db_name: str = "mlflow.db",
    artifact_dirname: str = "mlruns",
) -> Path:
    """Fixe un store MLflow sqlite portable, versionnable via git.

    - Métriques / params / tags -> ``<project_root>/<db_name>`` (petit, à committer).
    - Artefacts (checkpoints, configs) -> ``<project_root>/<artifact_dirname>``
      (lourd, gardé hors git).

    Le chemin est résolu en absolu : le tracking pointe toujours vers le même
    fichier quel que soit le dossier d'exécution (local, SSH, autre serveur).
    """
    global _ARTIFACT_LOCATION, _ARTIFACT_ROOT
    root = Path(project_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    db_path = root / db_name
    artifact_dir = root / artifact_dirname
    artifact_dir.mkdir(parents=True, exist_ok=True)

    migrated = _migrate_stale_artifact_locations(db_path, artifact_dir)
    if migrated:
        print(
            f"MLflow: {migrated} chemin(s) d'artefact migré(s) vers {artifact_dir}"
        )

    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    _ARTIFACT_ROOT = artifact_dir
    _ARTIFACT_LOCATION = artifact_dir.as_uri()
    # Portabilité inter-machines : réécrit les chemins d'artefact gravés par une
    # autre machine (ex: /home/kasekou/...) vers le mlruns local. Sans ça, MLflow
    # tente d'écrire dans un chemin inexistant -> PermissionError.
    migrated = _migrate_artifact_paths(db_path, artifact_dir, artifact_dirname)
    if migrated:
        print(
            f"MLflow: {migrated} chemin(s) d'artefact réécrit(s) vers {artifact_dir} "
            f"(portabilité inter-machines)"
        )
    return db_path


def setup_experiment(experiment_name: str) -> None:
    if (
        _ARTIFACT_LOCATION is not None
        and mlflow.get_experiment_by_name(experiment_name) is None
    ):
        mlflow.create_experiment(experiment_name, artifact_location=_ARTIFACT_LOCATION)
    mlflow.set_experiment(experiment_name)


def default_run_name(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}"


@contextmanager
def start_training_run(
    experiment_name: str,
    *,
    run_id: str | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> Iterator[str]:
    setup_experiment(experiment_name)
    if run_id:
        with mlflow.start_run(run_id=run_id):
            if tags:
                mlflow.set_tags(tags)
            active = mlflow.active_run()
            assert active is not None
            yield active.info.run_id
    else:
        with mlflow.start_run(run_name=run_name):
            if tags:
                mlflow.set_tags(tags)
            active = mlflow.active_run()
            assert active is not None
            yield active.info.run_id


def log_hyperparameters(params: dict) -> None:
    mlflow.log_params(params)


def log_epoch_metrics(
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
) -> None:
    mlflow.log_metrics(
        {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        },
        step=epoch,
    )


def log_final_metrics(best_val_acc: float, best_epoch: int) -> None:
    mlflow.log_metrics(
        {
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
        }
    )


def _safe_log_artifact(local_path: Path, *, artifact_path: str) -> None:
    try:
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
    except OSError as exc:
        logger.warning(
            "MLflow artifact upload skipped (%s -> %s): %s",
            local_path,
            artifact_path,
            exc,
        )


def log_artifacts(config_path: str | Path, checkpoint_path: str | Path) -> None:
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)

    if config_path.exists():
        _safe_log_artifact(config_path, artifact_path="config")
    if checkpoint_path.exists():
        _safe_log_artifact(checkpoint_path, artifact_path="model")


def log_checkpoint_artifact(checkpoint_path: str | Path, artifact_name: str) -> None:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        _safe_log_artifact(
            checkpoint_path,
            artifact_path=f"checkpoints/{artifact_name}",
        )
