#!/usr/bin/env python3
"""Dump Optuna study best trial → best_params.yml (format scripts.tune)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

import optuna  # noqa: E402
from optuna_search import optuna_storage_url, save_best_params  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Export Optuna study → best_params.yml")
    p.add_argument("--study", required=True, help="Nom de l'étude Optuna")
    p.add_argument("--output", required=True, help="Fichier best_params.yml de sortie")
    p.add_argument(
        "--hpst-root",
        type=Path,
        default=ROOT / "HPSTAtten",
        help="Racine HPSTAtten (optuna.db)",
    )
    args = p.parse_args()

    storage = optuna_storage_url(args.hpst_root)
    try:
        study = optuna.load_study(study_name=args.study, storage=storage)
    except KeyError as exc:
        raise SystemExit(f"Étude introuvable: {args.study} ({exc})") from exc

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise SystemExit(f"Aucun essai COMPLETE pour {args.study}")

    out = save_best_params(args.output, study)
    print(
        f"{args.study}: trial #{study.best_trial.number} "
        f"val={study.best_value:.4f}% → {out}"
    )


if __name__ == "__main__":
    main()
