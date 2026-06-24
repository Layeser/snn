#!/usr/bin/env python3
"""Télécharge les jeux de données SNN (CIFAR-10 / CIFAR-10-DVS)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SNN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = SNN_ROOT / "data"

sys.path.insert(0, str(SNN_ROOT / "common"))

from data_download import (
    download_cifar10_dvs_archives,
    ensure_cifar10,
    is_cifar10_dvs_archives_ready,
    is_cifar10_ready,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Téléchargement accéléré des datasets SNN")
    p.add_argument(
        "dataset",
        nargs="?",
        default="all",
        choices=["cifar10", "cifar10-dvs", "all"],
        help="Jeu à télécharger (défaut: all)",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Répertoire racine des données (défaut: {DEFAULT_DATA_DIR})",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Téléchargements parallèles pour CIFAR-10-DVS (défaut: 4)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("cifar10", "all"):
        ensure_cifar10(data_dir)
        assert is_cifar10_ready(data_dir)

    if args.dataset in ("cifar10-dvs", "all"):
        dvs_root = data_dir / "CIFAR10DVS"
        download_cifar10_dvs_archives(dvs_root, max_workers=args.workers)
        assert is_cifar10_dvs_archives_ready(dvs_root)

    print(f"\nTerminé. Données dans {data_dir}")


if __name__ == "__main__":
    main()
