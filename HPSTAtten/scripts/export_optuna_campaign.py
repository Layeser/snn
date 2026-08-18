#!/usr/bin/env python3
"""Exporte best_params.yml (Optuna) vers un YAML de campagne train (200 ep)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "common"))
sys.path.insert(0, str(ROOT / "src"))

from utils.config import load_and_validate_config
from utils.config_schema import HPSTATTEN_CONFIG_SCHEMA, validate_hpstattn_config


def _merge_best_params(base: dict[str, Any], best: dict[str, Any], *, epochs: int) -> dict[str, Any]:
    out = dict(base)
    out["epochs"] = epochs
    bp = best.get("best_params", best)

    mapping = {
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "mixup": "mixup",
        "label_smoothing": "label_smoothing",
        "scheduler": "scheduler",
        "warmup_epochs": "warmup_epochs",
        "embed_dim": "emb_dim",
        "depth": "depth",
        "num_heads": "num_heads",
        "random_erasing": "random_erasing",
        "batch_size": "batch_size",
    }
    for src, dst in mapping.items():
        if src in bp:
            out[dst] = bp[src]

    if "rand_augment" in bp:
        out["rand_augment"] = "true" if bp["rand_augment"] else "false"

    if out.get("scheduler") != "cosine":
        out.setdefault("warmup_epochs", int(base.get("warmup_epochs", 0)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Export Optuna best_params → campaign YAML")
    p.add_argument("--best-params", type=str, required=True, help="best_params.yml Optuna")
    p.add_argument("--base-config", type=str, default=str(ROOT / "config" / "train_cifar10.yml"))
    p.add_argument("--output", type=str, required=True, help="Fichier YAML de sortie")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--attention-mode", type=str, default=None, choices=["factorized", "factorized_hgr", "mk_hgr", "sdt", "contrast", "contrast_sdt"])
    p.add_argument("--hybrid-qkv", type=str, default=None, choices=["true", "false"])
    args = p.parse_args()

    with Path(args.best_params).open(encoding="utf-8") as f:
        best_payload = yaml.safe_load(f)

    base = load_and_validate_config(
        Path(args.base_config),
        HPSTATTEN_CONFIG_SCHEMA,
        extra_validators=[validate_hpstattn_config],
    )
    if args.attention_mode is not None:
        base["attention_mode"] = args.attention_mode
    if args.hybrid_qkv is not None:
        base["hybrid_qkv"] = args.hybrid_qkv

    merged = _merge_best_params(base, best_payload, epochs=args.epochs)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header_lines = ["# Campagne — HP exportés depuis Optuna"]
    if "study_name" in best_payload:
        header_lines.append(f"# Study: {best_payload['study_name']}")
    if "best_trial" in best_payload:
        header_lines.append(f"# Best trial: #{best_payload['best_trial']}")
    if "best_value" in best_payload:
        header_lines.append(f"# Best val @ tune: {float(best_payload['best_value']):.4f}%")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(header_lines) + "\n")
        yaml.safe_dump(merged, f, sort_keys=False, allow_unicode=True)
    print(f"Campagne écrite → {out_path}")


if __name__ == "__main__":
    main()
