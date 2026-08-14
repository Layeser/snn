#!/usr/bin/env bash
# Optuna — Contrast-hyb (contrast + hybrid_qkv), D=512, CIFAR-10.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"

run_tune \
    "hpstattn-cifar10-opt512-contrast-hyb" \
    "$HPST/save/optuna_opt512/contrast_hyb" \
    "HP-STAtten-CIFAR10-Opt512-contrast-hyb" \
    --attention-mode contrast \
    --hybrid-qkv true \
    --vct-num 16
