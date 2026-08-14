#!/usr/bin/env bash
# Optuna — Contrast-sdt-hyb (contrast_sdt + hybrid_qkv), D=512, CIFAR-10.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"

run_tune \
    "hpstattn-cifar10-opt512-contrast-sdt-hyb" \
    "$HPST/save/optuna_opt512/contrast_sdt_hyb" \
    "HP-STAtten-CIFAR10-Opt512-contrast-sdt-hyb" \
    --attention-mode contrast_sdt \
    --hybrid-qkv true \
    --vct-num 16
