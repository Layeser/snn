#!/usr/bin/env bash
# Optuna — HP-STAtten (factorized + hybrid_qkv), D=512, CIFAR-10.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"

run_tune \
    "hpstattn-cifar10-opt512-hp" \
    "$HPST/save/optuna_opt512/hp" \
    "HP-STAtten-CIFAR10-Opt512-hp" \
    --attention-mode factorized \
    --hybrid-qkv true
