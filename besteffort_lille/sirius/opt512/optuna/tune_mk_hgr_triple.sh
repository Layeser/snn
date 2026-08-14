#!/usr/bin/env bash
# Optuna — MK-HGR triple (3+7+15), hybrid_qkv, D=512, CIFAR-10.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"

run_tune \
    "hpstattn-cifar10-opt512-mk-hgr-triple" \
    "$HPST/save/optuna_opt512/mk_hgr_triple" \
    "HP-STAtten-CIFAR10-Opt512-mk-hgr-triple" \
    --attention-mode mk_hgr \
    --hybrid-qkv true \
    --mk-dual-scale false
