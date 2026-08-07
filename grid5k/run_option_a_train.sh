#!/usr/bin/env bash
# Phase 2 Option A : 6 entraînements 200 ep @ 100% train (HP exportés).
#
# Prérequis : bash grid5k/export_option_a_campaigns.sh
#
# Usage :
#   bash grid5k/run_option_a_train.sh
#   bash grid5k/run_option_a_train.sh besteffort_lille/option_a/train/oa_train_hp.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SEQ="${SCRIPT_DIR}/run_sequential.sh"

if [[ $# -gt 0 ]]; then
    exec bash "$SEQ" "$@"
fi

mapfile -t SCRIPTS < <(
    find "$PROJECT_DIR/besteffort_lille/option_a/train" -maxdepth 1 -name 'oa_train_*.sh' -type f | sort
)

if [[ ${#SCRIPTS[@]} -eq 0 ]]; then
    echo "Aucun script dans besteffort_lille/option_a/train/" >&2
    exit 1
fi

exec bash "$SEQ" "${SCRIPTS[@]#$PROJECT_DIR/}"
