#!/usr/bin/env bash
# Phase 1 Option A : 8 études Optuna (grille 4×2, 50% train, 20×30 ep) séquentiellement.
#
# Usage :
#   bash grid5k/run_option_a_optuna.sh
#   bash grid5k/run_option_a_optuna.sh besteffort_lille/option_a/optuna/oa_tune_hp.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SEQ="${SCRIPT_DIR}/run_sequential.sh"

if [[ $# -gt 0 ]]; then
    exec bash "$SEQ" "$@"
fi

mapfile -t SCRIPTS < <(
    find "$PROJECT_DIR/besteffort_lille/option_a/optuna" -maxdepth 1 -name 'oa_tune_*.sh' -type f | sort
)

if [[ ${#SCRIPTS[@]} -eq 0 ]]; then
    echo "Aucun script dans besteffort_lille/option_a/optuna/" >&2
    exit 1
fi

exec bash "$SEQ" "${SCRIPTS[@]#$PROJECT_DIR/}"
