#!/usr/bin/env bash
# Optuna lite — 4 études (1 par attention_mode : factorized, contrast, sdt, contrast_sdt).
#
# Usage :
#   bash grid5k/run_option_a_lite_optuna.sh
#   bash grid5k/run_option_a_lite_optuna.sh besteffort_lille/option_a/optuna/oa_tune_statten_sota.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SEQ="${SCRIPT_DIR}/run_sequential.sh"

if [[ $# -gt 0 ]]; then
    exec bash "$SEQ" "$@"
fi

mapfile -t SCRIPTS < <(
    find "$PROJECT_DIR/besteffort_lille/option_a/optuna" -maxdepth 1 -name 'oa_tune_*_sota.sh' -type f | sort
)

if [[ ${#SCRIPTS[@]} -eq 0 ]]; then
    echo "Aucun script oa_tune_*_sota.sh dans besteffort_lille/option_a/optuna/" >&2
    exit 1
fi

exec bash "$SEQ" "${SCRIPTS[@]#$PROJECT_DIR/}"
