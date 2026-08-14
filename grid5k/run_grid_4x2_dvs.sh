#!/usr/bin/env bash
# Lance les 8 ablations 4×2 sur CIFAR-10-DVS (200 ep, séquentiel).
#
# Usage :
#   bash grid5k/run_grid_4x2_dvs.sh
#   bash grid5k/run_grid_4x2_dvs.sh besteffort_lille/grid_4x2_dvs/dvs_grid_hp.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SEQ="${SCRIPT_DIR}/run_sequential.sh"
DVS_DIR="${PROJECT_DIR}/besteffort_lille/grid_4x2_dvs"

# Régénère les scripts si le dossier est absent ou vide
if [[ ! -d "$DVS_DIR" ]] || [[ -z "$(find "$DVS_DIR" -maxdepth 1 -name 'dvs_grid_*.sh' -print -quit)" ]]; then
    bash "${SCRIPT_DIR}/generate_grid_4x2_dvs.sh"
fi

if [[ $# -gt 0 ]]; then
    exec bash "$SEQ" "$@"
fi

mapfile -t SCRIPTS < <(
    find "$DVS_DIR" -maxdepth 1 -name 'dvs_grid_*.sh' -type f | sort
)

if [[ ${#SCRIPTS[@]} -eq 0 ]]; then
    echo "Aucun script dans $DVS_DIR" >&2
    exit 1
fi

exec bash "$SEQ" "${SCRIPTS[@]#$PROJECT_DIR/}"
