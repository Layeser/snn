#!/usr/bin/env bash
# Phase 2 Opt512 — train 310 ep, 100 % train, HP exportés depuis Optuna.
#
# Prérequis : bash grid5k/export_opt512_campaigns.sh
#
# Usage :
#   bash grid5k/run_opt512_train.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HPST="${PROJECT_DIR}/HPSTAtten"
PY="${PROJECT_DIR}/.venv/bin/python"
DATA="${PROJECT_DIR}/data"
CAMPAIGN_DIR="${HPST}/config/campaigns/opt512"
GRID_DIR="${HPST}/save/grid/cifar10"
EPOCHS="${EPOCHS:-310}"

# shellcheck source=opt512_variants.sh
source "${SCRIPT_DIR}/opt512_variants.sh"

for entry in "${OPT512_VARIANTS[@]}"; do
    IFS='|' read -r id attn hybrid _study _save_rel mlflow <<< "$entry"

    config="${CAMPAIGN_DIR}/cifar10_${id}_best.yml"
    save_dir="${GRID_DIR}/opt512_${id}"
    extra_args=(--vct-num 16)
    if [[ "$attn" == "factorized" ]]; then
        extra_args=()
    fi

    if [[ ! -f "$config" ]]; then
        echo "ERREUR: $config absent — lancer: bash grid5k/export_opt512_campaigns.sh" >&2
        exit 1
    fi

    echo "=== Train opt512/$id | ${EPOCHS} ep | 100% train ==="
    cd "$HPST"
    "$PY" -m scripts.train \
        --config "$config" \
        --dataset cifar10 \
        --data-dir "$DATA" \
        --epochs "$EPOCHS" \
        --train-fraction 1.0 \
        --seed 42 \
        --fresh \
        --save-dir "$save_dir" \
        --mlflow-experiment "$mlflow" \
        --attention-mode "$attn" \
        --hybrid-qkv "$hybrid" \
        "${extra_args[@]}"
done

echo "Entraînements Opt512 terminés."
