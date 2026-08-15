#!/usr/bin/env bash
# Phase 2 Opt512 — 3 trains en parallèle (1 GPU / variante).
#
# Prérequis :
#   bash grid5k/export_opt512_campaigns.sh
#
# Usage (Sirius, 3 GPU libres) :
#   bash grid5k/sirius_opt512_train_parallel.sh
#
# Logs : outputs/sirius_opt512_train/*.out
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HPST="${PROJECT_DIR}/HPSTAtten"
PY="${PROJECT_DIR}/.venv/bin/python"
DATA="${PROJECT_DIR}/data"
CAMPAIGN_DIR="${HPST}/config/campaigns/opt512"
GRID_DIR="${HPST}/save/grid/cifar10"
LOG="${PROJECT_DIR}/outputs/sirius_opt512_train"
EPOCHS="${EPOCHS:-310}"

# shellcheck source=opt512_variants.sh
source "${SCRIPT_DIR}/opt512_variants.sh"

mkdir -p "$LOG" "$GRID_DIR"

declare -a GPUS=(0 1 2)
pids=()
names=()

i=0
for entry in "${OPT512_VARIANTS[@]}"; do
    IFS='|' read -r id attn hybrid _study _save_rel mlflow <<< "$entry"
    gpu="${GPUS[$i]}"
    config="${CAMPAIGN_DIR}/cifar10_${id}_best.yml"
    save_dir="${GRID_DIR}/opt512_${id}"

    if [[ ! -f "$config" ]]; then
        echo "ERREUR: $config absent — lancer: bash grid5k/export_opt512_campaigns.sh" >&2
        exit 1
    fi

    extra=(--vct-num 16)
    [[ "$attn" == "factorized" ]] && extra=()

    echo "[GPU $gpu] opt512_$id → $LOG/${id}.out"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
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
            "${extra[@]}"
    ) >"$LOG/${id}.out" 2>"$LOG/${id}.err" &
    pids+=($!)
    names+=("$id")
    i=$((i + 1))
done

fail=0
for j in "${!pids[@]}"; do
    if wait "${pids[$j]}"; then
        echo "OK  ${names[$j]}"
    else
        echo "FAIL ${names[$j]} — tail -30 $LOG/${names[$j]}.err"
        fail=1
    fi
done
exit "$fail"
