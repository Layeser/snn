#!/usr/bin/env bash
# Variables partagées — Optuna CIFAR-10 D=512 (weekend Sirius, 4 GPU).
set -euo pipefail

SNN_ROOT="${SNN_ROOT:-$HOME/internship/snn}"
HPST="${SNN_ROOT}/HPSTAtten"
PYTHON="${PYTHON:-${SNN_ROOT}/.venv/bin/python}"
CONFIG="${CONFIG:-config/campaigns/cifar10_grid_sota_512.yml}"
DATA_DIR="${DATA_DIR:-${SNN_ROOT}/data}"

N_TRIALS="${N_TRIALS:-30}"
TUNE_EPOCHS="${TUNE_EPOCHS:-30}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.5}"
SEED="${SEED:-42}"

TUNE_EXTRA=(
    --train-fraction "$TRAIN_FRACTION"
    --seed "$SEED"
    --n-trials "$N_TRIALS"
    --tune-epochs "$TUNE_EPOCHS"
    --tune-aug
    --tune-batch
)

run_tune() {
    local study_name="$1"
    local save_root="$2"
    local mlflow_experiment="$3"
    shift 3
    mkdir -p "$save_root"
    cd "$HPST"
    "$PYTHON" -m scripts.tune \
        --config "$CONFIG" \
        --dataset cifar10 \
        --data-dir "$DATA_DIR" \
        --study-name "$study_name" \
        --save-dir "$save_root" \
        --mlflow-experiment "$mlflow_experiment" \
        "${TUNE_EXTRA[@]}" \
        "$@"
}
