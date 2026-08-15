#!/usr/bin/env bash
# Relance Optuna mk-hgr-triple seul (GPU 3 par défaut).
# MK triple @ D=512 : batch 32 fixe (pas --tune-batch) pour éviter OOM.
#
# Usage :
#   CUDA_VISIBLE_DEVICES=3 bash grid5k/sirius_opt512_mk_only.sh
#   CUDA_VISIBLE_DEVICES=0 N_TRIALS=30 bash grid5k/sirius_opt512_mk_only.sh
set -euo pipefail

SNN_ROOT="${SNN_ROOT:-$HOME/internship/snn}"
HPST="$SNN_ROOT/HPSTAtten"
PY="$SNN_ROOT/.venv/bin/python"
DATA="$SNN_ROOT/data"
CFG="$HPST/config/campaigns/cifar10_opt512_mk_hgr_triple.yml"
LOG="$SNN_ROOT/outputs/sirius_opt512"
GPU="${CUDA_VISIBLE_DEVICES:-3}"

N_TRIALS="${N_TRIALS:-30}"
TUNE_EPOCHS="${TUNE_EPOCHS:-30}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.5}"
MK_BATCH="${MK_BATCH:-32}"

mkdir -p "$LOG" "$HPST/save/optuna_opt512/mk_hgr_triple"
cd "$HPST"

echo "MK-HGR triple Optuna | GPU=$GPU batch=$MK_BATCH | $N_TRIALS trials × $TUNE_EPOCHS ep"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m scripts.tune \
    --config "$CFG" \
    --dataset cifar10 \
    --data-dir "$DATA" \
    --attention-mode mk_hgr \
    --hybrid-qkv true \
    --batch-size "$MK_BATCH" \
    --train-fraction "$TRAIN_FRACTION" \
    --seed 42 \
    --n-trials "$N_TRIALS" \
    --tune-epochs "$TUNE_EPOCHS" \
    --tune-aug \
    --study-name hpstattn-cifar10-opt512-mk-hgr-triple \
    --save-dir save/optuna_opt512/mk_hgr_triple \
    --mlflow-experiment HP-STAtten-CIFAR10-Opt512-mk-hgr-triple \
    2>&1 | tee "$LOG/mk_hgr_triple_retry.log"
