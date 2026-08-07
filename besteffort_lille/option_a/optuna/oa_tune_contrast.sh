#!/bin/bash
# Option A phase 1 — Optuna contrast (contrast, hybrid_qkv=true)
# 50% train, val 100%, 20 essais × 30 epochs, seed=42


STUDY="hpstattn-cifar10-oa-contrast"
SAVE_ROOT="$HOME/internship/snn/HPSTAtten/save/optuna_option_a/contrast"
mkdir -p "$SAVE_ROOT"

cd HPSTAtten
python -m scripts.tune \
    --config config/train_cifar10.yml \
    --dataset cifar10 \
    --attention-mode contrast \
    --hybrid-qkv true \
    --vct-num 16 \
    --train-fraction 0.5 \
    --seed 42 \
    --n-trials 20 \
    --tune-epochs 30 \
    --study-name "$STUDY" \
    --save-dir "$SAVE_ROOT"
