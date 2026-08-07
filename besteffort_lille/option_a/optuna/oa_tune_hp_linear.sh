#!/bin/bash
# Option A phase 1 — Optuna hp_linear (sdt, hybrid_qkv=true)
# 50% train, val 100%, 20 essais × 30 epochs, seed=42


STUDY="hpstattn-cifar10-oa-hp-linear"
SAVE_ROOT="$HOME/internship/snn/HPSTAtten/save/optuna_option_a/hp_linear"
mkdir -p "$SAVE_ROOT"

cd HPSTAtten
python -m scripts.tune \
    --config config/train_cifar10.yml \
    --dataset cifar10 \
    --attention-mode sdt \
    --hybrid-qkv true \
    --train-fraction 0.5 \
    --seed 42 \
    --n-trials 20 \
    --tune-epochs 30 \
    --study-name "$STUDY" \
    --save-dir "$SAVE_ROOT"
