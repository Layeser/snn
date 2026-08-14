#!/bin/bash
# Optuna lite — famille sdt (HP-STAtten-sdt / Hadamard, hybrid)
# Proxy hp_linear → transfert HP vers sdt (binary)
# 50% train, 20 essais × 30 ep

STUDY="hpstattn-cifar10-sota-sdt"
SAVE_ROOT="$HOME/internship/snn/HPSTAtten/save/optuna_sota/sdt"
mkdir -p "$SAVE_ROOT"

cd HPSTAtten
python -m scripts.tune \
    --config config/campaigns/cifar10_grid_sota.yml \
    --dataset cifar10 \
    --attention-mode sdt \
    --hybrid-qkv true \
    --train-fraction 0.5 \
    --seed 42 \
    --n-trials 20 \
    --tune-epochs 30 \
    --study-name "$STUDY" \
    --save-dir "$SAVE_ROOT" \
    --tune-aug \
    --tune-batch \
    --tune-arch
