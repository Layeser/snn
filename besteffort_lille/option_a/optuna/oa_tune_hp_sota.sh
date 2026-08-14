#!/bin/bash
# Optuna lite — famille factorized (HP-STAtten hybrid)
# Proxy → transfert HP vers statten (binary)
# 50% train, 20 essais × 30 ep

STUDY="hpstattn-cifar10-sota-factorized"
SAVE_ROOT="$HOME/internship/snn/HPSTAtten/save/optuna_sota/factorized"
mkdir -p "$SAVE_ROOT"

cd HPSTAtten
python -m scripts.tune \
    --config config/campaigns/cifar10_grid_sota.yml \
    --dataset cifar10 \
    --attention-mode factorized \
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
