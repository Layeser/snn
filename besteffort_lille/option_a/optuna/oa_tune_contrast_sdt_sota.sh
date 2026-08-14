#!/bin/bash
# Optuna lite — famille contrast_sdt (VCA Σ K⊙V, hybrid)
# Proxy → transfert HP vers contrast_sdt_binary
# 50% train, 20 essais × 30 ep

STUDY="hpstattn-cifar10-sota-contrast-sdt"
SAVE_ROOT="$HOME/internship/snn/HPSTAtten/save/optuna_sota/contrast_sdt"
mkdir -p "$SAVE_ROOT"

cd HPSTAtten
python -m scripts.tune \
    --config config/campaigns/cifar10_grid_sota.yml \
    --dataset cifar10 \
    --attention-mode contrast_sdt \
    --hybrid-qkv true \
    --vct-num 16 \
    --train-fraction 0.5 \
    --seed 42 \
    --n-trials 20 \
    --tune-epochs 30 \
    --study-name "$STUDY" \
    --save-dir "$SAVE_ROOT" \
    --tune-aug \
    --tune-batch \
    --tune-arch
