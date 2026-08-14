#!/bin/bash
# Optuna lite DVS — famille contrast_sdt (VCA Σ K⊙V, hybrid)
# Proxy → transfert HP vers contrast_sdt_binary
# 1/3 train, 20 essais × 30 ep, batch 8

STUDY="hpstattn-cifar10-dvs-oa-contrast-sdt"
SAVE_ROOT="$HOME/internship/snn/HPSTAtten/save/optuna_dvs/contrast_sdt"
mkdir -p "$SAVE_ROOT"

cd HPSTAtten
python -m scripts.tune \
    --config config/campaigns/cifar10_dvs_subset_optuna.yml \
    --dataset cifar10-dvs \
    --attention-mode contrast_sdt \
    --hybrid-qkv true \
    --vct-num 16 \
    --batch-size 8 \
    --train-fraction 0.333333 \
    --seed 42 \
    --n-trials 20 \
    --tune-epochs 30 \
    --study-name "$STUDY" \
    --save-dir "$SAVE_ROOT" \
    --tune-aug
