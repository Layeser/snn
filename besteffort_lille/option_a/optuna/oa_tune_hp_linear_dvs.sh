#!/bin/bash
# Optuna lite DVS — famille sdt (HP-STAtten-sdt / Hadamard, hybrid)
# Proxy hp_linear → transfert HP vers sdt (binary)
# 1/3 train, 20 essais × 30 ep, batch 8

STUDY="hpstattn-cifar10-dvs-oa-sdt"
SAVE_ROOT="$HOME/internship/snn/HPSTAtten/save/optuna_dvs/sdt"
mkdir -p "$SAVE_ROOT"

cd HPSTAtten
python -m scripts.tune \
    --config config/campaigns/cifar10_dvs_subset_optuna.yml \
    --dataset cifar10-dvs \
    --attention-mode sdt \
    --hybrid-qkv true \
    --batch-size 8 \
    --train-fraction 0.333333 \
    --seed 42 \
    --n-trials 20 \
    --tune-epochs 30 \
    --study-name "$STUDY" \
    --save-dir "$SAVE_ROOT" \
    --tune-aug
