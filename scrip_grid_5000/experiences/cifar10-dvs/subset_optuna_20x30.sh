#!/bin/bash
# CIFAR-10-DVS — Optuna AdamW + cosine, 20 essais × 30 epochs, 1/3 train stratifié

# Pilot_site lyon
# OAR_option -p sirius
# OAR_option -t exotic
# OAR_option -t night

RUN_NAME="cifar10dvs_subset_optuna_20x30"
export OUTPUT_DIR="$HOME/internship/snn/HPSTAtten/save/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

cd HPSTAtten
python -m scripts.tune \
    --config config/campaigns/cifar10_dvs_subset_optuna.yml \
    --batch-size 8 \
    --dataset cifar10-dvs \
    --n-trials 20 \
    --tune-epochs 30 \
    --train-fraction 0.333333 \
    --seed 42 \
    --study-name hpstattn-cifar10-dvs-subset-third \
    --save-dir "$OUTPUT_DIR"
