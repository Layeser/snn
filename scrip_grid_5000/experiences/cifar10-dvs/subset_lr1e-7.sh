#!/bin/bash
# CIFAR-10-DVS — LR fixe 1e-7, 1/3 train stratifié, diagnostic stabilité

# Pilot_site lille
# OAR_option -p chuc

RUN_NAME="cifar10dvs_subset_lr1e-7"
export OUTPUT_DIR="$HOME/internship/snn/HPSTAtten/save/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

cd HPSTAtten
python -m scripts.train \
    --config config/campaigns/cifar10_dvs_subset_lr_sweep.yml \
    --dataset cifar10-dvs \
    --lr 1e-7 \
    --train-fraction 0.333333 \
    --seed 42 \
    --save-dir "$OUTPUT_DIR"
