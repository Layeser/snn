#!/bin/bash
# Smoke test — Lille / chuc : CIFAR-10-DVS, 2 epochs, 5 % du train

RUN_NAME="smoke_lille_chuc_cifar10dvs"
export OUTPUT_DIR="$HOME/internship/snn/HPSTAtten/save/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

cd HPSTAtten
python -m scripts.train \
    --config config/campaigns/smoke_cifar10_dvs.yml \
    --dataset cifar10-dvs \
    --epochs 2 \
    --train-fraction 0.05 \
    --seed 42 \
    --save-dir "$OUTPUT_DIR"
