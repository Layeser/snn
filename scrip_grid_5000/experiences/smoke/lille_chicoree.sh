#!/bin/bash
# Smoke test — Lille / chicoree : CIFAR-10, 2 epochs (~5 min GPU)

RUN_NAME="smoke_lille_chicoree_cifar10"
export OUTPUT_DIR="$HOME/internship/snn/HPSTAtten/save/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

cd HPSTAtten
python -m scripts.train \
    --config config/campaigns/smoke_cifar10.yml \
    --dataset cifar10 \
    --epochs 2 \
    --seed 42 \
    --save-dir "$OUTPUT_DIR"
