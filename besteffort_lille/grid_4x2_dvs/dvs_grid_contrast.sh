#!/bin/bash
# CIFAR-10-DVS — grille 4×2 : contrast (contrast, hybrid_qkv=true)
# 200 epochs, recette STAtten/SDT, seed=42


RUN_NAME="cifar10_dvs_grid_contrast_200ep"
export OUTPUT_DIR="$HOME/internship/snn/HPSTAtten/save/grid_4x2_dvs/hp_contrast_hybrid"
mkdir -p "$OUTPUT_DIR"

cd HPSTAtten
python -m scripts.train \
    --config config/train_cifar10-dvs.yml \
    --dataset cifar10-dvs \
    --epochs 200 \
    --seed 42 \
    --train-fraction 1.0 \
    --fresh \
    --attention-mode contrast \
    --hybrid-qkv true \
    --vct-num 16 \
    --save-dir "$OUTPUT_DIR"
