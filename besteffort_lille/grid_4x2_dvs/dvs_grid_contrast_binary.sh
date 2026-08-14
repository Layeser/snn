#!/bin/bash
# CIFAR-10-DVS — grille 4×2 : contrast_binary (contrast, hybrid_qkv=false)
# 200 epochs, recette STAtten/SDT, seed=42


RUN_NAME="cifar10_dvs_grid_contrast_binary_200ep"
export OUTPUT_DIR="$HOME/internship/snn/HPSTAtten/save/grid_4x2_dvs/contrast_binary"
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
    --hybrid-qkv false \
    --vct-num 16 \
    --save-dir "$OUTPUT_DIR"
