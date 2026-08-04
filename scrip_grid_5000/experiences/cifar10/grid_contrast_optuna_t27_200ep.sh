#!/bin/bash
# CIFAR-10 — grille 3×2 : HP-STAtten-contrast (contrast + hybride)
# Hyperparamètres Optuna trial #27 @ 200 epochs, seed=42
# VCA-light : AvgPool(Q) + dual e± + différentiel linéaire (sans Softmax)


RUN_NAME="cifar10_grid_contrast_optuna_t27_200ep"
export OUTPUT_DIR="$HOME/internship/snn/HPSTAtten/save/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

cd HPSTAtten
python -m scripts.train \
    --config config/campaigns/cifar10_optuna_t27.yml \
    --dataset cifar10 \
    --epochs 200 \
    --seed 42 \
    --attention-mode contrast \
    --hybrid-qkv true \
    --vct-num 16 \
    --save-dir "$OUTPUT_DIR"
