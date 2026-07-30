#!/bin/bash
# CIFAR-10 — grille 2×2 : HP-STAtten (factorized + hybride)
# Hyperparamètres Optuna trial #27 @ 200 epochs, seed=42


RUN_NAME="cifar10_grid_hp_optuna_t27_200ep"
export OUTPUT_DIR="$HOME/internship/snn/HPSTAtten/save/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

cd HPSTAtten
python -m scripts.train \
    --config config/campaigns/cifar10_optuna_t27.yml \
    --dataset cifar10 \
    --epochs 200 \
    --seed 42 \
    --attention-mode factorized \
    --hybrid-qkv true \
    --save-dir "$OUTPUT_DIR"
