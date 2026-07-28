#!/bin/bash
# CIFAR-10 — grille 2×2 : STAtten (factorized + binaire)
# Hyperparamètres Optuna trial #27 @ 200 epochs, seed=42

# OAR_option -q default
# OAR_option -l host=1/gpu=1

RUN_NAME="cifar10_grid_statten_optuna_t27_200ep"
export OUTPUT_DIR="$HOME/internship/snn/HPSTAtten/save/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

cd HPSTAtten
python -m scripts.train \
    --config config/campaigns/cifar10_optuna_t27.yml \
    --dataset cifar10 \
    --epochs 200 \
    --seed 42 \
    --fresh \
    --attention-mode factorized \
    --hybrid-qkv false \
    --save-dir "$OUTPUT_DIR"
