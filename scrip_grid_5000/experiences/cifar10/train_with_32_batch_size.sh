#!/bin/bash
# Experience : ablation scheduler LR (StepLR) sur CIFAR-10.

# ---- Ressources OAR ----
OAR_option -q default
OAR_option -l host=1/gpu=1

# ---- Identite du run ----
RUN_NAME="cifar10_32_batch_size"
export OUTPUT_DIR="$HOME/internship/snn/HPSTAtten/save/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

# ---- Entrainement ----
cd HPSTAtten
python -m scripts.train \
    --config config/train_cifar10.yml \
    --dataset cifar10 \
    --save-dir "$OUTPUT_DIR"
