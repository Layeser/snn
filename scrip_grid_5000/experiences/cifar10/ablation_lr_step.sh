#!/bin/bash
# Experience : ablation scheduler LR (StepLR) sur CIFAR-10.

# ---- Ressources OAR ----
# OAR_option -l host=1/gpu=1
# OAR_option -q besteffort

# ---- Identite du run ----
RUN_NAME="cifar10_ablation_lr_step"
export OUTPUT_DIR="$HOME/snn/HPSTAtten/save/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

# ---- Entrainement ----
cd HPSTAtten
python -m scripts.train \
    --config config/ablation/step.yml \
    --dataset cifar10 \
    --save-dir "$OUTPUT_DIR"
