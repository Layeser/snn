#!/bin/bash
# Experience : entrainement de reference sur CIFAR-10-DVS (T=16, tet_loss).

# ---- Ressources OAR ----
# OAR_option -l host=1/gpu=1
# OAR_option -q besteffort

# ---- Identite du run ----
RUN_NAME="cifar10-dvs_train_ref"
export OUTPUT_DIR="$HOME/snn/HPSTAtten/save/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

# ---- Entrainement ----
cd HPSTAtten
python -m scripts.train \
    --config config/train_cifar10-dvs.yml \
    --dataset cifar10-dvs \
    --save-dir "$OUTPUT_DIR"
