#!/bin/bash
# CIFAR-10-DVS — grille 4×2 : sdt (sdt, hybrid_qkv=false)
# 200 epochs, recette STAtten/SDT, seed=42


RUN_NAME="cifar10_dvs_grid_sdt_200ep"
export OUTPUT_DIR="$HOME/internship/snn/HPSTAtten/save/grid_4x2_dvs/sdt_binary"
mkdir -p "$OUTPUT_DIR"

cd HPSTAtten
python -m scripts.train \
    --config config/train_cifar10-dvs.yml \
    --dataset cifar10-dvs \
    --epochs 200 \
    --seed 42 \
    --train-fraction 1.0 \
    --fresh \
    --attention-mode sdt \
    --hybrid-qkv false \
    --save-dir "$OUTPUT_DIR"
