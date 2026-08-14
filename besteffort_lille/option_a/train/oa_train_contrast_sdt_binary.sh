#!/bin/bash
# Option A phase 2 — contrast_sdt binary @ 200 epochs, 100% train (HP exportés)


RUN_NAME="cifar10_oa_contrast_sdt_binary_200ep"
export OUTPUT_DIR="$HOME/internship/snn/HPSTAtten/save/$RUN_NAME"
CONFIG="$HOME/internship/snn/HPSTAtten/config/campaigns/option_a/cifar10_contrast_sdt_binary_best.yml"
mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config absent: $CONFIG" >&2
    echo "Lancer: bash grid5k/export_option_a_campaigns.sh" >&2
    exit 1
fi

cd HPSTAtten
python -m scripts.train \
    --config "$CONFIG" \
    --dataset cifar10 \
    --epochs 200 \
    --seed 42 \
    --train-fraction 1.0 \
    --fresh \
    --attention-mode contrast_sdt \
    --hybrid-qkv false \
    --vct-num 16 \
    --save-dir "$OUTPUT_DIR"
