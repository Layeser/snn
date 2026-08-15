#!/usr/bin/env bash
# Variantes Opt512 — export Optuna → train 310 ep (3 études OK).
# Format: id|attention_mode|hybrid_qkv|study_name|save_rel|mlflow_experiment

OPT512_VARIANTS=(
    "contrast_hyb|contrast|true|hpstattn-cifar10-opt512-contrast-hyb|optuna_opt512/contrast_hyb|HP-STAtten-CIFAR10-Opt512-contrast-hyb"
    "hp|factorized|true|hpstattn-cifar10-opt512-hp|optuna_opt512/hp|HP-STAtten-CIFAR10-Opt512-hp"
    "contrast_sdt_hyb|contrast_sdt|true|hpstattn-cifar10-opt512-contrast-sdt-hyb|optuna_opt512/contrast_sdt_hyb|HP-STAtten-CIFAR10-Opt512-contrast-sdt-hyb"
)
