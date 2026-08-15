#!/usr/bin/env bash
# Variantes Optuna CIFAR-10-DVS (4 familles attention_mode, hybrid).
# Format: id|attention_mode|hybrid_qkv|study_name|save_rel|mlflow_experiment|extra_cli

OPT_DVS_VARIANTS=(
    "factorized|factorized|true|hpstattn-cifar10-dvs-optuna-factorized|optuna_dvs/factorized|HP-STAtten-CIFAR10-DVS-Optuna-factorized|"
    "contrast|contrast|true|hpstattn-cifar10-dvs-optuna-contrast|optuna_dvs/contrast|HP-STAtten-CIFAR10-DVS-Optuna-contrast|--vct-num 16"
    "sdt|sdt|true|hpstattn-cifar10-dvs-optuna-sdt|optuna_dvs/sdt|HP-STAtten-CIFAR10-DVS-Optuna-sdt|"
    "contrast_sdt|contrast_sdt|true|hpstattn-cifar10-dvs-optuna-contrast-sdt|optuna_dvs/contrast_sdt|HP-STAtten-CIFAR10-DVS-Optuna-contrast-sdt|--vct-num 16"
)
