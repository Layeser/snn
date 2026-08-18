#!/usr/bin/env bash
# Variantes Opt512 CIFAR-10 @ D=512.
# Format: id|attention_mode|hybrid_qkv|study_name|save_rel|mlflow_experiment|base_config

OPT512_ALL_VARIANTS=(
    "contrast_hyb|contrast|true|hpstattn-cifar10-opt512-contrast-hyb|optuna_opt512/contrast_hyb|HP-STAtten-CIFAR10-Opt512-contrast-hyb|cifar10_grid_sota_512.yml"
    "hp|factorized|true|hpstattn-cifar10-opt512-hp|optuna_opt512/hp|HP-STAtten-CIFAR10-Opt512-hp|cifar10_grid_sota_512.yml"
    "contrast_sdt_hyb|contrast_sdt|true|hpstattn-cifar10-opt512-contrast-sdt-hyb|optuna_opt512/contrast_sdt_hyb|HP-STAtten-CIFAR10-Opt512-contrast-sdt-hyb|cifar10_grid_sota_512.yml"
    "mk_hgr_triple|mk_hgr|true|hpstattn-cifar10-opt512-mk-hgr-triple|optuna_opt512/mk_hgr_triple|HP-STAtten-CIFAR10-Opt512-mk-hgr-triple|cifar10_opt512_mk_hgr_triple.yml"
)

OPT512_VARIANTS=("${OPT512_ALL_VARIANTS[@]}")
