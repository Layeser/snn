# Définitions partagées — Option A (6 variantes CIFAR-10)
# shellcheck shell=bash
OPTION_A_VARIANTS=(
    "hp|factorized|true|hpstattn-cifar10-oa-hp|optuna_option_a/hp"
    "statten|factorized|false|hpstattn-cifar10-oa-statten|optuna_option_a/statten"
    "contrast|contrast|true|hpstattn-cifar10-oa-contrast|optuna_option_a/contrast"
    "contrast_binary|contrast|false|hpstattn-cifar10-oa-contrast-bin|optuna_option_a/contrast_binary"
    "hp_linear|sdt|true|hpstattn-cifar10-oa-hp-linear|optuna_option_a/hp_linear"
    "sdt|sdt|false|hpstattn-cifar10-oa-sdt|optuna_option_a/sdt"
)
