# Définitions — Optuna lite 4×1 (une étude par attention_mode, proxy hybrid)
# shellcheck shell=bash
#
# Chaque famille : Optuna sur hybrid_qkv=true → transfert vers la cellule binaire.

OPTION_A_PROXY_VARIANTS=(
    "hp|factorized|true|hpstattn-cifar10-sota-factorized|optuna_sota/factorized"
    "contrast|contrast|true|hpstattn-cifar10-sota-contrast|optuna_sota/contrast"
    "hp_linear|sdt|true|hpstattn-cifar10-sota-sdt|optuna_sota/sdt"
    "contrast_sdt|contrast_sdt|true|hpstattn-cifar10-sota-contrast-sdt|optuna_sota/contrast_sdt"
)

# Transfert HP (1 cible binaire par proxy)
OPTION_A_PROXY_TRANSFER=(
    "hp|statten"
    "contrast|contrast_binary"
    "hp_linear|sdt"
    "contrast_sdt|contrast_sdt_binary"
)
