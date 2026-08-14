# Grille 4×2 CIFAR-10-DVS — 8 ablations (200 ep, recette STAtten/SDT)
# Format : id|attention_mode|hybrid_qkv|save_subdir
# shellcheck shell=bash
GRID_4X2_DVS_VARIANTS=(
    "hp|factorized|true|hp_factorized_hybrid"
    "statten|factorized|false|statten_factorized_binary"
    "hp_linear|sdt|true|hp_sdt_hybrid"
    "sdt|sdt|false|sdt_binary"
    "contrast|contrast|true|hp_contrast_hybrid"
    "contrast_binary|contrast|false|contrast_binary"
    "contrast_sdt|contrast_sdt|true|hp_contrast_sdt_hybrid"
    "contrast_sdt_binary|contrast_sdt|false|contrast_sdt_binary"
)
