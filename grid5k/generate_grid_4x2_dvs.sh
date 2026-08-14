#!/usr/bin/env bash
# Génère les scripts grid_4x2 DVS depuis grid_4x2_dvs_variants.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="${PROJECT_DIR}/besteffort_lille/grid_4x2_dvs"
TEMPLATE_HEADER='#!/bin/bash
# CIFAR-10-DVS — grille 4×2 : {{LABEL}}
# 200 epochs, recette STAtten/SDT (train_cifar10-dvs.yml), seed=42
'

# shellcheck source=grid_4x2_dvs_variants.sh
source "${SCRIPT_DIR}/grid_4x2_dvs_variants.sh"

mkdir -p "$OUT_DIR"

for entry in "${GRID_4X2_DVS_VARIANTS[@]}"; do
    IFS='|' read -r id attn hybrid save_sub <<< "$entry"
    script="${OUT_DIR}/dvs_grid_${id}.sh"
    extra_vct=""
    if [[ "$attn" == contrast* ]]; then
        extra_vct=$'\n    --vct-num 16 \\'
    fi

    cat > "$script" <<EOF
#!/bin/bash
# CIFAR-10-DVS — grille 4×2 : ${id} (${attn}, hybrid_qkv=${hybrid})
# 200 epochs, recette STAtten/SDT, seed=42


RUN_NAME="cifar10_dvs_grid_${id}_200ep"
export OUTPUT_DIR="\$HOME/internship/snn/HPSTAtten/save/grid_4x2_dvs/${save_sub}"
mkdir -p "\$OUTPUT_DIR"

cd HPSTAtten
python -m scripts.train \\
    --config config/train_cifar10-dvs.yml \\
    --dataset cifar10-dvs \\
    --epochs 200 \\
    --seed 42 \\
    --train-fraction 1.0 \\
    --fresh \\
    --attention-mode ${attn} \\
    --hybrid-qkv ${hybrid} \\${extra_vct}
    --save-dir "\$OUTPUT_DIR"
EOF
    chmod +x "$script"
    echo "Wrote $script"
done

echo "Done — ${#GRID_4X2_DVS_VARIANTS[@]} scripts in $OUT_DIR"
