#!/usr/bin/env bash
# Prépare le smoke AUTO complet dans scrip_run/ :
#   lille/chicoree  → 6 scripts (4 GPU + file)
#   lille/chuc      → 6 scripts (4 GPU + file)
#   lyon/sirius     → 10 scripts (8 GPU + file)
#
# Puis : make g5k-auto-smoke   ou   make g5k-test-auto-smoke (prépare + soumet)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE="$ROOT/scrip_run"
STATE="$ROOT/pilot_grid/run_status.json"

write_chicoree() {
    local dest_dir="${QUEUE}/lille/chicoree"
    mkdir -p "$dest_dir"
    rm -f "${dest_dir}"/*.sh
    local i seed dest
    for i in $(seq 1 6); do
        seed=$((40 + i))
        dest="${dest_dir}/smoke_${i}.sh"
        cat >"$dest" <<EOF
#!/bin/bash
# Smoke AUTO ${i}/6 — chicorée (1 epoch, seed ${seed})

RUN_NAME="smoke_auto_chicoree_${i}"
export OUTPUT_DIR="\$HOME/internship/snn/HPSTAtten/save/\${RUN_NAME}"
mkdir -p "\$OUTPUT_DIR"

cd HPSTAtten
python -m scripts.train \\
    --config config/campaigns/smoke_cifar10.yml \\
    --dataset cifar10 \\
    --epochs 1 \\
    --seed ${seed} \\
    --save-dir "\$OUTPUT_DIR"
EOF
        chmod +x "$dest"
        echo "  -> scrip_run/lille/chicoree/smoke_${i}.sh"
    done
}

write_chuc() {
    local dest_dir="${QUEUE}/lille/chuc"
    mkdir -p "$dest_dir"
    rm -f "${dest_dir}"/*.sh
    local i seed dest
    for i in $(seq 1 6); do
        seed=$((40 + i))
        dest="${dest_dir}/smoke_${i}.sh"
        cat >"$dest" <<EOF
#!/bin/bash
# Smoke AUTO ${i}/6 — chuc (1 epoch DVS, seed ${seed})

RUN_NAME="smoke_auto_chuc_${i}"
export OUTPUT_DIR="\$HOME/internship/snn/HPSTAtten/save/\${RUN_NAME}"
mkdir -p "\$OUTPUT_DIR"

cd HPSTAtten
python -m scripts.train \\
    --config config/campaigns/smoke_cifar10_dvs.yml \\
    --dataset cifar10-dvs \\
    --epochs 1 \\
    --train-fraction 0.05 \\
    --seed ${seed} \\
    --save-dir "\$OUTPUT_DIR"
EOF
        chmod +x "$dest"
        echo "  -> scrip_run/lille/chuc/smoke_${i}.sh"
    done
}

write_sirius() {
    local dest_dir="${QUEUE}/lyon/sirius"
    mkdir -p "$dest_dir"
    rm -f "${dest_dir}"/*.sh
    local i seed dest
    for i in $(seq 1 10); do
        seed=$((40 + i))
        dest="${dest_dir}/smoke_${i}.sh"
        cat >"$dest" <<EOF
#!/bin/bash
# Smoke AUTO ${i}/10 — sirius (1 epoch DVS, seed ${seed})

RUN_NAME="smoke_auto_sirius_${i}"
export OUTPUT_DIR="\$HOME/internship/snn/HPSTAtten/save/\${RUN_NAME}"
mkdir -p "\$OUTPUT_DIR"

cd HPSTAtten
python -m scripts.train \\
    --config config/campaigns/smoke_cifar10_dvs.yml \\
    --dataset cifar10-dvs \\
    --epochs 1 \\
    --train-fraction 0.05 \\
    --seed ${seed} \\
    --save-dir "\$OUTPUT_DIR"
EOF
        chmod +x "$dest"
        echo "  -> scrip_run/lyon/sirius/smoke_${i}.sh"
    done
}

echo "=== Smoke AUTO — préparation scrip_run/ ==="
echo ""
echo "Lille / chicorée (6 scripts, 4 GPU + file) :"
write_chicoree
echo ""
echo "Lille / chuc (6 scripts, 4 GPU + file) :"
write_chuc
echo ""
echo "Lyon / sirius (10 scripts, 8 GPU + file) :"
write_sirius

echo '{}' >"$STATE"
echo ""
echo "run_status.json réinitialisé."
echo ""
echo "Résumé : 22 scripts, 3 jobs OAR au prochain make g5k-auto-smoke"
echo "  chicorée 6 → 4 parallèles + 2 en file"
echo "  chuc     6 → 4 parallèles + 2 en file"
echo "  sirius  10 → 8 parallèles + 2 en file"
echo ""
echo "Lancement automatique : make g5k-test-auto-smoke"
