#!/bin/bash
# Prépare 6 smoke tests dans chicoree_experiences/ pour tester
# le parallélisme (4 GPU) + la file d'attente (2 jobs suivants).
#
# Usage (sur flille) :
#   make g5k-test-chicoree
#   # ou : bash scrip_grid_5000/prepare_chicoree_smoke.sh
#
# Puis réservation day 15 min + orchestrateur (voir Notes/gpureser.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE="${ROOT}/chicoree_experiences"
ARCHIVE="${QUEUE}/archive"

echo "=== Nettoyage file chicoree_experiences (smoke) ==="
mkdir -p "$QUEUE" "$ARCHIVE/done" "$ARCHIVE/failed"
find "$QUEUE" -maxdepth 1 -name 'smoke_*.sh' -delete

echo "=== Génération de 6 scripts smoke (1 epoch, seeds distinctes) ==="
for i in $(seq 1 6); do
    seed=$((40 + i))
    dest="${QUEUE}/smoke_${i}.sh"
    cat >"$dest" <<EOF
#!/bin/bash
# Smoke queue test ${i}/6 — chicorée (1 epoch, seed ${seed})

RUN_NAME="smoke_chicoree_queue_${i}"
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
    echo "  -> chicoree_experiences/smoke_${i}.sh  (RUN_NAME=smoke_chicoree_queue_${i})"
done

echo ""
echo "File prête : 6 jobs → 4 en parallèle puis 2 en file."
echo ""
echo "Prochaines etapes (mode MANUEL — sur flille) :"
echo "  oarsub -I -p chicoree -t exotic -t day -l host=1/gpu=4,walltime=0:15:00 -q default"
echo "  bash scrip_grid_5000/run_chicoree_queue.sh"
echo ""
echo "Pour smoke AUTO sans intervention : make g5k-test-auto-smoke  (depuis votre PC)"
echo ""
echo "Suivi :"
echo "  tail -f outputs/chicoree_queue/scheduler.log"
echo "  watch -n 3 nvidia-smi"
echo "  ls scrip_grid_5000/chicoree_experiences/archive/done/"
