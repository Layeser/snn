#!/bin/bash
# Prépare 6 smoke tests dans chuc_experiences/ pour tester
# le parallélisme (4 GPU) + la file d'attente (2 jobs suivants).
#
# Usage (sur flille) :
#   make g5k-test-chuc
#   # ou : bash scrip_grid_5000/prepare_chuc_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE="${ROOT}/chuc_experiences"
ARCHIVE="${QUEUE}/archive"

echo "=== Nettoyage file chuc_experiences (smoke) ==="
mkdir -p "$QUEUE" "$ARCHIVE/done" "$ARCHIVE/failed"
find "$QUEUE" -maxdepth 1 -name 'smoke_*.sh' -delete

echo "=== Génération de 6 scripts smoke DVS (1 epoch, seeds distinctes) ==="
for i in $(seq 1 6); do
    seed=$((40 + i))
    dest="${QUEUE}/smoke_${i}.sh"
    cat >"$dest" <<EOF
#!/bin/bash
# Smoke queue test ${i}/6 — chuc (1 epoch DVS, seed ${seed})

RUN_NAME="smoke_chuc_queue_${i}"
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
    echo "  -> chuc_experiences/smoke_${i}.sh  (RUN_NAME=smoke_chuc_queue_${i})"
done

echo ""
echo "File prête : 6 jobs → 4 en parallèle puis 2 en file."
echo ""
echo "Prochaines etapes (mode MANUEL — sur flille) :"
echo "  oarsub -I -p chuc -t day -l host=1/gpu=4,walltime=0:15:00 -q default"
echo "  bash scrip_grid_5000/run_chuc_queue.sh"
echo ""
echo "Pour smoke AUTO sans intervention : make g5k-test-auto-smoke  (depuis votre PC)"
echo ""
echo "Suivi :"
echo "  tail -f outputs/chuc_queue/scheduler.log"
echo "  ls scrip_grid_5000/chuc_experiences/archive/done/"
