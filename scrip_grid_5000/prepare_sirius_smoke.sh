#!/bin/bash
# Prépare 10 smoke tests dans sirius_experiences/ :
# 8 en parallèle (8× A100) + 2 en file d'attente.
#
# Usage (sur flyon ou en local avant git push) :
#   ./scrip_grid_5000/prepare_sirius_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE="${ROOT}/sirius_experiences"
ARCHIVE="${QUEUE}/archive"

echo "=== Nettoyage file sirius_experiences (smoke) ==="
mkdir -p "$QUEUE" "$ARCHIVE/done" "$ARCHIVE/failed"
find "$QUEUE" -maxdepth 1 -name 'smoke_*.sh' -delete

echo "=== Génération de 10 scripts smoke DVS (1 epoch, seeds distinctes) ==="
for i in $(seq 1 10); do
    seed=$((40 + i))
    dest="${QUEUE}/smoke_${i}.sh"
    cat >"$dest" <<EOF
#!/bin/bash
# Smoke queue test ${i}/10 — sirius Lyon (1 epoch DVS, seed ${seed})

RUN_NAME="smoke_sirius_queue_${i}"
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
    echo "  -> sirius_experiences/smoke_${i}.sh  (RUN_NAME=smoke_sirius_queue_${i})"
done

echo ""
echo "File prête : 10 jobs → 8 en parallèle puis 2 en file."
echo ""
echo "Prochaines étapes sur flyon (Lyon) :"
echo "  1) git pull"
echo "  2) Réservation day (voir Notes/gpures_sirius.md)"
echo "  3) bash scrip_grid_5000/run_sirius_queue.sh --job-id <JOB_ID>"
echo ""
echo "Suivi :"
echo "  tail -f outputs/sirius_queue/scheduler.log"
echo "  watch -n 3 nvidia-smi"
