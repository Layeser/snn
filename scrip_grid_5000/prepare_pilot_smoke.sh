#!/bin/bash
# Prépare une file smoke test (3 jobs : Lille chicoree/chuc + Lyon sirius).
# Lance ensuite : git push && make pilot-grid-smoke
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE="$ROOT/scrip_run"
EXP="$ROOT/experiences/smoke"
STATE="$ROOT/pilot_grid/run_status.json"

copy_smoke() {
    local site=$1 cluster=$2 script=$3
    local dest_dir="${QUEUE}/${site}/${cluster}"
    mkdir -p "$dest_dir"
    rm -f "${dest_dir}"/*.sh
    cp "${EXP}/${script}" "${dest_dir}/${script}"
    chmod +x "${dest_dir}/${script}"
    echo "  -> scrip_run/${site}/${cluster}/${script}"
}

echo "=== Nettoyage file d'attente (lille/, lyon/) ==="
rm -rf "${QUEUE}/lille" "${QUEUE}/lyon"

echo "=== Smoke Lille / chicoree ==="
copy_smoke lille chicoree lille_chicoree.sh

echo "=== Smoke Lille / chuc ==="
copy_smoke lille chuc lille_chuc.sh

echo "=== Smoke Lyon / sirius ==="
copy_smoke lyon sirius lyon_sirius.sh

echo '{}' > "$STATE"
echo ""
echo "run_status.json reinitialise."
echo ""
echo "Prochaines etapes :"
echo "  git add -A && git commit ... && git push"
echo "  make pilot-grid-smoke          # soumet les 3 jobs (jour, 10 min max)"
echo "  make pilot-grid-smoke-watch    # suivre jusqu'a recuperation"
echo ""
echo "Campagne reelle ensuite :"
echo "  ./prepare_campaign_queue.sh && git push && make pilot-grid"
