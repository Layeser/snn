#!/bin/bash
# Campagne : 3 jobs OAR (1 bundle par cluster), parallélisme interne sur chicoree/chuc.
#
#   scrip_run/lille/chicoree/bundle_grid4.sh   → 4 grilles CIFAR-10 (gpu=4)
#   scrip_run/lille/chuc/bundle_dvs4.sh        → 4 LR DVS (gpu=4)
#   scrip_run/lyon/sirius/bundle_optuna.sh     → Optuna DVS (gpu=1)
#
# Alternative (9 jobs séparés) : copier les scripts individuels depuis experiences/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE="$ROOT/scrip_run"
EXP="$ROOT/experiences"

copy_bundle() {
    local site=$1 cluster=$2 src_name=$3 dest_name=$4
    local dest_dir="${QUEUE}/${site}/${cluster}"
    mkdir -p "$dest_dir"
    rm -f "${dest_dir}"/*.sh
    cp "${EXP}/bundles/${src_name}" "${dest_dir}/${dest_name}"
    chmod +x "${dest_dir}/${dest_name}"
    echo "  -> scrip_run/${site}/${cluster}/${dest_name}"
}

echo "=== 1 job Lille / chicoree (4 expériences en parallèle, gpu=4) ==="
copy_bundle lille chicoree lille_chicoree_grid4.sh bundle_grid4.sh

echo "=== 1 job Lille / chuc (4 LR DVS en parallèle, gpu=4) ==="
copy_bundle lille chuc lille_chuc_dvs4.sh bundle_dvs4.sh

echo "=== 1 job Lyon / sirius (Optuna DVS, gpu=1) ==="
copy_bundle lyon sirius lyon_sirius_optuna.sh bundle_optuna.sh

echo ""
echo "3 jobs OAR au total. Options par cluster : pilot_grid/cluster_defaults.yaml"
echo "Bundles : # OAR_option -l host=1/gpu=4 sur chicoree et chuc"
echo ""
echo "  echo '{}' > scrip_grid_5000/pilot_grid/run_status.json"
echo "  git push && make pilot-grid"
