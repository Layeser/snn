#!/bin/bash
# Copie les expériences campagne actuelle dans scrip_run/<site>/<cluster>/.
# Usage futur : cp scrip_grid_5000/experiences/cifar10/mon_run.sh scrip_run/lille/chicoree/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE="$ROOT/scrip_run"
EXP="$ROOT/experiences"

copy_exp() {
    local site=$1 cluster=$2 rel_path=$3
    local dest_dir="${QUEUE}/${site}/${cluster}"
    mkdir -p "$dest_dir"
    cp "${EXP}/${rel_path}" "${dest_dir}/$(basename "${rel_path}")"
    chmod +x "${dest_dir}/$(basename "${rel_path}")"
    echo "  -> scrip_run/${site}/${cluster}/$(basename "${rel_path}")"
}

echo "=== File d'attente Lille / chicoree (4 grilles CIFAR-10) ==="
copy_exp lille chicoree cifar10/grid_statten_optuna_t27_200ep.sh
copy_exp lille chicoree cifar10/grid_hp_optuna_t27_200ep.sh
copy_exp lille chicoree cifar10/grid_hp_linear_optuna_t27_200ep.sh
copy_exp lille chicoree cifar10/grid_sdt_optuna_t27_200ep.sh

echo "=== File d'attente Lille / chuc (4 LR DVS) ==="
copy_exp lille chuc cifar10-dvs/subset_lr1e-4.sh
copy_exp lille chuc cifar10-dvs/subset_lr1e-5.sh
copy_exp lille chuc cifar10-dvs/subset_lr1e-6.sh
copy_exp lille chuc cifar10-dvs/subset_lr1e-7.sh

echo "=== File d'attente Lyon / sirius (Optuna DVS) ==="
copy_exp lyon sirius cifar10-dvs/subset_optuna_20x30.sh

echo ""
echo "Structure : scrip_run/<site>/<cluster>/*.sh"
echo "Options OAR par cluster : pilot_grid/cluster_defaults.yaml"
echo ""
echo "  echo '{}' > scrip_grid_5000/pilot_grid/run_status.json"
echo "  git push && make pilot-grid"
