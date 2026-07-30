#!/bin/bash
# Bundle Lille / chicoree — 4 grilles CIFAR-10 en parallèle (1 nœud, 4 GPU).
# OAR : 1 job, host=1/gpu=4 (surcharge cluster_defaults via OAR_option ci-dessous).

# OAR_option -l host=1/gpu=4

set -uo pipefail

RUN_NAME="bundle_lille_chicoree_grid4"
export OUTPUT_DIR="${HOME}/internship/snn/HPSTAtten/save/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}/logs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/../../../experiences" && pwd)"

echo "=== Bundle chicoree : 4 grilles CIFAR-10 (parallèle) ==="
echo "Logs : ${OUTPUT_DIR}/logs/"
echo "Experiences : ${EXP_ROOT}/cifar10/"

pids=()
fail=0

launch() {
    local gpu=$1
    local script=$2
    local tag=$3
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Lancement GPU ${gpu} : ${script}"
    CUDA_VISIBLE_DEVICES="${gpu}" bash "${EXP_ROOT}/cifar10/${script}" \
        > "${OUTPUT_DIR}/logs/${tag}.log" 2>&1 &
    pids+=($!)
}

launch 0 grid_statten_optuna_t27_200ep.sh statten
launch 1 grid_hp_optuna_t27_200ep.sh hp
launch 2 grid_hp_linear_optuna_t27_200ep.sh hp_linear
launch 3 grid_sdt_optuna_t27_200ep.sh sdt

echo "PIDs : ${pids[*]} — attente..."
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        echo "ERREUR : processus ${pid} en échec."
        fail=1
    fi
done

if [ "${fail}" -ne 0 ]; then
    echo "=== Bundle chicoree terminé avec ERREUR(S) ==="
    exit 1
fi

echo "=== Bundle chicoree terminé avec succès ==="
