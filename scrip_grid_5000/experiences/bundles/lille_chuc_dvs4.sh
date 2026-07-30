#!/bin/bash
# Bundle Lille / chuc — 4 LR DVS en parallèle (1 nœud, 4 GPU).

# OAR_option -l host=1/gpu=4

set -uo pipefail

RUN_NAME="bundle_lille_chuc_dvs4"
export OUTPUT_DIR="${HOME}/internship/snn/HPSTAtten/save/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}/logs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/../../../experiences" && pwd)"

echo "=== Bundle chuc : 4 LR DVS (parallèle) ==="
echo "Logs : ${OUTPUT_DIR}/logs/"

pids=()
fail=0

launch() {
    local gpu=$1
    local script=$2
    local tag=$3
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Lancement GPU ${gpu} : ${script}"
    CUDA_VISIBLE_DEVICES="${gpu}" bash "${EXP_ROOT}/cifar10-dvs/${script}" \
        > "${OUTPUT_DIR}/logs/${tag}.log" 2>&1 &
    pids+=($!)
}

launch 0 subset_lr1e-4.sh lr1e-4
launch 1 subset_lr1e-5.sh lr1e-5
launch 2 subset_lr1e-6.sh lr1e-6
launch 3 subset_lr1e-7.sh lr1e-7

echo "PIDs : ${pids[*]} — attente..."
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        echo "ERREUR : processus ${pid} en échec."
        fail=1
    fi
done

if [ "${fail}" -ne 0 ]; then
    echo "=== Bundle chuc terminé avec ERREUR(S) ==="
    exit 1
fi

echo "=== Bundle chuc terminé avec succès ==="
