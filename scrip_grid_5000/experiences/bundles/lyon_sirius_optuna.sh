#!/bin/bash
# Bundle Lyon / sirius — Optuna CIFAR-10-DVS (1 GPU).
# Réservation OAR : host=1/gpu=1, exotic + night.

# Pilot_site lyon
# OAR_option -p sirius
# OAR_option -t exotic
# OAR_option -t night
# OAR_option -l host=1/gpu=1

set -euo pipefail

RUN_NAME="bundle_lyon_sirius_optuna"
export OUTPUT_DIR="${HOME}/internship/snn/HPSTAtten/save/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(dirname "${BUNDLE_DIR}")"

echo "=== Bundle sirius : Optuna DVS ==="
bash "${EXP_ROOT}/cifar10-dvs/subset_optuna_20x30.sh"
