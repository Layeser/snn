#!/bin/bash
# Bundle Lyon / sirius — Optuna CIFAR-10-DVS (1 GPU, 1 job).

set -euo pipefail

RUN_NAME="bundle_lyon_sirius_optuna"
export OUTPUT_DIR="${HOME}/internship/snn/HPSTAtten/save/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/../../../experiences" && pwd)"

echo "=== Bundle sirius : Optuna DVS ==="
bash "${EXP_ROOT}/cifar10-dvs/subset_optuna_20x30.sh"
