#!/usr/bin/env bash
# Lance les 4 Optuna Opt512 en parallèle (1 GPU / étude).
#
# Usage (depuis snn/, nœud Sirius avec 4 GPU) :
#   bash grid5k/run_sirius_opt512_parallel.sh
#
# Override :
#   N_TRIALS=40 TUNE_EPOCHS=30 bash grid5k/run_sirius_opt512_parallel.sh
#
# Logs : outputs/sirius_opt512/*.out / *.err
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OPT512_DIR="$PROJECT_DIR/besteffort_lille/sirius/opt512/optuna"
LOG_DIR="$PROJECT_DIR/outputs/sirius_opt512"
mkdir -p "$LOG_DIR"

declare -a JOBS=(
    "tune_contrast_hyb.sh:0"
    "tune_hp.sh:1"
    "tune_contrast_sdt_hyb.sh:2"
    "tune_mk_hgr_triple.sh:3"
)

echo "=== Sirius Opt512 — ${#JOBS[@]} études en parallèle ==="
echo "Logs → $LOG_DIR"
echo "N_TRIALS=${N_TRIALS:-30} TUNE_EPOCHS=${TUNE_EPOCHS:-30} TRAIN_FRACTION=${TRAIN_FRACTION:-0.5}"
echo ""

pids=()
for entry in "${JOBS[@]}"; do
    script="${entry%%:*}"
    gpu="${entry##*:}"
    name="${script%.sh}"
    out="$LOG_DIR/${name}.out"
    err="$LOG_DIR/${name}.err"
    echo "[GPU $gpu] $script → $out"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export N_TRIALS="${N_TRIALS:-30}"
        export TUNE_EPOCHS="${TUNE_EPOCHS:-30}"
        export TRAIN_FRACTION="${TRAIN_FRACTION:-0.5}"
        bash "$OPT512_DIR/$script"
    ) >"$out" 2>"$err" &
    pids+=($!)
done

fail=0
for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    entry="${JOBS[$i]}"
    name="${entry%%:*}"
    if wait "$pid"; then
        echo "OK  $name (pid $pid)"
    else
        echo "FAIL $name (pid $pid) — voir $LOG_DIR/${name%.sh}.err"
        fail=1
    fi
done

if [[ "$fail" -ne 0 ]]; then
    echo "Au moins une étude a échoué." >&2
    exit 1
fi
echo "Toutes les études sont terminées."
