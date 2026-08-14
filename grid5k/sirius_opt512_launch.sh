#!/usr/bin/env bash
# Optuna CIFAR-10 D=512 — 4 variantes en parallèle (Sirius, 4 GPU).
# Créer sur Sirius si absent :
#   nano ~/internship/snn/grid5k/sirius_opt512_launch.sh   (coller ce fichier)
#   chmod +x ~/internship/snn/grid5k/sirius_opt512_launch.sh
#
# Usage :
#   cd ~/internship/snn && bash grid5k/sirius_opt512_launch.sh
#   N_TRIALS=40 bash grid5k/sirius_opt512_launch.sh
set -uo pipefail

SNN_ROOT="${SNN_ROOT:-$HOME/internship/snn}"
HPST="$SNN_ROOT/HPSTAtten"
PY="$SNN_ROOT/.venv/bin/python"
DATA="$SNN_ROOT/data"
CFG="$HPST/config/campaigns/cifar10_grid_sota_512.yml"
LOG="$SNN_ROOT/outputs/sirius_opt512"

N_TRIALS="${N_TRIALS:-30}"
TUNE_EPOCHS="${TUNE_EPOCHS:-30}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.5}"

mkdir -p "$LOG" "$HPST/save/optuna_opt512"

if [[ ! -x "$PY" ]]; then
    echo "ERREUR: venv introuvable → $PY" >&2
    echo "  Sur Sirius: cd $SNN_ROOT && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi
if [[ ! -f "$CFG" ]]; then
    echo "ERREUR: config introuvable → $CFG" >&2
    exit 1
fi

cd "$HPST"

TUNE_BASE=(
    "$PY" -m scripts.tune
    --config "$CFG"
    --dataset cifar10
    --data-dir "$DATA"
    --train-fraction "$TRAIN_FRACTION"
    --seed 42
    --n-trials "$N_TRIALS"
    --tune-epochs "$TUNE_EPOCHS"
    --tune-aug
    --tune-batch
)

launch() {
    local name="$1" gpu="$2"
    shift 2
    echo "[GPU $gpu] $name"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        "${TUNE_BASE[@]}" "$@" >"$LOG/${name}.out" 2>"$LOG/${name}.err"
    ) &
    pids+=($!)
    names+=("$name")
}

pids=()
names=()

launch contrast_hyb 0 \
    --attention-mode contrast --hybrid-qkv true --vct-num 16 \
    --study-name hpstattn-cifar10-opt512-contrast-hyb \
    --save-dir save/optuna_opt512/contrast_hyb \
    --mlflow-experiment HP-STAtten-CIFAR10-Opt512-contrast-hyb

launch hp 1 \
    --attention-mode factorized --hybrid-qkv true \
    --study-name hpstattn-cifar10-opt512-hp \
    --save-dir save/optuna_opt512/hp \
    --mlflow-experiment HP-STAtten-CIFAR10-Opt512-hp

launch contrast_sdt_hyb 2 \
    --attention-mode contrast_sdt --hybrid-qkv true --vct-num 16 \
    --study-name hpstattn-cifar10-opt512-contrast-sdt-hyb \
    --save-dir save/optuna_opt512/contrast_sdt_hyb \
    --mlflow-experiment HP-STAtten-CIFAR10-Opt512-contrast-sdt-hyb

launch mk_hgr_triple 3 \
    --attention-mode mk_hgr --hybrid-qkv true --mk-dual-scale false \
    --study-name hpstattn-cifar10-opt512-mk-hgr-triple \
    --save-dir save/optuna_opt512/mk_hgr_triple \
    --mlflow-experiment HP-STAtten-CIFAR10-Opt512-mk-hgr-triple

echo "Logs → $LOG"
fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "OK  ${names[$i]}"
    else
        echo "FAIL ${names[$i]} — tail -30 $LOG/${names[$i]}.err"
        fail=1
    fi
done
exit "$fail"
