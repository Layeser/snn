#!/usr/bin/env bash
# Optuna CIFAR-10-DVS — 4 familles en parallèle (Sirius, 4 GPU).
#
# Protocole (aligné option_a lite DVS) :
#   config/campaigns/cifar10_dvs_subset_optuna.yml
#   1/3 train stratifié, batch 8 fixe, T=16, tet_loss, 20×30 ep (override N_TRIALS)
#   1 dossier MLflow par variante
#
# Usage :
#   cd ~/internship/snn && bash grid5k/sirius_opt_dvs_launch.sh
#   N_TRIALS=30 bash grid5k/sirius_opt_dvs_launch.sh
#
# Prérequis données :
#   ls ~/internship/snn/data/cifar10dvs/  (ou téléchargement auto au 1er run)
#
# Logs : outputs/sirius_opt_dvs/*.out / *.err
set -uo pipefail

SNN_ROOT="${SNN_ROOT:-$HOME/internship/snn}"
HPST="$SNN_ROOT/HPSTAtten"
PY="$SNN_ROOT/.venv/bin/python"
DATA="$SNN_ROOT/data"
CFG="$HPST/config/campaigns/cifar10_dvs_subset_optuna.yml"
LOG="$SNN_ROOT/outputs/sirius_opt_dvs"

N_TRIALS="${N_TRIALS:-20}"
TUNE_EPOCHS="${TUNE_EPOCHS:-30}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.333333}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# shellcheck source=opt_dvs_variants.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/opt_dvs_variants.sh"

mkdir -p "$LOG" "$HPST/save/optuna_dvs"

if [[ ! -x "$PY" ]]; then
    echo "ERREUR: venv introuvable → $PY" >&2
    exit 1
fi
if [[ ! -f "$CFG" ]]; then
    echo "ERREUR: config introuvable → $CFG" >&2
    exit 1
fi
if ! "$PY" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERREUR: CUDA indisponible sur $(hostname)." >&2
    echo "  Lancer ce script sur un nœud GPU (sirius-1), pas sur la frontale (flyon)." >&2
    exit 1
fi

cd "$HPST"

echo "Préparation CIFAR-10-DVS T=16 (une fois, avant les 4 GPU)…"
"$PY" "$SNN_ROOT/scripts/download_data.py" cifar10-dvs \
    --data-dir "$DATA" --prepare-frames --frames 16 || {
    echo "ERREUR: préparation CIFAR-10-DVS échouée." >&2
    exit 1
}

TUNE_BASE=(
    "$PY" -m scripts.tune
    --config "$CFG"
    --dataset cifar10-dvs
    --data-dir "$DATA"
    --batch-size "$BATCH_SIZE"
    --train-fraction "$TRAIN_FRACTION"
    --seed 42
    --n-trials "$N_TRIALS"
    --tune-epochs "$TUNE_EPOCHS"
    --tune-aug
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
gpu=0

for entry in "${OPT_DVS_VARIANTS[@]}"; do
    IFS='|' read -r id attn hybrid study save_rel mlflow extra <<< "$entry"
    # shellcheck disable=SC2206
    extra_args=($extra)
    launch "$id" "$gpu" \
        --attention-mode "$attn" \
        --hybrid-qkv "$hybrid" \
        --study-name "$study" \
        --save-dir "save/${save_rel}" \
        --mlflow-experiment "$mlflow" \
        "${extra_args[@]}"
    gpu=$((gpu + 1))
done

echo "=== Sirius Optuna DVS — ${#OPT_DVS_VARIANTS[@]} études ==="
echo "Logs → $LOG"
echo "N_TRIALS=$N_TRIALS TUNE_EPOCHS=$TUNE_EPOCHS TRAIN_FRACTION=$TRAIN_FRACTION BATCH=$BATCH_SIZE"
echo ""

fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "OK  ${names[$i]}"
    else
        echo "FAIL ${names[$i]} — tail -40 $LOG/${names[$i]}.err"
        fail=1
    fi
done
exit "$fail"
