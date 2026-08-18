#!/usr/bin/env bash
# Entraînement Opt512 @ D=512, 310 ep, 100 % train — 1 GPU, modèles en série.
#
# Prérequis :
#   bash grid5k/recover_opt512_best.sh    # sur Sirius si besoin
#   bash grid5k/export_opt512_campaigns.sh
#
# Usage (chicoree-1, 1 GPU) :
#   bash grid5k/chicoree_opt512_train_sequential.sh
#   VARIANT=hp bash grid5k/chicoree_opt512_train_sequential.sh
#   SKIP_MISSING=1 bash grid5k/chicoree_opt512_train_sequential.sh
#
# Logs : outputs/chicoree_opt512_train/<id>.{out,err}
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HPST="${PROJECT_DIR}/HPSTAtten"
PY="${PROJECT_DIR}/.venv/bin/python"
DATA="${PROJECT_DIR}/data"
CAMPAIGN_DIR="${HPST}/config/campaigns/opt512"
GRID_DIR="${HPST}/save/grid/cifar10"
LOG="${PROJECT_DIR}/outputs/chicoree_opt512_train"
EPOCHS="${EPOCHS:-310}"
GPU="${GPU:-0}"
VARIANT="${VARIANT:-all}"
SKIP_MISSING="${SKIP_MISSING:-0}"

# shellcheck source=opt512_variants.sh
source "${SCRIPT_DIR}/opt512_variants.sh"

mkdir -p "$LOG" "$GRID_DIR"

if [[ ! -x "$PY" ]]; then
    echo "ERREUR: venv introuvable → $PY" >&2
    exit 1
fi

if [[ -z "${MLFLOW_DB:-}" ]]; then
    bash "${SCRIPT_DIR}/ensure_mlflow_compat.sh" || exit 1
else
    echo "MLflow: base locale MLFLOW_DB=${MLFLOW_DB}"
fi

run_one() {
    local id="$1" attn="$2" hybrid="$3" mlflow="$4"
    local config="${CAMPAIGN_DIR}/cifar10_${id}_best.yml"
    local save_dir="${GRID_DIR}/opt512_${id}"
    local extra_args=(--vct-num 16)

    if [[ "$attn" == "factorized" || "$attn" == "mk_hgr" ]]; then
        extra_args=()
    fi

    if [[ ! -f "$config" ]]; then
        if [[ "$SKIP_MISSING" == "1" ]]; then
            echo "SKIP $id — config absente: $config"
            return 0
        fi
        echo "ERREUR: $config absent — lancer:" >&2
        echo "  bash grid5k/export_opt512_campaigns.sh" >&2
        echo "  (ou récupérer les HP depuis Sirius: bash grid5k/recover_opt512_best.sh)" >&2
        return 1
    fi

    echo ""
    echo "========================================"
    echo " Train opt512/$id | GPU=$GPU | ${EPOCHS} ep"
    echo " Config: $config"
    echo " Save:   $save_dir"
    echo " Log:    $LOG/${id}.out"
    echo "========================================"

    (
        export CUDA_VISIBLE_DEVICES="$GPU"
        cd "$HPST"
        "$PY" -m scripts.train \
            --config "$config" \
            --dataset cifar10 \
            --data-dir "$DATA" \
            --epochs "$EPOCHS" \
            --train-fraction 1.0 \
            --seed 42 \
            --fresh \
            --save-dir "$save_dir" \
            --mlflow-experiment "$mlflow" \
            --attention-mode "$attn" \
            --hybrid-qkv "$hybrid" \
            "${extra_args[@]}"
    ) >"$LOG/${id}.out" 2>"$LOG/${id}.err"

    local rc=$?
    if [[ "$rc" -eq 0 ]]; then
        echo "OK  $id"
    else
        echo "FAIL $id (exit $rc) — tail -40 $LOG/${id}.err"
    fi
    return "$rc"
}

fail=0
ran=0

for entry in "${OPT512_ALL_VARIANTS[@]}"; do
    IFS='|' read -r id attn hybrid _study _save_rel mlflow _base <<< "$entry"

    if [[ "$VARIANT" != "all" && "$VARIANT" != "$id" ]]; then
        continue
    fi

    if ! run_one "$id" "$attn" "$hybrid" "$mlflow"; then
        fail=1
    fi
    ran=$((ran + 1))
done

if [[ "$ran" -eq 0 ]]; then
    echo "ERREUR: VARIANT=$VARIANT ne correspond à aucune variante." >&2
    echo "Variantes: all | contrast_hyb | hp | contrast_sdt_hyb | mk_hgr_triple" >&2
    exit 1
fi

echo ""
if [[ "$fail" -eq 0 ]]; then
    echo "Entraînements Opt512 terminés ($ran modèle(s))."
else
    echo "Terminé avec erreur(s) — voir $LOG/*.err"
fi
exit "$fail"
