#!/usr/bin/env bash
# Exécute une liste d'expériences SÉQUENTIELLEMENT sur 1 GPU (nœud réservé).
#
# Usage (sur chicoree-* après oarsub / oarsh) :
#   cd ~/internship/snn
#   bash grid5k/run_sequential.sh
#
# Liste personnalisée :
#   bash grid5k/run_sequential.sh besteffort_lille/exp1.sh besteffort_lille/exp2.sh
#
# Tous les .sh d'un dossier :
#   bash grid5k/run_sequential.sh --dir besteffort_lille
#
# tmux recommandé (connexion SSH peut couper) :
#   tmux new -s train
#   bash grid5k/run_sequential.sh
#   # Ctrl+B puis D
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RUN_ONE="${SCRIPT_DIR}/run_experiment.sh"
LOG_ROOT="${PROJECT_DIR}/outputs/sequential_queue"
SENTINEL='=== EXPERIENCE TERMINEE AVEC SUCCES ==='

DEFAULT_SCRIPTS=(
    besteffort_lille/grid_contrast_binary_optuna_t27_200ep.sh
    besteffort_lille/grid_contrast_optuna_t27_200ep.sh
    besteffort_lille/grid_hp_linear_optuna_t27_200ep.sh
    besteffort_lille/grid_hp_optuna_t27_200ep.sh
    besteffort_lille/grid_sdt_optuna_t27_200ep.sh
    besteffort_lille/grid_statten_optuna_t27_200ep.sh
)

usage() {
    cat <<EOF
Usage :
  bash grid5k/run_sequential.sh                    # 6 expériences Lille (défaut)
  bash grid5k/run_sequential.sh script1.sh script2.sh
  bash grid5k/run_sequential.sh --dir besteffort_lille
EOF
}

SCRIPTS=()
if [[ $# -eq 0 ]]; then
    SCRIPTS=("${DEFAULT_SCRIPTS[@]}")
elif [[ "${1:-}" == "--dir" ]]; then
    dir="${2:?--dir requiert un chemin}"
    mapfile -t SCRIPTS < <(find "$PROJECT_DIR/$dir" -maxdepth 1 -name '*.sh' -type f | sort)
elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
else
    SCRIPTS=("$@")
fi

if [[ ! -x "$RUN_ONE" ]]; then
    chmod +x "$RUN_ONE"
fi

cd "$PROJECT_DIR"
mkdir -p "$LOG_ROOT"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "=== GPU ==="
    nvidia-smi -L || true
    echo ""
fi

echo "============================================================"
echo "File séquentielle — ${#SCRIPTS[@]} expérience(s)"
echo "Projet : $PROJECT_DIR"
echo "Logs   : $LOG_ROOT"
echo "============================================================"

OK=0
FAIL=0
idx=0

for rel in "${SCRIPTS[@]}"; do
    idx=$((idx + 1))
    if [[ "$rel" != /* ]]; then
        script="$PROJECT_DIR/$rel"
    else
        script="$rel"
    fi

    if [[ ! -f "$script" ]]; then
        echo "[${idx}/${#SCRIPTS[@]}] ABSENT : $script" >&2
        FAIL=$((FAIL + 1))
        continue
    fi

    name="$(basename "$script" .sh)"
    log_dir="${LOG_ROOT}/$(date +%Y%m%d_%H%M%S)_${idx}_${name}"
    mkdir -p "$log_dir"
    log_file="${log_dir}/run.log"

    echo ""
    echo "[${idx}/${#SCRIPTS[@]}] START $(date -Iseconds) — $name"
    echo "  script : $script"
    echo "  log    : $log_file"

    start_ts=$(date +%s)
    if bash "$RUN_ONE" "$script" >"$log_file" 2>&1; then
        end_ts=$(date +%s)
        if grep -qF "$SENTINEL" "$log_file" 2>/dev/null; then
            echo "[${idx}/${#SCRIPTS[@]}] OK ($(( end_ts - start_ts ))s)"
            OK=$((OK + 1))
        else
            echo "[${idx}/${#SCRIPTS[@]}] FAIL (pas de sentinelle succès)" >&2
            tail -n 20 "$log_file" >&2 || true
            FAIL=$((FAIL + 1))
        fi
    else
        end_ts=$(date +%s)
        echo "[${idx}/${#SCRIPTS[@]}] FAIL ($(( end_ts - start_ts ))s, code $?)" >&2
        tail -n 20 "$log_file" >&2 || true
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "============================================================"
echo "Terminé : ${OK} succès, ${FAIL} échec(s) / ${#SCRIPTS[@]} total"
echo "Logs : $LOG_ROOT"
echo "============================================================"

[[ "$FAIL" -eq 0 ]]
