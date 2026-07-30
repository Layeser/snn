#!/usr/bin/env bash
# =============================================================
# Orchestrateur GPU sur nœud réservé — file + parallélisme
#
# Clusters supportés : chicoree (4 GPU), sirius (8 GPU)
#
# Usage :
#   bash scrip_grid_5000/run_gpu_queue.sh --cluster chicoree
#   bash scrip_grid_5000/run_gpu_queue.sh --cluster sirius --job-id <ID>
#
# Wrappers :
#   run_chicoree_queue.sh  → --cluster chicoree
#   run_sirius_queue.sh    → --cluster sirius
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
START_RUN="${SCRIPT_DIR}/start_run.sh"
SENTINEL='=== EXPERIENCE TERMINEE AVEC SUCCES ==='

CLUSTER=""
JOB_ID=""
MAX_GPUS=""
DO_GIT_PULL=1
DRY_RUN=0
SELF_SCRIPT="${SCRIPT_DIR}/run_gpu_queue.sh"

usage() {
    cat <<EOF
Orchestrateur GPU Grid'5000 (file + parallélisme)

Usage :
  bash scrip_grid_5000/run_gpu_queue.sh --cluster <chicoree|sirius> [options]

Options :
  --cluster NAME   chicoree (4 GPU) ou sirius (8 GPU) — requis
  --job-id ID      Connexion OAR depuis la frontale + lancement
  --max-gpus N     Slots parallèles (défaut : auto via nvidia-smi)
  --no-git-pull    Ne pas git pull avant démarrage
  --dry-run        Affiche la file sans lancer

Dossiers :
  scrip_grid_5000/<cluster>_experiences/*.sh
  outputs/<cluster>_queue/
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster)
            CLUSTER="${2:?--cluster requiert chicoree ou sirius}"
            shift 2
            ;;
        --job-id)
            JOB_ID="${2:?--job-id requiert un ID}"
            shift 2
            ;;
        --max-gpus)
            MAX_GPUS="${2:?--max-gpus requiert un entier}"
            shift 2
            ;;
        --no-git-pull)
            DO_GIT_PULL=0
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage 0
            ;;
        *)
            echo "Option inconnue : $1" >&2
            usage 1
            ;;
    esac
done

if [[ -z "$CLUSTER" ]]; then
    echo "Erreur : --cluster chicoree|sirius requis." >&2
    usage 1
fi

case "$CLUSTER" in
    chicoree)
        DEFAULT_GPUS=4
        FRONTEND="flille"
        ;;
    sirius)
        DEFAULT_GPUS=8
        FRONTEND="flyon"
        ;;
    *)
        echo "Erreur : cluster inconnu '${CLUSTER}' (chicoree|sirius)." >&2
        exit 1
        ;;
esac

QUEUE_DIR="${GPU_QUEUE_DIR:-${SCRIPT_DIR}/${CLUSTER}_experiences}"
ARCHIVE_OK="${QUEUE_DIR}/archive/done"
ARCHIVE_FAIL="${QUEUE_DIR}/archive/failed"
LOG_ROOT="${GPU_LOG_DIR:-${PROJECT_DIR}/outputs/${CLUSTER}_queue}"

detect_max_gpus() {
    if [[ -n "$MAX_GPUS" ]]; then
        echo "$MAX_GPUS"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | wc -l
    else
        echo "$DEFAULT_GPUS"
    fi
}

connect_and_run() {
    local state
    state="$(oarstat -f -j "$JOB_ID" 2>/dev/null | awk -F' = ' '/^[[:space:]]*state =/{print $2; exit}')"
    if [[ "$state" != "Running" ]]; then
        echo "Erreur : le job ${JOB_ID} n'est pas Running (state=${state:-inconnu})." >&2
        echo "Vérifiez : oarstat -fj ${JOB_ID}" >&2
        exit 1
    fi

    local remote_cmd
    remote_cmd=$(printf 'cd %q' "$PROJECT_DIR")
    if [[ "$DO_GIT_PULL" -eq 1 ]]; then
        remote_cmd+=$(printf ' && git pull --ff-only')
    fi
    remote_cmd+=$(printf ' && bash %q --cluster %q --no-git-pull' "$SELF_SCRIPT" "$CLUSTER")
    if [[ "$DRY_RUN" -eq 1 ]]; then
        remote_cmd+=' --dry-run'
    fi
    if [[ -n "$MAX_GPUS" ]]; then
        remote_cmd+=$(printf ' --max-gpus %q' "$MAX_GPUS")
    fi

    echo "Connexion au job OAR ${JOB_ID} (${CLUSTER}) et lancement..."
    exec oarsub -C "$JOB_ID" -- bash -lc "$remote_cmd"
}

if [[ -n "$JOB_ID" ]]; then
    connect_and_run
fi

cd "$PROJECT_DIR"
mkdir -p "$ARCHIVE_OK" "$ARCHIVE_FAIL" "$LOG_ROOT"

if [[ ! -x "$START_RUN" ]]; then
    echo "Erreur : $START_RUN introuvable ou non exécutable." >&2
    exit 1
fi

MAX_GPUS="$(detect_max_gpus)"
if [[ "$MAX_GPUS" -lt 1 ]]; then
    MAX_GPUS=1
fi

echo "============================================================"
echo "Orchestrateur GPU — ${CLUSTER} (frontale ${FRONTEND})"
echo "  Projet     : $PROJECT_DIR"
echo "  File       : $QUEUE_DIR/*.sh"
echo "  GPU slots  : $MAX_GPUS"
echo "  Logs       : $LOG_ROOT"
echo "============================================================"

if [[ "$DO_GIT_PULL" -eq 1 ]] && [[ -d "$PROJECT_DIR/.git" ]]; then
    echo "[git] git pull --ff-only..."
    git -C "$PROJECT_DIR" pull --ff-only
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[gpu] nvidia-smi -L :"
    nvidia-smi -L || true
fi

mapfile -t PENDING < <(find "$QUEUE_DIR" -maxdepth 1 -name '*.sh' -type f | sort)

if [[ ${#PENDING[@]} -eq 0 ]]; then
    echo "Aucun script dans $QUEUE_DIR (fichiers *.sh attendus)."
    exit 0
fi

echo "File d'attente (${#PENDING[@]} expérience(s)) :"
printf '  - %s\n' "${PENDING[@]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] Arrêt sans lancement."
    exit 0
fi

SCHEDULER_LOG="${LOG_ROOT}/scheduler.log"
echo "[$(date -Iseconds)] Démarrage ${CLUSTER} (${#PENDING[@]} jobs, ${MAX_GPUS} GPU)" >>"$SCHEDULER_LOG"

declare -A PID_TO_GPU=()
declare -A PID_TO_SCRIPT=()
declare -A PID_TO_LOG=()
declare -A GPU_BUSY=()
PENDING_IDX=0
FAIL_COUNT=0
OK_COUNT=0

archive_script() {
    local script=$1 dest_base=$2
    local name dest
    name="$(basename "$script")"
    dest="${dest_base}/${name}"
    if [[ -e "$dest" ]]; then
        dest="${dest_base}/$(date +%Y%m%d_%H%M%S)_${name}"
    fi
    mv "$script" "$dest"
    echo "$dest"
}

launch_on_gpu() {
    local gpu=$1 script=$2
    local name log_dir log_file

    name="$(basename "$script" .sh)"
    log_dir="${LOG_ROOT}/$(date +%Y%m%d_%H%M%S)_gpu${gpu}_${name}"
    mkdir -p "$log_dir"
    log_file="${log_dir}/run.log"

    echo "[$(date -Iseconds)] GPU ${gpu} START ${script}" | tee -a "$SCHEDULER_LOG"
    cp "$script" "${log_dir}/script.sh"

    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        bash "$START_RUN" "$script"
    ) >"$log_file" 2>&1 &

    local pid=$!
    PID_TO_GPU[$pid]=$gpu
    PID_TO_SCRIPT[$pid]=$script
    PID_TO_LOG[$pid]=$log_file
    GPU_BUSY[$gpu]=$pid
    echo "$pid" >"${log_dir}/pid"
}

fill_slots() {
    local gpu
    for ((gpu = 0; gpu < MAX_GPUS; gpu++)); do
        if [[ -n "${GPU_BUSY[$gpu]:-}" ]]; then
            continue
        fi
        if [[ "$PENDING_IDX" -ge ${#PENDING[@]} ]]; then
            continue
        fi
        launch_on_gpu "$gpu" "${PENDING[$PENDING_IDX]}"
        PENDING_IDX=$((PENDING_IDX + 1))
    done
}

reap_finished() {
    local pid gpu script log_file archived status

    for pid in "${!PID_TO_GPU[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            continue
        fi

        wait "$pid" || true
        gpu="${PID_TO_GPU[$pid]}"
        script="${PID_TO_SCRIPT[$pid]}"
        log_file="${PID_TO_LOG[$pid]:-}"
        unset 'PID_TO_GPU[$pid]'
        unset 'PID_TO_SCRIPT[$pid]'
        unset 'PID_TO_LOG[$pid]'
        unset 'GPU_BUSY[$gpu]'

        if [[ -n "$log_file" ]] && grep -qF "$SENTINEL" "$log_file" 2>/dev/null; then
            status="OK"
            archived="$(archive_script "$script" "$ARCHIVE_OK")"
            OK_COUNT=$((OK_COUNT + 1))
        else
            status="FAIL"
            archived="$(archive_script "$script" "$ARCHIVE_FAIL")"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi

        echo "[$(date -Iseconds)] GPU ${gpu} ${status} ${script} -> ${archived}" | tee -a "$SCHEDULER_LOG"
        if [[ "$status" == "FAIL" && -n "$log_file" ]]; then
            echo "  Dernières lignes du log (${log_file}) :" >&2
            tail -n 15 "$log_file" >&2 || true
        fi
        return 0
    done
    return 1
}

running_count() {
    echo "${#PID_TO_GPU[@]}"
}

while [[ "$PENDING_IDX" -lt ${#PENDING[@]} ]] || [[ "$(running_count)" -gt 0 ]]; do
    fill_slots

    if [[ "$(running_count)" -eq 0 && "$PENDING_IDX" -ge ${#PENDING[@]} ]]; then
        break
    fi

    if [[ "$(running_count)" -gt 0 ]]; then
        if ! reap_finished; then
            sleep 2
        fi
    fi
done

echo "============================================================"
echo "Terminé : ${OK_COUNT} succès, ${FAIL_COUNT} échec(s)."
echo "Logs    : $LOG_ROOT"
echo "Archive : $ARCHIVE_OK | $ARCHIVE_FAIL"
echo "============================================================"

if [[ "$FAIL_COUNT" -gt 0 ]]; then
    exit 1
fi
