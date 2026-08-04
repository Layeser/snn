#!/usr/bin/env bash
# Lance la campagne réelle sur réservations -r existantes (pas de nouvel oarsub).
#
# Lit manual_jobs.env → oarsub -C <JOB_ID> → run_gpu_queue.sh par cluster.
# File d'attente : scrip_run/<site>/<cluster>/*.sh (même layout que g5k-auto).
#
# Usage (sur flille / flyon) :
#   bash scrip_grid_5000/run_manual_site.sh lille --scrip-run
#   bash scrip_grid_5000/run_manual_site.sh lyon --scrip-run --dry-run
#
# Make :
#   make g5k-run-lille-scrip
#   make g5k-check-lille-scrip
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$ROOT")"
JOBS_FILE="${ROOT}/manual_jobs.env"
RUN_QUEUE="${ROOT}/run_gpu_queue.sh"
SITE="${1:?site requis : lille ou lyon}"
shift || true

DRY_RUN=0
USE_SCRIP_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --scrip-run) USE_SCRIP_RUN=1; shift ;;
        --fresh)
            echo "Utilisez : make g5k-restart-${SITE}  (ou g5k-clean-manual puis g5k-run-${SITE})" >&2
            exit 1
            ;;
        *) echo "Option inconnue : $1" >&2; exit 1 ;;
    esac
done

if [[ -f "$JOBS_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$JOBS_FILE"
fi

queue_dir_for() {
    local cluster=$1
    if [[ "$USE_SCRIP_RUN" -eq 1 ]]; then
        echo "scrip_grid_5000/scrip_run/${SITE}/${cluster}"
    else
        echo "scrip_grid_5000/${cluster}_experiences"
    fi
}

check_running() {
    local job_id=$1 label=$2
    if [[ -z "$job_id" ]]; then
        echo "Erreur : ${label} non défini dans ${JOBS_FILE}" >&2
        exit 1
    fi
    local state
    state="$(oarstat -f -j "$job_id" 2>/dev/null | awk -F' = ' '/^[[:space:]]*state =/{print $2; exit}')"
    echo "  ${label} job ${job_id} → state=${state:-inconnu}"
    if [[ "$DRY_RUN" -eq 0 && "$state" != "Running" ]]; then
        echo "Erreur : le job ${job_id} (${label}) n'est pas Running." >&2
        echo "Attendez le début du créneau (-r) ou vérifiez : oarstat -fj ${job_id}" >&2
        exit 1
    fi
}

count_scripts() {
    local queue_dir=$1
    find "${PROJECT}/${queue_dir}" -maxdepth 1 -name '*.sh' -type f 2>/dev/null | wc -l
}

launch_cluster() {
    local cluster=$1 job_var=$2
    local job_id="${!job_var:-}"
    local queue_dir
    queue_dir="$(queue_dir_for "$cluster")"
    local nb
    nb="$(count_scripts "$queue_dir")"

    echo ""
    echo "=== ${cluster} : ${nb} script(s) dans ${queue_dir} ==="
    if [[ "$nb" -eq 0 ]]; then
        echo "Erreur : aucun .sh dans ${queue_dir}/" >&2
        if [[ "$USE_SCRIP_RUN" -eq 1 ]]; then
            echo "→ Copiez vos scripts dans scrip_run/${SITE}/${cluster}/ (local)" >&2
            echo "→ Puis : make g5k-sync-scrip-run (PC) ou copie manuelle sur la frontale" >&2
        else
            echo "→ Copiez vos scripts dans scrip_grid_5000/${cluster}_experiences/" >&2
        fi
        exit 1
    fi

    check_running "$job_id" "$job_var"

    local -a args=(--cluster "$cluster" --queue-dir "$queue_dir" --no-git-pull --job-id "$job_id")
    [[ "$DRY_RUN" -eq 1 ]] && args+=(--dry-run)

    echo "=== ${cluster} : oarsub -C ${job_id} → run_gpu_queue ==="
    bash "$RUN_QUEUE" "${args[@]}" &
    echo "$!"
}

cd "$PROJECT"
if [[ -d .git ]] && [[ "$DRY_RUN" -eq 0 ]]; then
    echo "[git] git pull --ff-only..."
    git pull --ff-only
fi

case "$SITE" in
    lille)
        CLUSTERS=(chicoree chuc)
        JOB_VARS=(JOB_CHICOREE JOB_CHUC)
        ;;
    lyon)
        CLUSTERS=(sirius)
        JOB_VARS=(JOB_SIRIUS)
        ;;
    *)
        echo "Site inconnu : ${SITE}" >&2
        exit 1
        ;;
esac

mode="manuel (*_experiences/)"
[[ "$USE_SCRIP_RUN" -eq 1 ]] && mode="campagne (scrip_run/)"
echo "=== Lancement ${SITE} — ${mode} ==="
echo "Jobs : ${JOBS_FILE}"

pids=()
for i in "${!CLUSTERS[@]}"; do
    pid="$(launch_cluster "${CLUSTERS[$i]}" "${JOB_VARS[$i]}")"
    if [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]]; then
        pids+=("$pid")
    fi
done

if [[ ${#pids[@]} -gt 0 && "$DRY_RUN" -eq 0 ]]; then
    echo ""
    echo "Attente des orchestrateurs ${SITE} (PIDs : ${pids[*]})..."
    wait "${pids[@]}"
    echo "Terminé."
fi
