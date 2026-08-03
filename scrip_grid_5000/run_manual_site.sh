#!/usr/bin/env bash
# Lance les orchestrateurs manuels (file *_experiences/) pour un site.
#
# Usage :
#   bash scrip_grid_5000/run_manual_site.sh lille
#   bash scrip_grid_5000/run_manual_site.sh lyon
#
# Job IDs : scrip_grid_5000/manual_jobs.env (ou variables d'environnement)
# Nettoyage préalable : make g5k-clean-manual
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$ROOT")"
JOBS_FILE="${ROOT}/manual_jobs.env"
RUN_QUEUE="${ROOT}/run_gpu_queue.sh"
SITE="${1:?site requis : lille ou lyon}"
shift || true

DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
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

launch_cluster() {
    local cluster=$1 job_var=$2
    local job_id="${!job_var:-}"
    local -a args=(--cluster "$cluster" --no-git-pull)
    [[ "$DRY_RUN" -eq 1 ]] && args+=(--dry-run)

    if [[ -n "$job_id" && "$DRY_RUN" -eq 0 ]]; then
        echo "=== ${cluster} : oarsub -C ${job_id} → run_gpu_queue ==="
        bash "$RUN_QUEUE" "${args[@]}" --job-id "$job_id" &
        echo "$!"
        return
    fi

    echo "=== ${cluster} : exécution locale (${job_var}=${job_id:-non défini}) ==="
    bash "$RUN_QUEUE" "${args[@]}"
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

pids=()
for i in "${!CLUSTERS[@]}"; do
    pid="$(launch_cluster "${CLUSTERS[$i]}" "${JOB_VARS[$i]}")"
    if [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]]; then
        pids+=("$pid")
    fi
done

if [[ ${#pids[@]} -gt 0 ]]; then
    echo ""
    echo "Attente des orchestrateurs ${SITE} (PIDs : ${pids[*]})..."
    wait "${pids[@]}"
    echo "Terminé."
fi
