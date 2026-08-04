#!/usr/bin/env bash
# Lance le smoke sur des réservations -r EXISTANTES (pas de nouvel oarsub).
#
# Prérequis :
#   - scrip_grid_5000/manual_jobs.env avec JOB_CHICOREE, JOB_CHUC, JOB_SIRIUS
#   - Au créneau : oarstat -fj <ID> → state = Running
#
# Usage (sur flille / flyon, pas depuis le PC) :
#   bash scrip_grid_5000/run_smoke_reserved.sh lille
#   bash scrip_grid_5000/run_smoke_reserved.sh lyon
#   bash scrip_grid_5000/run_smoke_reserved.sh lille --dry-run
#
# Équivalent Makefile (sur la frontale) :
#   make g5k-run-smoke-reserved-lille
#   make g5k-run-smoke-reserved-lyon
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
        --no-prepare) SKIP_PREPARE=1; shift ;;
        *) echo "Option inconnue : $1" >&2; exit 1 ;;
    esac
done
SKIP_PREPARE="${SKIP_PREPARE:-0}"

if [[ -f "$JOBS_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$JOBS_FILE"
fi

cd "$PROJECT"
if [[ -d .git ]] && [[ "$DRY_RUN" -eq 0 ]]; then
    echo "[git] git pull --ff-only..."
    git pull --ff-only
fi

if [[ "$SKIP_PREPARE" -eq 0 ]]; then
    echo "=== Préparation smoke dans scrip_run/ (22 scripts) ==="
    bash "${ROOT}/prepare_pilot_smoke.sh"
fi

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

launch_smoke() {
    local cluster=$1 job_id=$2 rel_queue=$3
    local queue_dir="${PROJECT}/${rel_queue}"
    local -a args=(
        --cluster "$cluster"
        --queue-dir "$queue_dir"
        --no-git-pull
        --job-id "$job_id"
    )
    [[ "$DRY_RUN" -eq 1 ]] && args+=(--dry-run)

    echo "=== ${cluster} : file ${rel_queue} sur job ${job_id} ==="
    bash "$RUN_QUEUE" "${args[@]}" &
    echo "$!"
}

echo "=== Smoke sur réservations existantes (${SITE}) ==="
echo ""

case "$SITE" in
    lille)
        check_running "${JOB_CHICOREE:-}" "JOB_CHICOREE"
        check_running "${JOB_CHUC:-}" "JOB_CHUC"
        echo ""
        pids=()
        pid="$(launch_smoke chicoree "${JOB_CHICOREE}" "scrip_grid_5000/scrip_run/lille/chicoree")"
        [[ "$pid" =~ ^[0-9]+$ ]] && pids+=("$pid")
        pid="$(launch_smoke chuc "${JOB_CHUC}" "scrip_grid_5000/scrip_run/lille/chuc")"
        [[ "$pid" =~ ^[0-9]+$ ]] && pids+=("$pid")
        ;;
    lyon)
        check_running "${JOB_SIRIUS:-}" "JOB_SIRIUS"
        echo ""
        pids=()
        pid="$(launch_smoke sirius "${JOB_SIRIUS}" "scrip_grid_5000/scrip_run/lyon/sirius")"
        [[ "$pid" =~ ^[0-9]+$ ]] && pids+=("$pid")
        ;;
    *)
        echo "Site inconnu : ${SITE}" >&2
        exit 1
        ;;
esac

if [[ ${#pids[@]} -gt 0 && "$DRY_RUN" -eq 0 ]]; then
    echo ""
    echo "Attente des orchestrateurs (PIDs : ${pids[*]})..."
    wait "${pids[@]}"
    echo "Terminé."
fi
