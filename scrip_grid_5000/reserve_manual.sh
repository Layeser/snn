#!/usr/bin/env bash
# Soumet les réservations manuelles (-r) sur le site courant.
#
# Usage (flille — Lille) :
#   RESERVE_START="2026-08-04 19:00:00" \
#   RESERVE_END="2026-08-05 09:00:00" \
#   RESERVE_TAG=04 \
#   bash scrip_grid_5000/reserve_manual.sh lille
#
# Usage (flyon — Lyon) :
#   RESERVE_START="2026-08-04 19:00:00" \
#   RESERVE_END="2026-08-05 09:00:00" \
#   RESERVE_TAG=04 \
#   bash scrip_grid_5000/reserve_manual.sh lyon
#
# Les JOB_ID sont enregistrés dans scrip_grid_5000/manual_jobs.env
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS_FILE="${ROOT}/manual_jobs.env"
SITE="${1:?site requis : lille ou lyon}"

RESERVE_START="${RESERVE_START:?RESERVE_START requis (ex: 2026-08-04 19:00:00)}"
RESERVE_END="${RESERVE_END:?RESERVE_END requis (ex: 2026-08-05 09:00:00)}"
RESERVE_TAG="${RESERVE_TAG:-run}"
CHICOREE_GPU="${CHICOREE_GPU:-4}"
CHUC_GPU="${CHUC_GPU:-4}"
SIRIUS_GPU="${SIRIUS_GPU:-8}"
CHICOREE_OAR_TYPES="${CHICOREE_OAR_TYPES:-exotic}"
CHUC_OAR_TYPES="${CHUC_OAR_TYPES:-}"
SIRIUS_OAR_TYPES="${SIRIUS_OAR_TYPES:-exotic}"

submit_job() {
    local var_name=$1 cluster=$2 gpu=$3 extra_types=$4 job_name=$5
    local -a types=()
    local out err cmd result job_id

    out="${HOME}/${job_name}.out"
    err="${HOME}/${job_name}.err"

    read -r -a types <<<"$extra_types"
    cmd=(oarsub -p "$cluster" -l "host=1/gpu=${gpu}"
        -r "${RESERVE_START}, ${RESERVE_END}"
        -q default -n "$job_name" -O "$out" -E "$err")
    for t in "${types[@]}"; do
        [[ -n "$t" ]] && cmd+=(-t "$t")
    done
    cmd+=(-- /bin/sleep 999999)

    echo ""
    echo ">>> ${cluster} (gpu=${gpu}) : ${cmd[*]}"
    result="$("${cmd[@]}" 2>&1)" || true
    echo "$result"
    job_id="$(grep -oE 'OAR_JOB_ID=[0-9]+' <<<"$result" | tail -1 | cut -d= -f2 || true)"
    if [[ -z "$job_id" ]]; then
        echo "Erreur : pas de OAR_JOB_ID pour ${cluster}." >&2
        if grep -qi 'cannot have more than 2 waiting reservations' <<<"$result"; then
            echo "" >&2
            echo "Limite Grid'5000 : max 2 réservations en attente (état W)." >&2
            echo "→ Utilisez vos réservations existantes : make g5k-run-smoke-reserved-lille|lyon" >&2
            echo "→ Ou attendez qu'une réservation en attente expire / se termine." >&2
        fi
        return 1
    fi
    echo "${var_name}=${job_id}" >>"$JOBS_FILE"
    echo "    → ${var_name}=${job_id}"
}

touch "$JOBS_FILE"
echo "# Réservations manuelles — $(date -Iseconds) tag=${RESERVE_TAG}" >>"$JOBS_FILE"
echo "# ${RESERVE_START} → ${RESERVE_END}" >>"$JOBS_FILE"

case "$SITE" in
    lille)
        hostname_short="$(hostname -s 2>/dev/null || hostname)"
        if [[ "$hostname_short" != "flille" ]]; then
            echo "Attention : lille doit être soumis depuis flille (actuel : ${hostname_short})." >&2
        fi
        submit_job JOB_CHICOREE chicoree "$CHICOREE_GPU" "$CHICOREE_OAR_TYPES" "hpstattn_chicoree_${RESERVE_TAG}"
        submit_job JOB_CHUC chuc "$CHUC_GPU" "$CHUC_OAR_TYPES" "hpstattn_chuc_${RESERVE_TAG}"
        ;;
    lyon)
        hostname_short="$(hostname -s 2>/dev/null || hostname)"
        if [[ "$hostname_short" != "flyon" ]]; then
            echo "Attention : lyon doit être soumis depuis flyon (actuel : ${hostname_short})." >&2
        fi
        submit_job JOB_SIRIUS sirius "$SIRIUS_GPU" "$SIRIUS_OAR_TYPES" "hpstattn_sirius_${RESERVE_TAG}"
        ;;
    *)
        echo "Site inconnu : ${SITE} (lille|lyon)" >&2
        exit 1
        ;;
esac

echo ""
echo "Job IDs enregistrés dans ${JOBS_FILE}"
echo "Vérifier : oarstat -u \${USER}"
echo "Au créneau : make g5k-run-${SITE}"
