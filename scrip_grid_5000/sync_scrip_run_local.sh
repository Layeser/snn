#!/usr/bin/env bash
# Envoie scrip_run/ (local) vers les frontales — sans oarsub.
#
# Usage (PC local) :
#   bash scrip_grid_5000/sync_scrip_run_local.sh
#   bash scrip_grid_5000/sync_scrip_run_local.sh --dry-run
#
# Make :
#   make g5k-sync-scrip-run
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$ROOT")"
CONFIG="${ROOT}/pilot_grid/config.yaml"
LOCAL_SCRIP="${ROOT}/scrip_run"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help)
            sed -n '2,9p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Option inconnue : $1" >&2; exit 1 ;;
    esac
done

[[ -f "$CONFIG" ]] || { echo "Config introuvable : $CONFIG" >&2; exit 1; }
G5K_USER="$(grep '^user:' "$CONFIG" | sed 's/user:[[:space:]]*//')"
REMOTE_PROJECT="$(grep '^remote_project_dir:' "$CONFIG" | sed 's/remote_project_dir:[[:space:]]*//')"
SSH_GATEWAY="$(grep '^ssh_gateway:' "$CONFIG" | sed 's/ssh_gateway:[[:space:]]*//')"
SSH_GATEWAY="${SSH_GATEWAY:-access.grid5000.fr}"

sync_site() {
    local site=$1 host=$2
    local local_dir="${LOCAL_SCRIP}/${site}"
    local remote_dir="~/${REMOTE_PROJECT}/scrip_grid_5000/scrip_run/${site}"

    echo ""
    echo "=== ${site} (${host}) ==="

    if [[ ! -d "$local_dir" ]]; then
        echo "  (dossier local absent : ${local_dir})"
        return 0
    fi

    mapfile -t scripts < <(find "$local_dir" -name '*.sh' -type f | sort)
    if [[ ${#scripts[@]} -eq 0 ]]; then
        echo "  Aucun .sh local sous ${local_dir}/"
        return 0
    fi

    echo "  ${#scripts[@]} script(s) à envoyer :"
    printf '    - %s\n' "${scripts[@]#"${PROJECT}/"}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  [dry-run] scp → ${host}:${remote_dir}/"
        return 0
    fi

    ssh -J "${G5K_USER}@${SSH_GATEWAY}" \
        -o BatchMode=yes \
        -o StrictHostKeyChecking=accept-new \
        "${G5K_USER}@${host}" \
        "mkdir -p ${remote_dir}"

    for script in "${scripts[@]}"; do
        rel="${script#"${local_dir}/"}"
        remote_subdir="$(dirname "$rel")"
        if [[ "$remote_subdir" != "." ]]; then
            ssh -J "${G5K_USER}@${SSH_GATEWAY}" \
                -o BatchMode=yes \
                -o StrictHostKeyChecking=accept-new \
                "${G5K_USER}@${host}" \
                "mkdir -p ${remote_dir}/${remote_subdir}"
        fi
        scp -J "${G5K_USER}@${SSH_GATEWAY}" \
            -o BatchMode=yes \
            -o StrictHostKeyChecking=accept-new \
            "$script" \
            "${G5K_USER}@${host}:${remote_dir}/${rel}"
        ssh -J "${G5K_USER}@${SSH_GATEWAY}" \
            -o BatchMode=yes \
            -o StrictHostKeyChecking=accept-new \
            "${G5K_USER}@${host}" \
            "chmod +x ${remote_dir}/${rel}"
    done

    ssh -J "${G5K_USER}@${SSH_GATEWAY}" \
        -o BatchMode=yes \
        -o StrictHostKeyChecking=accept-new \
        "${G5K_USER}@${host}" \
        "find ${remote_dir} -name '*.sh' -type f | wc -l" | {
        read -r nb
        echo "  Frontale : ${nb} script(s) sous scrip_run/${site}/"
    }
}

echo "=== Sync scrip_run/ → frontales ==="
echo "Local  : ${LOCAL_SCRIP}/"
echo "Remote : ~/${REMOTE_PROJECT}/scrip_grid_5000/scrip_run/"

sync_site lille lille
sync_site lyon lyon

echo ""
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] Aucun fichier envoyé."
else
    echo "Sync terminé. Au créneau : make g5k-run-lille-scrip / g5k-run-lyon-scrip (sur frontales)"
fi
