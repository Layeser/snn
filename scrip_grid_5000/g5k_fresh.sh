#!/usr/bin/env bash
# Nettoyage unifié scrip_grid_5000 — local et/ou frontales Grid'5000.
#
# Usage :
#   bash scrip_grid_5000/g5k_fresh.sh              # local + flille + flyon
#   bash scrip_grid_5000/g5k_fresh.sh --local    # cette machine seulement
#   bash scrip_grid_5000/g5k_fresh.sh --remote    # frontales seulement (SSH)
#
# Effets (--local) :
#   - run_status.json → {}
#   - archives runtime (scrip_run/, *_experiences/, outputs/*_queue/)
#   - outputs/ rapatriés (racine projet)
#   - ne supprime PAS les .sh actifs dans scrip_run/ ni *_experiences/
#
# Effets (--remote, depuis le PC) :
#   - git pull --ff-only + git restore scrip_grid_5000/ sur flille et flyon
#   - puis le même nettoyage runtime sur chaque frontale
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$ROOT")"
CONFIG="${ROOT}/pilot_grid/config.yaml"

DO_LOCAL=0
DO_REMOTE=0

usage() {
    sed -n '2,16p' "$0" | sed 's/^# \?//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)  DO_LOCAL=1; shift ;;
        --remote) DO_REMOTE=1; shift ;;
        -h|--help) usage 0 ;;
        *)
            echo "Option inconnue : $1" >&2
            usage 1
            ;;
    esac
done

if [[ "$DO_LOCAL" -eq 0 && "$DO_REMOTE" -eq 0 ]]; then
    DO_LOCAL=1
    DO_REMOTE=1
fi

read_config() {
    [[ -f "$CONFIG" ]] || { echo "Config introuvable : $CONFIG" >&2; exit 1; }
    G5K_USER="$(grep '^user:' "$CONFIG" | sed 's/user:[[:space:]]*//')"
    REMOTE_PROJECT="$(grep '^remote_project_dir:' "$CONFIG" | sed 's/remote_project_dir:[[:space:]]*//')"
    GIT_BRANCH="$(grep '^git_branch:' "$CONFIG" | sed 's/git_branch:[[:space:]]*//')"
    SSH_GATEWAY="$(grep '^ssh_gateway:' "$CONFIG" | sed 's/ssh_gateway:[[:space:]]*//')"
    GIT_BRANCH="${GIT_BRANCH:-main}"
    SSH_GATEWAY="${SSH_GATEWAY:-access.grid5000.fr}"
    [[ -n "$G5K_USER" && -n "$REMOTE_PROJECT" ]] || {
        echo "user et remote_project_dir requis dans $CONFIG" >&2
        exit 1
    }
}

clean_runtime() {
    local label="${1:-local}"
    echo "=== Nettoyage runtime ($label) ==="

    echo '{}' >"${ROOT}/pilot_grid/run_status.json"
    echo "  pilot_grid/run_status.json réinitialisé"

    if [[ -d "${ROOT}/scrip_run/archive" ]]; then
        find "${ROOT}/scrip_run/archive" -mindepth 1 -delete 2>/dev/null \
            || rm -rf "${ROOT}/scrip_run/archive/"*
        echo "  scrip_run/archive/ vidé"
    fi

    local site_dir cluster_dir
    for site_dir in "${ROOT}/scrip_run"/*/; do
        [[ -d "$site_dir" ]] || continue
        [[ "$(basename "$site_dir")" == "archive" ]] && continue
        for cluster_dir in "$site_dir"*/; do
            [[ -d "$cluster_dir" ]] || continue
            if [[ -d "${cluster_dir}/archive" ]]; then
                find "${cluster_dir}/archive" -mindepth 1 -delete 2>/dev/null || true
                echo "  scrip_run/$(basename "$site_dir")/$(basename "$cluster_dir")/archive/ vidé"
            fi
        done
    done

    local cluster q
    for cluster in chicoree chuc sirius; do
        q="${ROOT}/${cluster}_experiences"
        [[ -d "$q" ]] || continue
        mkdir -p "${q}/archive/done" "${q}/archive/failed"
        find "${q}/archive/done" "${q}/archive/failed" -mindepth 1 -delete 2>/dev/null || true
        echo "  ${cluster}_experiences/archive/ vidé"
    done

    if [[ -d "${PROJECT}/outputs" ]]; then
        rm -rf "${PROJECT}/outputs"
        echo "  outputs/ supprimé"
    fi

    echo ""
    echo "Files actives :"
    local found=0 site cluster scripts
    for site_dir in "${ROOT}/scrip_run"/*/; do
        [[ -d "$site_dir" ]] || continue
        site="$(basename "$site_dir")"
        [[ "$site" == "archive" ]] && continue
        for cluster_dir in "$site_dir"*/; do
            [[ -d "$cluster_dir" ]] || continue
            cluster="$(basename "$cluster_dir")"
            mapfile -t scripts < <(find "$cluster_dir" -maxdepth 1 -name '*.sh' -type f | sort)
            if [[ ${#scripts[@]} -gt 0 ]]; then
                found=1
                echo "  scrip_run/${site}/${cluster}/ (${#scripts[@]} script(s))"
                printf '    - %s\n' "${scripts[@]##*/}"
            fi
        done
    done
    for cluster in chicoree chuc sirius; do
        q="${ROOT}/${cluster}_experiences"
        [[ -d "$q" ]] || continue
        mapfile -t scripts < <(find "$q" -maxdepth 1 -name '*.sh' -type f | sort)
        if [[ ${#scripts[@]} -gt 0 ]]; then
            found=1
            echo "  ${cluster}_experiences/ (${#scripts[@]} script(s))"
            printf '    - %s\n' "${scripts[@]##*/}"
        fi
    done
    if [[ "$found" -eq 0 ]]; then
        echo "  (aucun .sh en file — copiez vos scripts avant g5k-auto ou g5k-run)"
    fi
    echo ""
    echo "manual_jobs.env conservé (réservations manuelles en cours)."
}

sync_frontend() {
    local host=$1
    echo ""
    echo ">>> Frontale ${host} (${G5K_USER})"

    ssh -J "${SSH_GATEWAY}" -o BatchMode=yes "${G5K_USER}@${host}" bash -lc "
        set -euo pipefail
        cd \"\$HOME/${REMOTE_PROJECT}\"
        echo '[git] fetch + pull ${GIT_BRANCH}...'
        git fetch --all --prune
        git checkout ${GIT_BRANCH}
        git pull --ff-only origin ${GIT_BRANCH}
        echo '[git] restore scrip_grid_5000/ depuis HEAD...'
        git restore scrip_grid_5000/
        bash scrip_grid_5000/g5k_fresh.sh --local
        echo \"[git] commit deploye : \$(git rev-parse --short HEAD)\"
    "
}

read_config

if [[ "$DO_LOCAL" -eq 1 ]]; then
    clean_runtime "local"
fi

if [[ "$DO_REMOTE" -eq 1 ]]; then
    echo "=== Synchronisation frontales (git + nettoyage) ==="
    sync_frontend flille
    sync_frontend flyon
    echo ""
    echo "Frontales alignées sur origin/${GIT_BRANCH}."
fi

echo "=== g5k-fresh terminé ==="
