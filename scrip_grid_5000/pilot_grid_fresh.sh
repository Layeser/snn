#!/usr/bin/env bash
# Nettoyage mode auto : état, archives, outputs — puis affiche la file active par cluster.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$ROOT")"
STATE="${ROOT}/pilot_grid/run_status.json"
ARCHIVE="${ROOT}/scrip_run/archive"
OUTPUTS="${PROJECT}/outputs"
SCRIP_RUN="${ROOT}/scrip_run"

echo "=== g5k-auto-restart : nettoyage mode auto ==="

echo '{}' >"$STATE"
echo "  run_status.json réinitialisé"

if [[ -d "$ARCHIVE" ]]; then
    find "$ARCHIVE" -mindepth 1 -delete 2>/dev/null || rm -rf "${ARCHIVE:?}/"*
    echo "  scrip_run/archive/ vidé"
fi

if [[ -d "$OUTPUTS" ]]; then
    rm -rf "$OUTPUTS"
    echo "  outputs/ supprimé"
fi

echo ""
echo "File active (1 job OAR par dossier cluster) :"
found=0
for site_dir in "$SCRIP_RUN"/*/; do
    [[ -d "$site_dir" ]] || continue
    site="$(basename "$site_dir")"
    [[ "$site" == "archive" ]] && continue
    for cluster_dir in "$site_dir"*/; do
        [[ -d "$cluster_dir" ]] || continue
        cluster="$(basename "$cluster_dir")"
        mapfile -t scripts < <(find "$cluster_dir" -maxdepth 1 -name '*.sh' -type f | sort)
        if [[ ${#scripts[@]} -gt 0 ]]; then
            found=1
            echo "  scrip_run/${site}/${cluster}/ → 1 job OAR, ${#scripts[@]} expérience(s) en file GPU :"
            printf '    - %s\n' "${scripts[@]##*/}"
        fi
    done
done

if [[ "$found" -eq 0 ]]; then
    echo "  (aucun — déposez des .sh dans scrip_run/<site>/<cluster>/)"
fi
