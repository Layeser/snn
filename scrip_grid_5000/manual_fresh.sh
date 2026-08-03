#!/usr/bin/env bash
# Nettoyage mode manuel : archives *_experiences/, logs outputs/*_queue/.
# Les scripts à exécuter sont ceux présents à la racine de *_experiences/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$ROOT")"

echo "=== g5k-clean-manual : nettoyage mode manuel ==="

for cluster in chicoree chuc sirius; do
    q="${ROOT}/${cluster}_experiences"
    [[ -d "$q" ]] || continue
    mkdir -p "${q}/archive/done" "${q}/archive/failed"
    find "${q}/archive/done" "${q}/archive/failed" -mindepth 1 -delete 2>/dev/null || true
    echo "  ${cluster}_experiences/archive/ vidé"

    log="${PROJECT}/outputs/${cluster}_queue"
    if [[ -d "$log" ]]; then
        rm -rf "$log"
        echo "  outputs/${cluster}_queue/ supprimé"
    fi
done

echo ""
echo "Files actives (scripts à lancer au prochain g5k-run) :"
found=0
for cluster in chicoree chuc sirius; do
    q="${ROOT}/${cluster}_experiences"
    [[ -d "$q" ]] || continue
    mapfile -t scripts < <(find "$q" -maxdepth 1 -name '*.sh' -type f | sort)
    if [[ ${#scripts[@]} -gt 0 ]]; then
        found=1
        echo "  ${cluster}_experiences/ (${#scripts[@]} script(s)) :"
        printf '    - %s\n' "${scripts[@]##*/}"
    fi
done

if [[ "$found" -eq 0 ]]; then
    echo "  (aucun — copiez des .sh dans chicoree_experiences/, chuc_experiences/, sirius_experiences/)"
fi

echo ""
echo "manual_jobs.env conservé (JOB_ID des réservations en cours)."
