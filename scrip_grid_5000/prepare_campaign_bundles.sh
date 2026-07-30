#!/bin/bash
# Déploie les 3 jobs bundle (chicoree×4, chuc×4, sirius×1) dans scrip_run/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$ROOT/scrip_run"
BUNDLES="$ROOT/experiences/bundles"

mkdir -p "$DEST"

# Retirer les anciens scripts unitaires (9 jobs) s'ils traînent à la racine.
while IFS= read -r -d '' f; do
    base=$(basename "$f")
    case "$base" in
        lille_chicoree_grid4.sh|lille_chuc_dvs4.sh|lyon_sirius_optuna.sh) ;;
        *) rm -f "$f"; echo "  (retire ancien) $base" ;;
    esac
done < <(find "$DEST" -maxdepth 1 -name '*.sh' -print0 2>/dev/null)

copy_bundle() {
    local name=$1
    cp "${BUNDLES}/${name}" "${DEST}/${name}"
    chmod +x "${DEST}/${name}"
    echo "  -> scrip_run/${name}"
}

echo "=== 3 jobs bundle (parallèle sur 4 GPU pour Lille) ==="
copy_bundle lille_chicoree_grid4.sh
copy_bundle lille_chuc_dvs4.sh
copy_bundle lyon_sirius_optuna.sh

echo ""
echo "Annulez les anciens jobs OAR individuels si encore en file (oardel ...)."
echo "Puis :"
echo "  echo '{}' > scrip_grid_5000/pilot_grid/run_status.json"
echo "  git push && make pilot-grid"
