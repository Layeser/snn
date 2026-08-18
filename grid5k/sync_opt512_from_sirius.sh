#!/usr/bin/env bash
# Copie les best_params Opt512 depuis Sirius (Lyon) vers ce nœud (Lille).
#
# Prérequis : accès SSH Grid5000 (clé enregistrée).
# À lancer depuis flille ou chicoree :
#   bash grid5k/sync_opt512_from_sirius.sh
#
# Puis :
#   TRY_RECOVER=0 bash grid5k/export_opt512_campaigns.sh
set -euo pipefail

REMOTE="${REMOTE:-sirius-1.grid5000.fr}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HPST="${PROJECT_DIR}/HPSTAtten"
REMOTE_ROOT='~/internship/snn/HPSTAtten/save/optuna_opt512'

# shellcheck source=opt512_variants.sh
source "${SCRIPT_DIR}/opt512_variants.sh"

echo "Sync depuis $REMOTE → $HPST/save/optuna_opt512/"
echo ""

for entry in "${OPT512_ALL_VARIANTS[@]}"; do
    IFS='|' read -r id _attn _hybrid study save_rel _mlflow _base <<< "$entry"
    remote="${REMOTE_ROOT}/${id}/${study}/best_params.yml"
    local_dir="${HPST}/save/${save_rel}/${study}"
    local_file="${local_dir}/best_params.yml"

    if scp -q "${REMOTE}:${remote}" "$local_file" 2>/dev/null; then
        val="$(grep '^best_value:' "$local_file" | awk '{print $2}')"
        echo "OK  $id — val=${val:-?}%"
    else
        echo "MISS $id — $remote"
    fi
done

echo ""
echo "Vérification locale :"
find "${HPST}/save/optuna_opt512" -name best_params.yml 2>/dev/null || true
