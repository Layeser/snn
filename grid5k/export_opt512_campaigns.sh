#!/usr/bin/env bash
# Exporte best_params.yml (Optuna Opt512) → config/campaigns/opt512/cifar10_<id>_best.yml
#
# Prérequis : études terminées dans save/optuna_opt512/<id>/<study>/best_params.yml
#
# Usage :
#   bash grid5k/export_opt512_campaigns.sh
#   EPOCHS=310 bash grid5k/export_opt512_campaigns.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HPST="${PROJECT_DIR}/HPSTAtten"
CAMPAIGN_DIR="${HPST}/config/campaigns/opt512"
PY="${PROJECT_DIR}/.venv/bin/python"
BASE_CONFIG="${HPST}/config/campaigns/cifar10_grid_sota_512.yml"
EPOCHS="${EPOCHS:-310}"

# shellcheck source=opt512_variants.sh
source "${SCRIPT_DIR}/opt512_variants.sh"

cd "$HPST"
mkdir -p "$CAMPAIGN_DIR"

ok=0
skip=0

for entry in "${OPT512_VARIANTS[@]}"; do
    IFS='|' read -r id attn hybrid study save_rel _mlflow base_cfg <<< "$entry"
    best="${HPST}/save/${save_rel}/${study}/best_params.yml"
    out="${CAMPAIGN_DIR}/cifar10_${id}_best.yml"
    base_config="${HPST}/config/campaigns/${base_cfg:-cifar10_grid_sota_512.yml}"

    if [[ ! -f "$best" ]]; then
        echo "SKIP $id — absent: $best"
        skip=$((skip + 1))
        continue
    fi

    "$PY" -m scripts.export_optuna_campaign \
        --best-params "$best" \
        --base-config "$base_config" \
        --attention-mode "$attn" \
        --hybrid-qkv "$hybrid" \
        --epochs "$EPOCHS" \
        --output "$out"
    ok=$((ok + 1))
done

echo ""
echo "Export terminé : $ok campagne(s) @ ${EPOCHS} epochs, $skip ignorée(s)"
echo "Dossier : $CAMPAIGN_DIR"
