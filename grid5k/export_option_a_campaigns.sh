#!/usr/bin/env bash
# Exporte best_params.yml → config/campaigns/option_a/cifar10_<id>_best.yml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HPST="${PROJECT_DIR}/HPSTAtten"
CAMPAIGN_DIR="${HPST}/config/campaigns/option_a"

# shellcheck source=option_a_variants.sh
source "${SCRIPT_DIR}/option_a_variants.sh"

cd "$HPST"
mkdir -p "$CAMPAIGN_DIR"

ok=0
skip=0

for entry in "${OPTION_A_VARIANTS[@]}"; do
    IFS='|' read -r id attn hybrid study save_rel <<< "$entry"
    best="${HPST}/save/${save_rel}/${study}/best_params.yml"
    out="${CAMPAIGN_DIR}/cifar10_${id}_best.yml"

    if [[ ! -f "$best" ]]; then
        echo "SKIP $id — absent: $best"
        skip=$((skip + 1))
        continue
    fi

    python -m scripts.export_optuna_campaign \
        --best-params "$best" \
        --base-config config/train_cifar10.yml \
        --attention-mode "$attn" \
        --hybrid-qkv "$hybrid" \
        --epochs 200 \
        --output "$out"
    ok=$((ok + 1))
done

echo ""
echo "Export terminé : $ok campagne(s), $skip ignorée(s) (best_params manquant)"
echo "Dossier : $CAMPAIGN_DIR"
