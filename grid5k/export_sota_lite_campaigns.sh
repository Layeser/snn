#!/usr/bin/env bash
# Exporte best_params des 4 études proxy → config/campaigns/sota_lite/
# + YAML transférés vers la cellule binaire de chaque famille.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HPST="${PROJECT_DIR}/HPSTAtten"
BASE_CONFIG="${HPST}/config/campaigns/cifar10_grid_sota.yml"
OUT_DIR="${HPST}/config/campaigns/sota_lite"

# shellcheck source=option_a_proxy_variants.sh
source "${SCRIPT_DIR}/option_a_proxy_variants.sh"

variant_attn_hybrid() {
    case "$1" in
        hp) echo "factorized true" ;;
        statten) echo "factorized false" ;;
        contrast) echo "contrast true" ;;
        contrast_binary) echo "contrast false" ;;
        hp_linear) echo "sdt true" ;;
        sdt) echo "sdt false" ;;
        contrast_sdt) echo "contrast_sdt true" ;;
        contrast_sdt_binary) echo "contrast_sdt false" ;;
        *) return 1 ;;
    esac
}

export_from_best() {
    local best="$1" attn="$2" hybrid="$3" out="$4"
    python -m scripts.export_optuna_campaign \
        --best-params "$best" \
        --base-config "$BASE_CONFIG" \
        --attention-mode "$attn" \
        --hybrid-qkv "$hybrid" \
        --epochs 200 \
        --output "$out"
}

cd "$HPST"
mkdir -p "$OUT_DIR"

declare -A BEST_BY_PROXY

for entry in "${OPTION_A_PROXY_VARIANTS[@]}"; do
    IFS='|' read -r id attn hybrid study save_rel <<< "$entry"
    best="${HPST}/save/${save_rel}/${study}/best_params.yml"
    out="${OUT_DIR}/cifar10_${id}_best.yml"

    if [[ ! -f "$best" ]]; then
        echo "SKIP proxy $id — absent: $best"
        continue
    fi

    export_from_best "$best" "$attn" "$hybrid" "$out"
    BEST_BY_PROXY["$id"]="$best"
    echo "OK proxy → $out"
done

transfer() {
    local src_id="$1" target_id="$2"
    local best="${BEST_BY_PROXY[$src_id]:-}"
    [[ -n "$best" ]] || return 0
    read -r attn hybrid <<< "$(variant_attn_hybrid "$target_id")"
    local out="${OUT_DIR}/cifar10_${target_id}_transferred.yml"
    export_from_best "$best" "$attn" "$hybrid" "$out"
    {
        echo "# Transfert HP depuis proxy ${src_id} (Optuna lite 4 familles)"
        cat "$out"
    } > "${out}.tmp" && mv "${out}.tmp" "$out"
    echo "OK transfert $src_id → $target_id"
}

for entry in "${OPTION_A_PROXY_TRANSFER[@]}"; do
    IFS='|' read -r src_id target_id <<< "$entry"
    transfer "$src_id" "$target_id"
done

echo ""
echo "Campagnes lite : $OUT_DIR"
