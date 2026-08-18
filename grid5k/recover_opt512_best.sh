#!/usr/bin/env bash
# Récupère best_params.yml pour les 4 études Opt512 CIFAR-10.
#
# À lancer sur le nœud où tournent / ont tourné les études (ex. Sirius/Lyon) :
#   cd ~/internship/snn && bash grid5k/recover_opt512_best.sh
#
# Puis copier vers Lille si besoin (home Grid5000 ≠ partagé entre sites) :
#   scp -r sirius-1.grid5000.fr:~/internship/snn/HPSTAtten/save/optuna_opt512/{contrast_hyb,hp,contrast_sdt_hyb,mk_hgr_triple} \
#       ~/internship/snn/HPSTAtten/save/optuna_opt512/
#
# Ensuite (flille ou chicoree) :
#   bash grid5k/export_opt512_campaigns.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HPST="${PROJECT_DIR}/HPSTAtten"
PY="${PROJECT_DIR}/.venv/bin/python"

# shellcheck source=opt512_variants.sh
source "${SCRIPT_DIR}/opt512_variants.sh"

echo "=== Récupération Opt512 best_params ==="
echo "HPSTAtten: $HPST"
echo "optuna.db: ${HPST}/optuna.db"
echo ""

ok=0
missing=0

for entry in "${OPT512_ALL_VARIANTS[@]}"; do
    IFS='|' read -r id _attn _hybrid study save_rel _mlflow _base <<< "$entry"
    dest="${HPST}/save/${save_rel}/${study}/best_params.yml"

    if [[ -f "$dest" ]]; then
        val="$(grep '^best_value:' "$dest" | awk '{print $2}')"
        echo "OK  $id — déjà présent (val=${val:-?}%) → $dest"
        ok=$((ok + 1))
        continue
    fi

    if [[ ! -f "${HPST}/optuna.db" ]]; then
        echo "SKIP $id — pas de optuna.db"
        missing=$((missing + 1))
        continue
    fi

    mkdir -p "$(dirname "$dest")"
    if "$PY" "${SCRIPT_DIR}/dump_optuna_best.py" --study "$study" --output "$dest"; then
        ok=$((ok + 1))
    else
        echo "MISS $id — ni fichier ni étude Optuna « $study »"
        missing=$((missing + 1))
    fi
done

echo ""
echo "Résumé : $ok trouvé(s), $missing manquant(s)"
if [[ "$missing" -gt 0 ]]; then
    echo ""
    echo "Si les études ont tourné sur Sirius, vérifie aussi :"
    echo "  ls ~/internship/snn/HPSTAtten/save/optuna_opt512/*/hpstattn-cifar10-opt512-*/best_params.yml"
    echo "  tail ~/internship/snn/outputs/sirius_opt512/*.out"
    exit 1
fi
