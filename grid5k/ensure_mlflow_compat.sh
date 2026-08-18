#!/usr/bin/env bash
# Vérifie que mlflow.db est lisible par le MLflow installé dans .venv.
#
# Usage :
#   bash grid5k/ensure_mlflow_compat.sh
#   bash grid5k/ensure_mlflow_compat.sh --fix   # pip install mlflow>=3.15.1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HPST="${PROJECT_DIR}/HPSTAtten"
PY="${PROJECT_DIR}/.venv/bin/python"
DB="${HPST}/mlflow.db"
FIX=0
[[ "${1:-}" == "--fix" ]] && FIX=1

if [[ ! -x "$PY" ]]; then
    echo "ERREUR: venv introuvable → $PY" >&2
    exit 1
fi

MLFLOW_VER="$("$PY" -c "import mlflow; print(mlflow.__version__)")"
echo "MLflow installé: $MLFLOW_VER"
echo "Base: $DB"

if "$PY" - <<PY
import mlflow
from pathlib import Path

db = Path("${DB}").resolve()
mlflow.set_tracking_uri(f"sqlite:///{db}")
mlflow.tracking.MlflowClient()
print("OK")
PY
then
    echo "MLflow DB compatible."
    exit 0
fi

echo ""
echo "ERREUR: mlflow.db incompatible (souvent après push depuis Lyon avec MLflow plus récent)."
echo ""
echo "Solution recommandée (même store, historique unifié) :"
echo "  ${PY} -m pip install 'mlflow>=3.15.1'"
echo "  ${PY} -m mlflow db upgrade sqlite:///${DB}"
echo ""
echo "Alternative temporaire (base locale séparée, entraînement sans erreur) :"
echo "  export MLFLOW_DB=mlflow_chicoree.db"
echo "  VARIANT=contrast_hyb bash grid5k/chicoree_opt512_train_sequential.sh"

if [[ "$FIX" -eq 1 ]]; then
    echo ""
    echo "→ pip install mlflow>=3.15.1 ..."
    "$PY" -m pip install 'mlflow>=3.15.1'
    echo "→ mlflow db upgrade ..."
    "$PY" -m mlflow db upgrade "sqlite:///${DB}"
    echo "OK — relancer l'entraînement."
    exit 0
fi

exit 1
