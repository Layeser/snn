#!/bin/bash
# =============================================================
# Lanceur GENERIQUE (mono-GPU) appele par l'orchestrateur :
#   oarsub ... "start_run.sh <chemin_experience.sh>"
#
# Role : se placer a la racine du projet, activer le venv, puis
# executer le script d'experience passe en argument. Aucune
# hypothese sur le dataset / le modele : tout est dans l'experience.
# =============================================================
set -euo pipefail

# --- Securite : on attend le chemin du script d'experience ---
if [ -z "${1:-}" ]; then
    echo "Erreur : chemin du script d'experience attendu en argument."
    exit 1
fi
SCRIPT_EXPERIENCE="$1"

# --- Racine du projet = dossier parent de scrip_grid_5000/ ---
# (ce script se trouve dans <projet>/scrip_grid_5000/start_run.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- Activation de l'environnement Python (si present) ---
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo "Environnement virtuel active : $PROJECT_DIR/.venv"
else
    echo "Attention : aucun .venv trouve a la racine ($PROJECT_DIR)."
fi

echo "=== Lancement de l'experience ==="
echo "Projet      : $PROJECT_DIR"
echo "Experience  : $SCRIPT_EXPERIENCE"
echo "================================="

bash "$SCRIPT_EXPERIENCE"

# Ligne sentinelle : sert a l'orchestrateur pour detecter la fin OK du run.
echo "=== EXPERIENCE TERMINEE AVEC SUCCES ==="
