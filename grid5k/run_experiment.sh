#!/usr/bin/env bash
# Lance une experience depuis un job OAR besteffort (n'importe quel nœud GPU).
# Usage : oarsub -q besteffort ... grid5k/run_experiment.sh /chemin/exp.sh
set -euo pipefail

if [[ -z "${1:-}" ]]; then
    echo "Erreur : chemin du script d'experience attendu." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

echo "=== Lancement experience ==="
echo "Projet : $PROJECT_DIR"
echo "Script : $1"
bash "$1"
echo "=== EXPERIENCE TERMINEE AVEC SUCCES ==="
