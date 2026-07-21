#!/usr/bin/env bash
# Lance (ou reutilise) le serveur MLflow UI pour un projet donne.
# Usage: ./mlflow-ui.sh [nom_projet]      (defaut: HPSTAtten)
set -euo pipefail

PROJECT="${1:-HPSTAtten}"
ROOT="$HOME/internship/snn"
DB="$ROOT/$PROJECT/mlflow.db"
PORT=5000

if [ ! -f "$DB" ]; then
  echo "ERREUR: base introuvable -> $DB" >&2
  echo "Projets disponibles :" >&2
  ls -1 "$ROOT" >&2
  exit 1
fi

# Deja en ecoute sur le port ? on ne relance pas (evite 'Address already in use').
if ss -ltn | grep -q ":$PORT "; then
  echo "MLflow UI deja actif sur le port $PORT -> on le reutilise."
  echo "  Pour le pointer sur une AUTRE base, arrete-le d'abord :  fuser -k $PORT/tcp"
  exit 0
fi

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
setsid mlflow ui \
  --backend-store-uri "sqlite:///$DB" \
  --host 127.0.0.1 --port "$PORT" \
  > /tmp/mlflow_ui.log 2>&1 < /dev/null &

# Attente que le serveur reponde (max ~30s)
for i in $(seq 1 15); do
  sleep 2
  if curl -s -o /dev/null http://127.0.0.1:$PORT/ 2>/dev/null; then
    echo "MLflow UI demarre pour le projet '$PROJECT'."
    echo "  base : $DB"
    echo "  logs : /tmp/mlflow_ui.log"
    echo "Cote local, garde un tunnel ouvert :  ssh -L $PORT:localhost:$PORT kasekou@flille"
    echo "Puis ouvre : http://localhost:$PORT"
    exit 0
  fi
done
echo "Le serveur ne repond pas encore, regarde /tmp/mlflow_ui.log" >&2
exit 1

