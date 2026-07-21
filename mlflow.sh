#!/usr/bin/env bash
# MLflow UI — lancer depuis la racine SNN ou depuis un dossier projet.
#
# Usage:
#   ./mlflow.sh                     # auto: dossier courant ou HPSTAtten
#   ./mlflow.sh HPSTAtten           # projet explicite
#   ./mlflow.sh hpstattn            # alias
#   cd HPSTAtten && ../mlflow.sh    # détecte HPSTAtten
#   ./mlflow.sh list                # projets avec mlflow.db
#   ./mlflow.sh status              # port / projet actif
#   ./mlflow.sh stop                # arrête le serveur sur PORT
#   ./mlflow.sh restart spikformer  # stop + relance sur une autre base
#
# Variables d'environnement:
#   PORT=5001 ./mlflow.sh HPSTAtten
#   SNN_ROOT=/chemin/snn ./mlflow.sh
#
# Grid5000 (flille) : lancer ici, pas sur le nœud GPU.
# Tunnel depuis votre PC :
#   ssh -L 5000:localhost:5000 kasekou@flille.lille.grid5000.fr

set -euo pipefail

SNN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SNN_ROOT/.venv"
PORT="${PORT:-5000}"
HOST="${HOST:-127.0.0.1}"
STATE_FILE="/tmp/mlflow_ui_${USER}.env"
LOG_FILE="/tmp/mlflow_ui_${USER}.log"

# Projets reconnus (nom dossier = clé canonique)
PROJECTS=(spikformer spikdrivenformer spatialtemporal A2OS2A HPSTAtten)

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \?//'
  echo
  echo "Projets : ${PROJECTS[*]}"
  echo "Aliases : hpstattn→HPSTAtten  a2os2a→A2OS2A  spikdriven→spikdrivenformer"
}

normalize_name() {
  local raw="${1,,}"
  case "$raw" in
    hpstattn|hp-statten) echo "HPSTAtten" ;;
    a2os2a) echo "A2OS2A" ;;
    spikdriven|spikdrivenformer) echo "spikdrivenformer" ;;
    spikformer) echo "spikformer" ;;
    spatialtemporal|stattent) echo "spatialtemporal" ;;
    *) echo "$1" ;;
  esac
}

is_project_dir() {
  local dir="$1"
  local name
  name="$(basename "$dir")"
  for p in "${PROJECTS[@]}"; do
    [[ "$name" == "$p" ]] && return 0
  done
  return 1
}

resolve_project() {
  local arg="${1:-}"

  if [[ -n "$arg" && "$arg" != "list" && "$arg" != "status" && "$arg" != "stop" && "$arg" != "restart" && "$arg" != "help" && "$arg" != "-h" && "$arg" != "--help" ]]; then
    echo "$(normalize_name "$arg")"
    return
  fi

  local cwd
  cwd="$(pwd -P)"

  if [[ -f "$cwd/mlflow.db" ]] && is_project_dir "$cwd"; then
    basename "$cwd"
    return
  fi

  for p in "${PROJECTS[@]}"; do
    if [[ "$cwd" == "$SNN_ROOT/$p"* && -f "$SNN_ROOT/$p/mlflow.db" ]]; then
      echo "$p"
      return
    fi
  done

  echo "HPSTAtten"
}

project_db() {
  echo "$SNN_ROOT/$1/mlflow.db"
}

list_projects() {
  echo "Projets avec mlflow.db :"
  for p in "${PROJECTS[@]}"; do
    local db
    db="$(project_db "$p")"
    if [[ -f "$db" ]]; then
      local runs
      runs="$(sqlite3 "$db" "select count(*) from runs;" 2>/dev/null || echo "?")"
      printf "  %-20s %s  (%s runs)\n" "$p" "$db" "$runs"
    else
      printf "  %-20s (pas encore de mlflow.db)\n" "$p"
    fi
  done
}

port_in_use() {
  ss -ltn 2>/dev/null | grep -q ":${PORT} "
}

stop_server() {
  if ! port_in_use; then
    echo "Aucun serveur sur le port $PORT."
    rm -f "$STATE_FILE"
    return 0
  fi

  echo "Arrêt du serveur MLflow sur le port $PORT..."
  pkill -f "uvicorn.*mlflow.server.fastapi_app.*--port ${PORT}" 2>/dev/null || true
  fuser -k "${PORT}/tcp" 2>/dev/null || true
  sleep 1

  if port_in_use; then
    echo "ERREUR: le port $PORT est encore occupé." >&2
    ss -ltnp 2>/dev/null | grep ":${PORT} " || true
    exit 1
  fi

  rm -f "$STATE_FILE"
  echo "Serveur arrêté."
}

show_status() {
  echo "SNN_ROOT=$SNN_ROOT"
  echo "PORT=$PORT"
  if port_in_use; then
    echo "Serveur: ACTIF sur http://${HOST}:${PORT}"
    ss -ltnp 2>/dev/null | grep ":${PORT} " || true
  else
    echo "Serveur: inactif"
  fi
  if [[ -f "$STATE_FILE" ]]; then
    echo "--- dernière session ---"
    cat "$STATE_FILE"
  fi
  echo "--- log ---"
  echo "$LOG_FILE"
  [[ -f "$LOG_FILE" ]] && tail -3 "$LOG_FILE" || true
}

start_server() {
  local project="$1"
  local db
  db="$(project_db "$project")"

  if [[ ! -f "$db" ]]; then
    echo "ERREUR: base introuvable -> $db" >&2
    echo "Lancez d'abord un entraînement pour ce projet, ou choisissez :" >&2
    list_projects >&2
    exit 1
  fi

  if [[ ! -x "$VENV/bin/mlflow" ]]; then
    echo "ERREUR: venv absent ou mlflow non installé -> $VENV" >&2
    echo "  cd $SNN_ROOT && make setup" >&2
    exit 1
  fi

  if port_in_use; then
    if [[ -f "$STATE_FILE" ]]; then
      # shellcheck disable=SC1090
      source "$STATE_FILE"
      if [[ "${MLFLOW_PROJECT:-}" == "$project" && "${MLFLOW_DB:-}" == "$db" ]]; then
        echo "MLflow UI déjà actif pour '$project' sur le port $PORT."
        print_access_hint
        exit 0
      fi
      echo "ATTENTION: port $PORT utilisé par le projet '${MLFLOW_PROJECT:-?}'." >&2
      echo "  Base active : ${MLFLOW_DB:-?}" >&2
      echo "  Base demandée : $db" >&2
      echo "  Relancez avec : ./mlflow.sh restart $project" >&2
      exit 1
    fi
    echo "ERREUR: port $PORT déjà occupé (autre processus ?)." >&2
    echo "  ./mlflow.sh stop   ou   ./mlflow.sh status" >&2
    exit 1
  fi

  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  setsid mlflow ui \
    --backend-store-uri "sqlite:///$db" \
    --host "$HOST" --port "$PORT" \
    >"$LOG_FILE" 2>&1 < /dev/null &

  cat >"$STATE_FILE" <<EOF
MLFLOW_PROJECT=$project
MLFLOW_DB=$db
MLFLOW_PORT=$PORT
MLFLOW_STARTED=$(date -Iseconds)
EOF

  for _ in $(seq 1 15); do
    sleep 2
    if curl -s -o /dev/null "http://${HOST}:${PORT}/" 2>/dev/null; then
      echo "MLflow UI démarré pour '$project'."
      echo "  base : $db"
      echo "  logs : $LOG_FILE"
      print_access_hint
      exit 0
    fi
  done

  echo "ERREUR: le serveur ne répond pas. Voir $LOG_FILE" >&2
  tail -20 "$LOG_FILE" >&2 || true
  exit 1
}

print_access_hint() {
  echo
  echo "Tunnel SSH (depuis votre PC local, pas depuis flille) :"
  echo "  ssh -L ${PORT}:localhost:${PORT} kasekou@flille.lille.grid5000.fr"
  echo "Navigateur : http://localhost:${PORT}"
  echo "Expérience : ouvrir celle du dataset (ex. HP-STAtten-CIFAR10-DVS pour HPSTAtten + DVS)."
}

main() {
  local cmd="${1:-}"

  case "$cmd" in
    help|-h|--help)
      usage
      ;;
    list)
      list_projects
      ;;
    status)
      show_status
      ;;
    stop)
      stop_server
      ;;
    restart)
      stop_server
      start_server "$(resolve_project "${2:-}")"
      ;;
    "")
      start_server "$(resolve_project "")"
      ;;
    *)
      start_server "$(resolve_project "$cmd")"
      ;;
  esac
}

main "$@"
