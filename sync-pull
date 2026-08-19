#!/usr/bin/env bash
# git pull avec résolution auto des conflits mlflow.db / optuna.db (frontales Lille/Lyon).
#
# Usage (depuis flille ou flyon) :
#   bash sync-pull.sh                 # pull ; conflits .db → version distante (origin)
#   bash sync-pull.sh remote          # idem
#   bash sync-pull.sh local           # conflits .db → version locale ; restaure le stash
#   bash sync-pull.sh most-runs       # garde le .db avec le plus de runs/trials
#   bash sync-pull.sh remote -- git pull --ff-only origin main
#
# Sauvegardes : .db-backup/<timestamp>/
set -euo pipefail

SNN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SNN_ROOT"

STRATEGY="remote"
if [[ $# -gt 0 && "$1" =~ ^(remote|local|most-runs)$ ]]; then
  STRATEGY="$1"
  shift
fi

if [[ "${1:-}" == "--" ]]; then
  shift
  if [[ $# -eq 0 ]]; then
    echo "Usage: bash sync-pull.sh [remote|local|most-runs] -- <commande git pull...>" >&2
    exit 1
  fi
  PULL_CMD=("$@")
else
  PULL_CMD=(git pull --no-rebase)
fi

mapfile -t DB_FILES < <(git ls-files | grep -E '(mlflow|optuna)\.db$' || true)

backup_dbs() {
  local dest="${SNN_ROOT}/.db-backup/$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$dest"
  local n=0
  for f in "${DB_FILES[@]}"; do
    if [[ -f "$f" ]]; then
      install -D "$f" "${dest}/$(echo "$f" | tr '/' '_')"
      n=$((n + 1))
    fi
  done
  echo "Sauvegarde : ${dest}/ (${n} fichier(s))" >&2
  echo "$dest"
}

db_entry_count() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo 0
    return
  fi
  if [[ "$f" == *optuna* ]]; then
    sqlite3 "$f" "SELECT count(*) FROM trials;" 2>/dev/null || echo 0
  else
    sqlite3 "$f" "SELECT count(*) FROM runs;" 2>/dev/null || echo 0
  fi
}

extract_stage() {
  local stage="$1" path="$2" out="$3"
  git show "${stage}:${path}" >"$out" 2>/dev/null || : >"$out"
}

resolve_db_conflict() {
  local f="$1"
  echo "  Conflit sur $f → stratégie: $STRATEGY"

  case "$STRATEGY" in
    remote)
      git checkout --theirs -- "$f"
      ;;
    local)
      git checkout --ours -- "$f"
      ;;
    most-runs)
      local tmp ours theirs c_ours c_theirs
      tmp="$(mktemp -d)"
      ours="${tmp}/ours.db"
      theirs="${tmp}/theirs.db"
      extract_stage ":2" "$f" "$ours"
      extract_stage ":3" "$f" "$theirs"
      c_ours="$(db_entry_count "$ours")"
      c_theirs="$(db_entry_count "$theirs")"
      if [[ "$c_theirs" -gt "$c_ours" ]]; then
        echo "    distante gagne (${c_theirs} > ${c_ours} entrées)"
        cp "$theirs" "$f"
      elif [[ "$c_ours" -gt "$c_theirs" ]]; then
        echo "    locale gagne (${c_ours} > ${c_theirs} entrées)"
        cp "$ours" "$f"
      else
        echo "    égalité (${c_ours}) → distante par défaut"
        cp "$theirs" "$f"
      fi
      rm -rf "$tmp"
      ;;
  esac
  git add -- "$f"
}

finish_merge_if_needed() {
  if git rev-parse -q --verify MERGE_HEAD >/dev/null 2>&1; then
    local unresolved
    unresolved="$(git diff --name-only --diff-filter=U || true)"
    if [[ -z "$unresolved" ]]; then
      git -c core.editor=true commit --no-edit
      echo "Merge terminé."
    else
      echo "Conflits restants (hors .db) :" >&2
      echo "$unresolved" >&2
      exit 1
    fi
  fi
}

STASHED=0
stash_local_changes() {
  if git diff --quiet && git diff --cached --quiet; then
    return 0
  fi
  local msg="sync-pull pre-pull $(date -Iseconds) [${STRATEGY}]"
  echo "Modifications locales (hors pull) → git stash :"
  git status --short
  git stash push -m "$msg"
  STASHED=1
}

restore_stash_if_local() {
  if [[ "$STASHED" -eq 0 ]]; then
    return 0
  fi
  case "$STRATEGY" in
    local|most-runs)
      echo "Restauration du stash local..."
      if git stash pop; then
        echo "Stash restauré."
      else
        echo "Conflit au stash pop — résolvez puis: git stash drop (si déjà appliqué)" >&2
        exit 1
      fi
      ;;
    remote)
      echo ""
      echo "Stash conservé (origin prioritaire pour le code) :"
      git stash list | head -3
      echo "  git stash show -p   # voir les changements locaux"
      echo "  git stash pop       # les réappliquer"
      echo "  git stash drop      # les abandonner"
      ;;
  esac
}

if [[ ${#DB_FILES[@]} -gt 0 ]]; then
  backup_dbs >/dev/null
fi

stash_local_changes

echo "→ ${PULL_CMD[*]}"
if "${PULL_CMD[@]}"; then
  :
else
  pull_status=$?
  if [[ $pull_status -ne 0 ]] && ! git rev-parse -q --verify MERGE_HEAD >/dev/null 2>&1; then
    exit "$pull_status"
  fi
fi

conflicts=()
if git rev-parse -q --verify MERGE_HEAD >/dev/null 2>&1; then
  while IFS= read -r f; do
    [[ -n "$f" ]] && conflicts+=("$f")
  done < <(git diff --name-only --diff-filter=U | grep -E '(mlflow|optuna)\.db$' || true)
fi

if [[ ${#conflicts[@]} -gt 0 ]]; then
  echo "Résolution de ${#conflicts[@]} conflit(s) .db..."
  for f in "${conflicts[@]}"; do
    resolve_db_conflict "$f"
  done
fi

finish_merge_if_needed
restore_stash_if_local

echo "OK — $(git log -1 --oneline)"
echo ""
echo "Stratégies : remote = origin gagne | local = cette frontale gagne | most-runs = plus de runs/trials"
echo "MLflow UI  : ./mlflow.sh HPSTAtten"
