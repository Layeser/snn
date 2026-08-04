#!/usr/bin/env bash
# Réserve chicorée + chuc (lille) et sirius (lyon) pour le smoke — depuis le PC local.
#
# Config : scrip_grid_5000/reserve_smoke.yaml (voir reserve_smoke.yaml.example)
#
# Usage :
#   bash scrip_grid_5000/book_smoke_local.sh
#   bash scrip_grid_5000/book_smoke_local.sh --dry-run
#   RESERVE_SMOKE_CONFIG=/chemin/reserve_smoke.yaml bash scrip_grid_5000/book_smoke_local.sh
#
# Make :
#   make g5k-book-smoke
#   make g5k-book-smoke-check
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$ROOT")"
CONFIG="${RESERVE_SMOKE_CONFIG:-${ROOT}/reserve_smoke.yaml}"
JOBS_FILE="${ROOT}/manual_jobs.env"
RESERVE_SCRIPT="${ROOT}/reserve_manual.sh"
DRY_RUN=0

usage() {
    sed -n '2,13p' "$0" | sed 's/^# \?//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage 0 ;;
        *)
            echo "Option inconnue : $1" >&2
            usage 1
            ;;
    esac
done

[[ -f "$CONFIG" ]] || {
    echo "Config introuvable : $CONFIG" >&2
    echo "Copiez reserve_smoke.yaml.example → reserve_smoke.yaml et éditez le créneau." >&2
    exit 1
}

readarray -t CFG_LINES < <(python3 - "$CONFIG" <<'PY'
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML requis : pip install pyyaml")

path = Path(sys.argv[1])
cfg = yaml.safe_load(path.read_text()) or {}

def req(key):
    val = cfg.get(key)
    if val is None or val == "":
        sys.exit(f"Clé requise manquante dans {path}: {key}")
    return val

def oar_types(key, default=""):
    val = cfg.get(key, default)
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(x) for x in val)
    return str(val)

user = req("user")
gateway = cfg.get("ssh_gateway", "access.grid5000.fr")
remote = req("remote_project_dir")
start = str(req("reserve_start")).strip()
end = cfg.get("reserve_end")
duration = int(cfg.get("duration_minutes", 20))

if not end:
    dt = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    end = (dt + timedelta(minutes=duration)).strftime("%Y-%m-%d %H:%M:%S")

tag = str(cfg.get("tag", "smoke")).strip()
chicoree_gpu = int(cfg.get("chicoree_gpu", 4))
chuc_gpu = int(cfg.get("chuc_gpu", 4))
sirius_gpu = int(cfg.get("sirius_gpu", 8))

fields = [
    ("USER", user),
    ("SSH_GATEWAY", gateway),
    ("REMOTE_PROJECT", remote),
    ("RESERVE_START", start),
    ("RESERVE_END", end),
    ("RESERVE_TAG", tag),
    ("CHICOREE_GPU", chicoree_gpu),
    ("CHUC_GPU", chuc_gpu),
    ("SIRIUS_GPU", sirius_gpu),
    ("CHICOREE_OAR_TYPES", oar_types("chicoree_oar_types", "exotic day")),
    ("CHUC_OAR_TYPES", oar_types("chuc_oar_types", "day")),
    ("SIRIUS_OAR_TYPES", oar_types("sirius_oar_types", "exotic day")),
]
for key, val in fields:
    print(f"{key}={val}")
PY
)

declare -A CFG=()
for line in "${CFG_LINES[@]}"; do
    CFG["${line%%=*}"]="${line#*=}"
done

G5K_USER="${CFG[USER]}"
SSH_GATEWAY="${CFG[SSH_GATEWAY]}"
REMOTE_PROJECT="${CFG[REMOTE_PROJECT]}"
RESERVE_START="${CFG[RESERVE_START]}"
RESERVE_END="${CFG[RESERVE_END]}"
RESERVE_TAG="${CFG[RESERVE_TAG]}"

ssh_frontend() {
    local host=$1
    shift
    ssh -J "${G5K_USER}@${SSH_GATEWAY}" \
        -o BatchMode=yes \
        -o StrictHostKeyChecking=accept-new \
        "${G5K_USER}@${host}" "$@"
}

sync_remote_scripts() {
    local host=$1
    scp -J "${G5K_USER}@${SSH_GATEWAY}" \
        -o BatchMode=yes \
        -o StrictHostKeyChecking=accept-new \
        "${ROOT}/reserve_manual.sh" \
        "${G5K_USER}@${host}:~/${REMOTE_PROJECT}/scrip_grid_5000/reserve_manual.sh"
}

write_remote_env() {
    local site=$1 out=$2
    {
        printf 'export RESERVE_START=%q\n' "$RESERVE_START"
        printf 'export RESERVE_END=%q\n' "$RESERVE_END"
        printf 'export RESERVE_TAG=%q\n' "$RESERVE_TAG"
        case "$site" in
            lille)
                printf 'export CHICOREE_GPU=%q\n' "${CFG[CHICOREE_GPU]}"
                printf 'export CHUC_GPU=%q\n' "${CFG[CHUC_GPU]}"
                printf 'export CHICOREE_OAR_TYPES=%q\n' "${CFG[CHICOREE_OAR_TYPES]}"
                printf 'export CHUC_OAR_TYPES=%q\n' "${CFG[CHUC_OAR_TYPES]}"
                ;;
            lyon)
                printf 'export SIRIUS_GPU=%q\n' "${CFG[SIRIUS_GPU]}"
                printf 'export SIRIUS_OAR_TYPES=%q\n' "${CFG[SIRIUS_OAR_TYPES]}"
                ;;
        esac
    } >"$out"
}

run_remote_reserve() {
    local site=$1 host=$2
    local tmp_env remote_env="${REMOTE_PROJECT}/scrip_grid_5000/.book_smoke.env"

    echo ""
    echo "=== ${site} (${host}) : ${RESERVE_START} → ${RESERVE_END} tag=${RESERVE_TAG} ==="

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] scp reserve_manual.sh + .book_smoke.env → ${host}"
        echo "[dry-run] ssh ${host} source .book_smoke.env && bash reserve_manual.sh ${site}"
        case "$site" in
            lille)
                echo "  CHICOREE_OAR_TYPES=${CFG[CHICOREE_OAR_TYPES]}"
                echo "  CHUC_OAR_TYPES=${CFG[CHUC_OAR_TYPES]}"
                ;;
            lyon)
                echo "  SIRIUS_OAR_TYPES=${CFG[SIRIUS_OAR_TYPES]}"
                ;;
        esac
        return 0
    fi

    sync_remote_scripts "$host"

    tmp_env="$(mktemp)"
    write_remote_env "$site" "$tmp_env"
    scp -J "${G5K_USER}@${SSH_GATEWAY}" \
        -o BatchMode=yes \
        -o StrictHostKeyChecking=accept-new \
        "$tmp_env" \
        "${G5K_USER}@${host}:~/${remote_env}"
    rm -f "$tmp_env"

    ssh_frontend "$host" bash -s -- "$REMOTE_PROJECT" "$site" "$remote_env" <<'REMOTE'
set -euo pipefail
REMOTE_PROJECT=$1
SITE=$2
ENV_FILE=$3
# shellcheck disable=SC1090
source "$HOME/$ENV_FILE"
cd "$HOME/$REMOTE_PROJECT"
bash scrip_grid_5000/reserve_manual.sh "$SITE"
rm -f "$HOME/$ENV_FILE"
REMOTE
}

merge_job_ids() {
    local site=$1
    local content
    content="$(ssh_frontend "$site" bash -lc "grep -E '^JOB_(CHICOREE|CHUC|SIRIUS)=' \"\$HOME/${REMOTE_PROJECT}/scrip_grid_5000/manual_jobs.env\" 2>/dev/null | tail -3" || true)"
    if [[ -z "$content" ]]; then
        echo "Attention : aucun JOB_ID récupéré depuis ${site}." >&2
        return 0
    fi
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        local var="${line%%=*}"
        local val="${line#*=}"
        if grep -q "^${var}=" "$JOBS_FILE" 2>/dev/null; then
            sed -i "s/^${var}=.*/${var}=${val}/" "$JOBS_FILE"
        else
            echo "${var}=${val}" >>"$JOBS_FILE"
        fi
        echo "  local ${var}=${val}"
    done <<<"$content"
}

echo "=== Réservations smoke (local → SSH) ==="
echo "Config : ${CONFIG}"
echo "Créneau : ${RESERVE_START} → ${RESERVE_END} (tag=${RESERVE_TAG})"
echo "User   : ${G5K_USER} via ${SSH_GATEWAY}"

run_remote_reserve lille lille
run_remote_reserve lyon lyon

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo ""
    echo "[dry-run] Aucune réservation soumise."
    echo "Au créneau : make g5k-run-smoke-reserved-lille && make g5k-run-smoke-reserved-lyon (sur frontales)"
    exit 0
fi

touch "$JOBS_FILE"
{
    echo ""
    echo "# Smoke réservé depuis le PC — $(date -Iseconds) tag=${RESERVE_TAG}"
    echo "# ${RESERVE_START} → ${RESERVE_END}"
} >>"$JOBS_FILE"

echo ""
echo ">>> Synchronisation manual_jobs.env local"
merge_job_ids lille
merge_job_ids lyon

echo ""
echo "Job IDs locaux : ${JOBS_FILE}"
grep -E '^JOB_(CHICOREE|CHUC|SIRIUS)=' "$JOBS_FILE" || true
echo ""
echo "Vérifier : ssh lille/flyon puis oarstat -u ${G5K_USER}"
echo "Au créneau (frontales) :"
echo "  make g5k-run-smoke-reserved-lille"
echo "  make g5k-run-smoke-reserved-lyon"
