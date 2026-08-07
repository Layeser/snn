#!/usr/bin/env bash
# Reconcilie run_status.json : 1 job_id canonique par script (le plus recent).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="${ROOT}/besteffort_state/run_status.json"
SITE="${1:-lille}"

python3 - <<PY
import json
from pathlib import Path

state_path = Path("${STATE}")
site = "${SITE}"
state = json.loads(state_path.read_text()) if state_path.exists() else {}

# Garder une entree par script ; preferer le job_id le plus recent (updated)
by_script = {}
for key, info in list(state.items()):
    if not key.startswith(f"{site}/"):
        continue
    name = key.split("/", 1)[1]
    prev = by_script.get(name)
    if prev is None or (info.get("updated") or "") >= (prev.get("updated") or ""):
        by_script[name] = (key, info)

# Supprimer doublons / entrees obsoletes pour ce site
for key in list(state.keys()):
    if not key.startswith(f"{site}/"):
        continue
    name = key.split("/", 1)[1]
    canon_key, _ = by_script.get(name, (None, None))
    if canon_key != key:
        del state[key]

# Forcer ETAPE_2 si job_id present
for key, info in state.items():
    if key.startswith(f"{site}/") and info.get("job_id"):
        info["etape"] = "ETAPE_2"

state_path.write_text(json.dumps(state, indent=2))
print(f"Etat reconcilie pour {site} : {sum(1 for k in state if k.startswith(site+'/'))} entree(s)")
PY
