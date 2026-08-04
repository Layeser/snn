#!/usr/bin/env bash
# Reinitialise l'etat besteffort (garde les .sh actifs).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="${ROOT}/besteffort_state/run_status.json"

mkdir -p "$(dirname "$STATE")"
echo '{}' >"$STATE"
echo "  besteffort_state/run_status.json reinitialise"

for site in lille lyon; do
    dir="${ROOT}/besteffort_${site}"
    [[ -d "$dir/archive/done" ]] && find "$dir/archive/done" -mindepth 1 -delete 2>/dev/null || true
    [[ -d "$dir" ]] || continue
    mapfile -t scripts < <(find "$dir" -maxdepth 1 -name '*.sh' | sort)
    echo "  besteffort_${site}/ : ${#scripts[@]} script(s)"
done
echo "=== besteffort-fresh termine ==="
