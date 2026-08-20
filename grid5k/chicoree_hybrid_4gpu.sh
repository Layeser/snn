#!/usr/bin/env bash
# 4 variantes HYBRIDE CIFAR-10 — 1 GPU / modèle (chicoree, 4 GPU).
# Défaut : D=512 depth=2 (EMBED_DIM=512). Campagne depth=4 :
#   GRID_CAMPAIGN=4-384-400 bash grid5k/chicoree_hybrid_4gpu.sh --fresh --background
#
#   GPU 0 → grid-fresh-hp           (factorized + hybrid)
#   GPU 1 → grid-fresh-hp-linear    (sdt + hybrid)
#   GPU 2 → grid-fresh-contrast     (contrast + hybrid)
#   GPU 3 → grid-fresh-contrast-sdt (contrast_sdt + hybrid)
#
# Usage (sur chicoree-1, dans le job OAR) :
#   cd ~/internship/snn
#   bash grid5k/chicoree_hybrid_4gpu.sh --background
#
#   bash grid5k/chicoree_hybrid_4gpu.sh              # foreground
#   bash grid5k/chicoree_hybrid_4gpu.sh --resume     # reprise last.pt
#   bash grid5k/chicoree_hybrid_4gpu.sh --stop-all
#
# Suivi :
#   tail -f outputs/chicoree_hybrid_4gpu/LATEST/gpu0_hp.log
#   nvidia-smi
#
# Variables :
#   GRID_CAMPAIGN=4-384-400  depth=4, embed_dim=384, 400 ep (save/grid/cifar10_4-384-400/)
#   EMBED_DIM=512            legacy si GRID_CAMPAIGN vide (depth=2)
#   EPOCHS=310               override epochs si GRID_CAMPAIGN vide
#   FRESH=1  FORCE=0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNN_ROOT="$(dirname "$SCRIPT_DIR")"
HPST="${SNN_ROOT}/HPSTAtten"
GRID_CAMPAIGN="${GRID_CAMPAIGN:-}"
EMBED_DIM="${EMBED_DIM:-512}"
if [[ -n "$GRID_CAMPAIGN" ]]; then
  EPOCHS="${EPOCHS:-400}"
else
  EPOCHS="${EPOCHS:-310}"
fi
FRESH="${FRESH:-1}"
BACKGROUND=0
FORCE="${FORCE:-0}"
STOP_ALL=0

# gpu|make_suffix|log_suffix
TARGETS=(
  "0|hp|hp"
  "1|hp-linear|hp_linear"
  "2|contrast|contrast"
  "3|contrast-sdt|contrast_sdt"
)

usage() {
  sed -n '2,21p' "$0" | sed 's/^# \?//'
}

stop_all_trainers() {
  local pattern="${SNN_ROOT}/.venv/bin/python -m scripts.train"
  local pids
  pids=$(pgrep -f "$pattern" 2>/dev/null || true)
  if [[ -z "$pids" ]]; then
    echo "Aucun scripts.train en cours sur $(hostname)."
    return 0
  fi
  echo "Arrêt PIDs: ${pids//$'\n'/ }"
  kill $pids 2>/dev/null || true
  sleep 3
  pids=$(pgrep -f "$pattern" 2>/dev/null || true)
  [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null || true
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --background|-b) BACKGROUND=1; shift ;;
    --resume)        FRESH=0; shift ;;
    --fresh)         FRESH=1; shift ;;
    --stop-all)      STOP_ALL=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *)
      echo "Option inconnue : $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$STOP_ALL" -eq 1 ]]; then
  stop_all_trainers
  exit 0
fi

if [[ "$FRESH" == 1 ]]; then
  MAKE_PREFIX="grid-fresh"
else
  MAKE_PREFIX="grid"
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${SNN_ROOT}/outputs/chicoree_hybrid_4gpu/${RUN_TAG}"

require_gpu_node() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERREUR: lancez sur un nœud GPU (chicoree-*), pas flille." >&2
    exit 1
  fi
  if ! "${SNN_ROOT}/.venv/bin/python" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERREUR: torch.cuda indisponible sur $(hostname)." >&2
    exit 1
  fi
}

preflight_gpus() {
  [[ "$FORCE" == 1 ]] && return 0
  local gpu pid busy=0
  for gpu in 0 1 2 3; do
    while IFS= read -r pid; do
      [[ -z "$pid" ]] && continue
      busy=1
      echo "  GPU ${gpu}: PID ${pid} déjà actif" >&2
    done < <(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
  done
  if [[ "$busy" -eq 1 ]]; then
    echo "ERREUR: GPU(s) occupés. FORCE=1 ou --stop-all" >&2
    exit 1
  fi
}

write_meta() {
  mkdir -p "$LOG_ROOT"
  cat >"${LOG_ROOT}/status.txt" <<EOF
run_tag=${RUN_TAG}
hostname=$(hostname)
started=$(date -Iseconds)
fresh=${FRESH}
grid_campaign=${GRID_CAMPAIGN:-}
embed_dim=${EMBED_DIM}
epochs=${EPOCHS}
log_root=${LOG_ROOT}
gpu0=${MAKE_PREFIX}-hp
gpu1=${MAKE_PREFIX}-hp-linear
gpu2=${MAKE_PREFIX}-contrast
gpu3=${MAKE_PREFIX}-contrast-sdt
EOF
  ln -sfn "$LOG_ROOT" "${SNN_ROOT}/outputs/chicoree_hybrid_4gpu/LATEST"
}

run_one() {
  local gpu="$1" suffix="$2" log_name="$3"
  local target="${MAKE_PREFIX}-${suffix}"
  local log="${LOG_ROOT}/gpu${gpu}_${log_name}.log"
  local -a make_args=(DATASET=cifar10 EXTRA_ARGS="--epochs ${EPOCHS}")

  if [[ -n "$GRID_CAMPAIGN" ]]; then
    make_args+=("GRID_CAMPAIGN=${GRID_CAMPAIGN}")
  else
    make_args+=("EMBED_DIM=${EMBED_DIM}")
  fi

  {
    echo "============================================================"
    echo "GPU ${gpu} | ${target} | $(hostname) | $(date -Iseconds)"
    echo "GRID_CAMPAIGN=${GRID_CAMPAIGN:-<none>} EMBED_DIM=${EMBED_DIM} EPOCHS=${EPOCHS}"
    echo "============================================================"
    cd "$HPST"
    make "${target}" "${make_args[@]}"
    echo "=== OK ${target} $(date -Iseconds) ==="
  } >>"$log" 2>&1
}

launch_workers() {
  require_gpu_node
  preflight_gpus
  write_meta

  echo "Run : ${RUN_TAG}"
  echo "Logs : ${LOG_ROOT}/"
  nvidia-smi -L || true
  echo ""

  local fail=0
  local -a pids=()

  for entry in "${TARGETS[@]}"; do
    IFS='|' read -r gpu suffix log_name <<< "$entry"
    CUDA_VISIBLE_DEVICES="$gpu" run_one "$gpu" "$suffix" "$log_name" &
    pids+=($!)
    echo "$!" >"${LOG_ROOT}/gpu${gpu}.pid"
    echo "GPU ${gpu} → ${MAKE_PREFIX}-${suffix} (PID ${pids[-1]})"
  done

  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done

  {
    echo "finished=$(date -Iseconds)"
    echo "exit_code=${fail}"
  } >>"${LOG_ROOT}/status.txt"

  if [[ "$fail" -eq 0 ]]; then
    echo "Les 4 runs hybrides sont terminés."
  else
    echo "Au moins un run a échoué — voir ${LOG_ROOT}/gpu*.log" >&2
    exit 1
  fi
}

if [[ "$BACKGROUND" -eq 1 ]]; then
  mkdir -p "$LOG_ROOT"
  launcher_log="${LOG_ROOT}/launcher.log"
  nohup env RUN_TAG="$RUN_TAG" FRESH="$FRESH" GRID_CAMPAIGN="$GRID_CAMPAIGN" \
    EMBED_DIM="$EMBED_DIM" EPOCHS="$EPOCHS" \
    bash "$0" >>"$launcher_log" 2>&1 &
  child=$!
  disown "$child" 2>/dev/null || true
  ln -sfn "$LOG_ROOT" "${SNN_ROOT}/outputs/chicoree_hybrid_4gpu/LATEST"
  sleep 2
  echo "Lancé en arrière-plan (1 terminal suffit)."
  echo "  Logs : outputs/chicoree_hybrid_4gpu/LATEST/"
  echo "  Suivi : tail -f outputs/chicoree_hybrid_4gpu/LATEST/gpu0_hp.log"
  echo "  GPU   : nvidia-smi"
  exit 0
fi

launch_workers
