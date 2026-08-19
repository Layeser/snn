#!/usr/bin/env bash
# Ablation 4×2 — variantes HYBRIDE uniquement (4 modèles × 2 datasets).
#
# Plan GPU (gemini, 4× V100) :
#   GPU 0 : CIFAR-10      → grid-hp, grid-hp-linear        (séquentiel)
#   GPU 1 : CIFAR-10      → grid-contrast, grid-contrast-sdt
#   GPU 2 : CIFAR-10-DVS  → grid-hp, grid-hp-linear
#   GPU 3 : CIFAR-10-DVS  → grid-contrast, grid-contrast-sdt
#
# Usage (sur le nœud GPU, ex. gemini-2) :
#   cd ~/internship/snn
#   bash grid5k/gemini_ablation_hybrid_4gpu.sh              # foreground
#   bash grid5k/gemini_ablation_hybrid_4gpu.sh --background # détaché (PC éteint OK)
#   bash grid5k/gemini_ablation_hybrid_4gpu.sh --resume     # reprise last.pt
#   bash grid5k/gemini_ablation_hybrid_4gpu.sh --stop-all   # tuer tous les scripts.train
#
# Suivi :
#   tail -f outputs/gemini_ablation_hybrid/LATEST/gpu0.log
#   cat outputs/gemini_ablation_hybrid/LATEST/status.txt
#
# Variables :
#   GRID_CAMPAIGN=4-512-400  depth=4, embed_dim=512, 400 epochs (CIFAR-10 + DVS)
#   EMBED_DIM=512            (legacy, si GRID_CAMPAIGN vide)
#   FRESH=1|0
#   FORCE=1                 lancer même si un GPU a déjà des process CUDA
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNN_ROOT="$(dirname "$SCRIPT_DIR")"
HPST="${SNN_ROOT}/HPSTAtten"
GRID_CAMPAIGN="${GRID_CAMPAIGN:-4-512-400}"
EMBED_DIM="${EMBED_DIM:-512}"
FRESH="${FRESH:-1}"
BACKGROUND=0
FORCE="${FORCE:-0}"
STOP_ALL=0

usage() {
  sed -n '2,21p' "$0" | sed 's/^# \?//'
}

stop_all_trainers() {
  local pattern="${SNN_ROOT}/.venv/bin/python -m scripts.train"
  local pids
  pids=$(pgrep -f "$pattern" 2>/dev/null || true)
  if [[ -z "$pids" ]]; then
    echo "Aucun entraînement scripts.train en cours sur $(hostname)."
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q . \
      && nvidia-smi || true
    return 0
  fi
  echo "Arrêt des entraînements (PIDs: ${pids//$'\n'/ })..."
  kill $pids 2>/dev/null || true
  sleep 4
  pids=$(pgrep -f "$pattern" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "SIGKILL sur PIDs restants: ${pids//$'\n'/ }"
    kill -9 $pids 2>/dev/null || true
    sleep 1
  fi
  echo "GPU après arrêt :"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
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
  PREFIX="grid-fresh"
else
  PREFIX="grid"
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${SNN_ROOT}/outputs/gemini_ablation_hybrid/${RUN_TAG}"

require_gpu_node() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERREUR: nvidia-smi absent — lancez sur un nœud GPU (gemini-*), pas flyon/flille." >&2
    exit 1
  fi
  if ! "${SNN_ROOT}/.venv/bin/python" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERREUR: torch.cuda indisponible sur $(hostname)." >&2
    echo "  gemini/V100 : pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126" >&2
    exit 1
  fi
}

preflight_gpus() {
  [[ "$FORCE" == 1 ]] && return 0
  local gpu count pid line busy=0
  for gpu in 0 1 2 3; do
    count=0
    while IFS= read -r pid; do
      [[ -z "$pid" ]] && continue
      ((count++)) || true
      busy=1
      line=$(nvidia-smi -i "$gpu" --query-compute-apps=pid,used_gpu_memory --format=csv,noheader 2>/dev/null \
        | awk -F', ' -v p="$pid" '$1==p {print}')
      echo "  GPU ${gpu}: PID ${pid} (${line:-occupé})" >&2
    done < <(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
    if [[ "$count" -gt 0 ]]; then
      echo "ERREUR: GPU ${gpu} a déjà ${count} process(es) CUDA." >&2
    fi
  done
  if [[ "$busy" -eq 1 ]]; then
    echo "" >&2
    echo "Arrêtez les runs précédents avant de relancer, ex. :" >&2
    echo "  bash grid5k/gemini_ablation_hybrid_4gpu.sh --stop-all" >&2
    echo "  nvidia-smi" >&2
    echo "Ou relance forcée (risque OOM) : FORCE=1 bash $0 --background" >&2
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
grid_campaign=${GRID_CAMPAIGN}
embed_dim=${EMBED_DIM}
epochs=400
depth=4
log_root=${LOG_ROOT}
gpu0=${PREFIX}-hp, ${PREFIX}-hp-linear (cifar10)
gpu1=${PREFIX}-contrast, ${PREFIX}-contrast-sdt (cifar10)
gpu2=${PREFIX}-hp, ${PREFIX}-hp-linear (cifar10-dvs)
gpu3=${PREFIX}-contrast, ${PREFIX}-contrast-sdt (cifar10-dvs)
EOF
  ln -sfn "$LOG_ROOT" "${SNN_ROOT}/outputs/gemini_ablation_hybrid/LATEST"
}

run_gpu_queue() {
  local gpu="$1"
  local dataset="$2"
  shift 2
  local targets=("$@")
  local log="${LOG_ROOT}/gpu${gpu}.log"
  local -a make_args

  make_args=(DATASET="${dataset}")
  if [[ -n "$GRID_CAMPAIGN" ]]; then
    make_args+=("GRID_CAMPAIGN=${GRID_CAMPAIGN}")
  elif [[ "$dataset" == "cifar10" ]]; then
    make_args+=("EMBED_DIM=${EMBED_DIM}")
  fi

  {
    echo "============================================================"
    echo "GPU ${gpu} | ${dataset} | $(hostname) | $(date -Iseconds)"
    echo "Targets: ${targets[*]}"
    echo "============================================================"
    cd "$HPST"
    local target
    for target in "${targets[@]}"; do
      echo ""
      echo "--- START ${target} $(date -Iseconds) ---"
      if make "${target}" "${make_args[@]}"; then
        echo "--- OK ${target} $(date -Iseconds) ---"
      else
        echo "--- FAIL ${target} $(date -Iseconds) ---"
        exit 1
      fi
    done
    echo ""
    echo "GPU ${gpu} terminé."
    echo "=== EXPERIENCE TERMINEE AVEC SUCCES ==="
  } >>"$log" 2>&1
}

launch_workers() {
  require_gpu_node
  preflight_gpus
  write_meta

  echo "Run : ${RUN_TAG}"
  echo "Logs : ${LOG_ROOT}/gpu{0,1,2,3}.log"
  nvidia-smi -L || true
  echo ""

  CUDA_VISIBLE_DEVICES=0 run_gpu_queue 0 cifar10 \
    "${PREFIX}-hp" "${PREFIX}-hp-linear" &
  echo $! >"${LOG_ROOT}/gpu0.pid"

  CUDA_VISIBLE_DEVICES=1 run_gpu_queue 1 cifar10 \
    "${PREFIX}-contrast" "${PREFIX}-contrast-sdt" &
  echo $! >"${LOG_ROOT}/gpu1.pid"

  CUDA_VISIBLE_DEVICES=2 run_gpu_queue 2 cifar10-dvs \
    "${PREFIX}-hp" "${PREFIX}-hp-linear" &
  echo $! >"${LOG_ROOT}/gpu2.pid"

  CUDA_VISIBLE_DEVICES=3 run_gpu_queue 3 cifar10-dvs \
    "${PREFIX}-contrast" "${PREFIX}-contrast-sdt" &
  echo $! >"${LOG_ROOT}/gpu3.pid"

  local p0 p1 p2 p3 fail=0
  read -r p0 <"${LOG_ROOT}/gpu0.pid"
  read -r p1 <"${LOG_ROOT}/gpu1.pid"
  read -r p2 <"${LOG_ROOT}/gpu2.pid"
  read -r p3 <"${LOG_ROOT}/gpu3.pid"
  echo "PIDs : gpu0=$p0 gpu1=$p1 gpu2=$p2 gpu3=$p3"
  echo "$p0 $p1 $p2 $p3" >"${LOG_ROOT}/all.pids"

  wait "$p0" || fail=1
  wait "$p1" || fail=1
  wait "$p2" || fail=1
  wait "$p3" || fail=1

  {
    echo "finished=$(date -Iseconds)"
    echo "exit_code=${fail}"
  } >>"${LOG_ROOT}/status.txt"

  if [[ "$fail" -eq 0 ]]; then
    echo "Tous les runs terminés avec succès."
  else
    echo "Au moins un worker a échoué — voir ${LOG_ROOT}/gpu*.log" >&2
    exit 1
  fi
}

if [[ "$BACKGROUND" -eq 1 ]]; then
  mkdir -p "$LOG_ROOT"
  launcher_log="${LOG_ROOT}/launcher.log"
  nohup env RUN_TAG="$RUN_TAG" FRESH="$FRESH" GRID_CAMPAIGN="$GRID_CAMPAIGN" \
    EMBED_DIM="$EMBED_DIM" \
    bash "$0" >>"$launcher_log" 2>&1 &
  child=$!
  disown "$child" 2>/dev/null || true
  ln -sfn "$LOG_ROOT" "${SNN_ROOT}/outputs/gemini_ablation_hybrid/LATEST"
  sleep 2
  if grep -q "^ERREUR:" "$launcher_log" 2>/dev/null; then
    echo "Échec au démarrage — voir ${launcher_log}" >&2
    cat "$launcher_log" >&2
    exit 1
  fi
  echo "Lancé en arrière-plan."
  echo "  Wrapper PID : ${child} (se termine vite — normal)"
  echo "  Workers     : voir ${LOG_ROOT}/gpu*.pid après ~5 s"
  echo "  Logs        : ${LOG_ROOT}/"
  echo "  Suivi       : tail -f ${LOG_ROOT}/gpu0.log"
  echo "  GPU         : sleep 15 && nvidia-smi   # laisser le temps au chargement CUDA"
  exit 0
fi

launch_workers
