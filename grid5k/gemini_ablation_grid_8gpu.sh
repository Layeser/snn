#!/usr/bin/env bash
# Ablation grille 4×2 complète — 8 runs en parallèle (1 variante / GPU).
#
# 4 attention_mode × 2 hybrid_qkv = 8 cases (make grid-fresh-*).
# Plan (gemini, 8× V100) — vague CIFAR-10 puis vague CIFAR-10-DVS :
#   GPU 0 → grid-fresh-hp
#   GPU 1 → grid-fresh-hp-linear
#   GPU 2 → grid-fresh-contrast
#   GPU 3 → grid-fresh-contrast-sdt
#   GPU 4 → grid-fresh-statten
#   GPU 5 → grid-fresh-sdt
#   GPU 6 → grid-fresh-contrast-binary
#   GPU 7 → grid-fresh-contrast-sdt-binary
#
# Usage (sur le nœud GPU, ex. gemini-2) :
#   cd ~/internship/snn
#   bash grid5k/gemini_ablation_grid_8gpu.sh --background
#   bash grid5k/gemini_ablation_grid_8gpu.sh --resume
#   bash grid5k/gemini_ablation_grid_8gpu.sh --stop-all
#
# Variables :
#   GRID_CAMPAIGN=2-512-310        campagne CIFAR-10 ( défaut )
#   GRID_CAMPAIGN_DVS=2-512-310    campagne DVS ( défaut = même )
#   DATASETS=cifar10               une seule vague (pas DVS)
#   FRESH=1|0  FORCE=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNN_ROOT="$(dirname "$SCRIPT_DIR")"
HPST="${SNN_ROOT}/HPSTAtten"
NUM_GPUS=8

GRID_CAMPAIGN="${GRID_CAMPAIGN:-2-512-310}"
GRID_CAMPAIGN_DVS="${GRID_CAMPAIGN_DVS:-${GRID_CAMPAIGN}}"
DATASETS="${DATASETS:-cifar10 cifar10-dvs}"
FRESH="${FRESH:-1}"
BACKGROUND=0
FORCE="${FORCE:-0}"
STOP_ALL=0

# Ordre fixe = mapping GPU 0..7 (grille 4×2 Makefile)
GRID_TARGETS=(
  grid-fresh-hp
  grid-fresh-hp-linear
  grid-fresh-contrast
  grid-fresh-contrast-sdt
  grid-fresh-statten
  grid-fresh-sdt
  grid-fresh-contrast-binary
  grid-fresh-contrast-sdt-binary
)

usage() {
  sed -n '2,28p' "$0" | sed 's/^# \?//'
}

stop_all_trainers() {
  local pattern="${SNN_ROOT}/.venv/bin/python -m scripts.train"
  local pids
  pids=$(pgrep -f "$pattern" 2>/dev/null || true)
  if [[ -z "$pids" ]]; then
    echo "Aucun entraînement scripts.train en cours sur $(hostname)."
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv 2>/dev/null || true
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
  : # targets déjà grid-fresh-*
else
  GRID_TARGETS=(
    grid-hp grid-hp-linear grid-contrast grid-contrast-sdt
    grid-statten grid-sdt grid-contrast-binary grid-contrast-sdt-binary
  )
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${SNN_ROOT}/outputs/gemini_ablation_grid_8gpu/${RUN_TAG}"

require_gpu_node() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERREUR: nvidia-smi absent — lancez sur un nœud GPU (gemini-*)." >&2
    exit 1
  fi
  local count
  count=$(nvidia-smi -L 2>/dev/null | wc -l)
  if [[ "$count" -lt "$NUM_GPUS" ]]; then
    echo "ERREUR: ${NUM_GPUS} GPU requis, ${count} détecté(s)." >&2
    exit 1
  fi
  if ! "${SNN_ROOT}/.venv/bin/python" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERREUR: torch.cuda indisponible sur $(hostname)." >&2
    exit 1
  fi
}

preflight_gpus() {
  [[ "$FORCE" == 1 ]] && return 0
  local gpu count pid line busy=0
  for gpu in $(seq 0 $((NUM_GPUS - 1))); do
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
    echo "  bash grid5k/gemini_ablation_grid_8gpu.sh --stop-all" >&2
    echo "Ou FORCE=1 bash $0 --background" >&2
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
grid_campaign_cifar10=${GRID_CAMPAIGN}
grid_campaign_dvs=${GRID_CAMPAIGN_DVS}
datasets=${DATASETS}
log_root=${LOG_ROOT}
mapping=GPU0..7 -> ${GRID_TARGETS[*]}
EOF
  ln -sfn "$LOG_ROOT" "${SNN_ROOT}/outputs/gemini_ablation_grid_8gpu/LATEST"
}

campaign_for_dataset() {
  local ds="$1"
  if [[ "$ds" == "cifar10-dvs" ]]; then
    echo "$GRID_CAMPAIGN_DVS"
  else
    echo "$GRID_CAMPAIGN"
  fi
}

run_single() {
  local gpu="$1"
  local dataset="$2"
  local target="$3"
  local campaign="$4"
  local wave_log="${LOG_ROOT}/${dataset}/gpu${gpu}.log"
  local -a make_args=(DATASET="${dataset}")

  if [[ -n "$campaign" ]]; then
    make_args+=("GRID_CAMPAIGN=${campaign}")
  fi

  mkdir -p "${LOG_ROOT}/${dataset}"
  {
    echo "============================================================"
    echo "GPU ${gpu} | ${dataset} | ${target} | $(hostname) | $(date -Iseconds)"
    echo "GRID_CAMPAIGN=${campaign}"
    echo "============================================================"
    cd "$HPST"
    if make "${target}" "${make_args[@]}"; then
      echo "--- OK ${target} $(date -Iseconds) ---"
    else
      echo "--- FAIL ${target} $(date -Iseconds) ---"
      exit 1
    fi
    echo "=== EXPERIENCE TERMINEE AVEC SUCCES ==="
  } >>"$wave_log" 2>&1
}

launch_wave() {
  local dataset="$1"
  local campaign="$2"
  local fail=0
  local gpu target pid

  echo ""
  echo "========== Vague ${dataset} (GRID_CAMPAIGN=${campaign}) =========="
  mkdir -p "${LOG_ROOT}/${dataset}"

  for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    target="${GRID_TARGETS[$gpu]}"
    CUDA_VISIBLE_DEVICES="${gpu}" run_single "${gpu}" "${dataset}" "${target}" "${campaign}" &
    pid=$!
    echo "${pid}" >"${LOG_ROOT}/${dataset}/gpu${gpu}.pid"
    echo "  GPU ${gpu} → ${target}  PID ${pid}"
  done

  for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    read -r pid <"${LOG_ROOT}/${dataset}/gpu${gpu}.pid"
    if ! wait "${pid}"; then
      fail=1
      echo "Échec GPU ${gpu} (${dataset}) — voir ${LOG_ROOT}/${dataset}/gpu${gpu}.log" >&2
    fi
  done

  {
    echo "wave_${dataset}_finished=$(date -Iseconds)"
    echo "wave_${dataset}_exit_code=${fail}"
  } >>"${LOG_ROOT}/status.txt"

  return "${fail}"
}

launch_workers() {
  local ds fail=0
  require_gpu_node
  preflight_gpus
  write_meta

  echo "Run : ${RUN_TAG}"
  echo "Logs : ${LOG_ROOT}/<dataset>/gpu{0..7}.log"
  nvidia-smi -L || true

  for ds in ${DATASETS}; do
    if ! launch_wave "${ds}" "$(campaign_for_dataset "${ds}")"; then
      fail=1
    fi
  done

  {
    echo "finished=$(date -Iseconds)"
    echo "exit_code=${fail}"
  } >>"${LOG_ROOT}/status.txt"

  if [[ "$fail" -eq 0 ]]; then
    echo "Toutes les vagues terminées avec succès."
  else
    echo "Au moins un run a échoué — voir ${LOG_ROOT}/*/gpu*.log" >&2
    exit 1
  fi
}

if [[ "$BACKGROUND" -eq 1 ]]; then
  mkdir -p "$LOG_ROOT"
  launcher_log="${LOG_ROOT}/launcher.log"
  nohup env RUN_TAG="$RUN_TAG" FRESH="$FRESH" FORCE="$FORCE" \
    GRID_CAMPAIGN="$GRID_CAMPAIGN" GRID_CAMPAIGN_DVS="$GRID_CAMPAIGN_DVS" \
    DATASETS="$DATASETS" \
    bash "$0" >>"$launcher_log" 2>&1 &
  child=$!
  disown "$child" 2>/dev/null || true
  ln -sfn "$LOG_ROOT" "${SNN_ROOT}/outputs/gemini_ablation_grid_8gpu/LATEST"
  sleep 2
  if grep -q "^ERREUR:" "$launcher_log" 2>/dev/null; then
    echo "Échec au démarrage — voir ${launcher_log}" >&2
    cat "$launcher_log" >&2
    exit 1
  fi
  echo "Lancé en arrière-plan (grille 4×2, 8 GPU par vague)."
  echo "  Logs : ${LOG_ROOT}/"
  echo "  Suivi : tail -f ${LOG_ROOT}/cifar10/gpu0.log"
  echo "  GPU   : sleep 20 && nvidia-smi"
  exit 0
fi

launch_workers
