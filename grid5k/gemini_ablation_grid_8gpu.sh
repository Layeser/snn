#!/usr/bin/env bash
# Ablation grille 4×2 — jusqu'à 8 runs en parallèle (1 variante / GPU).
#
# 4 attention_mode × 2 hybrid_qkv = 8 cases (make grid-fresh-*).
# Si le nœud a < 8 GPU (ex. gemini-2 avec 6), les 8 variantes tournent
# en sous-vagues : 6 parallèles puis 2 parallèles.
#
# Usage (sur le nœud GPU, ex. gemini-2) :
#   cd ~/internship/snn
#   bash grid5k/gemini_ablation_grid_8gpu.sh --background
#   bash grid5k/gemini_ablation_grid_8gpu.sh --resume
#   bash grid5k/gemini_ablation_grid_8gpu.sh --stop-all
#
# Variables :
#   NUM_GPUS=auto                  auto-détecté via nvidia-smi (défaut)
#   GRID_CAMPAIGN=2-512-310        campagne CIFAR-10 ( défaut )
#   GRID_CAMPAIGN_DVS=2-512-310    campagne DVS ( défaut = même )
#   DATASETS=cifar10               une seule vague (pas DVS)
#   VARIANTS="6 7"                 sous-ensemble (v6=contrast-binary, v7=contrast_sdt-binary)
#   GPU_IDS="0 1"                  GPU physiques libres (défaut : 0..N-1)
#   FRESH=1|0  FORCE=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNN_ROOT="$(dirname "$SCRIPT_DIR")"
HPST="${SNN_ROOT}/HPSTAtten"
NUM_GPUS="${NUM_GPUS:-auto}"

GRID_CAMPAIGN="${GRID_CAMPAIGN:-2-512-310}"
GRID_CAMPAIGN_DVS="${GRID_CAMPAIGN_DVS:-${GRID_CAMPAIGN}}"
DATASETS="${DATASETS:-cifar10 cifar10-dvs}"
VARIANTS="${VARIANTS:-}"
GPU_IDS="${GPU_IDS:-}"
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
VARIANT_LIST=()
GPU_LIST=()

build_variant_list() {
  VARIANT_LIST=()
  if [[ -n "$VARIANTS" ]]; then
    local v
    for v in $VARIANTS; do
      if [[ "$v" -lt 0 || "$v" -ge 8 ]]; then
        echo "ERREUR: VARIANTS index invalide v${v} (attendu 0..7)." >&2
        exit 1
      fi
      VARIANT_LIST+=("$v")
    done
  else
    local i
    for ((i=0; i<${#GRID_TARGETS[@]}; i++)); do
      VARIANT_LIST+=("$i")
    done
  fi
}

build_gpu_list() {
  GPU_LIST=()
  if [[ -n "$GPU_IDS" ]]; then
    local g
    for g in $GPU_IDS; do
      GPU_LIST+=("$g")
    done
  else
    local g
    for ((g=0; g<NUM_GPUS; g++)); do
      GPU_LIST+=("$g")
    done
  fi
  if [[ ${#GPU_LIST[@]} -lt 1 ]]; then
    echo "ERREUR: aucun GPU dans GPU_IDS." >&2
    exit 1
  fi
}

detect_num_gpus() {
  if [[ "$NUM_GPUS" != "auto" ]]; then
    return 0
  fi
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
}

require_gpu_node() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERREUR: nvidia-smi absent — lancez sur un nœud GPU (gemini-*)." >&2
    exit 1
  fi
  detect_num_gpus
  if [[ "$NUM_GPUS" -lt 1 ]]; then
    echo "ERREUR: aucun GPU détecté." >&2
    exit 1
  fi
  if [[ "$NUM_GPUS" -lt "$NUM_VARIANTS" ]]; then
    echo "Note: ${NUM_GPUS} GPU(s) — ${NUM_VARIANTS} variantes en sous-vagues."
  fi
  if ! "${SNN_ROOT}/.venv/bin/python" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERREUR: torch.cuda indisponible sur $(hostname)." >&2
    exit 1
  fi
}

preflight_gpus() {
  [[ "$FORCE" == 1 ]] && return 0
  local gpu count pid line busy=0
  for gpu in "${GPU_LIST[@]}"; do
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
    echo "Ou GPU_IDS=\"4 5\" …  FORCE=1 bash $0 --background" >&2
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
num_gpus=${NUM_GPUS}
variants=${VARIANTS:-v${VARIANT_LIST[*]}}
gpu_ids=${GPU_IDS:-${GPU_LIST[*]}}
mapping=v0..v7 -> ${GRID_TARGETS[*]}
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
  local phys_gpu="$1"
  local variant_idx="$2"
  local dataset="$3"
  local target="$4"
  local campaign="$5"
  local wave_log="${LOG_ROOT}/${dataset}/v${variant_idx}.log"
  local -a make_args=(DATASET="${dataset}")

  if [[ -n "$campaign" ]]; then
    make_args+=("GRID_CAMPAIGN=${campaign}")
  fi

  mkdir -p "${LOG_ROOT}/${dataset}"
  {
    echo "============================================================"
    echo "variant v${variant_idx} | phys GPU ${phys_gpu} | ${dataset} | ${target} | $(hostname) | $(date -Iseconds)"
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
  local variant batch_start batch_end slot phys_gpu target pid
  local batch_size=${#GPU_LIST[@]}
  local num_selected=${#VARIANT_LIST[@]}

  echo ""
  echo "========== Vague ${dataset} (GRID_CAMPAIGN=${campaign}, GPU: ${GPU_LIST[*]}) =========="
  mkdir -p "${LOG_ROOT}/${dataset}"

  for ((batch_start=0; batch_start<num_selected; batch_start+=batch_size)); do
    batch_end=$((batch_start + batch_size - 1))
    if [[ "$batch_end" -ge "$num_selected" ]]; then
      batch_end=$((num_selected - 1))
    fi
    echo "  Sous-vague $(((batch_start + 1)))-$(($batch_end + 1))/${num_selected}"

    slot=0
    for ((i=batch_start; i<=batch_end; i++)); do
      variant="${VARIANT_LIST[$i]}"
      phys_gpu="${GPU_LIST[$slot]}"
      slot=$((slot + 1))
      target="${GRID_TARGETS[$variant]}"
      CUDA_VISIBLE_DEVICES="${phys_gpu}" run_single "${phys_gpu}" "${variant}" "${dataset}" "${target}" "${campaign}" &
      pid=$!
      echo "${pid}" >"${LOG_ROOT}/${dataset}/v${variant}.pid"
      echo "    v${variant} (GPU ${phys_gpu}) → ${target}  PID ${pid}"
    done

    slot=0
    for ((i=batch_start; i<=batch_end; i++)); do
      variant="${VARIANT_LIST[$i]}"
      read -r pid <"${LOG_ROOT}/${dataset}/v${variant}.pid"
      if ! wait "${pid}"; then
        fail=1
        echo "Échec v${variant} (${dataset}) — voir ${LOG_ROOT}/${dataset}/v${variant}.log" >&2
      fi
      slot=$((slot + 1))
    done
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
  build_variant_list
  build_gpu_list
  preflight_gpus
  write_meta

  echo "Run : ${RUN_TAG}"
  echo "Variantes : v${VARIANT_LIST[*]}"
  echo "Logs : ${LOG_ROOT}/<dataset>/v*.log"
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
  nohup env RUN_TAG="$RUN_TAG" FRESH="$FRESH" FORCE="$FORCE" NUM_GPUS="$NUM_GPUS" \
    GRID_CAMPAIGN="$GRID_CAMPAIGN" GRID_CAMPAIGN_DVS="$GRID_CAMPAIGN_DVS" \
    DATASETS="$DATASETS" VARIANTS="$VARIANTS" GPU_IDS="$GPU_IDS" \
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
  echo "Lancé en arrière-plan (grille 4×2, ${NUM_GPUS:-auto} GPU détectés)."
  echo "  Logs : ${LOG_ROOT}/"
  echo "  Suivi : tail -f ${LOG_ROOT}/cifar10-dvs/v0.log"
  echo "  GPU   : sleep 20 && nvidia-smi"
  exit 0
fi

launch_workers
