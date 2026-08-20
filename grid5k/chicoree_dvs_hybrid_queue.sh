#!/usr/bin/env bash
# File DVS hybride : 8 runs (2-512 + 4-384 × 4 variantes), 1 GPU libre → 1 job.
#
# Attend qu'un GPU se libère (runs CIFAR-10 en cours OK), lance le suivant, boucle
# jusqu'à épuisement de la file. Persiste l'état pour reprise après crash.
#
# Usage (sur chicoree-*, dans le job OAR) :
#   cd ~/internship/snn
#   bash grid5k/chicoree_dvs_hybrid_queue.sh --background
#
#   bash grid5k/chicoree_dvs_hybrid_queue.sh --status
#   bash grid5k/chicoree_dvs_hybrid_queue.sh --stop-scheduler
#
# Variables : POLL_SEC=30  FRESH=1  EPOCHS=400
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNN_ROOT="$(dirname "$SCRIPT_DIR")"
HPST="${SNN_ROOT}/HPSTAtten"
EPOCHS="${EPOCHS:-400}"
FRESH="${FRESH:-1}"
POLL_SEC="${POLL_SEC:-30}"
BACKGROUND=0
STOP_SCHEDULER=0
SHOW_STATUS=0
RESET_QUEUE=0

# job_id|GRID_CAMPAIGN|make_suffix|log_slug
JOBS=(
  "dvs_2-512_hp|2-512-400|hp|hp"
  "dvs_2-512_hp_linear|2-512-400|hp-linear|hp_linear"
  "dvs_2-512_contrast|2-512-400|contrast|contrast"
  "dvs_2-512_contrast_sdt|2-512-400|contrast-sdt|contrast_sdt"
  "dvs_4-384_hp|4-384-400|hp|hp"
  "dvs_4-384_hp_linear|4-384-400|hp-linear|hp_linear"
  "dvs_4-384_contrast|4-384-400|contrast|contrast"
  "dvs_4-384_contrast_sdt|4-384-400|contrast-sdt|contrast_sdt"
)

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${SNN_ROOT}/outputs/chicoree_dvs_hybrid_queue/${RUN_TAG}"
STATE_DIR="${SNN_ROOT}/outputs/chicoree_dvs_hybrid_queue"
STATE_FILE="${STATE_DIR}/queue.state"
LOCK_FILE="${STATE_DIR}/scheduler.lock"
LATEST_LINK="${STATE_DIR}/LATEST"

usage() {
  sed -n '2,15p' "$0" | sed 's/^# \?//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --background|-b) BACKGROUND=1; shift ;;
    --status)        SHOW_STATUS=1; shift ;;
    --stop-scheduler) STOP_SCHEDULER=1; shift ;;
    --reset)         RESET_QUEUE=1; shift ;;
    --fresh)         FRESH=1; shift ;;
    --resume)        FRESH=0; shift ;;
    -h|--help)       usage; exit 0 ;;
    *)
      echo "Option inconnue : $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

make_prefix() {
  [[ "$FRESH" == 1 ]] && echo "grid-fresh" || echo "grid"
}

save_dir_for_job() {
  local campaign="$1" suffix="$2"
  local subdir
  case "$suffix" in
    hp) subdir=hp_factorized_hybrid ;;
    hp-linear) subdir=hp_sdt_hybrid ;;
    contrast) subdir=hp_contrast_hybrid ;;
    contrast-sdt) subdir=hp_contrast_sdt_hybrid ;;
    *) echo "suffix inconnu: $suffix" >&2; return 1 ;;
  esac
  echo "${HPST}/save/grid/cifar10-dvs_${campaign}/${subdir}"
}

gpu_is_free() {
  local gpu="$1"
  local line
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    return 1
  done < <(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
  return 0
}

init_state_if_missing() {
  mkdir -p "$STATE_DIR" "$LOG_ROOT"
  if [[ ! -f "$STATE_FILE" ]]; then
    : >"$STATE_FILE"
    local entry job_id campaign suffix log_slug
    for entry in "${JOBS[@]}"; do
      IFS='|' read -r job_id campaign suffix log_slug <<< "$entry"
      echo "${job_id}|pending||||" >>"$STATE_FILE"
    done
  fi
  ln -sfn "$LOG_ROOT" "$LATEST_LINK" 2>/dev/null || true
}

read_state_line() {
  local job_id="$1"
  grep "^${job_id}|" "$STATE_FILE" 2>/dev/null | head -1 || true
}

write_state_line() {
  local job_id="$1" status="$2" gpu="$3" pid="$4" started="$5" finished="$6"
  local tmp="${STATE_FILE}.tmp.$$"
  local found=0 line
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" == "${job_id}|"* ]]; then
      echo "${job_id}|${status}|${gpu}|${pid}|${started}|${finished}" >>"$tmp"
      found=1
    else
      [[ -n "$line" ]] && echo "$line" >>"$tmp"
    fi
  done <"$STATE_FILE"
  if [[ "$found" -eq 0 ]]; then
    echo "${job_id}|${status}|${gpu}|${pid}|${started}|${finished}" >>"$tmp"
  fi
  mv "$tmp" "$STATE_FILE"
}

show_status() {
  init_state_if_missing
  echo "État file DVS hybride — $(hostname)"
  echo "State : $STATE_FILE"
  echo ""
  printf "%-24s %-10s %-4s %-8s %s\n" "JOB" "STATUS" "GPU" "PID" "started/finished"
  local line job_id status gpu pid started finished
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" ]] && continue
    IFS='|' read -r job_id status gpu pid started finished <<< "$line"
    printf "%-24s %-10s %-4s %-8s %s → %s\n" "$job_id" "$status" "${gpu:--}" "${pid:--}" "${started:--}" "${finished:--}"
  done <"$STATE_FILE"
  echo ""
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
  fi
}

stop_scheduler() {
  if [[ -f "$LOCK_FILE" ]]; then
    local spid
    spid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
    if [[ -n "$spid" ]] && kill -0 "$spid" 2>/dev/null; then
      echo "Arrêt planificateur PID ${spid}"
      kill "$spid" 2>/dev/null || true
    fi
    rm -f "$LOCK_FILE"
  else
    echo "Aucun planificateur actif (pas de lock)."
  fi
}

run_job_foreground() {
  local gpu="$1" job_id="$2" campaign="$3" suffix="$4"
  local target log
  target="$(make_prefix)-${suffix}"
  log="${LOG_ROOT}/${job_id}.log"

  {
    echo "============================================================"
    echo "JOB ${job_id} | GPU ${gpu} | ${target} | $(hostname) | $(date -Iseconds)"
    echo "DATASET=cifar10-dvs GRID_CAMPAIGN=${campaign} EPOCHS=${EPOCHS}"
    echo "SAVE_DIR=$(save_dir_for_job "$campaign" "$suffix")"
    echo "============================================================"
    cd "$HPST"
    make "${target}" \
      DATASET=cifar10-dvs \
      GRID_CAMPAIGN="${campaign}" \
      EXTRA_ARGS="--epochs ${EPOCHS}"
    echo "=== JOB OK ${job_id} $(date -Iseconds) ==="
  } >>"$log" 2>&1
}

reap_finished() {
  local -n _pids=$1
  local -n _jobs=$2
  local gpu pid job_id rc line status started
  for gpu in "${!_pids[@]}"; do
    pid="${_pids[$gpu]}"
    job_id="${_jobs[$gpu]}"
    [[ -z "$pid" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    rc=0
    wait "$pid" 2>/dev/null || rc=$?
    line="$(read_state_line "$job_id")"
    IFS='|' read -r _j status _g _p started _f <<< "$line"
    if [[ "$rc" -eq 0 ]] && grep -qF "=== JOB OK ${job_id}" "${LOG_ROOT}/${job_id}.log" 2>/dev/null; then
      write_state_line "$job_id" "done" "$gpu" "$pid" "$started" "$(date -Iseconds)"
      echo "$(date -Iseconds) DONE ${job_id} on GPU ${gpu}"
    else
      write_state_line "$job_id" "failed" "$gpu" "$pid" "$started" "$(date -Iseconds)"
      echo "$(date -Iseconds) FAIL ${job_id} on GPU ${gpu} (exit ${rc})" >&2
    fi
    unset "_pids[$gpu]" "_jobs[$gpu]"
  done
}

next_pending_job() {
  local line job_id status
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" ]] && continue
    IFS='|' read -r job_id status _ _ _ _ <<< "$line"
    if [[ "$status" == "pending" || "$status" == "failed" ]]; then
      echo "$line"
      return 0
    fi
  done <"$STATE_FILE"
  return 1
}

recover_running_jobs() {
  declare -n _pids=$1
  declare -n _jobs=$2
  local line job_id status gpu pid started
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" ]] && continue
    IFS='|' read -r job_id status gpu pid started _f <<< "$line"
    if [[ "$status" == "running" && -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      _pids["$gpu"]="$pid"
      _jobs["$gpu"]="$job_id"
      echo "Reprise suivi : ${job_id} GPU ${gpu} PID ${pid}"
    elif [[ "$status" == "running" ]]; then
      write_state_line "$job_id" "pending" "" "" "$started" ""
      echo "Job ${job_id} était running sans PID → remis en pending"
    fi
  done <"$STATE_FILE"
}

scheduler_loop() {
  require_gpu_node
  init_state_if_missing

  mkdir -p "$LOG_ROOT"
  cat >"${LOG_ROOT}/status.txt" <<EOF
run_tag=${RUN_TAG}
hostname=$(hostname)
started=$(date -Iseconds)
fresh=${FRESH}
epochs=${EPOCHS}
poll_sec=${POLL_SEC}
jobs_total=${#JOBS[@]}
log_root=${LOG_ROOT}
state_file=${STATE_FILE}
EOF
  ln -sfn "$LOG_ROOT" "$LATEST_LINK"

  echo $$ >"$LOCK_FILE"
  trap 'rm -f "$LOCK_FILE"; exit' EXIT INT TERM

  declare -A GPU_PID=() GPU_JOB=()
  recover_running_jobs GPU_PID GPU_JOB

  echo "Planificateur DVS hybride — ${#JOBS[@]} jobs, poll ${POLL_SEC}s"
  echo "Logs : ${LOG_ROOT}/"

  if [[ ! -d "${SNN_ROOT}/data/CIFAR10DVS" && ! -d "${SNN_ROOT}/data/cifar10dvs" && ! -d "${SNN_ROOT}/data/cifar10-dvs" ]]; then
    echo "Téléchargement CIFAR-10-DVS (si absent)..."
    "${SNN_ROOT}/.venv/bin/python" "${SNN_ROOT}/scripts/download_data.py" cifar10-dvs \
      --data-dir "${SNN_ROOT}/data" >>"${LOG_ROOT}/download_dvs.log" 2>&1 || true
  fi

  nvidia-smi -L || true

  local pending_count running_count line job_id campaign suffix log_slug gpu pid

  while true; do
    reap_finished GPU_PID GPU_JOB

    pending_count=0
    while IFS= read -r line || [[ -n "$line" ]]; do
      [[ -z "$line" ]] && continue
      IFS='|' read -r job_id status _ _ _ _ <<< "$line"
      [[ "$status" == "pending" || "$status" == "failed" ]] && pending_count=$((pending_count + 1))
    done <"$STATE_FILE"

    running_count=${#GPU_PID[@]}

    if [[ "$pending_count" -eq 0 && "$running_count" -eq 0 ]]; then
      echo "$(date -Iseconds) Tous les jobs DVS hybrides sont terminés."
      {
        echo "finished=$(date -Iseconds)"
        echo "exit_code=0"
      } >>"${LOG_ROOT}/status.txt"
      return 0
    fi

    for gpu in 0 1 2 3; do
      [[ -n "${GPU_PID[$gpu]:-}" ]] && continue
      gpu_is_free "$gpu" || continue
      line="$(next_pending_job)" || continue
      IFS='|' read -r job_id _status _g _p _s _f <<< "$line"
      campaign="" suffix=""
      for entry in "${JOBS[@]}"; do
        IFS='|' read -r jid camp suf _slug <<< "$entry"
        if [[ "$jid" == "$job_id" ]]; then
          campaign="$camp"
          suffix="$suf"
          break
        fi
      done
      [[ -n "$campaign" && -n "$suffix" ]] || continue

      write_state_line "$job_id" "running" "$gpu" "..." "$(date -Iseconds)" ""
      echo "$(date -Iseconds) START ${job_id} on GPU ${gpu}"
      CUDA_VISIBLE_DEVICES="$gpu" run_job_foreground "$gpu" "$job_id" "$campaign" "$suffix" &
      pid=$!
      GPU_PID["$gpu"]="$pid"
      GPU_JOB["$gpu"]="$job_id"
      write_state_line "$job_id" "running" "$gpu" "$pid" "$(date -Iseconds)" ""
    done

    if [[ "$pending_count" -gt 0 && ${#GPU_PID[@]} -eq 0 ]]; then
      echo "$(date -Iseconds) poll: ${pending_count} en attente, GPUs occupés — prochain essai dans ${POLL_SEC}s"
    fi

    sleep "$POLL_SEC"
  done
}

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

if [[ "$RESET_QUEUE" -eq 1 ]]; then
  rm -f "$STATE_FILE" "$LOCK_FILE"
  echo "File réinitialisée."
fi

if [[ "$SHOW_STATUS" -eq 1 ]]; then
  show_status
  exit 0
fi

if [[ "$STOP_SCHEDULER" -eq 1 ]]; then
  stop_scheduler
  exit 0
fi

if [[ "$BACKGROUND" -eq 1 ]]; then
  mkdir -p "$LOG_ROOT" "$STATE_DIR"
  # Nouvelle session : reset state seulement si pas de file en cours
  if [[ ! -f "$STATE_FILE" ]]; then
    init_state_if_missing
  else
    # Réutiliser state existant ; logs dans nouveau RUN_TAG pour ce lancement
    init_state_if_missing
  fi
  if [[ -f "$LOCK_FILE" ]] && kill -0 "$(cat "$LOCK_FILE" 2>/dev/null)" 2>/dev/null; then
    echo "Planificateur déjà actif (PID $(cat "$LOCK_FILE")). --status pour voir la file." >&2
    exit 1
  fi
  launcher_log="${LOG_ROOT}/scheduler.log"
  nohup env RUN_TAG="$RUN_TAG" FRESH="$FRESH" EPOCHS="$EPOCHS" POLL_SEC="$POLL_SEC" \
    bash "$0" >>"$launcher_log" 2>&1 &
  child=$!
  disown "$child" 2>/dev/null || true
  ln -sfn "$LOG_ROOT" "$LATEST_LINK"
  sleep 2
  echo "Planificateur DVS lancé en arrière-plan."
  echo "  Scheduler PID : ${child}"
  echo "  État file     : ${STATE_FILE}"
  echo "  Logs          : ${LOG_ROOT}/"
  echo "  Suivi         : bash grid5k/chicoree_dvs_hybrid_queue.sh --status"
  echo "  Jobs en cours : tail -f ${LOG_ROOT}/dvs_2-512_hp.log  (ex.)"
  exit 0
fi

scheduler_loop
