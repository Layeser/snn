#!/usr/bin/env bash
# MK-HGR ablation CIFAR-10 @ D=512 — 4 runs séquentiels (1 GPU).
#
# Usage :
#   bash grid5k/chicoree_mk_hgr_ablation.sh
#   RUN=2 bash grid5k/chicoree_mk_hgr_ablation.sh
#   GPU=0 RUN=all bash grid5k/chicoree_mk_hgr_ablation.sh
#
# Voir Notes/mk_hgr_cifar10_next_steps.md
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HPST="${PROJECT_DIR}/HPSTAtten"
LOG="${PROJECT_DIR}/outputs/chicoree_mk_hgr_ablation"
GPU="${GPU:-0}"
RUN="${RUN:-all}"

mkdir -p "$LOG"

run_make() {
    local id="$1" target="$2" extra="${3:-}"
    echo ""
    echo "========================================"
    echo " Run $id — make $target"
    echo " Log: $LOG/run${id}.out"
    echo "========================================"
    (
        export CUDA_VISIBLE_DEVICES="$GPU"
        cd "$HPST"
        make "$target" DATASET=cifar10 EMBED_DIM=512 $extra
    ) >"$LOG/run${id}.out" 2>"$LOG/run${id}.err"
    local rc=$?
    if [[ "$rc" -eq 0 ]]; then
        echo "OK  run $id ($target)"
    else
        echo "FAIL run $id — tail -30 $LOG/run${id}.err"
    fi
    return "$rc"
}

fail=0

if [[ "$RUN" == "all" || "$RUN" == "1" ]]; then
    run_make 1 grid-fresh-factorized-hgr || fail=1
fi

if [[ "$RUN" == "all" || "$RUN" == "2" ]]; then
    run_make 2 grid-fresh-mk-hgr-binary || fail=1
fi

if [[ "$RUN" == "all" || "$RUN" == "3" ]]; then
    run_make 3 grid-fresh-mk-hgr-binary \
        'EXTRA_ARGS="--batch-size 32 --save-dir save/grid/cifar10/mk_hgr_binary_b32"' || fail=1
fi

if [[ "$RUN" == "all" || "$RUN" == "4" ]]; then
    run_make 4 grid-mk-hgr-triple-binary || fail=1
fi

if [[ "$RUN" != "all" && "$RUN" != "1" && "$RUN" != "2" && "$RUN" != "3" && "$RUN" != "4" ]]; then
    echo "RUN doit être : all | 1 | 2 | 3 | 4" >&2
    exit 1
fi

exit "$fail"
