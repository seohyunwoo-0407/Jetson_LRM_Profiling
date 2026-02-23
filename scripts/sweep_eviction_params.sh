#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
# sweep_eviction_params.sh — Eviction 파라미터 스윕 (top_k / window / freq)
#
# 사용법:
#   bash scripts/sweep_eviction_params.sh experiments/exp5_baseline_eviction/run.py
# ═══════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

EXP_SCRIPT="${1:?Usage: $0 <exp_script.py> [args...]}"
shift
EXTRA_ARGS="${@:-}"

TAG=${TAG:-"esweep_$(date +%Y%m%d_%H%M%S)"}
CTX=${CTX:-2048}
GEN=${GEN:-256}
OUTDIR="results/eviction_sweep_${TAG}"
mkdir -p "$OUTDIR"

TOPK_LIST=(64 128 256 512)
WINDOW_LIST=(16 32 64 128)
FREQ_LIST=(1 2 4 8)

echo "═══════════════════════════════════════════════════"
echo " Eviction Param Sweep: $EXP_SCRIPT"
echo " TOPK: ${TOPK_LIST[*]}"
echo " WINDOW: ${WINDOW_LIST[*]}"
echo " FREQ: ${FREQ_LIST[*]}"
echo "═══════════════════════════════════════════════════"

for TOPK in "${TOPK_LIST[@]}"; do
    for WINDOW in "${WINDOW_LIST[@]}"; do
        for FREQ in "${FREQ_LIST[@]}"; do
            RUN_TAG="${TAG}_tk${TOPK}_w${WINDOW}_f${FREQ}"
            echo ""
            echo "▶ top_k=$TOPK window=$WINDOW freq=$FREQ"

            timeout 180 python3 "$EXP_SCRIPT" \
                --max_context "$CTX" \
                --gen_length "$GEN" \
                --repeats 2 \
                --warmup 1 \
                --eviction_top_k "$TOPK" \
                --eviction_window "$WINDOW" \
                --eviction_freq "$FREQ" \
                --output_dir "$OUTDIR" \
                --tag "$RUN_TAG" \
                --tegrastats \
                $EXTRA_ARGS \
                2>&1 | tee "$OUTDIR/${RUN_TAG}.log" || {
                    echo "  ⚠️ FAILED (top_k=$TOPK window=$WINDOW freq=$FREQ)"
                    echo "TOPK=$TOPK,WINDOW=$WINDOW,FREQ=$FREQ,STATUS=FAIL" >> "$OUTDIR/param_sweep_summary.csv"
                    continue
                }

            echo "TOPK=$TOPK,WINDOW=$WINDOW,FREQ=$FREQ,STATUS=OK" >> "$OUTDIR/param_sweep_summary.csv"
        done
    done
done

echo ""
echo "═══════════════════════════════════════════════════"
echo " ✅ Eviction sweep done! Summary → $OUTDIR/param_sweep_summary.csv"
echo "═══════════════════════════════════════════════════"

