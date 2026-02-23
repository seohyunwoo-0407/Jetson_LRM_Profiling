#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
# sweep_bottleneck.sh — 붕괴 지점 탐색 스윕
#
# 목적: context length & gen_length를 단계적으로 키워서
#       OOM / EMC 포화 / 성능 급락 지점을 찾음.
#
# 사용법:
#   bash scripts/sweep_bottleneck.sh experiments/exp1_baseline/run.py
#   bash scripts/sweep_bottleneck.sh experiments/exp5_baseline_eviction/run.py
# ═══════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

EXP_SCRIPT="${1:?Usage: $0 <exp_script.py> [args...]}"
shift
EXTRA_ARGS="${@:-}"

TAG=${TAG:-"sweep_$(date +%Y%m%d_%H%M%S)"}
OUTDIR="results/sweep_${TAG}"
mkdir -p "$OUTDIR"

# ── 스윕 파라미터 ──
CTX_LIST=(512 1024 2048 4096 8192)
GEN_LIST=(64 128 256 512 1024)

echo "═══════════════════════════════════════════════════"
echo " Bottleneck Sweep: $EXP_SCRIPT"
echo " CTX: ${CTX_LIST[*]}"
echo " GEN: ${GEN_LIST[*]}"
echo " Output → $OUTDIR"
echo "═══════════════════════════════════════════════════"

for CTX in "${CTX_LIST[@]}"; do
    for GEN in "${GEN_LIST[@]}"; do
        RUN_TAG="${TAG}_ctx${CTX}_gen${GEN}"
        echo ""
        echo "▶ CTX=$CTX GEN=$GEN"

        timeout 300 python3 "$EXP_SCRIPT" \
            --max_context "$CTX" \
            --gen_length "$GEN" \
            --repeats 2 \
            --warmup 1 \
            --output_dir "$OUTDIR" \
            --tag "$RUN_TAG" \
            --tegrastats \
            $EXTRA_ARGS \
            2>&1 | tee "$OUTDIR/${RUN_TAG}.log" || {
                echo "  ⚠️ FAILED or TIMEOUT (CTX=$CTX, GEN=$GEN) — possible OOM/crash"
                echo "CTX=$CTX,GEN=$GEN,STATUS=FAIL" >> "$OUTDIR/sweep_summary.csv"
                continue
            }

        echo "CTX=$CTX,GEN=$GEN,STATUS=OK" >> "$OUTDIR/sweep_summary.csv"
    done
done

echo ""
echo "═══════════════════════════════════════════════════"
echo " ✅ Sweep completed! Summary → $OUTDIR/sweep_summary.csv"
echo "═══════════════════════════════════════════════════"

