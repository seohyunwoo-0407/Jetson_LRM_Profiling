#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
# profile_nsys.sh — nsys 프로파일링 (NVTX 타임라인 + CUDA 커널)
#
# 사용법:
#   bash scripts/profile_nsys.sh exp1_baseline/run.py [추가 인자...]
#   bash scripts/profile_nsys.sh exp5_baseline_eviction/run.py --eviction_top_k 128
# ═══════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

EXP_SCRIPT="${1:?Usage: $0 <exp_script.py> [args...]}"
shift
EXTRA_ARGS="$@"

TAG=${TAG:-"nsys_$(date +%Y%m%d_%H%M%S)"}
CTX=${CTX:-2048}
GEN=${GEN:-128}
OUTDIR="results/nsys_${TAG}"
mkdir -p "$OUTDIR"

# 파일명: scheme_ctx_genlen_date
BASENAME=$(basename "$(dirname "$EXP_SCRIPT")")
REPORT_NAME="${BASENAME}_ctx${CTX}_gen${GEN}_${TAG}"

echo "═══════════════════════════════════════════════════"
echo " nsys profiling: $EXP_SCRIPT"
echo " Report → $OUTDIR/${REPORT_NAME}.nsys-rep"
echo "═══════════════════════════════════════════════════"

nsys profile \
    --trace=cuda,nvtx,osrt \
    --nvtx-capture=RUN \
    --sample=none \
    --output="$OUTDIR/${REPORT_NAME}" \
    --force-overwrite=true \
    --stats=true \
    python3 "$EXP_SCRIPT" \
        --max_context "$CTX" \
        --gen_length "$GEN" \
        --repeats 1 \
        --warmup 1 \
        --output_dir "$OUTDIR" \
        --tag "$TAG" \
        $EXTRA_ARGS

echo ""
echo "✅ nsys report: $OUTDIR/${REPORT_NAME}.nsys-rep"
echo "   열기: nsys-ui $OUTDIR/${REPORT_NAME}.nsys-rep"
echo "   CLI stats: nsys stats $OUTDIR/${REPORT_NAME}.nsys-rep"

