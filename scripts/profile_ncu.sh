#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
# profile_ncu.sh — ncu (Nsight Compute) attention 커널 프로파일링
#
# 사용법:
#   bash scripts/profile_ncu.sh experiments/exp1_baseline/run.py [추가 인자...]
#
# 주의: ncu는 매우 느림 → gen_length를 작게 (16~32) 권장
# ═══════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

EXP_SCRIPT="${1:?Usage: $0 <exp_script.py> [args...]}"
shift
EXTRA_ARGS="$@"

TAG=${TAG:-"ncu_$(date +%Y%m%d_%H%M%S)"}
CTX=${CTX:-512}
GEN=${GEN:-16}
OUTDIR="results/ncu_${TAG}"
mkdir -p "$OUTDIR"

BASENAME=$(basename "$(dirname "$EXP_SCRIPT")")
REPORT_NAME="${BASENAME}_ctx${CTX}_gen${GEN}_${TAG}"

echo "═══════════════════════════════════════════════════"
echo " ncu profiling: $EXP_SCRIPT"
echo " Report → $OUTDIR/${REPORT_NAME}.ncu-rep"
echo "═══════════════════════════════════════════════════"

# attention 커널만 타겟 (이름에 "attn", "attention", "sdpa", "flash" 포함)
ncu \
    --set full \
    --kernel-name "regex:attn|attention|sdpa|flash|gemm" \
    --launch-skip 5 \
    --launch-count 20 \
    --output "$OUTDIR/${REPORT_NAME}" \
    --force-overwrite \
    python3 "$EXP_SCRIPT" \
        --max_context "$CTX" \
        --gen_length "$GEN" \
        --repeats 1 \
        --warmup 0 \
        --output_dir "$OUTDIR" \
        --tag "$TAG" \
        $EXTRA_ARGS

echo ""
echo "✅ ncu report: $OUTDIR/${REPORT_NAME}.ncu-rep"
echo "   열기: ncu-ui $OUTDIR/${REPORT_NAME}.ncu-rep"

