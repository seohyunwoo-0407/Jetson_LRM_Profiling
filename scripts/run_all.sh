#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
# run_all.sh — 전체 실험 순차 실행 (baseline → ToT → Debate → MCTS → eviction 버전)
# 사용법: bash scripts/run_all.sh [--ctx 2048] [--gen 256] [--repeats 5] [--tag v1]
# ═══════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── 기본값 ──
CTX=${CTX:-2048}
GEN=${GEN:-256}
REPEATS=${REPEATS:-5}
WARMUP=${WARMUP:-2}
TAG=${TAG:-"$(date +%Y%m%d_%H%M%S)"}
MODEL=${MODEL:-"ibm-granite/granite-3.1-3b-a800m-instruct"}
OUTDIR="results/ctx${CTX}_gen${GEN}_${TAG}"

# eviction 파라미터
EVICT_TOPK=${EVICT_TOPK:-256}
EVICT_WINDOW=${EVICT_WINDOW:-64}
EVICT_FREQ=${EVICT_FREQ:-1}

mkdir -p "$OUTDIR"

echo "═══════════════════════════════════════════════════"
echo " Jetson LRM Profiling — Full Experiment Suite"
echo " CTX=$CTX  GEN=$GEN  REPEATS=$REPEATS  TAG=$TAG"
echo " EVICT: topk=$EVICT_TOPK  window=$EVICT_WINDOW  freq=$EVICT_FREQ"
echo " Output → $OUTDIR"
echo "═══════════════════════════════════════════════════"

COMMON_ARGS="--model_name $MODEL --max_context $CTX --gen_length $GEN \
             --repeats $REPEATS --warmup $WARMUP --output_dir $OUTDIR --tag $TAG \
             --tegrastats --tegrastats_interval 100"

EVICT_ARGS="--eviction_top_k $EVICT_TOPK --eviction_window $EVICT_WINDOW --eviction_freq $EVICT_FREQ"

echo ""
echo "▶ [1/8] Exp1: Baseline"
python3 experiments/exp1_baseline/run.py $COMMON_ARGS 2>&1 | tee "$OUTDIR/exp1_baseline.log"

echo ""
echo "▶ [2/8] Exp2: ToT"
python3 experiments/exp2_tot/run.py $COMMON_ARGS --n_proposals 3 --depth 2 --mode bfs 2>&1 | tee "$OUTDIR/exp2_tot.log"

echo ""
echo "▶ [3/8] Exp3: Multi-Agent Debate"
python3 experiments/exp3_debate/run.py $COMMON_ARGS --n_rounds 3 2>&1 | tee "$OUTDIR/exp3_debate.log"

echo ""
echo "▶ [4/8] Exp4: MCTS"
python3 experiments/exp4_mcts/run.py $COMMON_ARGS --n_iterations 20 --n_rollouts 3 --max_depth 3 --rollout_length 64 2>&1 | tee "$OUTDIR/exp4_mcts.log"

echo ""
echo "▶ [5/8] Exp5: Baseline + Eviction"
python3 experiments/exp5_baseline_eviction/run.py $COMMON_ARGS $EVICT_ARGS 2>&1 | tee "$OUTDIR/exp5_baseline_eviction.log"

echo ""
echo "▶ [6/8] Exp6: ToT + Eviction"
python3 experiments/exp6_tot_eviction/run.py $COMMON_ARGS $EVICT_ARGS --n_proposals 3 --depth 2 --mode bfs 2>&1 | tee "$OUTDIR/exp6_tot_eviction.log"

echo ""
echo "▶ [7/8] Exp7: Debate + Eviction"
python3 experiments/exp7_debate_eviction/run.py $COMMON_ARGS $EVICT_ARGS --n_rounds 3 2>&1 | tee "$OUTDIR/exp7_debate_eviction.log"

echo ""
echo "▶ [8/8] Exp8: MCTS + Eviction"
python3 experiments/exp8_mcts_eviction/run.py $COMMON_ARGS $EVICT_ARGS --n_iterations 20 --n_rollouts 3 --max_depth 3 --rollout_length 64 2>&1 | tee "$OUTDIR/exp8_mcts_eviction.log"

echo ""
echo "═══════════════════════════════════════════════════"
echo " ✅ All 8 experiments completed! Results → $OUTDIR"
echo "═══════════════════════════════════════════════════"

