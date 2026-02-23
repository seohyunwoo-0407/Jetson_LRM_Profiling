#!/usr/bin/env python3
"""
실험 2: Tree of Thoughts (ToT) — BFS/DFS 탐색.
NVTX: RUN / ToT/PROPOSE / ToT/EVAL / ToT/SELECT / ToT/EXPAND + PREFILL / DECODE_STEP
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from typing import List, Tuple
from common.args import get_common_parser, setup_seed
from common.model_loader import load_model_and_tokenizer
from common.nvtx_utils import (nvtx_run, nvtx_tot_propose, nvtx_tot_eval,
                                nvtx_tot_select, nvtx_tot_expand)
from common.profiling import TegrastatsLogger, collect_cuda_metrics
from common.metrics import MetricsCollector, LatencyTracker
from common.generation import generate_with_profiling
from common.kv_cache_eviction import HeavyHitterEvictionManager

DEFAULT_PROBLEM = (
    "You are an expert mathematician and logician. Solve the following complex problem step by step, "
    "showing all intermediate calculations and logical reasoning:

"
    "A company is planning to expand its operations across three cities: City A, City B, and City C. "
    "The initial investment required is $2.5 million. City A requires 40% of the total investment, "
    "City B requires 35% of the remaining amount after City A's investment, and City C requires the rest.

"
    "Additionally, the company must account for operational costs: City A has monthly costs of $15,000, "
    "City B has monthly costs that are 20% higher than City A, and City C has monthly costs that are "
    "the average of City A and City B combined.

"
    "Questions to answer:
"
    "1. Calculate the exact investment amount for each city.
"
    "2. Calculate the monthly operational costs for each city.
"
    "3. If the company expects to break even after 18 months of operation, what should be the minimum "
    "monthly revenue per city?
"
    "4. Considering that City B has a 15% higher revenue potential than City A, and City C has a "
    "revenue potential that is 80% of the average of City A and City B, determine the optimal "
    "revenue distribution strategy.

"
    "Show all your work, explain each step clearly, and verify your calculations. Consider edge cases "
    "and potential risks in your analysis."
)

# ─── ToT Prompt Templates ───
PROPOSE_TEMPLATE = (
    "Problem: {problem}\n\n"
    "Propose {n} distinct solution approaches. "
    "For each, give a 1-sentence summary.\n\n"
    "Approaches:"
)

EVAL_TEMPLATE = (
    "Problem: {problem}\n\n"
    "Proposed approach: {approach}\n\n"
    "Rate this approach on a scale of 1-10 and explain briefly.\n"
    "Score:"
)

EXPAND_TEMPLATE = (
    "Problem: {problem}\n\n"
    "Chosen approach: {approach}\n\n"
    "Now expand this into {n} detailed sub-plans.\n\n"
    "Sub-plans:"
)


def tot_generate(model, tokenizer, input_text: str, device: str, gen_length: int,
                 eviction_mgr=None) -> Tuple[str, LatencyTracker, list]:
    """단일 생성 호출 (프로파일링 포함)."""
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        all_ids, tracker, eviction_stats = generate_with_profiling(
            model, input_ids, attention_mask,
            gen_length=gen_length,
            eviction_mgr=eviction_mgr,
        )
    text = tokenizer.decode(all_ids[0], skip_special_tokens=True)
    return text, tracker, eviction_stats


def run_tot(model, tokenizer, problem: str, device: str, gen_length: int,
            n_proposals: int = 3, depth: int = 2, mode: str = "bfs",
            eviction_mgr=None):
    """
    ToT BFS/DFS 실행.
    Returns: (best_result, all_trackers, all_eviction_stats)
    """
    all_trackers: List[LatencyTracker] = []
    all_eviction_stats = []

    # ── PHASE 1: Propose ──
    with nvtx_tot_propose():
        propose_prompt = PROPOSE_TEMPLATE.format(problem=problem, n=n_proposals)
        proposals_text, trk, ev = tot_generate(
            model, tokenizer, propose_prompt, device, gen_length, eviction_mgr)
        all_trackers.append(trk)
        all_eviction_stats.extend(ev)

    # 간단한 파싱: 줄 단위로 proposal 추출
    proposals = [line.strip() for line in proposals_text.split("\n")
                 if line.strip() and len(line.strip()) > 10][:n_proposals]
    if not proposals:
        proposals = [f"Approach {i+1}: direct calculation" for i in range(n_proposals)]
    print(f"  [ToT] Proposals: {len(proposals)}")

    # ── PHASE 2: Eval each proposal ──
    scores = []
    for i, prop in enumerate(proposals):
        with nvtx_tot_eval():
            eval_prompt = EVAL_TEMPLATE.format(problem=problem, approach=prop)
            eval_text, trk, ev = tot_generate(
                model, tokenizer, eval_prompt, device, min(64, gen_length), eviction_mgr)
            all_trackers.append(trk)
            all_eviction_stats.extend(ev)
            # 점수 파싱 (숫자 추출)
            score = _extract_score(eval_text)
            scores.append(score)
            print(f"  [ToT] Proposal {i+1} score: {score}")

    # ── PHASE 3: Select best ──
    with nvtx_tot_select():
        best_idx = scores.index(max(scores))
        best_proposal = proposals[best_idx]
        print(f"  [ToT] Selected: {best_idx+1} ({best_proposal[:60]}...)")

    # ── PHASE 4: Expand (depth iterations) ──
    current_approach = best_proposal
    for d in range(depth):
        with nvtx_tot_expand():
            expand_prompt = EXPAND_TEMPLATE.format(
                problem=problem, approach=current_approach, n=n_proposals)
            expand_text, trk, ev = tot_generate(
                model, tokenizer, expand_prompt, device, gen_length, eviction_mgr)
            all_trackers.append(trk)
            all_eviction_stats.extend(ev)
            current_approach = expand_text[-500:]  # 마지막 부분을 다음 입력으로
            print(f"  [ToT] Depth {d+1}/{depth} expanded")

    return current_approach, all_trackers, all_eviction_stats


def _extract_score(text: str) -> float:
    """텍스트에서 숫자 점수 추출."""
    import re
    numbers = re.findall(r'(\d+(?:\.\d+)?)', text)
    for n in numbers:
        val = float(n)
        if 1 <= val <= 10:
            return val
    return 5.0  # 기본값


def main():
    parser = get_common_parser("Exp2: Tree of Thoughts")
    parser.add_argument("--n_proposals", type=int, default=3, help="제안 수")
    parser.add_argument("--depth", type=int, default=2, help="탐색 깊이")
    parser.add_argument("--mode", type=str, default="bfs", choices=["bfs", "dfs"])
    args = parser.parse_args()
    setup_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device)
    problem = args.prompt or DEFAULT_PROBLEM

    eviction_mgr = None
    if args.eviction_enabled:
        eviction_mgr = HeavyHitterEvictionManager(
            top_k=args.eviction_top_k,
            window=args.eviction_window,
            eviction_freq=args.eviction_freq,
            device=args.device,
        )

    collector = MetricsCollector("exp2_tot", args.output_dir)
    collector.extra = {
        "n_proposals": args.n_proposals, "depth": args.depth, "mode": args.mode,
        "max_context": args.max_context, "gen_length": args.gen_length,
        "eviction_enabled": args.eviction_enabled,
    }

    teg_path = os.path.join(args.output_dir, f"exp2_tot_tegrastats{('_' + args.tag) if args.tag else ''}.jsonl")
    teg = TegrastatsLogger(interval_ms=args.tegrastats_interval, output_path=teg_path) if args.tegrastats else None

    # Warmup
    print(f"[Exp2] Warmup...")
    inputs = tokenizer("Hello", return_tensors="pt").to(args.device)
    for _ in range(args.warmup):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=8)
    torch.cuda.synchronize()

    # 측정
    if teg:
        teg.start()

    for rep in range(args.repeats):
        print(f"[Exp2] Repeat {rep+1}/{args.repeats}")
        torch.cuda.reset_peak_memory_stats()
        if eviction_mgr:
            eviction_mgr.reset()

        with nvtx_run():
            result, trackers, ev_stats = run_tot(
                model, tokenizer, problem, args.device, args.gen_length,
                n_proposals=args.n_proposals, depth=args.depth, mode=args.mode,
                eviction_mgr=eviction_mgr,
            )

        # 모든 tracker를 합산
        merged = LatencyTracker()
        for trk in trackers:
            merged.records.extend(trk.records)
        collector.add_repeat(merged)

        summary = merged.summary()
        print(f"  total_tokens={summary.get('total_decode_tokens',0)}  "
              f"tok/s={summary.get('tokens_per_sec',0):.1f}")

    if teg:
        teg.stop()

    if eviction_mgr:
        collector.add_eviction_overhead(eviction_mgr.get_overhead_summary())

    agg = collector.save(tag=args.tag)
    print(f"\n[Exp2] === 집계 결과 ===")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

