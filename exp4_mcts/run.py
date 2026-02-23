#!/usr/bin/env python3
"""
실험 4: Monte Carlo Tree Search (MCTS)
Rollout 시뮬레이션으로 Reward 계산, 반복하여 최적 논리 완성.
NVTX: RUN / MCTS/SELECT / MCTS/EXPAND / MCTS/ROLLOUT_N / MCTS/BACKPROP + PREFILL / DECODE_STEP
"""
import sys, os, math, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from common.args import get_common_parser, setup_seed
from common.model_loader import load_model_and_tokenizer
from common.nvtx_utils import (nvtx_run, nvtx_mcts_select, nvtx_mcts_expand,
                                nvtx_mcts_rollout, nvtx_mcts_backprop)
from common.profiling import TegrastatsLogger
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

# ─── MCTS Node ───
@dataclass
class MCTSNode:
    text: str
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    depth: int = 0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    def ucb1(self, exploration: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        exploit = self.avg_reward
        explore = exploration * math.sqrt(math.log(self.parent.visits) / self.visits) if self.parent else 0
        return exploit + explore


def llm_generate(model, tokenizer, prompt: str, device: str, gen_length: int,
                 eviction_mgr=None) -> Tuple[str, LatencyTracker, list]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        all_ids, tracker, ev_stats = generate_with_profiling(
            model, input_ids, attention_mask,
            gen_length=gen_length,
            eviction_mgr=eviction_mgr,
        )
    text = tokenizer.decode(all_ids[0], skip_special_tokens=True)
    return text, tracker, ev_stats


def reward_function(text: str) -> float:
    """
    간단한 reward 함수:
    - 길이, 구조(step/번호), 논리 키워드 존재로 점수 부여.
    """
    score = 0.0
    score += min(len(text) / 500.0, 1.0) * 3.0  # 길이 (max 3)
    for kw in ["step", "first", "then", "next", "finally", "because", "therefore"]:
        if kw in text.lower():
            score += 0.5
    import re
    numbered = re.findall(r'\d+[\.\)]', text)
    score += min(len(numbered) * 0.3, 2.0)
    return min(score, 10.0)


def run_mcts(model, tokenizer, problem: str, device: str, gen_length: int,
             n_iterations: int = 20, n_rollouts: int = 3, max_depth: int = 3,
             n_children: int = 3, rollout_length: int = 64,
             eviction_mgr=None):
    """
    MCTS 실행.
    """
    all_trackers: List[LatencyTracker] = []
    all_eviction_stats = []

    root = MCTSNode(text=problem, depth=0)

    for it in range(n_iterations):
        print(f"  [MCTS] Iteration {it+1}/{n_iterations}")

        # ── SELECT ──
        with nvtx_mcts_select():
            node = root
            while node.children and node.depth < max_depth:
                node = max(node.children, key=lambda c: c.ucb1())

        # ── EXPAND ──
        if node.depth < max_depth and len(node.children) < n_children:
            with nvtx_mcts_expand():
                expand_prompt = (
                    f"Problem: {problem}\n"
                    f"Current reasoning: {node.text[-300:]}\n\n"
                    f"Continue with a new reasoning step:\n"
                )
                new_text, trk, ev = llm_generate(
                    model, tokenizer, expand_prompt, device, gen_length, eviction_mgr)
                all_trackers.append(trk)
                all_eviction_stats.extend(ev)
                if eviction_mgr:
                    eviction_mgr.reset()

                child = MCTSNode(
                    text=new_text[-500:],
                    parent=node,
                    depth=node.depth + 1,
                )
                node.children.append(child)
                node = child

        # ── ROLLOUT ──
        total_reward = 0.0
        for r_idx in range(n_rollouts):
            with nvtx_mcts_rollout(r_idx):
                rollout_prompt = (
                    f"Problem: {problem}\n"
                    f"Reasoning so far: {node.text[-200:]}\n\n"
                    f"Complete the solution:\n"
                )
                rollout_text, trk, ev = llm_generate(
                    model, tokenizer, rollout_prompt, device, rollout_length, eviction_mgr)
                all_trackers.append(trk)
                all_eviction_stats.extend(ev)
                if eviction_mgr:
                    eviction_mgr.reset()

                reward = reward_function(rollout_text)
                total_reward += reward

        avg_reward = total_reward / n_rollouts

        # ── BACKPROP ──
        with nvtx_mcts_backprop():
            backprop_node = node
            while backprop_node is not None:
                backprop_node.visits += 1
                backprop_node.total_reward += avg_reward
                backprop_node = backprop_node.parent

    # 최적 결과 추출
    best_node = root
    while best_node.children:
        best_node = max(best_node.children, key=lambda c: c.avg_reward)

    return best_node.text, all_trackers, all_eviction_stats


def main():
    parser = get_common_parser("Exp4: MCTS")
    parser.add_argument("--n_iterations", type=int, default=20, help="MCTS 반복 수")
    parser.add_argument("--n_rollouts", type=int, default=3, help="롤아웃 수")
    parser.add_argument("--max_depth", type=int, default=3, help="트리 최대 깊이")
    parser.add_argument("--n_children", type=int, default=3, help="자식 노드 수")
    parser.add_argument("--rollout_length", type=int, default=64, help="롤아웃 생성 토큰 수")
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

    collector = MetricsCollector("exp4_mcts", args.output_dir)
    collector.extra = {
        "n_iterations": args.n_iterations, "n_rollouts": args.n_rollouts,
        "max_depth": args.max_depth, "n_children": args.n_children,
        "rollout_length": args.rollout_length,
        "max_context": args.max_context, "gen_length": args.gen_length,
        "eviction_enabled": args.eviction_enabled,
    }

    teg_path = os.path.join(args.output_dir, f"exp4_mcts_tegrastats{('_' + args.tag) if args.tag else ''}.jsonl")
    teg = TegrastatsLogger(interval_ms=args.tegrastats_interval, output_path=teg_path) if args.tegrastats else None

    # Warmup
    print(f"[Exp4] Warmup...")
    inputs = tokenizer("Hello", return_tensors="pt").to(args.device)
    for _ in range(args.warmup):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=8)
    torch.cuda.synchronize()

    if teg:
        teg.start()

    for rep in range(args.repeats):
        print(f"[Exp4] Repeat {rep+1}/{args.repeats}")
        torch.cuda.reset_peak_memory_stats()
        if eviction_mgr:
            eviction_mgr.reset()

        with nvtx_run():
            result, trackers, ev_stats = run_mcts(
                model, tokenizer, problem, args.device, args.gen_length,
                n_iterations=args.n_iterations, n_rollouts=args.n_rollouts,
                max_depth=args.max_depth, n_children=args.n_children,
                rollout_length=args.rollout_length,
                eviction_mgr=eviction_mgr,
            )

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
    print(f"\n[Exp4] === 집계 결과 ===")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

