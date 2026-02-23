#!/usr/bin/env python3
"""
실험 3: Multi-Agent Debate
단일 모델에 역할별 프롬프트(생성/비판/검증)를 번갈아 호출.
NVTX: RUN / Debate/AGENT_A / Debate/AGENT_B / Debate/AGENT_C / Debate/JUDGE + PREFILL / DECODE_STEP
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from typing import List, Tuple
from common.args import get_common_parser, setup_seed
from common.model_loader import load_model_and_tokenizer
from common.nvtx_utils import nvtx_run, nvtx_debate_agent, nvtx_debate_judge
from common.profiling import TegrastatsLogger
from common.metrics import MetricsCollector, LatencyTracker
from common.generation import generate_with_profiling
from common.kv_cache_eviction import HeavyHitterEvictionManager

DEFAULT_PROBLEM = (
    "Should we prioritize economic growth over environmental protection? "
    "Provide a well-reasoned argument."
)

# ─── Agent Role Templates ───
AGENT_TEMPLATES = {
    "GENERATOR": (
        "You are a creative problem solver. "
        "Problem: {problem}\n"
        "Previous discussion:\n{history}\n\n"
        "Generate a new argument or solution:\n"
    ),
    "CRITIC": (
        "You are a critical thinker who finds flaws. "
        "Problem: {problem}\n"
        "Current proposal:\n{proposal}\n\n"
        "Critique this proposal. Identify weaknesses:\n"
    ),
    "VERIFIER": (
        "You are a fact-checker and verifier. "
        "Problem: {problem}\n"
        "Proposal: {proposal}\n"
        "Critique: {critique}\n\n"
        "Verify the claims. What is correct and what is wrong?\n"
    ),
    "JUDGE": (
        "You are a fair judge. "
        "Problem: {problem}\n"
        "All arguments:\n{all_arguments}\n\n"
        "Provide a final verdict with reasoning:\n"
    ),
}


def agent_generate(model, tokenizer, prompt: str, device: str, gen_length: int,
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


def run_debate(model, tokenizer, problem: str, device: str, gen_length: int,
               n_rounds: int = 3, eviction_mgr=None):
    """
    Multi-Agent Debate 실행.
    n_rounds번의 (Generator → Critic → Verifier) 사이클 + 최종 Judge.
    """
    all_trackers: List[LatencyTracker] = []
    all_eviction_stats = []
    history = ""
    proposal = ""
    critique = ""

    for rnd in range(n_rounds):
        print(f"  [Debate] Round {rnd+1}/{n_rounds}")

        # Agent A: Generator
        with nvtx_debate_agent("AGENT_A_GENERATOR"):
            prompt_a = AGENT_TEMPLATES["GENERATOR"].format(
                problem=problem, history=history[-1000:])
            proposal, trk, ev = agent_generate(
                model, tokenizer, prompt_a, device, gen_length, eviction_mgr)
            all_trackers.append(trk)
            all_eviction_stats.extend(ev)
            history += f"\n[Generator R{rnd+1}]: {proposal[-300:]}"
            if eviction_mgr:
                eviction_mgr.reset()

        # Agent B: Critic
        with nvtx_debate_agent("AGENT_B_CRITIC"):
            prompt_b = AGENT_TEMPLATES["CRITIC"].format(
                problem=problem, proposal=proposal[-500:])
            critique, trk, ev = agent_generate(
                model, tokenizer, prompt_b, device, gen_length, eviction_mgr)
            all_trackers.append(trk)
            all_eviction_stats.extend(ev)
            history += f"\n[Critic R{rnd+1}]: {critique[-300:]}"
            if eviction_mgr:
                eviction_mgr.reset()

        # Agent C: Verifier
        with nvtx_debate_agent("AGENT_C_VERIFIER"):
            prompt_c = AGENT_TEMPLATES["VERIFIER"].format(
                problem=problem, proposal=proposal[-300:], critique=critique[-300:])
            verification, trk, ev = agent_generate(
                model, tokenizer, prompt_c, device, gen_length, eviction_mgr)
            all_trackers.append(trk)
            all_eviction_stats.extend(ev)
            history += f"\n[Verifier R{rnd+1}]: {verification[-300:]}"
            if eviction_mgr:
                eviction_mgr.reset()

    # Judge
    with nvtx_debate_judge():
        prompt_j = AGENT_TEMPLATES["JUDGE"].format(
            problem=problem, all_arguments=history[-2000:])
        verdict, trk, ev = agent_generate(
            model, tokenizer, prompt_j, device, gen_length, eviction_mgr)
        all_trackers.append(trk)
        all_eviction_stats.extend(ev)

    return verdict, all_trackers, all_eviction_stats


def main():
    parser = get_common_parser("Exp3: Multi-Agent Debate")
    parser.add_argument("--n_rounds", type=int, default=3, help="디베이트 라운드 수")
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

    collector = MetricsCollector("exp3_debate", args.output_dir)
    collector.extra = {
        "n_rounds": args.n_rounds,
        "max_context": args.max_context, "gen_length": args.gen_length,
        "eviction_enabled": args.eviction_enabled,
    }

    teg_path = os.path.join(args.output_dir, f"exp3_debate_tegrastats{('_' + args.tag) if args.tag else ''}.jsonl")
    teg = TegrastatsLogger(interval_ms=args.tegrastats_interval, output_path=teg_path) if args.tegrastats else None

    # Warmup
    print(f"[Exp3] Warmup...")
    inputs = tokenizer("Hello", return_tensors="pt").to(args.device)
    for _ in range(args.warmup):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=8)
    torch.cuda.synchronize()

    if teg:
        teg.start()

    for rep in range(args.repeats):
        print(f"[Exp3] Repeat {rep+1}/{args.repeats}")
        torch.cuda.reset_peak_memory_stats()
        if eviction_mgr:
            eviction_mgr.reset()

        with nvtx_run():
            verdict, trackers, ev_stats = run_debate(
                model, tokenizer, problem, args.device, args.gen_length,
                n_rounds=args.n_rounds, eviction_mgr=eviction_mgr,
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
    print(f"\n[Exp3] === 집계 결과 ===")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

