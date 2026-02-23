#!/usr/bin/env python3
"""
실험 7: Multi-Agent Debate + H2O KV Cache Eviction
실험 3과 동일 로직 + eviction 강제 활성화.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from common.args import get_common_parser, setup_seed
from common.model_loader import load_model_and_tokenizer
from common.nvtx_utils import nvtx_run
from common.profiling import TegrastatsLogger
from common.metrics import MetricsCollector, LatencyTracker
from common.kv_cache_eviction import HeavyHitterEvictionManager

from exp3_debate.run import run_debate, DEFAULT_PROBLEM


def main():
    parser = get_common_parser("Exp7: Debate + KV Cache Eviction")
    parser.add_argument("--n_rounds", type=int, default=3)
    args = parser.parse_args()
    args.eviction_enabled = True
    setup_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device)
    problem = args.prompt or DEFAULT_PROBLEM

    eviction_mgr = HeavyHitterEvictionManager(
        top_k=args.eviction_top_k,
        window=args.eviction_window,
        eviction_freq=args.eviction_freq,
        device=args.device,
    )

    collector = MetricsCollector("exp7_debate_eviction", args.output_dir)
    collector.extra = {
        "n_rounds": args.n_rounds,
        "max_context": args.max_context, "gen_length": args.gen_length,
        "eviction_top_k": args.eviction_top_k,
        "eviction_window": args.eviction_window,
        "eviction_freq": args.eviction_freq,
    }

    teg_path = os.path.join(args.output_dir, f"exp7_debate_eviction_tegrastats{('_' + args.tag) if args.tag else ''}.jsonl")
    teg = TegrastatsLogger(interval_ms=args.tegrastats_interval, output_path=teg_path) if args.tegrastats else None

    print(f"[Exp7] Warmup...")
    inputs = tokenizer("Hello", return_tensors="pt").to(args.device)
    for _ in range(args.warmup):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=8)
    torch.cuda.synchronize()

    if teg:
        teg.start()

    for rep in range(args.repeats):
        print(f"[Exp7] Repeat {rep+1}/{args.repeats}")
        torch.cuda.reset_peak_memory_stats()
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
        collector.add_eviction_overhead(eviction_mgr.get_overhead_summary())

        summary = merged.summary()
        print(f"  total_tokens={summary.get('total_decode_tokens',0)}  "
              f"tok/s={summary.get('tokens_per_sec',0):.1f}")

    if teg:
        teg.stop()

    agg = collector.save(tag=args.tag)
    print(f"\n[Exp7] === 집계 결과 ===")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

