#!/usr/bin/env python3
"""
실험 6: ToT + H2O KV Cache Eviction
실험 2와 동일 로직 + eviction 강제 활성화.
"""
import sys, os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "experiments"))

import torch
from src.args import get_common_parser, setup_seed
from src.model_loader import load_model_and_tokenizer
from src.nvtx_utils import nvtx_run
from src.profiling import TegrastatsLogger
from src.metrics import MetricsCollector, LatencyTracker
from src.kv_cache_eviction import HeavyHitterEvictionManager

# exp2의 로직 재사용
from exp2_tot.run import run_tot, DEFAULT_PROBLEM


def main():
    parser = get_common_parser("Exp6: ToT + KV Cache Eviction")
    parser.add_argument("--n_proposals", type=int, default=3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--mode", type=str, default="bfs", choices=["bfs", "dfs"])
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

    collector = MetricsCollector("exp6_tot_eviction", args.output_dir)
    collector.extra = {
        "n_proposals": args.n_proposals, "depth": args.depth, "mode": args.mode,
        "max_context": args.max_context, "gen_length": args.gen_length,
        "eviction_top_k": args.eviction_top_k,
        "eviction_window": args.eviction_window,
        "eviction_freq": args.eviction_freq,
    }

    teg_path = os.path.join(args.output_dir, f"exp6_tot_eviction_tegrastats{('_' + args.tag) if args.tag else ''}.jsonl")
    teg = TegrastatsLogger(interval_ms=args.tegrastats_interval, output_path=teg_path) if args.tegrastats else None

    # Warmup
    print(f"[Exp6] Warmup...")
    inputs = tokenizer("Hello", return_tensors="pt").to(args.device)
    for _ in range(args.warmup):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=8)
    torch.cuda.synchronize()

    if teg:
        teg.start()

    for rep in range(args.repeats):
        print(f"[Exp6] Repeat {rep+1}/{args.repeats}")
        torch.cuda.reset_peak_memory_stats()
        eviction_mgr.reset()

        with nvtx_run():
            result, trackers, ev_stats = run_tot(
                model, tokenizer, problem, args.device, args.gen_length,
                n_proposals=args.n_proposals, depth=args.depth, mode=args.mode,
                eviction_mgr=eviction_mgr,
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
    print(f"\n[Exp6] === 집계 결과 ===")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

