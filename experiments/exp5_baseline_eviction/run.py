#!/usr/bin/env python3
"""
실험 5: Baseline + H2O KV Cache Eviction
실험 1과 동일하되, 매 디코드 스텝마다 attention 기반 eviction 수행.
NVTX: RUN / PREFILL / DECODE_STEP + EVICT/ATTN_METRIC_COLLECT / EVICT/SCORE_UPDATE / EVICT/EVICT_DECISION / EVICT/EVICT_MOVE
"""
import sys, os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "experiments"))

import torch
from src.args import get_common_parser, setup_seed
from src.model_loader import load_model_and_tokenizer
from src.nvtx_utils import nvtx_run
from src.profiling import TegrastatsLogger, collect_cuda_metrics
from src.metrics import MetricsCollector
from src.generation import generate_with_profiling
from src.kv_cache_eviction import HeavyHitterEvictionManager

DEFAULT_PROMPT = (
    "You are an expert mathematician and logician. Solve the following complex problem step by step, "
    "showing all intermediate calculations and logical reasoning.\n\n"
    "A company is planning to expand its operations across three cities: City A, City B, and City C. "
    "The initial investment required is $2.5 million. City A requires 40% of the total investment, "
    "City B requires 35% of the remaining amount after City A's investment, and City C requires the rest.\n\n"
    "Additionally, the company must account for operational costs: City A has monthly costs of $15,000, "
    "City B has monthly costs that are 20% higher than City A, and City C has monthly costs that are "
    "the average of City A and City B combined.\n\n"
    "Questions to answer:\n"
    "1. Calculate the exact investment amount for each city.\n"
    "2. Calculate the monthly operational costs for each city.\n"
    "3. If the company expects to break even after 18 months of operation, what should be the minimum "
    "monthly revenue per city?\n"
    "4. Considering that City B has a 15% higher revenue potential than City A, and City C has a "
    "revenue potential that is 80% of the average of City A and City B, determine the optimal "
    "revenue distribution strategy.\n\n"
    "Show all your work, explain each step clearly, and verify your calculations. Consider edge cases "
    "and potential risks in your analysis."
)


def main():
    parser = get_common_parser("Exp5: Baseline + KV Cache Eviction")
    args = parser.parse_args()
    # eviction 강제 활성화
    args.eviction_enabled = True
    setup_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device)
    prompt = args.prompt or DEFAULT_PROMPT
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_context)
    input_ids = inputs["input_ids"].to(args.device)
    attention_mask = inputs["attention_mask"].to(args.device)
    print(f"[Exp5] Prompt tokens: {input_ids.shape[1]}, Gen length: {args.gen_length}")
    print(f"[Exp5] Eviction: top_k={args.eviction_top_k}, window={args.eviction_window}, freq={args.eviction_freq}")

    eviction_mgr = HeavyHitterEvictionManager(
        top_k=args.eviction_top_k,
        window=args.eviction_window,
        eviction_freq=args.eviction_freq,
        device=args.device,
    )

    collector = MetricsCollector("exp5_baseline_eviction", args.output_dir)
    collector.extra = {
        "max_context": args.max_context, "gen_length": args.gen_length,
        "prompt_tokens": input_ids.shape[1],
        "eviction_top_k": args.eviction_top_k,
        "eviction_window": args.eviction_window,
        "eviction_freq": args.eviction_freq,
    }

    teg_path = os.path.join(args.output_dir, f"exp5_baseline_eviction_tegrastats{('_' + args.tag) if args.tag else ''}.jsonl")
    teg = TegrastatsLogger(interval_ms=args.tegrastats_interval, output_path=teg_path) if args.tegrastats else None

    # Warmup
    print(f"[Exp5] Warmup {args.warmup} runs...")
    for _ in range(args.warmup):
        eviction_mgr.reset()
        with torch.no_grad():
            generate_with_profiling(model, input_ids, attention_mask,
                                    gen_length=min(16, args.gen_length),
                                    eviction_mgr=eviction_mgr)
        torch.cuda.synchronize()

    if teg:
        teg.start()

    for rep in range(args.repeats):
        print(f"[Exp5] Repeat {rep+1}/{args.repeats}")
        torch.cuda.reset_peak_memory_stats()
        eviction_mgr.reset()

        with nvtx_run():
            with torch.no_grad():
                all_ids, tracker, ev_stats = generate_with_profiling(
                    model, input_ids, attention_mask,
                    gen_length=args.gen_length,
                    eviction_mgr=eviction_mgr,
                )

        summary = tracker.summary()
        print(f"  prefill={summary.get('prefill_latency_ms',0):.1f}ms  "
              f"decode_p50={summary.get('decode_p50_ms',0):.2f}ms  "
              f"tok/s={summary.get('tokens_per_sec',0):.1f}")

        collector.add_repeat(tracker)
        collector.add_kv_timeline([r.kv_bytes for r in tracker.records])
        collector.add_eviction_overhead(eviction_mgr.get_overhead_summary())

    if teg:
        teg.stop()

    agg = collector.save(tag=args.tag)
    print(f"\n[Exp5] === 집계 결과 ===")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    decoded_text = tokenizer.decode(all_ids[0], skip_special_tokens=True)
    print(f"\n[Exp5] Generated ({len(all_ids[0]) - input_ids.shape[1]} tokens):")
    print(decoded_text[:500])


if __name__ == "__main__":
    main()

