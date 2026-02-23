"""측정 지표 수집/집계 유틸리티."""
import time
import json
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class LatencyRecord:
    """단일 디코드 스텝(또는 prefill) 레이턴시 레코드."""
    step: int
    latency_ms: float
    phase: str = "decode"          # "prefill" | "decode"
    kv_bytes: int = 0              # 현재 KV cache 크기 (bytes)
    keep_ratio: float = 1.0        # eviction 후 유지 비율
    evicted_count: int = 0         # 이 스텝에서 evict된 토큰 수
    mem_allocated_gb: float = 0.0


class LatencyTracker:
    """토큰별 레이턴시를 추적하고 p50/p95/p99 통계 계산."""

    def __init__(self):
        self.records: List[LatencyRecord] = []

    def add(self, record: LatencyRecord):
        self.records.append(record)

    def get_decode_latencies(self) -> np.ndarray:
        return np.array([r.latency_ms for r in self.records if r.phase == "decode"])

    def get_prefill_latency(self) -> float:
        prefill = [r.latency_ms for r in self.records if r.phase == "prefill"]
        return prefill[0] if prefill else 0.0

    def summary(self) -> Dict[str, Any]:
        dec = self.get_decode_latencies()
        if len(dec) == 0:
            return {}
        return {
            "prefill_latency_ms": self.get_prefill_latency(),
            "decode_p50_ms": float(np.percentile(dec, 50)),
            "decode_p95_ms": float(np.percentile(dec, 95)),
            "decode_p99_ms": float(np.percentile(dec, 99)),
            "decode_mean_ms": float(np.mean(dec)),
            "decode_std_ms": float(np.std(dec)),
            "tokens_per_sec": 1000.0 / float(np.mean(dec)) if np.mean(dec) > 0 else 0,
            "total_decode_tokens": len(dec),
            "total_decode_time_ms": float(np.sum(dec)),
        }


class MetricsCollector:
    """
    실험 전체 메트릭을 수집/저장.
    - 여러 반복(repeat)의 LatencyTracker를 모아서 통계.
    - eviction 오버헤드, kv_bytes 추적.
    """

    def __init__(self, experiment_name: str, output_dir: str = "results"):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.repeat_summaries: List[Dict[str, Any]] = []
        self.eviction_overheads: List[Dict[str, float]] = []
        self.kv_bytes_timeline: List[List[int]] = []
        self.extra: Dict[str, Any] = {}

    def add_repeat(self, tracker: LatencyTracker):
        self.repeat_summaries.append(tracker.summary())

    def add_eviction_overhead(self, overhead: Dict[str, float]):
        """
        overhead keys: collect_ms, score_update_ms, decision_ms, move_ms
        """
        self.eviction_overheads.append(overhead)

    def add_kv_timeline(self, timeline: List[int]):
        self.kv_bytes_timeline.append(timeline)

    def aggregate(self) -> Dict[str, Any]:
        """모든 repeat의 통계를 집계."""
        if not self.repeat_summaries:
            return {}
        keys = self.repeat_summaries[0].keys()
        agg = {}
        for k in keys:
            vals = [s[k] for s in self.repeat_summaries if k in s]
            if vals:
                agg[f"{k}_mean"] = float(np.mean(vals))
                agg[f"{k}_std"] = float(np.std(vals))
        # eviction 오버헤드 평균
        if self.eviction_overheads:
            ov_keys = self.eviction_overheads[0].keys()
            for ok in ov_keys:
                ov_vals = [o[ok] for o in self.eviction_overheads]
                agg[f"eviction_{ok}_mean"] = float(np.mean(ov_vals))
                agg[f"eviction_{ok}_std"] = float(np.std(ov_vals))
        # peak kv bytes
        if self.kv_bytes_timeline:
            all_peaks = [max(tl) if tl else 0 for tl in self.kv_bytes_timeline]
            agg["peak_kv_bytes_mean"] = float(np.mean(all_peaks))
        agg["experiment"] = self.experiment_name
        agg.update(self.extra)
        return agg

    def save(self, tag: str = ""):
        """JSON으로 결과 저장."""
        agg = self.aggregate()
        fname = f"{self.experiment_name}"
        if tag:
            fname += f"_{tag}"
        fname += ".json"
        out_path = self.output_dir / fname
        with open(out_path, "w") as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)
        print(f"[MetricsCollector] Saved → {out_path}")

        # 개별 repeat 저장
        detail_path = self.output_dir / fname.replace(".json", "_detail.json")
        with open(detail_path, "w") as f:
            json.dump({
                "repeat_summaries": self.repeat_summaries,
                "eviction_overheads": self.eviction_overheads,
            }, f, indent=2, ensure_ascii=False)
        print(f"[MetricsCollector] Detail saved → {detail_path}")
        return agg

