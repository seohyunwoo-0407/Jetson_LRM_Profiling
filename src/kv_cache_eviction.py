"""
Heavy-Hitter Oracle (H2O) 스타일 KV Cache Eviction 엔진.

매 토큰 생성 시:
  1. attention metric 수집 (각 past token이 받은 attention weight 누적)
  2. 누적 점수 업데이트
  3. eviction 판정: 상위 top_k + 최근 window → keep, 나머지 → evict
  4. KV cache 텐서에서 evict 대상 제거 (compaction)
"""
import torch
import time
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field

from .nvtx_utils import (
    nvtx_evict_attn_collect,
    nvtx_evict_score_update,
    nvtx_evict_decision,
    nvtx_evict_move,
)


@dataclass
class EvictionStats:
    """eviction 한 스텝의 오버헤드 기록."""
    collect_ms: float = 0.0
    score_update_ms: float = 0.0
    decision_ms: float = 0.0
    move_ms: float = 0.0
    evicted_count: int = 0
    kept_count: int = 0
    kv_bytes_before: int = 0
    kv_bytes_after: int = 0

    def total_ms(self) -> float:
        return self.collect_ms + self.score_update_ms + self.decision_ms + self.move_ms

    def to_dict(self) -> Dict[str, float]:
        return {
            "collect_ms": self.collect_ms,
            "score_update_ms": self.score_update_ms,
            "decision_ms": self.decision_ms,
            "move_ms": self.move_ms,
            "evicted_count": self.evicted_count,
            "kept_count": self.kept_count,
            "kv_bytes_before": self.kv_bytes_before,
            "kv_bytes_after": self.kv_bytes_after,
        }


class HeavyHitterEvictionManager:
    """
    H2O-inspired KV Cache Eviction.

    Parameters:
        top_k: 누적 attention 점수 상위 K개 keep
        window: 최근 window 토큰 항상 keep
        eviction_freq: N 스텝마다 eviction 수행 (1=매 스텝)
        num_layers: 모델 레이어 수
        num_heads: 어텐션 헤드 수
    """

    def __init__(
        self,
        top_k: int = 256,
        window: int = 64,
        eviction_freq: int = 1,
        num_layers: int = 0,
        num_heads: int = 0,
        device: str = "cuda",
    ):
        self.top_k = top_k
        self.window = window
        self.eviction_freq = eviction_freq
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.device = device

        # 누적 attention 점수 (layer, head, seq)
        # generate 시작 시 초기화
        self.cumulative_scores: Optional[torch.Tensor] = None
        self.step_count: int = 0
        self.stats_history: List[EvictionStats] = []

    def reset(self):
        """새 생성 시작 시 초기화."""
        self.cumulative_scores = None
        self.step_count = 0
        self.stats_history = []

    # ─────────────────────────────────────────────
    # 1) Attention Metric 수집
    # ─────────────────────────────────────────────
    def collect_attention_metrics(
        self, attentions: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """
        attentions: transformers output.attentions
           tuple of (batch, num_heads, q_len, kv_len) per layer
        Returns: (num_layers, num_heads, kv_len) — 이번 스텝 attention 합산
        """
        with nvtx_evict_attn_collect():
            start = time.perf_counter()
            # 마지막 query position이 각 kv에 준 attention weight 합산
            layer_scores = []
            for layer_attn in attentions:
                # layer_attn: (batch, heads, q_len, kv_len) — 마지막 q만
                score = layer_attn[:, :, -1, :].squeeze(0)  # (heads, kv_len)
                layer_scores.append(score)
            current = torch.stack(layer_scores, dim=0)  # (layers, heads, kv_len)
            elapsed = (time.perf_counter() - start) * 1000
        return current, elapsed

    # ─────────────────────────────────────────────
    # 2) 누적 점수 업데이트
    # ─────────────────────────────────────────────
    def update_scores(self, current_scores: torch.Tensor) -> float:
        """current_scores: (layers, heads, kv_len)."""
        with nvtx_evict_score_update():
            start = time.perf_counter()
            if self.cumulative_scores is None:
                self.cumulative_scores = current_scores.clone()
            else:
                seq_len = current_scores.shape[-1]
                old_len = self.cumulative_scores.shape[-1]
                if seq_len > old_len:
                    # 새 토큰 분 확장
                    pad = torch.zeros(
                        *self.cumulative_scores.shape[:-1],
                        seq_len - old_len,
                        device=self.device,
                        dtype=self.cumulative_scores.dtype,
                    )
                    self.cumulative_scores = torch.cat(
                        [self.cumulative_scores, pad], dim=-1
                    )
                self.cumulative_scores[:, :, :seq_len] += current_scores
            elapsed = (time.perf_counter() - start) * 1000
        return elapsed

    # ─────────────────────────────────────────────
    # 3) Eviction 판정
    # ─────────────────────────────────────────────
    def decide_keep_indices(self, seq_len: int) -> Tuple[torch.Tensor, float]:
        """
        keep할 인덱스를 결정.
        Returns: (keep_indices (sorted, 1-D), elapsed_ms)
        """
        with nvtx_evict_decision():
            start = time.perf_counter()
            # 모든 layer/head 합산 → (seq_len,)
            total_score = self.cumulative_scores.sum(dim=(0, 1))[:seq_len]

            # 최근 window 보호
            window_start = max(0, seq_len - self.window)
            window_indices = set(range(window_start, seq_len))

            # 상위 top_k (window 밖에서)
            mask = torch.ones(seq_len, dtype=torch.bool, device=self.device)
            for idx in window_indices:
                mask[idx] = False

            if mask.any():
                non_window_scores = total_score.clone()
                non_window_scores[~mask] = -float("inf")
                k = min(self.top_k, mask.sum().item())
                _, topk_indices = torch.topk(non_window_scores, k=int(k))
                keep_set = window_indices | set(topk_indices.cpu().tolist())
            else:
                keep_set = window_indices

            keep_indices = torch.tensor(sorted(keep_set), dtype=torch.long, device=self.device)
            elapsed = (time.perf_counter() - start) * 1000
        return keep_indices, elapsed

    # ─────────────────────────────────────────────
    # 4) KV Cache Compaction (실제 evict)
    # ─────────────────────────────────────────────
    def evict_kv_cache(
        self, past_key_values, keep_indices: torch.Tensor
    ) -> Tuple:
        """
        past_key_values: tuple of (key, value) per layer
           key/value shape: (batch, heads, seq_len, head_dim)
        keep_indices: (num_keep,) — 유지할 시퀀스 인덱스

        Returns: new past_key_values (compacted)
        """
        with nvtx_evict_move():
            start = time.perf_counter()
            new_past = []
            for layer_kv in past_key_values:
                k, v = layer_kv
                new_k = k[:, :, keep_indices, :]
                new_v = v[:, :, keep_indices, :]
                new_past.append((new_k, new_v))
            # 누적 점수도 compaction
            if self.cumulative_scores is not None:
                self.cumulative_scores = self.cumulative_scores[:, :, keep_indices]
            elapsed = (time.perf_counter() - start) * 1000
        return tuple(new_past), elapsed

    # ─────────────────────────────────────────────
    # 통합 인터페이스
    # ─────────────────────────────────────────────
    def step(
        self,
        attentions: Tuple[torch.Tensor, ...],
        past_key_values,
    ) -> Tuple:
        """
        하나의 디코드 스텝에서 eviction 전체 파이프라인 실행.
        Returns: (new_past_key_values, eviction_stats)
        """
        self.step_count += 1
        stats = EvictionStats()

        # KV cache 크기 측정
        if past_key_values and len(past_key_values) > 0:
            k0 = past_key_values[0][0]
            seq_len = k0.shape[2]
            stats.kv_bytes_before = sum(
                k.nelement() * k.element_size() + v.nelement() * v.element_size()
                for k, v in past_key_values
            )
        else:
            seq_len = 0
            stats.kv_bytes_before = 0

        # eviction_freq 체크
        if self.step_count % self.eviction_freq != 0:
            stats.kv_bytes_after = stats.kv_bytes_before
            stats.kept_count = seq_len
            self.stats_history.append(stats)
            return past_key_values, stats

        # 1) collect
        current_scores, stats.collect_ms = self.collect_attention_metrics(attentions)

        # 2) score update
        stats.score_update_ms = self.update_scores(current_scores)

        # 3) decide
        keep_needed = self.top_k + self.window
        if seq_len <= keep_needed:
            # 아직 evict 불필요
            stats.kv_bytes_after = stats.kv_bytes_before
            stats.kept_count = seq_len
            self.stats_history.append(stats)
            return past_key_values, stats

        keep_indices, stats.decision_ms = self.decide_keep_indices(seq_len)

        # 4) evict (move/compaction)
        new_past, stats.move_ms = self.evict_kv_cache(past_key_values, keep_indices)

        stats.evicted_count = seq_len - len(keep_indices)
        stats.kept_count = len(keep_indices)
        if new_past and len(new_past) > 0:
            stats.kv_bytes_after = sum(
                k.nelement() * k.element_size() + v.nelement() * v.element_size()
                for k, v in new_past
            )
        else:
            stats.kv_bytes_after = 0

        self.stats_history.append(stats)
        return new_past, stats

    def get_overhead_summary(self) -> Dict[str, float]:
        """전체 스텝의 eviction 오버헤드 평균."""
        if not self.stats_history:
            return {}
        import numpy as np
        return {
            "collect_ms": float(np.mean([s.collect_ms for s in self.stats_history])),
            "score_update_ms": float(np.mean([s.score_update_ms for s in self.stats_history])),
            "decision_ms": float(np.mean([s.decision_ms for s in self.stats_history])),
            "move_ms": float(np.mean([s.move_ms for s in self.stats_history])),
        }

