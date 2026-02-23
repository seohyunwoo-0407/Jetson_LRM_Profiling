"""
공통 토큰 생성 루틴 — prefill/decode 분리, NVTX 마커, 스텝별 레이턴시 측정.
Eviction 매니저 주입 가능.
"""
import time
import torch
from typing import Optional, Dict, Any, List, Tuple

from .nvtx_utils import nvtx_range, nvtx_prefill, nvtx_decode_step
from .profiling import ProfilingContext, collect_cuda_metrics
from .metrics import LatencyTracker, LatencyRecord
from .kv_cache_eviction import HeavyHitterEvictionManager, EvictionStats


def generate_with_profiling(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_length: int = 256,
    eviction_mgr: Optional[HeavyHitterEvictionManager] = None,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> Tuple[torch.Tensor, LatencyTracker, List[EvictionStats]]:
    """
    수동 디코드 루프 — prefill / decode 분리, 스텝별 계측.

    Returns:
        generated_ids: (1, total_len) 전체 시퀀스
        tracker: LatencyTracker (prefill + 각 decode step)
        eviction_stats: list of EvictionStats per step
    """
    tracker = LatencyTracker()
    eviction_stats_list: List[EvictionStats] = []
    kv_bytes_timeline: List[int] = []
    past_key_values = None
    generated_tokens: List[int] = []
    device = input_ids.device

    need_attentions = eviction_mgr is not None

    # ═══════════════════════════════════════════
    # PREFILL
    # ═══════════════════════════════════════════
    with nvtx_prefill():
        with ProfilingContext("prefill") as pctx:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
                output_attentions=need_attentions,
            )
    past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :]
    mem = collect_cuda_metrics()

    # prefill 기록
    kv_bytes = _calc_kv_bytes(past_key_values)
    tracker.add(LatencyRecord(
        step=0,
        latency_ms=pctx.elapsed_ms,
        phase="prefill",
        kv_bytes=kv_bytes,
        mem_allocated_gb=mem.get("mem_allocated_gb", 0),
    ))
    kv_bytes_timeline.append(kv_bytes)

    # 첫 토큰 샘플링
    next_token = _sample_token(logits, temperature, do_sample)
    generated_tokens.append(next_token.item())

    # eviction (prefill 직후)
    if eviction_mgr and need_attentions and outputs.attentions is not None:
        past_key_values, e_stat = eviction_mgr.step(outputs.attentions, past_key_values)
        eviction_stats_list.append(e_stat)

    # ═══════════════════════════════════════════
    # DECODE LOOP
    # ═══════════════════════════════════════════
    for step in range(1, gen_length):
        with nvtx_decode_step(step):
            cur_input = next_token.unsqueeze(0)
            cur_mask = torch.ones(
                1, past_key_values[0][0].shape[2] + 1,
                dtype=torch.long, device=device
            )

            with ProfilingContext(f"decode_{step}") as pctx:
                outputs = model(
                    input_ids=cur_input,
                    attention_mask=cur_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=need_attentions,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            # eviction
            e_stat = None
            if eviction_mgr and need_attentions and outputs.attentions is not None:
                past_key_values, e_stat = eviction_mgr.step(
                    outputs.attentions, past_key_values
                )
                eviction_stats_list.append(e_stat)

            kv_bytes = _calc_kv_bytes(past_key_values)
            kv_bytes_timeline.append(kv_bytes)
            mem = collect_cuda_metrics()

            tracker.add(LatencyRecord(
                step=step,
                latency_ms=pctx.elapsed_ms,
                phase="decode",
                kv_bytes=kv_bytes,
                keep_ratio=(e_stat.kept_count / (e_stat.kept_count + e_stat.evicted_count))
                    if e_stat and (e_stat.kept_count + e_stat.evicted_count) > 0 else 1.0,
                evicted_count=e_stat.evicted_count if e_stat else 0,
                mem_allocated_gb=mem.get("mem_allocated_gb", 0),
            ))

            # 샘플링
            next_token = _sample_token(logits, temperature, do_sample)
            generated_tokens.append(next_token.item())

            # EOS 체크
            if next_token.item() == model.config.eos_token_id:
                break

    all_ids = torch.cat([input_ids, torch.tensor([generated_tokens], device=device)], dim=-1)
    return all_ids, tracker, eviction_stats_list


def _sample_token(logits: torch.Tensor, temperature: float, do_sample: bool) -> torch.Tensor:
    if do_sample and temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    else:
        return logits.argmax(dim=-1)


def _calc_kv_bytes(past_key_values) -> int:
    if past_key_values is None:
        return 0
    total = 0

    import torch

    for layer in past_key_values:
        # layer가 바로 텐서인 경우 (드물지만)
        if torch.is_tensor(layer):
            tensors = [layer]
        # 튜플 / 리스트인 경우: 안에 텐서 여러 개 들어 있음
        elif isinstance(layer, (tuple, list)):
            tensors = [t for t in layer if torch.is_tensor(t)]
        # dict 형태인 경우 (일부 모델)
        elif isinstance(layer, dict):
            tensors = [t for t in layer.values() if torch.is_tensor(t)]
        else:
            continue

        for t in tensors:
            total += t.nelement() * t.element_size()

    return total