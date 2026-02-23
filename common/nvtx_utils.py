"""NVTX 마커 유틸리티 — nsys 타임라인에서 scheme/eviction 단계 분리 가시화."""
import torch
from contextlib import contextmanager

# ═══════════════════════════════════════════════════════════
# 기본 NVTX wrapper
# ═══════════════════════════════════════════════════════════

@contextmanager
def nvtx_range(name: str, color: str = "blue"):
    """NVTX push/pop 을 context manager 로 래핑."""
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def nvtx_push(name: str):
    torch.cuda.nvtx.range_push(name)


def nvtx_pop():
    torch.cuda.nvtx.range_pop()


# ═══════════════════════════════════════════════════════════
# Scheme-별 NVTX 마커 (삽입 포인트 가이드)
# ═══════════════════════════════════════════════════════════

# ── 공통 ──
@contextmanager
def nvtx_run():
    """전체 실험 구간."""
    with nvtx_range("RUN", color="green"):
        yield

@contextmanager
def nvtx_prefill():
    with nvtx_range("PREFILL", color="yellow"):
        yield

@contextmanager
def nvtx_decode_step(step: int):
    with nvtx_range(f"DECODE_STEP_{step}", color="cyan"):
        yield

# ── ToT 단계 ──
@contextmanager
def nvtx_tot_propose():
    with nvtx_range("ToT/PROPOSE", color="orange"):
        yield

@contextmanager
def nvtx_tot_eval():
    with nvtx_range("ToT/EVAL", color="red"):
        yield

@contextmanager
def nvtx_tot_select():
    with nvtx_range("ToT/SELECT", color="purple"):
        yield

@contextmanager
def nvtx_tot_expand():
    with nvtx_range("ToT/EXPAND", color="pink"):
        yield

# ── Debate 단계 ──
@contextmanager
def nvtx_debate_agent(agent_name: str):
    with nvtx_range(f"Debate/{agent_name}", color="blue"):
        yield

@contextmanager
def nvtx_debate_judge():
    with nvtx_range("Debate/JUDGE", color="red"):
        yield

# ── MCTS 단계 ──
@contextmanager
def nvtx_mcts_select():
    with nvtx_range("MCTS/SELECT", color="green"):
        yield

@contextmanager
def nvtx_mcts_expand():
    with nvtx_range("MCTS/EXPAND", color="yellow"):
        yield

@contextmanager
def nvtx_mcts_rollout(rollout_idx: int):
    with nvtx_range(f"MCTS/ROLLOUT_{rollout_idx}", color="orange"):
        yield

@contextmanager
def nvtx_mcts_backprop():
    with nvtx_range("MCTS/BACKPROP", color="red"):
        yield

# ── Eviction 단계 ──
@contextmanager
def nvtx_evict_attn_collect():
    with nvtx_range("EVICT/ATTN_METRIC_COLLECT", color="magenta"):
        yield

@contextmanager
def nvtx_evict_score_update():
    with nvtx_range("EVICT/SCORE_UPDATE", color="magenta"):
        yield

@contextmanager
def nvtx_evict_decision():
    with nvtx_range("EVICT/EVICT_DECISION", color="magenta"):
        yield

@contextmanager
def nvtx_evict_move():
    with nvtx_range("EVICT/EVICT_MOVE", color="magenta"):
        yield

