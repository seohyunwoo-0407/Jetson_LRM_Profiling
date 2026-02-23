# Jetson LRM Profiling

Jetson AGX Orin에서 KV Cache Eviction 스킴의 성능 분석 및 병목 검증 프로젝트입니다.

## 📁 프로젝트 구조

```
Jetson_LRM_Profiling/
├── src/                              # 공통 모듈
│   ├── __init__.py
│   ├── args.py                      # CLI 인자 파서 + 시드 고정
│   ├── model_loader.py              # HF 모델/토크나이저 로드
│   ├── nvtx_utils.py                # NVTX 마커 (scheme/eviction 단계별)
│   ├── profiling.py                 # TegrastatsLogger + CUDA ProfilingContext
│   ├── metrics.py                   # LatencyTracker + MetricsCollector (p50/p95/p99)
│   ├── kv_cache_eviction.py         # H2O 스타일 KV Cache Eviction 엔진
│   └── generation.py                # 수동 디코드 루프 (prefill/decode 분리 + eviction 주입)
│
├── experiments/                      # 실험 실행 스크립트
│   ├── exp1_baseline/
│   │   └── run.py                   # 실험1: 기본 generate()
│   ├── exp2_tot/
│   │   └── run.py                   # 실험2: Tree of Thoughts (BFS/DFS)
│   ├── exp3_debate/
│   │   └── run.py                   # 실험3: Multi-Agent Debate (3에이전트+Judge)
│   ├── exp4_mcts/
│   │   └── run.py                   # 실험4: Monte Carlo Tree Search
│   ├── exp5_baseline_eviction/
│   │   └── run.py                   # 실험5: Baseline + H2O Eviction
│   ├── exp6_tot_eviction/
│   │   └── run.py                   # 실험6: ToT + H2O Eviction
│   ├── exp7_debate_eviction/
│   │   └── run.py                   # 실험7: Debate + H2O Eviction
│   └── exp8_mcts_eviction/
│       └── run.py                   # 실험8: MCTS + H2O Eviction
│
├── configs/                          # 설정 파일
│   └── default.yaml                  # 기본 설정 + 스윕 범위
│
├── scripts/                          # 실행 스크립트
│   ├── run_all.sh                   # 전체 8실험 순차 실행
│   ├── profile_nsys.sh              # nsys 프로파일링 (NVTX 타임라인)
│   ├── profile_ncu.sh               # ncu 프로파일링 (attention 커널 카운터)
│   ├── sweep_bottleneck.sh          # CTX×GEN 붕괴 지점 스윕
│   └── sweep_eviction_params.sh     # top_k×window×freq 스윕
│
├── results/                          # 결과 저장 디렉토리 (자동 생성)
└── requirements.txt
```

## 🚀 빠른 시작

### 1. 환경 설치

```bash
cd Jetson_LRM_Profiling
pip install -r requirements.txt
```

### 2. 단일 실험 실행

```bash
python3 experiments/exp1_baseline/run.py \
    --max_context 2048 \
    --gen_length 256 \
    --repeats 5 \
    --tegrastats
    --tag kvtrace
```
prefill_latency -> prefil 과정 걸리는 시간 (밀리초)

decode_(p50, p95, p99)_ms -> per-token decoding latency 분포의 percentage  (토큰 하나 생성하는데 걸린 시간 (밀리초) ) -> 200 나오면 0.2s마다 1토큰 -> 5tokens/s 라는 뜻

tokens_per_sec->초당 생성 토큰 수

peak_kv_bytes_mean -> kv cache peak memory

tegrastats 로 잴 수 있는것 -> RAM 사용량, free memory, EMC 메모리 컨트롤러 사용률, GR3D(GPU)사용률, 온도, 전력

### 3. 전체 8실험 한번에 실행

```bash
bash scripts/run_all.sh
```

환경변수로 파라미터 조절:

```bash
CTX=4096 GEN=512 REPEATS=3 bash scripts/run_all.sh
```

### 4. nsys 프로파일링

Baseline 실험:

```bash
bash scripts/profile_nsys.sh experiments/exp1_baseline/run.py
```

MCTS + Eviction 실험:

```bash
bash scripts/profile_nsys.sh experiments/exp8_mcts_eviction/run.py \
    --eviction_top_k 128 \
    --eviction_window 32
```

### 5. ncu 프로파일링 (attention 커널)

```bash
CTX=512 GEN=16 bash scripts/profile_ncu.sh experiments/exp1_baseline/run.py
```

### 6. tegrastats 단독 로깅

```bash
sudo tegrastats --interval 100 --logfile results/tegrastats_manual.log &
# 실험 실행 후 kill
```

### 7. 붕괴 지점 스윕

Baseline:

```bash
bash scripts/sweep_bottleneck.sh experiments/exp1_baseline/run.py
```

Eviction:

```bash
bash scripts/sweep_bottleneck.sh experiments/exp5_baseline_eviction/run.py
```

### 8. Eviction 파라미터 스윕

```bash
bash scripts/sweep_eviction_params.sh experiments/exp5_baseline_eviction/run.py
```

## 📊 NVTX 마커 구조

nsys 타임라인에서 보이는 NVTX 구간 구조:

| 실험 | NVTX 구간 |
|------|-----------|
| **공통** | `RUN` → `PREFILL` → `DECODE_STEP_N` |
| **ToT** | `ToT/PROPOSE` → `ToT/EVAL` → `ToT/SELECT` → `ToT/EXPAND` |
| **Debate** | `Debate/AGENT_A_GENERATOR` → `Debate/AGENT_B_CRITIC` → `Debate/AGENT_C_VERIFIER` → `Debate/JUDGE` |
| **MCTS** | `MCTS/SELECT` → `MCTS/EXPAND` → `MCTS/ROLLOUT_N` → `MCTS/BACKPROP` |
| **Eviction** | `EVICT/ATTN_METRIC_COLLECT` → `EVICT/SCORE_UPDATE` → `EVICT/EVICT_DECISION` → `EVICT/EVICT_MOVE` |

## 📈 측정 지표

JSON 자동 저장되는 지표:

### Prefill
- `prefill_latency_ms`: Prefill 단계 지연시간

### Decode
- `decode_p50_ms`: Decode p50 지연시간
- `decode_p95_ms`: Decode p95 지연시간
- `decode_p99_ms`: Decode p99 지연시간
- `tokens_per_sec`: 초당 생성 토큰 수

### KV Cache
- `kv_bytes`: 스텝별 KV cache 크기
- `peak_kv_bytes`: 피크 KV cache 크기
- `keep_ratio`: 유지된 토큰 비율

### Eviction 오버헤드
- `collect_ms`: Attention metric 수집 시간
- `score_update_ms`: 점수 업데이트 시간
- `decision_ms`: Eviction 결정 시간
- `move_ms`: KV cache 이동/압축 시간

### HW (tegrastats)
- `RAM/lfb`: RAM 사용량 및 LFB (Last Level Buffer)
- `EMC`: External Memory Controller 대역폭
- `GR3D`: GPU 3D 엔진 사용률
- `전력`: 전력 소비량
- `온도`: GPU/CPU 온도
- `클럭`: 클럭 주파수

## 🔍 병목 판정 체크리스트

| 한계 | 로그/카운터 패턴 | 확인 방법 |
|------|------------------|-----------|
| **RAM/lfb 고갈** | `lfb → 0~1x4MB`, `RAM used ≈ total` | `tegrastats ram_used_mb ≈ ram_total_mb` |
| **EMC 대역폭 포화** | `EMC_FREQ ≥ 95%` | `tegrastats emc_freq`, `ncu dram_throughput` |
| **memcpy/compaction 지배** | `eviction move_ms > decode latency의 30%+` | JSON의 `eviction_move_ms_mean` |
| **DVFS/Thermal throttle** | 클럭 하락 + 온도 > 90°C + decode tail(p99) 급증 | `tegrastats GPU_temp`, `gr3d_freq` 하락 |
| **OOM** | CUDA OOM / 프로세스 kill | sweep의 `STATUS=FAIL` 로그 |
| **커널 stall** | `ncu stall_memory_dependency > 50%` | ncu report의 stall breakdown |

## 🎯 실험 시나리오

### 실험 1: Baseline
- 기본 `generate()` 메서드 사용

### 실험 2: ToT (Tree of Thoughts)
- BFS/DFS 탐색 방식
- "문제 정의 → 해결 방향 3가지 → 각 방향 평가 → 최고 점수 방향으로 세부 계획 3가지 → ..."

### 실험 3: Multi-Agent Debate
- 단일 모델을 메모리에 올려두고, 역할이 다른 프롬프트(생성/비판/검증 등)를 번갈아 호출
- 핑퐁(ReAct 사이클 반복)으로 결론 도출

### 실험 4: MCTS
- 생성 시 rollout 시뮬레이션으로 Reward 계산
- 수백 번 반복해 최적 논리 완성

### 실험 5-8: Eviction 적용
- 실험 5: Baseline + KV Cache Eviction
- 실험 6: ToT + KV Cache Eviction
- 실험 7: Debate + KV Cache Eviction
- 실험 8: MCTS + KV Cache Eviction

## 🔧 환경 정보

- **엣지 디바이스**: JETSON AGX ORIN 64GB
- **JetPack**: 6.2.1+b38
- **L4T 버전**: 36.4.7
- **CUDA 버전**: 12.6
- **프레임워크**: PyTorch + HF transformers
- **모델**: `ibm-granite/granite-3.1-3b-a800m-instruct`

## 📝 라이선스

이 프로젝트는 연구 목적으로 사용됩니다.

