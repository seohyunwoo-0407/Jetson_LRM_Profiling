"""공통 CLI 인자 파서 및 시드 설정."""
import argparse
import random
import numpy as np
import torch


def get_common_parser(description: str = "Jetson LRM Profiling Experiment") -> argparse.ArgumentParser:
    """모든 실험에서 공통으로 사용하는 CLI 인자 파서."""
    parser = argparse.ArgumentParser(description=description)

    # ── 모델 / 생성 설정 ──
    parser.add_argument("--model_name", type=str,
                        default="ibm-granite/granite-3.1-3b-a800m-instruct",
                        help="HuggingFace 모델 이름 또는 경로")
    parser.add_argument("--max_context", type=int, default=2048,
                        help="최대 컨텍스트 길이 (prefill)")
    parser.add_argument("--gen_length", type=int, default=256,
                        help="생성할 토큰 수")
    parser.add_argument("--prompt", type=str, default=None,
                        help="커스텀 프롬프트 (없으면 기본 사용)")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="프롬프트 파일 경로 (.txt)")

    # ── 실험 제어 ──
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드")
    parser.add_argument("--warmup", type=int, default=2,
                        help="워밍업 반복 수")
    parser.add_argument("--repeats", type=int, default=5,
                        help="측정 반복 수")
    parser.add_argument("--device", type=str, default="cuda",
                        help="디바이스 (cuda / cpu)")

    # ── Eviction 파라미터 (exp5~8에서 사용) ──
    parser.add_argument("--eviction_enabled", action="store_true",
                        help="KV cache eviction 활성화")
    parser.add_argument("--eviction_top_k", type=int, default=256,
                        help="상위 K개 토큰 유지")
    parser.add_argument("--eviction_window", type=int, default=64,
                        help="최근 window 토큰 유지")
    parser.add_argument("--eviction_freq", type=int, default=1,
                        help="N 스텝마다 eviction 수행")

    # ── 프로파일링 ──
    parser.add_argument("--tegrastats", action="store_true",
                        help="tegrastats 로깅 활성화")
    parser.add_argument("--tegrastats_interval", type=int, default=100,
                        help="tegrastats 샘플링 간격 (ms)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="결과 저장 디렉토리")
    parser.add_argument("--tag", type=str, default="",
                        help="실험 태그 (파일명에 포함)")

    return parser


def setup_seed(seed: int):
    """재현성을 위한 시드 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

