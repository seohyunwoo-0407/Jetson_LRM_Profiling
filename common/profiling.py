"""프로파일링 도구 (tegrastats 로거, CUDA 메트릭 수집)."""
import os
import time
import subprocess
import threading
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch


class TegrastatsLogger:
    """
    tegrastats 데몬을 백그라운드 스레드로 돌리며 파싱/저장.
    사용법:
        with TegrastatsLogger(interval_ms=100, output_path="teg.jsonl") as teg:
            ...  # 실험 코드
        records = teg.records
    """

    def __init__(self, interval_ms: int = 100, output_path: str = "tegrastats.jsonl"):
        self.interval_ms = interval_ms
        self.output_path = output_path
        self.records: List[Dict[str, Any]] = []
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self):
        """tegrastats 프로세스를 시작하고 파싱 스레드를 띄움."""
        cmd = ["tegrastats", "--interval", str(self.interval_ms)]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
        except FileNotFoundError:
            print("[TegrastatsLogger] WARNING: tegrastats not found — skipping HW logging.")
            return
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        if self._thread:
            self._thread.join(timeout=5)
        # 결과 저장
        if self.records:
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "w") as f:
                for rec in self.records:
                    f.write(json.dumps(rec) + "\n")
            print(f"[TegrastatsLogger] Saved {len(self.records)} records → {self.output_path}")

    def _reader_loop(self):
        while not self._stop_event.is_set():
            if self._proc is None or self._proc.stdout is None:
                break
            line = self._proc.stdout.readline()
            if not line:
                break
            record = self._parse_line(line.strip())
            if record:
                self.records.append(record)

    @staticmethod
    def _parse_line(line: str) -> Optional[Dict[str, Any]]:
        """tegrastats 한 줄을 파싱하여 dict로 변환."""
        record: Dict[str, Any] = {"raw": line, "ts": datetime.now().isoformat()}
        try:
            # RAM 파싱: RAM XXXX/YYYYMB (lfb NxZMB)
            if "RAM" in line:
                ram_part = line.split("RAM")[1].split("(")[0].strip()
                used, total = ram_part.replace("MB", "").split("/")
                record["ram_used_mb"] = int(used)
                record["ram_total_mb"] = int(total)
            if "lfb" in line:
                lfb_part = line.split("lfb")[1].split(")")[0].strip()
                record["lfb"] = lfb_part
            # EMC 파싱: EMC_FREQ XX%@YYYY
            if "EMC_FREQ" in line:
                emc_part = line.split("EMC_FREQ")[1].split(" ")[0].strip()
                record["emc_freq"] = emc_part
            # GR3D 파싱: GR3D_FREQ XX%@YYYY
            if "GR3D_FREQ" in line:
                gr3d_part = line.split("GR3D_FREQ")[1].split(" ")[0].strip()
                record["gr3d_freq"] = gr3d_part
            # 전력: VDD_GPU_SOC / VDD_CPU_CV
            for pwr_tag in ["VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0"]:
                if pwr_tag in line:
                    pwr_part = line.split(pwr_tag)[1].split(" ")[0].strip()
                    record[pwr_tag.lower()] = pwr_part
            # 온도
            for temp_tag in ["CPU@", "GPU@", "tj@"]:
                if temp_tag in line:
                    idx = line.index(temp_tag)
                    temp_val = line[idx:].split(" ")[0]
                    record[temp_tag.replace("@", "_temp")] = temp_val
        except Exception:
            pass
        return record


class ProfilingContext:
    """CUDA 이벤트 기반 정밀 시간 측정 context manager."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_event.record()
        return self

    def __exit__(self, *exc):
        self.end_event.record()
        torch.cuda.synchronize()
        self.elapsed_ms = self.start_event.elapsed_time(self.end_event)


def collect_cuda_metrics() -> Dict[str, float]:
    """현재 CUDA 메모리 상태를 수집."""
    if not torch.cuda.is_available():
        return {}
    return {
        "mem_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "mem_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "mem_max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "mem_max_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
    }

