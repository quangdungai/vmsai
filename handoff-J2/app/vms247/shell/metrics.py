"""
L7 — Metric logger: FPS · latency · VRAM theo (module, engine).

TRÁI TIM của MVP1: không có cái này thì không "đo được tính khả thi".
VRAM đọc qua pynvml (chính xác) -> torch.cuda (fallback) -> 0.
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager

from vms247.core.schemas import Engine, Metric, ModuleId

# --- VRAM probe (thử pynvml, rồi torch, rồi bỏ qua) ---------------------------
_nvml_ok = False
try:  # pragma: no cover - phụ thuộc môi trường
    import pynvml

    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    _nvml_ok = True
except Exception:
    _nvml_handle = None


def read_vram_mb() -> float:
    if _nvml_ok and _nvml_handle is not None:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
            return info.used / (1024 * 1024)
        except Exception:
            pass
    try:  # pragma: no cover
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


class MetricLogger:
    """Giữ EMA FPS + latency gần nhất + VRAM cho từng (module, engine)."""

    def __init__(self, window: int = 30) -> None:
        self._lat: dict[tuple[str, str], deque[float]] = {}
        self._last: dict[tuple[str, str], Metric] = {}
        self.window = window

    @contextmanager
    def measure(self, module: ModuleId, engine: Engine):
        """Dùng: `with logger.measure(M1, BASELINE): dets = plugin.process(...)`"""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.record(module, engine, dt_ms)

    def record(self, module: ModuleId, engine: Engine, latency_ms: float) -> Metric:
        key = (module.value, engine.value)
        buf = self._lat.setdefault(key, deque(maxlen=self.window))
        buf.append(latency_ms)
        avg_ms = sum(buf) / len(buf)
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        m = Metric(
            module=module,
            engine=engine,
            fps=fps,
            latency_ms=latency_ms,
            vram_mb=read_vram_mb(),
            time=time.time(),
        )
        self._last[key] = m
        return m

    def snapshot(self) -> list[dict]:
        return [m.to_row() for m in self._last.values()]
