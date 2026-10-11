"""Фоновый сэмплер GPU: VRAM, температура, утилизация → JSONL.

Используется ``experiment_run_context.start_gpu_usage_sampler()``. Если pynvml
недоступен (или GPU нет) — сэмплер работает в режиме no-op, но не падает.

Каждая строка ``monitoring.jsonl``:
``{"ts": "<ISO>", "device": 0, "name": "...", "vram_alloc_mb": 1234,
"vram_reserved_mb": 4567, "vram_total_mb": 16384, "util_pct": 65, "temp_c": 71}``
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _NvmlBackend:
    """Обёртка над pynvml; ленивые импорты."""

    def __init__(self) -> None:
        self._handles: List[Any] = []
        self._nvml: Optional[Any] = None
        self._available = False
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            for i in range(n):
                self._handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
            self._nvml = pynvml
            self._available = bool(self._handles)
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def sample(self) -> List[Dict[str, Any]]:
        if not self._available or self._nvml is None:
            return []
        out: List[Dict[str, Any]] = []
        for idx, h in enumerate(self._handles):
            try:
                name = self._nvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                mem = self._nvml.nvmlDeviceGetMemoryInfo(h)
                util = self._nvml.nvmlDeviceGetUtilizationRates(h)
                try:
                    temp = self._nvml.nvmlDeviceGetTemperature(h, self._nvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    temp = None
                out.append(
                    {
                        "ts": _now_iso(),
                        "device": idx,
                        "name": name,
                        "vram_used_mb": int(mem.used) // (1024 * 1024),
                        "vram_total_mb": int(mem.total) // (1024 * 1024),
                        "util_pct": int(util.gpu),
                        "temp_c": int(temp) if temp is not None else None,
                    }
                )
            except Exception:
                continue
        return out

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


class _TorchBackend:
    """Fallback на torch.cuda — нет utilization, нет температуры, только VRAM."""

    def __init__(self) -> None:
        self._torch: Optional[Any] = None
        self._available = False
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self._torch = torch
                self._available = True
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def sample(self) -> List[Dict[str, Any]]:
        if not self._available or self._torch is None:
            return []
        out: List[Dict[str, Any]] = []
        for idx in range(self._torch.cuda.device_count()):
            try:
                name = self._torch.cuda.get_device_name(idx)
                alloc = int(self._torch.cuda.memory_allocated(idx))
                reserved = int(self._torch.cuda.memory_reserved(idx))
                total_props = self._torch.cuda.get_device_properties(idx)
                total = int(getattr(total_props, "total_memory", 0))
                out.append(
                    {
                        "ts": _now_iso(),
                        "device": idx,
                        "name": name,
                        "vram_alloc_mb": alloc // (1024 * 1024),
                        "vram_reserved_mb": reserved // (1024 * 1024),
                        "vram_total_mb": total // (1024 * 1024) if total else None,
                        "util_pct": None,
                        "temp_c": None,
                    }
                )
            except Exception:
                continue
        return out

    def close(self) -> None:
        return None


class GpuUsageSampler:
    """Фоновый поток, сэмплит GPU каждые ``interval_s`` и пишет JSONL.

    Если GPU нет — работает в no-op режиме (метод ``sample()`` вернёт пустой
    список, файл не создаётся). Идемпотентен: повторные ``start()`` игнорируются.
    """

    def __init__(self, output_path: Path, *, interval_s: float = 5.0) -> None:
        self.output_path = Path(output_path)
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._backend = _NvmlBackend()
        if not self._backend.available:
            self._backend = _TorchBackend()  # type: ignore[assignment]

    @property
    def available(self) -> bool:
        return getattr(self._backend, "available", False)

    def _run(self) -> None:
        if not self.available:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as f:
            while not self._stop.is_set():
                rows = self._backend.sample()
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                self._stop.wait(self.interval_s)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if not self.available:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gpu-usage-sampler", daemon=True)
        self._thread.start()

    def stop(self, *, timeout_s: float = 5.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout_s)
        self._thread = None
        try:
            self._backend.close()
        except Exception:
            pass


__all__ = ["GpuUsageSampler"]
