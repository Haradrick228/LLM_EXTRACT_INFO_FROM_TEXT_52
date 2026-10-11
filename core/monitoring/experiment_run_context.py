"""Context manager одного eval-прогона.

Создаёт каталог ``eval/runs/<YYYYMMDD_HHMMSS>_f<N>_<slug>/`` со всеми
обязательными файлами:

- ``manifest.json`` — phase, slug, scenario_map, embed/rerank, gold_sha, git_sha;
- ``env_snapshot.json`` — детерминированный os.environ без секретов;
- ``pip_freeze.txt`` — версии пакетов;
- ``run.log`` — копия stdout/stderr через tee;
- ``monitoring.jsonl`` — фоновый GPU-сэмплер (если доступно);
- ``traces/<scenario>.jsonl`` — пишутся вручную через ``ctx.append_trace(...)``;
- ``summary.json``, ``metrics_matrix.md`` — пишутся фазой по завершении.

Пример::

    with experiment_run_context(phase="f2", slug="hybrid_rrf_bge_m3") as ctx:
        ctx.log("dense.query", took_ms=120)
        ctx.append_trace("scenario_a", {"q_id": "x", ...})
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Dict, Iterator, Optional

from common import environment_snapshot, project_paths, reproducibility_seeds
from common.structured_logging import get_logger, log_event

from .gpu_usage_sampler import GpuUsageSampler

_log = get_logger(__name__)


def _format_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_slug(value: str) -> str:
    out = []
    for ch in (value or "").strip().lower():
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        elif ch.isspace() or ch in ".:":
            out.append("_")
    s = "".join(out).strip("_") or "run"
    return s[:120]


@dataclass
class ExperimentRunContext:
    """Immutable handle на каталог прогона."""

    phase: str
    slug: str
    run_dir: Path
    run_id: str
    gpu_sampler: Optional[GpuUsageSampler] = None
    _log_file: Optional[IO[str]] = field(default=None, repr=False)
    _trace_files: Dict[str, IO[str]] = field(default_factory=dict, repr=False)
    _closed: bool = field(default=False, repr=False)

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def env_snapshot_path(self) -> Path:
        return self.run_dir / "env_snapshot.json"

    @property
    def pip_freeze_path(self) -> Path:
        return self.run_dir / "pip_freeze.txt"

    @property
    def run_log_path(self) -> Path:
        return self.run_dir / "run.log"

    @property
    def monitoring_path(self) -> Path:
        return self.run_dir / "monitoring.jsonl"

    @property
    def traces_dir(self) -> Path:
        return self.run_dir / "traces"

    def trace_path(self, scenario: str) -> Path:
        return self.traces_dir / f"{_safe_slug(scenario)}.jsonl"

    def write_manifest(self, manifest: Dict[str, Any]) -> None:
        environment_snapshot.write_manifest_json(self.manifest_path, manifest)

    def write_summary(self, summary: Dict[str, Any]) -> None:
        environment_snapshot.write_manifest_json(self.run_dir / "summary.json", summary)

    def write_metrics_matrix_md(self, markdown: str) -> None:
        path = self.run_dir / "metrics_matrix.md"
        path.write_text(markdown, encoding="utf-8")

    def append_trace(self, scenario: str, payload: Dict[str, Any]) -> None:
        slug = _safe_slug(scenario)
        sink = self._trace_files.get(slug)
        if sink is None:
            self.traces_dir.mkdir(parents=True, exist_ok=True)
            sink = self.trace_path(scenario).open("a", encoding="utf-8")
            self._trace_files[slug] = sink
        sink.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sink.flush()

    def log(self, event: str, **fields: Any) -> None:
        log_event(_log, event, phase=self.phase, run_id=self.run_id, **fields)

    def start_gpu_usage_sampler(self, *, interval_s: float = 5.0) -> None:
        if self.gpu_sampler is not None:
            return
        sampler = GpuUsageSampler(self.monitoring_path, interval_s=interval_s)
        sampler.start()
        self.gpu_sampler = sampler
        self.log("gpu_sampler.started", available=sampler.available, interval_s=interval_s)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.gpu_sampler is not None:
            try:
                self.gpu_sampler.stop()
                self.log("gpu_sampler.stopped")
            except Exception:
                pass
        for sink in self._trace_files.values():
            try:
                sink.close()
            except Exception:
                pass
        if self._log_file is not None:
            try:
                self._log_file.flush()
                self._log_file.close()
            except Exception:
                pass


@contextmanager
def experiment_run_context(
    *,
    phase: str,
    slug: str,
    seed: int = 0,
    gold_path: Optional[Path] = None,
    runs_root: Optional[Path] = None,
    extra_manifest: Optional[Dict[str, Any]] = None,
    auto_start_gpu_sampler: bool = True,
    tee_stderr: bool = True,
) -> Iterator[ExperimentRunContext]:
    """Создать каталог прогона, записать manifest/env/pip_freeze, отдать контекст.

    На выходе автоматически останавливает GPU-сэмплер, закрывает trace-файлы,
    дофлашивает run.log.
    """
    reproducibility_seeds.set_global_seed(seed)

    root = runs_root or project_paths.runs_dir()
    ts = _format_timestamp()
    phase_clean = _safe_slug(phase)
    slug_clean = _safe_slug(slug)
    run_id = f"{ts}_{phase_clean}_{slug_clean}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = None
    if tee_stderr:
        log_file = (run_dir / "run.log").open("a", encoding="utf-8")
        _attach_stream_tee(log_file)

    ctx = ExperimentRunContext(
        phase=phase_clean,
        slug=slug_clean,
        run_dir=run_dir,
        run_id=run_id,
        _log_file=log_file,
    )

    manifest: Dict[str, Any] = {
        "phase": phase_clean,
        "slug": slug_clean,
        "run_id": run_id,
        "seed": seed,
        "environment": environment_snapshot.build_environment_manifest(
            project_paths.project_root(), gold_path=gold_path,
        ),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    ctx.write_manifest(manifest)

    snap = environment_snapshot.env_snapshot()
    ctx.env_snapshot_path.write_text(
        json.dumps(snap, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    freeze = environment_snapshot.pip_freeze_text()
    if freeze:
        ctx.pip_freeze_path.write_text(freeze, encoding="utf-8")

    ctx.log("run.start", run_dir=str(run_dir))

    if auto_start_gpu_sampler:
        try:
            ctx.start_gpu_usage_sampler()
        except Exception as exc:
            ctx.log("gpu_sampler.start_failed", error=repr(exc))

    try:
        yield ctx
    finally:
        ctx.log("run.end")
        ctx.close()


class _StreamTee:
    """Дублирует запись в исходный stream и в file."""

    def __init__(self, original: IO[str], extra: IO[str]) -> None:
        self._original = original
        self._extra = extra

    def write(self, data: str) -> int:
        try:
            n = self._original.write(data)
        except Exception:
            n = 0
        try:
            self._extra.write(data)
        except Exception:
            pass
        return n

    def flush(self) -> None:
        for s in (self._original, self._extra):
            try:
                s.flush()
            except Exception:
                pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


_TEE_ATTACHED: Dict[str, _StreamTee] = {}


def _attach_stream_tee(extra: IO[str]) -> None:
    """Подменяет sys.stderr/sys.stdout на tee. Идемпотентно по PID."""
    pid_key = f"{os.getpid()}::{id(extra)}"
    if pid_key in _TEE_ATTACHED:
        return
    new_err = _StreamTee(sys.stderr, extra)
    sys.stderr = new_err  # type: ignore[assignment]
    _TEE_ATTACHED[pid_key] = new_err


__all__ = ["ExperimentRunContext", "experiment_run_context"]
