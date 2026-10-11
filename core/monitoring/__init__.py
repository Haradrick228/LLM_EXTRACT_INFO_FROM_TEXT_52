"""Мониторинг и контекст eval-прогонов.

Точка входа всех CLI в ``eval/**``: ``experiment_run_context`` создаёт
каталог прогона ``eval/runs/<ts>_f<N>_<slug>/``, пишет manifest, делает
снапшот окружения, запускает фоновый сэмплер GPU, оборачивает stdout/stderr
в ``run.log``. ``gpu_usage_sampler`` — фоновый поток-семплер VRAM/util.
"""
from .experiment_run_context import ExperimentRunContext, experiment_run_context
from .gpu_usage_sampler import GpuUsageSampler

__all__ = [
    "ExperimentRunContext",
    "experiment_run_context",
    "GpuUsageSampler",
]
