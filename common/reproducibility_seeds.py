"""Глобальные seed'ы для воспроизводимости eval-прогонов.

Зачем
-----
Любая retrieval-метрика чувствительна к порядку обработки вопросов и
случайным выборкам. Чтобы прогоны были битово сравнимы между запусками,
вызывать ``set_global_seed(0)`` в начале CLI и оркестратора.

Используется в ``eval.retrieval.run_retrieval_eval`` и
``eval.retrieval.run_retrieval_matrix`` через
``core.monitoring.experiment_run_context``.
"""
from __future__ import annotations

import os
import random
from typing import Optional


def set_global_seed(seed: int = 0) -> int:
    """Зафиксировать seed для Python ``random``, NumPy, PyTorch (CPU+CUDA).

    Возвращает фактически применённый seed (для записи в manifest).
    """
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
    except Exception:
        pass

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    return seed


def env_seed(default: int = 0, var_name: str = "EVAL_SEED") -> int:
    """Прочитать seed из env-переменной, иначе вернуть ``default``."""
    raw: Optional[str] = os.getenv(var_name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


__all__ = ["set_global_seed", "env_seed"]
