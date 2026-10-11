"""Deprecated re-export. См. :mod:`eval.experiments.replay_experiment`."""
from __future__ import annotations

import warnings

from eval.experiments.replay_experiment import *  # noqa: F401,F403

warnings.warn(
    "eval.replay_experiment is deprecated; use eval.experiments.replay_experiment",
    DeprecationWarning,
    stacklevel=2,
)
