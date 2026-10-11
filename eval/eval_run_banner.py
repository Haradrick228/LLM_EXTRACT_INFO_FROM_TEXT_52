"""Deprecated re-export. См. :mod:`eval.reporting.log_eval_run_config_banner`."""
from __future__ import annotations

import warnings

from eval.reporting.log_eval_run_config_banner import *  # noqa: F401,F403

warnings.warn(
    "eval.eval_run_banner is deprecated; use eval.reporting.log_eval_run_config_banner",
    DeprecationWarning,
    stacklevel=2,
)
