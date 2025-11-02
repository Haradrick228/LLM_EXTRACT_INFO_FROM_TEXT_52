from __future__ import annotations

from typing import Dict


def format_number(value: object, digits: int = 0) -> str:
    try:
        digits = max(0, int(digits))
    except (TypeError, ValueError):
        digits = 0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if digits == 0 or numeric.is_integer():
        return format(int(round(numeric)), ",").replace(",", " ")
    return format(numeric, f",.{digits}f").replace(",", " ")


def format_percentage(value: object, digits: int = 1) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    digits = max(0, int(digits))
    return f"{numeric * 100:.{digits}f}%"


def format_metric_value(metric: Dict[str, object]) -> str:
    fmt = str(metric.get("format", "number"))
    digits = metric.get("digits", 2)
    value = metric.get("value", 0)
    if fmt == "percent":
        return format_percentage(value, digits)
    if fmt == "integer":
        return format_number(value, 0)
    return format_number(value, digits)
