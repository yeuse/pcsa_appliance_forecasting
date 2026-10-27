"""Shared, dependency-free checkpoint rules for training and monitoring."""
from __future__ import annotations

import math
from typing import Mapping, Any


def metric_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def on_safety_status(
    metrics: Mapping[str, Any], ceiling: float | None, scope: str = "home",
    prefix: str = "val/",
) -> tuple[bool, float, float]:
    """Return eligibility, worst ON risk and evaluated pair coverage.

    Missing/NaN diagnostics cannot satisfy an enabled constraint.
    A disabled constraint does not make a safety claim.
    """
    if scope not in {"home", "home_appliance"}:
        raise ValueError(f"Unknown ON-safety scope: {scope}")
    pair = scope == "home_appliance"
    key = ("domain/worst_home_appliance_normalized_on_power_MAE" if pair
           else "domain/worst_home_normalized_on_power_MAE")
    worst = metric_number(metrics.get(prefix + key))
    coverage = metric_number(metrics.get(prefix + "domain/on_pair_coverage"))
    if ceiling is None:
        return True, worst, coverage
    ceiling = float(ceiling)
    if not math.isfinite(ceiling) or ceiling <= 0:
        raise ValueError("ON safety ceiling must be finite and positive.")
    eligible = (math.isfinite(worst) and worst <= ceiling
                and (not pair or coverage == 1.0))
    return eligible, worst, coverage


def monitor_improved(value: float, best: float, mode: str, min_delta: float) -> bool:
    if mode not in {"min", "max"}:
        raise ValueError(f"Unknown monitor mode: {mode}")
    if not math.isfinite(value):
        return False
    return value < best - min_delta if mode == "min" else value > best + min_delta
