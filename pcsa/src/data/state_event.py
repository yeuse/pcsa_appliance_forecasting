# src/pisa/data/state_event.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class ApplianceActivityStats:
    """Basic activity statistics for one appliance."""

    appliance: str
    threshold: float
    on_minutes: int
    on_rate: float
    event_count: int
    start_count: int
    stop_count: int
    median_event_duration: float
    q25_event_duration: float
    q75_event_duration: float
    lag1_autocorrelation: float
    positive_p50: float
    p95_positive: float
    p99_positive: float
    p95_on_power: float
    max_power: float


def _as_numpy_2d(x: np.ndarray | pd.DataFrame | pd.Series) -> np.ndarray:
    """Convert input to a 2D numpy array."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        arr = x.to_numpy()
    else:
        arr = np.asarray(x)

    if arr.ndim == 1:
        arr = arr[:, None]

    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape={arr.shape}")

    return arr.astype(np.float32, copy=False)


def estimate_state_thresholds(
    df: pd.DataFrame,
    appliance_cols: Iterable[str],
    min_threshold_kw: float = 0.01,
    ratio_to_p95_positive: float = 0.05,
    positive_floor_kw: float = 1e-6,
) -> dict[str, float]:
    """Estimate ON/OFF thresholds from the training split.

    The file provided by the user is in kW, so the default minimum threshold
    is 0.01 kW, i.e. 10 W.

    threshold_i = max(min_threshold_kw, ratio_to_p95_positive * p95_positive_i)

    Args:
        df: Training dataframe only.
        appliance_cols: Appliance power columns, e.g. air1, refrigerator1, ...
        min_threshold_kw: Minimum ON threshold in kW.
        ratio_to_p95_positive: Ratio multiplied by positive p95 power.
        positive_floor_kw: Values above this are considered positive samples.

    Returns:
        Mapping from appliance name to threshold in kW.
    """
    thresholds: dict[str, float] = {}

    for col in appliance_cols:
        if col not in df.columns:
            raise KeyError(f"Appliance column not found: {col}")

        values = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        values = np.clip(values, a_min=0.0, a_max=None)
        positive = values[values > positive_floor_kw]

        if positive.size == 0:
            thresholds[col] = float(min_threshold_kw)
            continue

        p95_positive = float(np.percentile(positive, 95))
        threshold = max(float(min_threshold_kw), float(ratio_to_p95_positive * p95_positive))
        thresholds[col] = threshold

    return thresholds


def build_state_labels(
    power: np.ndarray | pd.DataFrame,
    thresholds: dict[str, float] | np.ndarray | list[float],
    appliance_cols: Iterable[str] | None = None,
) -> np.ndarray:
    """Build binary ON/OFF labels from appliance power.

    Args:
        power: Array/DataFrame with shape [N, A].
        thresholds: Either dict {appliance: threshold} or array [A].
        appliance_cols: Required when thresholds is a dict and power is not a DataFrame.

    Returns:
        state: float32 array with shape [N, A], values 0 or 1.
    """
    if isinstance(power, pd.DataFrame):
        arr = power.to_numpy(dtype=np.float32)
        cols = list(power.columns)
    else:
        arr = _as_numpy_2d(power)
        cols = list(appliance_cols) if appliance_cols is not None else None

    if isinstance(thresholds, dict):
        if cols is None:
            raise ValueError("appliance_cols must be provided when thresholds is a dict.")
        th = np.asarray([thresholds[c] for c in cols], dtype=np.float32)
    else:
        th = np.asarray(thresholds, dtype=np.float32)

    if th.ndim != 1 or th.shape[0] != arr.shape[1]:
        raise ValueError(f"Threshold shape should be [A]={arr.shape[1]}, got {th.shape}")

    return (arr > th[None, :]).astype(np.float32)


def build_event_labels(
    state: np.ndarray | pd.DataFrame,
    mode: str = "both",
) -> np.ndarray:
    """Build appliance event labels from binary states.

    event_t = 1 if state_t != state_{t-1}; otherwise 0.

    Args:
        state: Binary state array [N, A].
        mode:
            - "both": start and stop are both events.
            - "start": only OFF->ON transitions are events.
            - "stop": only ON->OFF transitions are events.

    Returns:
        event: float32 array [N, A]. First row is zeros.
    """
    s = _as_numpy_2d(state)
    prev = np.zeros_like(s)
    prev[1:] = s[:-1]

    if mode == "both":
        event = (s != prev).astype(np.float32)
        event[0] = 0.0
    elif mode == "start":
        event = ((prev == 0.0) & (s == 1.0)).astype(np.float32)
        event[0] = 0.0
    elif mode == "stop":
        event = ((prev == 1.0) & (s == 0.0)).astype(np.float32)
        event[0] = 0.0
    else:
        raise ValueError(f"Unsupported event mode: {mode}")

    return event.astype(np.float32)


def compute_appliance_activity_stats(
    df: pd.DataFrame,
    appliance_cols: Iterable[str],
    thresholds: dict[str, float],
    positive_floor_kw: float = 1e-6,
) -> list[ApplianceActivityStats]:
    """Compute appliance activity statistics for reporting and debugging."""
    stats: list[ApplianceActivityStats] = []

    power_df = df[list(appliance_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    states = build_state_labels(power_df, thresholds)
    events = build_event_labels(states, mode="both")

    prev = np.zeros_like(states)
    prev[1:] = states[:-1]
    starts = ((prev == 0.0) & (states == 1.0)).astype(np.float32)
    starts[0] = 0.0
    stops = ((prev == 1.0) & (states == 0.0)).astype(np.float32)
    stops[0] = 0.0

    values = power_df.to_numpy(dtype=np.float32)
    n = len(power_df)

    for j, col in enumerate(appliance_cols):
        positive = values[:, j][values[:, j] > positive_floor_kw]
        positive_p50 = float(np.percentile(positive, 50)) if positive.size else 0.0
        p95 = float(np.percentile(positive, 95)) if positive.size else 0.0
        p99 = float(np.percentile(positive, 99)) if positive.size else 0.0
        max_power = float(np.max(values[:, j])) if n else 0.0

        state_j = states[:, j].astype(bool)
        on_power = values[:, j][state_j]
        p95_on_power = (
            float(np.percentile(on_power, 95)) if on_power.size else 0.0
        )
        prev_j = np.zeros_like(state_j)
        prev_j[1:] = state_j[:-1]
        start_idx = np.flatnonzero((~prev_j) & state_j)
        durations: list[int] = []
        for idx in start_idx:
            after = state_j[idx:]
            off_positions = np.flatnonzero(~after)
            if off_positions.size:
                duration = int(off_positions[0])
                if duration <= 0:
                    duration = 1
            else:
                duration = int(len(after))
            durations.append(duration)

        if durations:
            duration_arr = np.asarray(durations, dtype=np.float32)
            median_duration = float(np.percentile(duration_arr, 50))
            q25_duration = float(np.percentile(duration_arr, 25))
            q75_duration = float(np.percentile(duration_arr, 75))
        else:
            median_duration = 0.0
            q25_duration = 0.0
            q75_duration = 0.0

        series = values[:, j].astype(np.float32)
        if n > 1 and float(series[:-1].std()) > 0.0 and float(series[1:].std()) > 0.0:
            x0 = series[:-1] - float(series[:-1].mean())
            x1 = series[1:] - float(series[1:].mean())
            denom = float(np.sqrt((x0 * x0).sum() * (x1 * x1).sum()))
            lag1_autocorrelation = float((x0 * x1).sum() / denom) if denom > 0.0 else 0.0
        else:
            lag1_autocorrelation = 0.0

        on_minutes = int(states[:, j].sum())
        stats.append(
            ApplianceActivityStats(
                appliance=col,
                threshold=float(thresholds[col]),
                on_minutes=on_minutes,
                on_rate=float(on_minutes / max(n, 1)),
                event_count=int(events[:, j].sum()),
                start_count=int(starts[:, j].sum()),
                stop_count=int(stops[:, j].sum()),
                median_event_duration=median_duration,
                q25_event_duration=q25_duration,
                q75_event_duration=q75_duration,
                lag1_autocorrelation=lag1_autocorrelation,
                positive_p50=positive_p50,
                p95_positive=p95,
                p99_positive=p99,
                p95_on_power=p95_on_power,
                max_power=max_power,
            )
        )

    return stats
