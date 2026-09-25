"""Causal, aggregate-only baselines for Home 7951 forecasting experiments.

These baselines use aggregate history at inference. Training labels estimate
fixed parameters, and validation labels select operating thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .metrics import find_best_event_thresholds, window_start_risk_metrics


def estimate_training_appliance_shares(train_dataset) -> Tensor:
    """Estimate selected-appliance aggregate shares from the training split.

    The returned vector is fixed before validation/test evaluation.  At
    inference, :func:`aggregate_history_share_persistence_power` combines it
    solely with the last observed aggregate value, never with ``y_hist_power``.
    """
    appliance_power = np.asarray(train_dataset.y_power, dtype=np.float64)
    mains = np.asarray(train_dataset.mains, dtype=np.float64)
    if appliance_power.ndim != 2 or mains.ndim != 1:
        raise ValueError("train_dataset should expose y_power [T, A] and mains [T].")
    if appliance_power.shape[0] != mains.shape[0]:
        raise ValueError("train_dataset y_power and mains lengths should match.")

    mean_mains = max(float(np.mean(np.clip(mains, 0.0, None))), 1e-8)
    shares = np.mean(np.clip(appliance_power, 0.0, None), axis=0) / mean_mains
    shares = np.clip(shares, 0.0, None)
    # Preserve unmodeled aggregate load; normalize only when shares exceed 1.
    if float(shares.sum()) > 1.0:
        shares = shares / float(shares.sum())
    return torch.tensor(shares, dtype=torch.float32)


def aggregate_history_share_persistence_power(
    y_hist_mains: Tensor,
    appliance_shares: Tensor | Sequence[float],
    horizon: int,
) -> Tensor:
    """Forecast appliance power from last observed aggregate and fixed shares.

    Args:
        y_hist_mains: observed aggregate history, ``[B, T]``.
        appliance_shares: training-only mean appliance-to-aggregate shares,
            ``[A]``.
        horizon: requested forecast length.
    """
    history = torch.as_tensor(y_hist_mains, dtype=torch.float32)
    if history.dim() != 2 or history.size(-1) < 1:
        raise ValueError("y_hist_mains should have shape [B, T] with T >= 1.")
    shares = torch.as_tensor(appliance_shares, dtype=history.dtype).view(-1)
    if shares.numel() == 0 or bool((shares < 0).any()):
        raise ValueError("appliance_shares should be a non-empty non-negative vector.")
    if int(horizon) < 1:
        raise ValueError("horizon should be >= 1.")

    last_mains = history[:, -1:].clamp_min(0.0)
    return (
        last_mains.unsqueeze(1)
        * shares.view(1, -1, 1).to(device=history.device)
    ).expand(-1, -1, int(horizon)).clone()


def never_start_probability(num_windows: int, num_appliances: int) -> Tensor:
    """Probability output for the valid no-alert/zero-risk reference."""
    return torch.zeros(num_windows, num_appliances, dtype=torch.float32)


@dataclass
class CalendarWindowStartPrior:
    """Smoothed causal start-risk prior indexed by hour and weekend status."""

    smoothing: float = 20.0
    probabilities: Tensor | None = None
    counts: Tensor | None = None
    global_rate: Tensor | None = None

    @staticmethod
    def _bin_from_timestamp(value: Any) -> int:
        timestamp = pd.to_datetime(value, errors="coerce")
        if pd.isna(timestamp):
            raise ValueError(f"Unable to parse target timestamp: {value!r}")
        return int(timestamp.hour) * 2 + int(timestamp.dayofweek >= 5)

    def fit(self, train_dataset) -> "CalendarWindowStartPrior":
        """Fit solely from train-window labels and known target timestamps."""
        if len(train_dataset) == 0:
            raise ValueError("Cannot fit a time prior on an empty dataset.")
        first = train_dataset[0]
        first_start = torch.as_tensor(first["y_start"])
        num_appliances = int(first_start.shape[0])
        counts = torch.zeros(48, dtype=torch.float64)
        positives = torch.zeros(48, num_appliances, dtype=torch.float64)

        for index in range(len(train_dataset)):
            sample = train_dataset[index]
            bin_index = self._bin_from_timestamp(sample["timestamp_start"])
            target = torch.as_tensor(sample["y_start"], dtype=torch.float64)
            if target.dim() != 2 or target.size(0) != num_appliances:
                raise ValueError("y_start should have stable shape [A, H].")
            counts[bin_index] += 1.0
            positives[bin_index] += (target.amax(dim=-1) > 0.5).to(torch.float64)

        total_windows = counts.sum().clamp_min(1.0)
        global_rate = positives.sum(dim=0) / total_windows
        alpha = max(float(self.smoothing), 0.0)
        probabilities = (positives + alpha * global_rate.unsqueeze(0)) / (
            counts.unsqueeze(-1) + alpha
        ).clamp_min(1e-8)
        self.probabilities = probabilities.to(torch.float32)
        self.counts = counts.to(torch.float32)
        self.global_rate = global_rate.to(torch.float32)
        return self

    def predict(self, dataset) -> tuple[Tensor, Tensor]:
        """Return prior probabilities and true start labels for a split."""
        if self.probabilities is None:
            raise RuntimeError("Call fit(train_dataset) before predict(dataset).")
        probabilities: list[Tensor] = []
        targets: list[Tensor] = []
        for index in range(len(dataset)):
            sample = dataset[index]
            bin_index = self._bin_from_timestamp(sample["timestamp_start"])
            probabilities.append(self.probabilities[bin_index])
            targets.append(torch.as_tensor(sample["y_start"], dtype=torch.float32))
        return torch.stack(probabilities), torch.stack(targets)

    def metadata(self, appliance_names: Sequence[str]) -> dict[str, Any]:
        if self.probabilities is None or self.counts is None or self.global_rate is None:
            raise RuntimeError("Call fit before requesting metadata.")
        return {
            "type": "calendar_hour_weekend_prior",
            "smoothing": float(self.smoothing),
            "num_calendar_bins": 48,
            "nonempty_bins": int((self.counts > 0).sum().item()),
            "global_start_rate": {
                name: float(self.global_rate[index].item())
                for index, name in enumerate(appliance_names)
            },
        }


def calibrate_window_start_thresholds(
    probabilities: Tensor,
    true_start: Tensor,
    appliance_names: Sequence[str],
    num_thresholds: int = 80,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Fit per-device F1 thresholds on validation windows only."""
    probability = torch.as_tensor(probabilities, dtype=torch.float32).clamp(
        1e-5,
        1.0 - 1e-5,
    )
    target = torch.as_tensor(true_start, dtype=torch.float32)
    if probability.dim() != 2 or target.dim() != 3:
        raise ValueError("probabilities should be [N, A] and true_start [N, A, H].")
    if probability.shape != target.shape[:2]:
        raise ValueError("probabilities and true_start should share [N, A].")
    window_target = (target.amax(dim=-1, keepdim=True) > 0.5).to(torch.float32)
    return find_best_event_thresholds(
        event_logits=torch.logit(probability).unsqueeze(-1),
        true_event=window_target,
        appliance_names=list(appliance_names),
        num_thresholds=num_thresholds,
        min_threshold=0.01,
        max_threshold=0.99,
    )


def evaluate_window_start_probabilities(
    probabilities: Tensor,
    true_start: Tensor,
    appliance_names: Sequence[str],
    thresholds: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Evaluate calibrated window-start probabilities on one split."""
    return window_start_risk_metrics(
        start_prob=probabilities,
        true_start=true_start,
        appliance_names=list(appliance_names),
        threshold=thresholds,
    )
