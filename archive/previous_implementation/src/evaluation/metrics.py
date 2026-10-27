from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


DEFAULT_APPLIANCE_NAMES = [
    "air1",
    "refrigerator1",
    "dishwasher1",
    "microwave1",
]


def _to_cpu_tensor(x: Tensor | np.ndarray | list | tuple) -> Tensor:
    if isinstance(x, Tensor):
        return x.detach().cpu().float()

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()

    return torch.tensor(x, dtype=torch.float32)


def _as_appliance_vector(
    values: Optional[Tensor | np.ndarray | list | tuple | dict[str, float]],
    appliance_names: list[str],
    default_value: float,
) -> Tensor:
    """
    Convert thresholds/scales into shape [A].
    """
    if values is None:
        return torch.full((len(appliance_names),), float(default_value))

    if isinstance(values, dict):
        return torch.tensor(
            [float(values[name]) for name in appliance_names],
            dtype=torch.float32,
        )

    values = _to_cpu_tensor(values).view(-1)

    if values.numel() != len(appliance_names):
        raise ValueError(
            f"Expected {len(appliance_names)} values, got {values.numel()}."
        )

    return values.float()


def _safe_div(numerator: Tensor, denominator: Tensor, eps: float = 1e-8) -> Tensor:
    return numerator / denominator.clamp_min(eps)


def _masked_values(
    value: Tensor,
    mask: Optional[Tensor],
) -> Tensor:
    if mask is None:
        return value.reshape(-1)

    mask = mask.to(dtype=torch.bool)
    return value[mask]


def _masked_sum(
    value: Tensor,
    mask: Optional[Tensor],
) -> Tensor:
    if mask is None:
        return value.sum()

    return value[mask.to(dtype=torch.bool)].sum()


def _masked_count(
    value: Tensor,
    mask: Optional[Tensor],
) -> Tensor:
    if mask is None:
        return torch.tensor(value.numel(), dtype=torch.float32)

    return mask.to(dtype=torch.float32).sum()


def _mean_or_nan(value: Tensor) -> float:
    if value.numel() == 0:
        return float("nan")
    return float(value.mean().item())


def _rmse_or_nan(error: Tensor) -> float:
    if error.numel() == 0:
        return float("nan")
    return float(torch.sqrt((error ** 2).mean()).item())


def _mae_or_nan(error: Tensor) -> float:
    if error.numel() == 0:
        return float("nan")
    return float(torch.abs(error).mean().item())


def _binary_metrics_from_counts(tp: Tensor, fp: Tensor, fn: Tensor) -> dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    return {
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "f1": float(f1.item()),
        "tp": float(tp.item()),
        "fp": float(fp.item()),
        "fn": float(fn.item()),
    }


def derive_event_from_state(
    state: Tensor,
    previous_state: Optional[Tensor] = None,
) -> Tensor:
    """
    Derive event labels from state sequence.

    state:
        [N, A, H], 0/1

    return:
        event: [N, A, H]

    previous_state:
        Optional state immediately before the forecasting horizon, with shape
        [N, A] or [N, A, 1]. If omitted, OFF is used as the prior state for
        backward compatibility.

    Note:
        event[..., 0] is state[..., 0] XOR previous_state.
        For t > 0, event_t = state_t XOR state_{t-1}.
    """
    if state.dim() != 3:
        raise ValueError(f"state should be [N, A, H], got {state.shape}")

    state = state.to(dtype=torch.bool)
    event = torch.zeros_like(state)

    if previous_state is None:
        previous = torch.zeros_like(state[..., 0])
    else:
        previous = previous_state.to(device=state.device, dtype=torch.bool)
        if previous.dim() == 3 and previous.size(-1) == 1:
            previous = previous.squeeze(-1)
        if previous.shape != state[..., 0].shape:
            raise ValueError(
                "previous_state should have shape [N, A] or [N, A, 1], "
                f"got {previous_state.shape}."
            )

    event[..., 0] = state[..., 0] ^ previous
    event[..., 1:] = state[..., 1:] ^ state[..., :-1]

    return event.to(dtype=torch.float32)


def postprocess_state_sequence(
    state: Tensor | np.ndarray,
    min_on_duration: int = 1,
    min_off_duration: int = 1,
) -> Tensor:
    """
    Clean binary appliance state sequences.

    Short OFF gaps inside an ON run are filled first, then ON runs shorter than
    min_on_duration are removed.
    """
    state_t = (_to_cpu_tensor(state) > 0.5).clone()

    min_on_duration = max(1, int(min_on_duration))
    min_off_duration = max(1, int(min_off_duration))

    if min_on_duration <= 1 and min_off_duration <= 1:
        return state_t.float()

    if state_t.dim() != 3:
        raise ValueError(f"state should be [N, A, H], got {state_t.shape}")

    processed = state_t.clone()
    num_samples, num_appliances, horizon = processed.shape

    for sample_idx in range(num_samples):
        for app_idx in range(num_appliances):
            seq = processed[sample_idx, app_idx]

            if min_off_duration > 1:
                start = 0
                while start < horizon:
                    value = bool(seq[start].item())
                    end = start + 1
                    while end < horizon and bool(seq[end].item()) == value:
                        end += 1

                    run_len = end - start
                    has_left_on = start > 0 and bool(seq[start - 1].item())
                    has_right_on = end < horizon and bool(seq[end].item())

                    if (
                        not value
                        and run_len < min_off_duration
                        and has_left_on
                        and has_right_on
                    ):
                        seq[start:end] = True

                    start = end

            if min_on_duration > 1:
                start = 0
                while start < horizon:
                    value = bool(seq[start].item())
                    end = start + 1
                    while end < horizon and bool(seq[end].item()) == value:
                        end += 1

                    if value and (end - start) < min_on_duration:
                        seq[start:end] = False

                    start = end

    return processed.float()


def find_best_state_thresholds(
    pred_power: Tensor | np.ndarray,
    true_state: Tensor | np.ndarray,
    true_event: Optional[Tensor | np.ndarray] = None,
    previous_state: Optional[Tensor | np.ndarray] = None,
    previous_power: Optional[Tensor | np.ndarray] = None,
    appliance_names: Optional[list[str]] = None,
    mask: Optional[Tensor | np.ndarray] = None,
    num_thresholds: int = 80,
    min_threshold: float = 0.0,
    max_threshold: Optional[float] = None,
    min_on_duration: int = 1,
    min_off_duration: int = 1,
    objective: str = "state_f1",
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Search one power threshold per appliance on validation predictions.
    """
    power = _to_cpu_tensor(pred_power)
    target = _to_cpu_tensor(true_state)
    event_target = None if true_event is None else _to_cpu_tensor(true_event)
    previous = None if previous_state is None else _to_cpu_tensor(previous_state)
    previous_power_t = (
        None if previous_power is None else _to_cpu_tensor(previous_power)
    )

    if power.shape != target.shape:
        raise ValueError(
            f"pred_power and true_state shape mismatch: {power.shape}, {target.shape}"
        )

    if event_target is not None and event_target.shape != target.shape:
        raise ValueError(
            f"true_event and true_state shape mismatch: "
            f"{event_target.shape}, {target.shape}"
        )

    if previous is not None:
        if previous.dim() == 3 and previous.size(-1) == 1:
            previous = previous.squeeze(-1)
        if previous.shape != target[..., 0].shape:
            raise ValueError(
                "previous_state should have shape [N, A] or [N, A, 1], "
                f"got {previous_state.shape}."
            )

    if previous_power_t is not None:
        if previous_power_t.dim() == 2:
            previous_power_t = previous_power_t.unsqueeze(-1)
        if previous_power_t.shape != target[..., :1].shape:
            raise ValueError(
                "previous_power should have shape [N, A] or [N, A, 1], "
                f"got {previous_power.shape}."
            )

    if power.dim() != 3:
        raise ValueError(f"pred_power should be [N, A, H], got {power.shape}")

    if objective not in {"state_f1", "event_f1", "mean_state_event_f1"}:
        raise ValueError(
            "objective should be one of: state_f1, event_f1, mean_state_event_f1."
        )

    _, num_appliances, _ = power.shape

    if appliance_names is None:
        appliance_names = DEFAULT_APPLIANCE_NAMES[:num_appliances]

    mask_t = None if mask is None else (_to_cpu_tensor(mask) > 0.5)

    thresholds: dict[str, float] = {}
    summary: dict[str, dict[str, float]] = {}
    num_thresholds = max(2, int(num_thresholds))

    for app_idx, app_name in enumerate(appliance_names):
        power_i = power[:, app_idx : app_idx + 1, :]
        target_i = target[:, app_idx : app_idx + 1, :]
        event_target_i = (
            None
            if event_target is None
            else event_target[:, app_idx : app_idx + 1, :]
        )
        mask_i = None if mask_t is None else mask_t[:, app_idx : app_idx + 1, :]
        previous_i = (
            None
            if previous is None
            else previous[:, app_idx : app_idx + 1]
        )
        previous_power_i = (
            None
            if previous_power_t is None
            else previous_power_t[:, app_idx : app_idx + 1, :]
        )

        if max_threshold is None:
            valid_power = power_i if mask_i is None else power_i[mask_i]
            if valid_power.numel() == 0:
                upper = float(min_threshold + 1.0)
            else:
                upper = float(torch.quantile(valid_power, 0.995).item())
                upper = max(upper, float(min_threshold) + 1e-4)
        else:
            upper = float(max_threshold)

        candidates = torch.linspace(float(min_threshold), upper, steps=num_thresholds)

        best_threshold = float(candidates[0].item())
        best_metrics = {
            "StatePrecision": 0.0,
            "StateRecall": 0.0,
            "StateF1": -1.0,
            "EventPrecision": 0.0,
            "EventRecall": 0.0,
            "EventF1": -1.0,
            "Score": -1.0,
        }

        for threshold in candidates:
            pred_i = (power_i >= threshold).float()
            pred_i = postprocess_state_sequence(
                pred_i,
                min_on_duration=min_on_duration,
                min_off_duration=min_off_duration,
            )
            metrics_i = state_classification_metrics(
                pred_state=pred_i,
                true_state=target_i,
                appliance_names=[app_name],
                mask=mask_i,
            )[app_name]

            event_metrics_i = {
                "EventPrecision": 0.0,
                "EventRecall": 0.0,
                "EventF1": 0.0,
            }
            if event_target_i is not None:
                event_previous_i = previous_i
                if previous_power_i is not None:
                    event_previous_i = (previous_power_i >= threshold).float()

                event_metrics_i = event_classification_metrics(
                    pred_event=derive_event_from_state(
                        pred_i,
                        previous_state=event_previous_i,
                    ),
                    true_event=event_target_i,
                    appliance_names=[app_name],
                    mask=mask_i,
                )[app_name]

            if objective == "state_f1":
                score = metrics_i["StateF1"]
            elif objective == "event_f1":
                score = event_metrics_i["EventF1"]
            else:
                score = 0.5 * (
                    metrics_i["StateF1"] + event_metrics_i["EventF1"]
                )

            if score > best_metrics["Score"]:
                best_threshold = float(threshold.item())
                best_metrics = {
                    **metrics_i,
                    **event_metrics_i,
                    "Score": score,
                }

        thresholds[app_name] = best_threshold
        summary[app_name] = {
            "threshold": best_threshold,
            "StatePrecision": best_metrics["StatePrecision"],
            "StateRecall": best_metrics["StateRecall"],
            "StateF1": best_metrics["StateF1"],
            "EventPrecision": best_metrics["EventPrecision"],
            "EventRecall": best_metrics["EventRecall"],
            "EventF1": best_metrics["EventF1"],
            "Score": best_metrics["Score"],
        }

    return thresholds, summary


def find_best_event_thresholds(
    event_logits: Tensor | np.ndarray,
    true_event: Tensor | np.ndarray,
    appliance_names: Optional[list[str]] = None,
    mask: Optional[Tensor | np.ndarray] = None,
    num_thresholds: int = 80,
    min_threshold: float = 0.05,
    max_threshold: float = 0.99,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Calibrate one event-probability threshold per appliance on validation data.

    Event probabilities are highly imbalanced, so their operating threshold
    should not be tied to the 0.5 state threshold.
    """
    logits = _to_cpu_tensor(event_logits)
    target = _to_cpu_tensor(true_event)

    if logits.shape != target.shape or logits.dim() != 3:
        raise ValueError(
            "event_logits and true_event should share shape [N, A, H], "
            f"got {logits.shape} and {target.shape}."
        )

    _, num_appliances, _ = logits.shape
    if appliance_names is None:
        appliance_names = DEFAULT_APPLIANCE_NAMES[:num_appliances]
    if len(appliance_names) != num_appliances:
        raise ValueError("len(appliance_names) should match appliance dimension.")

    mask_t = None if mask is None else (_to_cpu_tensor(mask) > 0.5)
    probs = torch.sigmoid(logits)
    lower = min(max(float(min_threshold), 0.0), 1.0)
    upper = min(max(float(max_threshold), lower + 1e-4), 1.0)
    candidates = torch.linspace(lower, upper, steps=max(2, int(num_thresholds)))

    thresholds: dict[str, float] = {}
    summary: dict[str, dict[str, float]] = {}

    for app_idx, app_name in enumerate(appliance_names):
        prob_i = probs[:, app_idx : app_idx + 1, :]
        target_i = target[:, app_idx : app_idx + 1, :]
        mask_i = None if mask_t is None else mask_t[:, app_idx : app_idx + 1, :]

        best_threshold = float(candidates[0].item())
        best_metrics = {
            "EventPrecision": 0.0,
            "EventRecall": 0.0,
            "EventF1": -1.0,
        }

        for threshold in candidates:
            metrics_i = event_classification_metrics(
                pred_event=(prob_i >= threshold).float(),
                true_event=target_i,
                appliance_names=[app_name],
                mask=mask_i,
            )[app_name]

            if metrics_i["EventF1"] > best_metrics["EventF1"]:
                best_threshold = float(threshold.item())
                best_metrics = metrics_i

        thresholds[app_name] = best_threshold
        summary[app_name] = {
            "threshold": best_threshold,
            "EventPrecision": best_metrics["EventPrecision"],
            "EventRecall": best_metrics["EventRecall"],
            "EventF1": best_metrics["EventF1"],
        }

    return thresholds, summary


def find_best_hierarchical_start_thresholds(
    window_start_prob: Tensor | np.ndarray,
    conditional_start_prob: Tensor | np.ndarray,
    true_start: Tensor | np.ndarray,
    appliance_names: Optional[list[str]] = None,
    mask: Optional[Tensor | np.ndarray] = None,
    num_thresholds: int = 80,
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
    tolerance: int = 0,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Calibrate the gate used by the hierarchical one-start decision.

    ``conditional_start_prob`` identifies *where* a start is most likely in
    a horizon while ``window_start_prob`` decides whether the horizon contains
    any start at all.  Thresholding their product minute-by-minute would allow
    multiple alerts in one window and would therefore evaluate a different
    decision rule from the deployed hierarchical forecaster.  This helper
    instead emits at most one start per appliance/window: the conditional
    argmax, gated by the calibrated window probability.

    If ``tolerance`` is positive, selection is performed using one-to-one
    tolerance-aware EventF1.  Exact event metrics are retained in the summary
    so calibration provenance remains explicit.
    """
    window_prob = _to_cpu_tensor(window_start_prob).float()
    conditional_prob = _to_cpu_tensor(conditional_start_prob).float()
    target = _to_cpu_tensor(true_start).float()

    if window_prob.dim() == 3 and window_prob.size(-1) == 1:
        window_prob = window_prob.squeeze(-1)
    if window_prob.dim() != 2:
        raise ValueError(
            "window_start_prob should have shape [N, A] or [N, A, 1], "
            f"got {window_prob.shape}."
        )
    if conditional_prob.shape != target.shape or conditional_prob.dim() != 3:
        raise ValueError(
            "conditional_start_prob and true_start should share shape [N, A, H], "
            f"got {conditional_prob.shape} and {target.shape}."
        )
    if window_prob.shape != conditional_prob.shape[:2]:
        raise ValueError(
            "window_start_prob should share [N, A] with conditional_start_prob; "
            f"got {window_prob.shape} and {conditional_prob.shape}."
        )

    _, num_appliances, _ = conditional_prob.shape
    if appliance_names is None:
        appliance_names = DEFAULT_APPLIANCE_NAMES[:num_appliances]
    if len(appliance_names) != num_appliances:
        raise ValueError("len(appliance_names) should match appliance dimension.")

    mask_t = None if mask is None else (_to_cpu_tensor(mask) > 0.5)
    if mask_t is not None and mask_t.shape != target.shape:
        raise ValueError("mask should share shape [N, A, H] with true_start.")

    # Do not select an invalid padded minute as the one permissible alert.
    conditional_for_argmax = conditional_prob
    if mask_t is not None:
        conditional_for_argmax = conditional_prob.masked_fill(~mask_t, -float("inf"))
    start_index = conditional_for_argmax.argmax(dim=-1, keepdim=True)
    one_start = torch.zeros_like(conditional_prob)
    one_start.scatter_(2, start_index, 1.0)
    if mask_t is not None:
        one_start = one_start * mask_t.to(dtype=one_start.dtype)

    lower = min(max(float(min_threshold), 0.0), 1.0)
    upper = min(max(float(max_threshold), lower + 1e-4), 1.0)
    candidates = torch.linspace(lower, upper, steps=max(2, int(num_thresholds)))
    thresholds: dict[str, float] = {}
    summary: dict[str, dict[str, float]] = {}

    for app_idx, app_name in enumerate(appliance_names):
        pred_template = one_start[:, app_idx : app_idx + 1, :]
        prob_i = window_prob[:, app_idx : app_idx + 1]
        target_i = target[:, app_idx : app_idx + 1, :]
        mask_i = None if mask_t is None else mask_t[:, app_idx : app_idx + 1, :]

        best_threshold = float(candidates[0].item())
        best_score = -1.0
        best_exact: dict[str, float] = {}
        best_tolerant: dict[str, float] = {}
        for threshold in candidates:
            pred_i = pred_template * (prob_i >= threshold).to(
                dtype=pred_template.dtype
            ).unsqueeze(-1)
            exact = event_classification_metrics(
                pred_event=pred_i,
                true_event=target_i,
                appliance_names=[app_name],
                mask=mask_i,
            )[app_name]
            tolerant = event_tolerance_metrics(
                pred_event=pred_i,
                true_event=target_i,
                tolerance=tolerance,
                appliance_names=[app_name],
                mask=mask_i,
            )[app_name]
            score = (
                tolerant["EventF1"] if int(tolerance) > 0 else exact["EventF1"]
            )
            if score > best_score:
                best_threshold = float(threshold.item())
                best_score = float(score)
                best_exact = exact
                best_tolerant = tolerant

        thresholds[app_name] = best_threshold
        summary[app_name] = {
            "threshold": best_threshold,
            "selection_metric": (
                "tolerant_event_f1" if int(tolerance) > 0 else "event_f1"
            ),
            "selection_tolerance_minutes": int(tolerance),
            "EventPrecision": best_exact["EventPrecision"],
            "EventRecall": best_exact["EventRecall"],
            "EventF1": best_exact["EventF1"],
            "TolerantEventPrecision": best_tolerant["EventPrecision"],
            "TolerantEventRecall": best_tolerant["EventRecall"],
            "TolerantEventF1": best_tolerant["EventF1"],
        }

    return thresholds, summary


def per_appliance_regression_metrics(
    pred_power: Tensor | np.ndarray,
    target_power: Tensor | np.ndarray,
    appliance_names: Optional[list[str]] = None,
    mask: Optional[Tensor | np.ndarray] = None,
) -> dict[str, dict[str, float]]:
    """
    Compute per-appliance MAE/RMSE/BIAS.

    Args:
        pred_power:
            [N, A, H], original power unit, e.g. kW.

        target_power:
            [N, A, H], original power unit.

        mask:
            [N, A, H], optional.

    Returns:
        {
            "air1": {"MAE": ..., "RMSE": ..., "Bias": ...},
            ...
            "macro_avg": {...}
        }
    """
    pred = _to_cpu_tensor(pred_power)
    target = _to_cpu_tensor(target_power)

    if pred.shape != target.shape:
        raise ValueError(f"pred and target shape mismatch: {pred.shape}, {target.shape}")

    if pred.dim() != 3:
        raise ValueError(f"Expected shape [N, A, H], got {pred.shape}")

    _, num_appliances, _ = pred.shape

    if appliance_names is None:
        appliance_names = DEFAULT_APPLIANCE_NAMES[:num_appliances]

    if len(appliance_names) != num_appliances:
        raise ValueError("len(appliance_names) should match appliance dimension.")

    mask_t = None if mask is None else _to_cpu_tensor(mask)

    result: dict[str, dict[str, float]] = {}
    mae_list = []
    rmse_list = []
    bias_list = []

    for app_idx, app_name in enumerate(appliance_names):
        err = pred[:, app_idx, :] - target[:, app_idx, :]
        app_mask = None if mask_t is None else mask_t[:, app_idx, :]

        err_valid = _masked_values(err, app_mask)

        mae = _mae_or_nan(err_valid)
        rmse = _rmse_or_nan(err_valid)
        bias = _mean_or_nan(err_valid)

        target_valid = _masked_values(target[:, app_idx, :], app_mask)
        pred_valid = _masked_values(pred[:, app_idx, :], app_mask)

        sae = float(
            torch.abs(pred_valid.sum() - target_valid.sum())
            / target_valid.sum().abs().clamp_min(1e-8)
        )

        result[app_name] = {
            "MAE": mae,
            "RMSE": rmse,
            "Bias": bias,
            "SAE": sae,
            "TargetMean": _mean_or_nan(target_valid),
            "PredMean": _mean_or_nan(pred_valid),
        }

        mae_list.append(mae)
        rmse_list.append(rmse)
        bias_list.append(bias)

    result["macro_avg"] = {
        "MAE": float(np.nanmean(mae_list)),
        "RMSE": float(np.nanmean(rmse_list)),
        "Bias": float(np.nanmean(bias_list)),
    }

    return result


def state_classification_metrics(
    pred_state: Tensor | np.ndarray,
    true_state: Tensor | np.ndarray,
    appliance_names: Optional[list[str]] = None,
    mask: Optional[Tensor | np.ndarray] = None,
) -> dict[str, dict[str, float]]:
    """
    Compute per-appliance state Precision/Recall/F1.

    pred_state:
        [N, A, H], 0/1

    true_state:
        [N, A, H], 0/1
    """
    pred = _to_cpu_tensor(pred_state) > 0.5
    target = _to_cpu_tensor(true_state) > 0.5

    if pred.shape != target.shape:
        raise ValueError(f"pred and target shape mismatch: {pred.shape}, {target.shape}")

    _, num_appliances, _ = pred.shape

    if appliance_names is None:
        appliance_names = DEFAULT_APPLIANCE_NAMES[:num_appliances]

    mask_t = None if mask is None else (_to_cpu_tensor(mask) > 0.5)

    result: dict[str, dict[str, float]] = {}
    f1_list = []
    precision_list = []
    recall_list = []

    total_tp = torch.tensor(0.0)
    total_fp = torch.tensor(0.0)
    total_fn = torch.tensor(0.0)

    for app_idx, app_name in enumerate(appliance_names):
        p = pred[:, app_idx, :]
        t = target[:, app_idx, :]
        m = None if mask_t is None else mask_t[:, app_idx, :]

        if m is not None:
            p = p[m]
            t = t[m]

        tp = ((p == 1) & (t == 1)).sum().float()
        fp = ((p == 1) & (t == 0)).sum().float()
        fn = ((p == 0) & (t == 1)).sum().float()

        metrics = _binary_metrics_from_counts(tp, fp, fn)
        result[app_name] = {
            "StatePrecision": metrics["precision"],
            "StateRecall": metrics["recall"],
            "StateF1": metrics["f1"],
            "StateTP": metrics["tp"],
            "StateFP": metrics["fp"],
            "StateFN": metrics["fn"],
        }

        precision_list.append(metrics["precision"])
        recall_list.append(metrics["recall"])
        f1_list.append(metrics["f1"])

        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro = _binary_metrics_from_counts(total_tp, total_fp, total_fn)

    result["macro_avg"] = {
        "StatePrecision": float(np.nanmean(precision_list)),
        "StateRecall": float(np.nanmean(recall_list)),
        "StateF1": float(np.nanmean(f1_list)),
    }

    result["micro_avg"] = {
        "StatePrecision": micro["precision"],
        "StateRecall": micro["recall"],
        "StateF1": micro["f1"],
    }

    return result


def event_classification_metrics(
    pred_event: Tensor | np.ndarray,
    true_event: Tensor | np.ndarray,
    appliance_names: Optional[list[str]] = None,
    mask: Optional[Tensor | np.ndarray] = None,
) -> dict[str, dict[str, float]]:
    """
    Compute per-appliance event Precision/Recall/F1.

    pred_event:
        [N, A, H], 0/1

    true_event:
        [N, A, H], 0/1
    """
    pred = _to_cpu_tensor(pred_event) > 0.5
    target = _to_cpu_tensor(true_event) > 0.5

    if pred.shape != target.shape:
        raise ValueError(f"pred and target shape mismatch: {pred.shape}, {target.shape}")

    _, num_appliances, _ = pred.shape

    if appliance_names is None:
        appliance_names = DEFAULT_APPLIANCE_NAMES[:num_appliances]

    mask_t = None if mask is None else (_to_cpu_tensor(mask) > 0.5)

    result: dict[str, dict[str, float]] = {}
    f1_list = []
    precision_list = []
    recall_list = []

    total_tp = torch.tensor(0.0)
    total_fp = torch.tensor(0.0)
    total_fn = torch.tensor(0.0)

    for app_idx, app_name in enumerate(appliance_names):
        p = pred[:, app_idx, :]
        t = target[:, app_idx, :]
        m = None if mask_t is None else mask_t[:, app_idx, :]

        if m is not None:
            p = p[m]
            t = t[m]

        tp = ((p == 1) & (t == 1)).sum().float()
        fp = ((p == 1) & (t == 0)).sum().float()
        fn = ((p == 0) & (t == 1)).sum().float()

        metrics = _binary_metrics_from_counts(tp, fp, fn)

        result[app_name] = {
            "EventPrecision": metrics["precision"],
            "EventRecall": metrics["recall"],
            "EventF1": metrics["f1"],
            "EventTP": metrics["tp"],
            "EventFP": metrics["fp"],
            "EventFN": metrics["fn"],
        }

        precision_list.append(metrics["precision"])
        recall_list.append(metrics["recall"])
        f1_list.append(metrics["f1"])

        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro = _binary_metrics_from_counts(total_tp, total_fp, total_fn)

    result["macro_avg"] = {
        "EventPrecision": float(np.nanmean(precision_list)),
        "EventRecall": float(np.nanmean(recall_list)),
        "EventF1": float(np.nanmean(f1_list)),
    }

    result["micro_avg"] = {
        "EventPrecision": micro["precision"],
        "EventRecall": micro["recall"],
        "EventF1": micro["f1"],
    }

    return result


def event_tolerance_metrics(
    pred_event: Tensor | np.ndarray,
    true_event: Tensor | np.ndarray,
    tolerance: int,
    appliance_names: Optional[list[str]] = None,
    mask: Optional[Tensor | np.ndarray] = None,
) -> dict[str, dict[str, float]]:
    """
    Event F1 with one-to-one matching inside a timing tolerance.

    A predicted event can match at most one true event and vice versa, avoiding
    inflated scores from a burst of predictions around one true transition.
    """
    pred = _to_cpu_tensor(pred_event) > 0.5
    target = _to_cpu_tensor(true_event) > 0.5

    if pred.shape != target.shape or pred.dim() != 3:
        raise ValueError(
            "pred_event and true_event should share shape [N, A, H], "
            f"got {pred.shape} and {target.shape}."
        )

    tolerance = max(0, int(tolerance))
    _, num_appliances, _ = pred.shape
    if appliance_names is None:
        appliance_names = DEFAULT_APPLIANCE_NAMES[:num_appliances]
    mask_t = None if mask is None else (_to_cpu_tensor(mask) > 0.5)

    result: dict[str, dict[str, float]] = {}
    precision_list = []
    recall_list = []
    f1_list = []
    total_tp = torch.tensor(0.0)
    total_fp = torch.tensor(0.0)
    total_fn = torch.tensor(0.0)

    for app_idx, app_name in enumerate(appliance_names):
        tp = 0
        fp = 0
        fn = 0

        for sample_idx in range(pred.size(0)):
            valid = (
                torch.ones(pred.size(-1), dtype=torch.bool)
                if mask_t is None
                else mask_t[sample_idx, app_idx]
            )
            pred_idx = torch.nonzero(
                pred[sample_idx, app_idx] & valid,
                as_tuple=False,
            ).view(-1).tolist()
            target_idx = torch.nonzero(
                target[sample_idx, app_idx] & valid,
                as_tuple=False,
            ).view(-1).tolist()

            unmatched_target = set(target_idx)
            for pred_time in pred_idx:
                candidates = [
                    target_time
                    for target_time in unmatched_target
                    if abs(target_time - pred_time) <= tolerance
                ]
                if not candidates:
                    fp += 1
                    continue

                matched_time = min(
                    candidates,
                    key=lambda target_time: (abs(target_time - pred_time), target_time),
                )
                unmatched_target.remove(matched_time)
                tp += 1

            fn += len(unmatched_target)

        metrics = _binary_metrics_from_counts(
            torch.tensor(float(tp)),
            torch.tensor(float(fp)),
            torch.tensor(float(fn)),
        )
        result[app_name] = {
            "EventPrecision": metrics["precision"],
            "EventRecall": metrics["recall"],
            "EventF1": metrics["f1"],
            "EventTP": metrics["tp"],
            "EventFP": metrics["fp"],
            "EventFN": metrics["fn"],
        }
        precision_list.append(metrics["precision"])
        recall_list.append(metrics["recall"])
        f1_list.append(metrics["f1"])
        total_tp += float(tp)
        total_fp += float(fp)
        total_fn += float(fn)

    micro = _binary_metrics_from_counts(total_tp, total_fp, total_fn)
    result["macro_avg"] = {
        "EventPrecision": float(np.nanmean(precision_list)),
        "EventRecall": float(np.nanmean(recall_list)),
        "EventF1": float(np.nanmean(f1_list)),
    }
    result["micro_avg"] = {
        "EventPrecision": micro["precision"],
        "EventRecall": micro["recall"],
        "EventF1": micro["f1"],
    }
    return result


def window_start_risk_metrics(
    start_prob: Tensor | np.ndarray,
    true_start: Tensor | np.ndarray,
    appliance_names: Optional[list[str]] = None,
    threshold: Tensor | np.ndarray | list | tuple | dict[str, float] | float = 0.5,
) -> dict[str, dict[str, float]]:
    """Thresholded F1 plus AP/Brier for any start inside the horizon."""
    prob = _to_cpu_tensor(start_prob)
    target_minute = _to_cpu_tensor(true_start)
    if prob.dim() != 2 or target_minute.dim() != 3:
        raise ValueError("start_prob should be [N, A] and true_start [N, A, H].")
    target = (target_minute.amax(dim=-1) > 0.5).float()
    if prob.shape != target.shape:
        raise ValueError(f"window start shape mismatch: {prob.shape} vs {target.shape}")

    _, num_appliances = prob.shape
    if appliance_names is None:
        appliance_names = DEFAULT_APPLIANCE_NAMES[:num_appliances]
    thresholds = _as_appliance_vector(
        threshold,
        appliance_names,
        default_value=0.5,
    )

    result: dict[str, dict[str, float]] = {}
    for app_idx, app_name in enumerate(appliance_names):
        p = prob[:, app_idx].clamp(0.0, 1.0)
        t = target[:, app_idx]
        pred = p >= float(thresholds[app_idx].item())
        truth = t > 0.5
        tp = (pred & truth).sum().float()
        fp = (pred & ~truth).sum().float()
        fn = (~pred & truth).sum().float()
        cls = _binary_metrics_from_counts(tp, fp, fn)

        order = torch.argsort(p, descending=True)
        sorted_truth = truth[order].float()
        positives = sorted_truth.sum()
        if positives > 0:
            precision_curve = sorted_truth.cumsum(0) / torch.arange(
                1,
                sorted_truth.numel() + 1,
                dtype=torch.float32,
            )
            average_precision = float(
                (precision_curve * sorted_truth).sum().div(positives).item()
            )
        else:
            average_precision = float("nan")

        result[app_name] = {
            "WindowStartPrecision": cls["precision"],
            "WindowStartRecall": cls["recall"],
            "WindowStartF1": cls["f1"],
            "WindowStartAP": average_precision,
            "WindowStartBrier": float(((p - t) ** 2).mean().item()),
            "WindowStartTargetRate": float(t.mean().item()),
            "WindowStartPredMean": float(p.mean().item()),
        }
    return result


def bucket_start_risk_metrics(
    bucket_prob: Tensor | np.ndarray,
    true_start: Tensor | np.ndarray,
    bucket_size: int,
    appliance_names: Optional[list[str]] = None,
) -> dict[str, dict[str, float]]:
    """Calibration, ranking and top-bucket accuracy for coarse start risk."""
    prob = _to_cpu_tensor(bucket_prob)
    target_minute = _to_cpu_tensor(true_start)
    if prob.dim() != 3 or target_minute.dim() != 3:
        raise ValueError("bucket_prob and true_start should be rank-3 tensors.")
    num_samples, num_appliances, num_buckets = prob.shape
    inferred_bucket_size = (
        target_minute.size(-1) + num_buckets - 1
    ) // num_buckets
    bucket_size = inferred_bucket_size
    padded_horizon = num_buckets * bucket_size
    pad = padded_horizon - target_minute.size(-1)
    target_padded = F.pad(target_minute, (0, pad)) if pad > 0 else target_minute
    target = (
        target_padded[..., :padded_horizon]
        .view(num_samples, num_appliances, num_buckets, bucket_size)
        .amax(dim=-1)
        > 0.5
    ).float()
    if appliance_names is None:
        appliance_names = DEFAULT_APPLIANCE_NAMES[:num_appliances]

    result: dict[str, dict[str, float]] = {}
    for app_idx, app_name in enumerate(appliance_names):
        p = prob[:, app_idx].clamp(0.0, 1.0)
        t = target[:, app_idx]
        p_flat = p.flatten()
        truth_flat = t.flatten() > 0.5
        order = torch.argsort(p_flat, descending=True)
        sorted_truth = truth_flat[order].float()
        positives = sorted_truth.sum()
        if positives > 0:
            precision_curve = sorted_truth.cumsum(0) / torch.arange(
                1, sorted_truth.numel() + 1, dtype=torch.float32
            )
            average_precision = float(
                (precision_curve * sorted_truth).sum().div(positives).item()
            )
        else:
            average_precision = float("nan")
        positive_windows = t.amax(dim=-1) > 0.5
        if positive_windows.any():
            top_bucket = p.argmax(dim=-1)
            top_accuracy = float(
                t[positive_windows]
                .gather(1, top_bucket[positive_windows].unsqueeze(-1))
                .float()
                .mean()
                .item()
            )
        else:
            top_accuracy = float("nan")
        result[app_name] = {
            "BucketStartAP": average_precision,
            "BucketStartBrier": float(((p - t) ** 2).mean().item()),
            "BucketStartTargetRate": float(t.mean().item()),
            "BucketStartPredMean": float(p.mean().item()),
            "TopBucketAccuracy": top_accuracy,
        }
    return result


def event_bucket_metrics(
    pred_event: Tensor | np.ndarray,
    true_event: Tensor | np.ndarray,
    bucket_size: int,
    appliance_names: Optional[list[str]] = None,
    mask: Optional[Tensor | np.ndarray] = None,
) -> dict[str, dict[str, float]]:
    """Event F1 after reducing each horizon into fixed-width time buckets."""
    pred = _to_cpu_tensor(pred_event) > 0.5
    target = _to_cpu_tensor(true_event) > 0.5

    if pred.shape != target.shape or pred.dim() != 3:
        raise ValueError(
            "pred_event and true_event should share shape [N, A, H], "
            f"got {pred.shape} and {target.shape}."
        )

    bucket_size = max(1, int(bucket_size))
    num_samples, num_appliances, horizon = pred.shape
    if appliance_names is None:
        appliance_names = DEFAULT_APPLIANCE_NAMES[:num_appliances]
    mask_t = None if mask is None else (_to_cpu_tensor(mask) > 0.5)

    result: dict[str, dict[str, float]] = {}
    precision_list = []
    recall_list = []
    f1_list = []
    total_tp = torch.tensor(0.0)
    total_fp = torch.tensor(0.0)
    total_fn = torch.tensor(0.0)

    for app_idx, app_name in enumerate(appliance_names):
        tp = 0
        fp = 0
        fn = 0

        for sample_idx in range(num_samples):
            for start in range(0, horizon, bucket_size):
                end = min(start + bucket_size, horizon)
                valid = (
                    torch.ones(end - start, dtype=torch.bool)
                    if mask_t is None
                    else mask_t[sample_idx, app_idx, start:end]
                )
                if not valid.any():
                    continue

                pred_bucket = bool(
                    (pred[sample_idx, app_idx, start:end] & valid).any()
                )
                target_bucket = bool(
                    (target[sample_idx, app_idx, start:end] & valid).any()
                )
                tp += int(pred_bucket and target_bucket)
                fp += int(pred_bucket and not target_bucket)
                fn += int(target_bucket and not pred_bucket)

        metrics = _binary_metrics_from_counts(
            torch.tensor(float(tp)),
            torch.tensor(float(fp)),
            torch.tensor(float(fn)),
        )
        result[app_name] = {
            "EventPrecision": metrics["precision"],
            "EventRecall": metrics["recall"],
            "EventF1": metrics["f1"],
            "EventTP": metrics["tp"],
            "EventFP": metrics["fp"],
            "EventFN": metrics["fn"],
        }
        precision_list.append(metrics["precision"])
        recall_list.append(metrics["recall"])
        f1_list.append(metrics["f1"])
        total_tp += float(tp)
        total_fp += float(fp)
        total_fn += float(fn)

    micro = _binary_metrics_from_counts(total_tp, total_fp, total_fn)
    result["macro_avg"] = {
        "EventPrecision": float(np.nanmean(precision_list)),
        "EventRecall": float(np.nanmean(recall_list)),
        "EventF1": float(np.nanmean(f1_list)),
    }
    result["micro_avg"] = {
        "EventPrecision": micro["precision"],
        "EventRecall": micro["recall"],
        "EventF1": micro["f1"],
    }
    return result


def ghost_metrics(
    pred_power: Tensor | np.ndarray,
    true_state: Tensor | np.ndarray,
    appliance_names: Optional[list[str]] = None,
    ghost_thresholds: Optional[Tensor | np.ndarray | list | tuple | dict[str, float]] = None,
    mask: Optional[Tensor | np.ndarray] = None,
    default_ghost_threshold: float = 0.01,
) -> dict[str, dict[str, float]]:
    """
    Compute ghost-load metrics.

    GhostMean:
        Mean predicted power when true appliance state is OFF.

    GhostRate:
        Proportion of OFF samples where predicted power exceeds ghost threshold.

    pred_power:
        [N, A, H], original unit, e.g. kW.

    true_state:
        [N, A, H], 1 means ON, 0 means OFF.
    """
    pred = _to_cpu_tensor(pred_power)
    state = _to_cpu_tensor(true_state) > 0.5

    if pred.shape != state.shape:
        raise ValueError(f"pred and state shape mismatch: {pred.shape}, {state.shape}")

    _, num_appliances, _ = pred.shape

    if appliance_names is None:
        appliance_names = DEFAULT_APPLIANCE_NAMES[:num_appliances]

    thresholds = _as_appliance_vector(
        ghost_thresholds,
        appliance_names,
        default_value=default_ghost_threshold,
    )

    mask_t = None if mask is None else (_to_cpu_tensor(mask) > 0.5)

    result: dict[str, dict[str, float]] = {}
    ghost_mean_list = []
    ghost_rate_list = []

    for app_idx, app_name in enumerate(appliance_names):
        pred_i = pred[:, app_idx, :]
        off_i = ~state[:, app_idx, :]

        if mask_t is not None:
            off_i = off_i & mask_t[:, app_idx, :]

        off_pred = pred_i[off_i]

        if off_pred.numel() == 0:
            ghost_mean = float("nan")
            ghost_rate = float("nan")
            off_count = 0.0
        else:
            threshold_i = thresholds[app_idx]
            ghost_mean = float(off_pred.mean().item())
            ghost_rate = float((off_pred > threshold_i).float().mean().item())
            off_count = float(off_pred.numel())

        result[app_name] = {
            "GhostMean": ghost_mean,
            "GhostRate": ghost_rate,
            "OffCount": off_count,
        }

        ghost_mean_list.append(ghost_mean)
        ghost_rate_list.append(ghost_rate)

    result["macro_avg"] = {
        "GhostMean": float(np.nanmean(ghost_mean_list)),
        "GhostRate": float(np.nanmean(ghost_rate_list)),
    }

    return result


def aggregate_consistency_metrics(
    pred_power: Tensor | np.ndarray,
    y_mains: Tensor | np.ndarray,
) -> dict[str, float]:
    """
    Since selected appliances are only a subset of total grid,
    we mainly measure whether sum(pred_appliance) exceeds grid.

    pred_power:
        [N, A, H]

    y_mains:
        [N, H]
    """
    pred = _to_cpu_tensor(pred_power)
    mains = _to_cpu_tensor(y_mains)

    if pred.dim() != 3:
        raise ValueError(f"Expected pred_power [N, A, H], got {pred.shape}")

    if mains.dim() != 2:
        raise ValueError(f"Expected y_mains [N, H], got {mains.shape}")

    pred_sum = pred.sum(dim=1)

    if pred_sum.shape != mains.shape:
        raise ValueError(
            f"pred_sum and mains shape mismatch: {pred_sum.shape}, {mains.shape}"
        )

    excess = torch.relu(pred_sum - mains)

    return {
        "AggregateUpperBoundMAE": float(excess.mean().item()),
        "AggregateUpperBoundRMSE": float(torch.sqrt((excess ** 2).mean()).item()),
        "AggregateViolationRate": float((excess > 0).float().mean().item()),
        "PredSelectedSumMean": float(pred_sum.mean().item()),
        "MainsMean": float(mains.mean().item()),
    }


def peak_window_rmse(
    pred_power: Tensor | np.ndarray,
    target_power: Tensor | np.ndarray,
    appliance_names: Optional[list[str]] = None,
    mask: Optional[Tensor | np.ndarray] = None,
    true_state: Optional[Tensor | np.ndarray] = None,
    peak_quantile: float = 0.80,
) -> dict[str, dict[str, float]]:
    """
    Compute RMSE on peak target moments.

    If true_state is provided, peak positions are selected only from
    true ON samples. This avoids treating tiny standby/noise values
    such as 0.001 kW as peak samples.

    pred_power:
        [N, A, H]

    target_power:
        [N, A, H]

    true_state:
        [N, A, H], 1 means appliance ON.
    """
    pred = _to_cpu_tensor(pred_power)
    target = _to_cpu_tensor(target_power)

    if pred.shape != target.shape:
        raise ValueError(f"pred and target shape mismatch: {pred.shape}, {target.shape}")

    if pred.dim() != 3:
        raise ValueError(f"Expected shape [N, A, H], got {pred.shape}")

    _, num_appliances, _ = pred.shape

    if appliance_names is None:
        appliance_names = DEFAULT_APPLIANCE_NAMES[:num_appliances]

    if len(appliance_names) != num_appliances:
        raise ValueError("len(appliance_names) should match appliance dimension.")

    mask_t = None if mask is None else (_to_cpu_tensor(mask) > 0.5)
    state_t = None if true_state is None else (_to_cpu_tensor(true_state) > 0.5)

    if state_t is not None and state_t.shape != target.shape:
        raise ValueError(
            f"true_state and target_power shape mismatch: {state_t.shape}, {target.shape}"
        )

    result: dict[str, dict[str, float]] = {}
    rmse_list = []

    for app_idx, app_name in enumerate(appliance_names):
        pred_i = pred[:, app_idx, :]
        target_i = target[:, app_idx, :]

        # Prefer true ON-state mask.
        # This prevents standby/noise values from entering peak evaluation.
        if state_t is not None:
            valid_mask = state_t[:, app_idx, :]
        else:
            valid_mask = target_i > 0.0

        if mask_t is not None:
            valid_mask = valid_mask & mask_t[:, app_idx, :]

        valid_target = target_i[valid_mask]

        if valid_target.numel() == 0:
            rmse = float("nan")
            count = 0.0
            threshold = float("nan")
        else:
            threshold = float(torch.quantile(valid_target, peak_quantile).item())
            peak_mask = valid_mask & (target_i >= threshold)

            error = pred_i[peak_mask] - target_i[peak_mask]
            rmse = _rmse_or_nan(error)
            count = float(error.numel())

        result[app_name] = {
            "PeakWindowRMSE": rmse,
            "PeakCount": count,
            "PeakThreshold": threshold,
        }

        rmse_list.append(rmse)

    result["macro_avg"] = {
        "PeakWindowRMSE": float(np.nanmean(rmse_list)),
    }

    return result


def compute_all_metrics(
    out: dict[str, Tensor],
    batch: dict[str, Tensor],
    appliance_names: Optional[list[str]] = None,
    state_thresholds: Optional[Tensor | np.ndarray | list | tuple | dict[str, float]] = None,
    state_prob_threshold: float = 0.5,
    event_prob_threshold: Optional[
        Tensor | np.ndarray | list | tuple | dict[str, float] | float
    ] = None,
    bucket_event_prob_threshold: Optional[
        Tensor | np.ndarray | list | tuple | dict[str, float] | float
    ] = None,
    window_start_prob_threshold: Optional[
        Tensor | np.ndarray | list | tuple | dict[str, float] | float
    ] = None,
    hierarchical_start_prob_threshold: Optional[
        Tensor | np.ndarray | list | tuple | dict[str, float] | float
    ] = None,
    event_tolerance_minutes: int = 0,
    event_bucket_size: int = 0,
    peak_quantile: float = 0.80,
    postprocess_state: bool = False,
    min_on_duration: int = 1,
    min_off_duration: int = 1,
    include_reference_baselines: bool = False,
) -> dict[str, Any]:
    """
    Compute all main PISA metrics from one model output and one batch.

    This function can be used for a single batch or for concatenated full test set.

    Required out:
        out["y_power"] [N, A, H]

    Optional out:
        out["p_on"]          [N, A, H]
        out["state_logits"]  [N, A, H]
        out["event_logits"]  [N, A, H]

    Required batch:
        batch["y_power"] [N, A, H]
        batch["y_state"] [N, A, H]
        batch["y_event"] [N, A, H]

    Optional batch:
        batch["target_mask"] [N, A, H]
        batch["y_mains"]     [N, H]
    """
    if "y_power" not in out:
        raise KeyError("out should contain 'y_power'.")

    if "y_power" not in batch:
        raise KeyError("batch should contain 'y_power'.")

    if "y_state" not in batch:
        raise KeyError("batch should contain 'y_state'.")

    if "y_event" not in batch:
        raise KeyError("batch should contain 'y_event'.")

    pred_power = _to_cpu_tensor(out["y_power"])
    target_power = _to_cpu_tensor(batch["y_power"])
    true_state = _to_cpu_tensor(batch["y_state"])
    true_event = _to_cpu_tensor(batch["y_event"])
    true_start = _to_cpu_tensor(batch["y_start"]) if "y_start" in batch else true_event
    true_stop = _to_cpu_tensor(batch["y_stop"]) if "y_stop" in batch else true_event
    hard_power = _to_cpu_tensor(out["hard_power"]) if "hard_power" in out else None
    state_binary = _to_cpu_tensor(out["state_binary"]) if "state_binary" in out else None

    mask = batch.get("target_mask", None)
    if mask is not None:
        mask = _to_cpu_tensor(mask)

    _, num_appliances, _ = pred_power.shape

    if appliance_names is None:
        appliance_names = DEFAULT_APPLIANCE_NAMES[:num_appliances]

    # Main predicted state is derived from final predicted power.
    # This keeps state/event metrics consistent with the final forecasting output.
    thresholds = _as_appliance_vector(
        state_thresholds,
        appliance_names,
        default_value=0.01,
    ).view(1, -1, 1)
    event_thresholds = _as_appliance_vector(
        event_prob_threshold,
        appliance_names,
        default_value=0.5,
    ).view(1, -1, 1)
    bucket_event_thresholds = _as_appliance_vector(
        bucket_event_prob_threshold,
        appliance_names,
        default_value=0.1,
    ).view(1, -1, 1)
    window_start_thresholds = _as_appliance_vector(
        window_start_prob_threshold,
        appliance_names,
        default_value=0.5,
    ).view(1, -1)
    hierarchical_start_thresholds = _as_appliance_vector(
        hierarchical_start_prob_threshold,
        appliance_names,
        default_value=0.5,
    ).view(1, -1)

    if "event_prev_power" in out:
        pred_previous_state = (
            _to_cpu_tensor(out["event_prev_power"]) >= thresholds
        ).float()
    elif "event_prev_p_on" in out:
        pred_previous_state = (
            _to_cpu_tensor(out["event_prev_p_on"]) >= state_prob_threshold
        ).float()
    elif "y_prev_state" in batch:
        pred_previous_state = _to_cpu_tensor(batch["y_prev_state"])
    elif "y_hist_state" in batch:
        pred_previous_state = _to_cpu_tensor(batch["y_hist_state"])[..., -1]
    else:
        pred_previous_state = None

    pred_state_raw = (pred_power >= thresholds).float()
    pred_state = pred_state_raw

    if postprocess_state:
        pred_state = postprocess_state_sequence(
            pred_state_raw,
            min_on_duration=min_on_duration,
            min_off_duration=min_off_duration,
        )

    # Optional auxiliary state-head prediction.
    # This is logged separately as "state_head", not used as the main state metric.
    pred_state_head = None

    if "p_on" in out:
        pred_state_head = (_to_cpu_tensor(out["p_on"]) >= state_prob_threshold).float()
    elif "state_logits" in out:
        pred_state_head = (
            torch.sigmoid(_to_cpu_tensor(out["state_logits"])) >= state_prob_threshold
        ).float()

    # Predicted event
    # Main event metric is derived from predicted state transitions.
    # This is more consistent because an appliance event is an ON/OFF state change.
    pred_event_raw = derive_event_from_state(
        pred_state_raw,
        previous_state=pred_previous_state,
    )
    pred_event = derive_event_from_state(
        pred_state,
        previous_state=pred_previous_state,
    )

    # Optional auxiliary event-head prediction.
    pred_event_head = None

    if "event_logits" in out:
        pred_event_head = (
            torch.sigmoid(_to_cpu_tensor(out["event_logits"])) >= event_thresholds
        ).float()

    pred_bucket_event_head = None
    if "bucket_event_logits" in out:
        pred_bucket_event_head = (
            torch.sigmoid(_to_cpu_tensor(out["bucket_event_logits"]))
            >= bucket_event_thresholds
        ).float()

    window_start_prob = None
    pred_hierarchical_start = None
    if "window_start_prob" in out:
        window_start_prob = _to_cpu_tensor(out["window_start_prob"])
    elif "window_start_logits" in out:
        window_start_prob = torch.sigmoid(
            _to_cpu_tensor(out["window_start_logits"])
        )

    if window_start_prob is not None and "conditional_start_prob" in out:
        conditional_start = _to_cpu_tensor(out["conditional_start_prob"])
        start_index = conditional_start.argmax(dim=-1, keepdim=True)
        pred_hierarchical_start = torch.zeros_like(conditional_start)
        pred_hierarchical_start.scatter_(2, start_index, 1.0)
        pred_hierarchical_start *= (
            window_start_prob >= hierarchical_start_thresholds
        ).float().unsqueeze(-1)

    pred_start_head = None
    if "start_logits" in out:
        pred_start_head = (
            torch.sigmoid(_to_cpu_tensor(out["start_logits"])) >= event_thresholds
        ).float()

    pred_stop_head = None
    if "stop_logits" in out:
        pred_stop_head = (
            torch.sigmoid(_to_cpu_tensor(out["stop_logits"])) >= event_thresholds
        ).float()

    metrics: dict[str, Any] = {}

    if postprocess_state:
        metrics["postprocess"] = {
            "enabled": 1.0,
            "min_on_duration": float(max(1, int(min_on_duration))),
            "min_off_duration": float(max(1, int(min_off_duration))),
        }

    metrics["regression"] = per_appliance_regression_metrics(
        pred_power=pred_power,
        target_power=target_power,
        appliance_names=appliance_names,
        mask=mask,
    )

    valid_mask = torch.ones_like(true_state) if mask is None else mask
    metrics["true_on_regression"] = per_appliance_regression_metrics(
        pred_power=pred_power,
        target_power=target_power,
        appliance_names=appliance_names,
        mask=valid_mask * (true_state > 0.5).to(valid_mask.dtype),
    )
    metrics["true_off_regression"] = per_appliance_regression_metrics(
        pred_power=pred_power,
        target_power=target_power,
        appliance_names=appliance_names,
        mask=valid_mask * (true_state <= 0.5).to(valid_mask.dtype),
    )

    if "expected_power" in out:
        expected_power = _to_cpu_tensor(out["expected_power"])
        metrics["expected_regression"] = per_appliance_regression_metrics(
            pred_power=expected_power,
            target_power=target_power,
            appliance_names=appliance_names,
            mask=mask,
        )
        metrics["expected_peak"] = peak_window_rmse(
            pred_power=expected_power,
            target_power=target_power,
            appliance_names=appliance_names,
            mask=mask,
            true_state=true_state,
            peak_quantile=peak_quantile,
        )

    if hard_power is not None:
        metrics["hard_regression"] = per_appliance_regression_metrics(
            pred_power=hard_power,
            target_power=target_power,
            appliance_names=appliance_names,
            mask=mask,
        )

        if state_binary is not None:
            hard_state = state_binary.float()
        else:
            hard_state = (hard_power >= thresholds).float()

        hard_event = derive_event_from_state(
            hard_state,
            previous_state=pred_previous_state,
        )

        metrics["hard_state"] = state_classification_metrics(
            pred_state=hard_state,
            true_state=true_state,
            appliance_names=appliance_names,
            mask=mask,
        )
        metrics["hard_event"] = event_classification_metrics(
            pred_event=hard_event,
            true_event=true_event,
            appliance_names=appliance_names,
            mask=mask,
        )
        metrics["hard_ghost"] = ghost_metrics(
            pred_power=hard_power,
            true_state=true_state,
            appliance_names=appliance_names,
            ghost_thresholds=state_thresholds,
            mask=mask,
        )
        metrics["hard_peak"] = peak_window_rmse(
            pred_power=hard_power,
            target_power=target_power,
            appliance_names=appliance_names,
            mask=mask,
            true_state=true_state,
            peak_quantile=peak_quantile,
        )

    if postprocess_state:
        metrics["state_raw"] = state_classification_metrics(
            pred_state=pred_state_raw,
            true_state=true_state,
            appliance_names=appliance_names,
            mask=mask,
        )

    metrics["state"] = state_classification_metrics(
        pred_state=pred_state,
        true_state=true_state,
        appliance_names=appliance_names,
        mask=mask,
    )

    if pred_state_head is not None:
        metrics["state_head"] = state_classification_metrics(
            pred_state=pred_state_head,
            true_state=true_state,
            appliance_names=appliance_names,
            mask=mask,
        )

    if postprocess_state:
        metrics["event_raw"] = event_classification_metrics(
            pred_event=pred_event_raw,
            true_event=true_event,
            appliance_names=appliance_names,
            mask=mask,
        )

    metrics["event"] = event_classification_metrics(
        pred_event=pred_event,
        true_event=true_event,
        appliance_names=appliance_names,
        mask=mask,
    )

    if event_tolerance_minutes > 0:
        metrics["event_tolerant"] = event_tolerance_metrics(
            pred_event=pred_event,
            true_event=true_event,
            tolerance=event_tolerance_minutes,
            appliance_names=appliance_names,
            mask=mask,
        )

        if pred_event_head is not None:
            metrics["event_head_tolerant"] = event_tolerance_metrics(
                pred_event=pred_event_head,
                true_event=true_event,
                tolerance=event_tolerance_minutes,
                appliance_names=appliance_names,
                mask=mask,
            )
        if pred_bucket_event_head is not None:
            metrics["bucket_event_head_tolerant"] = event_tolerance_metrics(
                pred_event=pred_bucket_event_head,
                true_event=true_event,
                tolerance=event_tolerance_minutes,
                appliance_names=appliance_names,
                mask=mask,
            )

    if event_bucket_size > 1:
        metrics["event_bucket"] = event_bucket_metrics(
            pred_event=pred_event,
            true_event=true_event,
            bucket_size=event_bucket_size,
            appliance_names=appliance_names,
            mask=mask,
        )

        if pred_event_head is not None:
            metrics["event_head_bucket"] = event_bucket_metrics(
                pred_event=pred_event_head,
                true_event=true_event,
                bucket_size=event_bucket_size,
                appliance_names=appliance_names,
                mask=mask,
            )
        if pred_bucket_event_head is not None:
            metrics["bucket_event_head_bucket"] = event_bucket_metrics(
                pred_event=pred_bucket_event_head,
                true_event=true_event,
                bucket_size=event_bucket_size,
                appliance_names=appliance_names,
                mask=mask,
            )

    if pred_event_head is not None:
        metrics["event_head"] = event_classification_metrics(
            pred_event=pred_event_head,
            true_event=true_event,
            appliance_names=appliance_names,
            mask=mask,
        )

    if pred_bucket_event_head is not None:
        metrics["bucket_event_head"] = event_classification_metrics(
            pred_event=pred_bucket_event_head,
            true_event=true_event,
            appliance_names=appliance_names,
            mask=mask,
        )

    if window_start_prob is not None:
        metrics["window_start_risk"] = window_start_risk_metrics(
            start_prob=window_start_prob,
            true_start=true_start,
            appliance_names=appliance_names,
            threshold=window_start_thresholds.view(-1),
        )

    if "bucket_start_risk" in out:
        metrics["bucket_start_risk"] = bucket_start_risk_metrics(
            bucket_prob=out["bucket_start_risk"],
            true_start=true_start,
            bucket_size=max(1, int(event_bucket_size)),
            appliance_names=appliance_names,
        )

    if (
        "pulse_duration_logits" in out
        and "pulse_amplitude_power" in out
        and "pulse_appliance_mask" in out
    ):
        duration_logits = _to_cpu_tensor(out["pulse_duration_logits"])
        amplitude_power = _to_cpu_tensor(out["pulse_amplitude_power"])
        pulse_mask = _to_cpu_tensor(out["pulse_appliance_mask"]).view(-1)
        has_start = true_start.sum(dim=-1) > 0.5
        first_start = true_start.argmax(dim=-1)
        minute_ids = torch.arange(true_state.size(-1)).view(1, 1, -1)
        after_start = minute_ids >= first_start.unsqueeze(-1)
        off_after_start = (1.0 - true_state) * after_start.float()
        event_on = (
            after_start
            & (off_after_start.cumsum(dim=-1) < 1.0)
            & (true_state > 0.5)
        )
        true_duration = event_on.sum(dim=-1).clamp_min(1).float()
        pred_duration = duration_logits.argmax(dim=-1).float() + 1.0
        true_amplitude = target_power.masked_fill(~event_on, 0.0).amax(dim=-1)
        pulse_metrics: dict[str, Any] = {}
        for app_idx, app_name in enumerate(appliance_names):
            if app_idx >= pulse_mask.numel() or pulse_mask[app_idx] <= 0.5:
                continue
            amplitude_valid = has_start[:, app_idx]
            duration_valid = amplitude_valid & (true_state[:, app_idx, -1] <= 0.5)
            if duration_valid.any():
                duration_error = (
                    pred_duration[duration_valid, app_idx]
                    - true_duration[duration_valid, app_idx]
                ).abs()
                duration_mae = float(duration_error.mean().item())
                duration_accuracy = float((duration_error == 0).float().mean().item())
            else:
                duration_mae = 0.0
                duration_accuracy = 0.0
            if amplitude_valid.any():
                amplitude_mae = float(
                    (
                        amplitude_power[amplitude_valid, app_idx]
                        - true_amplitude[amplitude_valid, app_idx]
                    ).abs().mean().item()
                )
            else:
                amplitude_mae = 0.0
            pulse_metrics[app_name] = {
                "StartWindows": float(amplitude_valid.sum().item()),
                "DurationWindows": float(duration_valid.sum().item()),
                "DurationMAE": duration_mae,
                "DurationAccuracy": duration_accuracy,
                "AmplitudeMAE": amplitude_mae,
            }
        metrics["pulse_parameters"] = pulse_metrics

    if pred_hierarchical_start is not None:
        metrics["hierarchical_start"] = event_classification_metrics(
            pred_event=pred_hierarchical_start,
            true_event=true_start,
            appliance_names=appliance_names,
            mask=mask,
        )
        if event_tolerance_minutes > 0:
            metrics["hierarchical_start_tolerant"] = event_tolerance_metrics(
                pred_event=pred_hierarchical_start,
                true_event=true_start,
                tolerance=event_tolerance_minutes,
                appliance_names=appliance_names,
                mask=mask,
            )
        if event_bucket_size > 1:
            metrics["hierarchical_start_bucket"] = event_bucket_metrics(
                pred_event=pred_hierarchical_start,
                true_event=true_start,
                bucket_size=event_bucket_size,
                appliance_names=appliance_names,
                mask=mask,
            )

    if "y_prev_state" in batch:
        previous_true = _to_cpu_tensor(batch["y_prev_state"])
        if previous_true.dim() == 3:
            previous_true = previous_true.squeeze(-1)
        future_start = true_start.amax(dim=-1) > 0.5
        group_specs = {
            "prev_on": previous_true > 0.5,
            "off_new_start": (previous_true <= 0.5) & future_start,
        }
        metrics["forecast_groups"] = {
            app_name: {
                "PrevOnWindows": float(group_specs["prev_on"][:, app_idx].sum().item()),
                "OffNewStartWindows": float(
                    group_specs["off_new_start"][:, app_idx].sum().item()
                ),
            }
            for app_idx, app_name in enumerate(appliance_names)
        }
        for group_name, sample_group in group_specs.items():
            group_mask = sample_group.float().unsqueeze(-1).expand_as(true_state)
            if mask is not None:
                group_mask = group_mask * mask
            metrics[f"{group_name}_regression"] = per_appliance_regression_metrics(
                pred_power=pred_power,
                target_power=target_power,
                appliance_names=appliance_names,
                mask=group_mask,
            )
            metrics[f"{group_name}_state"] = state_classification_metrics(
                pred_state=pred_state,
                true_state=true_state,
                appliance_names=appliance_names,
                mask=group_mask,
            )
            metrics[f"{group_name}_peak"] = peak_window_rmse(
                pred_power=pred_power,
                target_power=target_power,
                appliance_names=appliance_names,
                mask=group_mask,
                true_state=true_state,
                peak_quantile=peak_quantile,
            )

        if "conditional_power" in out:
            conditional_power = _to_cpu_tensor(out["conditional_power"])
            conditional_group_mask = group_specs["off_new_start"].float().unsqueeze(-1).expand_as(true_state)
            if mask is not None:
                conditional_group_mask = conditional_group_mask * mask
            metrics["off_new_start_conditional_regression"] = (
                per_appliance_regression_metrics(
                    pred_power=conditional_power,
                    target_power=target_power,
                    appliance_names=appliance_names,
                    mask=conditional_group_mask,
                )
            )
            metrics["off_new_start_conditional_peak"] = peak_window_rmse(
                pred_power=conditional_power,
                target_power=target_power,
                appliance_names=appliance_names,
                mask=conditional_group_mask,
                true_state=true_state,
                peak_quantile=peak_quantile,
            )

    if pred_start_head is not None:
        metrics["start_head"] = event_classification_metrics(
            pred_event=pred_start_head,
            true_event=true_start,
            appliance_names=appliance_names,
            mask=mask,
        )

    if pred_stop_head is not None:
        metrics["stop_head"] = event_classification_metrics(
            pred_event=pred_stop_head,
            true_event=true_stop,
            appliance_names=appliance_names,
            mask=mask,
        )

    metrics["ghost"] = ghost_metrics(
        pred_power=pred_power,
        true_state=true_state,
        appliance_names=appliance_names,
        ghost_thresholds=state_thresholds,
        mask=mask,
    )

    metrics["peak"] = peak_window_rmse(
        pred_power=pred_power,
        target_power=target_power,
        appliance_names=appliance_names,
        mask=mask,
        true_state=true_state,
        peak_quantile=peak_quantile,
    )

    if "y_mains" in batch:
        metrics["aggregate"] = aggregate_consistency_metrics(
            pred_power=pred_power,
            y_mains=batch["y_mains"],
        )

    if include_reference_baselines:
        metrics.update(
            compute_reference_baseline_metrics(
                batch=batch,
                appliance_names=appliance_names,
                state_thresholds=state_thresholds,
                state_prob_threshold=state_prob_threshold,
                event_prob_threshold=event_prob_threshold,
                event_tolerance_minutes=event_tolerance_minutes,
                event_bucket_size=event_bucket_size,
                peak_quantile=peak_quantile,
                postprocess_state=postprocess_state,
                min_on_duration=min_on_duration,
                min_off_duration=min_off_duration,
            )
        )

    return metrics


def _baseline_out_from_power(pred_power: Tensor) -> dict[str, Tensor]:
    return {
        "y_power": pred_power.clamp_min(0.0),
    }


def zero_power_baseline(batch: dict[str, Tensor]) -> Tensor:
    target = _to_cpu_tensor(batch["y_power"])
    return torch.zeros_like(target)


def persistence_power_baseline(batch: dict[str, Tensor]) -> Tensor:
    """Historical-submeter persistence oracle (diagnostic only).

    ``y_hist_power`` is an appliance submeter label and is unavailable to an
    aggregate-only deployed forecaster.  Keep this reference for diagnostics,
    never for fair forecasting comparisons.
    """
    target = _to_cpu_tensor(batch["y_power"])
    if "y_hist_power" not in batch:
        return torch.zeros_like(target)

    hist_power = _to_cpu_tensor(batch["y_hist_power"])
    last_power = hist_power[:, :, -1:].clamp_min(0.0)
    return last_power.expand_as(target).clone()


def aggregate_only_proxy_baseline(batch: dict[str, Tensor]) -> Tensor:
    """
    Future-aggregate plus submeter-mix oracle proxy (diagnostic only).

    It consumes *future* aggregate mains and the latest historical appliance
    mix. Neither is valid for an aggregate-only future forecaster, so this is
    intentionally excluded from the fair baseline report. It remains only for
    backwards-compatible diagnostic logging.
    """
    target = _to_cpu_tensor(batch["y_power"])
    if "y_mains" not in batch or "y_hist_power" not in batch:
        return zero_power_baseline(batch)

    y_mains = _to_cpu_tensor(batch["y_mains"])
    hist_power = _to_cpu_tensor(batch["y_hist_power"])
    last_power = hist_power[:, :, -1:].clamp_min(0.0)
    last_sum = last_power.sum(dim=1, keepdim=True).clamp_min(1e-8)
    mix = last_power / last_sum

    return mix * y_mains.unsqueeze(1)


def compute_reference_baseline_metrics(
    batch: dict[str, Tensor],
    appliance_names: list[str],
    state_thresholds: Optional[Tensor | np.ndarray | list | tuple | dict[str, float]] = None,
    state_prob_threshold: float = 0.5,
    event_prob_threshold: float = 0.5,
    event_tolerance_minutes: int = 0,
    event_bucket_size: int = 0,
    peak_quantile: float = 0.80,
    postprocess_state: bool = False,
    min_on_duration: int = 1,
    min_off_duration: int = 1,
) -> dict[str, Any]:
    baseline_builders = {
        "baseline_zero": zero_power_baseline,
        "baseline_persistence": persistence_power_baseline,
        "baseline_aggregate_only_proxy": aggregate_only_proxy_baseline,
    }

    results: dict[str, Any] = {}

    for name, builder in baseline_builders.items():
        pred_power = builder(batch)
        results[name] = compute_all_metrics(
            out=_baseline_out_from_power(pred_power),
            batch=batch,
            appliance_names=appliance_names,
            state_thresholds=state_thresholds,
            state_prob_threshold=state_prob_threshold,
            event_prob_threshold=event_prob_threshold,
            event_tolerance_minutes=event_tolerance_minutes,
            event_bucket_size=event_bucket_size,
            peak_quantile=peak_quantile,
            postprocess_state=postprocess_state,
            min_on_duration=min_on_duration,
            min_off_duration=min_off_duration,
            include_reference_baselines=False,
        )

    return results


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """
    Flatten nested metric dictionary into a single-level dictionary.

    Example:
        metrics["regression"]["air1"]["MAE"]
        -> "regression/air1/MAE"
    """
    flat: dict[str, float] = {}

    def _recursive(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}/{key}" if prefix else str(key)
                _recursive(new_prefix, value)
        else:
            if isinstance(obj, (int, float, np.floating)):
                flat[prefix] = float(obj)
            elif isinstance(obj, Tensor) and obj.numel() == 1:
                flat[prefix] = float(obj.item())

    _recursive("", metrics)

    return flat


def metrics_to_rows(
    metrics: dict[str, Any],
    appliance_names: Optional[list[str]] = None,
) -> list[dict[str, float | str]]:
    """
    Convert nested metrics into row format for CSV report.

    Each row corresponds to one appliance plus macro/micro rows.
    """
    if appliance_names is None:
        appliance_names = DEFAULT_APPLIANCE_NAMES

    row_names = list(appliance_names) + ["macro_avg", "micro_avg"]

    rows = []

    for name in row_names:
        row: dict[str, float | str] = {"appliance": name}

        for section, section_metrics in metrics.items():
            if not isinstance(section_metrics, dict):
                continue

            if name not in section_metrics:
                continue

            if not isinstance(section_metrics[name], dict):
                continue

            for metric_name, value in section_metrics[name].items():
                if isinstance(value, (int, float, np.floating)):
                    row[f"{section}_{metric_name}"] = float(value)

        if len(row) > 1:
            rows.append(row)

    return rows


@dataclass
class MetricsAccumulator:
    """
    Accumulate model outputs and batches over an epoch/test set.

    Usage:
        acc = MetricsAccumulator(appliance_names=APPLIANCE_COLS)
        for batch in loader:
            out = model(batch)
            acc.update(out, batch)
        metrics = acc.compute(state_thresholds=bundle.state_thresholds)
    """

    appliance_names: list[str] = field(default_factory=lambda: DEFAULT_APPLIANCE_NAMES.copy())

    pred_power_list: list[Tensor] = field(default_factory=list)
    target_power_list: list[Tensor] = field(default_factory=list)
    true_state_list: list[Tensor] = field(default_factory=list)
    true_event_list: list[Tensor] = field(default_factory=list)
    true_start_list: list[Tensor] = field(default_factory=list)
    true_stop_list: list[Tensor] = field(default_factory=list)
    hist_power_list: list[Tensor] = field(default_factory=list)
    previous_state_prob_list: list[Tensor] = field(default_factory=list)
    previous_power_list: list[Tensor] = field(default_factory=list)
    true_previous_state_list: list[Tensor] = field(default_factory=list)
    p_on_list: list[Tensor] = field(default_factory=list)
    hard_power_list: list[Tensor] = field(default_factory=list)
    state_binary_list: list[Tensor] = field(default_factory=list)
    event_logits_list: list[Tensor] = field(default_factory=list)
    bucket_event_logits_list: list[Tensor] = field(default_factory=list)
    window_start_prob_list: list[Tensor] = field(default_factory=list)
    conditional_start_prob_list: list[Tensor] = field(default_factory=list)
    conditional_power_list: list[Tensor] = field(default_factory=list)
    expected_power_list: list[Tensor] = field(default_factory=list)
    bucket_start_risk_list: list[Tensor] = field(default_factory=list)
    pulse_duration_logits_list: list[Tensor] = field(default_factory=list)
    pulse_amplitude_power_list: list[Tensor] = field(default_factory=list)
    pulse_appliance_mask: Optional[Tensor] = None
    start_logits_list: list[Tensor] = field(default_factory=list)
    stop_logits_list: list[Tensor] = field(default_factory=list)
    mask_list: list[Tensor] = field(default_factory=list)
    y_mains_list: list[Tensor] = field(default_factory=list)

    def update(self, out: dict[str, Tensor], batch: dict[str, Tensor]) -> None:
        self.pred_power_list.append(_to_cpu_tensor(out["y_power"]))
        self.target_power_list.append(_to_cpu_tensor(batch["y_power"]))
        self.true_state_list.append(_to_cpu_tensor(batch["y_state"]))
        self.true_event_list.append(_to_cpu_tensor(batch["y_event"]))

        if "y_start" in batch:
            self.true_start_list.append(_to_cpu_tensor(batch["y_start"]))

        if "y_stop" in batch:
            self.true_stop_list.append(_to_cpu_tensor(batch["y_stop"]))

        if "y_hist_power" in batch:
            self.hist_power_list.append(_to_cpu_tensor(batch["y_hist_power"]))

        if "event_prev_p_on" in out:
            self.previous_state_prob_list.append(
                _to_cpu_tensor(out["event_prev_p_on"])
            )

        if "event_prev_power" in out:
            self.previous_power_list.append(_to_cpu_tensor(out["event_prev_power"]))

        if "y_prev_state" in batch:
            self.true_previous_state_list.append(
                _to_cpu_tensor(batch["y_prev_state"])
            )

        if "p_on" in out:
            self.p_on_list.append(_to_cpu_tensor(out["p_on"]))

        if "hard_power" in out:
            self.hard_power_list.append(_to_cpu_tensor(out["hard_power"]))

        if "state_binary" in out:
            self.state_binary_list.append(_to_cpu_tensor(out["state_binary"]))

        if "event_logits" in out:
            self.event_logits_list.append(_to_cpu_tensor(out["event_logits"]))

        if "bucket_event_logits" in out:
            self.bucket_event_logits_list.append(
                _to_cpu_tensor(out["bucket_event_logits"])
            )

        if "window_start_prob" in out:
            self.window_start_prob_list.append(
                _to_cpu_tensor(out["window_start_prob"])
            )
        if "conditional_start_prob" in out:
            self.conditional_start_prob_list.append(
                _to_cpu_tensor(out["conditional_start_prob"])
            )
        if "conditional_power" in out:
            self.conditional_power_list.append(
                _to_cpu_tensor(out["conditional_power"])
            )
        if "expected_power" in out:
            self.expected_power_list.append(_to_cpu_tensor(out["expected_power"]))
        if "bucket_start_risk" in out:
            self.bucket_start_risk_list.append(
                _to_cpu_tensor(out["bucket_start_risk"])
            )
        if "pulse_duration_logits" in out:
            self.pulse_duration_logits_list.append(
                _to_cpu_tensor(out["pulse_duration_logits"])
            )
        if "pulse_amplitude_power" in out:
            self.pulse_amplitude_power_list.append(
                _to_cpu_tensor(out["pulse_amplitude_power"])
            )
        if "pulse_appliance_mask" in out and self.pulse_appliance_mask is None:
            self.pulse_appliance_mask = _to_cpu_tensor(out["pulse_appliance_mask"])

        if "start_logits" in out:
            self.start_logits_list.append(_to_cpu_tensor(out["start_logits"]))

        if "stop_logits" in out:
            self.stop_logits_list.append(_to_cpu_tensor(out["stop_logits"]))

        if "target_mask" in batch:
            self.mask_list.append(_to_cpu_tensor(batch["target_mask"]))

        if "y_mains" in batch:
            self.y_mains_list.append(_to_cpu_tensor(batch["y_mains"]))

    def compute(
        self,
        state_thresholds: Optional[Tensor | np.ndarray | list | tuple | dict[str, float]] = None,
        state_prob_threshold: float = 0.5,
        event_prob_threshold: Optional[
            Tensor | np.ndarray | list | tuple | dict[str, float] | float
        ] = None,
        bucket_event_prob_threshold: Optional[
            Tensor | np.ndarray | list | tuple | dict[str, float] | float
        ] = None,
        window_start_prob_threshold: Optional[
            Tensor | np.ndarray | list | tuple | dict[str, float] | float
        ] = None,
        hierarchical_start_prob_threshold: Optional[
            Tensor | np.ndarray | list | tuple | dict[str, float] | float
        ] = None,
        event_tolerance_minutes: int = 0,
        event_bucket_size: int = 0,
        peak_quantile: float = 0.80,
        postprocess_state: bool = False,
        min_on_duration: int = 1,
        min_off_duration: int = 1,
    ) -> dict[str, Any]:
        if len(self.pred_power_list) == 0:
            raise RuntimeError("No metrics accumulated. Call update() first.")

        out = {
            "y_power": torch.cat(self.pred_power_list, dim=0),
        }

        batch = {
            "y_power": torch.cat(self.target_power_list, dim=0),
            "y_state": torch.cat(self.true_state_list, dim=0),
            "y_event": torch.cat(self.true_event_list, dim=0),
        }

        if self.true_start_list:
            batch["y_start"] = torch.cat(self.true_start_list, dim=0)

        if self.true_stop_list:
            batch["y_stop"] = torch.cat(self.true_stop_list, dim=0)

        if self.hist_power_list:
            batch["y_hist_power"] = torch.cat(self.hist_power_list, dim=0)

        if self.previous_state_prob_list:
            out["event_prev_p_on"] = torch.cat(
                self.previous_state_prob_list,
                dim=0,
            )

        if self.previous_power_list:
            out["event_prev_power"] = torch.cat(self.previous_power_list, dim=0)

        if self.true_previous_state_list:
            batch["y_prev_state"] = torch.cat(
                self.true_previous_state_list,
                dim=0,
            )

        if self.p_on_list:
            out["p_on"] = torch.cat(self.p_on_list, dim=0)

        if self.hard_power_list:
            out["hard_power"] = torch.cat(self.hard_power_list, dim=0)

        if self.state_binary_list:
            out["state_binary"] = torch.cat(self.state_binary_list, dim=0)

        if self.event_logits_list:
            out["event_logits"] = torch.cat(self.event_logits_list, dim=0)

        if self.bucket_event_logits_list:
            out["bucket_event_logits"] = torch.cat(
                self.bucket_event_logits_list,
                dim=0,
            )

        if self.window_start_prob_list:
            out["window_start_prob"] = torch.cat(
                self.window_start_prob_list,
                dim=0,
            )
        if self.conditional_start_prob_list:
            out["conditional_start_prob"] = torch.cat(
                self.conditional_start_prob_list,
                dim=0,
            )
        if self.conditional_power_list:
            out["conditional_power"] = torch.cat(
                self.conditional_power_list,
                dim=0,
            )
        if self.expected_power_list:
            out["expected_power"] = torch.cat(self.expected_power_list, dim=0)
        if self.bucket_start_risk_list:
            out["bucket_start_risk"] = torch.cat(
                self.bucket_start_risk_list, dim=0
            )
        if self.pulse_duration_logits_list:
            out["pulse_duration_logits"] = torch.cat(
                self.pulse_duration_logits_list, dim=0
            )
        if self.pulse_amplitude_power_list:
            out["pulse_amplitude_power"] = torch.cat(
                self.pulse_amplitude_power_list, dim=0
            )
        if self.pulse_appliance_mask is not None:
            out["pulse_appliance_mask"] = self.pulse_appliance_mask

        if self.start_logits_list:
            out["start_logits"] = torch.cat(self.start_logits_list, dim=0)

        if self.stop_logits_list:
            out["stop_logits"] = torch.cat(self.stop_logits_list, dim=0)

        if self.mask_list:
            batch["target_mask"] = torch.cat(self.mask_list, dim=0)

        if self.y_mains_list:
            batch["y_mains"] = torch.cat(self.y_mains_list, dim=0)

        return compute_all_metrics(
            out=out,
            batch=batch,
            appliance_names=self.appliance_names,
            state_thresholds=state_thresholds,
            state_prob_threshold=state_prob_threshold,
            event_prob_threshold=event_prob_threshold,
            bucket_event_prob_threshold=bucket_event_prob_threshold,
            window_start_prob_threshold=window_start_prob_threshold,
            hierarchical_start_prob_threshold=hierarchical_start_prob_threshold,
            event_tolerance_minutes=event_tolerance_minutes,
            event_bucket_size=event_bucket_size,
            peak_quantile=peak_quantile,
            postprocess_state=postprocess_state,
            min_on_duration=min_on_duration,
            min_off_duration=min_off_duration,
            include_reference_baselines=True,
        )

    def reset(self) -> None:
        self.pred_power_list.clear()
        self.target_power_list.clear()
        self.true_state_list.clear()
        self.true_event_list.clear()
        self.true_start_list.clear()
        self.true_stop_list.clear()
        self.hist_power_list.clear()
        self.previous_state_prob_list.clear()
        self.previous_power_list.clear()
        self.true_previous_state_list.clear()
        self.p_on_list.clear()
        self.hard_power_list.clear()
        self.state_binary_list.clear()
        self.event_logits_list.clear()
        self.bucket_event_logits_list.clear()
        self.window_start_prob_list.clear()
        self.conditional_start_prob_list.clear()
        self.conditional_power_list.clear()
        self.expected_power_list.clear()
        self.bucket_start_risk_list.clear()
        self.pulse_duration_logits_list.clear()
        self.pulse_amplitude_power_list.clear()
        self.pulse_appliance_mask = None
        self.start_logits_list.clear()
        self.stop_logits_list.clear()
        self.mask_list.clear()
        self.y_mains_list.clear()
