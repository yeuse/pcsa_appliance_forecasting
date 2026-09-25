"""Evaluation-only event protocols and validation-only fusion diagnostics."""
from collections import defaultdict

import torch
from evaluation.metrics import (
    MetricsAccumulator, flatten_metrics, postprocess_state_sequence,
    event_classification_metrics, event_tolerance_metrics,
)
from training import move_batch_to_device


def cpu(value):
    return value.detach().float().cpu()


def directional(state, previous):
    previous = torch.cat((previous.reshape(*state.shape[:2], 1), state[..., :-1]), -1)
    return {"start": (state > .5) & (previous <= .5),
            "stop": (state <= .5) & (previous > .5)}


def predicted_previous(out, thresholds):
    # Prefer the history actually consumed by the residual TCN. Do not use
    # event_prev_power left over from an earlier, different reconstruction.
    for key in ("future_tcn_input_power", "nilm_past_power", "past_power", "event_prev_power"):
        if key in out:
            power = cpu(out[key])
            if power.ndim == 2:
                power = power.unsqueeze(-1)
            return (power[..., -1] >= thresholds.reshape(1, -1)).float(), key
    return None, "unavailable_no_predicted_appliance_history"


def event_report(pred, truth, mask, true_prev, pred_prev, names, source):
    """Boundary excluded BEFORE tolerant matching; both endpoints must be valid."""
    valid = mask > 0
    transition_mask = valid.clone()
    transition_mask[..., 1:] &= valid[..., :-1]
    true_events = directional(truth, true_prev)
    report = {"previous_state_source": source,
              "counting": "overlapping forecast windows, not unique household events",
              "full_predicted_history_available": pred_prev is not None,
              "protocols": {}}
    for processing in ("raw", "postprocessed_3on_2off"):
        states = pred if processing == "raw" else postprocess_state_sequence(
            pred, min_on_duration=3, min_off_duration=2)
        for scope in ("internal", "full_predicted_history", "full_true_history_DIAGNOSTIC_ONLY"):
            if scope == "full_predicted_history" and pred_prev is None:
                continue
            previous = true_prev if scope.endswith("DIAGNOSTIC_ONLY") else (
                pred_prev if pred_prev is not None else torch.zeros_like(true_prev))
            predicted = directional(states, previous)
            active_mask = transition_mask.clone()
            if scope == "internal":
                active_mask[..., 0] = False
            for direction in ("start", "stop"):
                key = f"{scope}/{processing}/{direction}"
                report["protocols"][key] = {
                    "valid_transition_positions": {name: int(active_mask[:, i].sum()) for i, name in enumerate(names)},
                    "exact": event_classification_metrics(predicted[direction], true_events[direction], names, active_mask),
                    "tolerance_2min": event_tolerance_metrics(predicted[direction], true_events[direction], 2, names, active_mask),
                }
    return report


TRACE_KEYS = (
    "y_power", "p_on", "amplitude_power", "state_gate", "y_raw", "state_logits",
    "base_y_power", "base_p_on", "base_y_raw", "base_state_logits",
    "future_tcn_power_gate", "future_tcn_power_residual_raw",
    "future_tcn_state_gate", "future_tcn_state_residual_logits",
    "future_tcn_uncalibrated_amplitude", "future_tcn_direct_power",
    "future_tcn_legacy_power", "future_tcn_base_power_blend",
    "future_tcn_target_power_scale", "future_tcn_target_power_bias",
)


def fusion_values(out):
    shape = out["y_power"].shape
    values = {key: cpu(value).expand(shape) for key in TRACE_KEYS
              if (value := out.get(key)) is not None}
    for prefix, residual in (("power", "raw"), ("state", "logits")):
        gate, delta = f"future_tcn_{prefix}_gate", f"future_tcn_{prefix}_residual_{residual}"
        if gate in values and delta in values:
            values[f"gated_{prefix}_residual_{residual}"] = values[gate] * values[delta]
    if "base_y_power" in values:
        values["final_minus_base_power_kW"] = values["y_power"] - values["base_y_power"]
    if "amplitude_power" in values:
        values["final_minus_amplitude_kW"] = values["y_power"] - values["amplitude_power"]
    return values


def fusion_report(values, target, truth, mask, names, model):
    result = {"split": "validation_only", "interpretation":
              "Observed route quantities, not additive causal attribution; raw/logit deltas are NOT kW.",
              "fusion_mode": getattr(model, "future_residual_fusion", None),
              "state_gate_floor": getattr(model, "state_gate_floor", None),
              "missing_fields": sorted(set(TRACE_KEYS) - set(values)), "devices": {}}
    for i, name in enumerate(names):
        result["devices"][name] = {}
        for condition, selection in (("all", mask[:, i] > 0),
                                     ("ON", (mask[:, i] > 0) & (truth[:, i] > .5)),
                                     ("OFF", (mask[:, i] > 0) & (truth[:, i] <= .5))):
            n = int(selection.sum())
            stats = {"count": n, "TargetMean": float(target[:, i][selection].mean()) if n else None, "fields": {}}
            for key, value in values.items():
                x = value[:, i][selection]
                stats["fields"][key] = None if not n else dict(
                    mean=float(x.mean()), mean_abs=float(x.abs().mean()),
                    p05=float(torch.quantile(x, .05)), p50=float(torch.quantile(x, .5)),
                    p95=float(torch.quantile(x, .95)))
            for key in ("y_power", "base_y_power", "amplitude_power"):
                if key in values:
                    stats[f"{key}_MAE_kW"] = float((values[key][:, i][selection] - target[:, i][selection]).abs().mean()) if n else None
            result["devices"][name][condition] = stats
    return result


@torch.inference_mode()
def evaluate_audited(model, loader, device, thresholds, names, use_amp, *, trace_fusion=False):
    model.to(device).eval()
    acc = MetricsAccumulator(appliance_names=names)
    chunks = defaultdict(list)
    traces = defaultdict(list)
    sources = set()
    threshold_tensor = torch.tensor([thresholds[n] for n in names]).reshape(1, -1, 1)
    for batch in loader:
        moved = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=use_amp and device.type == "cuda"):
            out = model(moved)
        acc.update(out, moved)
        truth = cpu(batch["y_state"])
        chunks["truth"].append(truth)
        chunks["target"].append(cpu(batch["y_power"]))
        chunks["mask"].append(cpu(batch.get("target_mask", torch.ones_like(truth))))
        chunks["pred"].append((cpu(out["y_power"]) >= threshold_tensor).float())
        if "y_prev_state" not in batch:
            raise ValueError("Explicit true historical boundary required for event reference")
        chunks["true_prev"].append(cpu(batch["y_prev_state"]).reshape(truth.shape[:2]))
        previous, source = predicted_previous(out, threshold_tensor)
        sources.add(source)
        if previous is not None:
            chunks["pred_prev"].append(previous)
        if trace_fusion:
            for key, value in fusion_values(out).items():
                traces[key].append(value)
    if len(sources) != 1:
        raise ValueError("Predicted historical boundary source changed between batches")
    tensors = {k: torch.cat(v) for k, v in chunks.items()}
    # Retain legacy metrics for comparison with archived reports; the event
    # audit below uses the corrected boundary protocol.
    metrics = acc.compute(state_thresholds=thresholds, event_tolerance_minutes=2,
                          event_bucket_size=5, postprocess_state=True,
                          min_on_duration=3, min_off_duration=2)
    result = {"nested": metrics, "flat": flatten_metrics(metrics),
              "legacy_event_warning": "Legacy event fields can fall back to true historical state; use event_audit for comparisons.",
              "event_audit": event_report(tensors["pred"], tensors["truth"], tensors["mask"],
                  tensors["true_prev"], tensors.get("pred_prev"), names, next(iter(sources)))}
    if trace_fusion:
        result["fusion_audit"] = fusion_report({k: torch.cat(v) for k, v in traces.items()},
            tensors["target"], tensors["truth"], tensors["mask"], names, model)
    return result
