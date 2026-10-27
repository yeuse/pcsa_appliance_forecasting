from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(ROOT))
sys.path.append(str(SRC))

from data import build_single_home_datasets  # noqa: E402
from training import move_batch_to_device  # noqa: E402
from scripts.run_baseline_cross_home_transfer import (  # noqa: E402
    APPLIANCES,
    INPUT_COLS,
    apply_caps,
    build_models,
    load_checkpoint_strict as load_baseline_checkpoint,
)
from scripts.run_pisa_cross_home_transfer import (  # noqa: E402
    DEFAULT_STATE_THRESHOLDS,
    apply_rated_power_caps,
    build_loaders,
    build_source_model,
    load_checkpoint_strict as load_pisa_checkpoint,
)


EVALUATOR_ID = "transfer_history_forecast_diagnostic_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the validation-selected PISA and Seq2Seq transfer models "
            "on (A) historical reconstruction and (B) future forecasting with "
            "the same oracle appliance history. No model is trained."
        )
    )
    parser.add_argument(
        "--pisa_run_dir",
        type=Path,
        required=True,
        help="Completed PISA cross-home transfer run directory.",
    )
    parser.add_argument(
        "--baseline_run_dir",
        type=Path,
        required=True,
        help="Completed baseline cross-home transfer run directory.",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.10],
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Smoke-test limit. Omit for the complete validation split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON. Defaults to "
            "<pisa_run_dir>/results/history_forecast_diagnostic.json."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=True)


def resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def fraction_key(value: float) -> str:
    if not 0.0 < float(value) <= 1.0:
        raise ValueError(f"Few-shot fraction must be in (0, 1], got {value}.")
    return f"{float(value):.3f}"


def percent_better(candidate: float, reference: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(reference) or reference == 0.0:
        return float("nan")
    return 100.0 * (reference - candidate) / reference


def safe_f1(tp: float, fp: float, fn: float) -> float:
    denominator = 2.0 * tp + fp + fn
    return 2.0 * tp / denominator if denominator > 0.0 else float("nan")


class PowerAccumulator:
    """Per-appliance power MAE/RMSE with the repository's macro convention."""

    def __init__(self, appliance_names: list[str]) -> None:
        count = len(appliance_names)
        self.appliance_names = list(appliance_names)
        self.absolute_error = torch.zeros(count, dtype=torch.float64)
        self.squared_error = torch.zeros(count, dtype=torch.float64)
        self.valid_count = torch.zeros(count, dtype=torch.float64)

    def update(self, prediction: Tensor, target: Tensor, mask: Tensor | None) -> None:
        prediction = prediction.detach().to(device="cpu", dtype=torch.float32)
        target = target.detach().to(device="cpu", dtype=torch.float32)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Power prediction/target shapes differ: {prediction.shape}, {target.shape}."
            )
        valid = (
            torch.ones_like(target, dtype=torch.bool)
            if mask is None
            else mask.detach().to(device="cpu") > 0.5
        )
        error = prediction - target
        valid_float = valid.to(error.dtype)
        self.absolute_error += (
            error.abs() * valid_float
        ).sum(dim=(0, 2)).double()
        self.squared_error += (
            error.square() * valid_float
        ).sum(dim=(0, 2)).double()
        self.valid_count += valid.sum(dim=(0, 2)).double()

    def compute(self) -> dict[str, Any]:
        per_appliance: dict[str, dict[str, float]] = {}
        maes: list[float] = []
        rmses: list[float] = []
        for index, appliance in enumerate(self.appliance_names):
            count = float(self.valid_count[index])
            if count <= 0.0:
                mae = rmse = float("nan")
            else:
                mae = float(self.absolute_error[index] / count)
                rmse = math.sqrt(float(self.squared_error[index] / count))
            per_appliance[appliance] = {
                "MAE_kW": mae,
                "RMSE_kW": rmse,
                "valid_points": count,
            }
            maes.append(mae)
            rmses.append(rmse)
        finite_maes = [value for value in maes if math.isfinite(value)]
        finite_rmses = [value for value in rmses if math.isfinite(value)]
        return {
            "per_appliance": per_appliance,
            "macro_avg": {
                "MAE_kW": (
                    sum(finite_maes) / len(finite_maes)
                    if finite_maes
                    else float("nan")
                ),
                "RMSE_kW": (
                    sum(finite_rmses) / len(finite_rmses)
                    if finite_rmses
                    else float("nan")
                ),
            },
        }


class MagnitudeAccumulator(PowerAccumulator):
    """Mean absolute/RMS magnitude of an [B,A,H] tensor."""

    def update_values(self, value: Tensor, mask: Tensor | None) -> None:
        self.update(value, torch.zeros_like(value), mask)

    def compute(self) -> dict[str, Any]:
        result = super().compute()
        for item in result["per_appliance"].values():
            item["mean_absolute"] = item.pop("MAE_kW")
            item["root_mean_square"] = item.pop("RMSE_kW")
        macro = result["macro_avg"]
        macro["mean_absolute"] = macro.pop("MAE_kW")
        macro["root_mean_square"] = macro.pop("RMSE_kW")
        return result


class HistoryAccumulator(PowerAccumulator):
    """Power errors plus state F1 for the history actually consumed downstream."""

    def __init__(self, appliance_names: list[str]) -> None:
        super().__init__(appliance_names)
        count = len(appliance_names)
        self.actual_tp = torch.zeros(count, dtype=torch.float64)
        self.actual_fp = torch.zeros(count, dtype=torch.float64)
        self.actual_fn = torch.zeros(count, dtype=torch.float64)
        self.raw_tp = torch.zeros(count, dtype=torch.float64)
        self.raw_fp = torch.zeros(count, dtype=torch.float64)
        self.raw_fn = torch.zeros(count, dtype=torch.float64)

    @staticmethod
    def confusion(
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        prediction = prediction.bool()
        target = target.bool()
        mask = mask.bool()
        dims = (0, 2)
        return (
            (prediction & target & mask).sum(dim=dims).double(),
            (prediction & ~target & mask).sum(dim=dims).double(),
            (~prediction & target & mask).sum(dim=dims).double(),
        )

    def update_history(
        self,
        prediction_power: Tensor,
        actual_input_p_on: Tensor,
        raw_head_p_on: Tensor,
        target_power: Tensor,
        target_state: Tensor,
        mask: Tensor | None,
    ) -> None:
        self.update(prediction_power, target_power, mask)
        target_state = target_state.detach().to(device="cpu") >= 0.5
        valid = (
            torch.ones_like(target_state, dtype=torch.bool)
            if mask is None
            else mask.detach().to(device="cpu") > 0.5
        )
        actual = actual_input_p_on.detach().to(device="cpu") >= 0.5
        raw = raw_head_p_on.detach().to(device="cpu") >= 0.5
        if actual.shape != target_state.shape or raw.shape != target_state.shape:
            raise ValueError("Historical state prediction and target shapes differ.")
        actual_counts = self.confusion(actual, target_state, valid)
        raw_counts = self.confusion(raw, target_state, valid)
        self.actual_tp += actual_counts[0]
        self.actual_fp += actual_counts[1]
        self.actual_fn += actual_counts[2]
        self.raw_tp += raw_counts[0]
        self.raw_fp += raw_counts[1]
        self.raw_fn += raw_counts[2]

    def compute(self) -> dict[str, Any]:
        result = super().compute()
        actual_values: list[float] = []
        raw_values: list[float] = []
        for index, appliance in enumerate(self.appliance_names):
            actual_f1 = safe_f1(
                float(self.actual_tp[index]),
                float(self.actual_fp[index]),
                float(self.actual_fn[index]),
            )
            raw_f1 = safe_f1(
                float(self.raw_tp[index]),
                float(self.raw_fp[index]),
                float(self.raw_fn[index]),
            )
            result["per_appliance"][appliance].update(
                {
                    "StateF1_actual_forecaster_input": actual_f1,
                    "StateF1_raw_reconstruction_head": raw_f1,
                }
            )
            actual_values.append(actual_f1)
            raw_values.append(raw_f1)
        valid_actual = [value for value in actual_values if math.isfinite(value)]
        valid_raw = [value for value in raw_values if math.isfinite(value)]
        result["macro_avg"].update(
            {
                "StateF1_actual_forecaster_input": (
                    sum(valid_actual) / len(valid_actual)
                    if valid_actual
                    else float("nan")
                ),
                "StateF1_raw_reconstruction_head": (
                    sum(valid_raw) / len(valid_raw)
                    if valid_raw
                    else float("nan")
                ),
            }
        )
        return result


def source_pisa_config_path(
    transfer_config: dict[str, Any],
    source_checkpoint: Path,
) -> Path:
    explicit = transfer_config.get("source_config")
    if explicit:
        return resolved(explicit)
    return source_checkpoint.parent.parent / "config.json"


def validate_protocol(
    pisa_config: dict[str, Any],
    baseline_config: dict[str, Any],
    pisa_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    pisa_source_config: dict[str, Any],
    baseline_source_config: dict[str, Any],
) -> tuple[Path, Path, int, int, int, list[str], list[str]]:
    pisa_source = resolved(pisa_config["source_csv"])
    pisa_target = resolved(pisa_config["target_csv"])
    baseline_source = resolved(baseline_config["source_csv"])
    baseline_target = resolved(baseline_config["target_csv"])
    if pisa_source != baseline_source or pisa_target != baseline_target:
        raise ValueError(
            "PISA and baseline runs do not use the same source/target CSV files."
        )

    input_cols = list(pisa_source_config["input_cols"])
    appliances = list(pisa_source_config["appliance_cols"])
    if input_cols != INPUT_COLS or appliances != APPLIANCES:
        raise ValueError(
            f"Expected inputs={INPUT_COLS}, appliances={APPLIANCES}; got "
            f"inputs={input_cols}, appliances={appliances}."
        )
    if list(baseline_source_config["input_cols"]) != input_cols:
        raise ValueError("PISA and baseline source input columns differ.")
    baseline_apps = baseline_source_config.get(
        "appliances", baseline_source_config.get("appliance_cols")
    )
    if list(baseline_apps) != appliances:
        raise ValueError("PISA and baseline source appliance order differs.")

    data_cfg = dict(pisa_source_config["data"])
    input_window = int(
        pisa_config.get("input_window")
        if pisa_config.get("input_window") is not None
        else data_cfg.get("input_window", 120)
    )
    horizon = int(
        pisa_config.get("horizon")
        if pisa_config.get("horizon") is not None
        else data_cfg.get("horizon", 30)
    )
    stride = int(
        pisa_config.get("stride")
        if pisa_config.get("stride") is not None
        else data_cfg.get("stride", 1)
    )
    baseline_args = baseline_source_config["args"]
    expected = {
        "input_window": input_window,
        "horizon": horizon,
        "stride": stride,
    }
    for field, value in expected.items():
        baseline_value = int(
            baseline_config.get(field, baseline_args.get(field))
        )
        if baseline_value != value:
            raise ValueError(
                f"PISA/baseline {field} mismatch: {value} vs {baseline_value}."
            )

    pisa_home = pisa_summary.get("protocol", {}).get("target_home")
    baseline_home = baseline_summary.get("protocol", {}).get("target_home")
    if pisa_home and baseline_home and pisa_home != baseline_home:
        raise ValueError(
            f"Transfer summaries target different homes: {pisa_home}, {baseline_home}."
        )
    return (
        pisa_source,
        pisa_target,
        input_window,
        horizon,
        stride,
        input_cols,
        appliances,
    )


def source_state_thresholds(
    source_checkpoint: Path,
    baseline_source_config: dict[str, Any],
    appliance_names: list[str],
) -> dict[str, float]:
    data_info_path = source_checkpoint.parent.parent / "results" / "data_info.json"
    data_info = read_json(data_info_path) if data_info_path.is_file() else {}
    pisa_values = data_info.get("state_thresholds", DEFAULT_STATE_THRESHOLDS)
    thresholds = {
        name: float(pisa_values[name]) for name in appliance_names
    }
    baseline_values = baseline_source_config.get("state_thresholds", {})
    if baseline_values:
        for name in appliance_names:
            if not math.isclose(
                thresholds[name], float(baseline_values[name]), rel_tol=0.0, abs_tol=1e-7
            ):
                raise ValueError(
                    f"Source state threshold differs for {name}: PISA "
                    f"{thresholds[name]} vs Seq2Seq {baseline_values[name]}."
                )
    return thresholds


def selected_pisa_model(
    run_dir: Path,
    transfer_config: dict[str, Any],
    source_config: dict[str, Any],
    summary_item: dict[str, Any],
    key: str,
) -> tuple[nn.Module, Path, dict[str, Any]]:
    adapter_dim: int | None = None
    if transfer_config.get("fine_tune_mode") == "power_adapter_heads":
        adapter_dim = int(transfer_config.get("power_adapter_dim", 0))
    effective_source_config = copy.deepcopy(source_config)
    if transfer_config.get("enhanced_forecast_head", False):
        effective_source_config.setdefault("model", {}).update(
            {
                "residual_tcn_use_future_context": True,
                "residual_tcn_target_adapter_dim": int(
                    transfer_config.get("target_adapter_dim", 32)
                ),
                "future_residual_fusion": "learned_blend",
                "future_base_blend_init": float(
                    transfer_config.get("future_base_blend_init", 4.0)
                ),
                "target_output_calibration": True,
            }
        )
    model = build_source_model(
        effective_source_config,
        power_adapter_dim=adapter_dim,
    )
    selected = str(summary_item["selected_model"])
    explicit_checkpoint = summary_item.get("selected_checkpoint")
    if explicit_checkpoint:
        checkpoint = resolved(explicit_checkpoint)
    elif selected == "few_shot_checkpoint":
        checkpoint = run_dir / "checkpoints" / f"fewshot_{key}" / "best.pt"
    elif selected in {"source_zero_shot", "support_calibrated_source"}:
        checkpoint = resolved(transfer_config["source_checkpoint"])
    else:
        raise ValueError(f"Unsupported PISA selected_model={selected!r}.")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected PISA checkpoint not found: {checkpoint}")
    load_pisa_checkpoint(model, checkpoint)
    cap_profile = read_json(run_dir / "results" / f"support_caps_{key}.json")
    apply_rated_power_caps(model, cap_profile["calibrated_caps_kw"])
    return model, checkpoint, cap_profile


def selected_seq2seq_model(
    run_dir: Path,
    source_config: dict[str, Any],
    method_summary: dict[str, Any],
    summary_item: dict[str, Any],
    key: str,
) -> tuple[nn.Module, Path, dict[str, Any]]:
    _, model = build_models(source_config)
    selected = str(summary_item["selected_model"])
    if selected == "few_shot_checkpoint":
        checkpoint = (
            run_dir / "checkpoints" / "two_stage" / f"fewshot_{key}" / "best.pt"
        )
    elif selected in {"source_zero_shot", "support_calibrated_source"}:
        checkpoint = resolved(method_summary["source_checkpoint"])
    else:
        raise ValueError(f"Unsupported Seq2Seq selected_model={selected!r}.")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected Seq2Seq checkpoint not found: {checkpoint}")
    load_baseline_checkpoint(model, checkpoint)
    cap_profile = read_json(run_dir / "results" / f"support_caps_{key}.json")
    apply_caps(model, cap_profile["calibrated_caps_kw"])
    return model, checkpoint, cap_profile


def check_cap_profiles(
    pisa_profile: dict[str, Any], baseline_profile: dict[str, Any], key: str
) -> list[float]:
    pisa_caps = torch.tensor(pisa_profile["calibrated_caps_kw"], dtype=torch.float64)
    baseline_caps = torch.tensor(
        baseline_profile["calibrated_caps_kw"], dtype=torch.float64
    )
    if not torch.allclose(pisa_caps, baseline_caps, rtol=0.0, atol=1e-7):
        raise ValueError(
            f"PISA and Seq2Seq use different target caps for fraction {key}: "
            f"{pisa_caps.tolist()} vs {baseline_caps.tolist()}."
        )
    pisa_windows = pisa_profile.get("support_windows")
    baseline_windows = baseline_profile.get("support_windows")
    if pisa_windows != baseline_windows:
        raise ValueError(
            f"PISA and Seq2Seq support-window counts differ for {key}: "
            f"{pisa_windows} vs {baseline_windows}."
        )
    return [float(value) for value in pisa_caps]


@torch.inference_mode()
def evaluate_pair(
    pisa: nn.Module,
    seq2seq: nn.Module,
    loader: Any,
    device: torch.device,
    appliance_names: list[str],
    rated_caps_kw: list[float],
    use_amp: bool,
    max_batches: int | None,
) -> dict[str, Any]:
    pisa.to(device).eval()
    seq2seq.to(device).eval()
    residual_tcn = getattr(pisa, "refined_history_residual_tcn", None)
    if residual_tcn is None:
        raise ValueError(
            "PISA checkpoint has no residual history TCN, so the controlled "
            "oracle-history forecast cannot be evaluated."
        )

    pisa_history = HistoryAccumulator(appliance_names)
    seq_history = HistoryAccumulator(appliance_names)
    pisa_history_soft_gated = PowerAccumulator(appliance_names)
    pisa_history_hard_gated = PowerAccumulator(appliance_names)
    pisa_history_floor_gated = PowerAccumulator(appliance_names)
    pisa_future = PowerAccumulator(appliance_names)
    seq_future = PowerAccumulator(appliance_names)
    pisa_base_future = PowerAccumulator(appliance_names)
    pisa_oracle_future = PowerAccumulator(appliance_names)
    seq_oracle_future = PowerAccumulator(appliance_names)
    pisa_power_residual = MagnitudeAccumulator(appliance_names)
    pisa_gated_power_residual = MagnitudeAccumulator(appliance_names)
    pisa_state_residual = MagnitudeAccumulator(appliance_names)
    pisa_gated_state_residual = MagnitudeAccumulator(appliance_names)
    pisa_oracle_change = MagnitudeAccumulator(appliance_names)
    seq_oracle_change = MagnitudeAccumulator(appliance_names)
    last_power_gate: Tensor | None = None
    last_state_gate: Tensor | None = None
    last_base_power_blend: Tensor | None = None
    last_base_state_blend: Tensor | None = None
    last_target_power_scale: Tensor | None = None
    last_target_power_bias: Tensor | None = None

    oracle_inputs: dict[str, Tensor] = {}
    oracle_mode = False

    def replace_tcn_history(
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        del module
        if not oracle_mode:
            return None
        if "power" not in oracle_inputs or "state" not in oracle_inputs:
            raise RuntimeError("Oracle-history hook was invoked before inputs were set.")
        replacement = dict(kwargs)
        replacement["past_power"] = oracle_inputs["power"]
        replacement["past_p_on"] = oracle_inputs["state"]
        return args, replacement

    hook = residual_tcn.register_forward_pre_hook(
        replace_tcn_history,
        with_kwargs=True,
    )
    caps = torch.tensor(
        rated_caps_kw, device=device, dtype=torch.float32
    ).view(1, -1, 1)
    batches = 0
    try:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            device_batch = move_batch_to_device(batch, device)
            future_target = device_batch["y_power"]
            future_mask = device_batch.get("target_mask")
            history_target = device_batch["y_hist_power"]
            history_state = device_batch["y_hist_state"]
            history_mask = device_batch.get("hist_mask")
            oracle_inputs["power"] = torch.minimum(history_target, caps)
            oracle_inputs["state"] = history_state

            oracle_mode = False
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                pisa_out = pisa(device_batch)
                seq_out = seq2seq(device_batch)

            pisa_history.update_history(
                prediction_power=pisa_out["future_tcn_input_power"],
                actual_input_p_on=pisa_out["future_tcn_input_p_on"],
                raw_head_p_on=pisa_out["past_p_on"],
                target_power=history_target,
                target_state=history_state,
                mask=history_mask,
            )
            pisa_history_power = pisa_out["future_tcn_input_power"]
            pisa_history_p_on = pisa_out["future_tcn_input_p_on"].clamp(0.0, 1.0)
            pisa_history_soft_gated.update(
                pisa_history_power * pisa_history_p_on,
                history_target,
                history_mask,
            )
            pisa_history_hard_gated.update(
                pisa_history_power * (pisa_history_p_on >= 0.5).to(
                    pisa_history_power.dtype
                ),
                history_target,
                history_mask,
            )
            floor = float(getattr(pisa, "state_gate_floor", 0.01))
            pisa_history_floor_gated.update(
                pisa_history_power * (floor + (1.0 - floor) * pisa_history_p_on),
                history_target,
                history_mask,
            )
            seq_history.update_history(
                prediction_power=seq_out["nilm_past_power"],
                actual_input_p_on=seq_out["nilm_past_p_on"],
                raw_head_p_on=seq_out["past_p_on"],
                target_power=history_target,
                target_state=history_state,
                mask=history_mask,
            )
            pisa_future.update(pisa_out["y_power"], future_target, future_mask)
            seq_future.update(seq_out["y_power"], future_target, future_mask)
            pisa_base_future.update(
                pisa_out["base_y_power"], future_target, future_mask
            )
            power_residual = pisa_out["future_tcn_power_residual_raw"]
            state_residual = pisa_out["future_tcn_state_residual_logits"]
            power_gate = pisa_out["future_tcn_power_gate"]
            state_gate = pisa_out["future_tcn_state_gate"]
            pisa_power_residual.update_values(power_residual, future_mask)
            pisa_gated_power_residual.update_values(
                power_gate * power_residual, future_mask
            )
            pisa_state_residual.update_values(state_residual, future_mask)
            pisa_gated_state_residual.update_values(
                state_gate * state_residual, future_mask
            )
            last_power_gate = power_gate.detach().cpu().reshape(-1)
            last_state_gate = state_gate.detach().cpu().reshape(-1)
            if "future_tcn_base_power_blend" in pisa_out:
                last_base_power_blend = (
                    pisa_out["future_tcn_base_power_blend"]
                    .detach()
                    .cpu()
                    .mean(dim=(0, 2))
                )
                last_base_state_blend = (
                    pisa_out["future_tcn_base_state_blend"]
                    .detach()
                    .cpu()
                    .mean(dim=(0, 2))
                )
                last_target_power_scale = (
                    pisa_out["future_tcn_target_power_scale"]
                    .detach()
                    .cpu()
                    .reshape(-1)
                )
                last_target_power_bias = (
                    pisa_out["future_tcn_target_power_bias"]
                    .detach()
                    .cpu()
                    .reshape(-1)
                )

            oracle_mode = True
            oracle_batch = dict(device_batch)
            oracle_batch["appliance_history_power"] = oracle_inputs["power"]
            oracle_batch["appliance_history_state"] = oracle_inputs["state"]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                pisa_oracle_out = pisa(device_batch)
                seq_oracle_out = seq2seq.forecaster(oracle_batch)
            oracle_mode = False
            pisa_oracle_future.update(
                pisa_oracle_out["y_power"], future_target, future_mask
            )
            seq_oracle_future.update(
                seq_oracle_out["y_power"], future_target, future_mask
            )
            pisa_oracle_change.update_values(
                pisa_oracle_out["y_power"] - pisa_out["y_power"], future_mask
            )
            seq_oracle_change.update_values(
                seq_oracle_out["y_power"] - seq_out["y_power"], future_mask
            )
            batches += 1
    finally:
        hook.remove()

    if batches == 0:
        raise RuntimeError("No validation batches were evaluated.")

    history_pisa = pisa_history.compute()
    history_seq = seq_history.compute()
    history_pisa_soft = pisa_history_soft_gated.compute()
    history_pisa_hard = pisa_history_hard_gated.compute()
    history_pisa_floor = pisa_history_floor_gated.compute()
    future_pisa = pisa_future.compute()
    future_seq = seq_future.compute()
    base_future_pisa = pisa_base_future.compute()
    oracle_pisa = pisa_oracle_future.compute()
    oracle_seq = seq_oracle_future.compute()
    if last_power_gate is None or last_state_gate is None:
        raise RuntimeError("Residual-TCN gate values were not collected.")
    gate_values = {
        appliance: {
            "power_gate": float(last_power_gate[index]),
            "state_gate": float(last_state_gate[index]),
        }
        for index, appliance in enumerate(appliance_names)
    }
    enhanced_values = None
    if last_base_power_blend is not None:
        enhanced_values = {
            appliance: {
                "mean_base_power_blend": float(last_base_power_blend[index]),
                "mean_base_state_blend": float(last_base_state_blend[index]),
                "target_power_scale": float(last_target_power_scale[index]),
                "target_power_bias_kw": float(last_target_power_bias[index]),
            }
            for index, appliance in enumerate(appliance_names)
        }
    return {
        "evaluated_batches": batches,
        "question_A_history_reconstruction": {
            "definition": (
                "Historical power/state actually consumed by each method's "
                "future forecaster, evaluated against the same validation labels."
            ),
            "PISA": history_pisa,
            "Seq2Seq": history_seq,
            "PISA_counterfactual_state_conditioning": {
                "soft_power_times_p_on": history_pisa_soft,
                "hard_power_times_state_at_0.5": history_pisa_hard,
                "floor_gated_power": history_pisa_floor,
                "soft_vs_current_MAE_improvement_percent": percent_better(
                    history_pisa_soft["macro_avg"]["MAE_kW"],
                    history_pisa["macro_avg"]["MAE_kW"],
                ),
                "hard_vs_current_MAE_improvement_percent": percent_better(
                    history_pisa_hard["macro_avg"]["MAE_kW"],
                    history_pisa["macro_avg"]["MAE_kW"],
                ),
                "note": (
                    "Read-only counterfactual. It does not alter the checkpoint "
                    "or the residual TCN input used for normal forecasting."
                ),
            },
            "PISA_vs_Seq2Seq_MAE_improvement_percent": percent_better(
                history_pisa["macro_avg"]["MAE_kW"],
                history_seq["macro_avg"]["MAE_kW"],
            ),
        },
        "normal_future_forecast": {
            "PISA": future_pisa,
            "Seq2Seq": future_seq,
            "PISA_vs_Seq2Seq_MAE_improvement_percent": percent_better(
                future_pisa["macro_avg"]["MAE_kW"],
                future_seq["macro_avg"]["MAE_kW"],
            ),
        },
        "PISA_route_diagnostics": {
            "base_y_power": base_future_pisa,
            "final_vs_base_MAE_improvement_percent": percent_better(
                future_pisa["macro_avg"]["MAE_kW"],
                base_future_pisa["macro_avg"]["MAE_kW"],
            ),
            "residual_TCN_gates": gate_values,
            "legacy_residual_gates_active": (
                getattr(pisa, "future_residual_fusion", "gated_residual")
                == "gated_residual"
            ),
            "enhanced_direct_blend_and_calibration": enhanced_values,
            "power_residual_raw_magnitude": pisa_power_residual.compute(),
            "power_residual_after_gate_magnitude": (
                pisa_gated_power_residual.compute()
            ),
            "state_residual_logits_magnitude": pisa_state_residual.compute(),
            "state_residual_after_gate_magnitude": (
                pisa_gated_state_residual.compute()
            ),
        },
        "question_B_oracle_history_forecast": {
            "definition": (
                "Inference-time diagnostic only: both already-trained future "
                "forecasters receive the same ground-truth target-home history, "
                "clipped by the same support-calibrated rated-power caps."
            ),
            "PISA": oracle_pisa,
            "Seq2Seq": oracle_seq,
            "PISA_vs_Seq2Seq_MAE_improvement_percent": percent_better(
                oracle_pisa["macro_avg"]["MAE_kW"],
                oracle_seq["macro_avg"]["MAE_kW"],
            ),
            "caveat": (
                "The forecasters are not retrained with oracle history; this "
                "isolates inference sensitivity, not oracle-trained capacity."
            ),
            "normal_to_oracle_prediction_change": {
                "PISA": pisa_oracle_change.compute(),
                "Seq2Seq": seq_oracle_change.compute(),
            },
        },
        "structural_audit": {
            "PISA_history_power_state_conditioned_before_TCN": False,
            "Seq2Seq_history_power_state_conditioned_before_TCN": True,
            "PISA_residual_TCN_uses_future_calendar_covariates": bool(
                getattr(residual_tcn, "use_future_context", False)
            ),
            "Seq2Seq_forecaster_uses_future_calendar_covariates": True,
            "PISA_forecast_role": (
                "learned blend of frozen base and independent direct forecast"
                if getattr(pisa, "future_residual_fusion", "gated_residual")
                == "learned_blend"
                else "gated raw-space residual over frozen base route"
            ),
            "Seq2Seq_forecast_role": "direct absolute power/state forecast",
        },
    }


def print_result(key: str, result: dict[str, Any]) -> None:
    history = result["question_A_history_reconstruction"]
    normal = result["normal_future_forecast"]
    oracle = result["question_B_oracle_history_forecast"]
    route = result["PISA_route_diagnostics"]
    print("\n" + "=" * 88)
    print(f"Target validation, target data {float(key) * 100:.0f}%")
    print("=" * 88)
    print(
        "History MAE  : "
        f"PISA={history['PISA']['macro_avg']['MAE_kW']:.6f}, "
        f"Seq2Seq={history['Seq2Seq']['macro_avg']['MAE_kW']:.6f}, "
        f"PISA improvement={history['PISA_vs_Seq2Seq_MAE_improvement_percent']:+.2f}%"
    )
    print(
        "History F1   : "
        f"PISA={history['PISA']['macro_avg']['StateF1_actual_forecaster_input']:.6f}, "
        f"Seq2Seq={history['Seq2Seq']['macro_avg']['StateF1_actual_forecaster_input']:.6f}"
    )
    counterfactual = history["PISA_counterfactual_state_conditioning"]
    print(
        "History gates: "
        f"current={history['PISA']['macro_avg']['MAE_kW']:.6f}, "
        f"soft={counterfactual['soft_power_times_p_on']['macro_avg']['MAE_kW']:.6f}, "
        f"hard={counterfactual['hard_power_times_state_at_0.5']['macro_avg']['MAE_kW']:.6f}"
    )
    print(
        "Normal future: "
        f"PISA={normal['PISA']['macro_avg']['MAE_kW']:.6f}, "
        f"Seq2Seq={normal['Seq2Seq']['macro_avg']['MAE_kW']:.6f}, "
        f"PISA improvement={normal['PISA_vs_Seq2Seq_MAE_improvement_percent']:+.2f}%"
    )
    print(
        "Oracle future: "
        f"PISA={oracle['PISA']['macro_avg']['MAE_kW']:.6f}, "
        f"Seq2Seq={oracle['Seq2Seq']['macro_avg']['MAE_kW']:.6f}, "
        f"PISA improvement={oracle['PISA_vs_Seq2Seq_MAE_improvement_percent']:+.2f}%"
    )
    print(
        "PISA base/final: "
        f"base={route['base_y_power']['macro_avg']['MAE_kW']:.6f}, "
        f"final={normal['PISA']['macro_avg']['MAE_kW']:.6f}, "
        f"residual improvement={route['final_vs_base_MAE_improvement_percent']:+.2f}%"
    )
    print(
        "Oracle change : "
        f"PISA={oracle['normal_to_oracle_prediction_change']['PISA']['macro_avg']['mean_absolute']:.6f}, "
        f"Seq2Seq={oracle['normal_to_oracle_prediction_change']['Seq2Seq']['macro_avg']['mean_absolute']:.6f}"
    )
    enhanced = route.get("enhanced_direct_blend_and_calibration")
    if enhanced:
        print("Enhanced route (base blend / scale / bias kW):")
        for appliance, values in enhanced.items():
            print(
                f"  {appliance:<16} "
                f"{values['mean_base_power_blend']:.4f} / "
                f"{values['target_power_scale']:.4f} / "
                f"{values['target_power_bias_kw']:+.5f}"
            )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch_size must be >= 1 and num_workers must be >= 0.")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("--max_batches must be >= 1 when supplied.")

    pisa_run = resolved(args.pisa_run_dir)
    baseline_run = resolved(args.baseline_run_dir)
    pisa_transfer_config = read_json(pisa_run / "config.json")
    baseline_transfer_config = read_json(baseline_run / "config.json")
    pisa_summary = read_json(pisa_run / "results" / "transfer_summary.json")
    baseline_summary = read_json(
        baseline_run / "results" / "baseline_transfer_summary.json"
    )

    pisa_source_checkpoint = resolved(pisa_transfer_config["source_checkpoint"])
    pisa_source_config = read_json(
        source_pisa_config_path(pisa_transfer_config, pisa_source_checkpoint)
    )
    baseline_source_run = resolved(baseline_transfer_config["source_run_dir"])
    baseline_source_config = read_json(baseline_source_run / "config.json")
    (
        source_csv,
        target_csv,
        input_window,
        horizon,
        stride,
        input_cols,
        appliance_names,
    ) = validate_protocol(
        pisa_transfer_config,
        baseline_transfer_config,
        pisa_summary,
        baseline_summary,
        pisa_source_config,
        baseline_source_config,
    )
    thresholds = source_state_thresholds(
        pisa_source_checkpoint, baseline_source_config, appliance_names
    )

    _, source_bundle = build_single_home_datasets(
        csv_path=source_csv,
        input_cols=input_cols,
        appliance_cols=appliance_names,
        input_window=input_window,
        horizon=horizon,
        stride=stride,
        target_mode="future",
        state_thresholds=thresholds,
        drop_unavailable_windows=True,
    )
    target_datasets, _ = build_single_home_datasets(
        csv_path=target_csv,
        input_cols=input_cols,
        appliance_cols=appliance_names,
        input_window=input_window,
        horizon=horizon,
        stride=stride,
        target_mode="future",
        reference_bundle=source_bundle,
        state_thresholds=thresholds,
        drop_unavailable_windows=True,
    )
    if "val" not in target_datasets:
        raise ValueError("The target dataset has no validation split.")
    validation_loader = build_loaders(
        {"val": target_datasets["val"]}, args.batch_size, args.num_workers
    )["val"]

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_amp = device.type == "cuda" and not args.no_amp
    seq_method = baseline_summary["methods"].get("Seq2Seq-NILM -> TCN")
    if seq_method is None:
        raise KeyError("Baseline summary has no 'Seq2Seq-NILM -> TCN' method.")

    output: dict[str, Any] = {
        "evaluator_id": EVALUATOR_ID,
        "split": "validation",
        "selection_policy": (
            "Uses each transfer run's validation-selected checkpoint; no retraining."
        ),
        "source_csv": str(source_csv),
        "target_csv": str(target_csv),
        "input_window": input_window,
        "horizon": horizon,
        "stride": stride,
        "appliances": appliance_names,
        "state_thresholds_kw": thresholds,
        "max_batches": args.max_batches,
        "fractions": {},
    }

    for fraction in args.fractions:
        key = fraction_key(fraction)
        if key not in pisa_summary.get("few_shot", {}):
            raise KeyError(f"PISA summary has no few-shot fraction {key}.")
        if key not in seq_method.get("few_shot", {}):
            raise KeyError(f"Seq2Seq summary has no few-shot fraction {key}.")
        pisa_item = pisa_summary["few_shot"][key]
        seq_item = seq_method["few_shot"][key]
        pisa_model, pisa_checkpoint, pisa_caps = selected_pisa_model(
            pisa_run,
            pisa_transfer_config,
            pisa_source_config,
            pisa_item,
            key,
        )
        seq_model, seq_checkpoint, seq_caps = selected_seq2seq_model(
            baseline_run,
            baseline_source_config,
            seq_method,
            seq_item,
            key,
        )
        caps = check_cap_profiles(pisa_caps, seq_caps, key)
        result = evaluate_pair(
            pisa=pisa_model,
            seq2seq=seq_model,
            loader=validation_loader,
            device=device,
            appliance_names=appliance_names,
            rated_caps_kw=caps,
            use_amp=use_amp,
            max_batches=args.max_batches,
        )
        result["support_fraction"] = float(fraction)
        result["support_windows"] = int(pisa_caps["support_windows"])
        result["rated_power_caps_kw"] = caps
        result["PISA_selection"] = {
            "selected_model": pisa_item["selected_model"],
            "selected_epoch": pisa_item.get("selected_epoch", 0),
            "checkpoint": str(pisa_checkpoint),
            "stored_selected_validation_MAE_kW": pisa_item.get(
                "selected_validation_mae"
            ),
            "residual_tcn_history_source": getattr(
                pisa_model, "residual_tcn_history_source", None
            ),
        }
        result["Seq2Seq_selection"] = {
            "selected_model": seq_item["selected_model"],
            "selected_epoch": seq_item.get("selected_epoch", 0),
            "checkpoint": str(seq_checkpoint),
            "stored_selected_validation_MAE_kW": seq_item.get(
                "selected_validation_mae"
            ),
        }
        if args.max_batches is None:
            pisa_recomputed = result["normal_future_forecast"]["PISA"][
                "macro_avg"
            ]["MAE_kW"]
            seq_recomputed = result["normal_future_forecast"]["Seq2Seq"][
                "macro_avg"
            ]["MAE_kW"]
            pisa_stored = pisa_item.get("selected_validation_mae")
            seq_stored = seq_item.get("selected_validation_mae")
            result["PISA_selection"]["recomputed_minus_stored_MAE_kW"] = (
                None if pisa_stored is None else pisa_recomputed - float(pisa_stored)
            )
            result["Seq2Seq_selection"]["recomputed_minus_stored_MAE_kW"] = (
                None if seq_stored is None else seq_recomputed - float(seq_stored)
            )
            for label, recomputed, stored in (
                ("PISA", pisa_recomputed, pisa_stored),
                ("Seq2Seq", seq_recomputed, seq_stored),
            ):
                if stored is not None and abs(recomputed - float(stored)) > 5e-5:
                    print(
                        f"WARNING: {label} recomputed validation MAE differs "
                        f"from the saved summary for {key}: "
                        f"{recomputed:.9f} vs {float(stored):.9f}."
                    )
        output["fractions"][key] = result
        print_result(key, result)

        # Release one pair before constructing the next on memory-limited GPUs.
        del pisa_model, seq_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_path = (
        resolved(args.output)
        if args.output is not None
        else pisa_run / "results" / "history_forecast_diagnostic.json"
    )
    save_json(output_path, output)
    print("\nDiagnostic results saved to:")
    print(output_path)
    print(
        "Note: oracle-history results are inference-time diagnostics; the "
        "forecasters were not retrained on oracle histories."
    )


if __name__ == "__main__":
    main()
