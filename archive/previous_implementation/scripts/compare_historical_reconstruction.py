from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(ROOT))
sys.path.append(str(SRC))

from data import build_single_home_dataloaders  # noqa: E402
from models import (  # noqa: E402
    HistoricalNILMSeq2Seq,
    build_pisa_model_from_config,
    infer_optional_pisa_architecture,
)


DEFAULT_INPUT_COLS = [
    "grid",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
]
DEFAULT_APPLIANCES = [
    "air1",
    "refrigerator1",
    "dishwasher1",
    "microwave1",
]
DEFAULT_STATE_THRESHOLDS_KW = {
    "air1": 0.50,
    "refrigerator1": 0.05,
    "dishwasher1": 0.05,
    "microwave1": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare PISA and Seq2Seq-NILM historical appliance reconstruction "
            "on an identical split."
        )
    )
    parser.add_argument(
        "--pisa_checkpoints",
        nargs="+",
        type=Path,
        required=True,
        help="PISA best.pt files, normally one per seed.",
    )
    parser.add_argument(
        "--nilm_checkpoints",
        nargs="+",
        type=Path,
        required=True,
        help="Seq2Seq-NILM nilm_best.pt files in matching seed order.",
    )
    parser.add_argument(
        "--pisa_config",
        type=Path,
        default=ROOT / "configs" / "home7951_pisa_sparse_fix.yaml",
        help="YAML model/data config used by the PISA checkpoints.",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=(
            ROOT
            / "data"
            / "austin_2018_sep_4homes"
            / "home_7951_2018_sep_1min.csv"
        ),
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--input_window", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit for the publishable full-split result.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "historical_reconstruction_comparison.json",
    )
    return parser.parse_args()


def load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(handle)
        else:
            value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}, got {type(value).__name__}.")
    return value


def checkpoint_payload(path: Path) -> tuple[dict[str, Any], dict[str, Tensor]]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dictionary checkpoint: {path}")
    state_dict = payload.get("model_state_dict", payload)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint does not contain a model state_dict: {path}")
    return payload, state_dict


def load_pisa(path: Path, config: dict[str, Any]) -> tuple[nn.Module, dict[str, Any]]:
    payload, state_dict = checkpoint_payload(path)
    checkpoint_config = copy.deepcopy(config)
    model_config = checkpoint_config.setdefault("model", {})
    inferred = infer_optional_pisa_architecture(state_dict)
    inferred_horizon = inferred.pop("residual_tcn_horizon", None)
    model_config.update(inferred)
    if inferred_horizon is not None:
        configured_horizon = int(model_config.get("horizon", inferred_horizon))
        if configured_horizon != int(inferred_horizon):
            raise ValueError(
                f"Checkpoint residual-TCN horizon={inferred_horizon}, but "
                f"config horizon={configured_horizon}: {path}"
            )
    model = build_pisa_model_from_config(checkpoint_config)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"PISA checkpoint/config mismatch for {path}. Pass the exact YAML "
            "used to train this checkpoint with --pisa_config.\n{exc}"
        ) from exc
    return model, payload


def load_nilm(
    path: Path,
    input_dim: int,
    num_appliances: int,
    rated_power: Tensor,
) -> tuple[nn.Module, dict[str, Any]]:
    payload, state_dict = checkpoint_payload(path)
    saved_args = payload.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = {}
    hidden_dim = int(saved_args.get("hidden_dim", 128))
    num_layers = int(saved_args.get("nilm_num_layers", 2))
    dropout = float(saved_args.get("dropout", 0.1))
    model = HistoricalNILMSeq2Seq(
        input_dim=input_dim,
        num_appliances=num_appliances,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        rated_power=rated_power,
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Seq2Seq-NILM checkpoint architecture mismatch for {path}. "
            "The checkpoint should contain the original training args.\n{exc}"
        ) from exc
    return model, payload


def safe_f1(tp: float, fp: float, fn: float) -> float:
    denominator = 2.0 * tp + fp + fn
    return 2.0 * tp / denominator if denominator > 0.0 else float("nan")


class HistoricalMetricAccumulator:
    def __init__(self, appliance_names: list[str], thresholds_kw: Tensor) -> None:
        count = len(appliance_names)
        self.appliance_names = appliance_names
        self.thresholds_kw = thresholds_kw.view(1, count, 1).cpu()
        self.abs_error = torch.zeros(count, dtype=torch.float64)
        self.valid_count = torch.zeros(count, dtype=torch.float64)
        self.head_tp = torch.zeros(count, dtype=torch.float64)
        self.head_fp = torch.zeros(count, dtype=torch.float64)
        self.head_fn = torch.zeros(count, dtype=torch.float64)
        self.power_tp = torch.zeros(count, dtype=torch.float64)
        self.power_fp = torch.zeros(count, dtype=torch.float64)
        self.power_fn = torch.zeros(count, dtype=torch.float64)

    @staticmethod
    def _confusion(
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        prediction = prediction.bool()
        target = target.bool()
        mask = mask.bool()
        dims = (0, 2)
        tp = (prediction & target & mask).sum(dim=dims).double()
        fp = (prediction & ~target & mask).sum(dim=dims).double()
        fn = (~prediction & target & mask).sum(dim=dims).double()
        return tp, fp, fn

    def update(
        self,
        pred_power: Tensor,
        pred_p_on: Tensor,
        target_power: Tensor,
        target_state: Tensor,
        mask: Tensor | None,
    ) -> None:
        pred_power = pred_power.detach().float().cpu()
        pred_p_on = pred_p_on.detach().float().cpu()
        target_power = target_power.detach().float().cpu()
        target_state = target_state.detach().float().cpu()
        if mask is None:
            mask = torch.ones_like(target_power, dtype=torch.bool)
        else:
            mask = mask.detach().cpu() > 0.5

        self.abs_error += (
            (pred_power - target_power).abs() * mask.to(pred_power.dtype)
        ).sum(dim=(0, 2)).double()
        self.valid_count += mask.sum(dim=(0, 2)).double()

        head_values = self._confusion(pred_p_on >= 0.5, target_state >= 0.5, mask)
        power_values = self._confusion(
            pred_power >= self.thresholds_kw,
            target_state >= 0.5,
            mask,
        )
        self.head_tp += head_values[0]
        self.head_fp += head_values[1]
        self.head_fn += head_values[2]
        self.power_tp += power_values[0]
        self.power_fp += power_values[1]
        self.power_fn += power_values[2]

    def compute(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for index, name in enumerate(self.appliance_names):
            count = float(self.valid_count[index].item())
            result[name] = {
                "MAE_kW": float(self.abs_error[index].item() / max(count, 1.0)),
                "StateF1_head_at_0.5": safe_f1(
                    float(self.head_tp[index]),
                    float(self.head_fp[index]),
                    float(self.head_fn[index]),
                ),
                "StateF1_power_threshold": safe_f1(
                    float(self.power_tp[index]),
                    float(self.power_fp[index]),
                    float(self.power_fn[index]),
                ),
                "valid_points": count,
            }
        return result


@torch.inference_mode()
def evaluate_history(
    model: nn.Module,
    model_kind: str,
    loader: Any,
    device: torch.device,
    appliance_names: list[str],
    thresholds_kw: Tensor,
    max_batches: int | None,
) -> dict[str, Any]:
    model.to(device).eval()
    metrics = HistoricalMetricAccumulator(appliance_names, thresholds_kw)
    auxiliary_abs_error = torch.zeros(len(appliance_names), dtype=torch.float64)
    auxiliary_count = torch.zeros(len(appliance_names), dtype=torch.float64)

    batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        model_batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        out = model(model_batch)
        target_power = batch["y_hist_power"]
        target_state = batch["y_hist_state"]
        mask = batch.get("hist_mask")

        if model_kind == "pisa":
            pred_power = out.get("past_reconstructed_power")
            if pred_power is None:
                pred_power = out["past_power"] * out["past_p_on"]
            # PISA's auxiliary reconstruction loss supervises ungated past_power.
            # Report it separately; the primary MAE uses deployment-available
            # reconstructed power after the rated-power cap.
            auxiliary = out["past_power"].detach().float().cpu()
            target_cpu = target_power.float().cpu()
            aux_mask = (
                torch.ones_like(target_cpu, dtype=torch.bool)
                if mask is None
                else mask.cpu() > 0.5
            )
            auxiliary_abs_error += (
                (auxiliary - target_cpu).abs() * aux_mask.to(auxiliary.dtype)
            ).sum(dim=(0, 2)).double()
            auxiliary_count += aux_mask.sum(dim=(0, 2)).double()
        elif model_kind == "nilm":
            pred_power = out["past_power"]
        else:
            raise ValueError(f"Unknown model kind: {model_kind}")

        metrics.update(
            pred_power=pred_power,
            pred_p_on=out.get("past_reconstructed_p_on", out["past_p_on"]),
            target_power=target_power,
            target_state=target_state,
            mask=mask,
        )
        batches += 1

    if batches == 0:
        raise RuntimeError("No evaluation batches were processed.")

    result: dict[str, Any] = {
        "metrics": metrics.compute(),
        "evaluated_batches": batches,
        "power_definition": (
            "min(past_power, rated_power); past_p_on is kept separate"
            if model_kind == "pisa"
            else "Seq2Seq-NILM past_power (already state-gated and rated-power capped)"
        ),
    }
    if model_kind == "pisa":
        result["auxiliary_ungated_MAE_kW"] = {
            name: float(auxiliary_abs_error[index] / max(float(auxiliary_count[index]), 1.0))
            for index, name in enumerate(appliance_names)
        }
    return result


def checkpoint_seed(payload: dict[str, Any], fallback: int) -> int | str:
    args = payload.get("args")
    if isinstance(args, dict) and "seed" in args:
        return int(args["seed"])
    config = payload.get("config")
    if isinstance(config, dict) and "seed" in config:
        return int(config["seed"])
    return fallback


def mean_std(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    valid = tensor[torch.isfinite(tensor)]
    if valid.numel() == 0:
        return {"mean": float("nan"), "std": float("nan")}
    std = valid.std(unbiased=True) if valid.numel() > 1 else valid.new_tensor(0.0)
    return {"mean": float(valid.mean()), "std": float(std)}


def summarize_runs(runs: list[dict[str, Any]], appliance_names: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in appliance_names:
        summary[name] = {}
        for metric in ("MAE_kW", "StateF1_head_at_0.5", "StateF1_power_threshold"):
            values = [float(run["metrics"][name][metric]) for run in runs]
            summary[name][metric] = {**mean_std(values), "values": values}
    return summary


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    if len(args.pisa_checkpoints) != len(args.nilm_checkpoints):
        raise ValueError(
            "--pisa_checkpoints and --nilm_checkpoints must contain the same "
            "number of seed-matched checkpoints."
        )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    config = load_mapping(args.pisa_config)
    data_config = config.setdefault("data", {})
    model_config = config.setdefault("model", {})
    appliance_names = list(data_config.get("appliance_cols", DEFAULT_APPLIANCES))
    input_cols = list(data_config.get("input_cols", DEFAULT_INPUT_COLS))
    thresholds = {
        name: float(DEFAULT_STATE_THRESHOLDS_KW[name]) for name in appliance_names
    }
    rated_power = torch.tensor(data_config["rated_power_kw"], dtype=torch.float32)
    model_config["input_dim"] = len(input_cols)
    model_config["num_appliances"] = len(appliance_names)
    model_config["horizon"] = args.horizon

    loaders, bundle = build_single_home_dataloaders(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_train=False,
        pin_memory=(device.type == "cuda"),
        input_window=args.input_window,
        horizon=args.horizon,
        stride=args.stride,
        input_cols=input_cols,
        appliance_cols=appliance_names,
        state_thresholds=thresholds,
    )
    if args.split not in loaders:
        raise KeyError(f"Split {args.split!r} is not present in {args.csv_path}.")
    threshold_tensor = torch.tensor(
        [bundle.state_thresholds[name] for name in appliance_names],
        dtype=torch.float32,
    )

    pisa_runs: list[dict[str, Any]] = []
    nilm_runs: list[dict[str, Any]] = []
    for run_index, (pisa_path, nilm_path) in enumerate(
        zip(args.pisa_checkpoints, args.nilm_checkpoints, strict=True)
    ):
        pisa, pisa_payload = load_pisa(pisa_path.resolve(), config)
        nilm, nilm_payload = load_nilm(
            nilm_path.resolve(),
            input_dim=len(input_cols),
            num_appliances=len(appliance_names),
            rated_power=rated_power,
        )
        pisa_result = evaluate_history(
            pisa,
            "pisa",
            loaders[args.split],
            device,
            appliance_names,
            threshold_tensor,
            args.max_batches,
        )
        nilm_result = evaluate_history(
            nilm,
            "nilm",
            loaders[args.split],
            device,
            appliance_names,
            threshold_tensor,
            args.max_batches,
        )
        pisa_result.update(
            checkpoint=str(pisa_path.resolve()),
            seed=checkpoint_seed(pisa_payload, run_index),
        )
        nilm_result.update(
            checkpoint=str(nilm_path.resolve()),
            seed=checkpoint_seed(nilm_payload, run_index),
        )
        pisa_runs.append(pisa_result)
        nilm_runs.append(nilm_result)
        del pisa, nilm

    focused_appliances = [
        name for name in ("air1", "refrigerator1") if name in appliance_names
    ]
    report = {
        "protocol": {
            "csv_path": str(args.csv_path.resolve()),
            "split": args.split,
            "input_window": args.input_window,
            "horizon": args.horizon,
            "stride": args.stride,
            "max_batches": args.max_batches,
            "state_head_threshold": 0.5,
            "power_state_thresholds_kw": bundle.state_thresholds,
            "note": (
                "MAE_kW uses deployment-available reconstructed history. "
                "StateF1_head_at_0.5 evaluates each model's state head; "
                "StateF1_power_threshold evaluates state inferred from reconstructed power."
            ),
        },
        "PISA": {
            "runs": pisa_runs,
            "summary": summarize_runs(pisa_runs, focused_appliances),
        },
        "Seq2Seq-NILM": {
            "runs": nilm_runs,
            "summary": summarize_runs(nilm_runs, focused_appliances),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(report), handle, indent=2, ensure_ascii=False)

    print(json.dumps(json_safe({
        "PISA": report["PISA"]["summary"],
        "Seq2Seq-NILM": report["Seq2Seq-NILM"]["summary"],
        "output": str(args.output.resolve()),
    }), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
