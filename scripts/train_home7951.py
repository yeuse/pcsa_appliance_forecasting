from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler

# ---------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.append(str(ROOT))
sys.path.append(str(SRC_DIR))


from data import (  # noqa: E402
    build_multi_home_dataloaders,
    build_single_home_dataloaders,
)
from data.pecanstreet_dataset import summarize_on_support  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    MetricsAccumulator,
    find_best_event_thresholds,
    find_best_hierarchical_start_thresholds,
    find_best_state_thresholds,
)
from models import AggregateOnlySeq2SeqBaseline, PISAModel  # noqa: E402
from models.pisa import (  # noqa: E402
    DEFAULT_APPLIANCE_TYPES,
    infer_optional_pisa_architecture,
    rated_power_tensor_for_appliances,
)
from training import (  # noqa: E402
    PISALoss,
    PISALossConfig,
    PISATrainer,
    TrainerConfig,
    move_batch_to_device,
    set_seed,
)


# ---------------------------------------------------------------------
# Default experiment setting
# ---------------------------------------------------------------------
DEFAULT_INPUT_COLS = [
    "grid",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
]

DEFAULT_APPLIANCE_COLS = [
    "air1",
    "refrigerator1",
    "dishwasher1",
    "microwave1",
]

# The four-home September protocol uses the defaults above. Water heating
# remains an explicitly supported legacy target for older single-home runs.
SUPPORTED_APPLIANCE_COLS = [*DEFAULT_APPLIANCE_COLS, "waterheater1"]


# These objectives describe future events/risk rather than the core NILM
# reconstruction task.  Base training deliberately suppresses them; the risk
# fine-tuning phase restores their requested weights with a warm-up/ramp.
RISK_LOSS_ARGUMENTS = (
    "lambda_event",
    "lambda_start",
    "lambda_stop",
    "lambda_bucket_start",
    "lambda_bucket_stop",
    "lambda_event_offset",
    "lambda_window_start",
    "lambda_conditional_bucket",
    "lambda_conditional_offset",
    "lambda_conditional_power",
    "lambda_pulse_duration",
    "lambda_pulse_amplitude",
)

CORE_MAE_MONITOR = "val/regression/macro_avg/MAE"
BASELINE_ZERO_MAE = "baseline_zero/regression/macro_avg/MAE"
# This diagnostic reference uses measured appliance history, which is
# unavailable to the deployed aggregate-only forecasting route.
PERSISTENCE_ORACLE_MAE = "baseline_persistence/regression/macro_avg/MAE"
# The risk stage is a one-start-per-window forecasting task.  Selecting it by
# the raw bidirectional event head rewards a different, noisier decision rule.
RISK_HIERARCHICAL_START_MONITOR = (
    "val/hierarchical_start_tolerant/macro_avg/EventF1"
)


def find_default_csv() -> Path:
    """
    Find default single-home CSV file.
    """
    candidates = [
        ROOT
        / "data"
        / "austin_2018_sep_4homes"
        / "home_7951_2018_sep_1min.csv",
        ROOT / "data" / "home_7951_2018_jun_aug_1min.csv",
        ROOT / "src" / "data" / "home_7951_2018_jun_aug_1min.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Cannot find a supported Home 7951 one-minute CSV. "
        f"Checked: {[str(p) for p in candidates]}"
    )


def json_safe(obj: Any) -> Any:
    """
    Convert tensors / numpy values / Path / dataclass / custom objects
    into JSON-safe values.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return json_safe(asdict(obj))

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]

    if isinstance(obj, set):
        return [json_safe(v) for v in obj]

    if isinstance(obj, Path):
        return str(obj)

    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return json_safe(obj.item())
        return obj.detach().cpu().tolist()

    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass

    if hasattr(obj, "__dict__"):
        return json_safe(vars(obj))

    return obj


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            json_safe(obj),
            f,
            indent=2,
            ensure_ascii=False,
        )


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def load_fixed_state_thresholds(
    checkpoint_path: str | Path,
    appliance_names: list[str],
    explicit_path: str | None = None,
) -> tuple[dict[str, float], Path]:
    """Load the MAE-safe state calibration for risk-checkpoint evaluation."""
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
    else:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        path = checkpoint.parent.parent / "results" / "calibrated_state_thresholds.json"
    if not path.exists():
        raise FileNotFoundError(
            "Risk-checkpoint evaluation needs the MAE-safe state calibration. "
            f"Expected: {path}"
        )

    calibration = load_json(path)
    thresholds = calibration.get("state_thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError(
            "State calibration does not contain a 'state_thresholds' mapping: "
            f"{path}"
        )

    result: dict[str, float] = {}
    for name in appliance_names:
        value = thresholds.get(name)
        if not isinstance(value, (int, float)) or not bool(
            torch.isfinite(torch.tensor(float(value)))
        ):
            raise ValueError(
                f"Invalid fixed state threshold for '{name}' in {path}."
            )
        result[name] = float(value)
    return result, path

class DeviceStratifiedBatchSampler(Sampler[list[int]]):
    """Keep rare-appliance start windows represented in every training batch.

    A single weighted sampler can repeatedly favour the most common sparse
    appliance.  This sampler reserves a configurable fraction of each batch
    for positive future-start windows and distributes those slots equally
    across the selected appliances.  Sampling is with replacement only for
    the rare positive strata; the remaining slots follow the natural training
    window distribution.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        appliance_indices: list[int],
        positive_fraction: float = 0.50,
        seed: int = 42,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if not 0.0 <= float(positive_fraction) < 1.0:
            raise ValueError("positive_fraction must be in [0, 1).")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.positive_fraction = float(positive_fraction)
        self.seed = int(seed)
        self._epoch = 0
        self.num_batches = len(dataset) // self.batch_size
        if self.num_batches < 1:
            raise ValueError("dataset must contain at least one complete batch.")

        valid_indices = sorted(
            {
                int(idx)
                for idx in appliance_indices
                if 0 <= int(idx)
            }
        )
        self.positive_indices: dict[int, torch.Tensor] = {}
        for app_idx in valid_indices:
            positives: list[int] = []
            for sample_idx in range(len(dataset)):
                sample = dataset[sample_idx]
                target = sample.get("y_start", sample["y_event"])
                if app_idx < int(target.shape[0]) and bool(
                    target[app_idx].amax().item() > 0.5
                ):
                    positives.append(sample_idx)
            if positives:
                self.positive_indices[app_idx] = torch.tensor(
                    positives,
                    dtype=torch.long,
                )

        if not self.positive_indices:
            raise ValueError(
                "No positive start windows were found for the requested "
                "device-stratified sampler."
            )

        requested_slots = int(round(self.batch_size * self.positive_fraction))
        requested_slots = min(self.batch_size - 1, requested_slots)
        requested_slots = max(requested_slots, len(self.positive_indices))
        self.positive_slots = requested_slots
        self.appliance_indices = list(self.positive_indices)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self._epoch)
        self._epoch += 1

        num_devices = len(self.appliance_indices)
        base_quota, remainder = divmod(self.positive_slots, num_devices)
        for _ in range(self.num_batches):
            batch_indices: list[torch.Tensor] = []
            for position, app_idx in enumerate(self.appliance_indices):
                quota = base_quota + int(position < remainder)
                choices = self.positive_indices[app_idx]
                draw = torch.randint(
                    high=choices.numel(),
                    size=(quota,),
                    generator=generator,
                )
                batch_indices.append(choices[draw])

            background_slots = self.batch_size - self.positive_slots
            if background_slots > 0:
                batch_indices.append(
                    torch.randint(
                        high=len(self.dataset),
                        size=(background_slots,),
                        generator=generator,
                    )
                )
            batch = torch.cat(batch_indices)
            order = torch.randperm(batch.numel(), generator=generator)
            yield batch[order].tolist()


def build_device_stratified_train_loader(
    train_dataset,
    batch_size: int,
    num_workers: int,
    sparse_indices: list[int],
    positive_fraction: float,
    seed: int,
) -> DataLoader:
    """Build an appliance-balanced sparse-start batch loader."""
    sampler = DeviceStratifiedBatchSampler(
        dataset=train_dataset,
        batch_size=batch_size,
        appliance_indices=sparse_indices,
        positive_fraction=positive_fraction,
        seed=seed,
    )
    print("Device-stratified sampler built.")
    print(f"positive fraction per batch: {sampler.positive_fraction:.2f}")
    print(f"positive slots per batch   : {sampler.positive_slots}")
    print(
        "positive windows by appliance: "
        + str(
            {
                int(app_idx): int(indices.numel())
                for app_idx, indices in sampler.positive_indices.items()
            }
        )
    )
    return DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_active_weighted_train_loader(
    train_dataset,
    batch_size: int,
    num_workers: int,
    sparse_indices: list[int] | None = None,
    sparse_active_weight: float = 8.0,
    sparse_event_weight: float = 20.0,
    any_event_weight: float = 2.0,
):
    """
    Build weighted sampler for sparse appliances.

    sparse_indices:
        [2, 3] means dishwasher1 and waterheater1.
    """
    weights = []

    print("Building active weighted sampler...")

    for i in range(len(train_dataset)):
        sample = train_dataset[i]

        y_state = sample["y_state"]
        y_event = sample["y_event"]

        valid_sparse_indices = [
            idx
            for idx in (sparse_indices or [])
            if 0 <= idx < int(y_state.shape[0])
        ]

        if valid_sparse_indices:
            sparse_active = y_state[valid_sparse_indices].max().item()
            sparse_event = y_event[valid_sparse_indices].max().item()
        else:
            sparse_active = 0.0
            sparse_event = 0.0

        any_event = y_event.max().item()

        w = (
            1.0
            + sparse_active_weight * sparse_active
            + sparse_event_weight * sparse_event
            + any_event_weight * any_event
        )

        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print("Weighted sampler built.")
    print("weight min :", float(weights.min()))
    print("weight max :", float(weights.max()))
    print("weight mean:", float(weights.mean()))

    return loader


@torch.no_grad()
def collect_metrics_accumulator(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    appliance_names: list[str],
) -> MetricsAccumulator:
    model.eval()
    metrics_acc = MetricsAccumulator(appliance_names=appliance_names)

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            out = model(batch)
        metrics_acc.update(out, batch)

    return metrics_acc


def calibrate_risk_event_thresholds(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    appliance_names: list[str],
    num_thresholds: int,
    min_threshold: float,
    max_threshold: float,
    hierarchical_tolerance_minutes: int,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, Any],
]:
    """Calibrate risk-head thresholds using a selected risk checkpoint only."""
    event_thresholds = {app_name: 0.5 for app_name in appliance_names}
    bucket_thresholds = {app_name: 0.10 for app_name in appliance_names}
    window_thresholds = {app_name: 0.5 for app_name in appliance_names}
    hierarchical_start_thresholds = {
        app_name: 0.5 for app_name in appliance_names
    }
    calibration: dict[str, Any] = {
        "enabled": False,
        "event_thresholds": event_thresholds,
        "bucket_event_thresholds": bucket_thresholds,
        "window_start_thresholds": window_thresholds,
        "hierarchical_start_thresholds": hierarchical_start_thresholds,
    }

    val_acc = collect_metrics_accumulator(
        model=model,
        loader=loader,
        device=device,
        use_amp=use_amp,
        appliance_names=appliance_names,
    )
    val_true_event = torch.cat(val_acc.true_event_list, dim=0)
    val_true_start = (
        torch.cat(val_acc.true_start_list, dim=0)
        if val_acc.true_start_list
        else val_true_event
    )
    val_mask = torch.cat(val_acc.mask_list, dim=0) if val_acc.mask_list else None

    if val_acc.event_logits_list:
        event_thresholds, event_summary = find_best_event_thresholds(
            event_logits=torch.cat(val_acc.event_logits_list, dim=0),
            true_event=val_true_event,
            appliance_names=appliance_names,
            mask=val_mask,
            num_thresholds=num_thresholds,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
        )
        calibration["event_thresholds"] = event_thresholds
        calibration["event_summary"] = event_summary

    if val_acc.bucket_event_logits_list:
        bucket_thresholds, bucket_summary = find_best_event_thresholds(
            event_logits=torch.cat(val_acc.bucket_event_logits_list, dim=0),
            true_event=val_true_event,
            appliance_names=appliance_names,
            mask=val_mask,
            num_thresholds=num_thresholds,
            min_threshold=0.01,
            max_threshold=max_threshold,
        )
        calibration["bucket_event_thresholds"] = bucket_thresholds
        calibration["bucket_event_summary"] = bucket_summary

    if val_acc.window_start_prob_list:
        window_prob = torch.cat(val_acc.window_start_prob_list, dim=0).clamp(
            1e-5,
            1.0 - 1e-5,
        )
        window_logits = torch.logit(window_prob).unsqueeze(-1)
        window_target = (val_true_start.amax(dim=-1, keepdim=True) > 0.5).float()
        window_thresholds, window_summary = find_best_event_thresholds(
            event_logits=window_logits,
            true_event=window_target,
            appliance_names=appliance_names,
            mask=None,
            num_thresholds=num_thresholds,
            min_threshold=0.01,
            max_threshold=max_threshold,
        )
        calibration["window_start_thresholds"] = window_thresholds
        calibration["window_start_summary"] = window_summary

    if (
        val_acc.window_start_prob_list
        and val_acc.conditional_start_prob_list
    ):
        (
            hierarchical_start_thresholds,
            hierarchical_start_summary,
        ) = find_best_hierarchical_start_thresholds(
            window_start_prob=torch.cat(val_acc.window_start_prob_list, dim=0),
            conditional_start_prob=torch.cat(
                val_acc.conditional_start_prob_list,
                dim=0,
            ),
            true_start=val_true_start,
            appliance_names=appliance_names,
            mask=val_mask,
            num_thresholds=num_thresholds,
            min_threshold=0.01,
            max_threshold=max_threshold,
            tolerance=hierarchical_tolerance_minutes,
        )
        calibration["hierarchical_start_thresholds"] = (
            hierarchical_start_thresholds
        )
        calibration["hierarchical_start_summary"] = hierarchical_start_summary
        calibration["hierarchical_start_selection_tolerance_minutes"] = int(
            hierarchical_tolerance_minutes
        )

    calibration["enabled"] = True
    calibration["event_num_thresholds"] = num_thresholds
    calibration["event_min_threshold"] = min_threshold
    calibration["event_max_threshold"] = max_threshold
    return (
        event_thresholds,
        bucket_thresholds,
        window_thresholds,
        hierarchical_start_thresholds,
        calibration,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PISA on home 7951 Austin 1-min data."
    )

    # Data
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument(
        "--source_csv_paths",
        nargs="+",
        default=None,
        help=(
            "Two or more source-home CSVs for balanced multi-home training. "
            "Source scalers are fitted only on their combined train splits."
        ),
    )
    parser.add_argument(
        "--held_out_csv_path",
        type=str,
        default=None,
        help=(
            "Optional unseen target-home CSV. It is never used for fitting or "
            "checkpoint selection; its test split is evaluated once afterward."
        ),
    )
    parser.add_argument(
        "--home_balanced_sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use equal source-home quotas in every multi-home training batch.",
    )
    parser.add_argument(
        "--appliances",
        nargs="+",
        default=None,
        help=(
            "Appliance columns to train, e.g. --appliances air1 or "
            "--appliances dishwasher1 microwave1. Default uses the four-home "
            "protocol targets: air1 refrigerator1 dishwasher1 microwave1."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1024,
        help="Batch size for held-out-home evaluation.",
    )
    parser.add_argument(
        "--rated_power_kw",
        nargs="+",
        type=float,
        default=None,
        help=(
            "One finite rated-power cap (kW) per selected appliance, in the "
            "same order as --appliances. Defaults to the documented Home 7951 caps."
        ),
    )
    parser.add_argument("--input_window", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument(
        "--target_mode",
        type=str,
        default="future",
        choices=["future", "history_tail"],
        help=(
            "future predicts the next horizon; history_tail disaggregates the "
            "last horizon inside the input window."
        ),
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--appliance_scale_mode", default="p99_positive",
        choices=["p99_positive", "p99_on", "none"],
        help="Source-train-only power loss scale; p99_on excludes standby readings.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--sampler_mode",
        type=str,
        default="weighted",
        choices=["weighted", "device_stratified", "none"],
        help=(
            "weighted oversamples active/event windows; device_stratified reserves "
            "positive future-start slots per sparse appliance; none uses normal "
            "shuffled training."
        ),
    )
    parser.add_argument(
        "--stratified_positive_fraction",
        type=float,
        default=0.50,
        help=(
            "For device_stratified sampling, fraction of every batch reserved "
            "for equally balanced sparse-appliance positive start windows."
        ),
    )

    # Model
    parser.add_argument(
        "--model_type",
        type=str,
        default="pisa",
        choices=["pisa", "aggregate_seq2seq"],
        help="Train the full PISA model or an aggregate-only seq2seq baseline.",
    )
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--encoder_layers", type=int, default=3)
    parser.add_argument(
        "--use_bridge_residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fuse the appliance-query bridge power directly into the decoder "
            "forecast. Use --no-use_bridge_residual only for a documented "
            "PISA component ablation."
        ),
    )
    parser.add_argument(
        "--state_conditioned_power",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use state_prob * amplitude-style final power. This separates "
            "future ON/OFF prediction from power magnitude prediction."
        ),
    )
    parser.add_argument(
        "--state_gate_floor",
        type=float,
        default=0.05,
        help=(
            "Lower bound of the soft state gate. 0 is strict p_on gating; "
            "small positive values keep gradients alive when p_on is low."
        ),
    )
    parser.add_argument(
        "--state_power_blend",
        type=float,
        default=1.0,
        help=(
            "Blend between ungated amplitude power and state-conditioned power. "
            "1.0 uses fully state-conditioned power."
        ),
    )
    parser.add_argument(
        "--disable_rated_power_cap",
        action="store_true",
        help=(
            "Disable PISA's explicit rated-power upper clamp by replacing "
            "the documented caps with a very large finite cap. Intended only "
            "for a component ablation, never for the main physical model."
        ),
    )
    parser.add_argument(
        "--hierarchical_future",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use window-start risk times conditional power for sparse forecasting.",
    )
    parser.add_argument(
        "--risk_adapter_dim",
        type=int,
        default=0,
        help=(
            "Bottleneck width of the risk-only residual adapter. 0 disables it; "
            "a positive value adapts hierarchical/event heads without changing "
            "the base power route."
        ),
    )
    parser.add_argument(
        "--power_adapter_dim",
        type=int,
        default=0,
        help=(
            "Bottleneck width of a per-appliance residual adapter in the core "
            "power route. Keep 0 for normal source-home base training."
        ),
    )
    parser.add_argument(
        "--use_history_recon_refiner",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable a deployment-only residual history refiner. It does not "
            "enter the existing PISA future decoder or risk paths."
        ),
    )
    parser.add_argument(
        "--history_recon_gate_init",
        type=float,
        default=-3.0,
        help="Initial per-appliance logit for the history residual gates.",
    )
    parser.add_argument(
        "--use_refined_history_residual_tcn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the causal future residual-TCN. This legacy option name is "
            "kept for checkpoint and command compatibility; the TCN may consume "
            "either base or refined histories."
        ),
    )
    parser.add_argument(
        "--use_residual_tcn",
        dest="use_refined_history_residual_tcn",
        action="store_true",
        help=(
            "Preferred alias for --use_refined_history_residual_tcn. With "
            "--residual_tcn_history_source base, no history refiner is required."
        ),
    )
    parser.add_argument("--residual_tcn_hidden_dim", type=int, default=128)
    parser.add_argument("--residual_tcn_num_layers", type=int, default=4)
    parser.add_argument("--residual_tcn_kernel_size", type=int, default=3)
    parser.add_argument("--residual_tcn_dropout", type=float, default=0.1)
    parser.add_argument(
        "--residual_tcn_history_source",
        choices=("base", "refined"),
        default="refined",
        help="Use base or history-refiner appliance histories as TCN inputs.",
    )
    parser.add_argument(
        "--residual_tcn_shared",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Share one causal TCN across appliances instead of independent TCNs.",
    )
    parser.add_argument(
        "--future_residual_fusion",
        choices=("gated_residual", "direct", "learned_blend"),
        default="gated_residual",
        help=(
            "Fuse with the PISA anchor, replace it, or learn a per-appliance/"
            "horizon blend with an independent direct forecast."
        ),
    )
    parser.add_argument(
        "--residual_tcn_use_future_context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add future calendar covariates and learned horizon-step embeddings "
            "to the residual-TCN decoder."
        ),
    )
    parser.add_argument(
        "--residual_tcn_target_adapter_dim",
        type=int,
        default=0,
        help=(
            "Bottleneck width of the small target-domain adapter applied only "
            "to the frozen Transformer context used by the enhanced TCN head."
        ),
    )
    parser.add_argument(
        "--future_base_blend_init",
        type=float,
        default=2.0,
        help="Initial logit weight assigned to the frozen base forecast.",
    )
    parser.add_argument(
        "--target_output_calibration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable per-appliance target-domain power scale and bias.",
    )
    parser.add_argument(
        "--future_residual_zero_init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zero-initialize TCN output heads; recommended for gated residual fusion.",
    )
    parser.add_argument(
        "--future_residual_gate_init",
        type=float,
        default=-3.0,
        help="Initial logit for future power/state residual fusion gates.",
    )
    parser.add_argument(
        "--conditional_peak_multiplier",
        type=float,
        default=1.0,
        help="Multiplier for the training-set positive-power p95 template cap.",
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint to initialize model weights from.",
    )
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        help=(
            "Skip fitting and evaluate --init_checkpoint only. For a risk_best "
            "checkpoint, also pass --risk_state_thresholds_path from the MAE-safe "
            "run whose state thresholds must be retained."
        ),
    )
    parser.add_argument(
        "--risk_state_thresholds_path",
        type=str,
        default=None,
        help=(
            "Optional calibrated_state_thresholds.json used as the fixed base "
            "state thresholds during risk fine-tuning or checkpoint evaluation."
        ),
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="full",
        choices=[
            "full",
            "decoder",
            "decoder_heads",
            "bucket_event_heads",
            "hierarchical_future_heads",
            "risk_heads",
            "history_reconstruction",
            "future_residual_tcn",
        ],
        help=(
            "full trains all parameters; decoder freezes encoder/bridge and "
            "trains the whole decoder; decoder_heads trains appliance heads; "
            "bucket_event_heads trains only the coarse event heads; risk_heads "
            "trains only hierarchical/event heads while preserving base power; "
            "history_reconstruction trains only the optional deployment history "
            "refiner; future_residual_tcn trains only the independent per-appliance "
            "causal TCN residual branch."
        ),
    )
    parser.add_argument(
        "--training_stage",
        type=str,
        default="base",
        choices=["base", "risk_finetune", "joint"],
        help=(
            "base trains only core NILM objectives; risk_finetune initializes "
            "from a calibrated base run and enables sparse risk objectives; "
            "joint preserves the legacy all-at-once behavior."
        ),
    )
    parser.add_argument(
        "--risk_warmup_epochs",
        type=int,
        default=3,
        help=(
            "For risk_finetune/joint, keep sparse event/risk losses at zero "
            "for this many initial epochs."
        ),
    )
    parser.add_argument(
        "--risk_ramp_epochs",
        type=int,
        default=10,
        help=(
            "For risk_finetune/joint, linearly ramp sparse event/risk losses "
            "to their configured weights over this many epochs."
        ),
    )
    parser.add_argument(
        "--hierarchical_power_blend",
        type=float,
        default=1.0,
        help=(
            "Maximum hierarchy contribution to structured-appliance y_power. "
            "0 keeps base power; 1 uses the full hierarchy scenario."
        ),
    )
    parser.add_argument(
        "--hierarchical_blend_warmup_epochs",
        type=int,
        default=0,
        help=(
            "For hierarchical future training, keep y_power on the frozen base "
            "route for this many initial epochs."
        ),
    )
    parser.add_argument(
        "--hierarchical_blend_ramp_epochs",
        type=int,
        default=0,
        help=(
            "Linearly increase hierarchy power blend to "
            "--hierarchical_power_blend over this many epochs."
        ),
    )
    parser.add_argument(
        "--risk_mae_tolerance_ratio",
        type=float,
        default=0.03,
        help=(
            "For risk_best.pt, permit validation MAE up to this relative "
            "increase over the base checkpoint before selecting by EventF1."
        ),
    )
    parser.add_argument(
        "--risk_scheduler_patience",
        type=int,
        default=8,
        help=(
            "Number of post-warmup EventF1 plateaus tolerated before reducing "
            "the risk-stage learning rate."
        ),
    )
    parser.add_argument(
        "--require_base_beats_baselines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before risk_finetune, require the base run's best validation MAE "
            "to beat the fair zero-power baseline. Historical-submeter "
            "persistence remains report-only because it is an oracle reference."
        ),
    )
    parser.add_argument(
        "--require_base_calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before risk_finetune, require validation-set threshold calibration "
            "from the base run."
        ),
    )

    # Optimization
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument(
        "--domain_generalization_weight",
        type=float,
        default=0.0,
        help=(
            "Weight on the mean-plus-dispersion normalized future-power risk "
            "added to the original ERM loss. Requires multi-home data with "
            "home_index; zero preserves ordinary ERM."
        ),
    )
    parser.add_argument(
        "--domain_generalization_dispersion_weight",
        type=float,
        default=0.25,
        help=(
            "Within the normalized future-power DG objective, weight applied "
            "to the standard deviation across source homes."
        ),
    )
    parser.add_argument(
        "--domain_generalization_warmup_epochs",
        type=int,
        default=0,
        help="Epochs optimized with ordinary ERM before cross-home regularization.",
    )
    parser.add_argument(
        "--domain_generalization_ramp_epochs",
        type=int,
        default=0,
        help="Linear ramp length for the cross-home regularization weight.",
    )
    parser.add_argument(
        "--source_on_monitor_weight",
        type=float,
        default=0.5,
        help=(
            "Weight assigned to normalized source-home ON-condition MAE in "
            "the combined checkpoint monitor; the remaining weight is assigned "
            "to ordinary all-step normalized MAE."
        ),
    )
    parser.add_argument(
        "--source_on_safety_ceiling",
        type=float,
        default=None,
        help=(
            "Optional hard checkpoint constraint on the worst source home's "
            "normalized ON-condition MAE. Eligible checkpoints are then ranked "
            "only by --monitor."
        ),
    )
    parser.add_argument(
        "--source_on_safety_scope", choices=["home", "home_appliance"], default="home",
    )
    parser.add_argument("--source_on_min_windows", type=int, default=5)

    # Trainer
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--min_epochs_before_stopping",
        type=int,
        default=0,
        help=(
            "Do not trigger early stopping before this epoch. Useful for "
            "hierarchical sparse-event training where validation loss is noisy "
            "during the warm-up phase."
        ),
    )
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_metrics_every", type=int, default=1)
    parser.add_argument(
        "--monitor",
        type=str,
        default=CORE_MAE_MONITOR,
        help=(
            "Validation metric used for learning-rate scheduling, early stopping "
            "and best.pt selection. Defaults to core macro MAE, not loss_total."
        ),
    )
    parser.add_argument(
        "--calibrate_state_thresholds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After loading the best checkpoint, sweep validation-set power "
            "thresholds per appliance and reuse them for validation/test metrics."
        ),
    )
    parser.add_argument("--threshold_search_steps", type=int, default=80)
    parser.add_argument("--threshold_min", type=float, default=0.0)
    parser.add_argument("--threshold_max", type=float, default=None)
    parser.add_argument(
        "--threshold_objective",
        type=str,
        default="state_f1",
        choices=["state_f1", "event_f1", "mean_state_event_f1"],
        help=(
            "Validation objective for threshold search. state_f1 is stable; "
            "event_f1 focuses on transitions; mean_state_event_f1 is a compromise."
        ),
    )
    parser.add_argument(
        "--calibrate_event_thresholds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Sweep validation event-head probabilities per appliance and reuse "
            "the F1-optimal thresholds for final validation/test reporting."
        ),
    )
    parser.add_argument("--event_threshold_search_steps", type=int, default=80)
    parser.add_argument("--event_threshold_min", type=float, default=0.05)
    parser.add_argument("--event_threshold_max", type=float, default=0.99)
    parser.add_argument(
        "--postprocess_state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply simple duration cleanup to predicted ON/OFF states before "
            "computing state/event F1."
        ),
    )
    parser.add_argument("--min_on_duration", type=int, default=3)
    parser.add_argument("--min_off_duration", type=int, default=2)
    parser.add_argument(
        "--event_tolerance_minutes",
        type=int,
        default=2,
        help="One-to-one event matching tolerance used in final reports.",
    )
    parser.add_argument(
        "--event_bucket_size",
        type=int,
        default=5,
        help="Bucket size in minutes for additional event F1 reporting.",
    )

    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable automatic mixed precision.",
    )

    # Loss weights
    parser.add_argument("--lambda_power", type=float, default=1.0)
    parser.add_argument("--lambda_bridge", type=float, default=0.3)
    parser.add_argument("--lambda_state", type=float, default=0.40)
    parser.add_argument("--lambda_event", type=float, default=0.50)
    parser.add_argument("--lambda_start", type=float, default=0.50)
    parser.add_argument("--lambda_stop", type=float, default=0.50)
    parser.add_argument(
        "--tcn_event_weight",
        type=float,
        default=0.05,
        help=(
            "Direct event/start/stop supervision used only while training the "
            "refined-history residual TCN. Set to 0 to reproduce power-only "
            "TCN training."
        ),
    )
    parser.add_argument("--lambda_bucket_start", type=float, default=0.0)
    parser.add_argument("--lambda_bucket_stop", type=float, default=0.0)
    parser.add_argument("--lambda_event_offset", type=float, default=0.0)
    parser.add_argument("--lambda_window_start", type=float, default=0.0)
    parser.add_argument("--lambda_conditional_bucket", type=float, default=0.0)
    parser.add_argument("--lambda_conditional_offset", type=float, default=0.0)
    parser.add_argument("--lambda_conditional_power", type=float, default=0.0)
    parser.add_argument("--lambda_pulse_duration", type=float, default=0.0)
    parser.add_argument("--lambda_pulse_amplitude", type=float, default=0.0)
    parser.add_argument("--lambda_agg", type=float, default=0.02)
    parser.add_argument("--lambda_ghost", type=float, default=0.08)
    parser.add_argument("--lambda_peak", type=float, default=0.30)
    parser.add_argument("--lambda_amp_on", type=float, default=0.0)
    parser.add_argument("--lambda_orth", type=float, default=0.005)
    parser.add_argument("--active_power_weight", type=float, default=5.0)
    parser.add_argument(
        "--balanced_on_off_power_loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For every appliance, average ON and OFF power error equally, then "
            "average across appliances. This prevents rare appliances from "
            "minimizing the power objective with an all-off prediction."
        ),
    )
    parser.add_argument(
        "--balanced_on_fraction",
        type=float,
        default=0.5,
        help=(
            "ON-state fraction used by --balanced_on_off_power_loss. OFF uses "
            "the complementary fraction. The controlled default is 0.5."
        ),
    )
    parser.add_argument("--active_bridge_weight", type=float, default=4.0)
    parser.add_argument(
        "--active_recon_weight",
        type=float,
        default=0.0,
        help=(
            "Extra ON-state weight for historical reconstruction. Zero preserves "
            "the original unweighted reconstruction objective."
        ),
    )
    parser.add_argument("--state_alpha", type=float, default=0.70)
    parser.add_argument("--event_alpha", type=float, default=0.92)
    parser.add_argument(
        "--stop_alpha",
        type=float,
        default=None,
        help="Optional positive weight for the stop head; defaults to --event_alpha.",
    )
    parser.add_argument("--lambda_recon_power", type=float, default=0.20)
    parser.add_argument("--lambda_recon_state", type=float, default=0.10)    
    parser.add_argument(
        "--lambda_direct_power",
        type=float,
        default=0.0,
        help="Auxiliary power loss for the enhanced direct future decoder.",
    )
    parser.add_argument(
        "--base_aux_warmup_epochs",
        type=int,
        default=0,
        help=(
            "For base-stage power stabilization, hold bridge/state/ghost/peak "
            "at their base-warmup weights for this many epochs. Disabled at 0."
        ),
    )
    parser.add_argument(
        "--base_aux_ramp_epochs",
        type=int,
        default=0,
        help=(
            "Linearly ramp base auxiliary weights from their warmup values to "
            "--lambda_bridge/state/ghost/peak over this many epochs."
        ),
    )
    parser.add_argument("--base_warmup_lambda_bridge", type=float, default=0.05)
    parser.add_argument("--base_warmup_lambda_state", type=float, default=0.10)
    parser.add_argument("--base_warmup_lambda_ghost", type=float, default=0.0)
    parser.add_argument("--base_warmup_lambda_peak", type=float, default=0.0)
    parser.add_argument(
        "--skip_test_evaluation",
        action="store_true",
        help=(
            "Do not evaluate the test split after training. Use for validation-only "
            "model selection and stabilization studies."
        ),
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT / "outputs" / "runs"),
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
    )

    return parser.parse_args()


def resolve_appliance_cols(raw_appliances: list[str] | None) -> list[str]:
    if raw_appliances is None:
        return DEFAULT_APPLIANCE_COLS.copy()

    appliance_cols: list[str] = []
    for value in raw_appliances:
        for name in value.split(","):
            name = name.strip()
            if name:
                appliance_cols.append(name)

    if not appliance_cols:
        raise ValueError("--appliances should contain at least one appliance name.")

    unknown = sorted(set(appliance_cols) - set(SUPPORTED_APPLIANCE_COLS))
    if unknown:
        raise ValueError(
            f"Unknown appliances: {unknown}. Available: {SUPPORTED_APPLIANCE_COLS}"
        )

    return list(dict.fromkeys(appliance_cols))


def _checkpoint_appliances(path: Path) -> list[str] | None:
    config_path = path.parent.parent / "config.json"
    if not config_path.exists():
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    appliances = config.get("appliance_cols")
    if not isinstance(appliances, list) or not appliances:
        return None
    return [str(name) for name in appliances]


def _remap_checkpoint_appliances(
    state_dict: dict[str, torch.Tensor],
    source_appliances: list[str],
    target_appliances: list[str],
) -> dict[str, torch.Tensor]:
    """Select appliance-specific weights by name when model widths differ."""
    if source_appliances == target_appliances:
        return state_dict

    missing = [name for name in target_appliances if name not in source_appliances]
    if missing:
        raise ValueError(
            f"Checkpoint does not contain target appliances: {missing}. "
            f"Available: {source_appliances}"
        )

    source_indices = [source_appliances.index(name) for name in target_appliances]
    index = torch.tensor(source_indices, dtype=torch.long)
    remapped = dict(state_dict)

    # Private decoder banks are indexed by position in ModuleList.
    decoder_prefix = "decoder.appliance_decoders."
    for key in list(remapped):
        if key.startswith(decoder_prefix):
            del remapped[key]
    for target_idx, source_idx in enumerate(source_indices):
        source_prefix = f"{decoder_prefix}{source_idx}."
        target_prefix = f"{decoder_prefix}{target_idx}."
        for key, value in state_dict.items():
            if key.startswith(source_prefix):
                remapped[target_prefix + key[len(source_prefix):]] = value

    row_keys = {
        "bridge.appliance_queries",
        "bridge.state_bias",
        "bridge.event_bias",
        "decoder.appliance_embedding.weight",
        "tail_disagg_head.appliance_embedding.weight",
        "bridge_fusion_logit",
        "tail_global_fusion_logit",
        "past_recon_head.power_head.weight",
        "past_recon_head.power_head.bias",
        "past_recon_head.state_head.weight",
        "past_recon_head.state_head.bias",
    }
    for key in row_keys:
        value = state_dict.get(key)
        if value is not None and value.size(0) == len(source_appliances):
            remapped[key] = value.index_select(0, index)

    # Type embeddings use sorted appliance-type names in PISAModel.
    source_types = sorted(
        {DEFAULT_APPLIANCE_TYPES.get(name, name) for name in source_appliances}
    )
    target_types = sorted(
        {DEFAULT_APPLIANCE_TYPES.get(name, name) for name in target_appliances}
    )
    source_type_indices = torch.tensor(
        [source_types.index(type_name) for type_name in target_types],
        dtype=torch.long,
    )
    for key in [
        "bridge.type_embedding.weight",
        "decoder.type_embedding.weight",
        "tail_disagg_head.type_embedding.weight",
    ]:
        value = state_dict.get(key)
        if value is not None and value.size(0) == len(source_types):
            remapped[key] = value.index_select(0, source_type_indices)

    # Keep target-model buffers whose values depend on the selected appliance set.
    for key in list(remapped):
        if key.endswith("appliance_type_ids") or key.endswith("sparse_appliance_mask"):
            del remapped[key]

    print(
        "Remapped checkpoint appliances: "
        f"{source_appliances} -> {target_appliances}"
    )
    return remapped


def load_model_weights(model: PISAModel, checkpoint_path: str | Path) -> None:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    state_dict = dict(checkpoint.get("model_state_dict", checkpoint))
    # This cap is derived from the current run's training split.
    state_dict.pop("conditional_peak_power", None)
    # Keep the current run's documented capacity caps rather than inheriting
    # an unbounded or differently specified cap from an initialization run.
    state_dict.pop("constraint.rated_power", None)
    state_dict.pop("refined_history_residual_tcn.rated_power", None)
    # A base checkpoint was trained without hierarchy; a risk run must start
    # from its own scheduled blend (normally zero), not inherit an unrelated
    # value from another risk run.
    state_dict.pop("hierarchical_power_blend", None)

    source_appliances = _checkpoint_appliances(path)
    target_appliances = list(getattr(model, "appliance_names", []))
    if source_appliances is not None and target_appliances:
        state_dict = _remap_checkpoint_appliances(
            state_dict,
            source_appliances=source_appliances,
            target_appliances=target_appliances,
        )

    current_state = model.state_dict()
    compatible_state = {}
    skipped_keys = []

    for key, value in state_dict.items():
        if key in current_state and current_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            skipped_keys.append(key)

    current_state.update(compatible_state)
    incompatible = model.load_state_dict(current_state, strict=True)

    print(f"Initialized model from checkpoint: {path}")
    print(f"Loaded compatible checkpoint keys: {len(compatible_state)}")
    if skipped_keys:
        print(f"Skipped incompatible checkpoint keys: {len(skipped_keys)}")
        print(skipped_keys[:10])
    if incompatible.missing_keys:
        print(f"Missing checkpoint keys: {len(incompatible.missing_keys)}")
        print(incompatible.missing_keys[:10])
    if incompatible.unexpected_keys:
        print(f"Unexpected checkpoint keys: {len(incompatible.unexpected_keys)}")
        print(incompatible.unexpected_keys[:10])


def apply_checkpoint_architecture_args(args: argparse.Namespace) -> None:
    """Auto-enable optional PISA modules already present in an init checkpoint."""
    if args.init_checkpoint is None:
        return
    path = Path(args.init_checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    payload = torch.load(path, map_location="cpu")
    state_dict = payload.get("model_state_dict", payload)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint does not contain a model state_dict: {path}")

    # Every staged run must reconstruct its frozen deterministic source route
    # exactly. Several behavior-defining values (for example state_gate_floor
    # and rated-power caps) are not inferable from tensor shapes. Recover them
    # from the source run rather than silently using parser defaults that may
    # describe another home or ablation.
    staged_frozen_route = (
        args.training_stage == "risk_finetune"
        or args.train_mode in {"history_reconstruction", "future_residual_tcn"}
        or getattr(args, "evaluate_only", False)
    )
    if staged_frozen_route:
        source_config_path = path.parent.parent / "config.json"
        if not source_config_path.is_file():
            raise FileNotFoundError(
                "Staged fine-tuning requires the source checkpoint config so the "
                f"deterministic route can be reconstructed exactly: {source_config_path}"
            )
        source_config = load_json(source_config_path)
        source_data = source_config.get("data", {})
        source_model = source_config.get("model", {})
        if not isinstance(source_data, dict) or not isinstance(source_model, dict):
            raise ValueError(
                f"Invalid data/model sections in source config: {source_config_path}"
            )

        model_arg_map = {
            "d_model": "d_model",
            "n_heads": "n_heads",
            "encoder_layers": "encoder_layers",
            "use_bridge_residual": "use_bridge_residual",
            "state_conditioned_power": "state_conditioned_power",
            "state_gate_floor": "state_gate_floor",
            "state_power_blend": "state_power_blend",
            "event_bucket_size": "event_bucket_size",
            "power_adapter_dim": "power_adapter_dim",
        }
        # Risk continuation and evaluation-only runs must preserve the complete
        # already-selected TCN route.  In particular, silently falling back to
        # the CLI default history source (``refined``) while evaluating a
        # checkpoint trained with ``base`` history changes its predictions.
        # When genuinely adding a new TCN to a history-only checkpoint, leave
        # the requested TCN hyperparameters untouched.
        if (
            args.training_stage == "risk_finetune"
            or getattr(args, "evaluate_only", False)
        ):
            model_arg_map.update(
                {
                    "residual_tcn_hidden_dim": "residual_tcn_hidden_dim",
                    "residual_tcn_num_layers": "residual_tcn_num_layers",
                    "residual_tcn_kernel_size": "residual_tcn_kernel_size",
                    "residual_tcn_dropout": "residual_tcn_dropout",
                    "residual_tcn_history_source": "residual_tcn_history_source",
                    "residual_tcn_shared": "residual_tcn_shared",
                    "future_residual_fusion": "future_residual_fusion",
                    "residual_tcn_use_future_context": (
                        "residual_tcn_use_future_context"
                    ),
                    "residual_tcn_target_adapter_dim": (
                        "residual_tcn_target_adapter_dim"
                    ),
                    "future_base_blend_init": "future_base_blend_init",
                    "target_output_calibration": "target_output_calibration",
                }
            )
        for config_name, arg_name in model_arg_map.items():
            if config_name in source_model:
                setattr(args, arg_name, source_model[config_name])

        for config_name, arg_name in {
            "input_window": "input_window",
            "horizon": "horizon",
            "target_mode": "target_mode",
            "appliance_scale_mode": "appliance_scale_mode",
        }.items():
            if config_name in source_data:
                setattr(args, arg_name, source_data[config_name])

        source_appliances = source_config.get("appliance_cols")
        if isinstance(source_appliances, list) and source_appliances:
            args.appliances = [str(name) for name in source_appliances]

        rated_power = source_data.get(
            "nominal_rated_power_kw",
            source_data.get("rated_power_kw"),
        )
        if isinstance(rated_power, list) and rated_power:
            args.rated_power_kw = [float(value) for value in rated_power]
        args.disable_rated_power_cap = not bool(
            source_data.get("rated_power_cap_enabled", True)
        )
        print(
            "Frozen deterministic-route config restored from source run: "
            f"{source_config_path}"
        )

    inferred = infer_optional_pisa_architecture(state_dict)
    if inferred.get("use_history_recon_refiner"):
        args.use_history_recon_refiner = True
    if inferred.get("use_refined_history_residual_tcn"):
        args.use_refined_history_residual_tcn = True
        for name in (
            "residual_tcn_hidden_dim",
            "residual_tcn_num_layers",
            "residual_tcn_kernel_size",
            "residual_tcn_dropout",
            "residual_tcn_shared",
        ):
            if name in inferred:
                previous = getattr(args, name)
                setattr(args, name, inferred[name])
                if previous != inferred[name]:
                    print(
                        f"Checkpoint architecture override: "
                        f"{name}={inferred[name]} (requested {previous})"
                    )
        checkpoint_horizon = inferred.get("residual_tcn_horizon")
        if (
            checkpoint_horizon is not None
            and int(checkpoint_horizon) != int(args.horizon)
        ):
            raise ValueError(
                f"Residual-TCN checkpoint horizon={checkpoint_horizon}, "
                f"but --horizon={args.horizon}."
            )
        if inferred.get("residual_tcn_use_future_context"):
            args.residual_tcn_use_future_context = True
            args.future_residual_fusion = "learned_blend"
            args.target_output_calibration = bool(
                inferred.get("target_output_calibration", True)
            )
            if "residual_tcn_target_adapter_dim" in inferred:
                args.residual_tcn_target_adapter_dim = int(
                    inferred["residual_tcn_target_adapter_dim"]
                )
    print(
        "Checkpoint optional modules: "
        f"history_refiner={bool(inferred.get('use_history_recon_refiner'))}, "
        "future_residual_tcn="
        f"{bool(inferred.get('use_refined_history_residual_tcn'))}; "
        "effective request: "
        f"history_refiner={args.use_history_recon_refiner}, "
        f"future_residual_tcn={args.use_refined_history_residual_tcn}"
    )


def configure_trainable_parameters(model: PISAModel, train_mode: str) -> None:
    # A model instance can be reconfigured in tests or staged workflows.
    # Clear train-mode guards first so a previous stage cannot leak into
    # the next call to ``model.train()``.
    model._history_reconstruction_only_training = False
    model._future_residual_only_training = False
    model._enhanced_forecast_heads_only_training = False
    model._risk_heads_only_training = False

    for param in model.parameters():
        param.requires_grad = False

    if train_mode == "full":
        for param in model.parameters():
            param.requires_grad = True
    elif train_mode == "decoder":
        for param in model.decoder.parameters():
            param.requires_grad = True
        if model.tail_disagg_head is not None:
            for param in model.tail_disagg_head.parameters():
                param.requires_grad = True
            model.tail_global_fusion_logit.requires_grad = True
        model.bridge_fusion_logit.requires_grad = True
    elif train_mode == "decoder_heads":
        head_names = [
            "base_power_head",
            "conditional_amplitude_head",
            "start_head",
            "event_head",
            "state_delta_head",
            "gate_head",
            "start_bucket_head",
            "stop_bucket_head",
            "start_offset_head",
            "stop_offset_head",
        ]
        if hasattr(model.decoder, "appliance_decoders"):
            for app_decoder in model.decoder.appliance_decoders:
                for name in head_names:
                    if hasattr(app_decoder, name):
                        for param in getattr(app_decoder, name).parameters():
                            param.requires_grad = True
        else:
            for name in head_names:
                if hasattr(model.decoder, name):
                    for param in getattr(model.decoder, name).parameters():
                        param.requires_grad = True
        if model.tail_disagg_head is not None:
            for param in model.tail_disagg_head.parameters():
                param.requires_grad = True
            model.tail_global_fusion_logit.requires_grad = True
        model.bridge_fusion_logit.requires_grad = True
    elif train_mode == "bucket_event_heads":
        head_names = [
            "start_bucket_head",
            "stop_bucket_head",
            "start_offset_head",
            "stop_offset_head",
        ]
        for app_decoder in model.decoder.appliance_decoders:
            for name in head_names:
                for param in getattr(app_decoder, name).parameters():
                    param.requires_grad = True
    elif train_mode == "hierarchical_future_heads":
        head_names = [
            "window_start_head",
            "conditional_profile_head",
            "start_bucket_head",
            "start_offset_head",
            "duration_head",
            "pulse_amplitude_head",
            "template_mix_head",
        ]
        for app_decoder in model.decoder.appliance_decoders:
            for name in head_names:
                if hasattr(app_decoder, name):
                    for param in getattr(app_decoder, name).parameters():
                        param.requires_grad = True
            if hasattr(app_decoder, "template_logits"):
                app_decoder.template_logits.requires_grad = True
    elif train_mode == "risk_heads":
        # Risk fine-tuning must not move the calibrated base power route.
        # In particular this leaves encoder, bridge, decoder trunk,
        # base_power_head, state heads and air/fridge amplitude paths frozen.
        head_names = [
            "start_head",
            "event_head",
            "start_bucket_head",
            "stop_bucket_head",
            "start_offset_head",
            "stop_offset_head",
            "window_start_head",
            "conditional_profile_head",
            "duration_head",
            "pulse_amplitude_head",
            "template_mix_head",
        ]
        for app_decoder in model.decoder.appliance_decoders:
            for name in head_names:
                if hasattr(app_decoder, name):
                    for param in getattr(app_decoder, name).parameters():
                        param.requires_grad = True
            if hasattr(app_decoder, "template_logits"):
                app_decoder.template_logits.requires_grad = True
        for param in model.risk_adapters.parameters():
            param.requires_grad = True
        model._risk_heads_only_training = True
    elif train_mode == "history_reconstruction":
        if model.history_recon_refiner is None:
            raise ValueError(
                "train_mode=history_reconstruction requires "
                "--use_history_recon_refiner."
            )
        for param in model.history_recon_refiner.parameters():
            param.requires_grad = True
        model._history_reconstruction_only_training = True
    elif train_mode == "future_residual_tcn":
        if model.refined_history_residual_tcn is None:
            raise ValueError(
                "train_mode=future_residual_tcn requires "
                "--use_residual_tcn."
            )
        for param in model.refined_history_residual_tcn.parameters():
            param.requires_grad = True
        if model.future_residual_fusion == "direct":
            model.refined_history_residual_tcn.power_gate_logit.requires_grad = False
            model.refined_history_residual_tcn.state_gate_logit.requires_grad = False
            model._future_residual_only_training = True
        elif model.future_residual_fusion == "learned_blend":
            # Reuse the already-trained causal TCN features. Only the compact
            # future-time decoder, Transformer-context adapter, physical-power
            # blend and affine calibration are allowed to move.
            residual_tcn = model.refined_history_residual_tcn
            for param in residual_tcn.parameters():
                param.requires_grad = False
            for module in residual_tcn.enhanced_adaptation_modules():
                for param in module.parameters():
                    param.requires_grad = True
            residual_tcn.base_power_blend_logit.requires_grad = True
            if residual_tcn.target_power_log_scale is not None:
                residual_tcn.target_power_log_scale.requires_grad = True
            if residual_tcn.target_power_bias is not None:
                residual_tcn.target_power_bias.requires_grad = True
            model._enhanced_forecast_heads_only_training = True
            model._future_residual_only_training = False
        else:
            model._future_residual_only_training = True
    else:
        raise ValueError(f"Unsupported train_mode: {train_mode}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0:
        raise RuntimeError(f"No trainable parameters for train_mode={train_mode}.")

    print(f"Train mode          : {train_mode}")
    print(f"Trainable parameters: {trainable:,} / {total:,}")


def apply_training_stage(args: argparse.Namespace) -> None:
    """Validate and materialize the requested base/fine-tuning stage."""
    if args.risk_warmup_epochs < 0 or args.risk_ramp_epochs < 0:
        raise ValueError("risk_warmup_epochs and risk_ramp_epochs must be >= 0.")
    if (
        int(getattr(args, "base_aux_warmup_epochs", 0)) < 0
        or int(getattr(args, "base_aux_ramp_epochs", 0)) < 0
    ):
        raise ValueError(
            "base_aux_warmup_epochs and base_aux_ramp_epochs must be >= 0."
        )
    for name in (
        "base_warmup_lambda_bridge",
        "base_warmup_lambda_state",
        "base_warmup_lambda_ghost",
        "base_warmup_lambda_peak",
    ):
        if float(getattr(args, name, 0.0)) < 0.0:
            raise ValueError(f"{name} must be >= 0.")
    if not 0.0 <= float(getattr(args, "hierarchical_power_blend", 1.0)) <= 1.0:
        raise ValueError("hierarchical_power_blend must be in [0, 1].")
    if (
        getattr(args, "hierarchical_blend_warmup_epochs", 0) < 0
        or getattr(args, "hierarchical_blend_ramp_epochs", 0) < 0
    ):
        raise ValueError("hierarchical blend warmup and ramp epochs must be >= 0.")
    if float(getattr(args, "risk_mae_tolerance_ratio", 0.03)) < 0.0:
        raise ValueError("risk_mae_tolerance_ratio must be >= 0.")
    if int(getattr(args, "risk_scheduler_patience", 8)) < 0:
        raise ValueError("risk_scheduler_patience must be >= 0.")
    if int(getattr(args, "risk_adapter_dim", 0)) < 0:
        raise ValueError("risk_adapter_dim must be >= 0.")
    if int(getattr(args, "power_adapter_dim", 0)) < 0:
        raise ValueError("power_adapter_dim must be >= 0.")
    if float(getattr(args, "active_recon_weight", 0.0)) < 0.0:
        raise ValueError("active_recon_weight must be >= 0.")
    if not math.isfinite(float(getattr(args, "history_recon_gate_init", -3.0))):
        raise ValueError("history_recon_gate_init must be finite.")
    if not math.isfinite(float(getattr(args, "future_residual_gate_init", -3.0))):
        raise ValueError("future_residual_gate_init must be finite.")
    if int(getattr(args, "residual_tcn_hidden_dim", 0)) <= 0:
        raise ValueError("residual_tcn_hidden_dim must be positive.")
    if int(getattr(args, "residual_tcn_num_layers", 0)) <= 0:
        raise ValueError("residual_tcn_num_layers must be positive.")
    if int(getattr(args, "residual_tcn_kernel_size", 0)) <= 0:
        raise ValueError("residual_tcn_kernel_size must be positive.")
    if int(getattr(args, "residual_tcn_target_adapter_dim", 0)) < 0:
        raise ValueError("residual_tcn_target_adapter_dim must be >= 0.")
    if not 0.0 <= float(getattr(args, "residual_tcn_dropout", 0.1)) < 1.0:
        raise ValueError("residual_tcn_dropout must be in [0, 1).")
    if not 0.0 <= float(getattr(args, "stratified_positive_fraction", 0.50)) < 1.0:
        raise ValueError("stratified_positive_fraction must be in [0, 1).")
    if not 0.0 <= float(getattr(args, "balanced_on_fraction", 0.50)) <= 1.0:
        raise ValueError("balanced_on_fraction must be in [0, 1].")
    if not 0.0 <= float(getattr(args, "source_on_monitor_weight", 0.50)) <= 1.0:
        raise ValueError("source_on_monitor_weight must be in [0, 1].")
    if (
        getattr(args, "source_on_safety_ceiling", None) is not None
        and (not math.isfinite(float(args.source_on_safety_ceiling))
             or float(args.source_on_safety_ceiling) <= 0.0)
    ):
        raise ValueError("source_on_safety_ceiling must be finite and positive.")
    if int(getattr(args, "source_on_min_windows", 5)) < 1:
        raise ValueError("source_on_min_windows must be positive.")
    if float(getattr(args, "tcn_event_weight", 0.0)) < 0.0:
        raise ValueError("tcn_event_weight must be >= 0.")
    if float(getattr(args, "lambda_direct_power", 0.0)) < 0.0:
        raise ValueError("lambda_direct_power must be >= 0.")

    if getattr(args, "evaluate_only", False):
        if args.init_checkpoint is None:
            raise ValueError("--evaluate_only requires --init_checkpoint.")
        if args.training_stage == "risk_finetune":
            if not args.hierarchical_future:
                raise ValueError(
                    "Risk-checkpoint evaluation requires --hierarchical_future."
                )
            # No optimisation occurs, but retaining the protected mode makes
            # model construction/provenance identical to the risk stage.
            args.train_mode = "risk_heads"
        elif args.train_mode not in {
            "full",
            "history_reconstruction",
            "future_residual_tcn",
        }:
            raise ValueError(
                "Base-stage --evaluate_only supports train_mode=full, "
                "history_reconstruction or future_residual_tcn."
            )
        return

    if args.train_mode == "history_reconstruction":
        if args.training_stage != "base":
            raise ValueError(
                "history_reconstruction uses --training_stage base with every "
                "existing future/risk path frozen."
            )
        if args.init_checkpoint is None:
            raise ValueError(
                "history_reconstruction requires --init_checkpoint from a "
                "completed PISA base run."
            )
        if not args.use_history_recon_refiner:
            raise ValueError(
                "history_reconstruction requires --use_history_recon_refiner."
            )
        if args.hierarchical_future:
            raise ValueError(
                "history_reconstruction must use --no-hierarchical_future."
            )
        if args.lambda_recon_power <= 0.0 and args.lambda_recon_state <= 0.0:
            raise ValueError(
                "history_reconstruction requires a positive "
                "--lambda_recon_power or --lambda_recon_state."
            )
        disabled_weights = (
            "lambda_power",
            "lambda_bridge",
            "lambda_state",
            "lambda_agg",
            "lambda_ghost",
            "lambda_peak",
            "lambda_amp_on",
            "lambda_orth",
            *RISK_LOSS_ARGUMENTS,
        )
        for name in disabled_weights:
            setattr(args, name, 0.0)
        args.risk_warmup_epochs = 0
        args.risk_ramp_epochs = 0
        args.base_aux_warmup_epochs = 0
        args.base_aux_ramp_epochs = 0
        args.base_warmup_lambda_bridge = 0.0
        args.base_warmup_lambda_state = 0.0
        args.base_warmup_lambda_ghost = 0.0
        args.base_warmup_lambda_peak = 0.0
        args.sampler_mode = "none"
        args.monitor = "val/loss_total"
        print(
            "History reconstruction stage: froze every existing PISA path and "
            "enabled only the deployment history residual refiner."
        )
        return

    if args.train_mode == "future_residual_tcn":
        if args.training_stage != "base":
            raise ValueError(
                "future_residual_tcn uses --training_stage base with every "
                "existing PISA/refiner/risk parameter frozen."
            )
        if args.init_checkpoint is None:
            raise ValueError(
                "future_residual_tcn requires a completed source-model "
                "--init_checkpoint."
            )
        if (
            args.residual_tcn_history_source == "refined"
            and not args.use_history_recon_refiner
        ):
            raise ValueError(
                "future_residual_tcn with --residual_tcn_history_source refined "
                "requires a checkpoint containing the history reconstruction "
                "refiner. Use history_source=base for the simplified main model."
            )
        if not args.use_refined_history_residual_tcn:
            raise ValueError(
                "future_residual_tcn requires "
                "--use_residual_tcn (legacy alias: "
                "--use_refined_history_residual_tcn)."
            )
        if args.hierarchical_future:
            raise ValueError(
                "future_residual_tcn currently requires "
                "--no-hierarchical_future."
            )
        if args.target_mode != "future":
            raise ValueError(
                "future_residual_tcn requires --target_mode future."
            )
        for name in (
            "lambda_bridge",
            "lambda_amp_on",
            "lambda_orth",
            "lambda_recon_power",
            "lambda_recon_state",
            *RISK_LOSS_ARGUMENTS,
        ):
            setattr(args, name, 0.0)
        # The hierarchy-specific objectives stay disabled, but the final TCN
        # state trajectory can now receive a small directional transition loss.
        # PISAModel recomputes event logits after TCN fusion, so these losses
        # supervise the same state path that produces final power.
        args.lambda_event = float(args.tcn_event_weight)
        args.lambda_start = float(args.tcn_event_weight)
        args.lambda_stop = float(args.tcn_event_weight)
        args.risk_warmup_epochs = 0
        args.risk_ramp_epochs = 0
        args.base_aux_warmup_epochs = 0
        args.base_aux_ramp_epochs = 0
        args.sampler_mode = "none"
        args.monitor = CORE_MAE_MONITOR
        args.eval_metrics_every = 1
        frozen_history_note = (
            " and the history refiner; "
            if args.use_history_recon_refiner
            else "; no history refiner is present; "
        )
        print(
            "Future residual-TCN stage: froze PISA"
            + frozen_history_note
            + f"enabled only {'one shared' if args.residual_tcn_shared else 'independent per-appliance'} "
            f"causal TCN parameters (history={args.residual_tcn_history_source}, "
            f"fusion={args.future_residual_fusion}, "
            f"event_weight={args.tcn_event_weight})."
        )
        return

    if args.training_stage == "base":
        if args.hierarchical_future:
            raise ValueError(
                "Base training must use --no-hierarchical_future. Run "
                "--training_stage risk_finetune after the core model beats "
                "the validation baselines."
            )
        requested_risk = {
            name: float(getattr(args, name))
            for name in RISK_LOSS_ARGUMENTS
            if float(getattr(args, name)) != 0.0
        }
        for name in RISK_LOSS_ARGUMENTS:
            setattr(args, name, 0.0)
        args.risk_warmup_epochs = 0
        args.risk_ramp_epochs = 0
        if args.sampler_mode != "none":
            print(
                "Base stage: switching sampler_mode from weighted to none so "
                "power calibration is learned on the natural window frequency."
            )
            args.sampler_mode = "none"
        if requested_risk:
            print(
                "Base stage: disabled future-risk loss weights: "
                f"{requested_risk}"
            )
        return

    if args.training_stage == "risk_finetune":
        if args.init_checkpoint is None:
            raise ValueError(
                "risk_finetune requires --init_checkpoint from a completed "
                "base training run."
            )
        if not args.hierarchical_future:
            raise ValueError(
                "risk_finetune requires --hierarchical_future so that window, "
                "conditional and pulse risk heads are active."
            )
        if args.target_mode != "future":
            raise ValueError("risk_finetune requires --target_mode future.")
        if args.train_mode == "full":
            print(
                "Risk stage: switching train_mode from full to risk_heads so "
                "the calibrated base encoder and power paths remain frozen."
            )
            args.train_mode = "risk_heads"
        if args.train_mode != "risk_heads":
            raise ValueError(
                "risk_finetune requires --train_mode risk_heads. This freezes "
                "the shared encoder and base power paths while training only "
                "hierarchical/event heads."
            )
        if (
            args.use_refined_history_residual_tcn
            and float(args.hierarchical_power_blend) != 0.0
        ):
            raise ValueError(
                "Risk fine-tuning on a residual-TCN checkpoint requires "
                "--hierarchical_power_blend 0. The hierarchical heads provide "
                "risk/scenario outputs, while residual TCN remains the fixed "
                "deterministic power route."
            )
        if (
            float(args.hierarchical_power_blend) > 0.0
            and args.hierarchical_blend_ramp_epochs == 0
        ):
            raise ValueError(
                "risk_finetune with a non-zero hierarchy blend requires "
                "--hierarchical_blend_ramp_epochs > 0. Use 0 blend to train "
                "only the risk heads first."
            )
        if not any(float(getattr(args, name)) > 0.0 for name in RISK_LOSS_ARGUMENTS):
            raise ValueError(
                "risk_finetune requires at least one non-zero future-risk loss "
                "weight (for example --lambda_window_start)."
            )
        return

    # joint is an explicit escape hatch for reproducing a legacy all-at-once
    # experiment. It remains available, but is not the default workflow.
    if args.training_stage == "joint" and not args.hierarchical_future:
        print(
            "Joint stage without hierarchical_future: conditional/window/pulse "
            "losses will have no model outputs and therefore remain zero."
        )


def verify_base_run_for_risk_finetune(args: argparse.Namespace) -> dict[str, Any] | None:
    """Enforce the base -> calibration -> risk-finetune hand-off contract."""
    if args.training_stage != "risk_finetune" or getattr(args, "evaluate_only", False):
        return None

    checkpoint_path = Path(args.init_checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {checkpoint_path}")

    base_run_dir = checkpoint_path.parent.parent
    metrics_path = base_run_dir / "results" / "best_val_metrics.json"
    calibration_path = base_run_dir / "results" / "calibrated_state_thresholds.json"
    provenance: dict[str, Any] = {
        "base_checkpoint": str(checkpoint_path),
        "base_run_dir": str(base_run_dir),
    }

    if args.require_base_beats_baselines:
        if not metrics_path.exists():
            raise FileNotFoundError(
                "risk_finetune requires the completed base run's validation "
                f"metrics: {metrics_path}"
            )
        base_metrics = load_json(metrics_path)
        model_mae = base_metrics.get("regression/macro_avg/MAE")
        zero_mae = base_metrics.get(BASELINE_ZERO_MAE)
        persistence_mae = base_metrics.get(PERSISTENCE_ORACLE_MAE)
        values = {
            "PISA": model_mae,
            "zero": zero_mae,
            "persistence_oracle": persistence_mae,
        }
        if not all(
            isinstance(value, (int, float))
            for value in (model_mae, zero_mae)
        ):
            raise ValueError(
                "Base validation metrics are incomplete; expected core PISA "
                "and zero-baseline macro MAE values."
            )
        if not float(model_mae) < float(zero_mae):
            raise ValueError(
                "Refusing risk_finetune because the base model has not beaten "
                "the fair zero baseline: "
                f"PISA={float(model_mae):.6f}, zero={float(zero_mae):.6f}. "
                "Historical-submeter persistence is report-only."
            )
        provenance["base_validation_mae"] = values

    if args.require_base_calibration:
        if not calibration_path.exists():
            raise FileNotFoundError(
                "risk_finetune requires validation-set threshold calibration "
                f"from the base run: {calibration_path}"
            )
        calibration = load_json(calibration_path)
        if not bool(calibration.get("enabled", False)):
            raise ValueError(
                "Base threshold calibration is disabled. Re-run the base stage "
                "with --calibrate_state_thresholds before risk fine-tuning."
            )
        provenance["base_calibration"] = str(calibration_path)

    print("Risk fine-tune base gate passed:")
    print(provenance)
    return provenance


def build_stage_summary(
    training_stage: str,
    monitor: str,
    best_epoch: int,
    best_metric: float,
    val_stats: dict[str, float],
    initialization: dict[str, Any] | None,
) -> dict[str, Any]:
    """Write a compact, machine-readable hand-off summary for each stage."""
    model_mae = val_stats.get("regression/macro_avg/MAE")
    zero_mae = val_stats.get(BASELINE_ZERO_MAE)
    persistence_mae = val_stats.get(PERSISTENCE_ORACLE_MAE)
    beats_zero = (
        isinstance(model_mae, (int, float))
        and isinstance(zero_mae, (int, float))
        and float(model_mae) < float(zero_mae)
    )
    beats_persistence = (
        isinstance(model_mae, (int, float))
        and isinstance(persistence_mae, (int, float))
        and float(model_mae) < float(persistence_mae)
    )
    return {
        "training_stage": training_stage,
        "monitor": monitor,
        "checkpoint_monitor": monitor,
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "validation_macro_mae": model_mae,
        "zero_macro_mae": zero_mae,
        "persistence_oracle_macro_mae": persistence_mae,
        "beats_zero": beats_zero,
        "beats_persistence_oracle": beats_persistence,
        "ready_for_risk_finetune": bool(beats_zero),
        "initialization": initialization,
    }


def main() -> None:
    args = parse_args()
    apply_checkpoint_architecture_args(args)
    apply_training_stage(args)
    if not args.state_conditioned_power and args.lambda_amp_on > 0:
        print("Notice: with direct power, amplitude_power and final power share the same "
              "output. lambda_amp_on adds ON weighting, not an independent amplitude task.")
    if args.hierarchical_future and args.target_mode != "future":
        raise ValueError("--hierarchical_future requires --target_mode future.")
    initialization = verify_base_run_for_risk_finetune(args)
    risk_mae_ceiling: float | None = None
    risk_checkpoint_start_epoch = 1
    if args.training_stage == "risk_finetune":
        base_mae = None
        if initialization is not None:
            base_mae = initialization.get("base_validation_mae", {}).get("PISA")
        if isinstance(base_mae, (int, float)):
            risk_mae_ceiling = float(base_mae) * (
                1.0 + float(args.risk_mae_tolerance_ratio)
            )
        risk_checkpoint_start_epoch = max(
            args.risk_warmup_epochs + args.risk_ramp_epochs + 1,
            args.hierarchical_blend_warmup_epochs + 1,
        )
    set_seed(args.seed)
    appliance_cols = resolve_appliance_cols(args.appliances)
    fixed_eval_state_thresholds: dict[str, float] | None = None
    fixed_eval_state_threshold_source: Path | None = None
    if args.evaluate_only or args.training_stage == "risk_finetune":
        fixed_eval_state_thresholds, fixed_eval_state_threshold_source = (
            load_fixed_state_thresholds(
                checkpoint_path=args.init_checkpoint,
                appliance_names=appliance_cols,
                explicit_path=args.risk_state_thresholds_path,
            )
        )
        print("Using MAE-safe fixed state thresholds for the risk protocol:")
        print(fixed_eval_state_thresholds)
        print(f"State-threshold source: {fixed_eval_state_threshold_source}")
    nominal_rated_power = rated_power_tensor_for_appliances(
        appliance_names=appliance_cols,
        rated_power_kw=args.rated_power_kw,
    )
    if args.disable_rated_power_cap:
        if args.model_type != "pisa":
            raise ValueError(
                "--disable_rated_power_cap is a PISA component ablation and "
                "cannot be used with model_type=aggregate_seq2seq."
            )
        # PISAModel validates rated_power as finite and positive. A very large
        # finite cap removes the clamp over the observed data range while
        # preserving numerical behaviour and an auditable configuration.
        rated_power = torch.full_like(nominal_rated_power, 1e6)
        print(
            "Component ablation: rated-power clamp disabled "
            "(effective cap = 1e6 kW for every appliance)."
        )
    else:
        rated_power = nominal_rated_power

    # -----------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------
    source_csv_paths: list[Path] = []
    if args.source_csv_paths:
        source_csv_paths = [
            Path(value).expanduser().resolve()
            for value in args.source_csv_paths
        ]
        if len(source_csv_paths) < 2:
            raise ValueError("--source_csv_paths requires at least two homes.")
        csv_path = source_csv_paths[0]
    elif args.csv_path is None:
        csv_path = find_default_csv()
    else:
        csv_path = Path(args.csv_path).expanduser().resolve()
    if not source_csv_paths:
        source_csv_paths = [csv_path]

    for source_path in source_csv_paths:
        if not source_path.exists():
            raise FileNotFoundError(f"Source CSV file not found: {source_path}")

    held_out_csv_path = (
        None
        if args.held_out_csv_path is None
        else Path(args.held_out_csv_path).expanduser().resolve()
    )
    if held_out_csv_path is not None:
        if not held_out_csv_path.exists():
            raise FileNotFoundError(
                f"Held-out CSV file not found: {held_out_csv_path}"
            )
        if held_out_csv_path in source_csv_paths:
            raise ValueError("The held-out home cannot also be a source home.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.run_name is None:
        run_name = f"home7951_pisa_{timestamp}"
    else:
        run_name = args.run_name

    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    checkpoint_dir = run_dir / "checkpoints"
    result_dir = run_dir / "results"

    if not args.evaluate_only and any(
        (checkpoint_dir / name).exists() for name in ("best.pt", "last.pt")
    ):
        raise FileExistsError(
            f"Training checkpoints already exist in {checkpoint_dir}. "
            "Use a new run_name; existing checkpoints will not be overwritten."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Save experiment config
    # -----------------------------------------------------------------
    experiment_config = {
        "csv_path": str(csv_path),
        "source_csv_paths": [str(path) for path in source_csv_paths],
        "held_out_csv_path": (
            None if held_out_csv_path is None else str(held_out_csv_path)
        ),
        "input_cols": DEFAULT_INPUT_COLS,
        "appliance_cols": appliance_cols,
        "data": {
            "batch_size": args.batch_size,
            "input_window": args.input_window,
            "horizon": args.horizon,
            "target_mode": args.target_mode,
            "stride": args.stride,
            "appliance_scale_mode": args.appliance_scale_mode,
            "num_workers": args.num_workers,
            "sampler_mode": args.sampler_mode,
            "home_balanced_sampling": args.home_balanced_sampling,
            "stratified_positive_fraction": args.stratified_positive_fraction,
            "rated_power_kw": rated_power.tolist(),
            "nominal_rated_power_kw": nominal_rated_power.tolist(),
            "rated_power_cap_enabled": not args.disable_rated_power_cap,
        },
        "model": {
            "input_dim": len(DEFAULT_INPUT_COLS),
            "num_appliances": len(appliance_cols),
            "horizon": args.horizon,
            "model_type": args.model_type,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "encoder_layers": args.encoder_layers,
            "use_bridge_residual": args.use_bridge_residual,
            "use_tail_disaggregation_head": args.target_mode == "history_tail",
            "state_conditioned_power": args.state_conditioned_power,
            "state_gate_floor": args.state_gate_floor,
            "state_power_blend": args.state_power_blend,
            "event_bucket_size": args.event_bucket_size,
            "hierarchical_future": args.hierarchical_future,
            "hierarchical_power_blend": args.hierarchical_power_blend,
            "hierarchical_blend_warmup_epochs": (
                args.hierarchical_blend_warmup_epochs
            ),
            "hierarchical_blend_ramp_epochs": args.hierarchical_blend_ramp_epochs,
            "conditional_peak_multiplier": args.conditional_peak_multiplier,
            "risk_adapter_dim": args.risk_adapter_dim,
            "power_adapter_dim": args.power_adapter_dim,
            "use_history_recon_refiner": args.use_history_recon_refiner,
            "history_recon_gate_init": args.history_recon_gate_init,
            "use_refined_history_residual_tcn": (
                args.use_refined_history_residual_tcn
            ),
            "use_residual_tcn": args.use_refined_history_residual_tcn,
            "residual_tcn_hidden_dim": args.residual_tcn_hidden_dim,
            "residual_tcn_num_layers": args.residual_tcn_num_layers,
            "residual_tcn_kernel_size": args.residual_tcn_kernel_size,
            "residual_tcn_dropout": args.residual_tcn_dropout,
            "residual_tcn_history_source": args.residual_tcn_history_source,
            "residual_tcn_shared": args.residual_tcn_shared,
            "future_residual_fusion": args.future_residual_fusion,
            "residual_tcn_use_future_context": (
                args.residual_tcn_use_future_context
            ),
            "residual_tcn_target_adapter_dim": (
                args.residual_tcn_target_adapter_dim
            ),
            "future_base_blend_init": args.future_base_blend_init,
            "target_output_calibration": args.target_output_calibration,
            "future_residual_zero_init": args.future_residual_zero_init,
            "future_residual_gate_init": args.future_residual_gate_init,
            "init_checkpoint": args.init_checkpoint,
            "evaluate_only": args.evaluate_only,
            "risk_state_thresholds_path": args.risk_state_thresholds_path,
            "train_mode": args.train_mode,
            "training_stage": args.training_stage,
            "initialization": initialization,
        },
        # Persist the actual component settings alongside every run.  This
        # prevents a result table from depending on ambiguous directory names.
        "ablation": {
            "state_conditioned_power": args.state_conditioned_power,
            "bridge_residual_enabled": args.use_bridge_residual,
            "rated_power_cap_enabled": not args.disable_rated_power_cap,
            "lambda_agg": args.lambda_agg,
            "lambda_ghost": args.lambda_ghost,
            "lambda_peak": args.lambda_peak,
        },
        "optimization": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
        },
        "loss": {
            "lambda_power": args.lambda_power,
            "lambda_bridge": args.lambda_bridge,
            "lambda_state": args.lambda_state,
            "lambda_event": args.lambda_event,
            "lambda_start": args.lambda_start,
            "lambda_stop": args.lambda_stop,
            "tcn_event_weight": args.tcn_event_weight,
            "lambda_bucket_start": args.lambda_bucket_start,
            "lambda_bucket_stop": args.lambda_bucket_stop,
            "lambda_event_offset": args.lambda_event_offset,
            "lambda_window_start": args.lambda_window_start,
            "lambda_conditional_bucket": args.lambda_conditional_bucket,
            "lambda_conditional_offset": args.lambda_conditional_offset,
            "lambda_conditional_power": args.lambda_conditional_power,
            "lambda_pulse_duration": args.lambda_pulse_duration,
            "lambda_pulse_amplitude": args.lambda_pulse_amplitude,
            "lambda_agg": args.lambda_agg,
            "lambda_ghost": args.lambda_ghost,
            "lambda_peak": args.lambda_peak,
            "lambda_amp_on": args.lambda_amp_on,
            "lambda_orth": args.lambda_orth,
            "lambda_recon_power": args.lambda_recon_power,
            "lambda_recon_state": args.lambda_recon_state,
            "lambda_direct_power": args.lambda_direct_power,
            "active_power_weight": args.active_power_weight,
            "balanced_on_off_power_loss": args.balanced_on_off_power_loss,
            "balanced_on_fraction": args.balanced_on_fraction,
            "active_bridge_weight": args.active_bridge_weight,
            "active_recon_weight": args.active_recon_weight,
            "state_alpha": args.state_alpha,
            "event_alpha": args.event_alpha,
            "stop_alpha": args.stop_alpha,
            "risk_warmup_epochs": args.risk_warmup_epochs,
            "risk_ramp_epochs": args.risk_ramp_epochs,
            "base_aux_warmup_epochs": args.base_aux_warmup_epochs,
            "base_aux_ramp_epochs": args.base_aux_ramp_epochs,
            "base_warmup_lambda_bridge": args.base_warmup_lambda_bridge,
            "base_warmup_lambda_state": args.base_warmup_lambda_state,
            "base_warmup_lambda_ghost": args.base_warmup_lambda_ghost,
            "base_warmup_lambda_peak": args.base_warmup_lambda_peak,
            "risk_mae_tolerance_ratio": args.risk_mae_tolerance_ratio,
            "risk_scheduler_patience": args.risk_scheduler_patience,
        },
        "trainer": {
            "seed": args.seed,
            "device": args.device,
            "use_amp": not args.no_amp,
            "patience": args.patience,
            "log_interval": args.log_interval,
            "eval_metrics_every": args.eval_metrics_every,
            "monitor": args.monitor,
            "skip_test_evaluation": args.skip_test_evaluation,
            "monitor_mode": "min",
            "require_base_beats_baselines": args.require_base_beats_baselines,
            "require_base_calibration": args.require_base_calibration,
            "calibrate_state_thresholds": args.calibrate_state_thresholds,
            "calibrate_event_thresholds": args.calibrate_event_thresholds,
            "threshold_search_steps": args.threshold_search_steps,
            "threshold_min": args.threshold_min,
            "threshold_max": args.threshold_max,
            "threshold_objective": args.threshold_objective,
            "event_threshold_search_steps": args.event_threshold_search_steps,
            "event_threshold_min": args.event_threshold_min,
            "event_threshold_max": args.event_threshold_max,
            "postprocess_state": args.postprocess_state,
            "min_on_duration": args.min_on_duration,
            "min_off_duration": args.min_off_duration,
            "event_tolerance_minutes": args.event_tolerance_minutes,
            "event_bucket_size": args.event_bucket_size,
            "risk_monitor": (
                RISK_HIERARCHICAL_START_MONITOR
                if args.training_stage == "risk_finetune"
                else None
            ),
            "risk_mae_ceiling": risk_mae_ceiling,
            "risk_checkpoint_start_epoch": risk_checkpoint_start_epoch,
            "scheduler_monitor": (
                "val/event/macro_avg/EventF1"
                if args.training_stage == "risk_finetune"
                else args.monitor
            ),
            "scheduler_start_epoch": (
                risk_checkpoint_start_epoch
                if args.training_stage == "risk_finetune"
                else 1
            ),
            "source_on_monitor_weight": args.source_on_monitor_weight,
            "source_on_safety_ceiling": args.source_on_safety_ceiling,
            "source_on_safety_scope": args.source_on_safety_scope,
            "source_on_min_windows": args.source_on_min_windows,
        },
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "result_dir": str(result_dir),
    }

    save_json(run_dir / "config.json", experiment_config)

    print("=" * 80)
    print("PISA training")
    print("=" * 80)
    print(f"Project root : {ROOT}")
    print(f"Source CSVs  : {[str(path) for path in source_csv_paths]}")
    print(f"Held-out CSV : {held_out_csv_path}")
    print(f"Run dir      : {run_dir}")
    print(f"Checkpoint   : {checkpoint_dir}")
    print(f"Input cols   : {DEFAULT_INPUT_COLS}")
    print(f"Appliances   : {appliance_cols}")
    print("=" * 80)

    # -----------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------
    manual_state_thresholds = {
        "air1": 0.50,
        "refrigerator1": 0.05,
        "dishwasher1": 0.05,
        "waterheater1": 0.50,
        "microwave1": 0.10,
    }
    source_home_loaders: dict[str, dict[str, DataLoader]] = {}
    if len(source_csv_paths) > 1:
        loaders, bundle, source_home_loaders = build_multi_home_dataloaders(
            csv_paths=source_csv_paths,
            batch_size=args.batch_size,
            input_window=args.input_window,
            horizon=args.horizon,
            target_mode=args.target_mode,
            stride=args.stride,
            input_cols=DEFAULT_INPUT_COLS,
            appliance_cols=appliance_cols,
            num_workers=args.num_workers,
            state_thresholds=manual_state_thresholds,
            balanced_train=args.home_balanced_sampling,
            seed=args.seed,
            appliance_scale_mode=args.appliance_scale_mode,
        )
    else:
        loaders, bundle = build_single_home_dataloaders(
            csv_path=csv_path,
            appliance_scale_mode=args.appliance_scale_mode,
            batch_size=args.batch_size,
            input_window=args.input_window,
            horizon=args.horizon,
            target_mode=args.target_mode,
            stride=args.stride,
            input_cols=DEFAULT_INPUT_COLS,
            appliance_cols=appliance_cols,
            num_workers=args.num_workers,
            state_thresholds=manual_state_thresholds,
        )

    held_out_loaders: dict[str, DataLoader] | None = None
    if held_out_csv_path is not None:
        held_out_loaders, _ = build_single_home_dataloaders(
            csv_path=held_out_csv_path,
            batch_size=args.eval_batch_size,
            input_window=args.input_window,
            horizon=args.horizon,
            target_mode=args.target_mode,
            stride=args.stride,
            input_cols=DEFAULT_INPUT_COLS,
            appliance_cols=appliance_cols,
            num_workers=args.num_workers,
            state_thresholds=manual_state_thresholds,
            reference_bundle=bundle,
            shuffle_train=False,
        )

    sparse_names = {"dishwasher1", "waterheater1", "microwave1"}
    sparse_indices = [
        idx for idx, app in enumerate(appliance_cols) if app in sparse_names
    ]

    if len(source_csv_paths) > 1:
        print(
            "Multi-home sampler: "
            + ("home-balanced" if args.home_balanced_sampling else "natural")
        )
    elif args.sampler_mode == "weighted":
        loaders["train"] = build_active_weighted_train_loader(
            train_dataset=loaders["train"].dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sparse_indices=sparse_indices,
        )
    elif args.sampler_mode == "device_stratified":
        loaders["train"] = build_device_stratified_train_loader(
            train_dataset=loaders["train"].dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sparse_indices=sparse_indices,
            positive_fraction=args.stratified_positive_fraction,
            seed=args.seed,
        )
    print("Data loaded.")
    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches  : {len(loaders['val'])}")

    if "test" in loaders:
        print(f"Test batches : {len(loaders['test'])}")
    if held_out_loaders is not None and "test" in held_out_loaders:
        print(f"Held-out test batches: {len(held_out_loaders['test'])}")

    # Save useful data statistics
    data_info = {
        "input_cols": bundle.input_cols,
        "appliance_cols": bundle.appliance_cols,
        "mains_col": bundle.mains_col,
        "state_thresholds": bundle.state_thresholds,
        "appliance_scales": bundle.appliance_scales,
        "scalers": bundle.scalers.state_dict(),
        "appliance_activity_stats": getattr(bundle, "appliance_activity_stats", None),
        "source_csv_paths": [str(path) for path in source_csv_paths],
        "source_home_window_counts": {
            home_id: {
                split: len(loader.dataset)
                for split, loader in split_loaders.items()
            }
            for home_id, split_loaders in source_home_loaders.items()
        },
        "held_out_csv_path": (
            None if held_out_csv_path is None else str(held_out_csv_path)
        ),
    }

    save_json(result_dir / "data_info.json", data_info)

    if source_home_loaders:
        on_support = {
            home: summarize_on_support(split_loaders["val"].dataset, args.source_on_min_windows)
            for home, split_loaders in source_home_loaders.items()
        }
        save_json(result_dir / "source_on_support.json", on_support)
    else:
        on_support = {}
    if not args.evaluate_only and args.source_on_safety_ceiling is not None:
        if len(on_support) < 2:
            raise ValueError("Source ON safety requires validation data from at least two source homes.")
        if args.source_on_safety_scope == "home_appliance":
            unsupported = [f"{home}/{app}" for home, support in on_support.items()
                           for app, counts in support["appliances"].items() if not counts["supported"]]
            if unsupported:
                save_json(checkpoint_dir / "selection_status.json", {
                    "status": "insufficient_source_on_support", "unsupported_pairs": unsupported,
                    "safety_scope": args.source_on_safety_scope,
                    "safety_ceiling": args.source_on_safety_ceiling,
                })
                raise ValueError("Cannot evaluate per-appliance ON safety: " + ", ".join(unsupported)
                                 + ". See results/source_on_support.json; no training was started.")

    print("State thresholds:")
    print(bundle.state_thresholds)

    print("Appliance scales:")
    print(bundle.appliance_scales)

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------
    if args.model_type == "pisa":
        conditional_peak_power = torch.tensor(
            [
                min(
                    max(
                        stats.p95_on_power * args.conditional_peak_multiplier,
                        stats.threshold,
                        1e-4,
                    ),
                    float(rated_power[idx]),
                )
                for idx, stats in enumerate(bundle.appliance_activity_stats)
            ],
            dtype=torch.float32,
        )
        if args.hierarchical_future:
            print("Conditional ON-state p95 template caps:")
            print(conditional_peak_power.tolist())
            save_json(
                result_dir / "conditional_template_caps.json",
                {
                    "source": "training_split_p95_on_state",
                    "multiplier": args.conditional_peak_multiplier,
                    "caps_kw": {
                        app: float(conditional_peak_power[idx])
                        for idx, app in enumerate(appliance_cols)
                    },
                    "state_thresholds_kw": bundle.state_thresholds,
                    "rated_power_kw": {
                        app: float(rated_power[idx])
                        for idx, app in enumerate(appliance_cols)
                    },
                },
            )
        model = PISAModel(
            input_dim=len(DEFAULT_INPUT_COLS),
            num_appliances=len(appliance_cols),
            horizon=args.horizon,
            appliance_names=appliance_cols,
            d_model=args.d_model,
            n_heads=args.n_heads,
            encoder_layers=args.encoder_layers,
            use_bridge_residual=args.use_bridge_residual,
            use_tail_disaggregation_head=args.target_mode == "history_tail",
            state_conditioned_power=args.state_conditioned_power,
            state_gate_floor=args.state_gate_floor,
            state_power_blend=args.state_power_blend,
            event_bucket_size=args.event_bucket_size,
            hierarchical_future=args.hierarchical_future,
            hierarchical_power_blend=(
                0.0
                if args.hierarchical_future
                and args.hierarchical_blend_ramp_epochs > 0
                else args.hierarchical_power_blend
            ),
            conditional_peak_power=conditional_peak_power,
            risk_adapter_dim=args.risk_adapter_dim,
            power_adapter_dim=args.power_adapter_dim,
            use_history_recon_refiner=args.use_history_recon_refiner,
            history_recon_gate_init=args.history_recon_gate_init,
            use_refined_history_residual_tcn=(
                args.use_refined_history_residual_tcn
            ),
            residual_tcn_hidden_dim=args.residual_tcn_hidden_dim,
            residual_tcn_num_layers=args.residual_tcn_num_layers,
            residual_tcn_kernel_size=args.residual_tcn_kernel_size,
            residual_tcn_dropout=args.residual_tcn_dropout,
            residual_tcn_history_source=args.residual_tcn_history_source,
            residual_tcn_shared=args.residual_tcn_shared,
            future_residual_fusion=args.future_residual_fusion,
            future_residual_zero_init=args.future_residual_zero_init,
            future_residual_gate_init=args.future_residual_gate_init,
            residual_tcn_input_scale=nominal_rated_power,
            residual_tcn_use_future_context=(
                args.residual_tcn_use_future_context
            ),
            residual_tcn_target_adapter_dim=(
                args.residual_tcn_target_adapter_dim
            ),
            future_base_blend_init=args.future_base_blend_init,
            target_output_calibration=args.target_output_calibration,
            rated_power=rated_power,
        )
    else:
        model = AggregateOnlySeq2SeqBaseline(
            input_dim=len(DEFAULT_INPUT_COLS),
            num_appliances=len(appliance_cols),
            horizon=args.horizon,
            hidden_dim=args.d_model,
            num_layers=max(1, args.encoder_layers),
            rated_power=rated_power,
        )

    state_prior = torch.tensor(
        [stats.on_rate for stats in bundle.appliance_activity_stats],
        dtype=torch.float32,
    )
    if hasattr(model, "set_state_prior"):
        model.set_state_prior(state_prior)
    print("State ON priors:")
    print(state_prior.tolist())

    if args.init_checkpoint is not None:
        load_model_weights(model, args.init_checkpoint)

    if args.model_type == "pisa":
        configure_trainable_parameters(model, args.train_mode)
    else:
        for param in model.parameters():
            param.requires_grad = True
        print("Train mode          : full baseline")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    save_json(
        result_dir / "model_info.json",
        {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_config": experiment_config["model"],
        },
    )

    # -----------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------
    loss_config = PISALossConfig(
        lambda_power=args.lambda_power,
        lambda_bridge=args.lambda_bridge,
        lambda_state=args.lambda_state,
        lambda_event=args.lambda_event,
        lambda_start=args.lambda_start,
        lambda_stop=args.lambda_stop,
        lambda_bucket_start=args.lambda_bucket_start,
        lambda_bucket_stop=args.lambda_bucket_stop,
        lambda_event_offset=args.lambda_event_offset,
        lambda_window_start=args.lambda_window_start,
        lambda_conditional_bucket=args.lambda_conditional_bucket,
        lambda_conditional_offset=args.lambda_conditional_offset,
        lambda_conditional_power=args.lambda_conditional_power,
        lambda_pulse_duration=args.lambda_pulse_duration,
        lambda_pulse_amplitude=args.lambda_pulse_amplitude,
        lambda_agg=args.lambda_agg,
        lambda_ghost=args.lambda_ghost,
        lambda_peak=args.lambda_peak,
        lambda_amp_on=args.lambda_amp_on,
        lambda_orth=args.lambda_orth,
        lambda_recon_power=args.lambda_recon_power,
        lambda_recon_state=args.lambda_recon_state,
        lambda_direct_power=args.lambda_direct_power,
        active_power_weight=args.active_power_weight,
        balanced_on_off_power_loss=args.balanced_on_off_power_loss,
        balanced_on_fraction=args.balanced_on_fraction,
        active_bridge_weight=args.active_bridge_weight,
        active_recon_weight=args.active_recon_weight,
        risk_warmup_epochs=args.risk_warmup_epochs,
        risk_ramp_epochs=args.risk_ramp_epochs,
        base_aux_warmup_epochs=args.base_aux_warmup_epochs,
        base_aux_ramp_epochs=args.base_aux_ramp_epochs,
        base_warmup_lambda_bridge=args.base_warmup_lambda_bridge,
        base_warmup_lambda_state=args.base_warmup_lambda_state,
        base_warmup_lambda_ghost=args.base_warmup_lambda_ghost,
        base_warmup_lambda_peak=args.base_warmup_lambda_peak,
    )

    loss_fn = PISALoss(
        appliance_scales=bundle.appliance_scales,
        config=loss_config,
        state_alpha=[args.state_alpha] * len(appliance_cols),
        event_alpha=[args.event_alpha] * len(appliance_cols),
        stop_alpha=[
            args.event_alpha if args.stop_alpha is None else args.stop_alpha
        ] * len(appliance_cols),
    )

    # -----------------------------------------------------------------
    # Optimizer and scheduler
    # -----------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if args.training_stage == "risk_finetune" else "min",
        factor=0.5,
        patience=(
            args.risk_scheduler_patience
            if args.training_stage == "risk_finetune"
            else 5
        ),
    )

    # -----------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------
    trainer_config = TrainerConfig(
        max_epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        use_amp=not args.no_amp,
        grad_clip_norm=args.grad_clip_norm,
        log_interval=args.log_interval,
        checkpoint_dir=str(checkpoint_dir),
        save_best=True,
        save_last=True,
        monitor=args.monitor,
        monitor_mode="min",
        early_stopping_patience=args.patience,
        min_epochs_before_stopping=args.min_epochs_before_stopping,
        eval_metrics_every=args.eval_metrics_every,
        hierarchical_power_blend=args.hierarchical_power_blend,
        hierarchical_blend_warmup_epochs=args.hierarchical_blend_warmup_epochs,
        hierarchical_blend_ramp_epochs=args.hierarchical_blend_ramp_epochs,
        risk_monitor=(
            RISK_HIERARCHICAL_START_MONITOR
            if args.training_stage == "risk_finetune"
            else None
        ),
        risk_monitor_mode="max",
        risk_mae_monitor=CORE_MAE_MONITOR,
        risk_mae_ceiling=risk_mae_ceiling,
        risk_checkpoint_start_epoch=risk_checkpoint_start_epoch,
        risk_checkpoint_name="risk_best.pt",
        scheduler_monitor=(
            RISK_HIERARCHICAL_START_MONITOR
            if args.training_stage == "risk_finetune"
            else args.monitor
        ),
        scheduler_start_epoch=(
            risk_checkpoint_start_epoch
            if args.training_stage == "risk_finetune"
            else 1
        ),
        event_tolerance_minutes=args.event_tolerance_minutes,
        event_bucket_size=args.event_bucket_size,
        domain_generalization_weight=args.domain_generalization_weight,
        domain_generalization_dispersion_weight=(
            args.domain_generalization_dispersion_weight
        ),
        domain_generalization_warmup_epochs=(
            args.domain_generalization_warmup_epochs
        ),
        domain_generalization_ramp_epochs=args.domain_generalization_ramp_epochs,
        source_on_monitor_weight=args.source_on_monitor_weight,
        source_on_safety_ceiling=args.source_on_safety_ceiling,
        source_on_safety_scope=args.source_on_safety_scope,
        source_on_min_windows=args.source_on_min_windows,
        history_file="history.json",
    )

    trainer = PISATrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=trainer_config,
        appliance_names=appliance_cols,
        state_thresholds=bundle.state_thresholds,
    )

    if args.train_mode == "future_residual_tcn" and not args.evaluate_only:
        initialization_label = (
            "zero-initialized"
            if args.future_residual_zero_init
            else "randomly initialized"
        )
        print(
            f"Evaluating the {initialization_label} future branch before training..."
        )
        initial_val_stats = trainer.validate(
            loaders["val"],
            compute_metrics=True,
            state_thresholds=bundle.state_thresholds,
            event_tolerance_minutes=args.event_tolerance_minutes,
            event_bucket_size=args.event_bucket_size,
            postprocess_state=args.postprocess_state,
            min_on_duration=args.min_on_duration,
            min_off_duration=args.min_off_duration,
        )
        save_json(result_dir / "initial_val_metrics.json", initial_val_stats)

    # -----------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------
    if args.evaluate_only:
        history: list[dict[str, float]] = []
        save_json(result_dir / "history.json", history)
        print("Evaluation-only mode: skipped training.")
    else:
        history = trainer.fit(
            train_loader=loaders["train"],
            val_loader=loaders["val"],
        )
        save_json(result_dir / "history.json", history)

    if args.training_stage == "risk_finetune" and not args.evaluate_only:
        risk_best_record = next(
            (
                record
                for record in history
                if int(record.get("epoch", -1)) == trainer.risk_best_epoch
            ),
            None,
        )
        risk_checkpoint_summary = {
            "checkpoint": str(checkpoint_dir / "risk_best.pt"),
            "exists": (checkpoint_dir / "risk_best.pt").exists(),
            "selection_metric": RISK_HIERARCHICAL_START_MONITOR,
            "selection_metric_value": (
                None
                if trainer.risk_best_epoch == 0
                else trainer.risk_best_metric
            ),
            "selection_epoch": trainer.risk_best_epoch,
            "early_stopping_monitor": RISK_HIERARCHICAL_START_MONITOR,
            "early_stopping_bad_epochs": trainer.risk_num_bad_epochs,
            "risk_checkpoint_start_epoch": risk_checkpoint_start_epoch,
            "mae_monitor": CORE_MAE_MONITOR,
            "mae_ceiling": risk_mae_ceiling,
            "selected_epoch_metrics": risk_best_record,
        }
        save_json(run_dir / "risk_checkpoint_summary.json", risk_checkpoint_summary)
        print("Risk checkpoint summary:")
        print(risk_checkpoint_summary)

    print("=" * 80)
    print("Evaluation setup finished." if args.evaluate_only else "Training finished.")
    if not args.evaluate_only:
        checkpoint_label = (
            "MAE-safe"
            if args.monitor == CORE_MAE_MONITOR
            else f"{args.monitor}-selected"
        )
        print(f"{checkpoint_label} best epoch : {trainer.best_epoch}")
        print(f"{checkpoint_label} best metric: {trainer.best_metric}")
        print(f"{checkpoint_label} checkpoint : {checkpoint_dir / 'best.pt'}")
        if args.training_stage == "risk_finetune":
            print(f"Risk best epoch     : {trainer.risk_best_epoch}")
            print(f"Risk best metric    : {trainer.risk_best_metric}")
            print(f"Risk checkpoint     : {checkpoint_dir / 'risk_best.pt'}")
    print("=" * 80)

    # -----------------------------------------------------------------
    # Final validation and optional test evaluation using best checkpoint
    # -----------------------------------------------------------------
    best_ckpt = (
        Path(args.init_checkpoint).expanduser().resolve()
        if args.evaluate_only
        else checkpoint_dir / "best.pt"
    )

    if best_ckpt.exists():
        trainer.load_checkpoint(
            best_ckpt,
            load_optimizer=False,
            load_scheduler=False,
        )

        eval_state_thresholds = (
            fixed_eval_state_thresholds
            if fixed_eval_state_thresholds is not None
            else bundle.state_thresholds
        )
        eval_event_thresholds = {
            app_name: 0.5
            for app_name in appliance_cols
        }
        eval_bucket_event_thresholds = {
            app_name: 0.10
            for app_name in appliance_cols
        }
        eval_window_start_thresholds = {
            app_name: 0.5
            for app_name in appliance_cols
        }
        eval_hierarchical_start_thresholds = {
            app_name: 0.5
            for app_name in appliance_cols
        }
        threshold_calibration = {
            "enabled": False,
            "state_thresholds": eval_state_thresholds,
            "event_thresholds": eval_event_thresholds,
            "bucket_event_thresholds": eval_bucket_event_thresholds,
            "window_start_thresholds": eval_window_start_thresholds,
            "hierarchical_start_thresholds": eval_hierarchical_start_thresholds,
        }
        if fixed_eval_state_threshold_source is not None:
            threshold_calibration["state_threshold_source"] = str(
                fixed_eval_state_threshold_source
            )

        if args.calibrate_state_thresholds or args.calibrate_event_thresholds:
            print("Collecting validation predictions for threshold calibration...")
            val_acc = collect_metrics_accumulator(
                model=trainer.model,
                loader=loaders["val"],
                device=trainer.device,
                use_amp=trainer.use_amp,
                appliance_names=appliance_cols,
            )
            val_pred_power = torch.cat(val_acc.pred_power_list, dim=0)
            val_true_state = torch.cat(val_acc.true_state_list, dim=0)
            val_true_event = torch.cat(val_acc.true_event_list, dim=0)
            val_true_start = (
                torch.cat(val_acc.true_start_list, dim=0)
                if val_acc.true_start_list
                else val_true_event
            )
            val_previous_state = (
                (
                    torch.cat(val_acc.previous_state_prob_list, dim=0) >= 0.5
                ).float()
                if val_acc.previous_state_prob_list
                else None
            )
            val_previous_power = (
                torch.cat(val_acc.previous_power_list, dim=0)
                if val_acc.previous_power_list
                else None
            )
            val_mask = (
                torch.cat(val_acc.mask_list, dim=0)
                if val_acc.mask_list
                else None
            )

            search_min_on = args.min_on_duration if args.postprocess_state else 1
            search_min_off = args.min_off_duration if args.postprocess_state else 1

            if args.calibrate_state_thresholds and fixed_eval_state_thresholds is None:
                print("Calibrating state thresholds on validation predictions...")
                eval_state_thresholds, threshold_summary = find_best_state_thresholds(
                    pred_power=val_pred_power,
                    true_state=val_true_state,
                    true_event=val_true_event,
                    previous_state=val_previous_state,
                    previous_power=val_previous_power,
                    appliance_names=appliance_cols,
                    mask=val_mask,
                    num_thresholds=args.threshold_search_steps,
                    min_threshold=args.threshold_min,
                    max_threshold=args.threshold_max,
                    min_on_duration=search_min_on,
                    min_off_duration=search_min_off,
                    objective=args.threshold_objective,
                )

                threshold_calibration["state_thresholds"] = eval_state_thresholds
                threshold_calibration["summary"] = threshold_summary
                threshold_calibration["num_thresholds"] = args.threshold_search_steps
                threshold_calibration["min_threshold"] = args.threshold_min
                threshold_calibration["max_threshold"] = args.threshold_max
                threshold_calibration["objective"] = args.threshold_objective
                threshold_calibration["postprocess_state"] = args.postprocess_state
                threshold_calibration["min_on_duration"] = search_min_on
                threshold_calibration["min_off_duration"] = search_min_off

                print("Calibrated state thresholds:")
                print(eval_state_thresholds)

            if args.calibrate_event_thresholds:
                if val_acc.event_logits_list:
                    print("Calibrating event-head thresholds on validation predictions...")
                    val_event_logits = torch.cat(val_acc.event_logits_list, dim=0)
                    eval_event_thresholds, event_threshold_summary = (
                        find_best_event_thresholds(
                            event_logits=val_event_logits,
                            true_event=val_true_event,
                            appliance_names=appliance_cols,
                            mask=val_mask,
                            num_thresholds=args.event_threshold_search_steps,
                            min_threshold=args.event_threshold_min,
                            max_threshold=args.event_threshold_max,
                        )
                    )
                    threshold_calibration["event_thresholds"] = eval_event_thresholds
                    threshold_calibration["event_summary"] = event_threshold_summary
                    threshold_calibration["event_num_thresholds"] = (
                        args.event_threshold_search_steps
                    )
                    threshold_calibration["event_min_threshold"] = (
                        args.event_threshold_min
                    )
                    threshold_calibration["event_max_threshold"] = (
                        args.event_threshold_max
                    )

                    print("Calibrated event thresholds:")
                    print(eval_event_thresholds)
                else:
                    print("No event logits available; skipping event calibration.")

                if val_acc.bucket_event_logits_list:
                    print("Calibrating bucket event-head thresholds on validation predictions...")
                    val_bucket_event_logits = torch.cat(
                        val_acc.bucket_event_logits_list,
                        dim=0,
                    )
                    (
                        eval_bucket_event_thresholds,
                        bucket_event_threshold_summary,
                    ) = find_best_event_thresholds(
                        event_logits=val_bucket_event_logits,
                        true_event=val_true_event,
                        appliance_names=appliance_cols,
                        mask=val_mask,
                        num_thresholds=args.event_threshold_search_steps,
                        min_threshold=0.01,
                        max_threshold=args.event_threshold_max,
                    )
                    threshold_calibration["bucket_event_thresholds"] = (
                        eval_bucket_event_thresholds
                    )
                    threshold_calibration["bucket_event_summary"] = (
                        bucket_event_threshold_summary
                    )
                    print("Calibrated bucket event thresholds:")
                    print(eval_bucket_event_thresholds)

                if val_acc.window_start_prob_list:
                    print("Calibrating 30-minute window-start thresholds...")
                    val_window_start_prob = torch.cat(
                        val_acc.window_start_prob_list,
                        dim=0,
                    ).clamp(1e-5, 1.0 - 1e-5)
                    val_window_start_logits = torch.logit(
                        val_window_start_prob
                    ).unsqueeze(-1)
                    val_window_start_target = (
                        val_true_start.amax(dim=-1, keepdim=True) > 0.5
                    ).float()
                    (
                        eval_window_start_thresholds,
                        window_start_threshold_summary,
                    ) = find_best_event_thresholds(
                        event_logits=val_window_start_logits,
                        true_event=val_window_start_target,
                        appliance_names=appliance_cols,
                        mask=None,
                        num_thresholds=args.event_threshold_search_steps,
                        min_threshold=0.01,
                        max_threshold=args.event_threshold_max,
                    )
                    threshold_calibration["window_start_thresholds"] = (
                        eval_window_start_thresholds
                    )
                    threshold_calibration["window_start_summary"] = (
                        window_start_threshold_summary
                    )
                    print("Calibrated window-start thresholds:")
                    print(eval_window_start_thresholds)

                if (
                    val_acc.window_start_prob_list
                    and val_acc.conditional_start_prob_list
                ):
                    print("Calibrating hierarchical one-start thresholds...")
                    (
                        eval_hierarchical_start_thresholds,
                        hierarchical_start_threshold_summary,
                    ) = find_best_hierarchical_start_thresholds(
                        window_start_prob=torch.cat(
                            val_acc.window_start_prob_list,
                            dim=0,
                        ),
                        conditional_start_prob=torch.cat(
                            val_acc.conditional_start_prob_list,
                            dim=0,
                        ),
                        true_start=val_true_start,
                        appliance_names=appliance_cols,
                        mask=val_mask,
                        num_thresholds=args.event_threshold_search_steps,
                        min_threshold=0.01,
                        max_threshold=args.event_threshold_max,
                        tolerance=args.event_tolerance_minutes,
                    )
                    threshold_calibration["hierarchical_start_thresholds"] = (
                        eval_hierarchical_start_thresholds
                    )
                    threshold_calibration["hierarchical_start_summary"] = (
                        hierarchical_start_threshold_summary
                    )
                    threshold_calibration[
                        "hierarchical_start_selection_tolerance_minutes"
                    ] = args.event_tolerance_minutes
                    print("Calibrated hierarchical one-start thresholds:")
                    print(eval_hierarchical_start_thresholds)

            threshold_calibration["enabled"] = bool(
                args.calibrate_state_thresholds or args.calibrate_event_thresholds
            )

            save_json(
                result_dir / "calibrated_state_thresholds.json",
                threshold_calibration,
            )

        val_stats = trainer.validate(
            loaders["val"],
            compute_metrics=True,
            state_thresholds=eval_state_thresholds,
            event_prob_threshold=eval_event_thresholds,
            bucket_event_prob_threshold=eval_bucket_event_thresholds,
            window_start_prob_threshold=eval_window_start_thresholds,
            hierarchical_start_prob_threshold=eval_hierarchical_start_thresholds,
            event_tolerance_minutes=args.event_tolerance_minutes,
            event_bucket_size=args.event_bucket_size,
            postprocess_state=args.postprocess_state,
            min_on_duration=args.min_on_duration,
            min_off_duration=args.min_off_duration,
        )
        val_stats["calibration/enabled"] = float(threshold_calibration["enabled"])
        for app_name, threshold in eval_state_thresholds.items():
            val_stats[f"calibration/{app_name}/state_threshold"] = float(threshold)
        for app_name, threshold in eval_event_thresholds.items():
            val_stats[f"calibration/{app_name}/event_probability_threshold"] = float(
                threshold
            )
        for app_name, threshold in eval_bucket_event_thresholds.items():
            val_stats[
                f"calibration/{app_name}/bucket_event_probability_threshold"
            ] = float(threshold)
        for app_name, threshold in eval_window_start_thresholds.items():
            val_stats[
                f"calibration/{app_name}/window_start_probability_threshold"
            ] = float(threshold)
        for app_name, threshold in eval_hierarchical_start_thresholds.items():
            val_stats[
                f"calibration/{app_name}/hierarchical_start_probability_threshold"
            ] = float(threshold)

        save_json(result_dir / "best_val_metrics.json", val_stats)

        if source_home_loaders:
            per_home_validation: dict[str, dict[str, float]] = {}
            for home_id, split_loaders in source_home_loaders.items():
                if "val" not in split_loaders:
                    continue
                home_stats = trainer.validate(
                    split_loaders["val"],
                    compute_metrics=True,
                    state_thresholds=eval_state_thresholds,
                    event_prob_threshold=eval_event_thresholds,
                    bucket_event_prob_threshold=eval_bucket_event_thresholds,
                    window_start_prob_threshold=eval_window_start_thresholds,
                    hierarchical_start_prob_threshold=(
                        eval_hierarchical_start_thresholds
                    ),
                    event_tolerance_minutes=args.event_tolerance_minutes,
                    event_bucket_size=args.event_bucket_size,
                    postprocess_state=args.postprocess_state,
                    min_on_duration=args.min_on_duration,
                    min_off_duration=args.min_off_duration,
                )
                per_home_validation[home_id] = home_stats
                save_json(
                    result_dir / f"source_home_{home_id}_val_metrics.json",
                    home_stats,
                )
            save_json(
                result_dir / "source_home_validation_summary.json",
                per_home_validation,
            )

        stage_summary = build_stage_summary(
            training_stage=args.training_stage,
            monitor=args.monitor,
            best_epoch=trainer.best_epoch,
            best_metric=trainer.best_metric,
            val_stats=val_stats,
            initialization=initialization,
        )
        save_json(run_dir / "stage_summary.json", stage_summary)

        print("Best validation metrics saved to:")
        print(result_dir / "best_val_metrics.json")
        print("Stage summary saved to:")
        print(run_dir / "stage_summary.json")
        if (
            args.training_stage == "base"
            and args.train_mode != "history_reconstruction"
        ):
            print(
                "Base-stage readiness for risk fine-tune: "
                f"{stage_summary['ready_for_risk_finetune']}"
            )
        elif args.train_mode == "history_reconstruction":
            print(
                "Risk-finetune readiness: not applicable to the "
                "history-reconstruction-only stage."
            )

        if (
            "test" in loaders
            and not args.skip_test_evaluation
            and args.training_stage != "risk_finetune"
        ):
            test_stats = trainer.validate(
                loaders["test"],
                compute_metrics=True,
                state_thresholds=eval_state_thresholds,
                event_prob_threshold=eval_event_thresholds,
                bucket_event_prob_threshold=eval_bucket_event_thresholds,
                window_start_prob_threshold=eval_window_start_thresholds,
                hierarchical_start_prob_threshold=eval_hierarchical_start_thresholds,
                event_tolerance_minutes=args.event_tolerance_minutes,
                event_bucket_size=args.event_bucket_size,
                postprocess_state=args.postprocess_state,
                min_on_duration=args.min_on_duration,
                min_off_duration=args.min_off_duration,
            )
            test_stats["calibration/enabled"] = float(
                threshold_calibration["enabled"]
            )
            for app_name, threshold in eval_state_thresholds.items():
                test_stats[f"calibration/{app_name}/state_threshold"] = float(
                    threshold
                )
            for app_name, threshold in eval_event_thresholds.items():
                test_stats[
                    f"calibration/{app_name}/event_probability_threshold"
                ] = float(threshold)
            for app_name, threshold in eval_bucket_event_thresholds.items():
                test_stats[
                    f"calibration/{app_name}/bucket_event_probability_threshold"
                ] = float(threshold)
            for app_name, threshold in eval_window_start_thresholds.items():
                test_stats[
                    f"calibration/{app_name}/window_start_probability_threshold"
                ] = float(threshold)
            for app_name, threshold in eval_hierarchical_start_thresholds.items():
                test_stats[
                    f"calibration/{app_name}/hierarchical_start_probability_threshold"
                ] = float(threshold)

            save_json(result_dir / "best_test_metrics.json", test_stats)

            print("Best test metrics saved to:")
            print(result_dir / "best_test_metrics.json")
        elif args.skip_test_evaluation:
            print("Test evaluation skipped: validation-only run.")
        elif args.training_stage == "risk_finetune":
            print(
                "Generic best-checkpoint test evaluation skipped for the risk "
                "stage; only validation-selected risk_best.pt is evaluated on test."
            )

        if (
            held_out_loaders is not None
            and "test" in held_out_loaders
            and not args.skip_test_evaluation
            and args.training_stage != "risk_finetune"
        ):
            held_out_stats = trainer.validate(
                held_out_loaders["test"],
                compute_metrics=True,
                state_thresholds=eval_state_thresholds,
                event_prob_threshold=eval_event_thresholds,
                bucket_event_prob_threshold=eval_bucket_event_thresholds,
                window_start_prob_threshold=eval_window_start_thresholds,
                hierarchical_start_prob_threshold=(
                    eval_hierarchical_start_thresholds
                ),
                event_tolerance_minutes=args.event_tolerance_minutes,
                event_bucket_size=args.event_bucket_size,
                postprocess_state=args.postprocess_state,
                min_on_duration=args.min_on_duration,
                min_off_duration=args.min_off_duration,
            )
            model_mae = held_out_stats.get("regression/macro_avg/MAE")
            zero_mae = held_out_stats.get(
                "baseline_zero/regression/macro_avg/MAE"
            )
            if (
                isinstance(model_mae, (int, float))
                and isinstance(zero_mae, (int, float))
                and float(zero_mae) > 0.0
            ):
                normalized_mae = float(model_mae) / float(zero_mae)
                held_out_stats["generalization/normalized_MAE_vs_zero"] = (
                    normalized_mae
                )
                held_out_stats["generalization/skill_vs_zero"] = (
                    1.0 - normalized_mae
                )
            save_json(result_dir / "held_out_test_metrics.json", held_out_stats)
            save_json(
                result_dir / "held_out_protocol.json",
                {
                    "target_csv": str(held_out_csv_path),
                    "selection_data": "source-home validation splits only",
                    "target_train_used": False,
                    "target_validation_used": False,
                    "target_test_evaluations": 1,
                    "source_scalers_only": True,
                },
            )
            print("Held-out-home test metrics saved to:")
            print(result_dir / "held_out_test_metrics.json")

        risk_ckpt = (
            Path(args.init_checkpoint).expanduser().resolve()
            if args.evaluate_only
            else checkpoint_dir / "risk_best.pt"
        )
        if args.training_stage == "risk_finetune" and risk_ckpt.exists():
            print("=" * 80)
            print("Evaluating risk_best.pt with risk-specific threshold calibration...")
            trainer.load_checkpoint(
                risk_ckpt,
                load_optimizer=False,
                load_scheduler=False,
            )
            (
                risk_event_thresholds,
                risk_bucket_thresholds,
                risk_window_thresholds,
                risk_hierarchical_start_thresholds,
                risk_threshold_calibration,
            ) = calibrate_risk_event_thresholds(
                model=trainer.model,
                loader=loaders["val"],
                device=trainer.device,
                use_amp=trainer.use_amp,
                appliance_names=appliance_cols,
                num_thresholds=args.event_threshold_search_steps,
                min_threshold=args.event_threshold_min,
                max_threshold=args.event_threshold_max,
                hierarchical_tolerance_minutes=args.event_tolerance_minutes,
            )
            risk_threshold_calibration["state_thresholds"] = eval_state_thresholds
            risk_threshold_calibration["state_threshold_source"] = (
                "mae_safe_best_checkpoint"
            )
            save_json(
                result_dir / "risk_best_calibrated_thresholds.json",
                risk_threshold_calibration,
            )

            risk_val_stats = trainer.validate(
                loaders["val"],
                compute_metrics=True,
                state_thresholds=eval_state_thresholds,
                event_prob_threshold=risk_event_thresholds,
                bucket_event_prob_threshold=risk_bucket_thresholds,
                window_start_prob_threshold=risk_window_thresholds,
                hierarchical_start_prob_threshold=risk_hierarchical_start_thresholds,
                event_tolerance_minutes=args.event_tolerance_minutes,
                event_bucket_size=args.event_bucket_size,
                postprocess_state=args.postprocess_state,
                min_on_duration=args.min_on_duration,
                min_off_duration=args.min_off_duration,
            )
            risk_val_stats["calibration/enabled"] = 1.0
            for app_name, threshold in risk_event_thresholds.items():
                risk_val_stats[
                    f"calibration/{app_name}/event_probability_threshold"
                ] = float(threshold)
            for app_name, threshold in risk_bucket_thresholds.items():
                risk_val_stats[
                    f"calibration/{app_name}/bucket_event_probability_threshold"
                ] = float(threshold)
            for app_name, threshold in risk_window_thresholds.items():
                risk_val_stats[
                    f"calibration/{app_name}/window_start_probability_threshold"
                ] = float(threshold)
            for app_name, threshold in risk_hierarchical_start_thresholds.items():
                risk_val_stats[
                    f"calibration/{app_name}/hierarchical_start_probability_threshold"
                ] = float(threshold)
            save_json(result_dir / "risk_best_val_metrics.json", risk_val_stats)

            risk_test_stats = None
            # Keep risk-stage model selection and threshold calibration strictly
            # validation-only when requested.  The generic best-checkpoint path
            # above already honours --skip_test_evaluation; the risk_best path
            # must do the same or it silently consumes the held-out test split.
            if "test" in loaders and not args.skip_test_evaluation:
                risk_test_stats = trainer.validate(
                    loaders["test"],
                    compute_metrics=True,
                    state_thresholds=eval_state_thresholds,
                    event_prob_threshold=risk_event_thresholds,
                    bucket_event_prob_threshold=risk_bucket_thresholds,
                    window_start_prob_threshold=risk_window_thresholds,
                    hierarchical_start_prob_threshold=risk_hierarchical_start_thresholds,
                    event_tolerance_minutes=args.event_tolerance_minutes,
                    event_bucket_size=args.event_bucket_size,
                    postprocess_state=args.postprocess_state,
                    min_on_duration=args.min_on_duration,
                    min_off_duration=args.min_off_duration,
                )
                risk_test_stats["calibration/enabled"] = 1.0
                save_json(result_dir / "risk_best_test_metrics.json", risk_test_stats)

            risk_summary_path = run_dir / "risk_checkpoint_summary.json"
            risk_summary = (
                load_json(risk_summary_path)
                if risk_summary_path.exists()
                else {
                    "checkpoint": str(risk_ckpt),
                    "exists": True,
                    "selection_metric": "not_reselected_evaluation_only",
                    "selection_metric_value": None,
                    "selection_epoch": None,
                    "mae_monitor": CORE_MAE_MONITOR,
                    "mae_ceiling": None,
                }
            )
            risk_summary["risk_thresholds"] = str(
                result_dir / "risk_best_calibrated_thresholds.json"
            )
            risk_summary["calibrated_val_event_f1"] = risk_val_stats.get(
                "event/macro_avg/EventF1"
            )
            risk_summary["calibrated_test_event_f1"] = (
                None
                if risk_test_stats is None
                else risk_test_stats.get("event/macro_avg/EventF1")
            )
            risk_summary["calibrated_val_hierarchical_start_f1"] = (
                risk_val_stats.get("hierarchical_start/macro_avg/EventF1")
            )
            risk_summary["calibrated_val_hierarchical_start_tolerant_f1"] = (
                risk_val_stats.get(
                    "hierarchical_start_tolerant/macro_avg/EventF1"
                )
            )
            risk_summary["calibrated_test_hierarchical_start_f1"] = (
                None
                if risk_test_stats is None
                else risk_test_stats.get("hierarchical_start/macro_avg/EventF1")
            )
            risk_summary[
                "calibrated_test_hierarchical_start_tolerant_f1"
            ] = (
                None
                if risk_test_stats is None
                else risk_test_stats.get(
                    "hierarchical_start_tolerant/macro_avg/EventF1"
                )
            )
            save_json(risk_summary_path, risk_summary)
            print("Risk validation metrics saved to:")
            print(result_dir / "risk_best_val_metrics.json")
            if risk_test_stats is not None:
                print("Risk test metrics saved to:")
                print(result_dir / "risk_best_test_metrics.json")

    print("Done.")


if __name__ == "__main__":
    main()
