from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(ROOT))
sys.path.append(str(SRC))

from data import (  # noqa: E402
    build_multi_home_dataloaders,
    build_single_home_dataloaders,
)
from evaluation.metrics import MetricsAccumulator, flatten_metrics  # noqa: E402
from models import (  # noqa: E402
    AggregateToApplianceTCN,
    ApplianceHistoryTCNForecaster,
    HistoricalNILMSeq2Seq,
    TwoStageNILMForecastPipeline,
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
from training.losses import apply_appliance_scale, masked_l1_loss  # noqa: E402
from scripts.compare_historical_reconstruction import evaluate_history  # noqa: E402


INPUT_COLS = [
    "grid",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
]
APPLIANCES = [
    "air1",
    "refrigerator1",
    "dishwasher1",
    "microwave1",
]
DEFAULT_STATE_THRESHOLDS = {
    "air1": 0.50,
    "refrigerator1": 0.05,
    "dishwasher1": 0.05,
    "microwave1": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a genuine disaggregate-then-forecast baseline for revision "
            "experiments."
        )
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=str(
            ROOT
            / "data"
            / "austin_2018_sep_4homes"
            / "home_7951_2018_sep_1min.csv"
        ),
    )
    parser.add_argument(
        "--source_csv_paths",
        nargs="+",
        default=None,
        help="Two or more source-home CSVs for balanced multi-home training.",
    )
    parser.add_argument(
        "--held_out_csv_path",
        type=str,
        default=None,
        help="Unseen target home evaluated once after source validation selection.",
    )
    parser.add_argument(
        "--home_balanced_sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--run_name", type=str, default="home7951_genuine_two_stage")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT / "outputs" / "runs"),
    )
    parser.add_argument("--input_window", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--nilm_num_layers", type=int, default=2)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--nilm_epochs", type=int, default=60)
    parser.add_argument("--forecast_epochs", type=int, default=80)
    parser.add_argument("--direct_epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument(
        "--min_epochs_before_stopping",
        type=int,
        default=0,
        help=(
            "Do not early-stop NILM or future forecasters before this epoch. "
            "Use 20 to match the current PISA forecasting protocol."
        ),
    )
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--protocol_reference_config",
        type=str,
        default=None,
        help=(
            "Optional PISA config.json used by a home-specific wrapper to inherit "
            "the exact data/evaluation protocol. The resolved path and SHA256 are "
            "stored for auditability."
        ),
    )
    parser.add_argument("--lambda_nilm_power", type=float, default=1.0)
    parser.add_argument("--lambda_nilm_state", type=float, default=0.25)
    parser.add_argument("--lambda_power", type=float, default=1.0)
    parser.add_argument("--lambda_state", type=float, default=0.35)
    parser.add_argument("--lambda_event", type=float, default=0.25)
    parser.add_argument("--lambda_start", type=float, default=0.25)
    parser.add_argument("--lambda_stop", type=float, default=0.25)
    parser.add_argument("--lambda_peak", type=float, default=0.20)
    parser.add_argument("--lambda_ghost", type=float, default=0.02)
    parser.add_argument("--active_power_weight", type=float, default=3.0)
    parser.add_argument("--state_alpha", type=float, default=0.85)
    parser.add_argument("--event_alpha", type=float, default=0.90)
    parser.add_argument(
        "--rated_power_kw",
        nargs=len(APPLIANCES),
        type=float,
        default=None,
        metavar=("AIR", "FRIDGE", "DISHWASHER", "MICROWAVE"),
        help=(
            "Explicit rated-power caps in appliance order. Omit to use the "
            "repository defaults [3.0, 0.5, 2.0, 2.5] kW."
        ),
    )
    parser.add_argument(
        "--state_thresholds_kw",
        nargs=len(APPLIANCES),
        type=float,
        default=None,
        metavar=("AIR", "FRIDGE", "DISHWASHER", "MICROWAVE"),
        help=(
            "ON-state power thresholds in appliance order. Omit to use "
            "[0.50, 0.05, 0.05, 0.10] kW."
        ),
    )
    parser.add_argument("--event_tolerance_minutes", type=int, default=2)
    parser.add_argument("--event_bucket_size", type=int, default=5)
    parser.add_argument(
        "--postprocess_state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply minimum-duration state postprocessing during final reporting. "
            "Use --no-postprocess_state for the current PISA matched protocol."
        ),
    )
    parser.add_argument("--min_on_duration", type=int, default=3)
    parser.add_argument("--min_off_duration", type=int, default=2)
    parser.add_argument(
        "--skip_test_evaluation",
        action="store_true",
        help="Select on validation only; do not touch the held-out test split.",
    )
    parser.add_argument(
        "--skip_oracle_forecaster",
        action="store_true",
        help="Skip the diagnostic ground-truth-history Oracle TCN stage.",
    )
    parser.add_argument(
        "--resume_completed_stages",
        action="store_true",
        help=(
            "Reuse completed stage checkpoints/results in an existing run. "
            "A partially trained forecast stage resumes from last.pt; missing "
            "or incomplete NILM training restarts because that stage has no "
            "optimizer-bearing last checkpoint."
        ),
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return json_safe(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(value), f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


RESUME_COMPATIBILITY_KEYS = (
    "csv_path",
    "source_csv_paths",
    "held_out_csv_path",
    "home_balanced_sampling",
    "input_window",
    "horizon",
    "stride",
    "batch_size",
    "hidden_dim",
    "num_layers",
    "nilm_num_layers",
    "kernel_size",
    "dropout",
    "nilm_epochs",
    "forecast_epochs",
    "direct_epochs",
    "lr",
    "weight_decay",
    "patience",
    "min_epochs_before_stopping",
    "grad_clip_norm",
    "seed",
    "lambda_nilm_power",
    "lambda_nilm_state",
    "lambda_power",
    "lambda_state",
    "lambda_event",
    "lambda_start",
    "lambda_stop",
    "lambda_peak",
    "lambda_ghost",
    "active_power_weight",
    "state_alpha",
    "event_alpha",
    "rated_power_kw",
    "state_thresholds_kw",
    "event_tolerance_minutes",
    "event_bucket_size",
    "postprocess_state",
    "min_on_duration",
    "min_off_duration",
    "skip_test_evaluation",
    "skip_oracle_forecaster",
)


def validate_resume_config(config_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(
            "Cannot resume safely because the original run config is missing: "
            f"{config_path}"
        )
    config = load_json(config_path)
    saved_args = config.get("args")
    if not isinstance(saved_args, dict):
        raise ValueError(f"Run config has no args mapping: {config_path}")
    mismatches: list[str] = []
    for key in RESUME_COMPATIBILITY_KEYS:
        if key not in saved_args:
            continue
        current = getattr(args, key)
        saved = saved_args[key]
        if isinstance(current, tuple):
            current = list(current)
        if current != saved:
            mismatches.append(f"{key}: saved={saved!r}, requested={current!r}")
    if mismatches:
        detail = "\n  ".join(mismatches)
        raise ValueError(
            "Refusing to resume with a different training protocol:\n  " + detail
        )
    if list(config.get("appliances", [])) != APPLIANCES:
        raise ValueError(
            "Refusing to resume because the saved appliance order differs from "
            f"the current four-appliance protocol: {config.get('appliances')!r}"
        )
    print(f"Resume protocol verified against: {config_path}")
    return config


def nilm_loss(
    out: dict[str, Tensor],
    batch: dict[str, Tensor],
    appliance_scales: Tensor,
    lambda_power: float,
    lambda_state: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    target_power_scaled = batch["y_hist_power_scaled"].to(
        device=out["past_power"].device,
        dtype=out["past_power"].dtype,
    )
    target_state = batch["y_hist_state"].to(
        device=out["past_power"].device,
        dtype=out["past_power"].dtype,
    )
    hist_mask = batch.get("hist_mask", None)
    if hist_mask is not None:
        hist_mask = hist_mask.to(device=out["past_power"].device, dtype=out["past_power"].dtype)

    pred_scaled = apply_appliance_scale(out["past_power"], appliance_scales)
    loss_power = masked_l1_loss(pred_scaled, target_power_scaled, mask=hist_mask)
    state_loss = F.binary_cross_entropy_with_logits(
        out["past_state_logits"],
        target_state,
        reduction="none",
    )
    if hist_mask is not None:
        state_loss = (state_loss * hist_mask).sum() / hist_mask.sum().clamp_min(1.0)
    else:
        state_loss = state_loss.mean()
    total = lambda_power * loss_power + lambda_state * state_loss
    return total, {
        "loss_total": total.detach(),
        "loss_power": loss_power.detach(),
        "loss_state": state_loss.detach(),
    }


@torch.no_grad()
def evaluate_nilm(
    model: HistoricalNILMSeq2Seq,
    loader,
    device: torch.device,
    appliance_scales: Tensor,
    lambda_power: float,
    lambda_state: float,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        _, loss_dict = nilm_loss(
            out,
            batch,
            appliance_scales=appliance_scales,
            lambda_power=lambda_power,
            lambda_state=lambda_state,
        )
        batch_size = int(batch["x_hist"].size(0))
        count += batch_size
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_nilm(
    model: HistoricalNILMSeq2Seq,
    loaders,
    device: torch.device,
    checkpoint_path: Path,
    appliance_scales: Tensor,
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    model.to(device)
    appliance_scales = appliance_scales.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(not args.no_amp and device.type == "cuda"))
    best_val = float("inf")
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.nilm_epochs + 1):
        model.train()
        start_time = time.time()
        totals: dict[str, float] = {}
        count = 0
        for batch in loaders["train"]:
            batch = move_batch_to_device(batch, device)
            batch_size = int(batch["x_hist"].size(0))
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(not args.no_amp and device.type == "cuda")):
                out = model(batch)
                loss, loss_dict = nilm_loss(
                    out,
                    batch,
                    appliance_scales=appliance_scales,
                    lambda_power=args.lambda_nilm_power,
                    lambda_state=args.lambda_nilm_state,
                )
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
            count += batch_size
            for key, value in loss_dict.items():
                totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size

        if count == 0:
            raise RuntimeError(
                "The training loader produced no batches. Reduce --stride or "
                "--batch_size (the training loader drops an incomplete batch)."
            )

        train_stats = {
            f"train/{key}": value / max(count, 1)
            for key, value in totals.items()
        }
        val_stats_raw = evaluate_nilm(
            model,
            loaders["val"],
            device=device,
            appliance_scales=appliance_scales,
            lambda_power=args.lambda_nilm_power,
            lambda_state=args.lambda_nilm_state,
        )
        val_stats = {f"val/{key}": value for key, value in val_stats_raw.items()}
        record = {
            "epoch": float(epoch),
            **train_stats,
            **val_stats,
            "epoch_time_sec": time.time() - start_time,
        }
        history.append(record)
        val_loss = val_stats["val/loss_total"]
        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val,
                    "args": vars(args),
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1

        print(
            f"[NILM {epoch:03d}] "
            f"train={record['train/loss_total']:.6f} "
            f"val={val_loss:.6f} best={best_val:.6f} bad={bad_epochs}"
        )
        if (
            epoch >= args.min_epochs_before_stopping
            and bad_epochs >= args.patience
        ):
            break

    return history


@torch.no_grad()
def evaluate_forecast_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    appliance_names: list[str],
    state_thresholds: dict[str, float],
    event_tolerance_minutes: int,
    event_bucket_size: int,
    postprocess_state: bool,
    min_on_duration: int,
    min_off_duration: int,
) -> dict[str, Any]:
    model.to(device)
    model.eval()
    acc = MetricsAccumulator(appliance_names=appliance_names)
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        acc.update(out, batch)
    metrics = acc.compute(
        state_thresholds=state_thresholds,
        event_tolerance_minutes=event_tolerance_minutes,
        event_bucket_size=event_bucket_size,
        postprocess_state=postprocess_state,
        min_on_duration=min_on_duration,
        min_off_duration=min_off_duration,
    )
    return {
        "nested": metrics,
        "flat": flatten_metrics(metrics),
    }


@torch.no_grad()
def evaluate_trivial_baseline(
    method: str,
    loader,
    device: torch.device,
    appliance_names: list[str],
    state_thresholds: dict[str, float],
    horizon: int,
    event_tolerance_minutes: int,
    event_bucket_size: int,
    postprocess_state: bool,
    min_on_duration: int,
    min_off_duration: int,
) -> dict[str, Any]:
    """Evaluate label-free Zero or diagnostic GT-history Persistence."""
    if method not in {"zero", "persistence"}:
        raise ValueError(f"Unknown trivial baseline: {method}")
    acc = MetricsAccumulator(appliance_names=appliance_names)
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        if method == "zero":
            pred = torch.zeros_like(batch["y_power"])
        else:
            if "y_hist_power" not in batch:
                raise KeyError("Persistence requires ground-truth y_hist_power.")
            pred = batch["y_hist_power"][..., -1:].expand(-1, -1, horizon)
        acc.update({"y_power": pred}, batch)
    metrics = acc.compute(
        state_thresholds=state_thresholds,
        event_tolerance_minutes=event_tolerance_minutes,
        event_bucket_size=event_bucket_size,
        postprocess_state=postprocess_state,
        min_on_duration=min_on_duration,
        min_off_duration=min_off_duration,
    )
    return {"nested": metrics, "flat": flatten_metrics(metrics)}


def make_forecast_trainer(
    model: torch.nn.Module,
    checkpoint_dir: Path,
    loss_config: PISALossConfig,
    appliance_scales: Tensor,
    bundle,
    args: argparse.Namespace,
    max_epochs: int,
) -> PISATrainer:
    loss_fn = PISALoss(
        appliance_scales=appliance_scales,
        config=loss_config,
        state_alpha=[args.state_alpha] * len(APPLIANCES),
        event_alpha=[args.event_alpha] * len(APPLIANCES),
        stop_alpha=[args.event_alpha] * len(APPLIANCES),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    return PISATrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=TrainerConfig(
            max_epochs=max_epochs,
            device=str(device_from_arg(args)),
            seed=args.seed,
            use_amp=not args.no_amp,
            grad_clip_norm=args.grad_clip_norm,
            checkpoint_dir=str(checkpoint_dir),
            monitor="val/regression/macro_avg/MAE",
            monitor_mode="min",
            early_stopping_patience=args.patience,
            min_epochs_before_stopping=args.min_epochs_before_stopping,
            eval_metrics_every=1,
            event_tolerance_minutes=args.event_tolerance_minutes,
            event_bucket_size=args.event_bucket_size,
        ),
        appliance_names=APPLIANCES,
        state_thresholds=bundle.state_thresholds,
    )


def fit_or_resume_forecast_stage(
    *,
    label: str,
    trainer: PISATrainer,
    train_loader,
    val_loader,
    result_history_path: Path,
    resume: bool,
) -> list[dict[str, float]]:
    """Finish one forecast stage without repeating completed work."""
    best_path = trainer.checkpoint_dir / "best.pt"
    last_path = trainer.checkpoint_dir / "last.pt"
    checkpoint_history_path = trainer.checkpoint_dir / trainer.cfg.history_file

    if resume and result_history_path.is_file():
        if not best_path.is_file():
            raise FileNotFoundError(
                f"{label} has a completion marker but no best checkpoint: {best_path}"
            )
        history = load_json(result_history_path)
        if not isinstance(history, list):
            raise ValueError(f"Expected a history list: {result_history_path}")
        trainer.history = history
        trainer.load_checkpoint(
            best_path,
            load_optimizer=False,
            load_scheduler=False,
        )
        print(f"Resume: skipped completed {label} stage.")
        return history

    if resume and last_path.is_file():
        checkpoint = trainer.load_checkpoint(
            last_path,
            load_optimizer=True,
            load_scheduler=True,
        )
        if checkpoint_history_path.is_file():
            saved_history = load_json(checkpoint_history_path)
            if isinstance(saved_history, list):
                trainer.history = saved_history
        last_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = last_epoch + 1
        stopped_early = (
            last_epoch >= int(trainer.cfg.min_epochs_before_stopping)
            and int(trainer.num_bad_epochs)
            >= int(trainer.cfg.early_stopping_patience)
        )
        exhausted_budget = last_epoch >= int(trainer.cfg.max_epochs)
        if stopped_early or exhausted_budget:
            reason = "early stopping" if stopped_early else "epoch budget"
            print(
                f"Resume: {label} training was already complete by {reason} "
                f"at epoch {last_epoch}; finalizing its saved checkpoint."
            )
        else:
            print(
                f"Resume: continuing interrupted {label} stage from "
                f"epoch {start_epoch}."
            )
            trainer.fit(train_loader, val_loader, start_epoch=start_epoch)
    else:
        trainer.fit(train_loader, val_loader)

    if not best_path.is_file():
        raise FileNotFoundError(f"{label} did not produce a best checkpoint: {best_path}")
    save_json(result_history_path, trainer.history)
    trainer.load_checkpoint(
        best_path,
        load_optimizer=False,
        load_scheduler=False,
    )
    return trainer.history


def device_from_arg(args: argparse.Namespace) -> torch.device:
    return torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = device_from_arg(args)
    source_csv_paths = (
        [Path(value).expanduser().resolve() for value in args.source_csv_paths]
        if args.source_csv_paths
        else [Path(args.csv_path).expanduser().resolve()]
    )
    if len(set(source_csv_paths)) != len(source_csv_paths):
        raise ValueError("Duplicate source CSV paths are not allowed.")
    for path in source_csv_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    csv_path = source_csv_paths[0]
    held_out_csv_path = (
        None
        if args.held_out_csv_path is None
        else Path(args.held_out_csv_path).expanduser().resolve()
    )
    if held_out_csv_path is not None:
        if not held_out_csv_path.exists():
            raise FileNotFoundError(held_out_csv_path)
        if held_out_csv_path in source_csv_paths:
            raise ValueError("The held-out home cannot also be a source home.")
    run_dir = Path(args.output_dir).expanduser().resolve() / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    result_dir = run_dir / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.min_epochs_before_stopping < 0:
        raise ValueError("--min_epochs_before_stopping must be >= 0.")
    if args.event_tolerance_minutes < 0 or args.event_bucket_size < 0:
        raise ValueError("Event tolerance and bucket size must be >= 0.")
    if args.min_on_duration < 1 or args.min_off_duration < 1:
        raise ValueError("Minimum ON/OFF durations must be >= 1.")

    state_threshold_values = (
        args.state_thresholds_kw
        if args.state_thresholds_kw is not None
        else [DEFAULT_STATE_THRESHOLDS[name] for name in APPLIANCES]
    )
    state_thresholds = {
        name: float(value)
        for name, value in zip(
            APPLIANCES,
            state_threshold_values,
            strict=True,
        )
    }

    source_home_loaders: dict[str, dict[str, Any]] = {}
    if len(source_csv_paths) > 1:
        loaders, bundle, source_home_loaders = build_multi_home_dataloaders(
            csv_paths=source_csv_paths,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            input_window=args.input_window,
            horizon=args.horizon,
            stride=args.stride,
            target_mode="future",
            input_cols=INPUT_COLS,
            appliance_cols=APPLIANCES,
            state_thresholds=state_thresholds,
            drop_unavailable_windows=True,
            balanced_train=args.home_balanced_sampling,
            seed=args.seed,
        )
    else:
        loaders, bundle = build_single_home_dataloaders(
            csv_path=csv_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            input_window=args.input_window,
            horizon=args.horizon,
            stride=args.stride,
            target_mode="future",
            input_cols=INPUT_COLS,
            appliance_cols=APPLIANCES,
            state_thresholds=state_thresholds,
            drop_unavailable_windows=True,
        )
    held_out_loaders = None
    if held_out_csv_path is not None:
        held_out_loaders, _ = build_single_home_dataloaders(
            csv_path=held_out_csv_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            input_window=args.input_window,
            horizon=args.horizon,
            stride=args.stride,
            target_mode="future",
            input_cols=INPUT_COLS,
            appliance_cols=APPLIANCES,
            state_thresholds=state_thresholds,
            drop_unavailable_windows=True,
            reference_bundle=bundle,
            shuffle_train=False,
        )
    appliance_scales = torch.tensor(bundle.appliance_scales, dtype=torch.float32)
    state_threshold_vector = torch.tensor(
        [bundle.state_thresholds[name] for name in APPLIANCES],
        dtype=torch.float32,
    )
    rated_power = rated_power_tensor_for_appliances(
        APPLIANCES,
        rated_power_kw=args.rated_power_kw,
    )

    config = {
        "args": vars(args),
        "source_csv_paths": [str(path) for path in source_csv_paths],
        "held_out_csv_path": (
            None if held_out_csv_path is None else str(held_out_csv_path)
        ),
        "input_cols": INPUT_COLS,
        "appliances": APPLIANCES,
        "state_thresholds": bundle.state_thresholds,
        "appliance_scales": bundle.appliance_scales.tolist(),
        "appliance_activity_stats": bundle.appliance_activity_stats,
        "rated_power_kw": rated_power.tolist(),
        "protocol": {
            "csv_sha256": sha256_file(csv_path),
            "source_csv_sha256": {
                str(path): sha256_file(path) for path in source_csv_paths
            },
            "reference_config": (
                str(Path(args.protocol_reference_config).expanduser().resolve())
                if args.protocol_reference_config
                else None
            ),
            "reference_config_sha256": (
                sha256_file(
                    Path(args.protocol_reference_config).expanduser().resolve()
                )
                if args.protocol_reference_config
                else None
            ),
            "split": "predeclared chronological train/val/test labels in the CSV",
            "input_window": args.input_window,
            "horizon": args.horizon,
            "stride": args.stride,
            "model_selection": "minimum validation macro-average MAE",
            "test_policy": "one final test evaluation per validation-selected method",
            "event_tolerance_minutes": args.event_tolerance_minutes,
            "event_bucket_size": args.event_bucket_size,
            "postprocess_state": args.postprocess_state,
            "min_on_duration": args.min_on_duration,
            "min_off_duration": args.min_off_duration,
            "two_stage_definition": (
                "frozen Seq2Seq-NILM histories are used to train, validate, and test "
                "the deployment TCN"
            ),
            "multi_home_preprocessing": (
                "source-train-only joint scalers and equal home quotas per batch"
                if len(source_csv_paths) > 1
                else "single source home"
            ),
            "held_out_policy": (
                "target train/val never used; target test evaluated once"
                if held_out_csv_path is not None
                else None
            ),
            "oracle_definition": (
                "ground-truth appliance histories train and evaluate a separately "
                "selected TCN with identical architecture, hyperparameters, and "
                "initialization"
            ),
            "persistence_definition": "ground-truth last appliance value; diagnostic only",
        },
    }
    config_path = run_dir / "config.json"
    if args.resume_completed_stages:
        validate_resume_config(config_path, args)
    else:
        save_json(config_path, config)

    nilm = HistoricalNILMSeq2Seq(
        input_dim=len(INPUT_COLS),
        num_appliances=len(APPLIANCES),
        hidden_dim=args.hidden_dim,
        num_layers=args.nilm_num_layers,
        dropout=args.dropout,
        rated_power=rated_power,
    )
    print("=" * 80)
    print("Stage 1: historical NILM")
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print("=" * 80)
    nilm_checkpoint_path = checkpoint_dir / "nilm_best.pt"
    nilm_history_path = result_dir / "nilm_history.json"
    if (
        args.resume_completed_stages
        and nilm_checkpoint_path.is_file()
        and nilm_history_path.is_file()
    ):
        nilm_history = load_json(nilm_history_path)
        if not isinstance(nilm_history, list):
            raise ValueError(f"Expected a history list: {nilm_history_path}")
        print("Resume: skipped completed historical NILM stage.")
    else:
        if args.resume_completed_stages and nilm_checkpoint_path.is_file():
            print(
                "Resume: NILM completion marker is missing; restarting the "
                "NILM stage from scratch."
            )
        nilm_history = train_nilm(
            nilm,
            loaders,
            device=device,
            checkpoint_path=nilm_checkpoint_path,
            appliance_scales=appliance_scales,
            args=args,
        )
        save_json(nilm_history_path, nilm_history)
    best_nilm = torch.load(nilm_checkpoint_path, map_location=device)
    nilm.load_state_dict(best_nilm["model_state_dict"])

    forecaster_template = ApplianceHistoryTCNForecaster(
        num_appliances=len(APPLIANCES),
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        rated_power=rated_power,
    )
    loss_config = PISALossConfig(
        lambda_power=args.lambda_power,
        lambda_bridge=0.0,
        lambda_state=args.lambda_state,
        lambda_event=args.lambda_event,
        lambda_start=args.lambda_start,
        lambda_stop=args.lambda_stop,
        lambda_agg=0.0,
        lambda_ghost=args.lambda_ghost,
        lambda_peak=args.lambda_peak,
        lambda_orth=0.0,
        lambda_recon_power=0.0,
        lambda_recon_state=0.0,
        active_power_weight=args.active_power_weight,
        active_bridge_weight=0.0,
    )

    # When requested, the Oracle reference starts from bit-identical TCN
    # weights. It is optional because it is not one of the three models in the
    # primary LOHO comparison and roughly doubles downstream training time.
    two_stage_forecaster = copy.deepcopy(forecaster_template)
    oracle_forecaster = (
        None
        if args.skip_oracle_forecaster
        else copy.deepcopy(forecaster_template)
    )
    two_stage_pipeline = TwoStageNILMForecastPipeline(
        nilm_model=nilm,
        forecaster=two_stage_forecaster,
        freeze_nilm=True,
        state_thresholds=state_threshold_vector,
    )
    two_stage_trainer = make_forecast_trainer(
        two_stage_pipeline,
        checkpoint_dir / "two_stage_forecaster",
        loss_config,
        appliance_scales,
        bundle,
        args,
        max_epochs=args.forecast_epochs,
    )

    print("=" * 80)
    print("Stage 2A: genuine Seq2Seq-NILM -> TCN forecaster")
    print("Training input: frozen NILM-predicted appliance histories")
    print("=" * 80)
    two_stage_history = fit_or_resume_forecast_stage(
        label="Seq2Seq-NILM -> TCN",
        trainer=two_stage_trainer,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        result_history_path=result_dir / "two_stage_forecaster_history.json",
        resume=args.resume_completed_stages,
    )

    eval_split = "val" if args.skip_test_evaluation else "test"
    predicted_history_metrics = evaluate_forecast_model(
        two_stage_trainer.model,
        loaders[eval_split],
        device=device,
        appliance_names=APPLIANCES,
        state_thresholds=bundle.state_thresholds,
        event_tolerance_minutes=args.event_tolerance_minutes,
        event_bucket_size=args.event_bucket_size,
        postprocess_state=args.postprocess_state,
        min_on_duration=args.min_on_duration,
        min_off_duration=args.min_off_duration,
    )
    save_json(
        result_dir / f"nilm_predicted_history_{eval_split}_metrics.json",
        predicted_history_metrics,
    )

    oracle_trainer = None
    oracle_history = None
    oracle_metrics = None
    if oracle_forecaster is not None:
        print("=" * 80)
        print("Stage 2B: Oracle appliance-history TCN upper reference")
        print("Training input: ground-truth appliance histories")
        print("=" * 80)
        oracle_trainer = make_forecast_trainer(
            oracle_forecaster,
            checkpoint_dir / "oracle_forecaster",
            loss_config,
            appliance_scales,
            bundle,
            args,
            max_epochs=args.forecast_epochs,
        )
        oracle_history = fit_or_resume_forecast_stage(
            label="Oracle appliance-history TCN",
            trainer=oracle_trainer,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            result_history_path=result_dir / "oracle_forecaster_history.json",
            resume=args.resume_completed_stages,
        )
        oracle_metrics = evaluate_forecast_model(
            oracle_trainer.model,
            loaders[eval_split],
            device=device,
            appliance_names=APPLIANCES,
            state_thresholds=bundle.state_thresholds,
            event_tolerance_minutes=args.event_tolerance_minutes,
            event_bucket_size=args.event_bucket_size,
            postprocess_state=args.postprocess_state,
            min_on_duration=args.min_on_duration,
            min_off_duration=args.min_off_duration,
        )
        save_json(
            result_dir / f"oracle_appliance_history_{eval_split}_metrics.json",
            oracle_metrics,
        )
    else:
        print("Skipping Oracle history forecaster (not part of primary LOHO comparison).")

    print("=" * 80)
    print("Direct baseline: aggregate history -> appliance TCN")
    print("=" * 80)
    direct_model = AggregateToApplianceTCN(
        input_dim=len(INPUT_COLS),
        num_appliances=len(APPLIANCES),
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        rated_power=rated_power,
    )
    direct_trainer = make_forecast_trainer(
        direct_model,
        checkpoint_dir / "aggregate_tcn",
        loss_config,
        appliance_scales,
        bundle,
        args,
        max_epochs=args.direct_epochs,
    )
    direct_history = fit_or_resume_forecast_stage(
        label="Aggregate -> appliance TCN",
        trainer=direct_trainer,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        result_history_path=result_dir / "aggregate_tcn_history.json",
        resume=args.resume_completed_stages,
    )
    direct_metrics = evaluate_forecast_model(
        direct_trainer.model,
        loaders[eval_split],
        device=device,
        appliance_names=APPLIANCES,
        state_thresholds=bundle.state_thresholds,
        event_tolerance_minutes=args.event_tolerance_minutes,
        event_bucket_size=args.event_bucket_size,
        postprocess_state=args.postprocess_state,
        min_on_duration=args.min_on_duration,
        min_off_duration=args.min_off_duration,
    )
    save_json(result_dir / f"aggregate_tcn_{eval_split}_metrics.json", direct_metrics)

    zero_metrics = evaluate_trivial_baseline(
        "zero", loaders[eval_split], device, APPLIANCES,
        bundle.state_thresholds, args.horizon,
        args.event_tolerance_minutes, args.event_bucket_size,
        args.postprocess_state, args.min_on_duration, args.min_off_duration,
    )
    persistence_metrics = evaluate_trivial_baseline(
        "persistence", loaders[eval_split], device, APPLIANCES,
        bundle.state_thresholds, args.horizon,
        args.event_tolerance_minutes, args.event_bucket_size,
        args.postprocess_state, args.min_on_duration, args.min_off_duration,
    )
    save_json(result_dir / f"zero_{eval_split}_metrics.json", zero_metrics)
    save_json(result_dir / f"persistence_{eval_split}_metrics.json", persistence_metrics)
    nilm_history_metrics = evaluate_history(
        model=nilm,
        model_kind="nilm",
        loader=loaders[eval_split],
        device=device,
        appliance_names=APPLIANCES,
        thresholds_kw=state_threshold_vector,
        max_batches=None,
    )
    nilm_history_metrics.update(
        {
            "seed": args.seed,
            "checkpoint": str(checkpoint_dir / "nilm_best.pt"),
            "evaluation_split": eval_split,
        }
    )
    save_json(
        result_dir / f"seq2seq_nilm_history_{eval_split}_metrics.json",
        nilm_history_metrics,
    )

    unified_methods = {
        "Zero": zero_metrics,
        "Persistence (GT history; diagnostic)": persistence_metrics,
        "Aggregate-to-appliance TCN": direct_metrics,
        "Seq2Seq-NILM -> TCN": predicted_history_metrics,
    }
    if oracle_metrics is not None:
        unified_methods["Oracle history -> matched TCN"] = oracle_metrics
    unified = {
        "seed": args.seed,
        "evaluation_split": eval_split,
        "methods": unified_methods,
    }
    save_json(result_dir / f"unified_baselines_{eval_split}.json", unified)

    if (
        held_out_loaders is not None
        and "test" in held_out_loaders
        and not args.skip_test_evaluation
    ):
        held_out_loader = held_out_loaders["test"]
        held_out_aggregate = evaluate_forecast_model(
            direct_trainer.model,
            held_out_loader,
            device=device,
            appliance_names=APPLIANCES,
            state_thresholds=bundle.state_thresholds,
            event_tolerance_minutes=args.event_tolerance_minutes,
            event_bucket_size=args.event_bucket_size,
            postprocess_state=args.postprocess_state,
            min_on_duration=args.min_on_duration,
            min_off_duration=args.min_off_duration,
        )
        held_out_seq2seq = evaluate_forecast_model(
            two_stage_trainer.model,
            held_out_loader,
            device=device,
            appliance_names=APPLIANCES,
            state_thresholds=bundle.state_thresholds,
            event_tolerance_minutes=args.event_tolerance_minutes,
            event_bucket_size=args.event_bucket_size,
            postprocess_state=args.postprocess_state,
            min_on_duration=args.min_on_duration,
            min_off_duration=args.min_off_duration,
        )
        held_out_zero = evaluate_trivial_baseline(
            "zero",
            held_out_loader,
            device,
            APPLIANCES,
            bundle.state_thresholds,
            args.horizon,
            args.event_tolerance_minutes,
            args.event_bucket_size,
            args.postprocess_state,
            args.min_on_duration,
            args.min_off_duration,
        )
        zero_mae = float(
            held_out_zero["flat"]["regression/macro_avg/MAE"]
        )
        held_out_methods = {
            "Aggregate-to-appliance TCN": held_out_aggregate,
            "Seq2Seq-NILM -> TCN": held_out_seq2seq,
            "Zero": held_out_zero,
        }
        generalization: dict[str, dict[str, float]] = {}
        for name, payload in held_out_methods.items():
            method_mae = float(
                payload["flat"]["regression/macro_avg/MAE"]
            )
            normalized = method_mae / zero_mae if zero_mae > 0.0 else float("nan")
            generalization[name] = {
                "macro_MAE": method_mae,
                "normalized_MAE_vs_zero": normalized,
                "skill_vs_zero": 1.0 - normalized,
            }
        held_out_summary = {
            "target_csv": str(held_out_csv_path),
            "selection_data": "source-home validation splits only",
            "target_train_used": False,
            "target_validation_used": False,
            "target_test_evaluations": 1,
            "source_scalers_only": True,
            "methods": held_out_methods,
            "generalization": generalization,
        }
        save_json(
            result_dir / "held_out_test_metrics.json",
            held_out_summary,
        )
        print("Held-out-home baseline metrics saved to:")
        print(result_dir / "held_out_test_metrics.json")

    model_info = {
        "nilm_parameters": sum(p.numel() for p in nilm.parameters()),
        "two_stage_forecaster_parameters": sum(
            p.numel() for p in two_stage_forecaster.parameters()
        ),
        "oracle_forecaster_parameters": (
            None
            if oracle_trainer is None
            else sum(p.numel() for p in oracle_trainer.model.parameters())
        ),
        "aggregate_tcn_parameters": sum(p.numel() for p in direct_trainer.model.parameters()),
        "loss": asdict(loss_config),
        "best_nilm_epoch": best_nilm.get("epoch"),
        "best_two_stage_forecaster_epoch": two_stage_trainer.best_epoch,
        "best_oracle_forecaster_epoch": (
            None if oracle_trainer is None else oracle_trainer.best_epoch
        ),
        "best_aggregate_tcn_epoch": direct_trainer.best_epoch,
        "forecaster_architecture": "one independent causal TCN per appliance",
        "two_stage_training_and_inference_history_source": (
            "frozen Seq2Seq-NILM predictions"
        ),
        "oracle_training_and_inference_history_source": (
            None if oracle_trainer is None else "ground-truth histories"
        ),
        "downstream_initialization": (
            "bit-identical before separate training"
            if oracle_trainer is not None
            else "not applicable; Oracle skipped"
        ),
    }
    save_json(result_dir / "model_info.json", model_info)
    print("Saved two-stage revision metrics to:")
    print(result_dir)


if __name__ == "__main__":
    main()
