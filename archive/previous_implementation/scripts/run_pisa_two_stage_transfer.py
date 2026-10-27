from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(ROOT))
sys.path.append(str(SRC))

from data import build_single_home_datasets  # noqa: E402
from training import PISATrainer, TrainerConfig, move_batch_to_device, set_seed  # noqa: E402
from scripts.run_pisa_cross_home_transfer import (  # noqa: E402
    APPLIANCES,
    DEFAULT_STATE_THRESHOLDS,
    apply_rated_power_caps,
    build_loaders,
    build_loss,
    build_source_model,
    calibrate_support_power_caps,
    evaluate,
    load_checkpoint_strict,
    macro_mae,
    model_rated_power_caps,
    resolve_source_config,
    save_json,
    subset_loader,
)


PROTOCOL_ID = "pisa_transfer_v3_history_then_forecast"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage PISA cross-home adaptation: first select historical "
            "reconstruction by validation history MAE, then freeze it and "
            "select residual-TCN forecasting by validation future MAE."
        )
    )
    parser.add_argument("--source_csv", required=True)
    parser.add_argument("--target_csv", required=True)
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument("--source_config", default=None)
    parser.add_argument(
        "--output_dir", default=str(ROOT / "outputs" / "transfer_runs_two_stage")
    )
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--input_window", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--few_shot_fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10]
    )
    parser.add_argument("--history_epochs", type=int, default=120)
    parser.add_argument("--forecast_epochs", type=int, default=200)
    parser.add_argument("--history_lr", type=float, default=3e-5)
    parser.add_argument("--forecast_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--history_patience", type=int, default=25)
    parser.add_argument("--forecast_patience", type=int, default=25)
    parser.add_argument("--history_min_epochs", type=int, default=40)
    parser.add_argument("--forecast_min_epochs", type=int, default=40)
    parser.add_argument(
        "--forecast_scope",
        choices=("heads", "enhanced_heads", "full_tcn"),
        default="heads",
        help=(
            "Use enhanced_heads for the budget-matched future-time/direct-blend "
            "adapter; heads preserves the legacy route; full_tcn is an ablation."
        ),
    )
    parser.add_argument(
        "--enhanced_forecast_head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add horizon embeddings, future-time covariates, a small frozen-"
            "Transformer context adapter, learned base/direct fusion and target "
            "power calibration."
        ),
    )
    parser.add_argument("--target_adapter_dim", type=int, default=32)
    parser.add_argument("--future_base_blend_init", type=float, default=4.0)
    parser.add_argument(
        "--forecast_parameter_budget",
        type=int,
        default=28_700,
        help="Maximum trainable forecast parameters for comparison with Seq2Seq.",
    )
    parser.add_argument("--lambda_recon_power", type=float, default=1.0)
    parser.add_argument("--lambda_recon_state", type=float, default=0.10)
    parser.add_argument("--lambda_direct_power", type=float, default=0.0)
    parser.add_argument("--active_recon_weight", type=float, default=1.0)
    parser.add_argument("--lambda_state", type=float, default=0.10)
    parser.add_argument("--lambda_agg", type=float, default=0.02)
    parser.add_argument("--lambda_ghost", type=float, default=0.03)
    parser.add_argument("--lambda_peak", type=float, default=0.05)
    parser.add_argument("--active_power_weight", type=float, default=1.0)
    parser.add_argument("--active_bridge_weight", type=float, default=0.0)
    parser.add_argument("--state_alpha", type=float, default=0.50)
    parser.add_argument("--adaptation_aux_warmup_epochs", type=int, default=5)
    parser.add_argument("--adaptation_aux_ramp_epochs", type=int, default=10)
    parser.add_argument("--adaptation_warmup_lambda_agg", type=float, default=0.0)
    parser.add_argument("--adaptation_warmup_lambda_state", type=float, default=0.0)
    parser.add_argument("--adaptation_warmup_lambda_ghost", type=float, default=0.0)
    parser.add_argument("--adaptation_warmup_lambda_peak", type=float, default=0.0)
    parser.add_argument(
        "--target_cap_mode",
        choices=("source", "support_quantile"),
        default="support_quantile",
    )
    parser.add_argument("--support_cap_quantile", type=float, default=0.995)
    parser.add_argument("--support_cap_margin", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each interrupted stage from its last.pt checkpoint.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_transfer_model(
    source_config: dict[str, Any],
    args: argparse.Namespace,
) -> nn.Module:
    """Build the source architecture plus optional target-only forecast modules."""
    config = copy.deepcopy(source_config)
    if args.enhanced_forecast_head:
        model_cfg = config.setdefault("model", {})
        model_cfg.update(
            {
                "residual_tcn_use_future_context": True,
                "residual_tcn_target_adapter_dim": int(args.target_adapter_dim),
                "future_residual_fusion": "learned_blend",
                "future_base_blend_init": float(args.future_base_blend_init),
                "target_output_calibration": True,
            }
        )
    return build_source_model(config)


@torch.no_grad()
def reset_target_output_calibration(model: nn.Module) -> None:
    """Start every target home's affine calibration from identity."""
    residual_tcn = getattr(model, "refined_history_residual_tcn", None)
    if residual_tcn is None:
        return
    log_scale = getattr(residual_tcn, "target_power_log_scale", None)
    bias = getattr(residual_tcn, "target_power_bias", None)
    if log_scale is not None:
        log_scale.zero_()
    if bias is not None:
        bias.zero_()


def configure_history_stage(model: nn.Module) -> int:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.past_recon_head.power_head, model.past_recon_head.state_head):
        for parameter in module.parameters():
            parameter.requires_grad = True
    if getattr(model, "residual_tcn_history_source", "base") == "refined":
        refiner = getattr(model, "history_recon_refiner", None)
        if refiner is None:
            raise ValueError("refined history source requires history_recon_refiner.")
        for module in (refiner.power_delta_head, refiner.state_delta_head):
            for parameter in module.parameters():
                parameter.requires_grad = True
        refiner.power_gate_logit.requires_grad = True
        refiner.state_gate_logit.requires_grad = True
    model._history_forecast_heads_only_training = True
    model._future_residual_only_training = False
    model._enhanced_forecast_heads_only_training = False
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count == 0:
        raise RuntimeError("No historical-reconstruction parameters selected.")
    return count


def configure_forecast_stage(
    model: nn.Module,
    scope: str,
    parameter_budget: int,
) -> int:
    for parameter in model.parameters():
        parameter.requires_grad = False
    residual_tcn = getattr(model, "refined_history_residual_tcn", None)
    if residual_tcn is None:
        raise ValueError("Source checkpoint has no residual history TCN.")
    model._future_residual_only_training = False
    model._history_forecast_heads_only_training = False
    model._enhanced_forecast_heads_only_training = False
    if scope == "full_tcn":
        for parameter in residual_tcn.parameters():
            parameter.requires_grad = True
        model._future_residual_only_training = True
        model._history_forecast_heads_only_training = False
    elif scope == "heads":
        for head in residual_tcn.residual_heads:
            for parameter in head.parameters():
                parameter.requires_grad = True
        residual_tcn.power_gate_logit.requires_grad = True
        residual_tcn.state_gate_logit.requires_grad = True
        model._history_forecast_heads_only_training = True
        model._future_residual_only_training = False
    else:
        modules = residual_tcn.enhanced_adaptation_modules()
        if not modules:
            raise ValueError(
                "forecast_scope=enhanced_heads requires --enhanced_forecast_head."
            )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
        for name in (
            "base_power_blend_logit",
            "target_power_log_scale",
            "target_power_bias",
        ):
            parameter = getattr(residual_tcn, name, None)
            if parameter is None:
                raise ValueError(f"Enhanced forecast parameter {name!r} is missing.")
            parameter.requires_grad = True
        model._enhanced_forecast_heads_only_training = True
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count == 0:
        raise RuntimeError("No residual-TCN parameters selected.")
    if parameter_budget > 0 and count > parameter_budget:
        raise ValueError(
            f"Forecast scope selects {count:,} trainable parameters, exceeding "
            f"the comparison budget of {parameter_budget:,}."
        )
    print(
        f"Forecast trainable parameters: {count:,} / "
        f"budget {parameter_budget:,}"
    )
    return count


@torch.no_grad()
def history_macro_mae(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.to(device).eval()
    absolute = torch.zeros(len(APPLIANCES), dtype=torch.float64)
    count = torch.zeros(len(APPLIANCES), dtype=torch.float64)
    for batch in loader:
        device_batch = move_batch_to_device(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp and device.type == "cuda",
        ):
            out = model(device_batch)
        prediction = out["future_tcn_input_power"].detach().cpu().float()
        target = batch["y_hist_power"].float()
        mask = batch.get("hist_mask")
        valid = torch.ones_like(target) if mask is None else (mask > 0.5).float()
        absolute += ((prediction - target).abs() * valid).sum((0, 2)).double()
        count += valid.sum((0, 2)).double()
    values = absolute / count.clamp_min(1.0)
    return float(values.mean())


class HistorySelectionTrainer(PISATrainer):
    """Inject exact kW history macro MAE into every validation epoch."""

    def validate(
        self,
        val_loader: Any,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, float]:
        stats = super().validate(val_loader, *args, **kwargs)
        stats["history/macro_avg/MAE"] = history_macro_mae(
            self.model,
            val_loader,
            self.device,
            self.use_amp,
        )
        return stats


def build_trainer(
    model: nn.Module,
    loss: nn.Module,
    train_loader: Any,
    val_loader: Any,
    checkpoint_dir: Path,
    bundle: Any,
    args: argparse.Namespace,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    min_epochs: int,
    monitor: str,
    history_selection: bool = False,
    resume: bool = False,
) -> PISATrainer:
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    trainer_class = HistorySelectionTrainer if history_selection else PISATrainer
    trainer = trainer_class(
        model=model,
        loss_fn=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        config=TrainerConfig(
            max_epochs=epochs,
            device=str(device),
            seed=args.seed,
            use_amp=not args.no_amp,
            checkpoint_dir=str(checkpoint_dir),
            monitor=monitor,
            monitor_mode="min",
            early_stopping_patience=patience,
            min_epochs_before_stopping=min_epochs,
            eval_metrics_every=1,
            scheduler_monitor=monitor,
            event_tolerance_minutes=2,
            event_bucket_size=5,
        ),
        appliance_names=APPLIANCES,
        state_thresholds=bundle.state_thresholds,
    )
    start_epoch = 1
    last_checkpoint = checkpoint_dir / "last.pt"
    if resume and last_checkpoint.is_file():
        payload = trainer.load_checkpoint(
            last_checkpoint,
            load_optimizer=True,
            load_scheduler=True,
        )
        start_epoch = int(payload.get("epoch", 0)) + 1
        print(
            f"Resuming {checkpoint_dir.name} from epoch "
            f"{start_epoch} (last checkpoint: {last_checkpoint})."
        )
    if start_epoch <= epochs:
        trainer.fit(train_loader, val_loader, start_epoch=start_epoch)
    else:
        print(
            f"Stage checkpoint already reached epoch {start_epoch - 1}; "
            "skipping additional optimization."
        )
    return trainer


def save_selected(path: Path, model: nn.Module, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "selection": metadata},
        path,
    )


def main() -> None:
    args = parse_args()
    if args.lambda_recon_power <= 0.0:
        raise ValueError("--lambda_recon_power must be positive.")
    if args.lambda_direct_power < 0.0:
        raise ValueError("--lambda_direct_power must be >= 0.")
    if args.history_epochs < 1 or args.forecast_epochs < 1:
        raise ValueError("Both adaptation stages require at least one epoch.")
    if args.target_adapter_dim < 0:
        raise ValueError("--target_adapter_dim must be >= 0.")
    if args.forecast_parameter_budget < 0:
        raise ValueError("--forecast_parameter_budget must be >= 0.")
    if args.forecast_scope == "enhanced_heads" and not args.enhanced_forecast_head:
        raise ValueError(
            "--forecast_scope enhanced_heads requires --enhanced_forecast_head."
        )
    if args.enhanced_forecast_head and args.forecast_scope != "enhanced_heads":
        raise ValueError(
            "--enhanced_forecast_head must use --forecast_scope enhanced_heads "
            "so its new parameters are actually trained."
        )
    set_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_amp = device.type == "cuda" and not args.no_amp
    source_config = resolve_source_config(args)
    data_cfg = dict(source_config["data"])
    input_cols = list(source_config["input_cols"])
    appliances = list(source_config["appliance_cols"])
    if appliances != APPLIANCES:
        raise ValueError(f"Expected appliance order {APPLIANCES}, got {appliances}.")
    input_window = int(
        data_cfg.get("input_window", 120)
        if args.input_window is None
        else args.input_window
    )
    horizon = int(
        data_cfg.get("horizon", 30) if args.horizon is None else args.horizon
    )
    stride = int(data_cfg.get("stride", 1) if args.stride is None else args.stride)
    source_run = Path(args.source_checkpoint).expanduser().resolve().parent.parent
    data_info_path = source_run / "results" / "data_info.json"
    data_info = read_json(data_info_path) if data_info_path.is_file() else {}
    thresholds_raw = data_info.get("state_thresholds", DEFAULT_STATE_THRESHOLDS)
    thresholds = {name: float(thresholds_raw[name]) for name in appliances}

    _, source_bundle = build_single_home_datasets(
        csv_path=args.source_csv,
        input_cols=input_cols,
        appliance_cols=appliances,
        input_window=input_window,
        horizon=horizon,
        stride=stride,
        target_mode="future",
        state_thresholds=thresholds,
        drop_unavailable_windows=True,
    )
    target_datasets, target_bundle = build_single_home_datasets(
        csv_path=args.target_csv,
        input_cols=input_cols,
        appliance_cols=appliances,
        input_window=input_window,
        horizon=horizon,
        stride=stride,
        target_mode="future",
        reference_bundle=source_bundle,
        state_thresholds=thresholds,
        drop_unavailable_windows=True,
    )
    loaders = build_loaders(target_datasets, args.eval_batch_size, args.num_workers)
    if "val" not in loaders:
        raise ValueError("Target validation split is required.")
    output_dir = Path(args.output_dir).expanduser().resolve() / args.run_name
    result_dir = output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    source_model = build_transfer_model(source_config, args)
    load_checkpoint_strict(source_model, args.source_checkpoint)
    source_caps = model_rated_power_caps(source_model)
    cap_profiles = {
        f"{fraction:.3f}": calibrate_support_power_caps(
            target_datasets["train"], fraction, appliances,
            source_bundle.state_thresholds, source_caps, args.target_cap_mode,
            args.support_cap_quantile, args.support_cap_margin,
        )
        for fraction in args.few_shot_fractions
    }
    for key, profile in cap_profiles.items():
        save_json(result_dir / f"support_caps_{key}.json", profile)

    summary: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "source_csv": str(Path(args.source_csv).resolve()),
        "target_csv": str(Path(args.target_csv).resolve()),
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "source_config": str(
            Path(args.source_config).resolve()
            if args.source_config
            else source_run / "config.json"
        ),
        "selection": {
            "stage_1": "target validation history macro MAE",
            "stage_2": "target validation future macro MAE",
            "test_used": False,
        },
        "forecast_architecture": {
            "enhanced_forecast_head": bool(args.enhanced_forecast_head),
            "future_time_and_horizon_context": bool(
                args.enhanced_forecast_head
            ),
            "fusion": (
                "learned_base_direct_blend"
                if args.enhanced_forecast_head
                else source_config["model"].get(
                    "future_residual_fusion", "gated_residual"
                )
            ),
            "target_adapter_dim": int(args.target_adapter_dim),
            "per_appliance_power_calibration": bool(
                args.enhanced_forecast_head
            ),
            "parameter_budget": int(args.forecast_parameter_budget),
            "history_power_state_gating": False,
        },
        "few_shot": {},
    }

    for fraction in args.few_shot_fractions:
        key = f"{fraction:.3f}"
        profile = cap_profiles[key]
        train_loader = subset_loader(
            target_datasets["train"], fraction, args.batch_size, args.num_workers
        )
        set_seed(args.seed)
        model = build_transfer_model(source_config, args)
        load_checkpoint_strict(model, args.source_checkpoint)
        apply_rated_power_caps(model, profile["calibrated_caps_kw"])
        reset_target_output_calibration(model)

        source_history_mae = history_macro_mae(
            copy.deepcopy(model), loaders["val"], device, use_amp
        )
        history_params = configure_history_stage(model)
        history_dir = output_dir / "checkpoints" / f"fewshot_{key}" / "history_stage"
        history_trainer = build_trainer(
            model, build_loss(target_bundle, args, stage="history"),
            train_loader, loaders["val"], history_dir, target_bundle, args,
            device, args.history_epochs, args.history_lr,
            args.history_patience, args.history_min_epochs,
            "val/history/macro_avg/MAE",
            history_selection=True,
            resume=args.resume,
        )
        history_trainer.load_checkpoint(
            history_dir / "best.pt", load_optimizer=False, load_scheduler=False
        )
        adapted_history_mae = history_macro_mae(
            history_trainer.model, loaders["val"], device, use_amp
        )
        if adapted_history_mae < source_history_mae:
            history_selected = "history_stage_checkpoint"
            history_epoch = history_trainer.best_epoch
            model = history_trainer.model
            selected_history_mae = adapted_history_mae
        else:
            history_selected = "support_calibrated_source"
            history_epoch = 0
            model = build_transfer_model(source_config, args)
            load_checkpoint_strict(model, args.source_checkpoint)
            apply_rated_power_caps(model, profile["calibrated_caps_kw"])
            reset_target_output_calibration(model)
            selected_history_mae = source_history_mae

        history_selected_path = (
            output_dir / "checkpoints" / f"fewshot_{key}" / "selected_history.pt"
        )
        save_selected(
            history_selected_path,
            model,
            {
                "selected": history_selected,
                "epoch": history_epoch,
                "validation_history_macro_mae": selected_history_mae,
            },
        )

        pre_forecast_metrics = evaluate(
            copy.deepcopy(model), loaders["val"], device,
            source_bundle.state_thresholds, appliances, use_amp,
        )
        forecast_params = configure_forecast_stage(
            model,
            args.forecast_scope,
            args.forecast_parameter_budget,
        )
        forecast_dir = output_dir / "checkpoints" / f"fewshot_{key}" / "forecast_stage"
        forecast_trainer = build_trainer(
            model, build_loss(target_bundle, args, stage="forecast"),
            train_loader, loaders["val"], forecast_dir, target_bundle, args,
            device, args.forecast_epochs, args.forecast_lr,
            args.forecast_patience, args.forecast_min_epochs,
            "val/regression/macro_avg/MAE",
            resume=args.resume,
        )
        if forecast_trainer.best_metric < macro_mae(pre_forecast_metrics):
            forecast_trainer.load_checkpoint(
                forecast_dir / "best.pt", load_optimizer=False, load_scheduler=False
            )
            final_model = forecast_trainer.model
            forecast_selected = "forecast_stage_checkpoint"
            forecast_epoch = forecast_trainer.best_epoch
        else:
            final_model = model
            load_checkpoint_strict(final_model, history_selected_path)
            forecast_selected = "selected_history_checkpoint"
            forecast_epoch = 0
        final_metrics = evaluate(
            final_model, loaders["val"], device,
            source_bundle.state_thresholds, appliances, use_amp,
        )
        final_path = output_dir / "checkpoints" / f"fewshot_{key}" / "selected.pt"
        save_selected(
            final_path,
            final_model,
            {
                "history_selected": history_selected,
                "forecast_selected": forecast_selected,
                "validation_future_macro_mae": macro_mae(final_metrics),
            },
        )
        save_json(result_dir / f"fewshot_{key}_validation_metrics.json", final_metrics)
        summary["few_shot"][key] = {
            "support_fraction": float(fraction),
            "support_windows": profile["support_windows"],
            "power_cap_calibration": profile,
            "history_stage": {
                "trainable_parameters": history_params,
                "source_validation_history_macro_mae": source_history_mae,
                "adapted_validation_history_macro_mae": adapted_history_mae,
                "selected_validation_history_macro_mae": selected_history_mae,
                "selected_model": history_selected,
                "selected_epoch": history_epoch,
            },
            "forecast_stage": {
                "scope": args.forecast_scope,
                "trainable_parameters": forecast_params,
                "pre_adaptation_validation_mae": macro_mae(pre_forecast_metrics),
                "best_validation_mae": forecast_trainer.best_metric,
                "selected_model": forecast_selected,
                "selected_epoch": forecast_epoch,
            },
            "selected_model": "two_stage_selected_checkpoint",
            "selected_epoch": forecast_epoch,
            "selected_checkpoint": str(final_path),
            "selected_validation_mae": macro_mae(final_metrics),
            "validation": final_metrics["flat"],
            "test": None,
        }
        # Persist after every fraction so an SSH interruption cannot discard
        # summaries for already completed fractions.
        save_json(
            result_dir / f"fewshot_{key}_stage_summary.json",
            summary["few_shot"][key],
        )
        save_json(result_dir / "transfer_summary.json", summary)
        print(
            f"Fraction {key}: history MAE {source_history_mae:.6f} -> "
            f"{selected_history_mae:.6f}; future MAE "
            f"{macro_mae(pre_forecast_metrics):.6f} -> {macro_mae(final_metrics):.6f}"
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_json(output_dir / "config.json", vars(args))
    save_json(result_dir / "transfer_summary.json", summary)
    print(f"Two-stage validation-only results saved to: {result_dir}")


if __name__ == "__main__":
    main()
