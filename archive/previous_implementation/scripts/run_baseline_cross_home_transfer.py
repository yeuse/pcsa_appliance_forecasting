from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(ROOT))
sys.path.append(str(SRC))

from data import build_single_home_datasets  # noqa: E402
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
    set_seed,
)
from scripts.run_pisa_cross_home_transfer import (  # noqa: E402
    build_loaders,
    calibrate_support_power_caps,
    evaluate,
    macro_mae,
    subset_loader,
)


PROTOCOL_ID = "transfer_baselines_v2_joint_history"
INPUT_COLS = ["grid", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
APPLIANCES = ["air1", "refrigerator1", "dishwasher1", "microwave1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen Aggregate-TCN and Seq2Seq-NILM->TCN cross-home transfer."
    )
    parser.add_argument("--source_csv", required=True)
    parser.add_argument("--target_csv", required=True)
    parser.add_argument("--source_run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--input_window", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--few_shot_fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10])
    parser.add_argument("--few_shot_epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_epochs_before_stopping", type=int, default=25)
    parser.add_argument("--lambda_state", type=float, default=0.10)
    parser.add_argument("--lambda_agg", type=float, default=0.02)
    parser.add_argument("--lambda_ghost", type=float, default=0.03)
    parser.add_argument("--lambda_peak", type=float, default=0.05)
    parser.add_argument("--lambda_recon_power", type=float, default=0.0)
    parser.add_argument("--lambda_recon_state", type=float, default=0.0)
    parser.add_argument("--active_recon_weight", type=float, default=0.0)
    parser.add_argument("--active_power_weight", type=float, default=1.0)
    parser.add_argument("--state_alpha", type=float, default=0.50)
    parser.add_argument("--adaptation_aux_warmup_epochs", type=int, default=5)
    parser.add_argument("--adaptation_aux_ramp_epochs", type=int, default=10)
    parser.add_argument("--target_cap_mode", choices=["source", "support_quantile"], default="support_quantile")
    parser.add_argument("--support_cap_quantile", type=float, default=0.995)
    parser.add_argument("--support_cap_margin", type=float, default=0.05)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--skip_test_evaluation", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)


def load_checkpoint_strict(model: nn.Module, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    print(f"Loaded source checkpoint: {path}")


def build_models(config: dict[str, Any]) -> tuple[nn.Module, nn.Module]:
    cfg = config["args"]
    rated_power = rated_power_tensor_for_appliances(APPLIANCES)
    direct = AggregateToApplianceTCN(
        input_dim=len(INPUT_COLS),
        num_appliances=len(APPLIANCES),
        horizon=int(cfg["horizon"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        kernel_size=int(cfg["kernel_size"]),
        dropout=float(cfg.get("dropout", 0.1)),
        rated_power=rated_power,
    )
    nilm = HistoricalNILMSeq2Seq(
        input_dim=len(INPUT_COLS),
        num_appliances=len(APPLIANCES),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["nilm_num_layers"]),
        dropout=float(cfg.get("dropout", 0.1)),
        rated_power=rated_power,
    )
    forecaster = ApplianceHistoryTCNForecaster(
        num_appliances=len(APPLIANCES),
        horizon=int(cfg["horizon"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        kernel_size=int(cfg["kernel_size"]),
        dropout=float(cfg.get("dropout", 0.1)),
        rated_power=rated_power,
    )
    state_thresholds = torch.tensor(
        [float(config["state_thresholds"][name]) for name in APPLIANCES],
        dtype=torch.float32,
    )
    two_stage = TwoStageNILMForecastPipeline(
        nilm_model=nilm,
        forecaster=forecaster,
        freeze_nilm=True,
        state_thresholds=state_thresholds,
    )
    return direct, two_stage


def source_caps(model: nn.Module) -> torch.Tensor:
    if isinstance(model, TwoStageNILMForecastPipeline):
        return model.forecaster.rated_power.detach().reshape(-1).cpu().clone()
    return model.rated_power.detach().reshape(-1).cpu().clone()


def apply_caps(model: nn.Module, caps: list[float] | torch.Tensor) -> None:
    values = torch.as_tensor(caps, dtype=torch.float32).reshape(-1)
    targets = (
        [model.nilm_model.rated_power, model.forecaster.rated_power]
        if isinstance(model, TwoStageNILMForecastPipeline)
        else [model.rated_power]
    )
    with torch.no_grad():
        for target in targets:
            if target.numel() != values.numel():
                raise ValueError("Rated-power cap size mismatch.")
            target.copy_(values.to(target).view_as(target))


def configure_direct_heads(model: AggregateToApplianceTCN) -> int:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for head in model.heads:
        for parameter in head.parameters():
            parameter.requires_grad = True
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count == 0:
        raise RuntimeError("No Aggregate-TCN head parameters selected.")
    return count


def configure_two_stage_heads(model: TwoStageNILMForecastPipeline) -> int:
    for parameter in model.parameters():
        parameter.requires_grad = False
    # Target supervision may calibrate the NILM output mapping, but the NILM
    # temporal encoder and all appliance TCN encoders remain source-frozen.
    model.freeze_nilm = False
    model.head_only_adaptation = True
    for module in (model.nilm_model.power_head, model.nilm_model.state_head):
        for parameter in module.parameters():
            parameter.requires_grad = True
    for head in model.forecaster.heads:
        for parameter in head.parameters():
            parameter.requires_grad = True
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count == 0:
        raise RuntimeError("No two-stage output-head parameters selected.")
    return count


def build_loss(bundle: Any, args: argparse.Namespace) -> PISALoss:
    return PISALoss(
        appliance_scales=torch.tensor(bundle.appliance_scales, dtype=torch.float32),
        config=PISALossConfig(
            lambda_power=1.0,
            lambda_bridge=0.0,
            lambda_state=args.lambda_state,
            lambda_event=0.0,
            lambda_start=0.0,
            lambda_stop=0.0,
            lambda_bucket_start=0.0,
            lambda_bucket_stop=0.0,
            lambda_event_offset=0.0,
            lambda_window_start=0.0,
            lambda_conditional_bucket=0.0,
            lambda_conditional_offset=0.0,
            lambda_conditional_power=0.0,
            lambda_pulse_duration=0.0,
            lambda_pulse_amplitude=0.0,
            lambda_agg=args.lambda_agg,
            lambda_ghost=args.lambda_ghost,
            lambda_peak=args.lambda_peak,
            lambda_amp_on=0.0,
            lambda_orth=0.0,
            lambda_recon_power=args.lambda_recon_power,
            lambda_recon_state=args.lambda_recon_state,
            active_power_weight=args.active_power_weight,
            active_bridge_weight=0.0,
            active_recon_weight=args.active_recon_weight,
            base_aux_warmup_epochs=args.adaptation_aux_warmup_epochs,
            base_aux_ramp_epochs=args.adaptation_aux_ramp_epochs,
            base_warmup_lambda_agg=0.0,
            base_warmup_lambda_state=0.0,
            base_warmup_lambda_ghost=0.0,
            base_warmup_lambda_peak=0.0,
        ),
        state_alpha=[args.state_alpha] * len(APPLIANCES),
        event_alpha=[0.50] * len(APPLIANCES),
        stop_alpha=[0.50] * len(APPLIANCES),
    )


def train_candidate(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    checkpoint_dir: Path,
    target_bundle: Any,
    args: argparse.Namespace,
) -> tuple[PISATrainer, list[dict[str, Any]]]:
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    trainer = PISATrainer(
        model=model,
        loss_fn=build_loss(target_bundle, args),
        optimizer=optimizer,
        scheduler=scheduler,
        config=TrainerConfig(
            max_epochs=args.few_shot_epochs,
            device=args.device,
            seed=args.seed,
            use_amp=not args.no_amp,
            checkpoint_dir=str(checkpoint_dir),
            monitor="val/regression/macro_avg/MAE",
            monitor_mode="min",
            early_stopping_patience=args.patience,
            min_epochs_before_stopping=args.min_epochs_before_stopping,
            eval_metrics_every=1,
            scheduler_monitor="val/regression/macro_avg/MAE",
            event_tolerance_minutes=2,
            event_bucket_size=5,
        ),
        appliance_names=APPLIANCES,
        state_thresholds=target_bundle.state_thresholds,
    )
    history = trainer.fit(train_loader, val_loader)
    return trainer, history


def run_method(
    name: str,
    source_model: nn.Module,
    model_factory: Callable[[], nn.Module],
    configure: Callable[[Any], int],
    source_checkpoint: Path,
    target_datasets: dict[str, Any],
    target_loaders: dict[str, Any],
    source_bundle: Any,
    target_bundle: Any,
    output_dir: Path,
    result_dir: Path,
    cap_profiles: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
) -> dict[str, Any]:
    zero_val = evaluate(
        source_model, target_loaders["val"], device,
        source_bundle.state_thresholds, APPLIANCES, use_amp,
    )
    zero_test = None
    if not args.skip_test_evaluation:
        zero_test = evaluate(
            source_model, target_loaders["test"], device,
            source_bundle.state_thresholds, APPLIANCES, use_amp,
        )
    method_slug = "aggregate_tcn" if name.startswith("Aggregate") else "two_stage"
    save_json(result_dir / f"{method_slug}_zero_shot_validation.json", zero_val)
    if zero_test is not None:
        save_json(result_dir / f"{method_slug}_zero_shot_test.json", zero_test)
    summary: dict[str, Any] = {
        "source_checkpoint": str(source_checkpoint),
        "zero_shot": {
            "validation": zero_val["flat"],
            "test": None if zero_test is None else zero_test["flat"],
        },
        "few_shot": {},
    }

    for fraction in args.few_shot_fractions:
        key = f"{fraction:.3f}"
        profile = cap_profiles[key]
        set_seed(args.seed)
        model = model_factory()
        load_checkpoint_strict(model, source_checkpoint)
        apply_caps(model, profile["calibrated_caps_kw"])
        trainable = configure(model)

        if args.target_cap_mode == "source":
            source_candidate_val = zero_val
            source_label = "source_zero_shot"
        else:
            source_candidate = model_factory()
            load_checkpoint_strict(source_candidate, source_checkpoint)
            apply_caps(source_candidate, profile["calibrated_caps_kw"])
            source_candidate_val = evaluate(
                source_candidate, target_loaders["val"], device,
                source_bundle.state_thresholds, APPLIANCES, use_amp,
            )
            source_label = "support_calibrated_source"

        train_loader = subset_loader(
            target_datasets["train"], fraction, args.batch_size, args.num_workers
        )
        checkpoint_dir = output_dir / "checkpoints" / method_slug / f"fewshot_{key}"
        trainer, history = train_candidate(
            model, train_loader, target_loaders["val"], checkpoint_dir,
            target_bundle, args,
        )
        save_json(result_dir / f"{method_slug}_fewshot_{key}_history.json", history)

        selected_label = source_label
        selected_epoch = 0
        selected_val = source_candidate_val
        if float(trainer.best_metric) < macro_mae(source_candidate_val):
            trainer.load_checkpoint(
                checkpoint_dir / "best.pt", load_optimizer=False, load_scheduler=False
            )
            selected_val = evaluate(
                trainer.model, target_loaders["val"], device,
                source_bundle.state_thresholds, APPLIANCES, use_amp,
            )
            selected_label = "few_shot_checkpoint"
            selected_epoch = int(trainer.best_epoch)

        selected_test = None
        if not args.skip_test_evaluation:
            if selected_label == "source_zero_shot":
                selected_test = zero_test
            elif selected_label == "support_calibrated_source":
                candidate = model_factory()
                load_checkpoint_strict(candidate, source_checkpoint)
                apply_caps(candidate, profile["calibrated_caps_kw"])
                selected_test = evaluate(
                    candidate, target_loaders["test"], device,
                    source_bundle.state_thresholds, APPLIANCES, use_amp,
                )
            else:
                selected_test = evaluate(
                    trainer.model, target_loaders["test"], device,
                    source_bundle.state_thresholds, APPLIANCES, use_amp,
                )

        save_json(result_dir / f"{method_slug}_fewshot_{key}_validation.json", selected_val)
        if selected_test is not None:
            save_json(result_dir / f"{method_slug}_fewshot_{key}_test.json", selected_test)
        summary["few_shot"][key] = {
            "support_fraction": float(fraction),
            "support_windows": int(profile["support_windows"]),
            "trainable_parameters": int(trainable),
            "power_cap_calibration": profile,
            "selected_model": selected_label,
            "selected_epoch": selected_epoch,
            "selected_validation_mae": macro_mae(selected_val),
            "source_candidate_validation_mae": macro_mae(source_candidate_val),
            "fine_tune_best_validation_mae": float(trainer.best_metric),
            "validation": selected_val["flat"],
            "test": None if selected_test is None else selected_test["flat"],
        }
        print(
            f"{name} few-shot {key}: val MAE={macro_mae(selected_val):.6f}; "
            f"selected={selected_label}@{selected_epoch}"
        )
    return summary


def main() -> None:
    args = parse_args()
    if args.lambda_recon_power < 0.0 or args.lambda_recon_state < 0.0:
        raise ValueError("Historical reconstruction weights must be >= 0.")
    if args.active_recon_weight < 0.0:
        raise ValueError("--active_recon_weight must be >= 0.")
    set_seed(args.seed)
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    source_run = Path(args.source_run_dir).expanduser().resolve()
    config = read_json(source_run / "config.json")
    if list(config["input_cols"]) != INPUT_COLS or list(config["appliances"]) != APPLIANCES:
        raise ValueError("Source baseline input/appliance order does not match frozen protocol.")
    source_args = config["args"]
    for field, expected in (
        ("input_window", args.input_window),
        ("horizon", args.horizon),
        ("stride", args.stride),
    ):
        if int(source_args[field]) != int(expected):
            raise ValueError(f"Source {field}={source_args[field]} does not match {expected}.")

    direct_ckpt = source_run / "checkpoints" / "aggregate_tcn" / "best.pt"
    two_stage_ckpt = source_run / "checkpoints" / "two_stage_forecaster" / "best.pt"
    if not direct_ckpt.exists() or not two_stage_ckpt.exists():
        raise FileNotFoundError(
            f"Missing source baseline checkpoint(s): {direct_ckpt}, {two_stage_ckpt}"
        )

    direct_template, two_stage_template = build_models(config)
    load_checkpoint_strict(direct_template, direct_ckpt)
    load_checkpoint_strict(two_stage_template, two_stage_ckpt)
    source_state_thresholds = {
        name: float(config["state_thresholds"][name]) for name in APPLIANCES
    }
    source_datasets, source_bundle = build_single_home_datasets(
        csv_path=args.source_csv,
        input_cols=INPUT_COLS,
        appliance_cols=APPLIANCES,
        input_window=args.input_window,
        horizon=args.horizon,
        stride=args.stride,
        target_mode="future",
        state_thresholds=source_state_thresholds,
        drop_unavailable_windows=True,
    )
    target_datasets, target_bundle = build_single_home_datasets(
        csv_path=args.target_csv,
        input_cols=INPUT_COLS,
        appliance_cols=APPLIANCES,
        input_window=args.input_window,
        horizon=args.horizon,
        stride=args.stride,
        target_mode="future",
        reference_bundle=source_bundle,
        state_thresholds=source_state_thresholds,
        drop_unavailable_windows=True,
    )
    if "val" not in target_datasets:
        raise ValueError("Target validation split is required.")
    if not args.skip_test_evaluation and "test" not in target_datasets:
        raise ValueError("Target test split is required unless --skip_test_evaluation.")
    target_loaders = build_loaders(
        target_datasets, args.eval_batch_size, args.num_workers
    )

    output_dir = Path(args.output_dir).expanduser().resolve() / args.run_name
    result_dir = output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    caps = source_caps(direct_template)
    if not torch.allclose(caps, source_caps(two_stage_template)):
        raise ValueError("Source direct and two-stage rated-power caps differ.")
    cap_profiles = {
        f"{fraction:.3f}": calibrate_support_power_caps(
            target_datasets["train"], fraction, APPLIANCES,
            source_bundle.state_thresholds, caps, args.target_cap_mode,
            args.support_cap_quantile, args.support_cap_margin,
        )
        for fraction in args.few_shot_fractions
    }
    for key, profile in cap_profiles.items():
        save_json(result_dir / f"support_caps_{key}.json", profile)

    direct_factory = lambda: copy.deepcopy(direct_template).cpu()
    two_stage_factory = lambda: copy.deepcopy(two_stage_template).cpu()
    summary = {
        "protocol_id": PROTOCOL_ID,
        "protocol": {
            "source_home": Path(args.source_csv).expanduser().resolve().stem,
            "target_home": Path(args.target_csv).expanduser().resolve().stem,
            "support": "chronologically first fraction of target train windows",
            "fractions": [0.0] + [float(x) for x in args.few_shot_fractions],
            "checkpoint_selection": "target validation macro MAE",
            "test_used_for_selection": False,
            "test_policy": "one evaluation per validation-selected method/configuration",
            "aggregate_adaptation": "forecast heads only; TCN encoder frozen",
            "two_stage_adaptation": (
                "NILM power/state output heads and forecast heads only; "
                "NILM and TCN encoders frozen"
            ),
            "history_reconstruction_supervision": {
                "lambda_power": args.lambda_recon_power,
                "lambda_state": args.lambda_recon_state,
                "active_power_weight": args.active_recon_weight,
            },
        },
        "seed": args.seed,
        "source_csv": str(Path(args.source_csv).resolve()),
        "target_csv": str(Path(args.target_csv).resolve()),
        "methods": {},
    }
    summary["methods"]["Aggregate-to-appliance TCN"] = run_method(
        "Aggregate-to-appliance TCN", direct_factory(), direct_factory,
        configure_direct_heads, direct_ckpt, target_datasets, target_loaders,
        source_bundle, target_bundle, output_dir, result_dir, cap_profiles,
        args, device, use_amp,
    )
    summary["methods"]["Seq2Seq-NILM -> TCN"] = run_method(
        "Seq2Seq-NILM -> TCN", two_stage_factory(), two_stage_factory,
        configure_two_stage_heads, two_stage_ckpt, target_datasets, target_loaders,
        source_bundle, target_bundle, output_dir, result_dir, cap_profiles,
        args, device, use_amp,
    )
    save_json(output_dir / "config.json", vars(args))
    save_json(result_dir / "baseline_transfer_summary.json", summary)
    print(f"Baseline transfer results saved to: {result_dir}")


if __name__ == "__main__":
    main()
