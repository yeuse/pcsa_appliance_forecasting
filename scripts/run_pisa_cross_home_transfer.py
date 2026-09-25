from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(ROOT))
sys.path.append(str(SRC))

from data import build_single_home_datasets  # noqa: E402
from evaluation.metrics import MetricsAccumulator, flatten_metrics  # noqa: E402
from models import (  # noqa: E402
    AggregateOnlySeq2SeqBaseline,
    build_pisa_model_from_config,
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


INPUT_COLS = ["grid", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
APPLIANCES = ["air1", "refrigerator1", "dishwasher1", "microwave1"]
PROTOCOL_ID = "pisa_transfer_v2_joint_history"
DEFAULT_STATE_THRESHOLDS = {
    "air1": 0.50,
    "refrigerator1": 0.05,
    "dishwasher1": 0.05,
    "microwave1": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot and few-shot PISA transfer between homes."
    )
    parser.add_argument(
        "--source_csv",
        type=str,
        default=str(
            ROOT
            / "data"
            / "austin_2018_sep_4homes"
            / "home_7951_2018_sep_1min.csv"
        ),
    )
    parser.add_argument(
        "--target_csv",
        type=str,
        default=str(
            ROOT
            / "data"
            / "austin_2018_sep_4homes"
            / "home_3039_2018_sep_1min.csv"
        ),
    )
    parser.add_argument(
        "--source_checkpoint",
        type=str,
        required=True,
        help="PISA checkpoint trained on the source home.",
    )
    parser.add_argument(
        "--source_config",
        type=str,
        default=None,
        help=(
            "Optional source run config.json. If omitted, it is resolved from "
            "the source checkpoint's run directory."
        ),
    )
    parser.add_argument(
        "--aggregate_only_checkpoint",
        type=str,
        default=None,
        help=(
            "Optional separately trained AggregateOnlySeq2Seq source checkpoint. "
            "When supplied, it is evaluated and few-shot adapted on exactly "
            "the same target support/validation protocol as PISA."
        ),
    )
    parser.add_argument(
        "--aggregate_only_config",
        type=str,
        default=None,
        help=(
            "Optional config.json for --aggregate_only_checkpoint. If omitted, "
            "it is resolved from that checkpoint's run directory."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT / "outputs" / "transfer_runs"),
    )
    parser.add_argument("--run_name", type=str, default="pisa_home7951_to_home3039")
    parser.add_argument(
        "--input_window",
        type=int,
        default=None,
        help="Optional override. Defaults to the source run configuration.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Optional override. Defaults to the source run configuration.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Optional override. Defaults to the source run configuration.",
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help=(
            "Validation/test batch size. Defaults to --batch_size; set this "
            "larger than the training batch when GPU memory permits."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--few_shot_fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10])
    parser.add_argument(
        "--zero_shot_only",
        action="store_true",
        help=(
            "Evaluate the frozen source checkpoint on the target validation/test "
            "splits without constructing or training a target support set."
        ),
    )
    parser.add_argument("--few_shot_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--min_epochs_before_stopping",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--fine_tune_mode",
        choices=[
            "power_heads",
            "power_adapter_heads",
            "history_forecast_heads",
            "decoder",
            "future_residual_tcn",
            "full",
        ],
        default="power_adapter_heads",
        help=(
            "power_adapter_heads adapts a small per-appliance power adapter plus "
            "power/state output heads; power_heads excludes the adapter; "
            "history_forecast_heads adapts the base historical-reconstruction "
            "output heads together with "
            "the residual-TCN output heads while keeping both temporal encoders "
            "frozen; decoder adapts the whole decoder; future_residual_tcn freezes "
            "the source PISA "
            "and history refiner and adapts only the refined-history residual TCN; "
            "full is an explicit full-model ablation."
        ),
    )
    parser.add_argument(
        "--power_adapter_dim",
        type=int,
        default=8,
        help=(
            "Per-appliance residual bottleneck used only for target-home PISA "
            "adaptation. It is zero-initialized so source zero-shot output is "
            "unchanged before optimization."
        ),
    )
    parser.add_argument(
        "--adapt_bridge_fusion",
        action="store_true",
        help=(
            "Also tune PISA's five per-appliance bridge-fusion logits during "
            "parameter-efficient target adaptation."
        ),
    )
    parser.add_argument(
        "--active_power_weight",
        type=float,
        default=2.0,
        help="ON-state power weight from the frozen Home 7951 base protocol.",
    )
    parser.add_argument(
        "--active_bridge_weight",
        type=float,
        default=1.0,
        help="Bridge ON-state weight from the frozen Home 7951 base protocol.",
    )
    parser.add_argument(
        "--state_alpha",
        type=float,
        default=0.50,
        help="Positive-class focal alpha frozen from the source-home protocol.",
    )
    parser.add_argument(
        "--lambda_agg",
        type=float,
        default=0.05,
        help="Aggregate upper-bound protection during target adaptation.",
    )
    parser.add_argument(
        "--lambda_ghost",
        type=float,
        default=0.10,
        help="OFF-state ghost-power protection during target adaptation.",
    )
    parser.add_argument(
        "--lambda_peak",
        type=float,
        default=0.20,
        help="ON-state peak-shape protection during target adaptation.",
    )
    parser.add_argument(
        "--lambda_state",
        type=float,
        default=0.40,
        help="State supervision weight during target adaptation.",
    )
    parser.add_argument(
        "--lambda_recon_power",
        type=float,
        default=0.0,
        help=(
            "Historical appliance-power reconstruction weight. Set this to the "
            "same positive value for PISA and the two-stage baseline when using "
            "a history-aware transfer comparison."
        ),
    )
    parser.add_argument(
        "--lambda_recon_state",
        type=float,
        default=0.0,
        help="Historical appliance-state reconstruction weight.",
    )
    parser.add_argument(
        "--active_recon_weight",
        type=float,
        default=0.0,
        help="Additional ON-state weight for historical power reconstruction.",
    )
    parser.add_argument(
        "--adaptation_aux_warmup_epochs",
        type=int,
        default=0,
        help=(
            "Number of target-adaptation epochs using the initial aggregate, "
            "state, ghost, and peak protection weights."
        ),
    )
    parser.add_argument(
        "--adaptation_aux_ramp_epochs",
        type=int,
        default=0,
        help=(
            "Number of epochs linearly ramping target-adaptation protection "
            "weights from their initial to final values."
        ),
    )
    parser.add_argument("--adaptation_warmup_lambda_agg", type=float, default=0.0)
    parser.add_argument("--adaptation_warmup_lambda_state", type=float, default=0.0)
    parser.add_argument("--adaptation_warmup_lambda_ghost", type=float, default=0.0)
    parser.add_argument("--adaptation_warmup_lambda_peak", type=float, default=0.0)
    parser.add_argument(
        "--target_cap_mode",
        choices=["source", "support_quantile"],
        default="source",
        help=(
            "Use source checkpoint caps unchanged, or derive support-only "
            "target operational caps and freeze them for adaptation/evaluation."
        ),
    )
    parser.add_argument(
        "--support_cap_quantile",
        type=float,
        default=0.995,
        help="Positive-power support quantile used by --target_cap_mode support_quantile.",
    )
    parser.add_argument(
        "--support_cap_margin",
        type=float,
        default=0.05,
        help="Non-negative multiplicative margin applied above the support quantile.",
    )
    parser.add_argument(
        "--skip_test_evaluation",
        action="store_true",
        help=(
            "Fit and select on target train/validation only; do not compute "
            "target-test metrics."
        ),
    )
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def build_loaders(datasets, batch_size: int, num_workers: int) -> dict[str, DataLoader]:
    loaders = {}
    for split, dataset in datasets.items():
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders


def subset_loader(dataset, fraction: float, batch_size: int, num_workers: int) -> DataLoader:
    fraction = min(max(float(fraction), 0.0), 1.0)
    count = max(1, int(round(len(dataset) * fraction)))
    indices = list(range(count))
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=len(indices) >= batch_size,
    )


def support_window_count(dataset, fraction: float) -> int:
    """Return the chronological support-window count used by ``subset_loader``."""
    fraction = min(max(float(fraction), 0.0), 1.0)
    return max(1, int(round(len(dataset) * fraction)))


def calibrate_support_power_caps(
    dataset: Any,
    fraction: float,
    appliance_names: list[str],
    state_thresholds: dict[str, float],
    source_caps: torch.Tensor,
    mode: str,
    quantile: float,
    margin: float,
) -> dict[str, Any]:
    """Derive non-leaking target operational caps from chronological support.

    The target labels inspected here are exactly the labels belonging to the
    few-shot training windows.  Validation/test rows are never read.  The
    source cap remains a lower bound, so this calibration can only relax an
    underestimated source-home cap; it cannot silently make a physical cap
    tighter on the target home.
    """
    source_caps = torch.as_tensor(source_caps, dtype=torch.float32).flatten().cpu()
    if source_caps.numel() != len(appliance_names):
        raise ValueError("source caps must contain one value per appliance.")
    if mode not in {"source", "support_quantile"}:
        raise ValueError(f"Unsupported target cap mode: {mode!r}.")

    support_count = support_window_count(dataset, fraction)
    result: dict[str, Any] = {
        "mode": mode,
        "support_fraction": float(fraction),
        "support_windows": support_count,
        "source_caps_kw": source_caps.tolist(),
        "support_quantile": float(quantile),
        "support_margin": float(margin),
        "appliances": {},
    }
    if mode == "source":
        result["calibrated_caps_kw"] = source_caps.tolist()
        return result

    required = ("indices", "input_window", "horizon", "y_power", "target_available")
    missing = [name for name in required if not hasattr(dataset, name)]
    if missing:
        raise TypeError(
            "support_quantile cap calibration requires the Pecan Street "
            f"dataset attributes {required}; missing {missing}."
        )

    starts = list(dataset.indices[:support_count])
    support_rows = sorted(
        {
            row
            for start in starts
            for row in range(
                int(start) + int(dataset.input_window),
                int(start) + int(dataset.input_window) + int(dataset.horizon),
            )
        }
    )
    power = torch.as_tensor(dataset.y_power[support_rows], dtype=torch.float32)
    available = torch.as_tensor(
        dataset.target_available[support_rows], dtype=torch.bool
    )
    calibrated = source_caps.clone()
    result["support_target_rows"] = len(support_rows)
    for index, appliance in enumerate(appliance_names):
        threshold = float(state_thresholds[appliance])
        positive = power[:, index][
            available[:, index] & (power[:, index] > threshold)
        ]
        item: dict[str, Any] = {
            "source_cap_kw": float(source_caps[index]),
            "state_threshold_kw": threshold,
            "positive_support_points": int(positive.numel()),
        }
        if positive.numel() > 0:
            observed = float(torch.quantile(positive, quantile))
            proposed = observed * (1.0 + margin)
            calibrated[index] = max(float(source_caps[index]), proposed)
            item["support_positive_quantile_kw"] = observed
            item["proposed_cap_kw"] = proposed
        else:
            item["support_positive_quantile_kw"] = None
            item["proposed_cap_kw"] = float(source_caps[index])
        item["calibrated_cap_kw"] = float(calibrated[index])
        result["appliances"][appliance] = item

    result["calibrated_caps_kw"] = calibrated.tolist()
    return result


def model_rated_power_caps(model: nn.Module) -> torch.Tensor:
    """Read a model's [A] hard power caps without assuming model internals."""
    if hasattr(model, "constraint") and hasattr(model.constraint, "rated_power"):
        caps = model.constraint.rated_power
    elif hasattr(model, "rated_power"):
        caps = model.rated_power
    else:
        raise TypeError(f"Model {type(model).__name__} does not expose rated_power.")
    return caps.detach().reshape(-1).to(device="cpu", dtype=torch.float32).clone()


def apply_rated_power_caps(model: nn.Module, caps: torch.Tensor | list[float]) -> None:
    """Set frozen operational caps identically for PISA and AggregateOnly."""
    caps = torch.as_tensor(caps, dtype=torch.float32).flatten()
    if hasattr(model, "constraint") and hasattr(model.constraint, "rated_power"):
        target = model.constraint.rated_power
    elif hasattr(model, "rated_power"):
        target = model.rated_power
    else:
        raise TypeError(f"Model {type(model).__name__} does not expose rated_power.")
    if target.numel() != caps.numel():
        raise ValueError(
            f"Cap size mismatch: model has {target.numel()}, received {caps.numel()}."
        )
    with torch.no_grad():
        target.copy_(caps.to(device=target.device, dtype=target.dtype).view_as(target))


def resolve_checkpoint_config(
    checkpoint: str | Path,
    config_path: str | Path | None,
    expected_model_type: str,
    label: str,
) -> dict[str, Any]:
    """Load the exact architecture/protocol that produced a source checkpoint."""
    config_path = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else Path(checkpoint).expanduser().resolve().parent.parent
        / "config.json"
    )
    config = load_json(config_path)
    required = ("data", "model", "input_cols", "appliance_cols")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(
            f"{label} config is incomplete at {config_path}; missing {missing}."
        )
    model_type = config["model"].get("model_type", "pisa")
    if model_type != expected_model_type:
        raise ValueError(
            f"{label} checkpoint requires model_type={expected_model_type!r}, "
            f"got {model_type!r}."
        )
    print(f"{label} config: {config_path}")
    return config


def resolve_source_config(args: argparse.Namespace) -> dict[str, Any]:
    return resolve_checkpoint_config(
        checkpoint=args.source_checkpoint,
        config_path=args.source_config,
        expected_model_type="pisa",
        label="PISA source",
    )


def build_source_model(
    config: dict[str, Any],
    power_adapter_dim: int | None = None,
) -> nn.Module:
    data_cfg = dict(config["data"])
    data_cfg["appliance_cols"] = list(config["appliance_cols"])
    model_cfg = dict(config["model"])
    if power_adapter_dim is not None:
        model_cfg["power_adapter_dim"] = int(power_adapter_dim)
    return build_pisa_model_from_config(
        {"data": data_cfg, "model": model_cfg}
    )


def build_aggregate_only_model(config: dict[str, Any]) -> nn.Module:
    """Rebuild the actual aggregate-only baseline from its saved source config."""
    appliance_names = list(config["appliance_cols"])
    data_cfg = dict(config["data"])
    model_cfg = dict(config["model"])
    rated_power = rated_power_tensor_for_appliances(
        appliance_names=appliance_names,
        rated_power_kw=data_cfg.get("rated_power_kw", data_cfg.get("rated_power")),
    )
    return AggregateOnlySeq2SeqBaseline(
        input_dim=int(model_cfg["input_dim"]),
        num_appliances=int(model_cfg["num_appliances"]),
        horizon=int(model_cfg["horizon"]),
        hidden_dim=int(model_cfg.get("d_model", 128)),
        num_layers=max(1, int(model_cfg.get("encoder_layers", 2))),
        rated_power=rated_power,
    )


def load_checkpoint_strict(model: nn.Module, path: str | Path) -> None:
    """Strictly restore a source model; fail instead of silently dropping weights."""
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = dict(checkpoint.get("model_state_dict", checkpoint))
    current_state = model.state_dict()
    if (
        "hierarchical_power_blend" in current_state
        and "hierarchical_power_blend" not in state
    ):
        state["hierarchical_power_blend"] = current_state[
            "hierarchical_power_blend"
        ]
    initialized_power_adapter = 0
    initialized_enhanced_forecast = 0
    for key, value in current_state.items():
        if key.startswith("power_adapters.") and key not in state:
            state[key] = value
            initialized_power_adapter += 1
        if (
            key.startswith(
                "refined_history_residual_tcn.horizon_embedding."
            )
            or key.startswith(
                "refined_history_residual_tcn.future_time_proj."
            )
            or key.startswith(
                "refined_history_residual_tcn.encoder_context_adapter."
            )
            or key.startswith(
                "refined_history_residual_tcn.future_step_norm."
            )
            or key.startswith(
                "refined_history_residual_tcn.direct_step_heads."
            )
            or key in {
                "refined_history_residual_tcn.base_power_blend_logit",
                "refined_history_residual_tcn.base_state_blend_logit",
                "refined_history_residual_tcn.target_power_log_scale",
                "refined_history_residual_tcn.target_power_bias",
            }
        ) and key not in state:
            # These parameters deliberately may be introduced only for target
            # transfer.  Initialize them from the freshly constructed model
            # while restoring every source-trained tensor strictly.
            state[key] = value
            initialized_enhanced_forecast += 1
    model.load_state_dict(state, strict=True)
    if initialized_power_adapter:
        print(
            "Source checkpoint predates power_adapters; initialized "
            f"{initialized_power_adapter} zero-output adapter tensors."
        )
    if initialized_enhanced_forecast:
        print(
            "Initialized enhanced target forecast head from its neutral "
            f"defaults ({initialized_enhanced_forecast} tensors); all source "
            "checkpoint tensors were restored strictly."
        )
    print(f"Loaded source checkpoint strictly: {checkpoint_path}")


def configure_transfer_parameters(
    model: nn.Module,
    mode: str,
    adapt_bridge_fusion: bool = False,
) -> int:
    """Choose a reproducible target-home adaptation scope."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    # These flags control train/eval behavior of frozen modules. Reset them so
    # one adaptation mode cannot leak into another model instance.
    model._history_forecast_heads_only_training = False
    model._future_residual_only_training = False

    if mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif mode == "decoder":
        for parameter in model.decoder.parameters():
            parameter.requires_grad = True
        if getattr(model, "tail_disagg_head", None) is not None:
            for parameter in model.tail_disagg_head.parameters():
                parameter.requires_grad = True
            model.tail_global_fusion_logit.requires_grad = True
        model.bridge_fusion_logit.requires_grad = True
    elif mode in {"power_heads", "power_adapter_heads"}:
        # Parameter-efficient cross-home calibration: preserve source encoder,
        # bridge, decoder representation, state route and physics constraint;
        # adjust only the per-appliance route producing future power.
        power_head_names = [
            "base_power_head",
            "residual_power_head",
            "persistence_gate_head",
            "conditional_amplitude_head",
        ]
        if mode == "power_adapter_heads":
            power_head_names.append("state_delta_head")
        for appliance_decoder in model.decoder.appliance_decoders:
            for name in power_head_names:
                for parameter in getattr(appliance_decoder, name).parameters():
                    parameter.requires_grad = True
        if mode == "power_adapter_heads":
            if not getattr(model, "has_power_adapter", False):
                raise ValueError(
                    "power_adapter_heads requires --power_adapter_dim > 0."
                )
            for parameter in model.power_adapters.parameters():
                parameter.requires_grad = True
        if adapt_bridge_fusion:
            # The direct bridge path is per-appliance but initially almost off.
            # Tuning these five logits lets the target choose its source/decoder
            # mixture without unfreezing the shared encoder or bridge network.
            model.bridge_fusion_logit.requires_grad = True
    elif mode == "history_forecast_heads":
        # Symmetric counterpart of Seq2Seq-NILM -> TCN head adaptation:
        # recalibrate the mapping from frozen aggregate-history features to
        # appliance histories, then recalibrate only the output mapping of the
        # frozen appliance-history TCNs.  The large temporal encoders remain
        # source-frozen and deterministic.
        residual_tcn = getattr(model, "refined_history_residual_tcn", None)
        if residual_tcn is None:
            raise ValueError(
                "history_forecast_heads requires a source checkpoint/config "
                "containing the trained residual TCN."
            )
        past_recon_head = getattr(model, "past_recon_head", None)
        if past_recon_head is None:
            raise ValueError(
                "history_forecast_heads requires the base historical "
                "reconstruction head."
            )
        for module in (past_recon_head.power_head, past_recon_head.state_head):
            for parameter in module.parameters():
                parameter.requires_grad = True
        for head in residual_tcn.residual_heads:
            for parameter in head.parameters():
                parameter.requires_grad = True
        residual_tcn.power_gate_logit.requires_grad = True
        residual_tcn.state_gate_logit.requires_grad = True
        model._history_forecast_heads_only_training = True
    elif mode == "future_residual_tcn":
        residual_tcn = getattr(model, "refined_history_residual_tcn", None)
        if residual_tcn is None:
            raise ValueError(
                "future_residual_tcn transfer requires a source checkpoint/config "
                "containing the trained refined-history residual TCN."
            )
        if getattr(model, "history_recon_refiner", None) is None:
            raise ValueError(
                "future_residual_tcn transfer requires a source checkpoint/config "
                "containing the trained history refiner."
            )
        for parameter in residual_tcn.parameters():
            parameter.requires_grad = True
        if getattr(model, "future_residual_fusion", "gated_residual") == "direct":
            residual_tcn.power_gate_logit.requires_grad = False
            residual_tcn.state_gate_logit.requires_grad = False
        # Keep every frozen source module in eval mode while retaining training
        # behavior (notably dropout) only inside the residual TCN.
        model._future_residual_only_training = True
    else:
        raise ValueError(f"Unsupported fine_tune_mode={mode!r}.")

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable == 0:
        raise RuntimeError("No trainable parameters selected for few-shot adaptation.")
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"Fine-tune mode      : {mode}")
    print(f"Trainable parameters: {trainable:,} / {total:,}")
    return trainable


def configure_aggregate_only_parameters(model: nn.Module) -> int:
    """Fine-tune only the actual baseline's power/state output heads."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for head_name in ("power_head", "state_head"):
        for parameter in getattr(model, head_name).parameters():
            parameter.requires_grad = True

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable == 0:
        raise RuntimeError("No trainable AggregateOnlySeq2Seq output-head parameters.")
    print(f"AggregateOnly trainable parameters: {trainable:,} / {total:,}")
    return trainable


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    state_thresholds: dict[str, float],
    appliance_names: list[str],
    use_amp: bool,
) -> dict[str, Any]:
    model.to(device).eval()
    acc = MetricsAccumulator(appliance_names=appliance_names)
    for batch in loader:
        device_batch = move_batch_to_device(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp and device.type == "cuda",
        ):
            out = model(device_batch)
        acc.update(out, device_batch)
    metrics = acc.compute(
        state_thresholds=state_thresholds,
        event_tolerance_minutes=2,
        event_bucket_size=5,
        postprocess_state=True,
        min_on_duration=3,
        min_off_duration=2,
    )
    return {"nested": metrics, "flat": flatten_metrics(metrics)}


def build_loss(
    bundle,
    args: argparse.Namespace,
    stage: str = "joint",
) -> PISALoss:
    """Build joint, history-only, or forecast-only target adaptation loss."""
    if stage not in {"joint", "history", "forecast"}:
        raise ValueError(f"Unsupported adaptation loss stage: {stage!r}.")
    history_only = stage == "history"
    forecast_only = stage == "forecast"
    return PISALoss(
        appliance_scales=torch.tensor(bundle.appliance_scales, dtype=torch.float32),
        config=PISALossConfig(
            lambda_power=0.0 if history_only else 1.0,
            lambda_bridge=0.0,
            lambda_state=0.0 if history_only else args.lambda_state,
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
            lambda_agg=0.0 if history_only else args.lambda_agg,
            lambda_ghost=0.0 if history_only else args.lambda_ghost,
            lambda_peak=0.0 if history_only else args.lambda_peak,
            lambda_amp_on=0.0,
            lambda_orth=0.0,
            lambda_recon_power=(
                0.0 if forecast_only else args.lambda_recon_power
            ),
            lambda_recon_state=(
                0.0 if forecast_only else args.lambda_recon_state
            ),
            lambda_direct_power=(
                0.0
                if history_only
                else float(getattr(args, "lambda_direct_power", 0.0))
            ),
            active_power_weight=(
                0.0 if history_only else args.active_power_weight
            ),
            active_bridge_weight=(
                0.0 if history_only else args.active_bridge_weight
            ),
            active_recon_weight=(
                0.0 if forecast_only else args.active_recon_weight
            ),
            base_aux_warmup_epochs=(
                0 if history_only else args.adaptation_aux_warmup_epochs
            ),
            base_aux_ramp_epochs=(
                0 if history_only else args.adaptation_aux_ramp_epochs
            ),
            base_warmup_lambda_agg=(
                0.0 if history_only else args.adaptation_warmup_lambda_agg
            ),
            base_warmup_lambda_state=(
                0.0 if history_only else args.adaptation_warmup_lambda_state
            ),
            base_warmup_lambda_ghost=(
                0.0 if history_only else args.adaptation_warmup_lambda_ghost
            ),
            base_warmup_lambda_peak=(
                0.0 if history_only else args.adaptation_warmup_lambda_peak
            ),
        ),
        state_alpha=[args.state_alpha] * len(bundle.appliance_cols),
        event_alpha=[0.50] * len(bundle.appliance_cols),
        stop_alpha=[0.50] * len(bundle.appliance_cols),
    )


def macro_mae(metrics: dict[str, Any]) -> float:
    """Extract the common regression metric for concise, auditable logging."""
    return float(metrics["flat"]["regression/macro_avg/MAE"])


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda" and not args.no_amp
    if args.power_adapter_dim < 0:
        raise ValueError("--power_adapter_dim must be >= 0.")
    if args.fine_tune_mode == "power_adapter_heads" and args.power_adapter_dim <= 0:
        raise ValueError(
            "--fine_tune_mode power_adapter_heads requires --power_adapter_dim > 0."
        )
    if args.adaptation_aux_warmup_epochs < 0 or args.adaptation_aux_ramp_epochs < 0:
        raise ValueError("adaptation auxiliary schedule epochs must be >= 0.")
    if args.lambda_recon_power < 0.0 or args.lambda_recon_state < 0.0:
        raise ValueError("Historical reconstruction weights must be >= 0.")
    if args.active_recon_weight < 0.0:
        raise ValueError("--active_recon_weight must be >= 0.")
    if (
        args.fine_tune_mode == "history_forecast_heads"
        and args.lambda_recon_power == 0.0
        and args.lambda_recon_state == 0.0
    ):
        raise ValueError(
            "history_forecast_heads requires a positive "
            "--lambda_recon_power and/or --lambda_recon_state."
        )
    if not 0.0 < args.support_cap_quantile <= 1.0:
        raise ValueError("--support_cap_quantile must be in (0, 1].")
    if args.support_cap_margin < 0.0:
        raise ValueError("--support_cap_margin must be >= 0.")
    transfer_power_adapter_dim = (
        args.power_adapter_dim
        if args.fine_tune_mode == "power_adapter_heads"
        else None
    )
    source_config = resolve_source_config(args)
    data_cfg = dict(source_config["data"])
    input_cols = list(source_config["input_cols"])
    appliance_names = list(source_config["appliance_cols"])
    if appliance_names != APPLIANCES:
        raise ValueError(
            "This transfer protocol currently requires the shared Home 7951/3039 "
            f"appliance order {APPLIANCES}, got {appliance_names}."
        )

    input_window = int(
        data_cfg.get("input_window", 120)
        if args.input_window is None
        else args.input_window
    )
    horizon = int(data_cfg.get("horizon", 30) if args.horizon is None else args.horizon)
    stride = int(data_cfg.get("stride", 1) if args.stride is None else args.stride)
    source_horizon = int(source_config["model"].get("horizon", horizon))
    if horizon != source_horizon:
        raise ValueError(
            "--horizon must match the source checkpoint architecture: "
            f"requested {horizon}, source {source_horizon}."
        )

    output_dir = Path(args.output_dir).expanduser().resolve() / args.run_name
    result_dir = output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    source_run_dir = Path(args.source_checkpoint).expanduser().resolve().parent.parent
    source_data_info = load_json(source_run_dir / "results" / "data_info.json")
    source_state_thresholds = source_data_info.get(
        "state_thresholds", DEFAULT_STATE_THRESHOLDS
    )
    source_state_thresholds = {
        appliance: float(source_state_thresholds[appliance])
        for appliance in appliance_names
    }

    source_datasets, source_bundle = build_single_home_datasets(
        csv_path=args.source_csv,
        input_cols=input_cols,
        appliance_cols=appliance_names,
        input_window=input_window,
        horizon=horizon,
        stride=stride,
        target_mode="future",
        state_thresholds=source_state_thresholds,
        drop_unavailable_windows=True,
    )
    target_datasets, target_bundle = build_single_home_datasets(
        csv_path=args.target_csv,
        input_cols=input_cols,
        appliance_cols=appliance_names,
        input_window=input_window,
        horizon=horizon,
        stride=stride,
        target_mode="future",
        reference_bundle=source_bundle,
        state_thresholds=source_state_thresholds,
        drop_unavailable_windows=True,
    )
    if "val" not in target_datasets:
        raise ValueError("Target CSV must contain a validation split.")
    if not args.skip_test_evaluation and "test" not in target_datasets:
        raise ValueError("Target CSV must contain a test split unless test evaluation is skipped.")

    eval_batch_size = args.batch_size if args.eval_batch_size is None else args.eval_batch_size
    if eval_batch_size < 1:
        raise ValueError("--eval_batch_size must be >= 1.")
    target_loaders = build_loaders(
        target_datasets,
        eval_batch_size,
        args.num_workers,
    )

    print("=" * 80)
    print("PISA cross-home transfer")
    print("=" * 80)
    print(f"Source CSV : {Path(args.source_csv).expanduser().resolve()}")
    print(f"Target CSV : {Path(args.target_csv).expanduser().resolve()}")
    print(f"Source ckpt: {Path(args.source_checkpoint).expanduser().resolve()}")
    print(f"Target test evaluation enabled: {not args.skip_test_evaluation}")
    print(
        f"Train/eval batch size: {args.batch_size}/{eval_batch_size}; "
        f"workers: {args.num_workers}"
    )
    print(f"Device: {device}; AMP enabled: {use_amp}")

    zero_model = build_source_model(
        source_config,
        power_adapter_dim=transfer_power_adapter_dim,
    )
    load_checkpoint_strict(zero_model, args.source_checkpoint)
    source_rated_power_caps = model_rated_power_caps(zero_model)
    aggregate_config: dict[str, Any] | None = None
    if args.aggregate_only_checkpoint is not None:
        # Validate shared architecture/caps before any PISA fine-tuning starts;
        # otherwise a mismatched baseline config could waste a long run.
        aggregate_config = resolve_checkpoint_config(
            checkpoint=args.aggregate_only_checkpoint,
            config_path=args.aggregate_only_config,
            expected_model_type="aggregate_seq2seq",
            label="AggregateOnlySeq2Seq source",
        )
        if list(aggregate_config["input_cols"]) != input_cols:
            raise ValueError("AggregateOnlySeq2Seq input columns differ from PISA source.")
        if list(aggregate_config["appliance_cols"]) != appliance_names:
            raise ValueError("AggregateOnlySeq2Seq appliance order differs from PISA source.")
        aggregate_data_cfg = dict(aggregate_config["data"])
        aggregate_horizon = int(aggregate_config["model"].get("horizon", horizon))
        if (
            int(aggregate_data_cfg.get("input_window", input_window)) != input_window
            or int(aggregate_data_cfg.get("stride", stride)) != stride
            or aggregate_horizon != horizon
        ):
            raise ValueError(
                "AggregateOnlySeq2Seq window/horizon/stride must match the PISA "
                "source and target protocol."
            )
        aggregate_preflight_model = build_aggregate_only_model(aggregate_config)
        load_checkpoint_strict(aggregate_preflight_model, args.aggregate_only_checkpoint)
        if not torch.allclose(
            model_rated_power_caps(aggregate_preflight_model),
            source_rated_power_caps,
        ):
            raise ValueError(
                "PISA and AggregateOnly source caps differ. Refusing to apply "
                "a target support calibration asymmetrically; align their "
                "source rated_power_kw configurations first."
            )
    zero_validation_metrics = evaluate(
        zero_model,
        target_loaders["val"],
        device=device,
        state_thresholds=source_bundle.state_thresholds,
        appliance_names=appliance_names,
        use_amp=use_amp,
    )
    save_json(result_dir / "zero_shot_validation_metrics.json", zero_validation_metrics)
    print(
        "Zero-shot target validation macro MAE="
        f"{macro_mae(zero_validation_metrics):.6f}"
    )

    zero_test_metrics = None
    if not args.skip_test_evaluation:
        zero_test_metrics = evaluate(
            zero_model,
            target_loaders["test"],
            device=device,
            state_thresholds=source_bundle.state_thresholds,
            appliance_names=appliance_names,
            use_amp=use_amp,
        )
        save_json(result_dir / "zero_shot_test_metrics.json", zero_test_metrics)

    summary = {
        "protocol_id": PROTOCOL_ID,
        "protocol": {
            "source_home": Path(args.source_csv).expanduser().resolve().stem,
            "target_home": Path(args.target_csv).expanduser().resolve().stem,
            "source_scalers_and_state_thresholds": "fixed from source run",
            "few_shot_support": "chronologically first fraction of target train split",
            "checkpoint_selection": "target validation macro appliance MAE",
            "zero_shot_is_a_few_shot_selection_candidate": True,
            "target_test_used_for_selection": False,
            "target_test_evaluated": not args.skip_test_evaluation,
            "pisa_fine_tune_mode": args.fine_tune_mode,
            "history_reconstruction_supervision": {
                "lambda_power": args.lambda_recon_power,
                "lambda_state": args.lambda_recon_state,
                "active_power_weight": args.active_recon_weight,
                "history_source": source_config["model"].get(
                    "residual_tcn_history_source", "refined"
                ),
            },
            "power_adapter_dim": transfer_power_adapter_dim or 0,
            "adapt_bridge_fusion": bool(args.adapt_bridge_fusion),
            "target_cap_calibration": {
                "mode": args.target_cap_mode,
                "support_quantile": args.support_cap_quantile,
                "support_margin": args.support_cap_margin,
                "cap_source": (
                    "source checkpoint only"
                    if args.target_cap_mode == "source"
                    else "chronological target support labels only"
                ),
            },
            "pisa_loss_protection": {
                "lambda_agg": args.lambda_agg,
                "lambda_ghost": args.lambda_ghost,
                "lambda_peak": args.lambda_peak,
                "lambda_state": args.lambda_state,
                "warmup_epochs": args.adaptation_aux_warmup_epochs,
                "ramp_epochs": args.adaptation_aux_ramp_epochs,
                "warmup_lambda_agg": args.adaptation_warmup_lambda_agg,
                "warmup_lambda_ghost": args.adaptation_warmup_lambda_ghost,
                "warmup_lambda_peak": args.adaptation_warmup_lambda_peak,
                "warmup_lambda_state": args.adaptation_warmup_lambda_state,
            },
        },
        "source_csv": str(Path(args.source_csv).resolve()),
        "target_csv": str(Path(args.target_csv).resolve()),
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "source_config": str(
            Path(args.source_config).resolve()
            if args.source_config is not None
            else source_run_dir / "config.json"
        ),
        "zero_shot": {
            "validation": zero_validation_metrics["flat"],
            "test": None if zero_test_metrics is None else zero_test_metrics["flat"],
        },
        "few_shot": {},
    }

    if args.zero_shot_only:
        save_json(output_dir / "config.json", vars(args))
        save_json(result_dir / "transfer_summary.json", summary)
        print(f"Zero-shot transfer results saved to: {result_dir}")
        return

    support_cap_profiles = {
        f"{fraction:.3f}": calibrate_support_power_caps(
            target_datasets["train"],
            fraction=fraction,
            appliance_names=appliance_names,
            state_thresholds=source_bundle.state_thresholds,
            source_caps=source_rated_power_caps,
            mode=args.target_cap_mode,
            quantile=args.support_cap_quantile,
            margin=args.support_cap_margin,
        )
        for fraction in args.few_shot_fractions
    }
    for key, cap_profile in support_cap_profiles.items():
        save_json(result_dir / f"support_caps_{key}.json", cap_profile)
        print(
            f"Support caps {key} ({cap_profile['mode']}): "
            f"{cap_profile['calibrated_caps_kw']}"
        )

    for fraction in args.few_shot_fractions:
        key = f"{fraction:.3f}"
        cap_profile = support_cap_profiles[key]
        model = build_source_model(
            source_config,
            power_adapter_dim=transfer_power_adapter_dim,
        )
        load_checkpoint_strict(model, args.source_checkpoint)
        apply_rated_power_caps(model, cap_profile["calibrated_caps_kw"])
        # Configure trainability before the optional support-calibrated source
        # evaluation.  ``evaluate`` uses torch.inference_mode(), after which
        # PyTorch forbids toggling requires_grad on tensors touched there.
        trainable_parameters = configure_transfer_parameters(
            model,
            args.fine_tune_mode,
            adapt_bridge_fusion=args.adapt_bridge_fusion,
        )
        source_candidate_label = (
            "source_zero_shot"
            if args.target_cap_mode == "source"
            else "support_calibrated_source"
        )
        if args.target_cap_mode == "source":
            source_candidate_validation = zero_validation_metrics
        else:
            # Do not pass the trainable instance through evaluate(): that
            # function moves a CPU model to CUDA under inference_mode, which
            # would turn BatchNorm buffers into inference tensors and make the
            # subsequent training update invalid.  The candidate is evaluation
            # only and is intentionally a separate source-model instance.
            source_candidate = build_source_model(
                source_config,
                power_adapter_dim=transfer_power_adapter_dim,
            )
            load_checkpoint_strict(source_candidate, args.source_checkpoint)
            apply_rated_power_caps(
                source_candidate,
                cap_profile["calibrated_caps_kw"],
            )
            source_candidate_validation = evaluate(
                source_candidate,
                target_loaders["val"],
                device=device,
                state_thresholds=source_bundle.state_thresholds,
                appliance_names=appliance_names,
                use_amp=use_amp,
            )
        save_json(
            result_dir / f"{source_candidate_label}_{key}_validation_metrics.json",
            source_candidate_validation,
        )
        train_loader = subset_loader(
            target_datasets["train"],
            fraction=fraction,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        checkpoint_dir = output_dir / "checkpoints" / f"fewshot_{fraction:.3f}"
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4,
        )
        trainer = PISATrainer(
            model=model,
            loss_fn=build_loss(target_bundle, args),
            optimizer=optimizer,
            scheduler=scheduler,
            config=TrainerConfig(
                max_epochs=args.few_shot_epochs,
                device=str(device),
                seed=args.seed,
                use_amp=use_amp,
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
            appliance_names=appliance_names,
            state_thresholds=source_bundle.state_thresholds,
        )
        trainer.fit(train_loader, target_loaders["val"])
        fine_tune_best_mae = trainer.best_metric
        selected_model = source_candidate_label
        selected_epoch = 0
        if fine_tune_best_mae < macro_mae(source_candidate_validation):
            trainer.load_checkpoint(
                checkpoint_dir / "best.pt",
                load_optimizer=False,
                load_scheduler=False,
            )
            validation_metrics = evaluate(
                trainer.model,
                target_loaders["val"],
                device=device,
                state_thresholds=source_bundle.state_thresholds,
                appliance_names=appliance_names,
                use_amp=use_amp,
            )
            selected_model = "few_shot_checkpoint"
            selected_epoch = trainer.best_epoch
        else:
            validation_metrics = source_candidate_validation
        save_json(result_dir / f"fewshot_{key}_validation_metrics.json", validation_metrics)

        test_metrics = None
        if not args.skip_test_evaluation and selected_model == "source_zero_shot":
            test_metrics = zero_test_metrics
        elif not args.skip_test_evaluation and selected_model == "support_calibrated_source":
            source_candidate = build_source_model(
                source_config,
                power_adapter_dim=transfer_power_adapter_dim,
            )
            load_checkpoint_strict(source_candidate, args.source_checkpoint)
            apply_rated_power_caps(source_candidate, cap_profile["calibrated_caps_kw"])
            test_metrics = evaluate(
                source_candidate,
                target_loaders["test"],
                device=device,
                state_thresholds=source_bundle.state_thresholds,
                appliance_names=appliance_names,
                use_amp=use_amp,
            )
        if not args.skip_test_evaluation and selected_model == "few_shot_checkpoint":
            test_metrics = evaluate(
                trainer.model,
                target_loaders["test"],
                device=device,
                state_thresholds=source_bundle.state_thresholds,
                appliance_names=appliance_names,
                use_amp=use_amp,
            )
        if test_metrics is not None:
            save_json(result_dir / f"fewshot_{key}_test_metrics.json", test_metrics)

        summary["few_shot"][key] = {
            "support_fraction": float(fraction),
            "support_windows": cap_profile["support_windows"],
            "power_cap_calibration": cap_profile,
            "trainable_parameters": trainable_parameters,
            "selected_model": selected_model,
            "selected_epoch": selected_epoch,
            "selected_validation_mae": macro_mae(validation_metrics),
            "source_candidate_validation_mae": macro_mae(source_candidate_validation),
            "fine_tune_best_epoch": trainer.best_epoch,
            "fine_tune_best_validation_mae": fine_tune_best_mae,
            "validation": validation_metrics["flat"],
            "test": None if test_metrics is None else test_metrics["flat"],
        }
        print(
            f"Few-shot {key} target validation macro MAE="
            f"{macro_mae(validation_metrics):.6f} "
            f"(selected {selected_model}, epoch {selected_epoch})"
        )

    # A diagnostic aggregate-share proxy is not a fair deployed baseline: it
    # has access to future mains and historical appliance labels.  When an
    # actual AggregateOnlySeq2Seq source checkpoint is supplied, evaluate it
    # under the same support, target-validation selection and target-test rules
    # as PISA instead.
    if args.aggregate_only_checkpoint is not None:
        assert aggregate_config is not None
        aggregate_model = build_aggregate_only_model(aggregate_config)
        load_checkpoint_strict(aggregate_model, args.aggregate_only_checkpoint)
        aggregate_zero_validation = evaluate(
            aggregate_model,
            target_loaders["val"],
            device=device,
            state_thresholds=source_bundle.state_thresholds,
            appliance_names=appliance_names,
            use_amp=use_amp,
        )
        save_json(
            result_dir / "aggregate_only_zero_shot_validation_metrics.json",
            aggregate_zero_validation,
        )
        print(
            "AggregateOnly zero-shot target validation macro MAE="
            f"{macro_mae(aggregate_zero_validation):.6f}"
        )

        aggregate_zero_test = None
        if not args.skip_test_evaluation:
            aggregate_zero_test = evaluate(
                aggregate_model,
                target_loaders["test"],
                device=device,
                state_thresholds=source_bundle.state_thresholds,
                appliance_names=appliance_names,
                use_amp=use_amp,
            )
            save_json(
                result_dir / "aggregate_only_zero_shot_test_metrics.json",
                aggregate_zero_test,
            )

        aggregate_summary: dict[str, Any] = {
            "source_checkpoint": str(Path(args.aggregate_only_checkpoint).resolve()),
            "source_config": str(
                Path(args.aggregate_only_config).resolve()
                if args.aggregate_only_config is not None
                else Path(args.aggregate_only_checkpoint).resolve().parent.parent
                / "config.json"
            ),
            "fine_tune_mode": "power_and_state_output_heads",
            "power_cap_calibration": "uses the same per-fraction caps as PISA",
            "zero_shot": {
                "validation": aggregate_zero_validation["flat"],
                "test": (
                    None if aggregate_zero_test is None else aggregate_zero_test["flat"]
                ),
            },
            "few_shot": {},
        }

        for fraction in args.few_shot_fractions:
            key = f"{fraction:.3f}"
            cap_profile = support_cap_profiles[key]
            model = build_aggregate_only_model(aggregate_config)
            load_checkpoint_strict(model, args.aggregate_only_checkpoint)
            apply_rated_power_caps(model, cap_profile["calibrated_caps_kw"])
            trainable_parameters = configure_aggregate_only_parameters(model)
            source_candidate_label = (
                "source_zero_shot"
                if args.target_cap_mode == "source"
                else "support_calibrated_source"
            )
            if args.target_cap_mode == "source":
                source_candidate_validation = aggregate_zero_validation
            else:
                source_candidate = build_aggregate_only_model(aggregate_config)
                load_checkpoint_strict(
                    source_candidate,
                    args.aggregate_only_checkpoint,
                )
                apply_rated_power_caps(
                    source_candidate,
                    cap_profile["calibrated_caps_kw"],
                )
                source_candidate_validation = evaluate(
                    source_candidate,
                    target_loaders["val"],
                    device=device,
                    state_thresholds=source_bundle.state_thresholds,
                    appliance_names=appliance_names,
                    use_amp=use_amp,
                )
            save_json(
                result_dir
                / f"aggregate_only_{source_candidate_label}_{key}_validation_metrics.json",
                source_candidate_validation,
            )
            train_loader = subset_loader(
                target_datasets["train"],
                fraction=fraction,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            checkpoint_dir = (
                output_dir / "checkpoints" / f"aggregate_only_fewshot_{fraction:.3f}"
            )
            optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=4,
            )
            trainer = PISATrainer(
                model=model,
                loss_fn=build_loss(target_bundle, args),
                optimizer=optimizer,
                scheduler=scheduler,
                config=TrainerConfig(
                    max_epochs=args.few_shot_epochs,
                    device=str(device),
                    seed=args.seed,
                    use_amp=use_amp,
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
                appliance_names=appliance_names,
                state_thresholds=source_bundle.state_thresholds,
            )
            trainer.fit(train_loader, target_loaders["val"])
            fine_tune_best_mae = trainer.best_metric
            selected_model = source_candidate_label
            selected_epoch = 0
            if fine_tune_best_mae < macro_mae(source_candidate_validation):
                trainer.load_checkpoint(
                    checkpoint_dir / "best.pt",
                    load_optimizer=False,
                    load_scheduler=False,
                )
                validation_metrics = evaluate(
                    trainer.model,
                    target_loaders["val"],
                    device=device,
                    state_thresholds=source_bundle.state_thresholds,
                    appliance_names=appliance_names,
                    use_amp=use_amp,
                )
                selected_model = "few_shot_checkpoint"
                selected_epoch = trainer.best_epoch
            else:
                validation_metrics = source_candidate_validation
            save_json(
                result_dir / f"aggregate_only_fewshot_{key}_validation_metrics.json",
                validation_metrics,
            )

            test_metrics = None
            if not args.skip_test_evaluation and selected_model == "source_zero_shot":
                test_metrics = aggregate_zero_test
            elif (
                not args.skip_test_evaluation
                and selected_model == "support_calibrated_source"
            ):
                source_candidate = build_aggregate_only_model(aggregate_config)
                load_checkpoint_strict(source_candidate, args.aggregate_only_checkpoint)
                apply_rated_power_caps(
                    source_candidate,
                    cap_profile["calibrated_caps_kw"],
                )
                test_metrics = evaluate(
                    source_candidate,
                    target_loaders["test"],
                    device=device,
                    state_thresholds=source_bundle.state_thresholds,
                    appliance_names=appliance_names,
                    use_amp=use_amp,
                )
            if not args.skip_test_evaluation and selected_model == "few_shot_checkpoint":
                test_metrics = evaluate(
                    trainer.model,
                    target_loaders["test"],
                    device=device,
                    state_thresholds=source_bundle.state_thresholds,
                    appliance_names=appliance_names,
                    use_amp=use_amp,
                )
            if test_metrics is not None:
                save_json(
                    result_dir / f"aggregate_only_fewshot_{key}_test_metrics.json",
                    test_metrics,
                )

            aggregate_summary["few_shot"][key] = {
                "support_fraction": float(fraction),
                "support_windows": cap_profile["support_windows"],
                "power_cap_calibration": cap_profile,
                "trainable_parameters": trainable_parameters,
                "selected_model": selected_model,
                "selected_epoch": selected_epoch,
                "selected_validation_mae": macro_mae(validation_metrics),
                "source_candidate_validation_mae": macro_mae(source_candidate_validation),
                "fine_tune_best_epoch": trainer.best_epoch,
                "fine_tune_best_validation_mae": fine_tune_best_mae,
                "validation": validation_metrics["flat"],
                "test": None if test_metrics is None else test_metrics["flat"],
            }
            print(
                f"AggregateOnly few-shot {key} target validation macro MAE="
                f"{macro_mae(validation_metrics):.6f} "
                f"(selected {selected_model}, epoch {selected_epoch})"
            )

        summary["aggregate_only_seq2seq"] = aggregate_summary

    save_json(output_dir / "config.json", vars(args))
    save_json(result_dir / "transfer_summary.json", summary)
    print(f"Transfer results saved to: {result_dir}")


if __name__ == "__main__":
    main()
