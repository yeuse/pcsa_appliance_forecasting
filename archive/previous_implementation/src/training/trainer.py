from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.evaluation.metrics import MetricsAccumulator, flatten_metrics
from src.checkpoint_selection import on_safety_status, monitor_improved


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """
    Move tensor values in batch to device.

    Non-tensor values are kept unchanged.
    """
    moved = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value

    return moved


def get_batch_size(batch: dict[str, Any]) -> int:
    """
    Infer batch size from common batch fields.
    """
    for key in ["x_hist", "y_power", "y_state", "y_event"]:
        if key in batch and torch.is_tensor(batch[key]):
            return int(batch[key].shape[0])

    return 1


class AverageMeter:
    """
    Weighted average meter for scalar logging.
    """

    def __init__(self) -> None:
        self.sum: dict[str, float] = {}
        self.count: dict[str, float] = {}

    def update(self, values: dict[str, Any], n: int = 1) -> None:
        for key, value in values.items():
            if torch.is_tensor(value):
                value = float(value.detach().cpu().item())
            elif isinstance(value, np.ndarray):
                value = float(value.item())
            else:
                value = float(value)

            if math.isnan(value) or math.isinf(value):
                continue

            self.sum[key] = self.sum.get(key, 0.0) + value * n
            self.count[key] = self.count.get(key, 0.0) + n

    def averages(self) -> dict[str, float]:
        return {
            key: self.sum[key] / max(self.count[key], 1.0)
            for key in self.sum.keys()
        }


@dataclass
class TrainerConfig:
    """
    Basic trainer configuration.
    """

    max_epochs: int = 100

    device: str | None = None
    seed: int = 42

    use_amp: bool = True
    grad_clip_norm: Optional[float] = 1.0

    log_interval: int = 50

    checkpoint_dir: str = "outputs/checkpoints"
    save_best: bool = True
    save_last: bool = True

    monitor: str = "val/loss_total"
    monitor_mode: str = "min"
    early_stopping_patience: int = 20
    min_epochs_before_stopping: int = 0
    min_delta: float = 1e-6

    eval_metrics_every: int = 1

    # For hierarchy risk fine-tuning, retain base y_power first and then
    # gradually blend hierarchy scenario power into structured appliances.
    hierarchical_power_blend: float = 1.0
    hierarchical_blend_warmup_epochs: int = 0
    hierarchical_blend_ramp_epochs: int = 0

    # Optional secondary checkpoint for risk fine-tuning.  It maximizes a
    # risk metric only while the core MAE remains below a fixed safety ceiling.
    risk_monitor: str | None = None
    risk_monitor_mode: str = "max"
    risk_mae_monitor: str = "val/regression/macro_avg/MAE"
    risk_mae_ceiling: float | None = None
    risk_checkpoint_start_epoch: int = 1
    risk_checkpoint_name: str = "risk_best.pt"

    # The optimization scheduler may follow a different metric from the
    # safety checkpoint monitor (for example EventF1 during risk fine-tuning).
    scheduler_monitor: str | None = None
    scheduler_start_epoch: int = 1

    # Metrics used by stage-level checkpoint selection.  Final evaluation may
    # override these values with calibrated thresholds, but training needs the
    # same tolerance-aware metric available every epoch.
    event_tolerance_minutes: int = 0
    event_bucket_size: int = 0

    # Optional multi-home domain-generalization auxiliary objective.  When a
    # balanced batch contains ``home_index``, retain the pooled ERM task loss
    # and add the mean plus dispersion of per-home future-power Macro MAE,
    # normalized by each home's zero-forecast Macro MAE.
    domain_generalization_weight: float = 0.0
    domain_generalization_dispersion_weight: float = 0.25
    domain_generalization_warmup_epochs: int = 0
    domain_generalization_ramp_epochs: int = 0
    # Convex weight of source-home ON-condition normalized MAE in the
    # checkpoint-selection metric.  The complementary weight is assigned to
    # ordinary all-step normalized MAE.
    source_on_monitor_weight: float = 0.5
    # Optional hard eligibility constraint for best.pt. When configured, an
    # epoch may improve the primary monitor only if the worst source home's
    # normalized ON-condition MAE is no larger than this ceiling.
    source_on_safety_ceiling: float | None = None
    source_on_safety_scope: str = "home"
    source_on_min_windows: int = 5

    history_file: str = "history.json"


class PISATrainer:
    """
    Trainer for PISA.

    Expected model:
        out = model(batch)

    Expected loss function:
        loss, loss_dict = loss_fn(out, batch)

    Expected dataloaders:
        train_loader, val_loader

    Optional metrics:
        MetricsAccumulator is used on validation set.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        config: Optional[TrainerConfig | dict[str, Any]] = None,
        scheduler: Optional[Any] = None,
        appliance_names: Optional[list[str]] = None,
        state_thresholds: Optional[Any] = None,
    ) -> None:
        if config is None:
            self.cfg = TrainerConfig()
        elif isinstance(config, dict):
            self.cfg = TrainerConfig(**config)
        elif isinstance(config, TrainerConfig):
            self.cfg = config
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        if self.cfg.monitor_mode not in ["min", "max"]:
            raise ValueError("monitor_mode should be 'min' or 'max'.")
        if self.cfg.risk_monitor_mode not in ["min", "max"]:
            raise ValueError("risk_monitor_mode should be 'min' or 'max'.")
        if not 0.0 <= float(self.cfg.hierarchical_power_blend) <= 1.0:
            raise ValueError("hierarchical_power_blend should be in [0, 1].")
        if (
            self.cfg.hierarchical_blend_warmup_epochs < 0
            or self.cfg.hierarchical_blend_ramp_epochs < 0
        ):
            raise ValueError(
                "hierarchical blend warmup and ramp epochs must be >= 0."
            )
        if self.cfg.risk_checkpoint_start_epoch < 1:
            raise ValueError("risk_checkpoint_start_epoch must be >= 1.")
        if self.cfg.scheduler_start_epoch < 1:
            raise ValueError("scheduler_start_epoch must be >= 1.")
        if self.cfg.event_tolerance_minutes < 0 or self.cfg.event_bucket_size < 0:
            raise ValueError("event tolerance and bucket size must be >= 0.")
        if self.cfg.domain_generalization_weight < 0.0:
            raise ValueError("domain_generalization_weight must be >= 0.")
        if self.cfg.domain_generalization_dispersion_weight < 0.0:
            raise ValueError(
                "domain_generalization_dispersion_weight must be >= 0."
            )
        if not 0.0 <= self.cfg.source_on_monitor_weight <= 1.0:
            raise ValueError("source_on_monitor_weight should be in [0, 1].")
        if self.cfg.source_on_safety_scope not in {"home", "home_appliance"}:
            raise ValueError("Invalid source_on_safety_scope.")
        if self.cfg.source_on_min_windows < 1:
            raise ValueError("source_on_min_windows must be positive.")
        if (
            self.cfg.source_on_safety_ceiling is not None
            and (
                not math.isfinite(float(self.cfg.source_on_safety_ceiling))
                or float(self.cfg.source_on_safety_ceiling) <= 0.0
            )
        ):
            raise ValueError(
                "source_on_safety_ceiling must be finite and positive."
            )
        if (
            self.cfg.domain_generalization_warmup_epochs < 0
            or self.cfg.domain_generalization_ramp_epochs < 0
        ):
            raise ValueError(
                "domain-generalization warmup and ramp epochs must be >= 0."
            )
        if (
            self.cfg.risk_mae_ceiling is not None
            and not math.isfinite(float(self.cfg.risk_mae_ceiling))
        ):
            raise ValueError("risk_mae_ceiling must be finite when provided.")

        set_seed(self.cfg.seed)

        self.device = get_device(self.cfg.device)

        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.appliance_names = appliance_names
        self.state_thresholds = state_thresholds

        self.use_amp = self.cfg.use_amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.checkpoint_dir = Path(self.cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: list[dict[str, float]] = []

        if self.cfg.monitor_mode == "min":
            self.best_metric = float("inf")
        else:
            self.best_metric = -float("inf")

        self.best_epoch = 0
        self.num_bad_epochs = 0
        self.risk_best_metric = (
            float("inf")
            if self.cfg.risk_monitor_mode == "min"
            else -float("inf")
        )
        self.risk_best_epoch = 0
        # Risk fine-tuning uses a secondary validation objective while the
        # deterministic MAE is intentionally frozen. Its early-stopping state
        # must therefore be independent from ``num_bad_epochs`` above.
        self.risk_num_bad_epochs = 0

    def _is_improved(self, metric: float) -> bool:
        return monitor_improved(metric, self.best_metric, self.cfg.monitor_mode, self.cfg.min_delta)

    def _is_risk_improved(self, metric: float) -> bool:
        if self.cfg.risk_monitor_mode == "min":
            return metric < self.risk_best_metric - self.cfg.min_delta
        return metric > self.risk_best_metric + self.cfg.min_delta

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _hierarchical_blend_for_epoch(self, epoch: int) -> float:
        """Return the hierarchy blend scheduled for a one-indexed epoch."""
        blend_max = float(self.cfg.hierarchical_power_blend)
        if blend_max == 0.0 or epoch <= self.cfg.hierarchical_blend_warmup_epochs:
            return 0.0
        if self.cfg.hierarchical_blend_ramp_epochs == 0:
            return blend_max

        progress = (epoch - self.cfg.hierarchical_blend_warmup_epochs) / float(
            self.cfg.hierarchical_blend_ramp_epochs
        )
        return blend_max * min(1.0, max(0.0, progress))

    def _domain_generalization_weight_for_epoch(self, epoch: int) -> float:
        """Return the cross-home risk penalty for a one-indexed epoch."""
        weight_max = float(self.cfg.domain_generalization_weight)
        warmup = int(self.cfg.domain_generalization_warmup_epochs)
        ramp = int(self.cfg.domain_generalization_ramp_epochs)
        if weight_max == 0.0 or epoch <= warmup:
            return 0.0
        if ramp == 0:
            return weight_max
        progress = (epoch - warmup) / float(ramp)
        return weight_max * min(1.0, max(0.0, progress))

    @staticmethod
    def _slice_batch_mapping(
        mapping: dict[str, Any],
        mask: Tensor,
        batch_size: int,
    ) -> dict[str, Any]:
        """Slice batch-first tensors while preserving metadata and constants."""
        sliced: dict[str, Any] = {}
        for key, value in mapping.items():
            if (
                torch.is_tensor(value)
                and value.ndim > 0
                and int(value.shape[0]) == batch_size
            ):
                sliced[key] = value[mask]
            elif isinstance(value, dict):
                sliced[key] = PISATrainer._slice_batch_mapping(
                    value, mask, batch_size
                )
            else:
                sliced[key] = value
        return sliced

    def _per_home_losses(
        self,
        out: dict[str, Any],
        batch: dict[str, Any],
        batch_size: int,
    ) -> list[tuple[int, Tensor, int]]:
        """Re-evaluate the loss for each home represented in one batch."""
        home_index = batch.get("home_index")
        if not torch.is_tensor(home_index) or home_index.ndim == 0:
            return []
        home_index = home_index.reshape(-1)
        if int(home_index.numel()) != batch_size:
            return []

        per_home: list[tuple[int, Tensor, int]] = []
        for home in torch.unique(home_index, sorted=True):
            mask = home_index == home
            count = int(mask.sum().item())
            if count == 0:
                continue
            home_batch = self._slice_batch_mapping(batch, mask, batch_size)
            home_out = self._slice_batch_mapping(out, mask, batch_size)
            home_loss, _ = self.loss_fn(home_out, home_batch)
            per_home.append((int(home.item()), home_loss, count))
        return per_home

    @staticmethod
    def _normalized_power_risk(
        out: dict[str, Any],
        batch: dict[str, Any],
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return raw, zero-baseline and normalized future-power Macro MAE.

        This deliberately matches the primary evaluation endpoint: unweighted
        MAE in the original power unit, macro-averaged across appliances.  The
        zero-forecast denominator makes risks from households with different
        load magnitudes comparable.  It is detached so the denominator cannot
        become an optimization route.
        """
        if "y_power" not in out or "y_power" not in batch:
            raise KeyError(
                "Cross-home power risk requires y_power in model output and batch."
            )
        pred = out["y_power"][mask]
        target = batch["y_power"][mask].to(
            device=pred.device, dtype=pred.dtype
        )
        valid = batch.get("target_mask")
        if valid is None:
            valid = torch.ones_like(target)
        else:
            valid = valid[mask].to(device=pred.device, dtype=pred.dtype)

        counts = valid.sum(dim=(0, 2))
        present = counts > 0
        if not torch.any(present):
            zero = pred.new_tensor(0.0)
            return zero, zero, zero
        safe_counts = counts.clamp_min(1.0)
        per_app_mae = (
            (pred - target).abs() * valid
        ).sum(dim=(0, 2)) / safe_counts
        per_app_zero_mae = (
            target.abs() * valid
        ).sum(dim=(0, 2)) / safe_counts
        raw_macro_mae = per_app_mae[present].mean()
        zero_macro_mae = per_app_zero_mae[present].mean()
        normalized_mae = raw_macro_mae / zero_macro_mae.detach().clamp_min(1e-6)
        return raw_macro_mae, zero_macro_mae, normalized_mae

    def _per_home_normalized_power_risks(
        self,
        out: dict[str, Any],
        batch: dict[str, Any],
        batch_size: int,
    ) -> list[tuple[int, Tensor, Tensor, Tensor, int]]:
        """Return differentiable future-power risks for homes in one batch."""
        home_index = batch.get("home_index")
        if not torch.is_tensor(home_index) or home_index.ndim == 0:
            return []
        home_index = home_index.reshape(-1)
        if int(home_index.numel()) != batch_size:
            return []

        risks: list[tuple[int, Tensor, Tensor, Tensor, int]] = []
        for home in torch.unique(home_index, sorted=True):
            mask = home_index == home
            count = int(mask.sum().item())
            if count == 0:
                continue
            raw_mae, zero_mae, normalized_mae = self._normalized_power_risk(
                out, batch, mask
            )
            risks.append(
                (int(home.item()), raw_mae, zero_mae, normalized_mae, count)
            )
        return risks

    def _cross_home_power_objective(
        self,
        per_home: list[tuple[int, Tensor, Tensor, Tensor, int]],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return mean, SD and risk-consistent normalized power objective."""
        risks = torch.stack([item[3] for item in per_home])
        mean_risk = risks.mean()
        risk_std = torch.sqrt(torch.mean((risks - mean_risk) ** 2) + 1e-12)
        objective = mean_risk + (
            float(self.cfg.domain_generalization_dispersion_weight) * risk_std
        )
        return mean_risk, risk_std, objective

    def _set_epoch_schedules(self, epoch: int) -> float:
        # PISALoss uses the epoch to delay and ramp sparse future-risk losses.
        # Keep this duck-typed so other loss functions remain supported.
        if hasattr(self.loss_fn, "set_epoch"):
            self.loss_fn.set_epoch(epoch)

        hierarchy_blend = 0.0
        if (
            getattr(self.model, "hierarchical_future", False)
            and hasattr(self.model, "set_hierarchical_power_blend")
        ):
            hierarchy_blend = self._hierarchical_blend_for_epoch(epoch)
            self.model.set_hierarchical_power_blend(hierarchy_blend)
        return hierarchy_blend

    def train_one_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        self.model.train()
        hierarchy_blend = self._set_epoch_schedules(epoch)
        dg_weight = self._domain_generalization_weight_for_epoch(epoch)

        meter = AverageMeter()

        start_time = time.time()
        supervision_counts = None
        supervision_batches = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, self.device)
            batch_size = get_batch_size(batch)
            if "y_state" in batch:
                state = batch["y_state"]
                valid = batch.get("target_mask", torch.ones_like(state))
                on_count = (valid * state).sum(dim=(0, 2)).detach().cpu().numpy()
                off_count = (valid * (1.0 - state)).sum(dim=(0, 2)).detach().cpu().numpy()
                if supervision_counts is None:
                    supervision_counts = np.zeros((4, len(on_count)), dtype=np.float64)
                supervision_counts += np.stack([on_count > 0, off_count > 0, on_count, off_count])
                supervision_batches += 1

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                out = self.model(batch)
                loss, loss_dict = self.loss_fn(out, batch)
                if dg_weight > 0.0:
                    per_home = self._per_home_normalized_power_risks(
                        out, batch, batch_size
                    )
                    if len(per_home) >= 2:
                        home_mean, home_std, home_objective = (
                            self._cross_home_power_objective(per_home)
                        )
                        pooled_loss = loss
                        loss = pooled_loss + dg_weight * home_objective
                        loss_dict = dict(loss_dict)
                        loss_dict["loss_pooled"] = pooled_loss.detach()
                        loss_dict["loss_cross_home_power_mean"] = home_mean
                        loss_dict["loss_cross_home_power_std"] = home_std
                        loss_dict["loss_cross_home_power_objective"] = (
                            home_objective
                        )
                        loss_dict["domain_generalization_weight"] = (
                            loss.new_tensor(dg_weight)
                        )
                        loss_dict["loss_total"] = loss

            if self.use_amp:
                self.scaler.scale(loss).backward()

                if self.cfg.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.grad_clip_norm,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                if self.cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.grad_clip_norm,
                    )

                self.optimizer.step()

            meter.update(loss_dict, n=batch_size)

            if self.cfg.log_interval > 0 and step % self.cfg.log_interval == 0:
                avg = meter.averages()
                msg = (
                    f"[Epoch {epoch:03d}] "
                    f"step {step:05d}/{len(train_loader):05d} "
                    f"loss={avg.get('loss_total', float('nan')):.6f} "
                    f"lr={self._current_lr():.3e}"
                )
                print(msg)

        stats = meter.averages()
        stats["epoch_time_sec"] = time.time() - start_time
        if supervision_counts is not None:
            for index in range(supervision_counts.shape[1]):
                app = self.appliance_names[index] if self.appliance_names else str(index)
                prefix = f"supervision/{app}"
                stats[f"{prefix}/on_batches"] = float(supervision_counts[0, index])
                stats[f"{prefix}/no_on_batches"] = float(supervision_batches - supervision_counts[0, index])
                stats[f"{prefix}/off_batches"] = float(supervision_counts[1, index])
                stats[f"{prefix}/on_target_steps"] = float(supervision_counts[2, index])
                stats[f"{prefix}/off_target_steps"] = float(supervision_counts[3, index])
        stats["lr"] = self._current_lr()
        stats["hierarchical_power_blend"] = hierarchy_blend
        stats["domain_generalization_weight"] = dg_weight

        return stats

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        compute_metrics: bool = True,
        state_thresholds: Optional[Any] = None,
        event_prob_threshold: Optional[Any] = None,
        bucket_event_prob_threshold: Optional[Any] = None,
        window_start_prob_threshold: Optional[Any] = None,
        hierarchical_start_prob_threshold: Optional[Any] = None,
        event_tolerance_minutes: Optional[int] = None,
        event_bucket_size: Optional[int] = None,
        postprocess_state: bool = False,
        min_on_duration: int = 1,
        min_off_duration: int = 1,
    ) -> dict[str, float]:
        self.model.eval()

        if event_tolerance_minutes is None:
            event_tolerance_minutes = self.cfg.event_tolerance_minutes
        if event_bucket_size is None:
            event_bucket_size = self.cfg.event_bucket_size

        meter = AverageMeter()
        home_loss_sum: dict[int, float] = {}
        home_sample_count: dict[int, int] = {}
        home_power_abs_sum: dict[int, np.ndarray] = {}
        home_zero_abs_sum: dict[int, np.ndarray] = {}
        home_power_count: dict[int, np.ndarray] = {}
        home_on_power_abs_sum: dict[int, np.ndarray] = {}
        home_on_zero_abs_sum: dict[int, np.ndarray] = {}
        home_on_power_count: dict[int, np.ndarray] = {}
        home_on_window_count: dict[int, np.ndarray] = {}

        metrics_acc = None
        if compute_metrics:
            metrics_acc = MetricsAccumulator(
                appliance_names=self.appliance_names
                if self.appliance_names is not None
                else None
            )

        for batch in val_loader:
            batch = move_batch_to_device(batch, self.device)
            batch_size = get_batch_size(batch)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                out = self.model(batch)
                loss, loss_dict = self.loss_fn(out, batch)

            # Always expose per-home validation risk when multi-home metadata
            # is available.  This enables checkpoint selection on the worst
            # observed source home without using held-out-home labels.
            per_home = self._per_home_losses(out, batch, batch_size)
            for home, home_loss, count in per_home:
                home_loss_sum[home] = home_loss_sum.get(home, 0.0) + (
                    float(home_loss.detach().cpu().item()) * count
                )
                home_sample_count[home] = home_sample_count.get(home, 0) + count

            home_index = batch.get("home_index")
            if torch.is_tensor(home_index) and home_index.numel() == batch_size:
                pred_power = out["y_power"].detach().double()
                target_power = batch["y_power"].to(
                    device=pred_power.device, dtype=pred_power.dtype
                )
                target_mask = batch.get("target_mask")
                if target_mask is None:
                    target_mask = torch.ones_like(target_power)
                else:
                    target_mask = target_mask.to(
                        device=pred_power.device, dtype=pred_power.dtype
                    )
                true_state = batch["y_state"].to(
                    device=pred_power.device, dtype=pred_power.dtype
                ).clamp(0.0, 1.0)
                flat_home_index = home_index.reshape(-1)
                for home_tensor in torch.unique(flat_home_index, sorted=True):
                    home = int(home_tensor.item())
                    home_mask = flat_home_index == home_tensor
                    valid = target_mask[home_mask]
                    abs_sum = (
                        (pred_power[home_mask] - target_power[home_mask]).abs()
                        * valid
                    ).sum(dim=(0, 2)).cpu().numpy()
                    zero_sum = (
                        target_power[home_mask].abs() * valid
                    ).sum(dim=(0, 2)).cpu().numpy()
                    counts = valid.sum(dim=(0, 2)).cpu().numpy()
                    on_valid = valid * true_state[home_mask]
                    on_abs_sum = (
                        (pred_power[home_mask] - target_power[home_mask]).abs()
                        * on_valid
                    ).sum(dim=(0, 2)).cpu().numpy()
                    on_zero_sum = (
                        target_power[home_mask].abs() * on_valid
                    ).sum(dim=(0, 2)).cpu().numpy()
                    on_counts = on_valid.sum(dim=(0, 2)).cpu().numpy()
                    on_windows = (on_valid > 0).any(dim=-1).sum(dim=0).cpu().numpy()
                    if home not in home_power_abs_sum:
                        home_power_abs_sum[home] = np.zeros_like(abs_sum)
                        home_zero_abs_sum[home] = np.zeros_like(zero_sum)
                        home_power_count[home] = np.zeros_like(counts)
                        home_on_power_abs_sum[home] = np.zeros_like(on_abs_sum)
                        home_on_zero_abs_sum[home] = np.zeros_like(on_zero_sum)
                        home_on_power_count[home] = np.zeros_like(on_counts)
                        home_on_window_count[home] = np.zeros_like(on_windows)
                    home_power_abs_sum[home] += abs_sum
                    home_zero_abs_sum[home] += zero_sum
                    home_power_count[home] += counts
                    home_on_power_abs_sum[home] += on_abs_sum
                    home_on_zero_abs_sum[home] += on_zero_sum
                    home_on_power_count[home] += on_counts
                    home_on_window_count[home] += on_windows

            meter.update(loss_dict, n=batch_size)

            if metrics_acc is not None:
                metrics_acc.update(out, batch)

        stats = meter.averages()

        if len(home_loss_sum) >= 2:
            home_means = {
                home: home_loss_sum[home] / max(home_sample_count[home], 1)
                for home in sorted(home_loss_sum)
            }
            values = np.asarray(list(home_means.values()), dtype=np.float64)
            for home, value in home_means.items():
                stats[f"domain/home_{home}/loss_total"] = float(value)
            stats["domain/mean_home_loss_total"] = float(values.mean())
            stats["domain/worst_home_loss_total"] = float(values.max())
            stats["domain/std_home_loss_total"] = float(values.std(ddof=0))

        normalized_power_risks: list[float] = []
        for home in sorted(home_power_abs_sum):
            counts = home_power_count[home]
            present = counts > 0
            if not np.any(present):
                continue
            safe_counts = np.maximum(counts, 1.0)
            macro_mae = float(
                np.mean((home_power_abs_sum[home] / safe_counts)[present])
            )
            zero_macro_mae = float(
                np.mean((home_zero_abs_sum[home] / safe_counts)[present])
            )
            normalized_mae = macro_mae / max(zero_macro_mae, 1e-6)
            stats[f"domain/home_{home}/power_macro_MAE"] = macro_mae
            stats[f"domain/home_{home}/zero_macro_MAE"] = zero_macro_mae
            stats[f"domain/home_{home}/normalized_power_MAE"] = normalized_mae
            normalized_power_risks.append(normalized_mae)

        overall_power_risk: float | None = None
        if len(normalized_power_risks) >= 2:
            power_values = np.asarray(normalized_power_risks, dtype=np.float64)
            power_mean = float(power_values.mean())
            power_std = float(power_values.std(ddof=0))
            stats["domain/mean_home_normalized_power_MAE"] = power_mean
            stats["domain/std_home_normalized_power_MAE"] = power_std
            stats["domain/worst_home_normalized_power_MAE"] = float(
                power_values.max()
            )
            overall_power_risk = (
                power_mean
                + float(self.cfg.domain_generalization_dispersion_weight)
                * power_std
            )
            stats["domain/risk_consistent_normalized_power_MAE"] = overall_power_risk

        normalized_on_power_risks: list[float] = []
        pair_on_risks: list[float] = []
        total_pairs = 0
        supported_pairs = 0
        for home in sorted(home_on_power_abs_sum):
            counts = home_on_power_count[home]
            for index, count in enumerate(counts):
                app = self.appliance_names[index] if self.appliance_names else str(index)
                prefix = f"domain/home_{home}/{app}"
                windows = int(home_on_window_count[home][index])
                supported = count > 0 and windows >= self.cfg.source_on_min_windows
                total_pairs += 1
                supported_pairs += int(supported)
                stats[f"{prefix}/on_valid_steps"] = float(count)
                stats[f"{prefix}/on_windows"] = float(windows)
                stats[f"{prefix}/on_supported"] = float(supported)
                if count > 0:
                    app_on_mae = float(home_on_power_abs_sum[home][index] / count)
                    app_zero_mae = float(home_on_zero_abs_sum[home][index] / count)
                    ratio = app_on_mae / max(app_zero_mae, 1e-6)
                    stats[f"{prefix}/on_MAE"] = app_on_mae
                    stats[f"{prefix}/zero_on_MAE"] = app_zero_mae
                    stats[f"{prefix}/normalized_on_MAE"] = ratio
                    if supported:
                        pair_on_risks.append(ratio)
            present = counts > 0
            if not np.any(present):
                continue
            safe_counts = np.maximum(counts, 1.0)
            on_macro_mae = float(
                np.mean((home_on_power_abs_sum[home] / safe_counts)[present])
            )
            zero_on_macro_mae = float(
                np.mean((home_on_zero_abs_sum[home] / safe_counts)[present])
            )
            normalized_on_mae = on_macro_mae / max(zero_on_macro_mae, 1e-6)
            stats[f"domain/home_{home}/on_power_macro_MAE"] = on_macro_mae
            stats[f"domain/home_{home}/zero_on_power_macro_MAE"] = zero_on_macro_mae
            stats[f"domain/home_{home}/normalized_on_power_MAE"] = normalized_on_mae
            normalized_on_power_risks.append(normalized_on_mae)

        if total_pairs:
            stats["domain/on_supported_pairs"] = float(supported_pairs)
            stats["domain/on_expected_pairs"] = float(total_pairs)
            stats["domain/on_pair_coverage"] = supported_pairs / total_pairs
            stats["domain/worst_home_appliance_normalized_on_power_MAE"] = (
                max(pair_on_risks) if pair_on_risks else float("nan")
            )

        on_power_risk: float | None = None
        if len(normalized_on_power_risks) >= 2:
            on_values = np.asarray(normalized_on_power_risks, dtype=np.float64)
            on_mean = float(on_values.mean())
            on_std = float(on_values.std(ddof=0))
            stats["domain/mean_home_normalized_on_power_MAE"] = on_mean
            stats["domain/std_home_normalized_on_power_MAE"] = on_std
            stats["domain/worst_home_normalized_on_power_MAE"] = float(
                on_values.max()
            )
            on_power_risk = (
                on_mean
                + float(self.cfg.domain_generalization_dispersion_weight)
                * on_std
            )
            stats["domain/risk_consistent_normalized_on_power_MAE"] = on_power_risk

        if overall_power_risk is not None and on_power_risk is not None:
            on_weight = float(self.cfg.source_on_monitor_weight)
            stats["domain/combined_overall_on_normalized_power_MAE"] = (
                (1.0 - on_weight) * overall_power_risk
                + on_weight * on_power_risk
            )

        if metrics_acc is not None:
            metrics = metrics_acc.compute(
                state_thresholds=(
                    self.state_thresholds
                    if state_thresholds is None
                    else state_thresholds
                ),
                event_prob_threshold=event_prob_threshold,
                bucket_event_prob_threshold=bucket_event_prob_threshold,
                window_start_prob_threshold=window_start_prob_threshold,
                hierarchical_start_prob_threshold=hierarchical_start_prob_threshold,
                event_tolerance_minutes=event_tolerance_minutes,
                event_bucket_size=event_bucket_size,
                postprocess_state=postprocess_state,
                min_on_duration=min_on_duration,
                min_off_duration=min_off_duration,
            )
            flat_metrics = flatten_metrics(metrics)

            for key, value in flat_metrics.items():
                stats[key] = value

        return stats

    def _step_scheduler(self, monitor_value: Optional[float]) -> None:
        if self.scheduler is None:
            return

        if isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):
            if monitor_value is not None:
                self.scheduler.step(monitor_value)
        else:
            self.scheduler.step()

    def _json_safe(self, obj: Any) -> Any:
        """
        Convert tensors/numpy/nan into JSON-safe values.
        """
        if isinstance(obj, dict):
            return {k: self._json_safe(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._json_safe(v) for v in obj]

        if torch.is_tensor(obj):
            if obj.numel() == 1:
                return self._json_safe(obj.item())
            return obj.detach().cpu().tolist()

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.generic):
            return obj.item()

        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj

        return obj

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        is_best: bool = False,
        checkpoint_role: str = "monitor_best",
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "num_bad_epochs": self.num_bad_epochs,
            "risk_best_metric": self.risk_best_metric,
            "risk_best_epoch": self.risk_best_epoch,
            "risk_num_bad_epochs": self.risk_num_bad_epochs,
            "config": asdict(self.cfg),
            "history": self.history,
            "is_best": is_best,
            "checkpoint_role": checkpoint_role,
        }

        if hasattr(self.loss_fn, "risk_loss_scale"):
            checkpoint["risk_loss_scale"] = float(self.loss_fn.risk_loss_scale())
        if hasattr(self.model, "get_hierarchical_power_blend"):
            checkpoint["hierarchical_power_blend"] = float(
                self.model.get_hierarchical_power_blend()
            )

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        temporary = path.with_name(path.name + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    def load_checkpoint(
        self,
        path: str | Path,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        map_location: Optional[str | torch.device] = None,
    ) -> dict[str, Any]:
        path = Path(path)

        if map_location is None:
            map_location = self.device

        checkpoint = torch.load(path, map_location=map_location)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if (
            load_scheduler
            and self.scheduler is not None
            and "scheduler_state_dict" in checkpoint
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_metric = checkpoint.get("best_metric", self.best_metric)
        self.best_epoch = checkpoint.get("best_epoch", self.best_epoch)
        self.num_bad_epochs = checkpoint.get(
            "num_bad_epochs", self.num_bad_epochs
        )
        self.risk_best_metric = checkpoint.get(
            "risk_best_metric", self.risk_best_metric
        )
        self.risk_best_epoch = checkpoint.get(
            "risk_best_epoch", self.risk_best_epoch
        )
        self.risk_num_bad_epochs = checkpoint.get(
            "risk_num_bad_epochs", self.risk_num_bad_epochs
        )
        self.history = checkpoint.get("history", self.history)

        return checkpoint

    def save_history(self) -> None:
        path = self.checkpoint_dir / self.cfg.history_file

        temporary = path.with_name(path.name + ".tmp")
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(
                self._json_safe(self.history),
                f,
                indent=2,
                ensure_ascii=False,
            )
        temporary.replace(path)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        start_epoch: int = 1,
    ) -> list[dict[str, float]]:
        start_epoch = int(start_epoch)
        if start_epoch < 1:
            raise ValueError("start_epoch must be >= 1.")
        if start_epoch == 1 and any(
            (self.checkpoint_dir / name).exists() for name in ("best.pt", "last.pt")
        ):
            raise FileExistsError("Fresh training requires a new checkpoint directory.")
        print(f"Using device: {self.device}")
        print(f"AMP enabled: {self.use_amp}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        if start_epoch > 1:
            print(f"Resuming training at epoch {start_epoch}.")

        scheduler_monitor = self.cfg.scheduler_monitor or self.cfg.monitor
        monitor_requires_metrics = (
            not self.cfg.monitor.startswith("val/loss_")
            or not scheduler_monitor.startswith("val/loss_")
        )
        if monitor_requires_metrics and self.cfg.eval_metrics_every != 1:
            print(
                "Forcing validation metrics every epoch because checkpoint "
                f"monitor '{self.cfg.monitor}' is not a loss metric."
            )

        for epoch in range(start_epoch, self.cfg.max_epochs + 1):
            compute_metrics = monitor_requires_metrics or (
                self.cfg.eval_metrics_every > 0
                and epoch % self.cfg.eval_metrics_every == 0
            )

            train_stats = self.train_one_epoch(train_loader, epoch)
            val_stats = self.validate(
                val_loader,
                compute_metrics=compute_metrics,
            )

            epoch_record: dict[str, float] = {"epoch": float(epoch)}

            for key, value in train_stats.items():
                epoch_record[f"train/{key}"] = value

            for key, value in val_stats.items():
                epoch_record[f"val/{key}"] = value

            monitor_value = epoch_record.get(self.cfg.monitor, None)

            if monitor_value is None:
                available = list(epoch_record.keys())
                raise KeyError(
                    f"Monitor key '{self.cfg.monitor}' not found. "
                    f"Available keys include: {available[:20]}"
                )

            source_on_safety_eligible, source_on_safety_value, coverage = on_safety_status(
                epoch_record, self.cfg.source_on_safety_ceiling, self.cfg.source_on_safety_scope,
            )
            epoch_record["checkpoint/source_on_safety_value"] = float(
                source_on_safety_value
            )
            epoch_record["checkpoint/source_on_safety_eligible"] = float(
                source_on_safety_eligible
            )
            epoch_record["checkpoint/source_on_pair_coverage"] = float(coverage)

            if epoch >= self.cfg.scheduler_start_epoch:
                scheduler_value = epoch_record.get(scheduler_monitor, None)
                if scheduler_value is None:
                    raise KeyError(
                        f"Scheduler monitor key '{scheduler_monitor}' not found."
                    )
                self._step_scheduler(scheduler_value)

            improved = (
                source_on_safety_eligible
                and math.isfinite(monitor_value)
                and self._is_improved(monitor_value)
            )

            if improved:
                self.best_metric = monitor_value
                self.best_epoch = epoch
                self.num_bad_epochs = 0

            else:
                # Before the first safety-eligible epoch, do not early-stop a
                # run merely because no checkpoint is yet admissible. Once a
                # safe best exists, unsafe epochs count as non-improvements.
                if source_on_safety_eligible or self.best_epoch > 0:
                    self.num_bad_epochs += 1
                else:
                    self.num_bad_epochs = 0

            risk_metric = float("nan")
            risk_mae = float("nan")
            risk_eligible = False
            risk_improved = False
            if self.cfg.risk_monitor is not None:
                risk_metric = epoch_record.get(self.cfg.risk_monitor, float("nan"))
                risk_mae = epoch_record.get(self.cfg.risk_mae_monitor, float("nan"))
                risk_eligible = (
                    epoch >= self.cfg.risk_checkpoint_start_epoch
                    and math.isfinite(risk_metric)
                    and math.isfinite(risk_mae)
                    and (
                        self.cfg.risk_mae_ceiling is None
                        or risk_mae <= float(self.cfg.risk_mae_ceiling)
                    )
                )
                if risk_eligible and self._is_risk_improved(risk_metric):
                    risk_improved = True
                    self.risk_best_metric = risk_metric
                    self.risk_best_epoch = epoch
                    self.risk_num_bad_epochs = 0
                elif epoch >= self.cfg.risk_checkpoint_start_epoch:
                    # Once the warm-up/ramp is complete, every epoch without an
                    # eligible risk improvement counts toward risk-specific
                    # early stopping. Frozen-MAE ``num_bad_epochs`` is ignored.
                    self.risk_num_bad_epochs += 1

                epoch_record["risk/checkpoint_eligible"] = float(risk_eligible)
                epoch_record["risk/bad_epochs"] = float(
                    self.risk_num_bad_epochs
                )

            epoch_record["checkpoint/selected_best_epoch"] = float(self.best_epoch)
            self.history.append(epoch_record)
            if improved and self.cfg.save_best:
                self.save_checkpoint(
                    self.checkpoint_dir / "best.pt", epoch=epoch, is_best=True,
                    checkpoint_role=("source_on_safe_best"
                        if self.cfg.source_on_safety_ceiling is not None else "mae_safe_best"),
                )
            if risk_improved:
                self.save_checkpoint(
                    self.checkpoint_dir / self.cfg.risk_checkpoint_name,
                    epoch=epoch, is_best=True,
                    checkpoint_role="risk_best_under_mae_ceiling",
                )
            if self.cfg.save_last:
                self.save_checkpoint(
                    self.checkpoint_dir / "last.pt",
                    epoch=epoch,
                    is_best=False,
                    checkpoint_role="last",
                )

            self.save_history()

            train_loss = epoch_record.get("train/loss_total", float("nan"))
            val_loss = epoch_record.get("val/loss_total", float("nan"))

            val_mae = epoch_record.get(
                "val/regression/macro_avg/MAE",
                float("nan"),
            )
            val_hard_mae = epoch_record.get(
                "val/hard_regression/macro_avg/MAE",
                float("nan"),
            )
            val_zero_mae = epoch_record.get(
                "val/baseline_zero/regression/macro_avg/MAE",
                float("nan"),
            )
            val_persistence_mae = epoch_record.get(
                "val/baseline_persistence/regression/macro_avg/MAE",
                float("nan"),
            )
            val_event_f1 = epoch_record.get(
                "val/event/macro_avg/EventF1",
                float("nan"),
            )
            val_hierarchical_start_f1 = epoch_record.get(
                "val/hierarchical_start_tolerant/macro_avg/EventF1",
                float("nan"),
            )
            val_ghost = epoch_record.get(
                "val/ghost/macro_avg/GhostMean",
                float("nan"),
            )
            risk_scale = epoch_record.get(
                "train/risk_loss_scale",
                float("nan"),
            )
            hierarchy_blend = epoch_record.get(
                "train/hierarchical_power_blend",
                float("nan"),
            )

            risk_selection_text = ""
            if self.cfg.risk_monitor is not None:
                ceiling_text = (
                    "none"
                    if self.cfg.risk_mae_ceiling is None
                    else f"{self.cfg.risk_mae_ceiling:.6f}"
                )
                risk_selection_text = (
                    f" risk_candidate={risk_metric:.6f}"
                    f" risk_eligible={risk_eligible}"
                    f" risk_mae_ceiling={ceiling_text}"
                    f" risk_best={self.risk_best_metric:.6f}@{self.risk_best_epoch}"
                    f" risk_bad_epochs={self.risk_num_bad_epochs}"
                )

            source_on_safety_text = ""
            if self.cfg.source_on_safety_ceiling is not None:
                source_on_safety_text = (
                    f" source_on_worst={source_on_safety_value:.6f}"
                    f" source_on_ceiling={self.cfg.source_on_safety_ceiling:.6f}"
                    f" source_on_eligible={source_on_safety_eligible}"
                    f" source_on_coverage={coverage:.3f}"
                )

            baseline_text = ""
            if math.isfinite(val_zero_mae):
                baseline_text += (
                    f" zero_MAE={val_zero_mae:.6f} "
                    f"beats_zero={val_mae < val_zero_mae}"
                )
            if math.isfinite(val_persistence_mae):
                baseline_text += (
                    f" persistence_oracle_MAE={val_persistence_mae:.6f}"
                )

            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"val_MAE={val_mae:.6f} "
                f"val_hard_MAE={val_hard_mae:.6f} "
                f"val_EventF1={val_event_f1:.6f} "
                f"val_HierStartTolF1={val_hierarchical_start_f1:.6f} "
                f"val_GhostMean={val_ghost:.6f} "
                f"risk_scale={risk_scale:.2f}"
                f" hierarchy_blend={hierarchy_blend:.2f}"
                f"{baseline_text} "
                f"monitor={monitor_value:.6f} "
                f"best={self.best_metric:.6f}@{self.best_epoch} "
                f"bad_epochs={self.num_bad_epochs}"
                f"{source_on_safety_text}"
                f"{risk_selection_text}"
            )

            risk_early_stopping = self.cfg.risk_monitor is not None
            early_stopping_bad_epochs = (
                self.risk_num_bad_epochs
                if risk_early_stopping
                else self.num_bad_epochs
            )
            early_stopping_monitor = (
                self.cfg.risk_monitor
                if risk_early_stopping
                else self.cfg.monitor
            )
            earliest_stop_epoch = max(
                0,
                int(self.cfg.min_epochs_before_stopping),
                (
                    int(self.cfg.risk_checkpoint_start_epoch)
                    if risk_early_stopping
                    else 0
                ),
            )
            can_stop = epoch >= earliest_stop_epoch
            if (
                can_stop
                and early_stopping_bad_epochs
                >= self.cfg.early_stopping_patience
            ):
                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best epoch: "
                    f"{self.risk_best_epoch if risk_early_stopping else self.best_epoch}, "
                    f"best {early_stopping_monitor}: "
                    f"{self.risk_best_metric if risk_early_stopping else self.best_metric:.6f}"
                )
                break

        selection_status = {
            "status": "selected" if self.best_epoch > 0 else "no_eligible_checkpoint",
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric if self.best_epoch > 0 else None,
            "monitor": self.cfg.monitor,
            "safety_scope": self.cfg.source_on_safety_scope,
            "safety_ceiling": self.cfg.source_on_safety_ceiling,
        }
        (self.checkpoint_dir / "selection_status.json").write_text(
            json.dumps(selection_status, indent=2, allow_nan=False), encoding="utf-8"
        )
        if self.best_epoch == 0:
            raise RuntimeError(
                "No eligible checkpoint was selected. See selection_status.json "
                "and history.json; last.pt is diagnostic only, not an accepted model."
            )
        return self.history


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device | None = None,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """
    Run model on a dataloader and concatenate outputs/batches.

    This is useful for evaluation and plotting.

    Returns:
        out_all, batch_all
    """
    device = get_device(device)
    model = model.to(device)
    model.eval()

    output_store: dict[str, list[Tensor]] = {}
    batch_store: dict[str, list[Tensor]] = {}

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)

        for key, value in out.items():
            if torch.is_tensor(value):
                output_store.setdefault(key, []).append(value.detach().cpu())

        for key, value in batch.items():
            if torch.is_tensor(value):
                batch_store.setdefault(key, []).append(value.detach().cpu())

    out_all = {
        key: torch.cat(values, dim=0)
        for key, values in output_store.items()
    }

    batch_all = {
        key: torch.cat(values, dim=0)
        for key, values in batch_store.items()
    }

    return out_all, batch_all
