from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .trainer import AverageMeter, get_device, move_batch_to_device, set_seed


@dataclass
class AggregateForecastLossConfig:
    lambda_mae: float = 1.0
    lambda_rmse: float = 0.25
    lambda_peak: float = 0.20
    lambda_aux_power: float = 0.20
    lambda_aux_state: float = 0.10
    eps: float = 1e-8


class AggregateForecastLoss(nn.Module):
    """Aggregate forecast objective with historical appliance auxiliaries."""

    def __init__(
        self,
        mains_scale: float,
        appliance_scales: Tensor,
        state_pos_weight: Optional[Tensor] = None,
        config: Optional[AggregateForecastLossConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = config or AggregateForecastLossConfig()
        self.register_buffer(
            "mains_scale", torch.tensor(max(float(mains_scale), self.cfg.eps))
        )
        self.register_buffer(
            "appliance_scales",
            torch.as_tensor(appliance_scales, dtype=torch.float32).view(1, -1, 1),
        )
        if state_pos_weight is None:
            state_pos_weight = torch.ones(self.appliance_scales.size(1))
        self.register_buffer(
            "state_pos_weight",
            torch.as_tensor(state_pos_weight, dtype=torch.float32).view(1, -1, 1),
        )

    @staticmethod
    def _masked_mean(value: Tensor, mask: Optional[Tensor]) -> Tensor:
        if mask is None:
            return value.mean()
        mask = mask.to(device=value.device, dtype=value.dtype)
        return (value * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(
        self, out: dict[str, Tensor], batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        pred = out["aggregate_power"]
        target = batch["y_mains"].to(device=pred.device, dtype=pred.dtype)
        scaled_error = (pred - target) / self.mains_scale
        loss_mae = scaled_error.abs().mean()
        loss_rmse = torch.sqrt((scaled_error.square()).mean() + self.cfg.eps)
        loss_peak = F.smooth_l1_loss(
            pred.amax(dim=-1) / self.mains_scale,
            target.amax(dim=-1) / self.mains_scale,
        )

        zero = pred.new_tensor(0.0)
        loss_aux_power = zero
        loss_aux_state = zero
        if "past_power" in out and "y_hist_power" in batch:
            target_power = batch["y_hist_power"].to(pred.device, pred.dtype)
            hist_mask = batch.get("hist_mask")
            if hist_mask is not None:
                hist_mask = hist_mask.to(pred.device, pred.dtype)
            aux_error = (
                out["past_power"] - target_power
            ) / self.appliance_scales.clamp_min(self.cfg.eps)
            loss_aux_power = self._masked_mean(aux_error.abs(), hist_mask)

        if "past_state_logits" in out and "y_hist_state" in batch:
            target_state = batch["y_hist_state"].to(pred.device, pred.dtype)
            hist_mask = batch.get("hist_mask")
            if hist_mask is not None:
                hist_mask = hist_mask.to(pred.device, pred.dtype)
            bce = F.binary_cross_entropy_with_logits(
                out["past_state_logits"], target_state, reduction="none"
            )
            class_weight = torch.where(
                target_state > 0.5,
                self.state_pos_weight,
                torch.ones_like(target_state),
            )
            loss_aux_state = self._masked_mean(bce * class_weight, hist_mask)

        total = (
            self.cfg.lambda_mae * loss_mae
            + self.cfg.lambda_rmse * loss_rmse
            + self.cfg.lambda_peak * loss_peak
            + self.cfg.lambda_aux_power * loss_aux_power
            + self.cfg.lambda_aux_state * loss_aux_state
        )
        return total, {
            "loss_total": total.detach(),
            "loss_mae": loss_mae.detach(),
            "loss_rmse": loss_rmse.detach(),
            "loss_peak": loss_peak.detach(),
            "loss_aux_power": loss_aux_power.detach(),
            "loss_aux_state": loss_aux_state.detach(),
        }


def _binary_scores(pred: Tensor, target: Tensor) -> dict[str, float]:
    pred_b = pred > 0.5
    target_b = target > 0.5
    tp = (pred_b & target_b).sum().float()
    fp = (pred_b & ~target_b).sum().float()
    fn = (~pred_b & target_b).sum().float()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {
        "Precision": float(precision.item()),
        "Recall": float(recall.item()),
        "F1": float(f1.item()),
    }


def aggregate_forecast_metrics(
    pred: Tensor,
    target: Tensor,
    hist_mains: Tensor,
    past_power: Optional[Tensor] = None,
    true_hist_power: Optional[Tensor] = None,
    past_state_prob: Optional[Tensor] = None,
    true_hist_state: Optional[Tensor] = None,
    hist_mask: Optional[Tensor] = None,
    appliance_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    pred = pred.float().cpu()
    target = target.float().cpu()
    hist_mains = hist_mains.float().cpu()
    error = pred - target
    peak_error = pred.amax(-1) - target.amax(-1)
    peak_time_error = (pred.argmax(-1) - target.argmax(-1)).abs().float()
    result: dict[str, Any] = {
        "aggregate": {
            "MAE": float(error.abs().mean().item()),
            "RMSE": float(torch.sqrt(error.square().mean()).item()),
            "Bias": float(error.mean().item()),
            "PeakValueRMSE": float(torch.sqrt(peak_error.square().mean()).item()),
            "PeakTimeMAE": float(peak_time_error.mean().item()),
            "TargetMean": float(target.mean().item()),
            "PredMean": float(pred.mean().item()),
        }
    }

    persistence = hist_mains[:, -1:].expand_as(target)
    zero = torch.zeros_like(target)
    result["baseline_zero"] = {
        "MAE": float((zero - target).abs().mean().item()),
        "RMSE": float(torch.sqrt((zero - target).square().mean()).item()),
    }
    result["baseline_persistence"] = {
        "MAE": float((persistence - target).abs().mean().item()),
        "RMSE": float(torch.sqrt((persistence - target).square().mean()).item()),
    }

    if appliance_names is not None and past_power is not None and true_hist_power is not None:
        past_power = past_power.float().cpu()
        true_hist_power = true_hist_power.float().cpu()
        mask = None if hist_mask is None else hist_mask.float().cpu()
        power_metrics: dict[str, Any] = {}
        for idx, name in enumerate(appliance_names):
            app_error = past_power[:, idx] - true_hist_power[:, idx]
            if mask is not None:
                valid = mask[:, idx] > 0.5
                app_error = app_error[valid]
            power_metrics[name] = {
                "MAE": float(app_error.abs().mean().item()),
                "RMSE": float(torch.sqrt(app_error.square().mean()).item()),
            }
        result["historical_aux_power"] = power_metrics

    if (
        appliance_names is not None
        and past_state_prob is not None
        and true_hist_state is not None
    ):
        past_state_prob = past_state_prob.float().cpu()
        true_hist_state = true_hist_state.float().cpu()
        mask = None if hist_mask is None else hist_mask.float().cpu()
        state_metrics: dict[str, Any] = {}
        for idx, name in enumerate(appliance_names):
            pred_i = past_state_prob[:, idx]
            target_i = true_hist_state[:, idx]
            if mask is not None:
                valid = mask[:, idx] > 0.5
                pred_i = pred_i[valid]
                target_i = target_i[valid]
            state_metrics[name] = _binary_scores(pred_i, target_i)
        result["historical_aux_state"] = state_metrics
    return result


def flatten_nested(data: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in data.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_nested(value, name))
        elif isinstance(value, (int, float)):
            flat[name] = float(value)
    return flat


@dataclass
class AggregateTrainerConfig:
    max_epochs: int = 80
    device: Optional[str] = None
    seed: int = 42
    use_amp: bool = True
    grad_clip_norm: float = 1.0
    patience: int = 15
    log_interval: int = 50
    checkpoint_dir: str = "outputs/checkpoints"


class AggregateForecastTrainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: AggregateForecastLoss,
        optimizer: Optimizer,
        config: AggregateTrainerConfig,
        appliance_names: list[str],
        scheduler: Optional[Any] = None,
    ) -> None:
        set_seed(config.seed)
        self.cfg = config
        self.device = get_device(config.device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.appliance_names = appliance_names
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.history: list[dict[str, float]] = []

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(training)
        meter = AverageMeter()
        start = time.time()
        for step, batch in enumerate(loader, start=1):
            batch = move_batch_to_device(batch, self.device)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                out = self.model(batch)
                loss, loss_dict = self.loss_fn(out, batch)
            if training:
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip_norm
                    )
                    self.optimizer.step()
            meter.update(loss_dict, n=int(batch["x_hist"].size(0)))
            if training and self.cfg.log_interval > 0 and step % self.cfg.log_interval == 0:
                print(
                    f"step {step:05d}/{len(loader):05d} "
                    f"loss={meter.averages().get('loss_total', float('nan')):.6f}"
                )
        stats = meter.averages()
        stats["epoch_time_sec"] = time.time() - start
        return stats

    def save_checkpoint(self, epoch: int, name: str) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
                "best_epoch": self.best_epoch,
                "config": asdict(self.cfg),
            },
            self.checkpoint_dir / name,
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> list[dict[str, float]]:
        bad_epochs = 0
        print(f"Using device: {self.device}")
        for epoch in range(1, self.cfg.max_epochs + 1):
            train_stats = self._run_epoch(train_loader, training=True)
            with torch.no_grad():
                val_stats = self._run_epoch(val_loader, training=False)
            record = {"epoch": float(epoch)}
            record.update({f"train/{k}": v for k, v in train_stats.items()})
            record.update({f"val/{k}": v for k, v in val_stats.items()})
            val_loss = val_stats["loss_total"]
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss - 1e-6:
                self.best_loss = val_loss
                self.best_epoch = epoch
                bad_epochs = 0
                self.save_checkpoint(epoch, "best.pt")
            else:
                bad_epochs += 1
            self.save_checkpoint(epoch, "last.pt")
            self.history.append(record)
            with open(self.checkpoint_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
            print(
                f"[Epoch {epoch:03d}] train={train_stats['loss_total']:.6f} "
                f"val={val_loss:.6f} best={self.best_loss:.6f}@{self.best_epoch}"
            )
            if bad_epochs >= self.cfg.patience:
                print(f"Early stopping at epoch {epoch}.")
                break
        return self.history

    def load_best(self) -> None:
        checkpoint = torch.load(self.checkpoint_dir / "best.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        self.model.eval()
        stores: dict[str, list[Tensor]] = {}
        for batch in loader:
            batch = move_batch_to_device(batch, self.device)
            out = self.model(batch)
            pairs = {
                "pred": out["aggregate_power"],
                "target": batch["y_mains"],
                "hist_mains": batch["y_hist_mains"],
                "true_hist_power": batch["y_hist_power"],
                "true_hist_state": batch["y_hist_state"],
                "hist_mask": batch["hist_mask"],
            }
            for optional in ["past_power", "past_p_on"]:
                if optional in out:
                    pairs[optional] = out[optional]
            for key, value in pairs.items():
                stores.setdefault(key, []).append(value.detach().cpu())
        values = {key: torch.cat(items, dim=0) for key, items in stores.items()}
        return aggregate_forecast_metrics(
            pred=values["pred"],
            target=values["target"],
            hist_mains=values["hist_mains"],
            past_power=values.get("past_power"),
            true_hist_power=values["true_hist_power"],
            past_state_prob=values.get("past_p_on"),
            true_hist_state=values["true_hist_state"],
            hist_mask=values["hist_mask"],
            appliance_names=self.appliance_names,
        )
