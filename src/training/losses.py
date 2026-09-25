from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _to_tensor(
    values: Any,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Convert list / tuple / dict / tensor into tensor.

    If dict, keep insertion order of values.
    """
    if values is None:
        raise ValueError("values should not be None.")

    if isinstance(values, Tensor):
        return values.to(device=device, dtype=dtype)

    if isinstance(values, dict):
        values = list(values.values())

    return torch.tensor(values, device=device, dtype=dtype)


def masked_mean(
    value: Tensor,
    mask: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """
    Compute masked mean.

    value:
        any shape

    mask:
        same shape or broadcastable to value
    """
    if mask is None:
        return value.mean()

    mask = mask.to(device=value.device, dtype=value.dtype)

    value = value * mask
    denom = mask.sum().clamp_min(eps)

    return value.sum() / denom


def apply_appliance_scale(
    x: Tensor,
    appliance_scales: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """
    Normalize appliance power by appliance-specific scales.

    x:
        [B, A, H]

    appliance_scales:
        [A]

    return:
        [B, A, H]
    """
    if x.dim() != 3:
        raise ValueError(f"Expected x shape [B, A, H], got {x.shape}")

    scales = appliance_scales.to(device=x.device, dtype=x.dtype)
    scales = scales.view(1, -1, 1).clamp_min(eps)

    return x / scales


def masked_l1_loss(
    pred: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    return masked_mean(torch.abs(pred - target), mask=mask)


def masked_mse_loss(
    pred: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    return masked_mean((pred - target) ** 2, mask=mask)


def binary_focal_loss_with_logits(
    logits: Tensor,
    targets: Tensor,
    mask: Optional[Tensor] = None,
    gamma: float = 2.0,
    alpha: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """
    Focal BCE loss for imbalanced state/event labels.

    logits:
        [B, A, H]

    targets:
        [B, A, H], values in {0, 1}

    alpha:
        None or [A].
        If given, positive samples are weighted by alpha_i,
        negative samples are weighted by 1 - alpha_i.
    """
    targets = targets.to(device=logits.device, dtype=logits.dtype)

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    prob = torch.sigmoid(logits)
    pt = prob * targets + (1.0 - prob) * (1.0 - targets)

    focal_weight = (1.0 - pt).clamp_min(eps) ** gamma

    if alpha is not None:
        alpha = alpha.to(device=logits.device, dtype=logits.dtype)
        alpha = alpha.view(1, -1, 1)

        alpha_weight = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        focal_weight = focal_weight * alpha_weight

    loss = focal_weight * bce
    return masked_mean(loss, mask=mask)


def binary_bce_loss_with_logits(
    logits: Tensor,
    targets: Tensor,
    mask: Optional[Tensor] = None,
    pos_weight: Optional[Tensor] = None,
) -> Tensor:
    """
    Standard BCEWithLogitsLoss with optional per-appliance pos_weight.

    pos_weight:
        None or [A].
    """
    targets = targets.to(device=logits.device, dtype=logits.dtype)

    if pos_weight is not None:
        pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)
        pos_weight = pos_weight.view(1, -1, 1)

        # Manual BCE with positive weight, because PyTorch BCEWithLogits
        # expects pos_weight shape to match class dimension but broadcasting
        # can be inconvenient for [B, A, H].
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        weight = torch.where(targets > 0.5, pos_weight, torch.ones_like(targets))
        loss = loss * weight
    else:
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

    return masked_mean(loss, mask=mask)


def appliance_latent_orthogonality_loss(
    z_app: Tensor,
    mask_diagonal: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """
    Penalize collapse among appliance latent vectors.

    z_app:
        [B, A, D]

    return:
        scalar loss

    The loss encourages different appliance representations to be less similar.
    """
    if z_app.dim() != 3:
        raise ValueError(f"Expected z_app shape [B, A, D], got {z_app.shape}")

    batch_size, num_appliances, _ = z_app.shape

    z_norm = F.normalize(z_app, p=2, dim=-1, eps=eps)
    sim = torch.matmul(z_norm, z_norm.transpose(1, 2))

    if mask_diagonal:
        eye = torch.eye(
            num_appliances,
            device=z_app.device,
            dtype=z_app.dtype,
        ).unsqueeze(0)

        off_diag = 1.0 - eye
        sim = sim * off_diag
        denom = off_diag.sum() * batch_size
    else:
        denom = torch.tensor(
            batch_size * num_appliances * num_appliances,
            device=z_app.device,
            dtype=z_app.dtype,
        )

    return (sim ** 2).sum() / denom.clamp_min(eps)


@dataclass
class PISALossConfig:
    """
    Loss weights and settings.
    """

    lambda_power: float = 1.0
    lambda_bridge: float = 0.3
    lambda_state: float = 0.25
    lambda_event: float = 0.10
    lambda_start: float = 0.20
    lambda_stop: float = 0.20
    lambda_bucket_start: float = 0.0
    lambda_bucket_stop: float = 0.0
    lambda_event_offset: float = 0.0
    lambda_window_start: float = 0.0
    lambda_conditional_bucket: float = 0.0
    lambda_conditional_offset: float = 0.0
    lambda_conditional_power: float = 0.0
    lambda_pulse_duration: float = 0.0
    lambda_pulse_amplitude: float = 0.0
    lambda_agg: float = 0.20
    lambda_ghost: float = 0.05
    lambda_peak: float = 0.10
    lambda_orth: float = 0.005
    lambda_amp_on: float = 0.0
    # Auxiliary supervision for the enhanced direct future decoder. This
    # prevents its gradient from being attenuated by a base-heavy blend while
    # keeping the deployed output safely anchored to the selected base model.
    lambda_direct_power: float = 0.0

    lambda_recon_power: float = 0.20
    lambda_recon_state: float = 0.10
    active_power_weight: float = 3.0
    active_bridge_weight: float = 2.0
    # Compute ON and OFF errors independently for every appliance and then
    # combine the two states with an explicit fraction before averaging all
    # appliances. This removes prevalence as an implicit loss weight, which
    # otherwise lets sparse appliances minimize MAE by predicting zero.
    balanced_on_off_power_loss: bool = False
    balanced_on_fraction: float = 0.5
    # Optional ON-state weighting for historical reconstruction fine-tuning.
    # Zero preserves the original unweighted auxiliary reconstruction loss.
    active_recon_weight: float = 0.0

    # Auxiliary future-risk objectives are deliberately introduced after the
    # core NILM task has formed a useful representation.  This prevents sparse
    # event/bucket losses (which can initially be close to random guessing)
    # from overwhelming power reconstruction and state estimation.
    risk_warmup_epochs: int = 0
    risk_ramp_epochs: int = 0

    # Base-stage power curriculum.  The core forecast must become calibrated
    # before bridge/state/peak regularizers can safely influence its route.
    # When both durations are zero (the default), target weights below are
    # used immediately and legacy training behaviour is unchanged.
    base_aux_warmup_epochs: int = 0
    base_aux_ramp_epochs: int = 0
    base_warmup_lambda_bridge: float = 0.0
    base_warmup_lambda_agg: float = 0.0
    base_warmup_lambda_state: float = 0.0
    base_warmup_lambda_ghost: float = 0.0
    base_warmup_lambda_peak: float = 0.0

    power_loss_type: str = "l1"
    bridge_loss_type: str = "l1"

    state_loss_type: str = "focal"
    event_loss_type: str = "focal"

    focal_gamma: float = 2.0

    peak_quantile: float = 0.80

    # Since only the selected appliances are modeled, their sum is only a subset
    # of the household aggregate.
    # Therefore, aggregate loss should mainly penalize over-estimation beyond grid.
    aggregate_mode: str = "upper_bound"

    eps: float = 1e-8


class PISALoss(nn.Module):
    """
    Multi-task loss for PISA.

    Expected model outputs:
        out["y_power"]          [B, A, H]
        out["bridge_power"]     [B, A, H]
        out["state_logits"]     [B, A, H]
        out["event_logits"]     [B, A, H]
        out["z_app"]            [B, A, D]

    Expected batch:
        batch["y_power"]        [B, A, H]
        batch["y_power_scaled"] [B, A, H], optional
        batch["y_state"]        [B, A, H]
        batch["y_event"]        [B, A, H]
        batch["y_mains"]        [B, H], optional
        batch["target_mask"]    [B, A, H], optional

    Important:
        Model outputs are assumed to be in original power unit, such as kW.
        The balanced power losses divide predictions by appliance-specific scales.
    """

    def __init__(
        self,
        appliance_scales: Tensor | list[float] | dict[str, float],
        config: Optional[PISALossConfig | dict[str, Any]] = None,
        state_alpha: Optional[Tensor | list[float] | dict[str, float]] = None,
        event_alpha: Optional[Tensor | list[float] | dict[str, float]] = None,
        stop_alpha: Optional[Tensor | list[float] | dict[str, float]] = None,
        state_pos_weight: Optional[Tensor | list[float] | dict[str, float]] = None,
        event_pos_weight: Optional[Tensor | list[float] | dict[str, float]] = None,
    ):
        super().__init__()

        if config is None:
            self.cfg = PISALossConfig()
        elif isinstance(config, dict):
            self.cfg = PISALossConfig(**config)
        elif isinstance(config, PISALossConfig):
            self.cfg = config
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        if not 0.0 <= float(self.cfg.balanced_on_fraction) <= 1.0:
            raise ValueError("balanced_on_fraction should be in [0, 1].")

        self.register_buffer(
            "appliance_scales",
            torch.as_tensor(
                list(appliance_scales.values())
                if isinstance(appliance_scales, dict)
                else appliance_scales,
                dtype=torch.float32,
            ),
            persistent=True,
        )

        self._register_optional_buffer("state_alpha", state_alpha)
        self._register_optional_buffer("event_alpha", event_alpha)
        self._register_optional_buffer(
            "stop_alpha",
            event_alpha if stop_alpha is None else stop_alpha,
        )
        self._register_optional_buffer("state_pos_weight", state_pos_weight)
        self._register_optional_buffer("event_pos_weight", event_pos_weight)
        # Keep direct loss-function use backward compatible: without a trainer
        # calling set_epoch(), no configured auxiliary objective is suppressed.
        self._current_epoch = 1

    def set_epoch(self, epoch: int) -> None:
        """Set the one-based epoch used by loss schedules."""
        self._current_epoch = max(0, int(epoch))

    def risk_loss_scale(self) -> float:
        """Return the current multiplier for sparse future-risk objectives.

        Core NILM losses (power, state, reconstruction, aggregate and ghost
        penalties) always keep their configured weight.  Event, bucket,
        conditional and pulse losses are held at zero during warm-up and then
        linearly ramp to their configured weights.
        """
        configured_risk_weights = (
            self.cfg.lambda_event,
            self.cfg.lambda_start,
            self.cfg.lambda_stop,
            self.cfg.lambda_bucket_start,
            self.cfg.lambda_bucket_stop,
            self.cfg.lambda_event_offset,
            self.cfg.lambda_window_start,
            self.cfg.lambda_conditional_bucket,
            self.cfg.lambda_conditional_offset,
            self.cfg.lambda_conditional_power,
            self.cfg.lambda_pulse_duration,
            self.cfg.lambda_pulse_amplitude,
        )
        if not any(float(weight) != 0.0 for weight in configured_risk_weights):
            return 0.0

        warmup = max(0, int(self.cfg.risk_warmup_epochs))
        ramp = max(0, int(self.cfg.risk_ramp_epochs))

        if self._current_epoch <= warmup:
            return 0.0
        if ramp == 0:
            return 1.0
        return min(1.0, (self._current_epoch - warmup) / float(ramp))

    def base_auxiliary_weights(self) -> dict[str, float]:
        """Return the active base-stage auxiliary weights for this epoch.

        Power remains the primary optimization signal.  Aggregate consistency
        stays active throughout unless an explicit base-stage schedule is
        configured; this permits a short target-domain calibration warm-up
        without changing the legacy default behaviour.
        """
        target = {
            "bridge": float(self.cfg.lambda_bridge),
            "agg": float(self.cfg.lambda_agg),
            "state": float(self.cfg.lambda_state),
            "ghost": float(self.cfg.lambda_ghost),
            "peak": float(self.cfg.lambda_peak),
        }
        warmup = max(0, int(self.cfg.base_aux_warmup_epochs))
        ramp = max(0, int(self.cfg.base_aux_ramp_epochs))
        if warmup == 0 and ramp == 0:
            return target

        initial = {
            "bridge": float(self.cfg.base_warmup_lambda_bridge),
            "agg": float(self.cfg.base_warmup_lambda_agg),
            "state": float(self.cfg.base_warmup_lambda_state),
            "ghost": float(self.cfg.base_warmup_lambda_ghost),
            "peak": float(self.cfg.base_warmup_lambda_peak),
        }
        if self._current_epoch <= warmup:
            return initial
        if ramp == 0:
            return target

        progress = min(
            1.0,
            max(0.0, (self._current_epoch - warmup) / float(ramp)),
        )
        return {
            name: initial[name] + progress * (target[name] - initial[name])
            for name in target
        }

    def base_auxiliary_schedule_progress(self) -> float:
        """Return 0 during base warm-up and 1 after the auxiliary ramp."""
        warmup = max(0, int(self.cfg.base_aux_warmup_epochs))
        ramp = max(0, int(self.cfg.base_aux_ramp_epochs))
        if warmup == 0 and ramp == 0:
            return 1.0
        if self._current_epoch <= warmup:
            return 0.0
        if ramp == 0:
            return 1.0
        return min(1.0, (self._current_epoch - warmup) / float(ramp))

    def _register_optional_buffer(self, name: str, values: Any) -> None:
        if values is None:
            self.register_buffer(name, None, persistent=False)
            return

        if isinstance(values, dict):
            values = list(values.values())

        self.register_buffer(
            name,
            torch.as_tensor(values, dtype=torch.float32),
            persistent=True,
        )

    def _get_target_power_scaled(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Prefer batch["y_power_scaled"] if available.
        Otherwise compute y_power / appliance_scales.
        """
        if "y_power_scaled" in batch:
            return batch["y_power_scaled"]

        if "y_power" not in batch:
            raise KeyError("batch should contain 'y_power' or 'y_power_scaled'.")

        return apply_appliance_scale(
            batch["y_power"],
            self.appliance_scales,
            eps=self.cfg.eps,
        )

    def _power_loss(
        self,
        pred_power: Tensor,
        target_power_scaled: Tensor,
        mask: Optional[Tensor],
        loss_type: str,
        y_state: Optional[Tensor] = None,
        active_weight: float = 0.0,
        balanced_on_off: bool = False,
        per_appliance_equal: bool = False,
    ) -> Tensor:
        """
        Balanced power loss with optional state-aware active weighting.

        pred_power:
            [B, A, H], original power unit

        target_power_scaled:
            [B, A, H], appliance-scaled target

        y_state:
            [B, A, H], 1 means appliance ON

        active_weight:
            If > 0, ON samples receive larger weight:
                weight = 1 + active_weight * y_state

        This prevents sparse appliances from collapsing to all-zero predictions.
        """
        pred_scaled = apply_appliance_scale(
            pred_power.float(),
            self.appliance_scales,
            eps=self.cfg.eps,
        )
        target_power_scaled = target_power_scaled.float()

        if loss_type == "l1":
            loss = torch.abs(pred_scaled - target_power_scaled)
        elif loss_type == "mse":
            loss = (pred_scaled - target_power_scaled) ** 2
        else:
            raise ValueError(f"Unsupported power loss type: {loss_type}")

        effective_mask = mask

        if per_appliance_equal:
            valid = (
                torch.ones_like(loss)
                if effective_mask is None
                else effective_mask.to(device=loss.device, dtype=loss.dtype)
            )
            count = valid.sum(dim=(0, 2))
            per_appliance = (
                (loss * valid).sum(dim=(0, 2)) / count.clamp_min(1.0)
            )
            present = count > 0
            if not torch.any(present):
                return loss.sum() * 0.0
            return per_appliance[present].mean()

        if balanced_on_off:
            if y_state is None:
                raise ValueError(
                    "balanced_on_off power loss requires y_state."
                )
            valid = (
                torch.ones_like(loss)
                if effective_mask is None
                else effective_mask.to(device=loss.device, dtype=loss.dtype)
            )
            state = y_state.to(device=loss.device, dtype=loss.dtype).clamp(0.0, 1.0)
            on_mask = valid * state
            off_mask = valid * (1.0 - state)
            reduce_dims = (0, 2)
            on_count = on_mask.sum(dim=reduce_dims)
            off_count = off_mask.sum(dim=reduce_dims)
            on_mean = (loss * on_mask).sum(dim=reduce_dims) / on_count.clamp_min(1.0)
            off_mean = (loss * off_mask).sum(dim=reduce_dims) / off_count.clamp_min(1.0)
            on_present = on_count > 0
            off_present = off_count > 0
            both_present = on_present & off_present
            on_fraction = float(self.cfg.balanced_on_fraction)
            weighted_both = (
                on_fraction * on_mean + (1.0 - on_fraction) * off_mean
            )
            # If a mini-batch contains only one state for an appliance, use
            # that available state rather than silently scaling it down.
            per_appliance = torch.where(
                both_present,
                weighted_both,
                torch.where(on_present, on_mean, off_mean),
            )
            present = on_present | off_present
            if not torch.any(present):
                return loss.sum() * 0.0
            return per_appliance[present].mean()

        if y_state is not None and active_weight > 0:
            state_weight = 1.0 + active_weight * y_state.to(
                device=loss.device,
                dtype=loss.dtype,
            ).clamp(0.0, 1.0)

            if effective_mask is None:
                effective_mask = state_weight
            else:
                effective_mask = effective_mask.to(
                    device=loss.device,
                    dtype=loss.dtype,
                ) * state_weight

        return masked_mean(loss, mask=effective_mask)
    
    def _classification_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        mask: Optional[Tensor],
        loss_type: str,
        alpha: Optional[Tensor],
        pos_weight: Optional[Tensor],
    ) -> Tensor:
        if loss_type == "focal":
            return binary_focal_loss_with_logits(
                logits=logits,
                targets=targets,
                mask=mask,
                gamma=self.cfg.focal_gamma,
                alpha=alpha,
            )

        if loss_type == "bce":
            return binary_bce_loss_with_logits(
                logits=logits,
                targets=targets,
                mask=mask,
                pos_weight=pos_weight,
            )

        raise ValueError(f"Unsupported classification loss type: {loss_type}")

    def _aggregate_consistency_loss(
        self,
        pred_power: Tensor,
        batch: dict[str, Tensor],
    ) -> Tensor:
        """
        Aggregate consistency loss.

        Selected appliances may account for only part of household demand, so
        their predicted sum is not constrained to equal aggregate demand.

        Default mode:
            upper_bound:
                penalize sum(pred_appliance) > grid

        Optional mode:
            match_target_sum:
                match sum(pred_appliance) with sum(true_selected_appliance)
        """
        mode = self.cfg.aggregate_mode

        if mode == "none":
            return pred_power.new_tensor(0.0)

        pred_sum = pred_power.sum(dim=1)

        if mode == "upper_bound":
            if "y_mains" not in batch:
                return pred_power.new_tensor(0.0)

            y_mains = batch["y_mains"].to(
                device=pred_power.device,
                dtype=pred_power.dtype,
            )

            if y_mains.dim() != 2:
                raise ValueError(f"Expected y_mains shape [B, H], got {y_mains.shape}")

            excess = F.relu(pred_sum - y_mains)
            return excess.mean()

        if mode == "match_target_sum":
            if "y_power" not in batch:
                return pred_power.new_tensor(0.0)

            target_sum = batch["y_power"].to(
                device=pred_power.device,
                dtype=pred_power.dtype,
            ).sum(dim=1)

            return F.l1_loss(pred_sum, target_sum)

        raise ValueError(f"Unsupported aggregate_mode: {mode}")

    def _ghost_loss(
        self,
        pred_power: Tensor,
        y_state: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        """
        Penalize predicted power when target appliance state is OFF.

        pred_power:
            [B, A, H], original unit

        y_state:
            [B, A, H], 1 means ON, 0 means OFF
        """
        pred_scaled = apply_appliance_scale(
            pred_power,
            self.appliance_scales,
            eps=self.cfg.eps,
        )

        off_mask = (1.0 - y_state.to(pred_scaled.dtype)).clamp(0.0, 1.0)

        if mask is not None:
            off_mask = off_mask * mask.to(pred_scaled.dtype)

        return masked_mean(pred_scaled, mask=off_mask)

    def _peak_loss(
        self,
        pred_power: Tensor,
        target_power_scaled: Tensor,
        mask: Optional[Tensor],
        y_state: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Peak loss only on true ON samples.

        This avoids treating tiny standby/noise values such as 0.001 kW
        as peak samples for sparse appliances.
        """
        pred_scaled = apply_appliance_scale(
            pred_power,
            self.appliance_scales,
            eps=self.cfg.eps,
        )

        with torch.no_grad():
            if y_state is None:
                active_mask = target_power_scaled > 0.0
            else:
                active_mask = y_state.to(
                    device=target_power_scaled.device,
                    dtype=torch.bool,
                )

            if mask is not None:
                active_mask = active_mask & mask.to(
                    device=target_power_scaled.device,
                    dtype=torch.bool,
                )

            peak_mask = torch.zeros_like(target_power_scaled, dtype=torch.bool)

            # Compute peak threshold per appliance on active samples only.
            num_appliances = target_power_scaled.shape[1]

            for app_idx in range(num_appliances):
                target_i = target_power_scaled[:, app_idx, :]
                active_i = active_mask[:, app_idx, :]

                values = target_i[active_i]

                if values.numel() == 0:
                    continue

                q = torch.quantile(values, self.cfg.peak_quantile)
                peak_i = active_i & (target_i >= q)

                peak_mask[:, app_idx, :] = peak_i

        if peak_mask.sum() < 1:
            return pred_power.new_tensor(0.0)

        return masked_l1_loss(
            pred_scaled,
            target_power_scaled,
            mask=peak_mask.to(pred_scaled.dtype),
        )

    @staticmethod
    def _event_bucket_targets(
        event: Tensor,
        mask: Optional[Tensor],
        bucket_size: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Convert minute event labels into occurrence and offset targets."""
        if event.dim() != 3:
            raise ValueError(f"event should be [B, A, H], got {event.shape}.")

        batch_size, num_appliances, horizon = event.shape
        bucket_size = max(1, int(bucket_size))
        num_buckets = (horizon + bucket_size - 1) // bucket_size
        padded_horizon = num_buckets * bucket_size
        pad = padded_horizon - horizon

        event_padded = F.pad(event, (0, pad)) if pad > 0 else event
        event_bucket = event_padded.view(
            batch_size,
            num_appliances,
            num_buckets,
            bucket_size,
        )

        if mask is None:
            valid_bucket = torch.ones_like(event_bucket[..., 0])
        else:
            mask_padded = F.pad(mask, (0, pad)) if pad > 0 else mask
            mask_bucket = mask_padded.view(
                batch_size,
                num_appliances,
                num_buckets,
                bucket_size,
            )
            valid_bucket = (mask_bucket.sum(dim=-1) > 0).to(event.dtype)
            event_bucket = event_bucket * mask_bucket

        event_count = event_bucket.sum(dim=-1)
        occurrence = (event_count > 0).to(event.dtype)
        offset = event_bucket.argmax(dim=-1).long()
        # Offset supervision is unambiguous only when the bucket has one event.
        offset_valid = (event_count == 1).to(event.dtype) * valid_bucket
        return occurrence, offset, offset_valid

    @staticmethod
    def _offset_loss(
        logits: Tensor,
        targets: Tensor,
        valid: Tensor,
    ) -> Tensor:
        if logits.dim() != 4:
            raise ValueError(f"offset logits should be [B, A, N, K], got {logits.shape}.")
        if logits.shape[:-1] != targets.shape or targets.shape != valid.shape:
            raise ValueError("offset targets and validity mask must match [B, A, N].")

        valid_flat = valid.reshape(-1) > 0.5
        if not torch.any(valid_flat):
            return logits.new_tensor(0.0)

        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1))[valid_flat],
            targets.reshape(-1)[valid_flat],
        )

    @staticmethod
    def _conditional_bucket_loss(
        logits: Tensor,
        occurrence: Tensor,
    ) -> Tensor:
        """Categorical bucket loss, evaluated only when a start exists."""
        if logits.shape != occurrence.shape or logits.dim() != 3:
            raise ValueError(
                "conditional bucket logits and targets should be [B, A, N]."
            )
        positive = occurrence.sum(dim=-1) > 0
        if not torch.any(positive):
            return logits.new_tensor(0.0)
        target_bucket = occurrence.argmax(dim=-1)
        return F.cross_entropy(logits[positive], target_bucket[positive])

    def forward(
        self,
        out: dict[str, Tensor],
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Return:
            total_loss:
                scalar tensor

            loss_dict:
                dictionary of detached scalar tensors for logging
        """
        required_out = [
            "y_power",
            "bridge_power",
            "state_logits",
            "event_logits",
            "z_app",
        ]

        for key in required_out:
            if key not in out:
                raise KeyError(f"Model output missing key: {key}")

        for key in ["y_state", "y_event"]:
            if key not in batch:
                raise KeyError(f"Batch missing key: {key}")

        pred_power = out["y_power"]
        optimization_power = out.get("expected_power", pred_power)
        bridge_power = out["bridge_power"]

        y_state = batch["y_state"].to(
            device=pred_power.device,
            dtype=pred_power.dtype,
        )
        y_event = batch["y_event"].to(
            device=pred_power.device,
            dtype=pred_power.dtype,
        )
        y_start = batch.get("y_start", y_event).to(
            device=pred_power.device,
            dtype=pred_power.dtype,
        )
        y_stop = batch.get("y_stop", y_event).to(
            device=pred_power.device,
            dtype=pred_power.dtype,
        )

        target_power_scaled = self._get_target_power_scaled(batch).to(
            device=pred_power.device,
            dtype=torch.float32,
        )

        target_mask = batch.get("target_mask", None)
        if target_mask is not None:
            target_mask = target_mask.to(
                device=pred_power.device,
                dtype=pred_power.dtype,
            )

        loss_power = self._power_loss(
            pred_power=optimization_power,
            target_power_scaled=target_power_scaled,
            mask=target_mask,
            loss_type=self.cfg.power_loss_type,
            y_state=y_state,
            active_weight=self.cfg.active_power_weight,
            balanced_on_off=self.cfg.balanced_on_off_power_loss,
        )

        loss_direct_power = pred_power.new_tensor(0.0)
        if "future_tcn_direct_power" in out:
            loss_direct_power = self._power_loss(
                pred_power=out["future_tcn_direct_power"],
                target_power_scaled=target_power_scaled,
                mask=target_mask,
                loss_type=self.cfg.power_loss_type,
                y_state=y_state,
                active_weight=self.cfg.active_power_weight,
                balanced_on_off=self.cfg.balanced_on_off_power_loss,
            )

        loss_bridge = self._power_loss(
            pred_power=bridge_power,
            target_power_scaled=target_power_scaled,
            mask=target_mask,
            loss_type=self.cfg.bridge_loss_type,
            y_state=y_state,
            active_weight=self.cfg.active_bridge_weight,
        )

        loss_state = self._classification_loss(
            logits=out["state_logits"],
            targets=y_state,
            mask=target_mask,
            loss_type=self.cfg.state_loss_type,
            alpha=self.state_alpha,
            pos_weight=self.state_pos_weight,
        )

        loss_event = self._classification_loss(
            logits=out["event_logits"],
            targets=y_event,
            mask=target_mask,
            loss_type=self.cfg.event_loss_type,
            alpha=self.event_alpha,
            pos_weight=self.event_pos_weight,
        )

        loss_start = pred_power.new_tensor(0.0)
        if "start_logits" in out:
            loss_start = self._classification_loss(
                logits=out["start_logits"],
                targets=y_start,
                mask=target_mask,
                loss_type=self.cfg.event_loss_type,
                alpha=self.event_alpha,
                pos_weight=self.event_pos_weight,
            )

        loss_stop = pred_power.new_tensor(0.0)
        if "stop_logits" in out:
            loss_stop = self._classification_loss(
                logits=out["stop_logits"],
                targets=y_stop,
                mask=target_mask,
                loss_type=self.cfg.event_loss_type,
                alpha=self.stop_alpha,
                pos_weight=self.event_pos_weight,
            )

        loss_bucket_start = pred_power.new_tensor(0.0)
        loss_bucket_stop = pred_power.new_tensor(0.0)
        loss_event_offset = pred_power.new_tensor(0.0)
        loss_window_start = pred_power.new_tensor(0.0)
        loss_conditional_bucket = pred_power.new_tensor(0.0)
        loss_conditional_offset = pred_power.new_tensor(0.0)
        loss_conditional_power = pred_power.new_tensor(0.0)
        loss_pulse_duration = pred_power.new_tensor(0.0)
        loss_pulse_amplitude = pred_power.new_tensor(0.0)
        if (
            "start_bucket_logits" in out
            and "stop_bucket_logits" in out
            and "start_offset_logits" in out
            and "stop_offset_logits" in out
        ):
            bucket_size = out["start_offset_logits"].size(-1)
            start_bucket_target, start_offset_target, start_offset_valid = (
                self._event_bucket_targets(y_start, target_mask, bucket_size)
            )
            stop_bucket_target, stop_offset_target, stop_offset_valid = (
                self._event_bucket_targets(y_stop, target_mask, bucket_size)
            )
            if target_mask is None:
                bucket_mask = torch.ones_like(start_bucket_target)
            else:
                horizon = target_mask.size(-1)
                num_buckets = (horizon + bucket_size - 1) // bucket_size
                pad = num_buckets * bucket_size - horizon
                mask_padded = F.pad(target_mask, (0, pad)) if pad > 0 else target_mask
                bucket_mask = (
                    mask_padded.view(
                        target_mask.size(0),
                        target_mask.size(1),
                        num_buckets,
                        bucket_size,
                    ).sum(dim=-1) > 0
                ).to(pred_power.dtype)

            loss_bucket_start = self._classification_loss(
                logits=out["start_bucket_logits"],
                targets=start_bucket_target,
                mask=bucket_mask,
                loss_type=self.cfg.event_loss_type,
                alpha=self.event_alpha,
                pos_weight=self.event_pos_weight,
            )
            loss_bucket_stop = self._classification_loss(
                logits=out["stop_bucket_logits"],
                targets=stop_bucket_target,
                mask=bucket_mask,
                loss_type=self.cfg.event_loss_type,
                alpha=self.stop_alpha,
                pos_weight=self.event_pos_weight,
            )
            loss_event_offset = 0.5 * (
                self._offset_loss(
                    out["start_offset_logits"],
                    start_offset_target,
                    start_offset_valid,
                )
                + self._offset_loss(
                    out["stop_offset_logits"],
                    stop_offset_target,
                    stop_offset_valid,
                )
            )

            window_start_target = (start_bucket_target.sum(dim=-1) > 0).to(
                pred_power.dtype
            )
            if "window_start_logits" in out:
                window_mask = (bucket_mask.sum(dim=-1) > 0).to(pred_power.dtype)
                loss_window_start = self._classification_loss(
                    logits=out["window_start_logits"].unsqueeze(-1),
                    targets=window_start_target.unsqueeze(-1),
                    mask=window_mask.unsqueeze(-1),
                    loss_type="bce",
                    alpha=None,
                    pos_weight=None,
                )
            loss_conditional_bucket = self._conditional_bucket_loss(
                out["start_bucket_logits"],
                start_bucket_target,
            )
            loss_conditional_offset = self._offset_loss(
                out["start_offset_logits"],
                start_offset_target,
                start_offset_valid,
            )

            if "conditional_power" in out:
                conditional_mask = window_start_target.unsqueeze(-1) * (
                    0.2 + 0.8 * y_state
                )
                if target_mask is not None:
                    conditional_mask = conditional_mask * target_mask
                loss_conditional_power = self._power_loss(
                    pred_power=out["conditional_power"],
                    target_power_scaled=target_power_scaled,
                    mask=conditional_mask,
                    loss_type=self.cfg.power_loss_type,
                    y_state=None,
                    active_weight=0.0,
                )

            if (
                "pulse_duration_logits" in out
                and "pulse_amplitude_power" in out
                and "pulse_appliance_mask" in out
            ):
                pulse_mask = out["pulse_appliance_mask"].to(
                    device=pred_power.device, dtype=pred_power.dtype
                ).squeeze(-1)
                has_start = (y_start.sum(dim=-1) > 0).to(pred_power.dtype)
                # Do not supervise duration when the event is still ON at the
                # right edge; its true duration is censored by the horizon.
                right_censored = (y_state[..., -1] > 0.5).to(pred_power.dtype)
                valid_duration = pulse_mask * has_start * (1.0 - right_censored)
                valid_pulse = pulse_mask * has_start

                first_start = y_start.argmax(dim=-1)
                minute_ids = torch.arange(
                    y_state.size(-1), device=pred_power.device
                ).view(1, 1, -1)
                after_start = minute_ids >= first_start.unsqueeze(-1)
                off_after_start = (1.0 - y_state) * after_start.to(y_state.dtype)
                before_first_off = off_after_start.cumsum(dim=-1) < 1.0
                event_on = after_start & before_first_off & (y_state > 0.5)
                duration_target = event_on.sum(dim=-1).clamp(
                    min=1, max=y_state.size(-1)
                ).long() - 1

                duration_ce = F.cross_entropy(
                    out["pulse_duration_logits"].reshape(-1, y_state.size(-1)),
                    duration_target.reshape(-1),
                    reduction="none",
                ).view_as(valid_pulse)
                loss_pulse_duration = (
                    duration_ce * valid_duration
                ).sum() / valid_duration.sum().clamp_min(1.0)

                target_power = batch["y_power"].to(
                    device=pred_power.device, dtype=pred_power.dtype
                )
                event_power = target_power.masked_fill(~event_on, 0.0)
                amplitude_target = event_power.amax(dim=-1)
                amplitude_error = F.smooth_l1_loss(
                    out["pulse_amplitude_power"],
                    amplitude_target,
                    reduction="none",
                )
                loss_pulse_amplitude = (
                    amplitude_error * valid_pulse
                ).sum() / valid_pulse.sum().clamp_min(1.0)

        loss_agg = self._aggregate_consistency_loss(
            pred_power=optimization_power,
            batch=batch,
        )

        loss_ghost = self._ghost_loss(
            pred_power=optimization_power,
            y_state=y_state,
            mask=target_mask,
        )

        peak_power = optimization_power
        if (
            "conditional_power" in out
            and "structured_appliance_mask" in out
        ):
            peak_power = torch.where(
                out["structured_appliance_mask"].bool(),
                out["conditional_power"],
                optimization_power,
            )
        loss_peak = self._peak_loss(
            pred_power=peak_power,
            target_power_scaled=target_power_scaled,
            mask=target_mask,
            y_state=y_state,
        )

        loss_amp_on = pred_power.new_tensor(0.0)
        if "amplitude_power" in out:
            amp_mask = y_state
            if target_mask is not None:
                amp_mask = amp_mask * target_mask

            loss_amp_on = self._power_loss(
                pred_power=out["amplitude_power"],
                target_power_scaled=target_power_scaled,
                mask=amp_mask,
                loss_type=self.cfg.power_loss_type,
                y_state=None,
                active_weight=0.0,
                per_appliance_equal=True,
            )

        loss_orth = appliance_latent_orthogonality_loss(out["z_app"])

        loss_recon_power = pred_power.new_tensor(0.0)
        loss_recon_state = pred_power.new_tensor(0.0)
        history_macro_mae = pred_power.new_tensor(float("nan"))
        
        if (
            "past_power" in out
            and "past_state_logits" in out
            and "y_hist_power_scaled" in batch
            and "y_hist_state" in batch
        
           ):
            # A deployment-only residual refiner may improve histories without
            # entering the existing PISA future/risk route.  When absent, these
            # fallbacks reproduce the original auxiliary reconstruction loss.
            # Supervise the exact history route consumed by the future TCN.
            # This matters when a checkpoint contains a refiner but the TCN is
            # explicitly configured to consume the base reconstruction.
            past_power = out.get(
                "future_tcn_input_power",
                out.get("past_refined_power", out["past_power"]),
            )
            past_state_logits = out.get(
                "future_tcn_input_state_logits",
                out.get("past_refined_state_logits", out["past_state_logits"]),
            )

            y_hist_power_scaled = batch["y_hist_power_scaled"].to(
               device=past_power.device,
               dtype=past_power.dtype,
            )

            y_hist_state = batch["y_hist_state"].to(
               device=past_power.device,
               dtype=past_power.dtype,
            )

            hist_mask = batch.get("hist_mask", None)
            if hist_mask is not None:
                hist_mask = hist_mask.to(
                   device=past_power.device,
                   dtype=past_power.dtype,
                )

            loss_recon_power = self._power_loss(
                pred_power=past_power,
                target_power_scaled=y_hist_power_scaled,
                mask=hist_mask,
                loss_type="l1",
                y_state=y_hist_state,
                active_weight=self.cfg.active_recon_weight,
            )

            loss_recon_state = self._classification_loss(
                logits=past_state_logits,
                targets=y_hist_state,
                mask=hist_mask,
                loss_type=self.cfg.state_loss_type,
                alpha=self.state_alpha,
                pos_weight=self.state_pos_weight,
            )

            if "y_hist_power" in batch:
                y_hist_power = batch["y_hist_power"].to(
                    device=past_power.device,
                    dtype=past_power.dtype,
                )
                valid = (
                    torch.ones_like(y_hist_power)
                    if hist_mask is None
                    else hist_mask
                )
                per_appliance_count = valid.sum(dim=(0, 2)).clamp_min(1.0)
                per_appliance_mae = (
                    (past_power - y_hist_power).abs() * valid
                ).sum(dim=(0, 2)) / per_appliance_count
                history_macro_mae = per_appliance_mae.mean()


        risk_scale = self.risk_loss_scale()
        base_aux_weights = self.base_auxiliary_weights()
        total_loss = (
            self.cfg.lambda_power * loss_power
            + self.cfg.lambda_direct_power * loss_direct_power
            + base_aux_weights["bridge"] * loss_bridge
            + base_aux_weights["state"] * loss_state
            + risk_scale * self.cfg.lambda_event * loss_event
            + risk_scale * self.cfg.lambda_start * loss_start
            + risk_scale * self.cfg.lambda_stop * loss_stop
            + risk_scale * self.cfg.lambda_bucket_start * loss_bucket_start
            + risk_scale * self.cfg.lambda_bucket_stop * loss_bucket_stop
            + risk_scale * self.cfg.lambda_event_offset * loss_event_offset
            + risk_scale * self.cfg.lambda_window_start * loss_window_start
            + risk_scale * self.cfg.lambda_conditional_bucket * loss_conditional_bucket
            + risk_scale * self.cfg.lambda_conditional_offset * loss_conditional_offset
            + risk_scale * self.cfg.lambda_conditional_power * loss_conditional_power
            + risk_scale * self.cfg.lambda_pulse_duration * loss_pulse_duration
            + risk_scale * self.cfg.lambda_pulse_amplitude * loss_pulse_amplitude
            + base_aux_weights["agg"] * loss_agg
            + base_aux_weights["ghost"] * loss_ghost
            + base_aux_weights["peak"] * loss_peak
            + self.cfg.lambda_amp_on * loss_amp_on
            + self.cfg.lambda_orth * loss_orth
            + self.cfg.lambda_recon_power * loss_recon_power
            + self.cfg.lambda_recon_state * loss_recon_state
        )

        

        loss_dict = {
            "loss_total": total_loss.detach(),
            "risk_loss_scale": pred_power.new_tensor(risk_scale).detach(),
            "base_aux_schedule_progress": pred_power.new_tensor(
                self.base_auxiliary_schedule_progress()
            ).detach(),
            "weight_bridge": pred_power.new_tensor(base_aux_weights["bridge"]).detach(),
            "weight_agg": pred_power.new_tensor(base_aux_weights["agg"]).detach(),
            "weight_state": pred_power.new_tensor(base_aux_weights["state"]).detach(),
            "weight_ghost": pred_power.new_tensor(base_aux_weights["ghost"]).detach(),
            "weight_peak": pred_power.new_tensor(base_aux_weights["peak"]).detach(),
            "loss_power": loss_power.detach(),
            "loss_direct_power": loss_direct_power.detach(),
            "loss_bridge": loss_bridge.detach(),
            "loss_state": loss_state.detach(),
            "loss_event": loss_event.detach(),
            "loss_start": loss_start.detach(),
            "loss_stop": loss_stop.detach(),
            "loss_bucket_start": loss_bucket_start.detach(),
            "loss_bucket_stop": loss_bucket_stop.detach(),
            "loss_event_offset": loss_event_offset.detach(),
            "loss_window_start": loss_window_start.detach(),
            "loss_conditional_bucket": loss_conditional_bucket.detach(),
            "loss_conditional_offset": loss_conditional_offset.detach(),
            "loss_conditional_power": loss_conditional_power.detach(),
            "loss_pulse_duration": loss_pulse_duration.detach(),
            "loss_pulse_amplitude": loss_pulse_amplitude.detach(),
            "loss_agg": loss_agg.detach(),
            "loss_ghost": loss_ghost.detach(),
            "loss_peak": loss_peak.detach(),
            "loss_amp_on": loss_amp_on.detach(),
            "loss_orth": loss_orth.detach(),
            "loss_recon_power": loss_recon_power.detach(),
            "loss_recon_state": loss_recon_state.detach(),
            "history/macro_avg/MAE": history_macro_mae.detach(),
        }

        return total_loss, loss_dict


def build_pisa_loss_from_config(
    cfg: dict[str, Any],
    appliance_scales: Tensor | list[float] | dict[str, float],
) -> PISALoss:
    """
    Build PISALoss from config dictionary.

    Example cfg:

    cfg = {
        "loss": {
            "lambda_power": 1.0,
            "lambda_bridge": 0.3,
            "lambda_state": 0.25,
            "lambda_event": 0.10,
            "lambda_start": 0.10,
            "lambda_stop": 0.10,
            "lambda_agg": 0.20,
            "lambda_ghost": 0.05,
            "lambda_peak": 0.10,
            "lambda_amp_on": 0.20,
            "lambda_orth": 0.005,
        }
    }
    """
    loss_cfg = dict(cfg.get("loss", {}))
    state_alpha = loss_cfg.pop("state_alpha", None)
    event_alpha = loss_cfg.pop("event_alpha", None)
    stop_alpha = loss_cfg.pop("stop_alpha", None)
    state_pos_weight = loss_cfg.pop("state_pos_weight", None)
    event_pos_weight = loss_cfg.pop("event_pos_weight", None)

    valid_keys = set(PISALossConfig.__dataclass_fields__.keys())
    unknown_keys = sorted(set(loss_cfg.keys()) - valid_keys)
    if unknown_keys:
        raise ValueError(f"Unsupported loss config keys: {unknown_keys}")

    config = PISALossConfig(**loss_cfg)

    return PISALoss(
        appliance_scales=appliance_scales,
        config=config,
        state_alpha=state_alpha,
        event_alpha=event_alpha,
        stop_alpha=stop_alpha,
        state_pos_weight=state_pos_weight,
        event_pos_weight=event_pos_weight,
    )
