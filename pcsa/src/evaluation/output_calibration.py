from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from training import move_batch_to_device


@dataclass
class ApplianceAffineCalibration:
    """Eight-parameter calibration for four appliance power trajectories."""

    scale: Tensor
    bias: Tensor

    def apply(self, power: Tensor, rated_power: Tensor | None = None) -> Tensor:
        scale = self.scale.to(device=power.device, dtype=power.dtype).view(1, -1, 1)
        bias = self.bias.to(device=power.device, dtype=power.dtype).view(1, -1, 1)
        calibrated = torch.clamp(power * scale + bias, min=0.0)
        if rated_power is not None:
            cap = rated_power.to(device=power.device, dtype=power.dtype).view(1, -1, 1)
            calibrated = torch.minimum(calibrated, cap)
        return calibrated

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "scale": self.scale.detach().cpu().tolist(),
            "bias": self.bias.detach().cpu().tolist(),
        }


@torch.no_grad()
def collect_power_predictions(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    model.to(device)
    model.eval()
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    masks: list[Tensor] = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        output = model(batch)
        predictions.append(output["y_power"].detach().cpu())
        targets.append(batch["y_power"].detach().cpu())
        masks.append(
            batch.get("target_mask", torch.ones_like(batch["y_power"]))
            .detach()
            .cpu()
        )
    if not predictions:
        raise RuntimeError("Calibration loader produced no samples.")
    return (
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(masks, dim=0),
    )


def fit_appliance_affine_calibration(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    max_scale: float = 5.0,
) -> ApplianceAffineCalibration:
    """Fit masked per-appliance least-squares scale and bias on support only."""

    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask should have equal shape.")
    if prediction.ndim != 3:
        raise ValueError("Expected power tensors with shape [N, A, H].")
    scales: list[Tensor] = []
    biases: list[Tensor] = []
    for appliance in range(prediction.size(1)):
        valid = mask[:, appliance].reshape(-1) > 0.5
        x = prediction[:, appliance].reshape(-1)[valid].double()
        y = target[:, appliance].reshape(-1)[valid].double()
        finite = torch.isfinite(x) & torch.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.numel() < 2:
            scales.append(torch.tensor(1.0))
            biases.append(torch.tensor(0.0))
            continue
        x_mean = x.mean()
        y_mean = y.mean()
        variance = ((x - x_mean) ** 2).sum()
        if float(variance) <= 1e-12:
            scale = torch.tensor(1.0, dtype=torch.float64)
        else:
            scale = (((x - x_mean) * (y - y_mean)).sum() / variance).clamp(
                min=0.0,
                max=float(max_scale),
            )
        bias = y_mean - scale * x_mean
        scales.append(scale.float())
        biases.append(bias.float())
    return ApplianceAffineCalibration(
        scale=torch.stack(scales),
        bias=torch.stack(biases),
    )


def masked_macro_mae(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> float:
    error = (prediction - target).abs() * mask
    count = mask.sum(dim=(0, 2)).clamp_min(1.0)
    per_appliance = error.sum(dim=(0, 2)) / count
    return float(per_appliance.mean().item())


def select_affine_or_identity(
    calibration: ApplianceAffineCalibration,
    validation_prediction: Tensor,
    validation_target: Tensor,
    validation_mask: Tensor,
    rated_power: Tensor | None = None,
) -> tuple[ApplianceAffineCalibration, dict[str, Any]]:
    """Use target validation only to choose calibrated or identity output."""

    identity_mae = masked_macro_mae(
        validation_prediction,
        validation_target,
        validation_mask,
    )
    calibrated_prediction = calibration.apply(
        validation_prediction,
        rated_power=rated_power,
    )
    calibrated_mae = masked_macro_mae(
        calibrated_prediction,
        validation_target,
        validation_mask,
    )
    selected = calibration if calibrated_mae < identity_mae else ApplianceAffineCalibration(
        scale=torch.ones_like(calibration.scale),
        bias=torch.zeros_like(calibration.bias),
    )
    return selected, {
        "identity_validation_macro_MAE": identity_mae,
        "calibrated_validation_macro_MAE": calibrated_mae,
        "selected": "affine" if calibrated_mae < identity_mae else "identity",
    }
