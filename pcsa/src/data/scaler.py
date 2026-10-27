# src/pisa/data/scaler.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


class StandardScaler1D:
    """Standard scaler for input features such as grid power.

    This scaler supports arrays with arbitrary leading dimensions as long as
    the last dimension corresponds to features.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray | pd.DataFrame) -> "StandardScaler1D":
        arr = x.to_numpy(dtype=np.float32) if isinstance(x, pd.DataFrame) else np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        self.mean_ = np.nanmean(arr, axis=0).astype(np.float32)
        self.std_ = np.nanstd(arr, axis=0).astype(np.float32)
        self.std_ = np.maximum(self.std_, self.eps)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return ((np.asarray(x, dtype=np.float32) - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (np.asarray(x, dtype=np.float32) * self.std_ + self.mean_).astype(np.float32)

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler1D has not been fitted.")

    def state_dict(self) -> dict[str, list[float]]:
        self._check_fitted()
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist(), "eps": self.eps}

    @classmethod
    def from_state_dict(cls, state: dict) -> "StandardScaler1D":
        obj = cls(eps=float(state.get("eps", 1e-6)))
        obj.mean_ = np.asarray(state["mean"], dtype=np.float32)
        obj.std_ = np.asarray(state["std"], dtype=np.float32)
        return obj


class ApplianceScaler:
    """Per-appliance target scaler.

    For PISA, target appliances should be scaled separately because air1 and
    waterheater1 can dominate the loss if raw kW values are used directly.
    """

    def __init__(self, mode: str = "p99", eps: float = 1e-6):
        if mode not in {"p95", "p99", "p95_positive", "p99_positive", "p99_on", "max", "none"}:
            raise ValueError(f"Unsupported scaler mode: {mode}")
        self.mode = mode
        self.eps = eps
        self.appliance_cols: list[str] | None = None
        self.scale_: np.ndarray | None = None
        self.fit_audit_: dict = {}

    def fit(
        self, df: pd.DataFrame, appliance_cols: Iterable[str],
        state_thresholds: dict[str, float] | None = None,
    ) -> "ApplianceScaler":
        self.appliance_cols = list(appliance_cols)
        values = df[self.appliance_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        values = np.clip(values, a_min=0.0, a_max=None)

        self.fit_audit_ = {}
        if self.mode == "p99_on":
            if state_thresholds is None:
                raise ValueError("p99_on requires training state thresholds.")
            scale = []
            for j, app in enumerate(self.appliance_cols):
                threshold = float(state_thresholds[app])
                if not np.isfinite(threshold) or threshold <= 0:
                    raise ValueError(f"Invalid ON threshold for {app}: {threshold}")
                valid = np.isfinite(values[:, j])
                availability = f"{app}_available"
                if availability in df.columns:
                    valid &= pd.to_numeric(df[availability], errors="coerce").fillna(0).to_numpy() > 0.5
                on = values[:, j][valid & (values[:, j] >= threshold)]
                # No target/validation statistics enter this fallback. Keep
                # all-off devices numerically safe and record them explicitly.
                value = float(np.percentile(on, 99)) if on.size else threshold
                scale.append(max(value, threshold, self.eps))
                self.fit_audit_[app] = {
                    "on_threshold_kw": threshold, "on_train_samples": int(on.size),
                    "scale_kw": scale[-1],
                    "fallback": None if on.size else "training_state_threshold",
                }
            scale = np.asarray(scale, dtype=np.float32)
        elif self.mode == "none":
            scale = np.ones(values.shape[1], dtype=np.float32)
        elif self.mode == "max":
            scale = np.max(values, axis=0)
        elif self.mode == "p95":
            scale = np.percentile(values, 95, axis=0)
        elif self.mode == "p99":
            scale = np.percentile(values, 99, axis=0)
        elif self.mode == "p95_positive":
            scale = []
            for j in range(values.shape[1]):
                positive = values[:, j][values[:, j] > self.eps]
                scale.append(np.percentile(positive, 95) if positive.size else 1.0)
            scale = np.asarray(scale, dtype=np.float32)
        else:  # p99_positive
            scale = []
            for j in range(values.shape[1]):
                positive = values[:, j][values[:, j] > self.eps]
                scale.append(np.percentile(positive, 99) if positive.size else 1.0)
            scale = np.asarray(scale, dtype=np.float32)

        scale = np.asarray(scale, dtype=np.float32)
        scale = np.maximum(scale, self.eps)
        self.scale_ = scale
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        self._check_fitted()
        # y can be [N, A], [B, A, H], or [A, H]
        arr = np.asarray(y, dtype=np.float32)
        if arr.ndim == 2:
            return (arr / self.scale_[None, :]).astype(np.float32)
        if arr.ndim == 3:
            return (arr / self.scale_[None, :, None]).astype(np.float32)
        raise ValueError(f"Expected y shape [N,A] or [B,A,H], got {arr.shape}")

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        self._check_fitted()
        arr = np.asarray(y_scaled, dtype=np.float32)
        if arr.ndim == 2:
            return (arr * self.scale_[None, :]).astype(np.float32)
        if arr.ndim == 3:
            return (arr * self.scale_[None, :, None]).astype(np.float32)
        raise ValueError(f"Expected y_scaled shape [N,A] or [B,A,H], got {arr.shape}")

    def _check_fitted(self) -> None:
        if self.scale_ is None:
            raise RuntimeError("ApplianceScaler has not been fitted.")

    def state_dict(self) -> dict:
        self._check_fitted()
        return {
            "mode": self.mode,
            "eps": self.eps,
            "appliance_cols": self.appliance_cols,
            "scale": self.scale_.tolist(),
            "fit_audit": self.fit_audit_,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "ApplianceScaler":
        obj = cls(mode=state.get("mode", "p99"), eps=float(state.get("eps", 1e-6)))
        obj.appliance_cols = list(state["appliance_cols"])
        obj.scale_ = np.asarray(state["scale"], dtype=np.float32)
        obj.fit_audit_ = dict(state.get("fit_audit", {}))
        return obj


@dataclass
class PISAScalers:
    """Container for all scalers used by the data pipeline."""

    input_scaler: StandardScaler1D
    appliance_scaler: ApplianceScaler

    def state_dict(self) -> dict:
        return {
            "input_scaler": self.input_scaler.state_dict(),
            "appliance_scaler": self.appliance_scaler.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "PISAScalers":
        return cls(
            input_scaler=StandardScaler1D.from_state_dict(state["input_scaler"]),
            appliance_scaler=ApplianceScaler.from_state_dict(state["appliance_scaler"]),
        )
