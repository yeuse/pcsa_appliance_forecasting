from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _CausalResidualBlock(nn.Module):
    """Small causal TCN block used by the frozen-protocol baselines."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.padding = padding
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def _chomp(self, x: Tensor) -> Tensor:
        return x[..., :-self.padding] if self.padding else x

    def forward(self, x: Tensor) -> Tensor:
        residual = self.residual(x)
        z = self.dropout(F.gelu(self.norm1(self._chomp(self.conv1(x)))))
        z = self.dropout(F.gelu(self.norm2(self._chomp(self.conv2(z)))))
        return F.gelu(z + residual)


class _TCNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        for layer in range(num_layers):
            blocks.append(
                _CausalResidualBlock(
                    input_dim if layer == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2**layer,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected [B, T, C], got {tuple(x.shape)}.")
        return self.network(x.transpose(1, 2))[..., -1]


class _HorizonHeads(nn.Module):
    """Decode a TCN context together with known future calendar covariates."""

    def __init__(
        self,
        hidden_dim: int,
        future_time_dim: int,
        horizon: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.future_time_dim = int(future_time_dim)
        self.time_proj = nn.Linear(future_time_dim, hidden_dim) if future_time_dim else None
        self.horizon_embedding = nn.Embedding(horizon, hidden_dim)
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(hidden_dim, 5)

    def forward(self, context: Tensor, future_time: Tensor | None) -> Tensor:
        batch_size = context.size(0)
        z = context[:, None, :].expand(-1, self.horizon, -1)
        step = torch.arange(self.horizon, device=context.device)
        z = z + self.horizon_embedding(step)[None, :, :]
        if self.time_proj is not None:
            if future_time is None:
                future_time = context.new_zeros(
                    batch_size,
                    self.horizon,
                    self.future_time_dim,
                )
            if future_time.shape != (batch_size, self.horizon, self.future_time_dim):
                raise ValueError(
                    "Expected x_future_time shape "
                    f"[{batch_size}, {self.horizon}, {self.future_time_dim}], "
                    f"got {tuple(future_time.shape)}."
                )
            z = z + self.time_proj(future_time)
        return self.output(self.decoder(z))


def _format_tcn_outputs(
    decoded: Tensor,
    rated_power: Tensor,
    z_app: Tensor,
) -> dict[str, Tensor]:
    """Convert [B, A, H, 5] decoder output to the common evaluation API."""
    power_raw = decoded[..., 0]
    state_logits = decoded[..., 1]
    event_logits = decoded[..., 2]
    start_logits = decoded[..., 3]
    stop_logits = decoded[..., 4]
    p_on = torch.sigmoid(state_logits)
    y_power = torch.minimum(F.softplus(power_raw) * p_on, rated_power)
    return {
        "y_power": y_power,
        "bridge_power": y_power,
        "state_logits": state_logits,
        "event_logits": event_logits,
        "start_logits": start_logits,
        "stop_logits": stop_logits,
        "p_on": p_on,
        "z_app": z_app,
        "y_raw": power_raw,
        "decoder_raw": power_raw,
    }


class AggregateToApplianceTCN(nn.Module):
    """Direct aggregate-history-to-appliance TCN without an explicit NILM stage."""

    def __init__(
        self,
        input_dim: int,
        num_appliances: int,
        horizon: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        future_time_dim: int = 5,
        rated_power: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_appliances = int(num_appliances)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.encoder = _TCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.appliance_context = nn.Linear(hidden_dim, num_appliances * hidden_dim)
        self.heads = nn.ModuleList(
            [
                _HorizonHeads(hidden_dim, future_time_dim, horizon, dropout)
                for _ in range(num_appliances)
            ]
        )
        if rated_power is None:
            rated_power = torch.full((num_appliances,), float("inf"))
        self.register_buffer(
            "rated_power",
            rated_power.float().view(1, num_appliances, 1),
            persistent=True,
        )

    def forward(self, batch_or_x: Tensor | dict[str, Tensor]) -> dict[str, Tensor]:
        if isinstance(batch_or_x, dict):
            x_hist = batch_or_x["x_hist"]
            future_time = batch_or_x.get("x_future_time")
        else:
            x_hist = batch_or_x
            future_time = None
        if x_hist.dim() != 3 or x_hist.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected x_hist [B, T, {self.input_dim}], got {tuple(x_hist.shape)}."
            )
        context = self.encoder(x_hist)
        z_app = self.appliance_context(context).view(
            x_hist.size(0), self.num_appliances, self.hidden_dim
        )
        decoded = torch.stack(
            [self.heads[index](z_app[:, index], future_time) for index in range(self.num_appliances)],
            dim=1,
        )
        return _format_tcn_outputs(decoded, self.rated_power, z_app)


class ApplianceHistoryTCNForecaster(nn.Module):
    """Independent per-appliance TCN used in matched two-stage/Oracle arms."""

    def __init__(
        self,
        num_appliances: int,
        horizon: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_state_history: bool = True,
        future_time_dim: int = 5,
        rated_power: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.num_appliances = int(num_appliances)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.use_state_history = bool(use_state_history)
        input_dim = 2 if use_state_history else 1
        self.encoders = nn.ModuleList(
            [
                _TCNEncoder(input_dim, hidden_dim, num_layers, kernel_size, dropout)
                for _ in range(num_appliances)
            ]
        )
        self.heads = nn.ModuleList(
            [
                _HorizonHeads(hidden_dim, future_time_dim, horizon, dropout)
                for _ in range(num_appliances)
            ]
        )
        if rated_power is None:
            rated_power = torch.full((num_appliances,), float("inf"))
        self.register_buffer(
            "rated_power",
            rated_power.float().view(1, num_appliances, 1),
            persistent=True,
        )

    def _extract_history(
        self, batch_or_hist: Tensor | dict[str, Tensor]
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if isinstance(batch_or_hist, dict):
            hist_power = batch_or_hist.get("appliance_history_power")
            if hist_power is None:
                hist_power = batch_or_hist.get("y_hist_power")
            if hist_power is None:
                raise KeyError("Missing appliance_history_power/y_hist_power.")
            hist_state = batch_or_hist.get("appliance_history_state")
            if hist_state is None:
                hist_state = batch_or_hist.get("y_hist_state")
            future_time = batch_or_hist.get("x_future_time")
        else:
            hist_power, hist_state, future_time = batch_or_hist, None, None
        if hist_power.dim() != 3 or hist_power.size(1) != self.num_appliances:
            raise ValueError(
                f"Expected history [B, {self.num_appliances}, T], "
                f"got {tuple(hist_power.shape)}."
            )
        if hist_state is not None and hist_state.shape != hist_power.shape:
            raise ValueError("Appliance state and power histories must have equal shape.")
        return hist_power, hist_state, future_time

    def forward(self, batch_or_hist: Tensor | dict[str, Tensor]) -> dict[str, Tensor]:
        hist_power, hist_state, future_time = self._extract_history(batch_or_hist)
        if self.use_state_history and hist_state is None:
            hist_state = (hist_power > 0).to(hist_power.dtype)
        contexts: list[Tensor] = []
        decoded: list[Tensor] = []
        for index in range(self.num_appliances):
            parts = [hist_power[:, index, :, None]]
            if self.use_state_history:
                parts.append(hist_state[:, index, :, None])
            context = self.encoders[index](torch.cat(parts, dim=-1))
            contexts.append(context)
            decoded.append(self.heads[index](context, future_time))
        z_app = torch.stack(contexts, dim=1)
        return _format_tcn_outputs(
            torch.stack(decoded, dim=1), self.rated_power, z_app
        )


class AggregateOnlySeq2SeqBaseline(nn.Module):
    """
    Lightweight aggregate-only seq2seq baseline.

    It consumes the same aggregate-history input as PCSA, without an
    appliance-history reconstruction stage or appliance queries.
    """

    def __init__(
        self,
        input_dim: int,
        num_appliances: int,
        horizon: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        rated_power: Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_appliances = num_appliances
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.context = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        out_dim = num_appliances * horizon
        self.power_head = nn.Linear(hidden_dim, out_dim)
        self.state_head = nn.Linear(hidden_dim, out_dim)
        self.event_head = nn.Linear(hidden_dim, out_dim)
        self.start_head = nn.Linear(hidden_dim, out_dim)

        self.z_proj = nn.Linear(hidden_dim, num_appliances * hidden_dim)

        if rated_power is None:
            rated_power = torch.full((num_appliances,), float("inf"))
        self.register_buffer(
            "rated_power",
            rated_power.float().view(1, num_appliances, 1),
            persistent=True,
        )

    def _extract_x(self, batch_or_x: Tensor | dict[str, Tensor]) -> Tensor:
        if isinstance(batch_or_x, dict):
            x_hist = batch_or_x["x_hist"]
        else:
            x_hist = batch_or_x

        if x_hist.dim() != 3 or x_hist.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected x_hist shape [B, T, {self.input_dim}], got {x_hist.shape}."
            )

        return x_hist

    def forward(self, batch_or_x: Tensor | dict[str, Tensor]) -> dict[str, Tensor]:
        x_hist = self._extract_x(batch_or_x)
        B = x_hist.size(0)

        _, h_n = self.encoder(x_hist)
        ctx = self.context(h_n[-1])

        power_raw = self.power_head(ctx).view(B, self.num_appliances, self.horizon)
        state_logits = self.state_head(ctx).view(B, self.num_appliances, self.horizon)
        event_logits = self.event_head(ctx).view(B, self.num_appliances, self.horizon)
        start_logits = self.start_head(ctx).view(B, self.num_appliances, self.horizon)

        p_on = torch.sigmoid(state_logits)
        y_power = F.softplus(power_raw) * p_on
        y_power = torch.minimum(y_power, self.rated_power)

        z_app = self.z_proj(ctx).view(B, self.num_appliances, self.hidden_dim)

        return {
            "y_power": y_power,
            "bridge_power": y_power,
            "state_logits": state_logits,
            "event_logits": event_logits,
            "start_logits": start_logits,
            "p_on": p_on,
            "z_app": z_app,
            "y_raw": power_raw,
            "decoder_raw": power_raw,
        }


class HistoricalNILMSeq2Seq(nn.Module):
    """
    Genuine historical disaggregation baseline.

    It consumes aggregate/history covariates and reconstructs appliance-level
    power and state over the observed history window. Its output is intended to
    feed an appliance-history forecaster in a two-stage NILM+forecast pipeline.
    """

    def __init__(
        self,
        input_dim: int,
        num_appliances: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        rated_power: Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.num_appliances = int(num_appliances)
        self.hidden_dim = int(hidden_dim)

        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        feature_dim = 2 * hidden_dim
        self.temporal_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.power_head = nn.Linear(feature_dim, num_appliances)
        self.state_head = nn.Linear(feature_dim, num_appliances)

        if rated_power is None:
            rated_power = torch.full((num_appliances,), float("inf"))
        self.register_buffer(
            "rated_power",
            rated_power.float().view(1, num_appliances, 1),
            persistent=True,
        )

    def forward(self, batch_or_x: Tensor | dict[str, Tensor]) -> dict[str, Tensor]:
        if isinstance(batch_or_x, dict):
            x_hist = batch_or_x["x_hist"]
        else:
            x_hist = batch_or_x

        if x_hist.dim() != 3 or x_hist.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected x_hist shape [B, T, {self.input_dim}], got {x_hist.shape}."
            )

        h, _ = self.encoder(x_hist)
        z = self.temporal_head(h)
        power_raw = self.power_head(z).transpose(1, 2)
        state_logits = self.state_head(z).transpose(1, 2)
        p_on = torch.sigmoid(state_logits)
        past_power = torch.minimum(F.softplus(power_raw) * p_on, self.rated_power)

        return {
            "past_power": past_power,
            "past_power_raw": power_raw,
            "past_state_logits": state_logits,
            "past_p_on": p_on,
        }


class ApplianceHistoryForecaster(nn.Module):
    """
    Forecast appliance demand from appliance-level histories.

    This is the second stage in a genuine NILM+forecasting baseline. At
    evaluation time it can be fed either predicted histories from a NILM model
    or ground-truth histories for an Oracle upper bound.
    """

    def __init__(
        self,
        num_appliances: int,
        horizon: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_state_history: bool = True,
        rated_power: Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        self.num_appliances = int(num_appliances)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.use_state_history = bool(use_state_history)
        input_dim = self.num_appliances * (2 if self.use_state_history else 1)

        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.context = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        out_dim = self.num_appliances * self.horizon
        self.power_head = nn.Linear(hidden_dim, out_dim)
        self.state_head = nn.Linear(hidden_dim, out_dim)
        self.event_head = nn.Linear(hidden_dim, out_dim)
        self.start_head = nn.Linear(hidden_dim, out_dim)
        self.stop_head = nn.Linear(hidden_dim, out_dim)
        self.z_proj = nn.Linear(hidden_dim, self.num_appliances * hidden_dim)

        if rated_power is None:
            rated_power = torch.full((num_appliances,), float("inf"))
        self.register_buffer(
            "rated_power",
            rated_power.float().view(1, self.num_appliances, 1),
            persistent=True,
        )

    def _extract_history(self, batch_or_hist: Tensor | dict[str, Tensor]) -> tuple[Tensor, Tensor | None]:
        if isinstance(batch_or_hist, dict):
            if "appliance_history_power" in batch_or_hist:
                hist_power = batch_or_hist["appliance_history_power"]
            elif "y_hist_power" in batch_or_hist:
                hist_power = batch_or_hist["y_hist_power"]
            else:
                raise KeyError(
                    "Batch should contain 'appliance_history_power' or 'y_hist_power'."
                )
            hist_state = batch_or_hist.get("appliance_history_state")
            if hist_state is None:
                hist_state = batch_or_hist.get("y_hist_state")
        else:
            hist_power = batch_or_hist
            hist_state = None

        if hist_power.dim() != 3 or hist_power.size(1) != self.num_appliances:
            raise ValueError(
                "Expected appliance history power shape "
                f"[B, {self.num_appliances}, T], got {hist_power.shape}."
            )

        if hist_state is not None:
            if hist_state.shape != hist_power.shape:
                raise ValueError(
                    "History state should have the same shape as history power, "
                    f"got {hist_state.shape} and {hist_power.shape}."
                )
            hist_state = hist_state.to(device=hist_power.device, dtype=hist_power.dtype)

        return hist_power, hist_state

    def forward(self, batch_or_hist: Tensor | dict[str, Tensor]) -> dict[str, Tensor]:
        hist_power, hist_state = self._extract_history(batch_or_hist)
        batch_size = hist_power.size(0)

        features = [hist_power.transpose(1, 2)]
        if self.use_state_history:
            if hist_state is None:
                hist_state = (hist_power > 0.0).to(hist_power.dtype)
            features.append(hist_state.transpose(1, 2))
        x = torch.cat(features, dim=-1)

        _, h_n = self.encoder(x)
        ctx = self.context(h_n[-1])

        power_raw = self.power_head(ctx).view(
            batch_size,
            self.num_appliances,
            self.horizon,
        )
        state_logits = self.state_head(ctx).view(
            batch_size,
            self.num_appliances,
            self.horizon,
        )
        event_logits = self.event_head(ctx).view(
            batch_size,
            self.num_appliances,
            self.horizon,
        )
        start_logits = self.start_head(ctx).view(
            batch_size,
            self.num_appliances,
            self.horizon,
        )
        stop_logits = self.stop_head(ctx).view(
            batch_size,
            self.num_appliances,
            self.horizon,
        )
        p_on = torch.sigmoid(state_logits)
        y_power = torch.minimum(F.softplus(power_raw) * p_on, self.rated_power)
        z_app = self.z_proj(ctx).view(
            batch_size,
            self.num_appliances,
            self.hidden_dim,
        )

        return {
            "y_power": y_power,
            "bridge_power": y_power,
            "state_logits": state_logits,
            "event_logits": event_logits,
            "start_logits": start_logits,
            "stop_logits": stop_logits,
            "p_on": p_on,
            "z_app": z_app,
            "y_raw": power_raw,
            "decoder_raw": power_raw,
        }


class TwoStageNILMForecastPipeline(nn.Module):
    """
    Frozen historical NILM followed by an appliance-history forecaster.

    This class represents the reviewer-requested two-stage comparison: a
    disaggregation model first estimates appliance histories, then a separate
    forecaster predicts future appliance demand from those histories.
    """

    def __init__(
        self,
        nilm_model: HistoricalNILMSeq2Seq,
        forecaster: nn.Module,
        freeze_nilm: bool = True,
        state_thresholds: Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        self.nilm_model = nilm_model
        self.forecaster = forecaster
        self.freeze_nilm = bool(freeze_nilm)
        # Cross-home transfer may tune only the NILM and forecaster output
        # heads.  This flag keeps both frozen temporal encoders deterministic
        # while still allowing gradients through them to the selected heads.
        self.head_only_adaptation = False

        if state_thresholds is not None:
            state_thresholds = state_thresholds.float().view(1, -1, 1)
        self.register_buffer(
            "state_thresholds",
            state_thresholds,
            persistent=False,
        )

        if self.freeze_nilm:
            for param in self.nilm_model.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True) -> "TwoStageNILMForecastPipeline":
        """Train the forecaster while keeping the frozen NILM deterministic."""
        super().train(mode)
        if mode and self.head_only_adaptation:
            self.nilm_model.eval()
            self.forecaster.eval()
            self.nilm_model.power_head.train(True)
            self.nilm_model.state_head.train(True)
            for head in self.forecaster.heads:
                head.train(True)
        elif self.freeze_nilm:
            # ``nn.Module.train`` recurses into every child.  Without this
            # override the frozen NILM's dropout would still perturb the
            # generated histories used to fit the second stage.
            self.nilm_model.eval()
        return self

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.freeze_nilm:
            with torch.no_grad():
                nilm_out = self.nilm_model(batch)
        else:
            nilm_out = self.nilm_model(batch)

        hist_power = nilm_out["past_power"].detach() if self.freeze_nilm else nilm_out["past_power"]
        hist_state = nilm_out["past_p_on"]
        if self.state_thresholds is not None:
            hist_state = (hist_power >= self.state_thresholds).to(hist_power.dtype)
        elif self.freeze_nilm:
            hist_state = hist_state.detach()

        forecast_batch = dict(batch)
        forecast_batch["appliance_history_power"] = hist_power
        forecast_batch["appliance_history_state"] = hist_state
        out = self.forecaster(forecast_batch)
        out["nilm_past_power"] = hist_power
        out["nilm_past_p_on"] = hist_state
        # Common aliases let PISALoss apply exactly the same historical
        # reconstruction supervision used by PISA during target adaptation.
        out["past_power"] = nilm_out["past_power"]
        out["past_state_logits"] = nilm_out["past_state_logits"]
        out["past_p_on"] = nilm_out["past_p_on"]
        return out
