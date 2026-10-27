from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .blocks import MultiScaleTemporalEncoder


class SingleApplianceHistoricalHead(nn.Module):
    """A private historical power/state head for one appliance."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(16, d_model // 2)
        self.features = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.power_head = nn.Linear(hidden_dim, 1)
        self.state_head = nn.Linear(hidden_dim, 1)

    def forward(self, memory: Tensor) -> dict[str, Tensor]:
        hidden = self.features(memory)
        state_logits = self.state_head(hidden).squeeze(-1)
        p_on = torch.sigmoid(state_logits)
        amplitude = F.softplus(self.power_head(hidden).squeeze(-1))
        power = amplitude * (0.02 + 0.98 * p_on)
        return {
            "power": power,
            "state_logits": state_logits,
            "p_on": p_on,
            "amplitude": amplitude,
        }


class HistoricalApplianceAuxiliaryHead(nn.Module):
    """Bank of independent historical heads, one for each appliance."""

    def __init__(self, d_model: int, num_appliances: int, dropout: float = 0.1):
        super().__init__()
        self.num_appliances = num_appliances
        self.appliance_heads = nn.ModuleList(
            [
                SingleApplianceHistoricalHead(d_model=d_model, dropout=dropout)
                for _ in range(num_appliances)
            ]
        )

    def forward(self, memory: Tensor) -> dict[str, Tensor]:
        outputs = [head(memory) for head in self.appliance_heads]
        return {
            "past_power": torch.stack([item["power"] for item in outputs], dim=1),
            "past_state_logits": torch.stack(
                [item["state_logits"] for item in outputs], dim=1
            ),
            "past_p_on": torch.stack([item["p_on"] for item in outputs], dim=1),
            "past_amplitude": torch.stack(
                [item["amplitude"] for item in outputs], dim=1
            ),
        }


class AggregateForecastDecoder(nn.Module):
    """Horizon-query decoder with cross-attention to historical memory."""

    def __init__(
        self,
        d_model: int,
        horizon: int,
        n_heads: int,
        hidden_dim: int,
        future_time_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon
        self.future_time_dim = future_time_dim
        self.horizon_embedding = nn.Embedding(horizon, d_model)
        self.register_buffer(
            "horizon_ids", torch.arange(horizon, dtype=torch.long), persistent=False
        )
        self.future_time_proj = (
            nn.Sequential(
                nn.Linear(future_time_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            if future_time_dim > 0
            else None
        )
        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_self = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.power_head = nn.Linear(d_model, 1)
        self.persistence_gate_head = nn.Linear(d_model, 1)

    def forward(
        self,
        memory: Tensor,
        future_time: Optional[Tensor],
        last_mains: Tensor,
    ) -> dict[str, Tensor]:
        batch_size = memory.size(0)
        query = self.horizon_embedding(self.horizon_ids).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        if self.future_time_proj is not None:
            if future_time is None:
                future_time = torch.zeros(
                    batch_size,
                    self.horizon,
                    self.future_time_dim,
                    device=memory.device,
                    dtype=memory.dtype,
                )
            query = query + self.future_time_proj(future_time.to(memory.dtype))

        self_out, _ = self.self_attention(query, query, query, need_weights=False)
        query = self.norm_self(query + self_out)
        cross_out, attention = self.cross_attention(
            query, memory, memory, need_weights=True, average_attn_weights=True
        )
        hidden = self.norm_cross(query + cross_out)
        hidden = self.norm_ffn(hidden + self.ffn(hidden))

        direct_power = F.softplus(self.power_head(hidden).squeeze(-1))
        persistence_gate = torch.sigmoid(
            self.persistence_gate_head(hidden).squeeze(-1)
        )
        persistence = last_mains.expand(-1, self.horizon)
        power = persistence_gate * persistence + (1.0 - persistence_gate) * direct_power
        return {
            "aggregate_power_raw": power,
            "persistence_gate": persistence_gate,
            "forecast_attention": attention,
        }


class ApplianceStateInformedForecaster(nn.Module):
    """Aggregate forecaster regularized by historical appliance supervision.

    Appliance labels are used only to supervise historical representations during
    training. The forecast target and primary output are aggregate household power.
    """

    def __init__(
        self,
        input_dim: int,
        num_appliances: int,
        horizon: int = 30,
        d_model: int = 128,
        n_heads: int = 4,
        encoder_layers: int = 3,
        encoder_ff_dim: int = 256,
        decoder_hidden_dim: int = 256,
        future_time_dim: Optional[int] = None,
        dropout: float = 0.1,
        appliance_scales: Optional[Tensor] = None,
        aggregate_cap: Optional[float] = None,
        use_state_injection: bool = True,
        use_physical_constraints: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_appliances = num_appliances
        self.horizon = horizon
        self.future_time_dim = (
            max(input_dim - 1, 0) if future_time_dim is None else future_time_dim
        )
        self.use_state_injection = bool(use_state_injection)
        self.use_physical_constraints = bool(use_physical_constraints)

        if appliance_scales is None:
            appliance_scales = torch.ones(num_appliances)
        self.register_buffer(
            "appliance_scales",
            torch.as_tensor(appliance_scales, dtype=torch.float32).view(
                1, num_appliances, 1
            ),
            persistent=True,
        )
        cap = float("inf") if aggregate_cap is None else float(aggregate_cap)
        self.register_buffer(
            "aggregate_cap", torch.tensor(cap, dtype=torch.float32), persistent=True
        )

        self.encoder = MultiScaleTemporalEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=encoder_layers,
            dim_feedforward=encoder_ff_dim,
            dropout=dropout,
        )
        self.auxiliary_head = HistoricalApplianceAuxiliaryHead(
            d_model=d_model,
            num_appliances=num_appliances,
            dropout=dropout,
        )
        self.state_feature_proj = nn.Sequential(
            nn.Linear(2 * num_appliances, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.memory_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )
        self.decoder = AggregateForecastDecoder(
            d_model=d_model,
            horizon=horizon,
            n_heads=n_heads,
            hidden_dim=decoder_hidden_dim,
            future_time_dim=self.future_time_dim,
            dropout=dropout,
        )

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        x_hist = batch["x_hist"]
        memory = self.encoder(x_hist)
        auxiliary = self.auxiliary_head(memory)

        scaled_past_power = auxiliary["past_power"] / self.appliance_scales.clamp_min(
            1e-6
        )
        state_features = torch.cat(
            [auxiliary["past_p_on"], scaled_past_power], dim=1
        ).transpose(1, 2)
        state_memory = self.state_feature_proj(state_features)
        if self.use_state_injection:
            gate = self.memory_gate(torch.cat([memory, state_memory], dim=-1))
            forecast_memory = memory + gate * state_memory
        else:
            gate = torch.zeros_like(memory)
            forecast_memory = memory

        if "y_hist_mains" not in batch:
            raise KeyError("batch should contain y_hist_mains for persistence context.")
        last_mains = batch["y_hist_mains"][:, -1:].to(x_hist.dtype)
        decoder = self.decoder(
            forecast_memory,
            batch.get("x_future_time"),
            last_mains,
        )
        aggregate_power = decoder["aggregate_power_raw"].clamp_min(0.0)
        if self.use_physical_constraints:
            aggregate_power = torch.minimum(aggregate_power, self.aggregate_cap)

        return {
            "aggregate_power": aggregate_power,
            "past_power": auxiliary["past_power"],
            "past_state_logits": auxiliary["past_state_logits"],
            "past_p_on": auxiliary["past_p_on"],
            "state_memory_gate": gate,
            **decoder,
        }


class AggregateGRUForecaster(nn.Module):
    """Aggregate-only GRU baseline using the same inputs and forecast horizon."""

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        aggregate_cap: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )
        cap = float("inf") if aggregate_cap is None else float(aggregate_cap)
        self.register_buffer("aggregate_cap", torch.tensor(cap), persistent=True)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        _, hidden = self.encoder(batch["x_hist"])
        power = F.softplus(self.head(hidden[-1]))
        power = torch.minimum(power, self.aggregate_cap)
        return {"aggregate_power": power, "aggregate_power_raw": power}
