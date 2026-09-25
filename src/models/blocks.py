from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for time-series Transformer encoder.

    Input:
        x: [B, T, D]

    Output:
        x + pe: [B, T, D]
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x shape [B, T, D], got {x.shape}")

        seq_len = x.size(1)

        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.pe.size(1)}."
            )

        return self.dropout(x + self.pe[:, :seq_len, :])


class MLP(nn.Module):
    """
    Two-layer MLP.

    Input:
        x: [..., in_dim]

    Output:
        y: [..., out_dim]
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            _build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MultiScaleTemporalStem(nn.Module):
    """
    Multi-scale temporal convolution stem.

    Purpose:
        Short kernels capture short appliance events such as microwave usage.
        Long kernels capture cyclic appliances such as refrigerator and air conditioner.

    Input:
        x: [B, T, C]

    Output:
        h: [B, T, D]
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        kernels: Iterable[int] = (3, 5, 9, 15),
        dropout: float = 0.1,
    ):
        super().__init__()

        kernels = list(kernels)
        if len(kernels) == 0:
            raise ValueError("kernels should not be empty.")

        branch_dim = max(8, d_model // len(kernels))

        self.branches = nn.ModuleList()
        for kernel_size in kernels:
            if kernel_size % 2 == 0:
                raise ValueError(
                    "Use odd kernel sizes to preserve sequence length."
                )

            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=input_dim,
                        out_channels=branch_dim,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.GELU(),
                    nn.BatchNorm1d(branch_dim),
                )
            )

        concat_dim = branch_dim * len(kernels)

        self.proj = nn.Sequential(
            nn.Conv1d(concat_dim, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x shape [B, T, C], got {x.shape}")

        # [B, T, C] -> [B, C, T]
        x_conv = x.transpose(1, 2)

        branch_outputs = [branch(x_conv) for branch in self.branches]
        h = torch.cat(branch_outputs, dim=1)

        h = self.proj(h)

        # [B, D, T] -> [B, T, D]
        h = h.transpose(1, 2)
        h = self.norm(h)
        return h


class MultiScaleTemporalEncoder(nn.Module):
    """
    Multi-scale temporal encoder.

    Structure:
        Multi-scale Conv1D stem
        + positional encoding
        + Transformer encoder

    Input:
        x: [B, T_in, C]

    Output:
        h_enc: [B, T_in, D]
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        kernels: Iterable[int] = (3, 5, 9, 15),
        max_len: int = 4096,
    ):
        super().__init__()

        self.stem = MultiScaleTemporalStem(
            input_dim=input_dim,
            d_model=d_model,
            kernels=kernels,
            dropout=dropout,
        )

        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.stem(x)
        h = self.positional_encoding(h)
        h = self.encoder(h)
        return h


class HouseholdFiLM(nn.Module):
    """
    Optional household FiLM adapter.

    For the current Home 7951 single-home experiment, this module can be disabled.
    For multi-home experiments, it can model household-specific amplitude and usage patterns.

    Input:
        h: [B, T, D]
        home_id: [B]

    Output:
        h_modulated: [B, T, D]
    """

    def __init__(
        self,
        num_homes: int,
        d_model: int,
        home_emb_dim: int = 16,
    ):
        super().__init__()

        self.home_embedding = nn.Embedding(num_homes, home_emb_dim)

        self.gamma = nn.Linear(home_emb_dim, d_model)
        self.beta = nn.Linear(home_emb_dim, d_model)

        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, h: Tensor, home_id: Tensor) -> Tensor:
        if home_id is None:
            return h

        emb = self.home_embedding(home_id.long())
        gamma = self.gamma(emb).unsqueeze(1)
        beta = self.beta(emb).unsqueeze(1)

        return gamma * h + beta


class ApplianceQueryStateBridge(nn.Module):
    """
    Appliance Query State Bridge, AQSB.

    This is the core bridge between aggregate history encoder and appliance-level decoder.

    It uses learnable appliance queries to extract appliance-specific latent states
    from the aggregate temporal representation.

    Input:
        h_enc: [B, T_in, D]

    Output:
        z_app:            [B, A, D]
        bridge_power_raw: [B, A, H]
        bridge_power:     [B, A, H]
        state_logits:     [B, A, H]
        event_logits:     [B, A, H]
        attn_weights:     [B, A, T_in]
    """

    def __init__(
        self,
        num_appliances: int,
        d_model: int,
        horizon: int,
        n_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_appliance_types: Optional[int] = None,
        appliance_type_ids: Optional[Tensor] = None,
        use_type_embedding: bool = True,
    ):
        super().__init__()

        self.num_appliances = num_appliances
        self.d_model = d_model
        self.horizon = horizon
        self.use_type_embedding = (
            use_type_embedding and num_appliance_types is not None
        )

        self.appliance_queries = nn.Parameter(
            torch.randn(num_appliances, d_model) * 0.02
        )

        if self.use_type_embedding:
            self.type_embedding = nn.Embedding(num_appliance_types, d_model)

            if appliance_type_ids is None:
                appliance_type_ids = torch.zeros(num_appliances, dtype=torch.long)

            if appliance_type_ids.numel() != num_appliances:
                raise ValueError(
                    "appliance_type_ids should have length num_appliances."
                )

            self.register_buffer(
                "appliance_type_ids",
                appliance_type_ids.long(),
                persistent=True,
            )
        else:
            self.type_embedding = None
            self.register_buffer(
                "appliance_type_ids",
                torch.zeros(num_appliances, dtype=torch.long),
                persistent=False,
            )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

        self.bridge_power_head = MLP(
            in_dim=d_model,
            hidden_dim=hidden_dim,
            out_dim=horizon,
            dropout=dropout,
        )

        self.state_head = MLP(
            in_dim=d_model,
            hidden_dim=hidden_dim,
            out_dim=horizon,
            dropout=dropout,
        )

        self.event_head = MLP(
            in_dim=d_model,
            hidden_dim=hidden_dim,
            out_dim=horizon,
            dropout=dropout,
        )

        self.state_bias = nn.Parameter(torch.zeros(num_appliances, horizon))
        self.event_bias = nn.Parameter(torch.zeros(num_appliances, horizon))

    @torch.no_grad()
    def set_state_prior(self, prior_on_prob: Tensor) -> None:
        """
        Initialize state bias using empirical ON probability.

        Args:
            prior_on_prob:
                shape [A] or [A, H], values in (0, 1).
        """
        eps = 1e-4
        prior = prior_on_prob.float().clamp(eps, 1.0 - eps)

        if prior.dim() == 1:
            prior = prior[:, None].repeat(1, self.horizon)

        if prior.shape != self.state_bias.shape:
            raise ValueError(
                f"Expected prior shape {self.state_bias.shape}, got {prior.shape}"
            )

        logit_prior = torch.log(prior / (1.0 - prior))
        self.state_bias.copy_(logit_prior)

    def _build_queries(self, batch_size: int) -> Tensor:
        q = self.appliance_queries

        if self.use_type_embedding:
            type_emb = self.type_embedding(self.appliance_type_ids)
            q = q + type_emb

        q = q.unsqueeze(0).expand(batch_size, -1, -1)
        return q

    def forward(self, h_enc: Tensor) -> dict[str, Tensor]:
        if h_enc.dim() != 3:
            raise ValueError(f"Expected h_enc shape [B, T, D], got {h_enc.shape}")

        batch_size = h_enc.size(0)
        q = self._build_queries(batch_size)

        attn_out, attn_weights = self.cross_attention(
            query=q,
            key=h_enc,
            value=h_enc,
            need_weights=True,
            average_attn_weights=True,
        )

        z = self.norm_attn(q + attn_out)
        z = self.norm_ffn(z + self.ffn(z))

        bridge_power_raw = self.bridge_power_head(z)
        bridge_power = F.softplus(bridge_power_raw)

        state_logits = self.state_head(z) + self.state_bias.unsqueeze(0)
        event_logits = self.event_head(z) + self.event_bias.unsqueeze(0)

        return {
            "z_app": z,
            "bridge_power_raw": bridge_power_raw,
            "bridge_power": bridge_power,
            "state_logits": state_logits,
            "event_logits": event_logits,
            "attn_weights": attn_weights,
        }


class ApplianceSpecificHead(nn.Module):
    """
    Lightweight appliance-specific forecasting head.

    Input:
        z_a: [B, D]

    Output:
        raw future power logits: [B, H]
    """

    def __init__(
        self,
        d_model: int,
        horizon: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, z_a: torch.Tensor) -> torch.Tensor:
        return self.net(z_a)


class SingleApplianceHorizonDecoder(nn.Module):
    """
    One appliance's horizon-query forecasting decoder.

    Each appliance owns its attention, feed-forward block, and output heads.
    This prevents high-activity appliances from dominating sparse appliances
    through a fully shared decoder.
    """

    def __init__(
        self,
        d_model: int,
        horizon: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        n_heads: int = 4,
        is_sparse: bool = False,
        future_time_dim: int = 0,
        event_bucket_size: int = 5,
    ):
        super().__init__()

        self.d_model = d_model
        self.horizon = horizon
        self.is_sparse = bool(is_sparse)
        self.future_time_dim = int(future_time_dim)
        self.event_bucket_size = max(1, int(event_bucket_size))
        self.num_event_buckets = (self.horizon + self.event_bucket_size - 1) // self.event_bucket_size

        self.horizon_embedding = nn.Embedding(horizon, d_model)
        self.register_buffer(
            "horizon_ids",
            torch.arange(horizon, dtype=torch.long),
            persistent=False,
        )

        if self.future_time_dim > 0:
            self.future_time_proj = nn.Sequential(
                nn.Linear(self.future_time_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
        else:
            self.future_time_proj = None

        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_self = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

        self.base_power_head = nn.Linear(d_model, 1)
        self.residual_power_head = nn.Linear(d_model, 1)
        self.persistence_gate_head = nn.Linear(d_model, 1)
        self.conditional_amplitude_head = nn.Linear(d_model, 1)
        self.start_head = nn.Linear(d_model, 1)
        self.event_head = nn.Linear(d_model, 1)
        self.state_delta_head = nn.Linear(d_model, 1)
        self.gate_head = nn.Linear(d_model, 3)

        # Event timing is intrinsically uncertain at one-minute resolution.
        # First predict whether a transition occurs in each coarse bucket, then
        # locate it inside the bucket with a conditional categorical head.
        self.start_bucket_head = nn.Linear(d_model, 1)
        self.stop_bucket_head = nn.Linear(d_model, 1)
        self.start_offset_head = nn.Linear(d_model, 1)
        self.stop_offset_head = nn.Linear(d_model, 1)
        self.window_start_head = nn.Linear(d_model, 1)
        self.conditional_profile_head = nn.Linear(d_model, 1)

    def _event_bucket_outputs(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return coarse start/stop logits and conditional minute offsets."""
        batch_size, horizon, d_model = z.shape
        if horizon != self.horizon or d_model != self.d_model:
            raise ValueError(f"Expected [B, {self.horizon}, {self.d_model}], got {z.shape}.")

        padded_horizon = self.num_event_buckets * self.event_bucket_size
        pad = padded_horizon - self.horizon
        if pad > 0:
            z_padded = F.pad(z, (0, 0, 0, pad))
        else:
            z_padded = z

        z_bucket = z_padded.view(
            batch_size,
            self.num_event_buckets,
            self.event_bucket_size,
            self.d_model,
        )
        valid = torch.ones(
            self.num_event_buckets,
            self.event_bucket_size,
            device=z.device,
            dtype=z.dtype,
        )
        if pad > 0:
            valid[-1, -pad:] = 0.0

        bucket_context = (z_bucket * valid.view(1, self.num_event_buckets, self.event_bucket_size, 1)).sum(dim=2)
        bucket_context = bucket_context / valid.sum(dim=-1).view(1, self.num_event_buckets, 1).clamp_min(1.0)

        start_bucket_logits = self.start_bucket_head(bucket_context).squeeze(-1)
        stop_bucket_logits = self.stop_bucket_head(bucket_context).squeeze(-1)
        start_offset_logits = self.start_offset_head(z_bucket).squeeze(-1)
        stop_offset_logits = self.stop_offset_head(z_bucket).squeeze(-1)

        if pad > 0:
            start_offset_logits[:, -1, -pad:] = -1e4
            stop_offset_logits[:, -1, -pad:] = -1e4

        return (
            start_bucket_logits,
            stop_bucket_logits,
            start_offset_logits,
            stop_offset_logits,
        )

    def forward(
        self,
        z_app: torch.Tensor,
        h_enc: torch.Tensor,
        future_time: torch.Tensor | None = None,
        persistence_power: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if z_app.dim() != 2:
            raise ValueError(f"Expected z_app shape [B, D], got {z_app.shape}.")

        if h_enc.dim() != 3:
            raise ValueError(f"Expected h_enc shape [B, T, D], got {h_enc.shape}.")

        B, D = z_app.shape

        if D != self.d_model or h_enc.size(0) != B or h_enc.size(2) != D:
            raise ValueError(
                f"Expected z_app [B, {self.d_model}] and h_enc [B, T, {self.d_model}], "
                f"got {z_app.shape} and {h_enc.shape}."
            )

        horizon_emb = self.horizon_embedding(self.horizon_ids)
        q = z_app.unsqueeze(1) + horizon_emb.unsqueeze(0)

        if self.future_time_proj is not None:
            if future_time is None:
                future_time = torch.zeros(
                    B,
                    self.horizon,
                    self.future_time_dim,
                    device=z_app.device,
                    dtype=z_app.dtype,
                )

            if (
                future_time.dim() != 3
                or future_time.size(0) != B
                or future_time.size(1) != self.horizon
                or future_time.size(2) != self.future_time_dim
            ):
                raise ValueError(
                    f"Expected future_time shape [B, {self.horizon}, "
                    f"{self.future_time_dim}], got {future_time.shape}."
                )

            q = q + self.future_time_proj(future_time.to(dtype=z_app.dtype))

        self_out, _ = self.self_attention(
            query=q,
            key=q,
            value=q,
            need_weights=False,
        )
        q = self.norm_self(q + self_out)

        attn_out, _ = self.cross_attention(
            query=q,
            key=h_enc,
            value=h_enc,
            need_weights=False,
        )

        z = self.norm_attn(q + attn_out)
        z = self.norm_ffn(z + self.ffn(z))

        base_raw = self.base_power_head(z).squeeze(-1)
        residual_raw = self.residual_power_head(z).squeeze(-1)
        persistence_gate = torch.sigmoid(
            self.persistence_gate_head(z).squeeze(-1)
        )
        amplitude_raw = self.conditional_amplitude_head(z).squeeze(-1)
        start_logits = self.start_head(z).squeeze(-1)
        event_logits = self.event_head(z).squeeze(-1)
        state_logits = self.state_delta_head(z).squeeze(-1)
        window_start_logits = self.window_start_head(z.mean(dim=1)).squeeze(-1)
        conditional_profile_logits = self.conditional_profile_head(z).squeeze(-1)
        (
            start_bucket_logits,
            stop_bucket_logits,
            start_offset_logits,
            stop_offset_logits,
        ) = self._event_bucket_outputs(z)

        direct_power = F.softplus(base_raw)

        if persistence_power is None:
            persistence = torch.zeros_like(direct_power)
        else:
            if persistence_power.dim() == 2:
                persistence_power = persistence_power.unsqueeze(-1)
            if persistence_power.shape != (B, 1, 1):
                raise ValueError(
                    f"Expected persistence_power shape [B, 1, 1] or [B, 1], "
                    f"got {persistence_power.shape}."
                )
            persistence = persistence_power.to(
                device=z_app.device,
                dtype=z_app.dtype,
            ).view(B, 1).expand(B, self.horizon)

        residual_power = torch.tanh(residual_raw) * direct_power
        y_power = persistence_gate * persistence + (1.0 - persistence_gate) * (
            direct_power + residual_power
        ).clamp_min(0.0)

        if self.is_sparse:
            sparse_gate = torch.sigmoid(start_logits)
            y_power = y_power + sparse_gate * F.softplus(amplitude_raw)
        else:
            sparse_gate = torch.zeros_like(base_raw)

        y_power = y_power.clamp_min(0.0)
        decoder_raw = torch.log(torch.expm1(y_power).clamp_min(1e-8))

        gate_weights = torch.softmax(self.gate_head(z.mean(dim=1)), dim=-1)

        return {
            "decoder_raw": decoder_raw,
            "y_raw": y_power,
            "gate_weights": gate_weights,
            "start_logits": start_logits,
            "decoder_event_logits": event_logits,
            "decoder_state_logits": state_logits,
            "start_bucket_logits": start_bucket_logits,
            "stop_bucket_logits": stop_bucket_logits,
            "start_offset_logits": start_offset_logits,
            "stop_offset_logits": stop_offset_logits,
            "window_start_logits": window_start_logits,
            "conditional_profile_logits": conditional_profile_logits,
            "sparse_gate": sparse_gate,
            "persistence_gate": persistence_gate,
            "conditional_profile": torch.sigmoid(conditional_profile_logits),
            "pulse_duration_logits": torch.zeros(
                B, self.horizon, device=z_app.device, dtype=z_app.dtype
            ),
            "pulse_amplitude_fraction": torch.zeros(
                B, device=z_app.device, dtype=z_app.dtype
            ),
            "pulse_scenario_profile": torch.zeros_like(y_power),
        }


class CyclicStateHorizonDecoder(SingleApplianceHorizonDecoder):
    """State-first decoder for cyclic loads such as HVAC compressors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cycle_state_head = nn.Linear(self.d_model, 1)
        self.cycle_amplitude_head = nn.Linear(self.d_model, 1)

    def forward(self, z_app: torch.Tensor, *args, **kwargs) -> dict[str, torch.Tensor]:
        out = super().forward(z_app, *args, **kwargs)
        cycle_state_logits = (
            out["decoder_state_logits"]
            + self.cycle_state_head(z_app).expand(-1, self.horizon)
        )
        state_prob = torch.sigmoid(cycle_state_logits)
        # The ordinary cross-attention power path supplies the amplitude while
        # the explicit state trajectory decides when the cyclic load is active.
        amplitude = F.softplus(
            out["decoder_raw"]
            + self.cycle_amplitude_head(z_app).expand(-1, self.horizon)
        )
        cycle_power = state_prob * amplitude
        out["decoder_state_logits"] = cycle_state_logits
        out["y_raw"] = cycle_power
        out["decoder_raw"] = torch.log(torch.expm1(cycle_power).clamp_min(1e-8))
        return out


class PersistenceHorizonDecoder(SingleApplianceHorizonDecoder):
    """Persistence-biased cyclic decoder for refrigerator duty cycles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retention_head = nn.Linear(self.d_model, 1)

    def forward(self, z_app: torch.Tensor, *args, **kwargs) -> dict[str, torch.Tensor]:
        out = super().forward(z_app, *args, **kwargs)
        # A refrigerator usually stays close to its boundary state and changes
        # gradually, so retain a cumulative-smoothed version of the forecast.
        raw_power = out["y_raw"]
        smooth_power = raw_power.cumsum(dim=-1) / torch.arange(
            1,
            self.horizon + 1,
            device=raw_power.device,
            dtype=raw_power.dtype,
        ).view(1, -1)
        retention = torch.sigmoid(
            torch.logit(out["persistence_gate"].clamp(1e-4, 1.0 - 1e-4))
            + self.retention_head(z_app).expand(-1, self.horizon)
        )
        power = retention * smooth_power + (1.0 - retention) * raw_power
        out["y_raw"] = power
        out["decoder_raw"] = torch.log(torch.expm1(power).clamp_min(1e-8))
        return out


class MultiStageTemplateHorizonDecoder(SingleApplianceHorizonDecoder):
    """Mixture-of-templates decoder for multi-stage appliance programmes."""

    def __init__(self, *args, num_templates: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_templates = int(num_templates)
        self.template_logits = nn.Parameter(
            torch.full((self.num_templates, self.horizon), -4.0)
        )
        self.template_mix_head = nn.Linear(self.d_model, self.num_templates)

    def forward(self, z_app: torch.Tensor, *args, **kwargs) -> dict[str, torch.Tensor]:
        out = super().forward(z_app, *args, **kwargs)
        mixture = torch.softmax(self.template_mix_head(z_app), dim=-1)
        templates = torch.sigmoid(self.template_logits)
        relative_profile = mixture @ templates
        bucket_prob = torch.softmax(out["start_bucket_logits"], dim=-1)
        offset_prob = torch.softmax(out["start_offset_logits"], dim=-1)
        start_prob = (bucket_prob.unsqueeze(-1) * offset_prob).flatten(1)
        start_prob = start_prob[:, : self.horizon]
        start_prob = start_prob / start_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        profile = torch.zeros_like(relative_profile)
        for minute in range(self.horizon):
            profile[:, minute] = (
                start_prob[:, : minute + 1]
                * relative_profile[:, : minute + 1].flip(-1)
            ).sum(dim=-1)
        profile = profile.clamp(0.0, 1.0)
        out["conditional_profile"] = profile
        out["conditional_profile_logits"] = torch.logit(
            profile.clamp(1e-4, 1.0 - 1e-4)
        )
        return out


class PulseHorizonDecoder(SingleApplianceHorizonDecoder):
    """Explicit start-time, duration and amplitude decoder for short bursts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.duration_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.horizon),
        )
        self.pulse_amplitude_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 1),
        )

    def _pulse_profile(
        self,
        start_prob: torch.Tensor,
        duration_prob: torch.Tensor,
    ) -> torch.Tensor:
        # survival[k] = P(duration > k), where durations are 1..H minutes.
        survival = duration_prob.flip(-1).cumsum(-1).flip(-1)
        profile = torch.zeros_like(start_prob)
        for minute in range(self.horizon):
            start_slice = start_prob[:, : minute + 1]
            lag_survival = survival[:, : minute + 1].flip(-1)
            profile[:, minute] = (start_slice * lag_survival).sum(dim=-1)
        return profile.clamp(0.0, 1.0)

    def forward(self, z_app: torch.Tensor, *args, **kwargs) -> dict[str, torch.Tensor]:
        out = super().forward(z_app, *args, **kwargs)
        bucket_prob = torch.softmax(out["start_bucket_logits"], dim=-1)
        offset_prob = torch.softmax(out["start_offset_logits"], dim=-1)
        start_prob = (bucket_prob.unsqueeze(-1) * offset_prob).flatten(1)
        start_prob = start_prob[:, : self.horizon]
        start_prob = start_prob / start_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        duration_logits = self.duration_head(z_app)
        duration_prob = torch.softmax(duration_logits, dim=-1)
        occupancy = self._pulse_profile(start_prob, duration_prob)
        amplitude_fraction = torch.sigmoid(
            self.pulse_amplitude_head(z_app).squeeze(-1)
        )
        profile = occupancy * amplitude_fraction.unsqueeze(-1)

        start_index = start_prob.argmax(dim=-1)
        duration = duration_prob.argmax(dim=-1) + 1
        minute_ids = torch.arange(self.horizon, device=z_app.device).view(1, -1)
        scenario = (
            (minute_ids >= start_index.unsqueeze(-1))
            & (minute_ids < (start_index + duration).unsqueeze(-1))
        ).to(z_app.dtype) * amplitude_fraction.unsqueeze(-1)

        out["conditional_profile"] = profile
        out["conditional_profile_logits"] = torch.logit(
            profile.clamp(1e-4, 1.0 - 1e-4)
        )
        out["pulse_duration_logits"] = duration_logits
        out["pulse_amplitude_fraction"] = amplitude_fraction
        out["pulse_scenario_profile"] = scenario
        return out


class TypeAwareApplianceDecoder(nn.Module):
    """
    Appliance-specific decoder bank.

    There is still a shared interface, but each appliance has its own
    SingleApplianceHorizonDecoder. The only shared pieces are appliance/type
    embeddings that condition the per-appliance latent before it enters its
    private decoder.
    """

    def __init__(
        self,
        num_appliances: int,
        d_model: int,
        horizon: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_appliance_types: int | None = None,
        appliance_type_ids: torch.Tensor | list[int] | None = None,
        n_heads: int = 4,
        sparse_appliance_mask: torch.Tensor | list[float] | None = None,
        future_time_dim: int = 0,
        event_bucket_size: int = 5,
        appliance_type_names: list[str] | None = None,
    ):
        super().__init__()

        self.num_appliances = num_appliances
        self.d_model = d_model
        self.horizon = horizon

        self.appliance_embedding = nn.Embedding(num_appliances, d_model)
        self.register_buffer(
            "appliance_ids",
            torch.arange(num_appliances, dtype=torch.long),
            persistent=False,
        )

        # Optional appliance type embedding
        if num_appliance_types is not None and appliance_type_ids is not None:
            self.type_embedding = nn.Embedding(num_appliance_types, d_model)
            self.register_buffer(
                "appliance_type_ids",
                torch.as_tensor(appliance_type_ids, dtype=torch.long),
                persistent=False,
            )
        else:
            self.type_embedding = None
            self.register_buffer(
                "appliance_type_ids",
                torch.zeros(num_appliances, dtype=torch.long),
                persistent=False,
            )

        if sparse_appliance_mask is None:
            sparse_appliance_mask = torch.zeros(num_appliances, dtype=torch.float32)
        else:
            sparse_appliance_mask = torch.as_tensor(
                sparse_appliance_mask,
                dtype=torch.float32,
            )

        if sparse_appliance_mask.numel() != num_appliances:
            raise ValueError(
                "sparse_appliance_mask should have length num_appliances."
            )

        self.register_buffer(
            "sparse_appliance_mask",
            sparse_appliance_mask.view(1, num_appliances, 1),
            persistent=True,
        )

        if appliance_type_names is None:
            appliance_type_names = ["generic"] * num_appliances
        if len(appliance_type_names) != num_appliances:
            raise ValueError("appliance_type_names should have length num_appliances.")
        self.appliance_type_names = list(appliance_type_names)

        decoder_classes = {
            "climate_cyclic": CyclicStateHorizonDecoder,
            "low_power_cyclic": PersistenceHorizonDecoder,
            "multi_stage_cycle": MultiStageTemplateHorizonDecoder,
            "thermal_burst": PulseHorizonDecoder,
            "sparse_event": PulseHorizonDecoder,
        }
        self.appliance_decoders = nn.ModuleList(
            [
                decoder_classes.get(
                    appliance_type_names[app_idx], SingleApplianceHorizonDecoder
                )(
                    d_model=d_model,
                    horizon=horizon,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    n_heads=n_heads,
                    is_sparse=bool(sparse_appliance_mask[app_idx].item() > 0.5),
                    future_time_dim=future_time_dim,
                    event_bucket_size=event_bucket_size,
                )
                for app_idx in range(num_appliances)
            ]
        )

    def forward(
        self,
        z_app: torch.Tensor,
        h_enc: torch.Tensor,
        future_time: torch.Tensor | None = None,
        persistence_power: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            z_app: [B, A, D]
            h_enc: [B, T, D]

        Returns:
            dict with decoder outputs.
        """
        B, A, D = z_app.shape

        if A != self.num_appliances:
            raise ValueError(
                f"Expected {self.num_appliances} appliances, but got {A}."
            )

        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {D}.")

        if h_enc.dim() != 3 or h_enc.size(0) != B or h_enc.size(2) != D:
            raise ValueError(
                f"Expected h_enc shape [B, T, {D}], got {h_enc.shape}."
            )

        app_emb = self.appliance_embedding(self.appliance_ids)

        if self.type_embedding is not None:
            type_emb = self.type_embedding(self.appliance_type_ids)
        else:
            type_emb = torch.zeros_like(app_emb)

        outputs: dict[str, list[torch.Tensor]] = {
            "decoder_raw": [],
            "y_raw": [],
            "gate_weights": [],
            "start_logits": [],
            "decoder_event_logits": [],
            "decoder_state_logits": [],
            "start_bucket_logits": [],
            "stop_bucket_logits": [],
            "start_offset_logits": [],
            "stop_offset_logits": [],
            "window_start_logits": [],
            "conditional_profile_logits": [],
            "sparse_gate": [],
            "persistence_gate": [],
            "conditional_profile": [],
            "pulse_duration_logits": [],
            "pulse_amplitude_fraction": [],
            "pulse_scenario_profile": [],
        }

        for app_idx, decoder in enumerate(self.appliance_decoders):
            z_i = z_app[:, app_idx, :] + app_emb[app_idx] + type_emb[app_idx]
            persistence_i = None
            if persistence_power is not None:
                persistence_i = persistence_power[:, app_idx : app_idx + 1, :]
            out_i = decoder(
                z_i,
                h_enc,
                future_time=future_time,
                persistence_power=persistence_i,
            )
            for key in outputs:
                outputs[key].append(out_i[key])

        return {
            "decoder_raw": torch.stack(outputs["decoder_raw"], dim=1),
            "y_raw": torch.stack(outputs["y_raw"], dim=1),
            "gate_weights": torch.stack(outputs["gate_weights"], dim=1),
            "start_logits": torch.stack(outputs["start_logits"], dim=1),
            "decoder_event_logits": torch.stack(
                outputs["decoder_event_logits"],
                dim=1,
            ),
            "decoder_state_logits": torch.stack(
                outputs["decoder_state_logits"],
                dim=1,
            ),
            "start_bucket_logits": torch.stack(
                outputs["start_bucket_logits"],
                dim=1,
            ),
            "stop_bucket_logits": torch.stack(
                outputs["stop_bucket_logits"],
                dim=1,
            ),
            "start_offset_logits": torch.stack(
                outputs["start_offset_logits"],
                dim=1,
            ),
            "stop_offset_logits": torch.stack(
                outputs["stop_offset_logits"],
                dim=1,
            ),
            "window_start_logits": torch.stack(
                outputs["window_start_logits"],
                dim=1,
            ),
            "conditional_profile_logits": torch.stack(
                outputs["conditional_profile_logits"],
                dim=1,
            ),
            "sparse_gate": torch.stack(outputs["sparse_gate"], dim=1),
            "persistence_gate": torch.stack(
                outputs["persistence_gate"],
                dim=1,
            ),
            "conditional_profile": torch.stack(
                outputs["conditional_profile"], dim=1
            ),
            "pulse_duration_logits": torch.stack(
                outputs["pulse_duration_logits"], dim=1
            ),
            "pulse_amplitude_fraction": torch.stack(
                outputs["pulse_amplitude_fraction"], dim=1
            ),
            "pulse_scenario_profile": torch.stack(
                outputs["pulse_scenario_profile"], dim=1
            ),
        }


class TailTemporalDisaggregationHead(nn.Module):
    """
    Per-time appliance head for same-window disaggregation.

    It uses the last H encoder states directly, preserving temporal alignment
    between aggregate changes and appliance targets.
    """

    def __init__(
        self,
        num_appliances: int,
        d_model: int,
        horizon: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_appliance_types: int | None = None,
        appliance_type_ids: torch.Tensor | list[int] | None = None,
    ):
        super().__init__()

        self.num_appliances = num_appliances
        self.d_model = d_model
        self.horizon = horizon

        self.appliance_embedding = nn.Embedding(num_appliances, d_model)
        self.register_buffer(
            "appliance_ids",
            torch.arange(num_appliances, dtype=torch.long),
            persistent=False,
        )

        if num_appliance_types is not None and appliance_type_ids is not None:
            self.type_embedding = nn.Embedding(num_appliance_types, d_model)
            self.register_buffer(
                "appliance_type_ids",
                torch.as_tensor(appliance_type_ids, dtype=torch.long),
                persistent=False,
            )
        else:
            self.type_embedding = None
            self.register_buffer(
                "appliance_type_ids",
                torch.zeros(num_appliances, dtype=torch.long),
                persistent=False,
            )

        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.power_head = nn.Linear(hidden_dim, 1)
        self.state_head = nn.Linear(hidden_dim, 1)
        self.event_head = nn.Linear(hidden_dim, 1)

    def forward(self, h_enc: torch.Tensor) -> dict[str, torch.Tensor]:
        if h_enc.dim() != 3:
            raise ValueError(f"Expected h_enc shape [B, T, D], got {h_enc.shape}")
        if h_enc.size(1) < self.horizon:
            raise ValueError(
                f"Encoder length {h_enc.size(1)} is shorter than horizon {self.horizon}."
            )

        B = h_enc.size(0)
        h_tail = h_enc[:, -self.horizon :, :]  # [B, H, D]

        x = h_tail.unsqueeze(1).expand(
            B,
            self.num_appliances,
            self.horizon,
            self.d_model,
        )

        app_emb = self.appliance_embedding(self.appliance_ids)
        x = x + app_emb.view(1, self.num_appliances, 1, self.d_model)

        if self.type_embedding is not None:
            type_emb = self.type_embedding(self.appliance_type_ids)
            x = x + type_emb.view(1, self.num_appliances, 1, self.d_model)

        z = self.net(x)

        return {
            "tail_power_raw": self.power_head(z).squeeze(-1),
            "tail_state_logits": self.state_head(z).squeeze(-1),
            "tail_event_logits": self.event_head(z).squeeze(-1),
        }


class PhysicsGuidedConstraintLayer(nn.Module):
    """
    Physics-guided appliance power constraint layer.

    Constraints:
        1. non-negative power,
        2. ON/OFF state-aware gating,
        3. optional rated-power cap,
        4. tiny ghost-load suppression during evaluation.

    Input:
        y_raw:        [B, A, H]
        state_logits: [B, A, H]

    Output:
        y_power:      [B, A, H]
        p_on:         [B, A, H]
    """

    def __init__(
        self,
        num_appliances: int,
        rated_power: Optional[Tensor] = None,
        off_threshold: float = 0.5,
        eps_s: float = 1e-4,
        hard_eval_gate: bool = False,
    ):
        super().__init__()

        self.num_appliances = num_appliances
        self.off_threshold = off_threshold
        self.eps_s = eps_s
        self.hard_eval_gate = hard_eval_gate

        if rated_power is None:
            rated_power = torch.full(
                (num_appliances,),
                float("inf"),
                dtype=torch.float32,
            )
        else:
            rated_power = rated_power.float()

        if rated_power.dim() != 1 or rated_power.numel() != num_appliances:
            raise ValueError(
                f"rated_power should be shape [A], got {rated_power.shape}"
            )

        self.register_buffer(
            "rated_power",
            rated_power.view(1, num_appliances, 1),
            persistent=True,
        )

    def forward(
        self,
        y_raw: Tensor,
        state_logits: Tensor,
    ) -> dict[str, Tensor]:
        if y_raw.shape != state_logits.shape:
            raise ValueError(
                f"y_raw and state_logits should have same shape, "
                f"got {y_raw.shape} and {state_logits.shape}"
            )

        power = F.softplus(y_raw)
        p_on = torch.sigmoid(state_logits)

        if self.training or not self.hard_eval_gate:
            gated_power = power * p_on
        else:
            on_mask = (p_on >= self.off_threshold).float()
            gated_power = power * on_mask

        capped_power = torch.minimum(gated_power, self.rated_power)

        if not self.training:
            capped_power = torch.where(
                capped_power < self.eps_s,
                torch.zeros_like(capped_power),
                capped_power,
            )

        return {
            "y_power": capped_power,
            "p_on": p_on,
        }
