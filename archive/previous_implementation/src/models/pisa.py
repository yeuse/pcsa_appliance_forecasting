from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .blocks import (
    ApplianceQueryStateBridge,
    HouseholdFiLM,
    MultiScaleTemporalEncoder,
    PhysicsGuidedConstraintLayer,
    TailTemporalDisaggregationHead,
    TypeAwareApplianceDecoder,
)


DEFAULT_APPLIANCE_NAMES = [
    "air1",
    "refrigerator1",
    "dishwasher1",
    "microwave1",
]

DEFAULT_APPLIANCE_TYPES = {
    "air1": "climate_cyclic",
    "refrigerator1": "low_power_cyclic",
    "dishwasher1": "multi_stage_cycle",
    "waterheater1": "thermal_burst",
    "microwave1": "sparse_event",
}

# Rated-power caps are in kW. Record run-specific values in the configuration;
# defaults must not be treated as estimates from a test split.
DEFAULT_RATED_POWER_KW = {
    "air1": 3.0,
    "refrigerator1": 0.5,
    "dishwasher1": 2.0,
    "waterheater1": 5.0,
    "microwave1": 2.5,
}

RISK_HEAD_MODULE_NAMES = (
    "start_head",
    "event_head",
    "start_bucket_head",
    "stop_bucket_head",
    "start_offset_head",
    "stop_offset_head",
    "window_start_head",
    "conditional_profile_head",
    "duration_head",
    "pulse_amplitude_head",
    "template_mix_head",
)


def rated_power_tensor_for_appliances(
    appliance_names: list[str],
    rated_power_kw: Optional[Tensor | list[float] | tuple[float, ...]] = None,
) -> Tensor:
    """Return validated, explicit rated-power caps for ``appliance_names``.

    The default map is limited to the Home 7951 appliance vocabulary.  Any
    other appliance must provide a cap explicitly so that the physical clamp
    cannot silently degrade into an unbounded output.
    """
    if rated_power_kw is None:
        missing = [name for name in appliance_names if name not in DEFAULT_RATED_POWER_KW]
        if missing:
            raise ValueError(
                "rated_power_kw must be supplied for appliances without a "
                f"default cap: {missing}."
            )
        rated_power_kw = [DEFAULT_RATED_POWER_KW[name] for name in appliance_names]

    rated_power = torch.as_tensor(rated_power_kw, dtype=torch.float32).view(-1)
    if rated_power.numel() != len(appliance_names):
        raise ValueError(
            "rated_power_kw should contain one positive value per appliance; "
            f"got {rated_power.numel()} values for {len(appliance_names)} appliances."
        )
    if not bool(torch.isfinite(rated_power).all()) or bool((rated_power <= 0).any()):
        raise ValueError("rated_power_kw values must all be finite and positive.")
    return rated_power


def build_appliance_type_ids(
    appliance_names: list[str],
    appliance_types: Optional[dict[str, str]] = None,
) -> tuple[Tensor, dict[str, int]]:
    """
    Convert appliance type names into integer type ids.

    Args:
        appliance_names:
            appliance column names.
        appliance_types:
            mapping from appliance name to type name.

    Returns:
        type_ids:
            LongTensor with shape [A].
        type_to_id:
            mapping from type name to integer id.
    """
    if appliance_types is None:
        appliance_types = DEFAULT_APPLIANCE_TYPES

    type_names = []
    for name in appliance_names:
        type_names.append(appliance_types.get(name, name))

    unique_types = sorted(set(type_names))
    type_to_id = {type_name: idx for idx, type_name in enumerate(unique_types)}

    type_ids = torch.tensor(
        [type_to_id[type_name] for type_name in type_names],
        dtype=torch.long,
    )

    return type_ids, type_to_id

class PastApplianceReconstructionHead(nn.Module):
    """
    Reconstruct historical appliance-level power and state
    from encoder hidden states.

    Input:
        h_enc: [B, T_h, D]

    Output:
        past_power: [B, A, T_h]
        past_state_logits: [B, A, T_h]
        past_p_on: [B, A, T_h]
    """

    def __init__(
        self,
        d_model: int,
        num_appliances: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.power_head = nn.Linear(d_model, num_appliances)
        self.state_head = nn.Linear(d_model, num_appliances)

    def forward(self, h_enc: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.net(h_enc)  # [B, T_h, D]

        past_power_raw = self.power_head(z).transpose(1, 2)        # [B, A, T_h]
        past_state_logits = self.state_head(z).transpose(1, 2)     # [B, A, T_h]

        past_power = F.softplus(past_power_raw)

        return {
            "past_power": past_power,
            "past_power_raw": past_power_raw,
            "past_state_logits": past_state_logits,
            "past_p_on": torch.sigmoid(past_state_logits),
        }


class PastApplianceReconstructionResidualRefiner(nn.Module):
    """
    Optional residual refiner for deployment-facing appliance histories.

    The base ``PastApplianceReconstructionHead`` remains untouched because its
    summaries already condition the frozen PISA future decoder.  This refiner
    reads the same aggregate-encoder sequence and corrects historical power and
    state only for downstream consumers such as a future residual TCN or HEMS.
    Zero-initialized residual heads make the initial refined history exactly
    equal to the base history reconstructed by an existing checkpoint.
    """

    def __init__(
        self,
        d_model: int,
        num_appliances: int,
        dropout: float = 0.1,
        gate_init: float = -3.0,
    ) -> None:
        super().__init__()
        self.num_appliances = int(num_appliances)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.power_delta_head = nn.Linear(d_model, num_appliances)
        self.state_delta_head = nn.Linear(d_model, num_appliances)
        self.power_gate_logit = nn.Parameter(
            torch.full((num_appliances, 1), float(gate_init))
        )
        self.state_gate_logit = nn.Parameter(
            torch.full((num_appliances, 1), float(gate_init))
        )

    @torch.no_grad()
    def zero_init_output(self) -> None:
        """Keep an existing checkpoint's reconstructed history unchanged."""
        for head in (self.power_delta_head, self.state_delta_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        h_enc: Tensor,
        base_power_raw: Tensor,
        base_state_logits: Tensor,
    ) -> dict[str, Tensor]:
        z = self.net(h_enc)
        power_delta_raw = self.power_delta_head(z).transpose(1, 2)
        state_delta_logits = self.state_delta_head(z).transpose(1, 2)
        power_gate = torch.sigmoid(self.power_gate_logit).view(
            1, self.num_appliances, 1
        )
        state_gate = torch.sigmoid(self.state_gate_logit).view(
            1, self.num_appliances, 1
        )
        refined_power_raw = base_power_raw + power_gate * power_delta_raw
        refined_state_logits = (
            base_state_logits + state_gate * state_delta_logits
        )
        return {
            "past_refined_power_raw": refined_power_raw,
            "past_refined_power": F.softplus(refined_power_raw),
            "past_refined_state_logits": refined_state_logits,
            "past_refined_p_on": torch.sigmoid(refined_state_logits),
            "past_refiner_power_delta_raw": power_delta_raw,
            "past_refiner_state_delta_logits": state_delta_logits,
            "past_refiner_power_gate": power_gate,
            "past_refiner_state_gate": state_gate,
        }


class _CausalTCNResidualBlock(nn.Module):
    """Two-layer causal residual block used by one appliance only."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        self.padding = (int(kernel_size) - 1) * int(dilation)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
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

    def _chomp(self, value: Tensor) -> Tensor:
        return value[..., :-self.padding] if self.padding else value

    def forward(self, value: Tensor) -> Tensor:
        residual = self.residual(value)
        hidden = self._chomp(self.conv1(value))
        hidden = self.dropout(F.gelu(self.norm1(hidden)))
        hidden = self._chomp(self.conv2(hidden))
        hidden = self.dropout(F.gelu(self.norm2(hidden)))
        return F.gelu(hidden + residual)


class _PerApplianceCausalTCN(nn.Module):
    """Causal temporal encoder with no parameter sharing across appliances."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        self.blocks = nn.ModuleList(
            [
                _CausalTCNResidualBlock(
                    in_channels=input_dim if layer == 0 else hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2**layer,
                    dropout=dropout,
                )
                for layer in range(num_layers)
            ]
        )

    def forward(self, history: Tensor) -> Tensor:
        if history.dim() != 3:
            raise ValueError(
                f"Expected appliance history [B,T,C], got {tuple(history.shape)}."
            )
        hidden = history.transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden[..., -1]


class RefinedHistoryResidualTCN(nn.Module):
    """
    Forecast residuals from deployment-available refined appliance histories.

    Each appliance has its own causal TCN and output head.  There is no temporal
    parameter sharing between appliances.  The causal encoder consumes only
    PISA-reconstructed appliance power/state history.  The optional enhanced
    decoder additionally consumes inference-available future calendar features,
    horizon embeddings and a small adapter over the frozen Transformer context;
    ground-truth appliance history is never consumed at inference time.
    """

    def __init__(
        self,
        num_appliances: int,
        horizon: int,
        rated_power: Tensor,
        hidden_dim: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        gate_init: float = -3.0,
        shared_tcn: bool = False,
        future_time_dim: int = 0,
        encoder_context_dim: int = 0,
        use_future_context: bool = False,
        target_adapter_dim: int = 0,
        base_blend_init: float = 2.0,
        use_target_power_calibration: bool = False,
    ) -> None:
        super().__init__()
        if num_appliances <= 0 or horizon <= 0:
            raise ValueError("num_appliances and horizon must be positive.")
        self.num_appliances = int(num_appliances)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self.shared_tcn_enabled = bool(shared_tcn)
        self.future_time_dim = int(future_time_dim)
        self.encoder_context_dim = int(encoder_context_dim)
        self.use_future_context = bool(use_future_context)
        self.target_adapter_dim = int(target_adapter_dim)
        self.use_target_power_calibration = bool(use_target_power_calibration)
        if self.future_time_dim < 0 or self.encoder_context_dim < 0:
            raise ValueError("future_time_dim and encoder_context_dim must be >= 0.")
        if self.target_adapter_dim < 0:
            raise ValueError("target_adapter_dim must be >= 0.")
        if self.use_future_context and self.future_time_dim <= 0:
            raise ValueError(
                "use_future_context=True requires a positive future_time_dim."
            )
        rated = torch.as_tensor(rated_power, dtype=torch.float32).reshape(
            1, self.num_appliances, 1
        )
        self.register_buffer("rated_power", rated, persistent=True)
        self.shared_tcn = (
            _PerApplianceCausalTCN(
                input_dim=2,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
            )
            if self.shared_tcn_enabled
            else None
        )
        self.appliance_tcns = (
            nn.ModuleList()
            if self.shared_tcn_enabled
            else nn.ModuleList(
                [
                    _PerApplianceCausalTCN(
                        input_dim=2,
                        hidden_dim=self.hidden_dim,
                        num_layers=self.num_layers,
                        kernel_size=self.kernel_size,
                        dropout=self.dropout,
                    )
                    for _ in range(self.num_appliances)
                ]
            )
        )
        # One head per appliance emits raw-power and state-logit residuals.
        self.residual_heads = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, 2 * self.horizon)
                for _ in range(self.num_appliances)
            ]
        )
        self.power_gate_logit = nn.Parameter(
            torch.full((self.num_appliances, 1), float(gate_init))
        )
        self.state_gate_logit = nn.Parameter(
            torch.full((self.num_appliances, 1), float(gate_init))
        )

        # Optional target-transfer decoder.  Unlike the legacy head, which maps
        # one pooled TCN vector to the complete horizon, this route constructs a
        # separate representation for every future step.  Future calendar
        # covariates and a learned horizon position therefore affect the output
        # before the final per-appliance projection.
        if self.use_future_context:
            self.horizon_embedding = nn.Embedding(self.horizon, self.hidden_dim)
            self.future_time_proj = nn.Linear(
                self.future_time_dim, self.hidden_dim, bias=False
            )
            self.encoder_context_adapter = (
                nn.Sequential(
                    nn.LayerNorm(self.encoder_context_dim),
                    nn.Linear(self.encoder_context_dim, self.target_adapter_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.target_adapter_dim, self.hidden_dim),
                )
                if self.target_adapter_dim > 0 and self.encoder_context_dim > 0
                else None
            )
            self.future_step_norm = nn.LayerNorm(self.hidden_dim)
            self.direct_step_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.hidden_dim, 2),
                    )
                    for _ in range(self.num_appliances)
                ]
            )
            self.base_power_blend_logit = nn.Parameter(
                torch.full(
                    (self.num_appliances, self.horizon), float(base_blend_init)
                )
            )
            self.base_state_blend_logit = nn.Parameter(
                torch.full(
                    (self.num_appliances, self.horizon), float(base_blend_init)
                )
            )
            if self.use_target_power_calibration:
                self.target_power_log_scale = nn.Parameter(
                    torch.zeros(self.num_appliances, 1)
                )
                self.target_power_bias = nn.Parameter(
                    torch.zeros(self.num_appliances, 1)
                )
            else:
                self.register_parameter("target_power_log_scale", None)
                self.register_parameter("target_power_bias", None)
        else:
            self.horizon_embedding = None
            self.future_time_proj = None
            self.encoder_context_adapter = None
            self.future_step_norm = None
            self.direct_step_heads = nn.ModuleList()
            self.register_parameter("base_power_blend_logit", None)
            self.register_parameter("base_state_blend_logit", None)
            self.register_parameter("target_power_log_scale", None)
            self.register_parameter("target_power_bias", None)
        self.zero_init_output()

    @torch.no_grad()
    def zero_init_output(self) -> None:
        """Make the residual branch exactly neutral at initialization."""
        for head in self.residual_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for head in self.direct_step_heads:
            final = head[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def enhanced_adaptation_modules(self) -> list[nn.Module]:
        """Return the small modules authorized for budget-matched transfer."""
        if not self.use_future_context:
            return []
        modules: list[nn.Module] = [
            self.horizon_embedding,
            self.future_time_proj,
            self.future_step_norm,
            self.direct_step_heads,
        ]
        if self.encoder_context_adapter is not None:
            modules.append(self.encoder_context_adapter)
        return modules

    def forward(
        self,
        past_power: Tensor,
        past_p_on: Tensor,
        future_time: Optional[Tensor] = None,
        encoder_context: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        if past_power.shape != past_p_on.shape or past_power.dim() != 3:
            raise ValueError(
                "past_power and past_p_on must share shape [B,A,T], got "
                f"{tuple(past_power.shape)} and {tuple(past_p_on.shape)}."
            )
        if past_power.size(1) != self.num_appliances:
            raise ValueError(
                f"Expected {self.num_appliances} appliances, got {past_power.size(1)}."
            )
        normalized_power = past_power / self.rated_power.clamp_min(1e-6)
        normalized_power = normalized_power.clamp(min=0.0, max=2.0)
        residuals: list[Tensor] = []
        contexts: list[Tensor] = []
        for appliance_index, head in enumerate(self.residual_heads):
            tcn = (
                self.shared_tcn
                if self.shared_tcn is not None
                else self.appliance_tcns[appliance_index]
            )
            history = torch.stack(
                [
                    normalized_power[:, appliance_index],
                    past_p_on[:, appliance_index].clamp(0.0, 1.0),
                ],
                dim=-1,
            )
            context = tcn(history)
            contexts.append(context)
            residuals.append(head(context))
        decoded = torch.stack(residuals, dim=1)
        power_residual_raw = decoded[..., : self.horizon]
        state_residual_logits = decoded[..., self.horizon :]
        power_gate = torch.sigmoid(self.power_gate_logit).view(
            1, self.num_appliances, 1
        )
        state_gate = torch.sigmoid(self.state_gate_logit).view(
            1, self.num_appliances, 1
        )
        result = {
            "future_tcn_power_residual_raw": power_residual_raw,
            "future_tcn_state_residual_logits": state_residual_logits,
            "future_tcn_power_gate": power_gate,
            "future_tcn_state_gate": state_gate,
        }
        if not self.use_future_context:
            return result

        batch_size = past_power.size(0)
        if future_time is None or future_time.shape != (
            batch_size,
            self.horizon,
            self.future_time_dim,
        ):
            actual = None if future_time is None else tuple(future_time.shape)
            raise ValueError(
                "Enhanced residual TCN requires x_future_time with shape "
                f"[{batch_size},{self.horizon},{self.future_time_dim}], got {actual}."
            )
        step_ids = torch.arange(self.horizon, device=past_power.device)
        step_features = self.horizon_embedding(step_ids).unsqueeze(0)
        step_features = step_features + self.future_time_proj(future_time)
        if self.encoder_context_adapter is not None:
            if encoder_context is None or encoder_context.shape != (
                batch_size,
                self.encoder_context_dim,
            ):
                actual = (
                    None if encoder_context is None else tuple(encoder_context.shape)
                )
                raise ValueError(
                    "Enhanced residual TCN requires pooled encoder context with "
                    f"shape [{batch_size},{self.encoder_context_dim}], got {actual}."
                )
            step_features = step_features + self.encoder_context_adapter(
                encoder_context
            ).unsqueeze(1)

        direct_outputs: list[Tensor] = []
        for appliance_index, head in enumerate(self.direct_step_heads):
            features = self.future_step_norm(
                contexts[appliance_index].unsqueeze(1) + step_features
            )
            direct_outputs.append(head(features))
        direct_decoded = torch.stack(direct_outputs, dim=1)
        power_delta_raw = direct_decoded[..., 0]
        state_delta_logits = direct_decoded[..., 1]

        # A persistence anchor gives the newly introduced direct route a
        # meaningful, source-independent initialization.  It does not use the
        # frozen future decoder and can move freely during target adaptation.
        last_power = past_power[..., -1:].clamp_min(1e-6)
        persistence_raw = torch.log(torch.expm1(last_power).clamp_min(1e-8))
        last_state_logits = torch.logit(
            past_p_on[..., -1:].clamp(1e-4, 1.0 - 1e-4)
        )
        direct_power_raw = persistence_raw + power_delta_raw
        direct_state_logits = last_state_logits + state_delta_logits
        result.update(
            {
                "future_tcn_direct_power_raw": direct_power_raw,
                "future_tcn_direct_state_logits": direct_state_logits,
                "future_tcn_base_power_blend": torch.sigmoid(
                    self.base_power_blend_logit
                ).unsqueeze(0),
                "future_tcn_base_state_blend": torch.sigmoid(
                    self.base_state_blend_logit
                ).unsqueeze(0),
            }
        )
        if self.use_target_power_calibration:
            result.update(
                {
                    "future_tcn_target_power_scale": torch.exp(
                        self.target_power_log_scale
                    ).view(1, self.num_appliances, 1),
                    "future_tcn_target_power_bias": self.target_power_bias.view(
                        1, self.num_appliances, 1
                    ),
                }
            )
        return result


class PISAModel(nn.Module):
    """
    PISA main model.

    Full pipeline:
        x_hist
            -> MultiScaleTemporalEncoder
            -> optional HouseholdFiLM
            -> ApplianceQueryStateBridge
            -> TypeAwareApplianceDecoder
            -> Bridge residual fusion
            -> PhysicsGuidedConstraintLayer

    Input:
        batch or tensor.

        If tensor:
            x_hist: [B, T_in, C]

        If dict:
            batch["x_hist"]: [B, T_in, C]
            batch["home_id"]: [B], optional

    Output:
        Dictionary containing:
            y_power:          [B, A, H]
            y_raw:            [B, A, H]
            decoder_raw:      [B, A, H]
            bridge_power:     [B, A, H]
            bridge_power_raw: [B, A, H]
            state_logits:     [B, A, H]
            event_logits:     [B, A, H]
            p_on:             [B, A, H]
            z_app:            [B, A, D]
            attn_weights:     [B, A, T_in]
            gate_weights:     [B, A, 3]
            h_enc:            [B, T_in, D]
    """

    def __init__(
        self,
        input_dim: int,
        num_appliances: int = 4,
        horizon: int = 30,
        appliance_names: Optional[list[str]] = None,
        appliance_types: Optional[dict[str, str]] = None,
        d_model: int = 128,
        n_heads: int = 4,
        encoder_layers: int = 3,
        encoder_ff_dim: int = 256,
        bridge_hidden_dim: int = 256,
        decoder_hidden_dim: int = 256,
        dropout: float = 0.1,
        max_len: int = 4096,
        kernels: tuple[int, ...] = (3, 5, 9, 15),
        future_time_dim: Optional[int] = None,
        rated_power: Optional[Tensor] = None,
        off_threshold: float = 0.5,
        eps_s: float = 1e-4,
        hard_eval_gate: bool = False,
        use_household_film: bool = False,
        num_homes: Optional[int] = None,
        home_emb_dim: int = 16,
        use_type_embedding: bool = True,
        use_bridge_residual: bool = True,
        use_tail_disaggregation_head: bool = False,
        sparse_appliance_names: Optional[list[str]] = None,
        state_conditioned_power: bool = True,
        state_gate_floor: float = 0.05,
        state_power_blend: float = 1.0,
        event_bucket_size: int = 5,
        hierarchical_future: bool = False,
        hierarchical_power_blend: float = 1.0,
        conditional_peak_power: Optional[Tensor] = None,
        risk_adapter_dim: int = 0,
        power_adapter_dim: int = 0,
        use_history_recon_refiner: bool = False,
        history_recon_gate_init: float = -3.0,
        use_refined_history_residual_tcn: bool = False,
        residual_tcn_hidden_dim: int = 128,
        residual_tcn_num_layers: int = 4,
        residual_tcn_kernel_size: int = 3,
        residual_tcn_dropout: float = 0.1,
        future_residual_gate_init: float = -3.0,
        residual_tcn_history_source: str = "refined",
        residual_tcn_shared: bool = False,
        future_residual_fusion: str = "gated_residual",
        future_residual_zero_init: bool = True,
        residual_tcn_input_scale: Optional[Tensor] = None,
        residual_tcn_use_future_context: bool = False,
        residual_tcn_target_adapter_dim: int = 0,
        future_base_blend_init: float = 2.0,
        target_output_calibration: bool = False,
    ):
        super().__init__()

        if appliance_names is None:
            appliance_names = DEFAULT_APPLIANCE_NAMES[:num_appliances]

        if len(appliance_names) != num_appliances:
            raise ValueError(
                f"len(appliance_names) should equal num_appliances. "
                f"Got {len(appliance_names)} and {num_appliances}."
            )

        rated_power = rated_power_tensor_for_appliances(
            appliance_names=appliance_names,
            rated_power_kw=rated_power,
        )

        self.input_dim = input_dim
        self.num_appliances = num_appliances
        self.horizon = horizon
        self.appliance_names = appliance_names
        self.d_model = d_model
        self.future_time_dim = (
            max(input_dim - 1, 0) if future_time_dim is None else int(future_time_dim)
        )
        self.use_household_film = use_household_film
        self.use_bridge_residual = use_bridge_residual
        self.use_tail_disaggregation_head = use_tail_disaggregation_head
        self.state_conditioned_power = bool(state_conditioned_power)
        self.state_gate_floor = float(state_gate_floor)
        self.state_power_blend = float(state_power_blend)
        self.event_bucket_size = max(1, int(event_bucket_size))
        self.hierarchical_future = bool(hierarchical_future)
        self.risk_adapter_dim = int(risk_adapter_dim)
        self.power_adapter_dim = int(power_adapter_dim)
        self.use_history_recon_refiner = bool(use_history_recon_refiner)
        self.use_refined_history_residual_tcn = bool(
            use_refined_history_residual_tcn
        )
        self.residual_tcn_history_source = str(residual_tcn_history_source)
        self.residual_tcn_shared = bool(residual_tcn_shared)
        self.future_residual_fusion = str(future_residual_fusion)
        self.future_residual_zero_init = bool(future_residual_zero_init)
        self.residual_tcn_use_future_context = bool(
            residual_tcn_use_future_context
        )
        self.residual_tcn_target_adapter_dim = int(
            residual_tcn_target_adapter_dim
        )
        self.target_output_calibration = bool(target_output_calibration)
        self._history_reconstruction_only_training = False
        self._future_residual_only_training = False
        self._history_forecast_heads_only_training = False
        self._enhanced_forecast_heads_only_training = False
        self._risk_heads_only_training = False
        if self.risk_adapter_dim < 0:
            raise ValueError("risk_adapter_dim must be >= 0.")
        if self.power_adapter_dim < 0:
            raise ValueError("power_adapter_dim must be >= 0.")
        if self.residual_tcn_history_source not in {"base", "refined"}:
            raise ValueError(
                "residual_tcn_history_source must be 'base' or 'refined'."
            )
        if self.residual_tcn_target_adapter_dim < 0:
            raise ValueError("residual_tcn_target_adapter_dim must be >= 0.")
        if self.future_residual_fusion not in {
            "gated_residual",
            "direct",
            "learned_blend",
        }:
            raise ValueError(
                "future_residual_fusion must be 'gated_residual', 'direct' "
                "or 'learned_blend'."
            )
        if self.future_residual_fusion == "learned_blend" and not (
            self.residual_tcn_use_future_context
        ):
            raise ValueError(
                "future_residual_fusion='learned_blend' requires "
                "residual_tcn_use_future_context=True."
            )
        if self.target_output_calibration and not self.residual_tcn_use_future_context:
            raise ValueError(
                "target_output_calibration requires "
                "residual_tcn_use_future_context=True."
            )
        if (
            self.use_refined_history_residual_tcn
            and self.residual_tcn_history_source == "refined"
            and not self.use_history_recon_refiner
        ):
            raise ValueError(
                "RefinedHistoryResidualTCN with history_source='refined' "
                "requires use_history_recon_refiner=True."
            )
        # The hierarchical route produces risk/scenario outputs, while the
        # refined-history residual TCN remains the final deterministic power
        # route. This allows risk heads to be fine-tuned without replacing or
        # perturbing the selected residual-TCN forecast.

        if not 0.0 <= float(hierarchical_power_blend) <= 1.0:
            raise ValueError("hierarchical_power_blend should be in [0, 1].")
        # This is a buffer rather than a Python float so the checkpoint records
        # the exact blend that produced a selected validation model.
        self.register_buffer(
            "hierarchical_power_blend",
            torch.tensor(float(hierarchical_power_blend), dtype=torch.float32),
            persistent=True,
        )

        if appliance_types is None:
            appliance_types = DEFAULT_APPLIANCE_TYPES
        appliance_type_names = [
            appliance_types.get(name, name) for name in appliance_names
        ]
        pulse_mask = torch.tensor(
            [
                1.0 if type_name in {"thermal_burst", "sparse_event"} else 0.0
                for type_name in appliance_type_names
            ],
            dtype=torch.float32,
        )
        structured_mask = torch.tensor(
            [
                1.0
                if type_name in {"thermal_burst", "sparse_event", "multi_stage_cycle"}
                else 0.0
                for type_name in appliance_type_names
            ],
            dtype=torch.float32,
        )
        self.register_buffer(
            "pulse_appliance_mask", pulse_mask.view(1, num_appliances, 1), persistent=True
        )
        self.register_buffer(
            "structured_appliance_mask",
            structured_mask.view(1, num_appliances, 1),
            persistent=True,
        )

        if conditional_peak_power is None:
            conditional_peak_power = torch.ones(num_appliances, dtype=torch.float32)
        conditional_peak_power = torch.as_tensor(
            conditional_peak_power,
            dtype=torch.float32,
        ).view(-1)
        if conditional_peak_power.numel() != num_appliances:
            raise ValueError(
                "conditional_peak_power should contain one value per appliance."
            )
        self.register_buffer(
            "conditional_peak_power",
            conditional_peak_power.clamp_min(1e-4).view(1, num_appliances, 1),
            persistent=True,
        )

        if not 0.0 <= self.state_gate_floor <= 1.0:
            raise ValueError("state_gate_floor should be in [0, 1].")
        if not 0.0 <= self.state_power_blend <= 1.0:
            raise ValueError("state_power_blend should be in [0, 1].")

        if sparse_appliance_names is None:
            sparse_appliance_names = [
                "dishwasher1",
                "waterheater1",
                "microwave1",
            ]

        sparse_appliance_mask = torch.tensor(
            [1.0 if name in sparse_appliance_names else 0.0 for name in appliance_names],
            dtype=torch.float32,
        )
        self.register_buffer(
            "sparse_appliance_mask",
            sparse_appliance_mask,
            persistent=True,
        )

        appliance_type_ids, type_to_id = build_appliance_type_ids(
            appliance_names=appliance_names,
            appliance_types=appliance_types,
        )

        self.type_to_id = type_to_id
        self.num_appliance_types = len(type_to_id)

        self.register_buffer(
            "appliance_type_ids",
            appliance_type_ids,
            persistent=True,
        )

        self.encoder = MultiScaleTemporalEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=encoder_layers,
            dim_feedforward=encoder_ff_dim,
            dropout=dropout,
            kernels=kernels,
            max_len=max_len,
        )

        if use_household_film:
            if num_homes is None:
                raise ValueError(
                    "num_homes should be provided when use_household_film=True."
                )

            self.household_film = HouseholdFiLM(
                num_homes=num_homes,
                d_model=d_model,
                home_emb_dim=home_emb_dim,
            )
        else:
            self.household_film = None

        self.bridge = ApplianceQueryStateBridge(
            num_appliances=num_appliances,
            d_model=d_model,
            horizon=horizon,
            n_heads=n_heads,
            hidden_dim=bridge_hidden_dim,
            dropout=dropout,
            num_appliance_types=self.num_appliance_types,
            appliance_type_ids=self.appliance_type_ids,
            use_type_embedding=use_type_embedding,
        )

        self.decoder = TypeAwareApplianceDecoder(
            num_appliances=num_appliances,
            d_model=d_model,
            horizon=horizon,
            hidden_dim=decoder_hidden_dim,
            dropout=dropout,
            num_appliance_types=self.num_appliance_types,
            appliance_type_ids=self.appliance_type_ids,
            n_heads=n_heads,
            sparse_appliance_mask=self.sparse_appliance_mask,
            future_time_dim=self.future_time_dim,
            event_bucket_size=self.event_bucket_size,
            appliance_type_names=appliance_type_names,
        )

        # The base power route may adapt the frozen appliance representation
        # during cross-home few-shot calibration without unfreezing the shared
        # encoder, bridge or temporal decoder.  Zero-init keeps a source
        # checkpoint's zero-shot output exactly unchanged at initialization.
        self.power_adapters = nn.ModuleList()
        if self.power_adapter_dim > 0:
            self.power_adapters = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(d_model),
                        nn.Linear(d_model, self.power_adapter_dim),
                        nn.GELU(),
                        nn.Linear(self.power_adapter_dim, d_model),
                    )
                    for _ in range(num_appliances)
                ]
            )

        # The risk route may adapt the frozen appliance representation without
        # entering the base decoder/power computation.  Each appliance gets a
        # small residual bottleneck so sparse loads do not have to compete for
        # a shared full-model update during risk fine-tuning.
        self.risk_adapters = nn.ModuleList()
        if self.risk_adapter_dim > 0:
            self.risk_adapters = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(d_model),
                        nn.Linear(d_model, self.risk_adapter_dim),
                        nn.GELU(),
                        nn.Linear(self.risk_adapter_dim, d_model),
                    )
                    for _ in range(num_appliances)
                ]
            )

        if use_tail_disaggregation_head:
            self.tail_disagg_head = TailTemporalDisaggregationHead(
                num_appliances=num_appliances,
                d_model=d_model,
                horizon=horizon,
                hidden_dim=decoder_hidden_dim,
                dropout=dropout,
                num_appliance_types=self.num_appliance_types,
                appliance_type_ids=self.appliance_type_ids,
            )
            self.tail_global_fusion_logit = nn.Parameter(
                torch.full((num_appliances, 1), -2.0)
            )
        else:
            self.tail_disagg_head = None
            self.tail_global_fusion_logit = None

        self.past_recon_head = PastApplianceReconstructionHead(
            d_model=d_model,
            num_appliances=num_appliances,
            dropout=dropout,
        )
        self.history_recon_refiner = (
            PastApplianceReconstructionResidualRefiner(
                d_model=d_model,
                num_appliances=num_appliances,
                dropout=dropout,
                gate_init=history_recon_gate_init,
            )
            if self.use_history_recon_refiner
            else None
        )
        self.refined_history_residual_tcn = (
            RefinedHistoryResidualTCN(
                num_appliances=num_appliances,
                horizon=horizon,
                rated_power=(
                    rated_power
                    if residual_tcn_input_scale is None
                    else residual_tcn_input_scale
                ),
                hidden_dim=residual_tcn_hidden_dim,
                num_layers=residual_tcn_num_layers,
                kernel_size=residual_tcn_kernel_size,
                dropout=residual_tcn_dropout,
                gate_init=future_residual_gate_init,
                shared_tcn=self.residual_tcn_shared,
                future_time_dim=self.future_time_dim,
                encoder_context_dim=self.d_model,
                use_future_context=self.residual_tcn_use_future_context,
                target_adapter_dim=self.residual_tcn_target_adapter_dim,
                base_blend_init=future_base_blend_init,
                use_target_power_calibration=self.target_output_calibration,
            )
            if self.use_refined_history_residual_tcn
            else None
        )

        self.recon_summary_proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            )

        self.constraint = PhysicsGuidedConstraintLayer(
            num_appliances=num_appliances,
            rated_power=rated_power,
            off_threshold=off_threshold,
            eps_s=eps_s,
            hard_eval_gate=hard_eval_gate,
        )

        # Per-appliance learnable residual weight.
        # It gives the bridge a direct path to final prediction,
        # which helps reduce encoder-decoder gradient weakening.
        self.bridge_fusion_logit = nn.Parameter(
            torch.full((num_appliances, 1), -6.0)
        )

        self._init_parameters()
        if self.history_recon_refiner is not None:
            self.history_recon_refiner.zero_init_output()
        if (
            self.refined_history_residual_tcn is not None
            and self.future_residual_zero_init
        ):
            self.refined_history_residual_tcn.zero_init_output()
        # Start from the exact frozen decoder representation.  The final
        # adapter projection receives gradients first, then opens a controlled
        # residual feature path for the risk heads without perturbing y_power.
        for adapter in [*self.power_adapters, *self.risk_adapters]:
            final_projection = adapter[-1]
            if isinstance(final_projection, nn.Linear):
                nn.init.zeros_(final_projection.weight)
                nn.init.zeros_(final_projection.bias)

    def train(self, mode: bool = True) -> "PISAModel":
        """
        Keep frozen forecasting modules deterministic in fine-tuning modes.

        ``nn.Module.train`` normally enables dropout in frozen modules.  During
        history-refiner, residual-TCN or risk-head-only training that would
        make the supposedly fixed routes drift from batch to batch and would
        also update frozen BatchNorm running statistics.
        """
        super().train(mode)
        if (
            mode
            and self._history_reconstruction_only_training
            and self.history_recon_refiner is not None
        ):
            for child in self.children():
                child.eval()
            self.history_recon_refiner.train(True)
        elif (
            mode
            and self._future_residual_only_training
            and self.refined_history_residual_tcn is not None
        ):
            for child in self.children():
                child.eval()
            self.refined_history_residual_tcn.train(True)
        elif (
            mode
            and self._history_forecast_heads_only_training
            and self.refined_history_residual_tcn is not None
        ):
            # Output-head-only transfer: keep every frozen temporal feature
            # extractor deterministic. Linear heads remain differentiable in
            # eval mode, but marking them train mode documents the intended
            # adaptation scope and supports future heads with dropout.
            for child in self.children():
                child.eval()
            self.past_recon_head.power_head.train(True)
            self.past_recon_head.state_head.train(True)
            if self.history_recon_refiner is not None and any(
                parameter.requires_grad
                for parameter in self.history_recon_refiner.parameters()
            ):
                self.history_recon_refiner.train(True)
            for head in self.refined_history_residual_tcn.residual_heads:
                head.train(True)
        elif (
            mode
            and self._enhanced_forecast_heads_only_training
            and self.refined_history_residual_tcn is not None
        ):
            # Parameter-matched target transfer: the source Transformer and
            # appliance-history TCN remain deterministic.  Only the small
            # future-step decoder, Transformer-context adapter, blending and
            # per-appliance calibration parameters are optimized.
            for child in self.children():
                child.eval()
            for module in (
                self.refined_history_residual_tcn.enhanced_adaptation_modules()
            ):
                module.train(True)
        elif mode and self._risk_heads_only_training:
            # ``requires_grad=False`` alone does not freeze BatchNorm running
            # statistics or dropout behavior. Keep the complete forecasting
            # route deterministic, then selectively re-enable only the risk
            # modules that were authorized by ``train_mode=risk_heads``.
            for child in self.children():
                child.eval()
            for appliance_decoder in self.decoder.appliance_decoders:
                for name in RISK_HEAD_MODULE_NAMES:
                    module = getattr(appliance_decoder, name, None)
                    if module is not None:
                        module.train(True)
            self.risk_adapters.train(True)
        return self

    def _init_parameters(self) -> None:
        for name, param in self.named_parameters():
            if param.dim() <= 1:
                continue

            if (
                "fusion_logit" in name
                or "blend_logit" in name
                or "gate_logit" in name
                or "target_power_log_scale" in name
                or "target_power_bias" in name
                or "template_logits" in name
            ):
                continue
            if "appliance_queries" in name:
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif "embedding" in name:
                nn.init.normal_(param, mean=0.0, std=0.02)
            else:
                nn.init.xavier_uniform_(param)

    @torch.no_grad()
    def set_state_prior(self, prior_on_prob: Tensor) -> None:
        """
        Set empirical ON-state prior for state prediction head.

        Args:
            prior_on_prob:
                Tensor with shape [A] or [A, H].
        """
        self.bridge.set_state_prior(prior_on_prob)

    @torch.no_grad()
    def set_hierarchical_power_blend(self, blend: float) -> None:
        """Set the structured-appliance hierarchy blend used for ``y_power``."""
        blend = float(blend)
        if not 0.0 <= blend <= 1.0:
            raise ValueError("hierarchical power blend should be in [0, 1].")
        self.hierarchical_power_blend.fill_(blend)

    def get_hierarchical_power_blend(self) -> float:
        """Return the current structured-appliance hierarchy blend."""
        return float(self.hierarchical_power_blend.detach().cpu().item())

    @property
    def has_risk_adapter(self) -> bool:
        """Whether the risk-only residual representation is enabled."""
        return len(self.risk_adapters) > 0

    @property
    def has_power_adapter(self) -> bool:
        """Whether the core power route has a few-shot residual adapter."""
        return len(self.power_adapters) > 0

    def _power_adapter_features(self, z_app: Tensor) -> Tensor:
        """Return per-appliance residual features for the core power route."""
        if not self.has_power_adapter:
            return z_app
        residual = torch.stack(
            [
                adapter(z_app[:, app_idx, :])
                for app_idx, adapter in enumerate(self.power_adapters)
            ],
            dim=1,
        )
        return z_app + residual

    def _risk_adapter_features(self, z_app: Tensor) -> Tensor:
        """Return per-appliance residual features for risk heads only."""
        if not self.has_risk_adapter:
            return z_app
        residual = torch.stack(
            [
                adapter(z_app[:, app_idx, :])
                for app_idx, adapter in enumerate(self.risk_adapters)
            ],
            dim=1,
        )
        return z_app + residual

    def _extract_inputs(
        self,
        batch_or_x: Tensor | dict[str, Tensor],
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        if isinstance(batch_or_x, dict):
            if "x_hist" not in batch_or_x:
                raise KeyError("batch should contain key 'x_hist'.")

            x_hist = batch_or_x["x_hist"]
            home_id = batch_or_x.get("home_id", None)
            x_future_time = batch_or_x.get("x_future_time", None)
            event_prev_hist_index = batch_or_x.get("event_prev_hist_index", None)
        else:
            x_hist = batch_or_x
            home_id = None
            x_future_time = None
            event_prev_hist_index = None

        if x_hist.dim() != 3:
            raise ValueError(
                f"x_hist should have shape [B, T_in, C], got {x_hist.shape}"
            )

        if x_hist.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, "
                f"but got x_hist.size(-1)={x_hist.size(-1)}."
            )

        if x_future_time is not None:
            if x_future_time.dim() != 3:
                raise ValueError(
                    "x_future_time should have shape [B, H, F], "
                    f"got {x_future_time.shape}"
                )
            if x_future_time.size(0) != x_hist.size(0):
                raise ValueError(
                    "x_future_time batch size should match x_hist, "
                    f"got {x_future_time.size(0)} and {x_hist.size(0)}"
                )
            if x_future_time.size(1) != self.horizon:
                raise ValueError(
                    f"x_future_time horizon should be {self.horizon}, "
                    f"got {x_future_time.size(1)}"
                )
            if x_future_time.size(2) != self.future_time_dim:
                raise ValueError(
                    f"x_future_time dim should be {self.future_time_dim}, "
                    f"got {x_future_time.size(2)}"
                )

        if event_prev_hist_index is not None:
            if event_prev_hist_index.dim() == 0:
                event_prev_hist_index = event_prev_hist_index.expand(x_hist.size(0))
            if (
                event_prev_hist_index.dim() != 1
                or event_prev_hist_index.size(0) != x_hist.size(0)
            ):
                raise ValueError(
                    "event_prev_hist_index should have shape [B], "
                    f"got {event_prev_hist_index.shape}."
                )

        return x_hist, home_id, x_future_time, event_prev_hist_index

    def _bucket_logits_to_minute_logits(
        self,
        bucket_logits: Tensor,
        offset_logits: Tensor,
    ) -> Tensor:
        """Expand bucket occurrence risk into a normalized minute distribution."""
        if bucket_logits.dim() != 3 or offset_logits.dim() != 4:
            raise ValueError(
                "bucket_logits should be [B, A, N] and offset_logits [B, A, N, K]."
            )
        if bucket_logits.shape != offset_logits.shape[:-1]:
            raise ValueError(
                "bucket and offset logits should share [B, A, N] dimensions."
            )

        minute_prob = torch.sigmoid(bucket_logits).unsqueeze(-1) * torch.softmax(
            offset_logits,
            dim=-1,
        )
        minute_prob = minute_prob.flatten(start_dim=2)[..., : self.horizon]
        return torch.logit(minute_prob.clamp(1e-4, 1.0 - 1e-4))

    def forward(
        self,
        batch_or_x: Tensor | dict[str, Tensor],
    ) -> dict[str, Tensor]:
        x_hist, home_id, x_future_time, event_prev_hist_index = self._extract_inputs(
            batch_or_x
        )

        h_enc = self.encoder(x_hist)
        

        if self.household_film is not None:
            h_enc = self.household_film(h_enc, home_id)
        
        past_recon_out = self.past_recon_head(h_enc)
        past_refined_out = None
        if self.history_recon_refiner is not None:
            past_refined_out = self.history_recon_refiner(
                h_enc=h_enc,
                base_power_raw=past_recon_out["past_power_raw"],
                base_state_logits=past_recon_out["past_state_logits"],
            )
        
        bridge_out = self.bridge(h_enc)
        z_app = bridge_out["z_app"]

        past_p_on = past_recon_out["past_p_on"]       # [B, A, T]
        past_power = past_recon_out["past_power"]     # [B, A, T]
        last_p_on = past_p_on[:, :, -1:]              # [B, A, 1]
        mean_p_on_30 = past_p_on[:, :, -30:].mean(dim=-1, keepdim=True)

        last_power = past_power[:, :, -1:]            # [B, A, 1]
        mean_power_30 = past_power[:, :, -30:].mean(dim=-1, keepdim=True)
        
        recon_summary = torch.cat(
            [last_p_on, mean_p_on_30, last_power, mean_power_30],
            dim=-1,
            )                                             # [B, A, 4]
        
        recon_summary_emb = self.recon_summary_proj(recon_summary)  # [B, A, D]
        
        z_app = z_app + recon_summary_emb

        power_z_app = self._power_adapter_features(z_app)
        decoder_out = self.decoder(
            power_z_app,
            h_enc,
            future_time=x_future_time,
            persistence_power=last_power,
        )
        # The base decoder output remains the only route into core power/state
        # prediction.  A second frozen-decoder pass over the adapter residual
        # is used exclusively by event/hierarchy outputs below.
        risk_decoder_out = decoder_out
        risk_z_app = z_app
        if self.has_risk_adapter:
            risk_z_app = self._risk_adapter_features(power_z_app)
            risk_decoder_out = self.decoder(
                risk_z_app,
                h_enc,
                future_time=x_future_time,
                persistence_power=last_power,
            )
        decoder_raw = decoder_out["decoder_raw"]
        state_logits = bridge_out["state_logits"] + decoder_out["decoder_state_logits"]
        stop_residual_logits = (
            bridge_out["event_logits"] + risk_decoder_out["decoder_event_logits"]
        )
        start_residual_logits = risk_decoder_out["start_logits"]
        bucket_start_logits = self._bucket_logits_to_minute_logits(
            risk_decoder_out["start_bucket_logits"],
            risk_decoder_out["start_offset_logits"],
        )
        bucket_stop_logits = self._bucket_logits_to_minute_logits(
            risk_decoder_out["stop_bucket_logits"],
            risk_decoder_out["stop_offset_logits"],
        )

        tail_out = None
        if self.tail_disagg_head is not None:
            tail_out = self.tail_disagg_head(h_enc)
            global_weight = torch.sigmoid(self.tail_global_fusion_logit).view(
                1,
                self.num_appliances,
                1,
            )
            tail_power = F.softplus(tail_out["tail_power_raw"])
            decoder_power = tail_power + global_weight * decoder_out["y_raw"]
            decoder_raw = torch.log(torch.expm1(decoder_power).clamp_min(1e-8))
            state_logits = state_logits + tail_out["tail_state_logits"]
            stop_residual_logits = (
                stop_residual_logits + tail_out["tail_event_logits"]
            )

        decoder_power = F.softplus(decoder_raw)

        if self.use_bridge_residual:
            fusion_weight = torch.sigmoid(self.bridge_fusion_logit).view(
                1,
                self.num_appliances,
                1,
            )
            y_power_direct = decoder_power + fusion_weight * bridge_out["bridge_power"]
            y_raw = torch.log(torch.expm1(y_power_direct).clamp_min(1e-8))
        else:
            fusion_weight = torch.zeros(
                1,
                self.num_appliances,
                1,
                device=decoder_raw.device,
                dtype=decoder_raw.dtype,
            )
            y_raw = decoder_raw

        constraint_out = self.constraint(
            y_raw=y_raw,
            state_logits=state_logits,
        )

        uncapped_amplitude_power = F.softplus(y_raw)
        amplitude_power = torch.minimum(
            uncapped_amplitude_power,
            self.constraint.rated_power,
        )

        p_on = constraint_out["p_on"]

        if event_prev_hist_index is None:
            event_prev_p_on = last_p_on
            event_prev_power = last_power
        else:
            event_prev_hist_index = event_prev_hist_index.to(
                device=past_p_on.device,
                dtype=torch.long,
            )
            if torch.any(event_prev_hist_index < 0) or torch.any(
                event_prev_hist_index >= past_p_on.size(-1)
            ):
                raise ValueError(
                    "event_prev_hist_index should refer to a valid position "
                    "inside x_hist."
                )

            gather_index = event_prev_hist_index.view(-1, 1, 1).expand(
                -1,
                self.num_appliances,
                1,
            )
            event_prev_p_on = torch.gather(past_p_on, dim=2, index=gather_index)
            event_prev_power = torch.gather(past_power, dim=2, index=gather_index)

        # Events are directional state transitions.  During deterministic
        # power training, derive them directly from the state trajectory so
        # untrained event-head parameters cannot contaminate the output.  The
        # residual event heads are introduced only by the explicit
        # hierarchical/risk stage.
        event_prev_and_current = torch.cat([event_prev_p_on, p_on[..., :-1]], dim=-1)
        start_base_prob = p_on * (1.0 - event_prev_and_current)
        stop_base_prob = (1.0 - p_on) * event_prev_and_current
        start_base_logits = torch.logit(start_base_prob.clamp(1e-4, 1.0 - 1e-4))
        stop_base_logits = torch.logit(stop_base_prob.clamp(1e-4, 1.0 - 1e-4))
        if self.hierarchical_future:
            start_logits = start_base_logits + start_residual_logits
            stop_logits = stop_base_logits + stop_residual_logits
        else:
            start_logits = start_base_logits
            stop_logits = stop_base_logits

        start_prob = torch.sigmoid(start_logits)
        stop_prob = torch.sigmoid(stop_logits)
        event_prob = start_prob + stop_prob - start_prob * stop_prob
        event_logits = torch.logit(event_prob.clamp(1e-4, 1.0 - 1e-4))

        bucket_start_prob = torch.sigmoid(bucket_start_logits)
        bucket_stop_prob = torch.sigmoid(bucket_stop_logits)
        bucket_event_prob = (
            bucket_start_prob
            + bucket_stop_prob
            - bucket_start_prob * bucket_stop_prob
        )
        bucket_event_logits = torch.logit(
            bucket_event_prob.clamp(1e-4, 1.0 - 1e-4)
        )

        window_start_logits = risk_decoder_out["window_start_logits"]
        window_start_prob = torch.sigmoid(window_start_logits)
        conditional_bucket_prob = torch.softmax(
            risk_decoder_out["start_bucket_logits"],
            dim=-1,
        )
        conditional_offset_prob = torch.softmax(
            risk_decoder_out["start_offset_logits"],
            dim=-1,
        )
        conditional_start_prob = (
            conditional_bucket_prob.unsqueeze(-1) * conditional_offset_prob
        ).flatten(start_dim=2)[..., : self.horizon]
        hierarchical_start_prob = (
            window_start_prob.unsqueeze(-1) * conditional_start_prob
        )
        bucket_start_risk = (
            window_start_prob.unsqueeze(-1) * conditional_bucket_prob
        )
        hierarchical_start_logits = torch.logit(
            hierarchical_start_prob.clamp(1e-4, 1.0 - 1e-4)
        )

        conditional_power = (
            risk_decoder_out["conditional_profile"] * self.conditional_peak_power
        )
        pulse_scenario_power = (
            risk_decoder_out["pulse_scenario_profile"] * self.conditional_peak_power
        )
        conditional_scenario_power = torch.where(
            self.pulse_appliance_mask.bool(),
            pulse_scenario_power,
            conditional_power,
        )

        prob_power = torch.minimum(
            amplitude_power * p_on,
            self.constraint.rated_power,
        )
        soft_state_gate = self.state_gate_floor + (1.0 - self.state_gate_floor) * p_on
        soft_power = torch.minimum(
            amplitude_power * soft_state_gate,
            self.constraint.rated_power,
        )
        state_binary = (p_on >= self.constraint.off_threshold).to(
            dtype=amplitude_power.dtype
        )
        hard_power = torch.minimum(
            amplitude_power * state_binary,
            self.constraint.rated_power,
        )

        if self.state_conditioned_power:
            blend = self.state_power_blend
            forecast_power = (
                (1.0 - blend) * amplitude_power
                + blend * soft_power
            )
        else:
            forecast_power = amplitude_power

        previous_on_prob = event_prev_p_on.clamp(0.0, 1.0)
        expected_start_power = window_start_prob.unsqueeze(-1) * conditional_power
        hierarchical_expected_power = (
            previous_on_prob * soft_power
            + (1.0 - previous_on_prob) * expected_start_power
        )
        previous_on_binary = (previous_on_prob >= 0.5).to(amplitude_power.dtype)
        window_start_binary = (window_start_prob >= 0.5).to(amplitude_power.dtype)
        hierarchical_scenario_power = (
            previous_on_binary * hard_power
            + (1.0 - previous_on_binary)
            * window_start_binary.unsqueeze(-1)
            * conditional_scenario_power
        )

        # Keep the calibrated base forecast as the anchor for risk fine-tuning.
        # The hierarchy can then be introduced continuously for the structured
        # appliances without replacing the base power path at epoch one.
        base_forecast_power = forecast_power
        base_hard_power = hard_power
        expected_power = base_forecast_power
        deterministic_power = base_forecast_power
        hierarchical_state_binary = (
            hierarchical_scenario_power >= (0.05 * self.conditional_peak_power)
        ).to(amplitude_power.dtype)
        if self.hierarchical_future:
            hierarchy_blend = self.hierarchical_power_blend.to(
                dtype=amplitude_power.dtype
            )
            blended_expected_power = (
                (1.0 - hierarchy_blend) * base_forecast_power
                + hierarchy_blend * hierarchical_expected_power
            )
            blended_deterministic_power = (
                (1.0 - hierarchy_blend) * base_forecast_power
                + hierarchy_blend * hierarchical_scenario_power
            )
            blended_hard_power = (
                (1.0 - hierarchy_blend) * base_hard_power
                + hierarchy_blend * hierarchical_scenario_power
            )
            expected_power = torch.where(
                self.structured_appliance_mask.bool(),
                blended_expected_power,
                base_forecast_power,
            )
            deterministic_power = torch.where(
                self.structured_appliance_mask.bool(),
                blended_deterministic_power,
                base_forecast_power,
            )
            # Keep the conditional median/decision separate from the
            # probability-weighted expectation. ``state_binary`` deliberately
            # remains the base state head: risk-head calibration must not
            # silently change the core state evaluation route.
            forecast_power = deterministic_power
            hard_power = torch.where(
                self.structured_appliance_mask.bool(),
                blended_hard_power,
                base_hard_power,
            )

        
        out = {
            "y_power": forecast_power,
            "gated_power": constraint_out["y_power"],
            "p_on": p_on,
            "y_raw": y_raw,
            "amplitude_power": amplitude_power,
            "uncapped_amplitude_power": uncapped_amplitude_power,
            "prob_power": prob_power,
            "soft_power": soft_power,
            "hard_power": hard_power,
            "state_binary": state_binary,
            "state_conditioned_power": soft_power,
            "state_gate": soft_state_gate,
            "decoder_raw": decoder_raw,
            "decoder_power": decoder_out["y_raw"],
            "bridge_power": bridge_out["bridge_power"],
            "bridge_power_raw": bridge_out["bridge_power_raw"],
            "state_logits": state_logits,
            "event_logits": event_logits,
            "start_logits": start_logits,
            "stop_logits": stop_logits,
            "bucket_event_logits": bucket_event_logits,
            "bucket_start_logits": bucket_start_logits,
            "bucket_stop_logits": bucket_stop_logits,
            "window_start_logits": window_start_logits,
            "window_start_prob": window_start_prob,
            "bucket_start_risk": bucket_start_risk,
            "minute_start_risk": hierarchical_start_prob,
            "event_risk_window": window_start_prob,
            "event_risk_bucket": bucket_start_risk,
            "event_risk_minute": hierarchical_start_prob,
            "conditional_bucket_prob": conditional_bucket_prob,
            "conditional_start_prob": conditional_start_prob,
            "hierarchical_start_logits": hierarchical_start_logits,
            "conditional_power": conditional_power,
            "hierarchical_expected_power": hierarchical_expected_power,
            "hierarchical_scenario_power": hierarchical_scenario_power,
            "hierarchical_state_binary": hierarchical_state_binary,
            "hierarchical_power_blend": self.hierarchical_power_blend,
            "deterministic_power": deterministic_power,
            "expected_power": expected_power,
            "pulse_duration_logits": risk_decoder_out["pulse_duration_logits"],
            "pulse_amplitude_fraction": risk_decoder_out["pulse_amplitude_fraction"],
            "pulse_amplitude_power": (
                risk_decoder_out["pulse_amplitude_fraction"].unsqueeze(-1)
                * self.conditional_peak_power
            ).squeeze(-1),
            "pulse_scenario_power": pulse_scenario_power,
            "pulse_appliance_mask": self.pulse_appliance_mask,
            "structured_appliance_mask": self.structured_appliance_mask,
            "start_bucket_logits": risk_decoder_out["start_bucket_logits"],
            "stop_bucket_logits": risk_decoder_out["stop_bucket_logits"],
            "start_offset_logits": risk_decoder_out["start_offset_logits"],
            "stop_offset_logits": risk_decoder_out["stop_offset_logits"],
            "event_prev_p_on": event_prev_p_on,
            "event_prev_power": event_prev_power,
            "event_transition_prob": event_prob,
            "event_residual_logits": stop_residual_logits,
            "start_residual_logits": start_residual_logits,
            "start_prob": start_prob,
            "stop_prob": stop_prob,
            "decoder_event_logits": risk_decoder_out["decoder_event_logits"],
            "decoder_state_logits": decoder_out["decoder_state_logits"],
            "sparse_gate": decoder_out["sparse_gate"],
            "persistence_gate": decoder_out["persistence_gate"],
            "z_app": z_app,
            "risk_z_app": risk_z_app,
            "attn_weights": bridge_out["attn_weights"],
            "gate_weights": decoder_out["gate_weights"],
            "h_enc": h_enc,
            "fusion_weight": fusion_weight.detach(),
        }

        if tail_out is not None:
            out.update(tail_out)
            out["tail_global_weight"] = global_weight.detach()

        if not self.hierarchical_future:
            for key in [
                "window_start_logits",
                "window_start_prob",
                "bucket_start_risk",
                "minute_start_risk",
                "event_risk_window",
                "event_risk_bucket",
                "event_risk_minute",
                "conditional_bucket_prob",
                "conditional_start_prob",
                "hierarchical_start_logits",
                "conditional_power",
                "hierarchical_expected_power",
                "hierarchical_scenario_power",
                "hierarchical_state_binary",
                "hierarchical_power_blend",
                "deterministic_power",
                "expected_power",
                "pulse_duration_logits",
                "pulse_amplitude_fraction",
                "pulse_amplitude_power",
                "pulse_scenario_power",
                "pulse_appliance_mask",
                "structured_appliance_mask",
            ]:
                out.pop(key, None)

        # This is the inference-available historical reconstruction used by
        # downstream consumers such as HEMS. The optional refiner is deliberately
        # excluded from ``recon_summary`` above, so fine-tuning it cannot alter
        # the existing PISA future decoder, risk heads, or their checkpoint
        # behavior. State probability remains a separate input and is never
        # multiplied into power a second time.
        deployment_past_power = (
            past_refined_out["past_refined_power"]
            if past_refined_out is not None
            else past_recon_out["past_power"]
        )
        deployment_past_state_logits = (
            past_refined_out["past_refined_state_logits"]
            if past_refined_out is not None
            else past_recon_out["past_state_logits"]
        )
        out["past_reconstructed_power"] = torch.minimum(
            deployment_past_power,
            self.constraint.rated_power,
        )
        out["past_reconstructed_state_logits"] = deployment_past_state_logits
        out["past_reconstructed_p_on"] = torch.sigmoid(
            deployment_past_state_logits
        )
        if self.refined_history_residual_tcn is not None:
            if self.residual_tcn_history_source == "refined":
                tcn_past_power = out["past_reconstructed_power"]
                tcn_past_state_logits = out[
                    "past_reconstructed_state_logits"
                ]
                tcn_past_p_on = out["past_reconstructed_p_on"]
            else:
                tcn_past_power = torch.minimum(
                    past_recon_out["past_power"],
                    self.constraint.rated_power,
                )
                tcn_past_state_logits = past_recon_out[
                    "past_state_logits"
                ]
                tcn_past_p_on = past_recon_out["past_p_on"]
            residual_out = self.refined_history_residual_tcn(
                past_power=tcn_past_power,
                past_p_on=tcn_past_p_on,
                future_time=x_future_time,
                encoder_context=h_enc.mean(dim=1),
            )
            base_y_raw = out["y_raw"]
            base_state_logits = out["state_logits"]
            direct_forecast_power = None
            legacy_tcn_forecast_power = None
            if self.future_residual_fusion == "gated_residual":
                fused_y_raw = (
                    base_y_raw
                    + residual_out["future_tcn_power_gate"]
                    * residual_out["future_tcn_power_residual_raw"]
                )
                fused_state_logits = (
                    base_state_logits
                    + residual_out["future_tcn_state_gate"]
                    * residual_out["future_tcn_state_residual_logits"]
                )
            elif self.future_residual_fusion == "direct":
                fused_y_raw = residual_out["future_tcn_power_residual_raw"]
                fused_state_logits = residual_out[
                    "future_tcn_state_residual_logits"
                ]
            else:
                # Blend the direct decoder with the complete gated TCN forecast
                # so this optional route retains the source residual model.
                legacy_tcn_y_raw = (
                    base_y_raw
                    + residual_out["future_tcn_power_gate"]
                    * residual_out["future_tcn_power_residual_raw"]
                )
                legacy_tcn_state_logits = (
                    base_state_logits
                    + residual_out["future_tcn_state_gate"]
                    * residual_out["future_tcn_state_residual_logits"]
                )
                legacy_tcn_constraint = self.constraint(
                    y_raw=legacy_tcn_y_raw,
                    state_logits=legacy_tcn_state_logits,
                )
                legacy_tcn_amplitude = torch.minimum(
                    F.softplus(legacy_tcn_y_raw),
                    self.constraint.rated_power,
                )
                legacy_tcn_p_on = legacy_tcn_constraint["p_on"]
                legacy_tcn_soft_gate = (
                    self.state_gate_floor
                    + (1.0 - self.state_gate_floor) * legacy_tcn_p_on
                )
                legacy_tcn_soft_power = torch.minimum(
                    legacy_tcn_amplitude * legacy_tcn_soft_gate,
                    self.constraint.rated_power,
                )
                legacy_tcn_forecast_power = (
                    (1.0 - self.state_power_blend) * legacy_tcn_amplitude
                    + self.state_power_blend * legacy_tcn_soft_power
                    if self.state_conditioned_power
                    else legacy_tcn_amplitude
                )
                base_power_blend = residual_out[
                    "future_tcn_base_power_blend"
                ]
                direct_amplitude = torch.minimum(
                    F.softplus(residual_out["future_tcn_direct_power_raw"]),
                    self.constraint.rated_power,
                )
                direct_soft_gate = (
                    self.state_gate_floor
                    + (1.0 - self.state_gate_floor) * legacy_tcn_p_on
                )
                direct_soft_power = torch.minimum(
                    direct_amplitude * direct_soft_gate,
                    self.constraint.rated_power,
                )
                direct_forecast_power = (
                    (1.0 - self.state_power_blend) * direct_amplitude
                    + self.state_power_blend * direct_soft_power
                    if self.state_conditioned_power
                    else direct_amplitude
                )
                # Blend physical non-negative power amplitudes, not raw
                # softplus coordinates.  Raw-space interpolation caused a
                # nominal 1.8% direct contribution to disproportionately move
                # high-power air-conditioning forecasts.
                fused_amplitude_uncalibrated = (
                    base_power_blend * legacy_tcn_amplitude
                    + (1.0 - base_power_blend)
                    * direct_amplitude
                )
                fused_y_raw = torch.log(
                    torch.expm1(
                        fused_amplitude_uncalibrated.clamp_min(1e-6)
                    ).clamp_min(1e-8)
                )
                # Keep the source state route when blending power forecasts.
                fused_state_logits = legacy_tcn_state_logits
            fused_constraint = self.constraint(
                y_raw=fused_y_raw,
                state_logits=fused_state_logits,
            )
            if self.future_residual_fusion != "learned_blend":
                fused_amplitude_uncalibrated = torch.minimum(
                    F.softplus(fused_y_raw),
                    self.constraint.rated_power,
                )
            if self.target_output_calibration:
                fused_amplitude = torch.clamp(
                    fused_amplitude_uncalibrated
                    * residual_out["future_tcn_target_power_scale"]
                    + residual_out["future_tcn_target_power_bias"],
                    min=0.0,
                )
                fused_amplitude = torch.minimum(
                    fused_amplitude, self.constraint.rated_power
                )
            else:
                fused_amplitude = fused_amplitude_uncalibrated
            fused_p_on = fused_constraint["p_on"]
            fused_prob_power = torch.minimum(
                fused_amplitude * fused_p_on,
                self.constraint.rated_power,
            )
            fused_soft_gate = (
                self.state_gate_floor
                + (1.0 - self.state_gate_floor) * fused_p_on
            )
            fused_soft_power = torch.minimum(
                fused_amplitude * fused_soft_gate,
                self.constraint.rated_power,
            )
            fused_state_binary = (
                fused_p_on >= self.constraint.off_threshold
            ).to(dtype=fused_amplitude.dtype)
            fused_hard_power = torch.minimum(
                fused_amplitude * fused_state_binary,
                self.constraint.rated_power,
            )
            if self.state_conditioned_power:
                fused_forecast_power = (
                    (1.0 - self.state_power_blend) * fused_amplitude
                    + self.state_power_blend * fused_soft_power
                )
            else:
                fused_forecast_power = fused_amplitude

            # Preserve the complete original route for auditing and ablations.
            out["base_y_power"] = out["y_power"]
            out["base_y_raw"] = base_y_raw
            out["base_state_logits"] = base_state_logits
            out["base_p_on"] = out["p_on"]
            out["base_hard_power"] = out["hard_power"]
            out["future_tcn_input_power"] = tcn_past_power
            out["future_tcn_input_state_logits"] = tcn_past_state_logits
            out["future_tcn_input_p_on"] = tcn_past_p_on
            out.update(residual_out)
            out["future_tcn_uncalibrated_amplitude"] = (
                fused_amplitude_uncalibrated
            )
            if direct_forecast_power is not None:
                out["future_tcn_direct_power"] = direct_forecast_power
            if legacy_tcn_forecast_power is not None:
                out["future_tcn_legacy_power"] = legacy_tcn_forecast_power

            # The final route uses the same non-negativity, state and rated-power
            # constraints as PISA. Risk/event heads may be fine-tuned, but their
            # outputs cannot feed back into this deterministic power route.
            out["y_power"] = fused_forecast_power
            out["gated_power"] = fused_prob_power
            out["p_on"] = fused_p_on
            out["y_raw"] = fused_y_raw
            out["amplitude_power"] = fused_amplitude
            out["prob_power"] = fused_prob_power
            out["soft_power"] = fused_soft_power
            out["hard_power"] = fused_hard_power
            out["state_binary"] = fused_state_binary
            out["state_conditioned_power"] = fused_soft_power
            out["state_gate"] = fused_soft_gate
            out["state_logits"] = fused_state_logits

            # The residual TCN changes the final state trajectory.  Recompute
            # directional event probabilities from that final trajectory so
            # power, state and event outputs describe the same forecast.  The
            # previous state comes from the same reconstructed/refined history
            # consumed by the TCN at inference time.
            fused_event_prev_p_on = tcn_past_p_on[..., -1:].clamp(0.0, 1.0)
            fused_prev_and_current = torch.cat(
                [fused_event_prev_p_on, fused_p_on[..., :-1]], dim=-1
            )
            fused_start_base_prob = fused_p_on * (1.0 - fused_prev_and_current)
            fused_stop_base_prob = (1.0 - fused_p_on) * fused_prev_and_current
            fused_start_logits = torch.logit(
                fused_start_base_prob.clamp(1e-4, 1.0 - 1e-4)
            )
            fused_stop_logits = torch.logit(
                fused_stop_base_prob.clamp(1e-4, 1.0 - 1e-4)
            )
            if self.hierarchical_future:
                fused_start_logits = (
                    fused_start_logits + out["start_residual_logits"]
                )
                fused_stop_logits = (
                    fused_stop_logits + out["event_residual_logits"]
                )
            fused_start_prob = torch.sigmoid(fused_start_logits)
            fused_stop_prob = torch.sigmoid(fused_stop_logits)
            fused_event_prob = (
                fused_start_prob
                + fused_stop_prob
                - fused_start_prob * fused_stop_prob
            )
            out["start_logits"] = fused_start_logits
            out["stop_logits"] = fused_stop_logits
            out["event_logits"] = torch.logit(
                fused_event_prob.clamp(1e-4, 1.0 - 1e-4)
            )
            out["start_prob"] = fused_start_prob
            out["stop_prob"] = fused_stop_prob
            out["event_transition_prob"] = fused_event_prob
            out["event_prev_p_on"] = fused_event_prev_p_on
            if self.hierarchical_future:
                # Hierarchical heads remain available through their explicit
                # risk/scenario outputs, but never replace the selected
                # residual-TCN deterministic forecast during risk-only
                # fine-tuning or HEMS evaluation.
                out["deterministic_power"] = fused_forecast_power
                out["expected_power"] = fused_forecast_power
            out.update(residual_out)
        if past_refined_out is not None:
            out.update(past_refined_out)
        out.update(past_recon_out)

        return out


def build_pisa_model_from_config(cfg: dict[str, Any]) -> PISAModel:
    """
    Build PISAModel from a config dictionary.

    Expected minimal config:

    cfg = {
        "data": {
            "appliance_cols": [
                "air1",
                "refrigerator1",
                "dishwasher1",
                "microwave1",
            ],
            "rated_power_kw": [3.0, 0.5, 2.0, 2.5],  # kW
        },
        "appliance_types": {
            "air1": "climate_cyclic",
            "refrigerator1": "low_power_cyclic",
            "dishwasher1": "multi_stage_cycle",
            "microwave1": "sparse_event",
        },
        "model": {
            "input_dim": 6,
            "num_appliances": 4,
            "horizon": 30,
            "d_model": 128,
        }
    }
    """
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    appliance_names = data_cfg.get(
        "appliance_cols",
        DEFAULT_APPLIANCE_NAMES,
    )

    num_appliances = model_cfg.get("num_appliances", len(appliance_names))

    rated_power = data_cfg.get("rated_power_kw", data_cfg.get("rated_power"))
    rated_power = rated_power_tensor_for_appliances(
        appliance_names=appliance_names,
        rated_power_kw=rated_power,
    )
    conditional_peak_power = data_cfg.get("conditional_peak_power", None)
    if conditional_peak_power is not None:
        conditional_peak_power = torch.tensor(
            conditional_peak_power,
            dtype=torch.float32,
        )
    residual_tcn_input_scale = data_cfg.get(
        "nominal_rated_power_kw",
        data_cfg.get("rated_power_kw", data_cfg.get("rated_power")),
    )
    if residual_tcn_input_scale is not None:
        residual_tcn_input_scale = torch.as_tensor(
            residual_tcn_input_scale,
            dtype=torch.float32,
        )

    appliance_types = cfg.get("appliance_types", DEFAULT_APPLIANCE_TYPES)

    input_dim = model_cfg.get("input_dim", None)
    if input_dim is None or input_dim == "auto":
        raise ValueError(
            "model.input_dim must be specified before building the model. "
            "For Home 7951, if using grid + 5 time features, input_dim=6."
        )

    return PISAModel(
        input_dim=int(input_dim),
        num_appliances=int(num_appliances),
        horizon=int(model_cfg.get("horizon", 30)),
        appliance_names=appliance_names,
        appliance_types=appliance_types,
        d_model=int(model_cfg.get("d_model", 128)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        encoder_layers=int(model_cfg.get("encoder_layers", 3)),
        encoder_ff_dim=int(model_cfg.get("encoder_ff_dim", 256)),
        bridge_hidden_dim=int(model_cfg.get("bridge_hidden_dim", 256)),
        decoder_hidden_dim=int(model_cfg.get("decoder_hidden_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        max_len=int(model_cfg.get("max_len", 4096)),
        kernels=tuple(model_cfg.get("kernels", (3, 5, 9, 15))),
        future_time_dim=model_cfg.get("future_time_dim", None),
        rated_power=rated_power,
        off_threshold=float(model_cfg.get("off_threshold", 0.5)),
        eps_s=float(model_cfg.get("eps_s", 1e-4)),
        hard_eval_gate=bool(model_cfg.get("hard_eval_gate", False)),
        use_household_film=bool(model_cfg.get("use_household_film", False)),
        num_homes=model_cfg.get("num_homes", None),
        home_emb_dim=int(model_cfg.get("home_emb_dim", 16)),
        use_type_embedding=bool(model_cfg.get("use_type_embedding", True)),
        use_bridge_residual=bool(model_cfg.get("use_bridge_residual", True)),
        use_tail_disaggregation_head=bool(
            model_cfg.get("use_tail_disaggregation_head", False)
        ),
        state_conditioned_power=bool(
            model_cfg.get("state_conditioned_power", True)
        ),
        state_gate_floor=float(model_cfg.get("state_gate_floor", 0.05)),
        state_power_blend=float(model_cfg.get("state_power_blend", 1.0)),
        event_bucket_size=int(model_cfg.get("event_bucket_size", 5)),
        hierarchical_future=bool(model_cfg.get("hierarchical_future", False)),
        hierarchical_power_blend=float(
            model_cfg.get("hierarchical_power_blend", 1.0)
        ),
        conditional_peak_power=conditional_peak_power,
        risk_adapter_dim=int(model_cfg.get("risk_adapter_dim", 0)),
        power_adapter_dim=int(model_cfg.get("power_adapter_dim", 0)),
        use_history_recon_refiner=bool(
            model_cfg.get("use_history_recon_refiner", False)
        ),
        history_recon_gate_init=float(
            model_cfg.get("history_recon_gate_init", -3.0)
        ),
        use_refined_history_residual_tcn=bool(
            model_cfg.get(
                "use_residual_tcn",
                model_cfg.get("use_refined_history_residual_tcn", False),
            )
        ),
        residual_tcn_hidden_dim=int(
            model_cfg.get("residual_tcn_hidden_dim", 128)
        ),
        residual_tcn_num_layers=int(
            model_cfg.get("residual_tcn_num_layers", 4)
        ),
        residual_tcn_kernel_size=int(
            model_cfg.get("residual_tcn_kernel_size", 3)
        ),
        residual_tcn_dropout=float(
            model_cfg.get("residual_tcn_dropout", 0.1)
        ),
        future_residual_gate_init=float(
            model_cfg.get("future_residual_gate_init", -3.0)
        ),
        residual_tcn_history_source=str(
            model_cfg.get("residual_tcn_history_source", "refined")
        ),
        residual_tcn_shared=bool(
            model_cfg.get("residual_tcn_shared", False)
        ),
        future_residual_fusion=str(
            model_cfg.get("future_residual_fusion", "gated_residual")
        ),
        future_residual_zero_init=bool(
            model_cfg.get("future_residual_zero_init", True)
        ),
        residual_tcn_input_scale=residual_tcn_input_scale,
        residual_tcn_use_future_context=bool(
            model_cfg.get("residual_tcn_use_future_context", False)
        ),
        residual_tcn_target_adapter_dim=int(
            model_cfg.get("residual_tcn_target_adapter_dim", 0)
        ),
        future_base_blend_init=float(
            model_cfg.get("future_base_blend_init", 2.0)
        ),
        target_output_calibration=bool(
            model_cfg.get("target_output_calibration", False)
        ),
    )


def infer_optional_pisa_architecture(
    state_dict: dict[str, Tensor],
) -> dict[str, Any]:
    """Infer optional history/refidual modules from checkpoint tensor keys."""
    result: dict[str, Any] = {
        "use_history_recon_refiner": any(
            str(key).startswith("history_recon_refiner.") for key in state_dict
        ),
        "use_refined_history_residual_tcn": any(
            str(key).startswith("refined_history_residual_tcn.")
            for key in state_dict
        ),
        "residual_tcn_use_future_context": any(
            str(key).startswith(
                "refined_history_residual_tcn.horizon_embedding."
            )
            for key in state_dict
        ),
        "target_output_calibration": any(
            str(key)
            == "refined_history_residual_tcn.target_power_log_scale"
            for key in state_dict
        ),
    }
    prefix = "refined_history_residual_tcn."
    if not result["use_refined_history_residual_tcn"]:
        return result

    shared = any(str(key).startswith(prefix + "shared_tcn.") for key in state_dict)
    result["residual_tcn_shared"] = shared
    tcn_prefix = (
        prefix + "shared_tcn."
        if shared
        else prefix + "appliance_tcns.0."
    )
    first_conv = state_dict.get(tcn_prefix + "blocks.0.conv1.weight")
    first_head = state_dict.get(prefix + "residual_heads.0.weight")
    layer_indices: set[int] = set()
    for key in state_dict:
        text = str(key)
        marker = tcn_prefix + "blocks."
        if not text.startswith(marker):
            continue
        remainder = text[len(marker) :]
        index_text = remainder.split(".", 1)[0]
        if index_text.isdigit():
            layer_indices.add(int(index_text))
    if first_conv is not None:
        result["residual_tcn_hidden_dim"] = int(first_conv.shape[0])
        result["residual_tcn_kernel_size"] = int(first_conv.shape[-1])
    if layer_indices:
        result["residual_tcn_num_layers"] = max(layer_indices) + 1
    if first_head is not None:
        result["residual_tcn_horizon"] = int(first_head.shape[0] // 2)
    adapter_down = state_dict.get(
        prefix + "encoder_context_adapter.1.weight"
    )
    if adapter_down is not None:
        result["residual_tcn_target_adapter_dim"] = int(
            adapter_down.shape[0]
        )
    # Dropout and gate initialization leave no uniquely inferable tensor shape.
    return result


def count_parameters(model: nn.Module) -> dict[str, int]:
    """
    Count model parameters.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
    }


if __name__ == "__main__":
    # Simple shape test.
    cfg = {
        "data": {
            "appliance_cols": [
                "air1",
                "refrigerator1",
                "dishwasher1",
                "microwave1",
            ],
            # Unit: kW. Use loose caps for debugging.
            "rated_power": [3.5, 1.0, 2.5, 2.5],
        },
        "appliance_types": {
            "air1": "climate_cyclic",
            "refrigerator1": "low_power_cyclic",
            "dishwasher1": "multi_stage_cycle",
            "microwave1": "sparse_event",
        },
        "model": {
            "input_dim": 6,
            "num_appliances": 4,
            "horizon": 30,
            "d_model": 128,
            "n_heads": 4,
            "encoder_layers": 3,
            "encoder_ff_dim": 256,
            "bridge_hidden_dim": 256,
            "decoder_hidden_dim": 256,
            "dropout": 0.1,
        },
    }

    model = build_pisa_model_from_config(cfg)

    x = torch.randn(8, 120, 6)
    out = model(x)

    for key, value in out.items():
        if torch.is_tensor(value):
            print(key, tuple(value.shape))

    print(count_parameters(model))
