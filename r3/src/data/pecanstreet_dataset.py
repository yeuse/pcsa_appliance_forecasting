# src/pisa/data/pecanstreet_dataset.py

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from .scaler import ApplianceScaler, PISAScalers, StandardScaler1D
from .state_event import (
    build_event_labels,
    build_state_labels,
    compute_appliance_activity_stats,
    estimate_state_thresholds,
)

SplitName = Literal["train", "val", "test"]


@dataclass
class PecanStreetDataBundle:
    """Shared preprocessing artifacts for train/val/test datasets."""

    csv_path: str
    input_cols: list[str]
    appliance_cols: list[str]
    mains_col: str
    split_col: str
    time_col: str | None
    state_thresholds: dict[str, float]
    scalers: PISAScalers
    appliance_activity_stats: list[Any]

    @property
    def appliance_scales(self) -> np.ndarray:
        return self.scalers.appliance_scaler.scale_.copy()


class PecanStreetSingleHomeDataset(Dataset):
    """Sliding-window dataset for one Pecan Street home.

    This class is designed for the one-minute Austin home files, including:
        data/austin_2018_sep_4homes/home_7951_2018_sep_1min.csv

    Expected columns include:
        split, localminute, grid,
        air1, refrigerator1, dishwasher1, microwave1,
        hour_sin, hour_cos, dow_sin, dow_cos, is_weekend,
        *_available columns.

    Returned batch item:
        x_hist: [T_in, C]
        y_power: [A, H]
        y_power_scaled: [A, H]
        y_state: [A, H]
        y_event: [A, H]
        y_mains: [H]
        target_mask: [A, H]
        timestamp_start / timestamp_end: strings, for debugging/reporting
    """

    def __init__(
        self,
        df: pd.DataFrame,
        split: SplitName,
        bundle: PecanStreetDataBundle,
        input_window: int = 120,
        horizon: int = 30,
        target_mode: Literal["future", "history_tail"] = "future",
        stride: int = 1,
        availability_cols: dict[str, str] | None = None,
        drop_unavailable_windows: bool = True,
        return_timestamps: bool = True,
        partial_train_labels: bool = False,
        require_future_mains: bool = True,
    ):
        super().__init__()

        if input_window <= 0:
            raise ValueError("input_window should be positive.")
        if horizon <= 0:
            raise ValueError("horizon should be positive.")
        if target_mode not in {"future", "history_tail"}:
            raise ValueError(
                "target_mode should be either 'future' or 'history_tail'."
            )
        if target_mode == "history_tail" and horizon > input_window:
            raise ValueError(
                "horizon should be <= input_window when target_mode='history_tail'."
            )
        if stride <= 0:
            raise ValueError("stride should be positive.")

        self.require_future_mains = bool(require_future_mains)
        self.split = split
        self.bundle = bundle
        self.input_window = int(input_window)
        self.horizon = int(horizon)
        self.target_mode = target_mode
        self.stride = int(stride)
        self.availability_cols = availability_cols or {}
        self.drop_unavailable_windows = drop_unavailable_windows
        self.return_timestamps = return_timestamps
        self.partial_train_labels = bool(partial_train_labels and split == "train")
        if self.partial_train_labels and (not drop_unavailable_windows or target_mode != "future"):
            raise ValueError("partial_train_labels requires future targets and strict mains filtering")

        self.df = df[df[bundle.split_col].astype(str).str.lower() == split].copy()
        if self.df.empty:
            raise ValueError(f"No rows found for split='{split}'.")

        self.df = self.df.reset_index(drop=True)
        self._prepare_arrays()
        self.indices = self._build_indices()

        if not self.indices:
            raise ValueError(
                f"No valid sliding windows for split={split}. "
                f"Check input_window={input_window}, horizon={horizon}, stride={stride}."
            )

    def _prepare_arrays(self) -> None:
        cols = list(dict.fromkeys(self.bundle.input_cols + [self.bundle.mains_col] + self.bundle.appliance_cols))
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            raise KeyError(f"Missing required columns in dataframe: {missing}")

        # Numeric cleanup.
        self.df[self.bundle.input_cols] = self.df[self.bundle.input_cols].apply(pd.to_numeric, errors="coerce")
        self.df[self.bundle.mains_col] = pd.to_numeric(self.df[self.bundle.mains_col], errors="coerce")
        self.df[self.bundle.appliance_cols] = self.df[self.bundle.appliance_cols].apply(pd.to_numeric, errors="coerce")

        # Dataport's grid channel is net grid exchange, so negative readings
        # can be PV export rather than household demand. Negative appliance
        # readings are invalid meter values as well. Keep their availability
        # masks at zero before filling numeric arrays; this prevents an invalid
        # reading from entering a usable model window.
        mains_nonnegative = self.df[self.bundle.mains_col].ge(0.0)
        target_nonnegative = self.df[self.bundle.appliance_cols].ge(0.0)

        # Forward/backward fill within the split, then remaining NaNs become 0.
        self.df[cols] = self.df[cols].ffill().bfill().fillna(0.0)

        input_raw = self.df[self.bundle.input_cols].to_numpy(dtype=np.float32)
        self.x = self.bundle.scalers.input_scaler.transform(input_raw)
        self.future_time_indices = [
            idx
            for idx, col in enumerate(self.bundle.input_cols)
            if col != self.bundle.mains_col
        ]

        self.mains = self.df[self.bundle.mains_col].to_numpy(dtype=np.float32)
        self.y_power = self.df[self.bundle.appliance_cols].to_numpy(dtype=np.float32)
        self.y_power = np.clip(self.y_power, a_min=0.0, a_max=None)
        self.y_power_scaled_full = self.bundle.scalers.appliance_scaler.transform(self.y_power)

        self.state = build_state_labels(
            self.y_power,
            thresholds=self.bundle.state_thresholds,
            appliance_cols=self.bundle.appliance_cols,
        )
        self.event = build_event_labels(self.state, mode="both")
        self.start_event = build_event_labels(self.state, mode="start")
        self.stop_event = build_event_labels(self.state, mode="stop")

        self.target_available = np.ones_like(self.y_power, dtype=np.float32)
        for j, app in enumerate(self.bundle.appliance_cols):
            av_col = self.availability_cols.get(app, f"{app}_available")
            if av_col in self.df.columns:
                av = pd.to_numeric(self.df[av_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                self.target_available[:, j] = (av > 0.5).astype(np.float32)
            self.target_available[:, j] *= target_nonnegative[app].to_numpy(dtype=np.float32)

        mains_av_col = self.availability_cols.get(self.bundle.mains_col, f"{self.bundle.mains_col}_available")
        if mains_av_col in self.df.columns:
            mains_av = pd.to_numeric(self.df[mains_av_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            self.mains_available = (mains_av > 0.5).astype(np.float32)
        else:
            self.mains_available = np.ones(len(self.df), dtype=np.float32)
        self.mains_available *= mains_nonnegative.to_numpy(dtype=np.float32)

        if self.bundle.time_col and self.bundle.time_col in self.df.columns:
            self.timestamps = self.df[self.bundle.time_col].astype(str).to_numpy()
        else:
            self.timestamps = np.asarray([str(i) for i in range(len(self.df))])

    def _build_indices(self) -> list[int]:
        n = len(self.df)
        if self.target_mode == "future":
            max_start = n - self.input_window - self.horizon
        else:
            max_start = n - self.input_window

        if max_start < 0:
            return []

        indices: list[int] = []
        for start in range(0, max_start + 1, self.stride):
            hist_slice = slice(start, start + self.input_window)
            if self.target_mode == "future":
                tgt_slice = slice(
                    start + self.input_window,
                    start + self.input_window + self.horizon,
                )
            else:
                tgt_slice = slice(
                    start + self.input_window - self.horizon,
                    start + self.input_window,
                )

            if self.drop_unavailable_windows:
                hist_ok = bool(np.all(self.mains_available[hist_slice] > 0.5))
                tgt_ok = bool(np.all(self.target_available[tgt_slice] > 0.5))
                if self.partial_train_labels:
                    tgt_ok = bool(np.any(self.target_available[tgt_slice] > 0.5))
                # Future mains is used by aggregate-consistency losses and
                # diagnostics, so invalid net-export minutes must be excluded
                # from the forecast interval as well as the input history.
                future_mains_ok = (not self.require_future_mains) or bool(
                    np.all(self.mains_available[tgt_slice] > 0.5)
                )
                if not (hist_ok and tgt_ok and future_mains_ok):
                    continue

            indices.append(start)

        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Tensor | str]:
        start = self.indices[idx]
        hist_start = start
        hist_end = start + self.input_window
        if self.target_mode == "future":
            tgt_start = hist_end
            tgt_end = hist_end + self.horizon
        else:
            tgt_start = hist_end - self.horizon
            tgt_end = hist_end

        x_hist = self.x[hist_start:hist_end]
        x_future_time = self.x[tgt_start:tgt_end, self.future_time_indices]
        
        y_hist_power = self.y_power[hist_start:hist_end].T
        y_hist_mains = self.mains[hist_start:hist_end]
        y_hist_power_scaled = self.y_power_scaled_full[hist_start:hist_end].T
        y_hist_state = self.state[hist_start:hist_end].T
        y_hist_event = self.event[hist_start:hist_end].T
        y_hist_start = self.start_event[hist_start:hist_end].T
        y_hist_stop = self.stop_event[hist_start:hist_end].T
        hist_mask = self.target_available[hist_start:hist_end].T

        # Transpose targets to [A, H], matching PISAModel output.
        y_power = self.y_power[tgt_start:tgt_end].T
        y_power_scaled = self.y_power_scaled_full[tgt_start:tgt_end].T
        y_state = self.state[tgt_start:tgt_end].T
        y_event = self.event[tgt_start:tgt_end].T
        y_start = self.start_event[tgt_start:tgt_end].T
        y_stop = self.stop_event[tgt_start:tgt_end].T
        target_mask = self.target_available[tgt_start:tgt_end].T
        event_mask = target_mask.copy()
        if self.partial_train_labels:
            # An event label requires both adjacent appliance labels. Filled
            # missing measurements must not create supervised transitions.
            event_mask[:, 1:] *= target_mask[:, :-1]
            event_mask[:, 0] *= self.target_available[tgt_start - 1]
        y_mains = self.mains[tgt_start:tgt_end]

        # The event at the first forecast step is defined relative to the
        # immediately preceding historical state, not relative to an implicit
        # OFF state at the start of every sliding window.
        event_prev_hist_index = tgt_start - hist_start - 1
        if event_prev_hist_index >= 0:
            y_prev_state = self.state[tgt_start - 1]
        else:
            event_prev_hist_index = -1
            y_prev_state = np.zeros(len(self.bundle.appliance_cols), dtype=np.float32)

        item: dict[str, Tensor | str] = {
            "x_hist": torch.from_numpy(x_hist.astype(np.float32)),
            "x_future_time": torch.from_numpy(x_future_time.astype(np.float32)),
            "y_power": torch.from_numpy(y_power.astype(np.float32)),
            "y_power_scaled": torch.from_numpy(y_power_scaled.astype(np.float32)),
            "y_state": torch.from_numpy(y_state.astype(np.float32)),
            "y_event": torch.from_numpy(y_event.astype(np.float32)),
            "y_start": torch.from_numpy(y_start.astype(np.float32)),
            "y_stop": torch.from_numpy(y_stop.astype(np.float32)),
            "y_prev_state": torch.from_numpy(y_prev_state.astype(np.float32)),
            "event_prev_hist_index": torch.tensor(
                event_prev_hist_index,
                dtype=torch.long,
            ),
            "y_mains": torch.from_numpy(y_mains.astype(np.float32)),
            "target_mask": torch.from_numpy(target_mask.astype(np.float32)),
            # Historical reconstruction targets.
            "y_hist_power": torch.from_numpy(y_hist_power.astype(np.float32)),
            "y_hist_mains": torch.from_numpy(y_hist_mains.astype(np.float32)),
            "y_hist_power_scaled": torch.from_numpy(y_hist_power_scaled.astype(np.float32)),
            "y_hist_state": torch.from_numpy(y_hist_state.astype(np.float32)),
            "y_hist_event": torch.from_numpy(y_hist_event.astype(np.float32)),
            "y_hist_start": torch.from_numpy(y_hist_start.astype(np.float32)),
            "y_hist_stop": torch.from_numpy(y_hist_stop.astype(np.float32)),
            "hist_mask": torch.from_numpy(hist_mask.astype(np.float32)),
        }

        if self.return_timestamps:
            item["timestamp_start"] = str(self.timestamps[tgt_start])
            item["timestamp_end"] = str(self.timestamps[tgt_end - 1])

        if self.partial_train_labels:
            item["event_mask"] = torch.from_numpy(event_mask.astype(np.float32))

        return item


def summarize_on_support(
    dataset: PecanStreetSingleHomeDataset, min_windows: int = 5,
) -> dict:
    """Audit label coverage over the actual windows, without model inference.

    ON-window counts include overlapping forecasts. Also report unique ON
    timesteps so repetition cannot be mistaken for independent observations.
    """
    if min_windows < 1:
        raise ValueError("min_windows must be positive.")
    starts = np.asarray(dataset.indices, dtype=np.int64) + dataset.input_window
    if dataset.target_mode == "history_tail":
        starts -= dataset.horizon
    ends = starts + dataset.horizon
    valid_on = (dataset.state > .5) & (dataset.target_available > .5)
    prefix = np.vstack([np.zeros((1, valid_on.shape[1]), dtype=np.int64),
                        np.cumsum(valid_on, axis=0, dtype=np.int64)])
    window_on_steps = prefix[ends] - prefix[starts]
    coverage_delta = np.zeros(len(valid_on) + 1, dtype=np.int64)
    np.add.at(coverage_delta, starts, 1)
    np.add.at(coverage_delta, ends, -1)
    covered = np.cumsum(coverage_delta[:-1]) > 0
    result = {}
    for index, app in enumerate(dataset.bundle.appliance_cols):
        windows = int((window_on_steps[:, index] > 0).sum())
        result[app] = {
            "on_windows": windows,
            "on_valid_steps_with_overlap": int(window_on_steps[:, index].sum()),
            "on_unique_steps": int((valid_on[:, index] & covered).sum()),
            "supported": windows >= min_windows,
        }
    return {"split": dataset.split, "windows": len(starts), "min_on_windows": min_windows,
            "overlapping_windows_are_not_independent": True, "appliances": result,
            "all_supported": all(item["supported"] for item in result.values())}


class PecanStreetMultiHomeDataset(ConcatDataset):
    """Concatenate home datasets while retaining auditable home identity."""

    def __init__(
        self,
        datasets: list[PecanStreetSingleHomeDataset],
        home_ids: list[str],
    ) -> None:
        if not datasets:
            raise ValueError("At least one home dataset is required.")
        if len(datasets) != len(home_ids):
            raise ValueError("datasets and home_ids should have equal length.")
        if len(set(home_ids)) != len(home_ids):
            raise ValueError(f"Home identifiers should be unique: {home_ids}")
        super().__init__(datasets)
        self.home_ids = [str(value) for value in home_ids]
        self.home_sizes = [len(dataset) for dataset in datasets]

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        home_index = bisect.bisect_right(self.cumulative_sizes, index)
        item = dict(super().__getitem__(index))
        item["home_index"] = torch.tensor(home_index, dtype=torch.long)
        item["home_id"] = self.home_ids[home_index]
        return item


class BalancedHomeBatchSampler(Sampler[list[int]]):
    """Yield batches with an equal quota from every source household.

    Smaller household datasets are reshuffled and recycled within an epoch.
    The total number of samples per epoch remains close to the natural
    concatenated dataset size, avoiding an accidental increase in optimizer
    steps when one source household has fewer valid windows.
    """

    def __init__(
        self,
        dataset: PecanStreetMultiHomeDataset,
        batch_size: int,
        seed: int = 42,
        drop_last: bool = True,
    ) -> None:
        if batch_size < len(dataset.home_sizes):
            raise ValueError(
                "batch_size should be at least the number of source homes."
            )
        if any(size < 1 for size in dataset.home_sizes):
            raise ValueError("Every source home should contain at least one window.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0
        total = len(dataset)
        self.num_batches = (
            total // self.batch_size
            if self.drop_last
            else (total + self.batch_size - 1) // self.batch_size
        )
        self.num_batches = max(1, self.num_batches)
        self.offsets = [0, *dataset.cumulative_sizes[:-1]]

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self._epoch)
        self._epoch += 1

        permutations = [
            torch.randperm(size, generator=generator)
            for size in self.dataset.home_sizes
        ]
        cursors = [0 for _ in self.dataset.home_sizes]

        def draw(home_index: int, count: int) -> list[int]:
            selected: list[int] = []
            size = self.dataset.home_sizes[home_index]
            while len(selected) < count:
                remaining = size - cursors[home_index]
                take = min(count - len(selected), remaining)
                start = cursors[home_index]
                selected.extend(
                    permutations[home_index][start : start + take].tolist()
                )
                cursors[home_index] += take
                if cursors[home_index] >= size:
                    permutations[home_index] = torch.randperm(
                        size,
                        generator=generator,
                    )
                    cursors[home_index] = 0
            offset = self.offsets[home_index]
            return [offset + value for value in selected]

        num_homes = len(self.dataset.home_sizes)
        base_quota, remainder = divmod(self.batch_size, num_homes)
        for batch_index in range(self.num_batches):
            batch: list[int] = []
            # Rotate remainder slots so no household systematically receives
            # the larger quota when batch_size is not divisible by num_homes.
            remainder_start = batch_index % num_homes
            for home_index in range(num_homes):
                relative = (home_index - remainder_start) % num_homes
                quota = base_quota + int(relative < remainder)
                batch.extend(draw(home_index, quota))
            order = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[position] for position in order]


def _read_csv(csv_path: str | Path, time_col: str | None = "localminute") -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if time_col and time_col in df.columns:
        # Keep original timezone string for reporting; sort using parsed UTC-like timestamp when possible.
        parsed = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        if parsed.notna().any():
            df = df.assign(_parsed_time=parsed).sort_values("_parsed_time").drop(columns=["_parsed_time"])
        else:
            df = df.sort_values(time_col)

    return df.reset_index(drop=True)


def _mask_negative_power_readings(
    df: pd.DataFrame,
    mains_col: str,
    appliance_cols: list[str],
    availability_cols: dict[str, str] | None,
) -> pd.DataFrame:
    """Mark negative power as unavailable before fitting preprocessing state."""
    result = df.copy()
    availability_cols = availability_cols or {}
    for column in [mains_col, *appliance_cols]:
        numeric = pd.to_numeric(result[column], errors="coerce")
        valid = numeric.notna() & numeric.ge(0.0)
        availability_column = availability_cols.get(column, f"{column}_available")
        if availability_column in result.columns:
            recorded_available = (
                pd.to_numeric(result[availability_column], errors="coerce")
                .fillna(0.0)
                .gt(0.5)
            )
            result[availability_column] = (recorded_available & valid).astype(np.int64)
        else:
            result[availability_column] = valid.astype(np.int64)
        result[column] = numeric.where(valid)
    return result


def build_single_home_datasets(
    csv_path: str | Path,
    input_cols: Iterable[str] | None = None,
    appliance_cols: Iterable[str] = (
        "air1",
        "refrigerator1",
        "dishwasher1",
        "microwave1",
    ),
    mains_col: str = "grid",
    split_col: str = "split",
    time_col: str | None = "localminute",
    time_feature_cols: Iterable[str] = (
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
    ),
    input_window: int = 120,
    horizon: int = 30,
    target_mode: Literal["future", "history_tail"] = "future",
    stride: int = 1,
    state_thresholds: dict[str, float] | None = None,
    state_min_threshold_kw: float = 0.01,
    state_ratio_to_p95_positive: float = 0.05,
    appliance_scale_mode: str = "p99_positive",
    availability_cols: dict[str, str] | None = None,
    drop_unavailable_windows: bool = True,
    return_timestamps: bool = True,
    reference_bundle: PecanStreetDataBundle | None = None,
    partial_train_labels: bool = False,
    require_future_mains: bool = True,
) -> tuple[dict[str, PecanStreetSingleHomeDataset], PecanStreetDataBundle]:
    """Build train/val/test datasets for the Home 7951 CSV.

    If input_cols is not given, input columns are:
        [mains_col] + time_feature_cols

    If input_cols is given, use input_cols directly.
    Target columns are appliance_cols.
    """

    appliance_cols = list(appliance_cols)

    if input_cols is None:
        time_feature_cols = list(time_feature_cols)
        input_cols = [mains_col] + time_feature_cols
    else:
        input_cols = list(input_cols)

    df = _read_csv(csv_path, time_col=time_col)

    if split_col not in df.columns:
        raise KeyError(
            f"split_col='{split_col}' not found. "
            "The CSV should contain train/val/test split labels."
        )

    # 检查输入列是否存在
    missing_input_cols = [col for col in input_cols if col not in df.columns]
    if missing_input_cols:
        raise KeyError(f"Missing input columns in CSV: {missing_input_cols}")

    # 检查目标电器列是否存在
    missing_appliance_cols = [col for col in appliance_cols if col not in df.columns]
    if missing_appliance_cols:
        raise KeyError(f"Missing appliance columns in CSV: {missing_appliance_cols}")

    if mains_col not in df.columns:
        raise KeyError(f"mains_col='{mains_col}' not found in CSV.")

    # Apply the nonnegative-demand policy before estimating thresholds,
    # fitting scalers or constructing any split. Availability flags retain the
    # invalid minutes so sliding windows can exclude them without collapsing
    # gaps in the time axis.
    df = _mask_negative_power_readings(
        df,
        mains_col=mains_col,
        appliance_cols=appliance_cols,
        availability_cols=availability_cols,
    )

    train_df = df[df[split_col].astype(str).str.lower() == "train"].copy()
    if train_df.empty:
        raise ValueError(
            "No training rows found. Expected split column to contain 'train'."
        )

    if reference_bundle is not None:
        if list(reference_bundle.input_cols) != list(input_cols):
            raise ValueError("reference_bundle input columns do not match requested input_cols.")
        if list(reference_bundle.appliance_cols) != appliance_cols:
            raise ValueError(
                "reference_bundle appliance columns do not match requested appliance_cols."
            )
        state_thresholds = {
            app: float(reference_bundle.state_thresholds[app])
            for app in appliance_cols
        }
    elif state_thresholds is None:
        state_thresholds = estimate_state_thresholds(
            train_df,
            appliance_cols=appliance_cols,
            min_threshold_kw=state_min_threshold_kw,
            ratio_to_p95_positive=state_ratio_to_p95_positive,
        )
    else:
        state_thresholds = {
            app: float(state_thresholds[app])
            for app in appliance_cols
        }

    if reference_bundle is None:
        input_scaler = StandardScaler1D().fit(
            train_df[input_cols]
            .apply(pd.to_numeric, errors="coerce")
            .ffill()
            .bfill()
            .fillna(0.0)
        )

        appliance_scaler = ApplianceScaler(mode=appliance_scale_mode).fit(
            train_df,
            appliance_cols,
            state_thresholds=state_thresholds,
        )

        scalers = PISAScalers(
            input_scaler=input_scaler,
            appliance_scaler=appliance_scaler,
        )
        stats = compute_appliance_activity_stats(
            train_df,
            appliance_cols=appliance_cols,
            thresholds=state_thresholds,
        )
    else:
        scalers = reference_bundle.scalers
        stats = reference_bundle.appliance_activity_stats

    bundle = PecanStreetDataBundle(
        csv_path=str(csv_path),
        input_cols=input_cols,
        appliance_cols=appliance_cols,
        mains_col=mains_col,
        split_col=split_col,
        time_col=time_col,
        state_thresholds=state_thresholds,
        scalers=scalers,
        appliance_activity_stats=stats,
    )

    datasets: dict[str, PecanStreetSingleHomeDataset] = {}

    for split in ("train", "val", "test"):
        split_mask = df[split_col].astype(str).str.lower() == split

        if split_mask.any():
            datasets[split] = PecanStreetSingleHomeDataset(
                df=df,
                split=split,  # type: ignore[arg-type]
                bundle=bundle,
                input_window=input_window,
                horizon=horizon,
                target_mode=target_mode,
                stride=stride,
                availability_cols=availability_cols,
                drop_unavailable_windows=drop_unavailable_windows,
                return_timestamps=return_timestamps,
                partial_train_labels=partial_train_labels,
                require_future_mains=require_future_mains,
            )

    return datasets, bundle


def build_single_home_dataloaders(
    csv_path: str | Path,
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle_train: bool = True,
    pin_memory: bool = True,
    **dataset_kwargs: Any,
) -> tuple[dict[str, DataLoader], PecanStreetDataBundle]:
    """Build train/val/test dataloaders for a single home CSV."""
    datasets, bundle = build_single_home_datasets(csv_path=csv_path, **dataset_kwargs)

    loaders: dict[str, DataLoader] = {}
    for split, ds in datasets.items():
        # The validation/test loaders are iterated more than once by the fair
        # evaluation runner.  Keeping workers alive avoids paying process
        # start-up cost for every model/checkpoint pass.  `prefetch_factor` is
        # illegal when num_workers is zero, hence the conditional kwargs.
        loader_kwargs: dict[str, Any] = {}
        if num_workers > 0:
            loader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=2,
            )

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train" and shuffle_train),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
            **loader_kwargs,
        )

    return loaders, bundle


def _home_id_from_path(path: str | Path) -> str:
    stem = Path(path).stem
    match = re.search(r"home[_-]?(\d+)", stem, flags=re.IGNORECASE)
    return match.group(1) if match else stem


def build_multi_home_datasets(
    csv_paths: Iterable[str | Path],
    input_cols: Iterable[str] | None = None,
    appliance_cols: Iterable[str] = (
        "air1",
        "refrigerator1",
        "dishwasher1",
        "microwave1",
    ),
    mains_col: str = "grid",
    split_col: str = "split",
    time_col: str | None = "localminute",
    time_feature_cols: Iterable[str] = (
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
    ),
    input_window: int = 120,
    horizon: int = 30,
    target_mode: Literal["future", "history_tail"] = "future",
    stride: int = 1,
    state_thresholds: dict[str, float] | None = None,
    state_min_threshold_kw: float = 0.01,
    state_ratio_to_p95_positive: float = 0.05,
    appliance_scale_mode: str = "p99_positive",
    availability_cols: dict[str, str] | None = None,
    drop_unavailable_windows: bool = True,
    return_timestamps: bool = True,
) -> tuple[
    dict[str, PecanStreetMultiHomeDataset],
    PecanStreetDataBundle,
    dict[str, dict[str, PecanStreetSingleHomeDataset]],
]:
    """Build source-only preprocessing and datasets for multiple homes.

    Scalers, state thresholds, and appliance activity statistics are fitted on
    the concatenation of *training splits from source homes only*. Validation
    and test rows never affect preprocessing. The returned per-home datasets
    make it possible to report household-level metrics in addition to pooled
    source validation metrics.
    """

    resolved_paths = [Path(path).expanduser().resolve() for path in csv_paths]
    if len(resolved_paths) < 2:
        raise ValueError("Multi-home training requires at least two source homes.")
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Duplicate source CSV paths are not allowed.")

    appliance_cols = list(appliance_cols)
    if input_cols is None:
        input_cols = [mains_col, *list(time_feature_cols)]
    else:
        input_cols = list(input_cols)

    home_ids = [_home_id_from_path(path) for path in resolved_paths]
    if len(set(home_ids)) != len(home_ids):
        raise ValueError(f"Could not derive unique home IDs from: {resolved_paths}")

    frames: dict[str, pd.DataFrame] = {}
    train_frames: list[pd.DataFrame] = []
    required = list(dict.fromkeys([*input_cols, mains_col, *appliance_cols]))
    for home_id, path in zip(home_ids, resolved_paths, strict=True):
        df = _read_csv(path, time_col=time_col)
        if split_col not in df.columns:
            raise KeyError(f"split_col='{split_col}' not found in {path}.")
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns in {path}: {missing}")
        df = _mask_negative_power_readings(
            df,
            mains_col=mains_col,
            appliance_cols=appliance_cols,
            availability_cols=availability_cols,
        )
        train_df = df[df[split_col].astype(str).str.lower() == "train"].copy()
        if train_df.empty:
            raise ValueError(f"No training rows found in source home {path}.")
        frames[home_id] = df
        train_frames.append(train_df)

    combined_train = pd.concat(train_frames, axis=0, ignore_index=True)
    if state_thresholds is None:
        fitted_thresholds = estimate_state_thresholds(
            combined_train,
            appliance_cols=appliance_cols,
            min_threshold_kw=state_min_threshold_kw,
            ratio_to_p95_positive=state_ratio_to_p95_positive,
        )
    else:
        fitted_thresholds = {
            app: float(state_thresholds[app]) for app in appliance_cols
        }

    # Fill within each household before concatenation. Filling the combined
    # frame would allow a boundary value from one home to impute another home.
    input_fit_frame = pd.concat(
        [
            frame[input_cols]
            .apply(pd.to_numeric, errors="coerce")
            .ffill()
            .bfill()
            .fillna(0.0)
            for frame in train_frames
        ],
        axis=0,
        ignore_index=True,
    )
    input_scaler = StandardScaler1D().fit(input_fit_frame)
    appliance_scaler = ApplianceScaler(mode=appliance_scale_mode).fit(
        combined_train,
        appliance_cols,
        state_thresholds=fitted_thresholds,
    )
    scalers = PISAScalers(
        input_scaler=input_scaler,
        appliance_scaler=appliance_scaler,
    )
    activity_stats = compute_appliance_activity_stats(
        combined_train,
        appliance_cols=appliance_cols,
        thresholds=fitted_thresholds,
    )
    bundle = PecanStreetDataBundle(
        csv_path=";".join(str(path) for path in resolved_paths),
        input_cols=list(input_cols),
        appliance_cols=appliance_cols,
        mains_col=mains_col,
        split_col=split_col,
        time_col=time_col,
        state_thresholds=fitted_thresholds,
        scalers=scalers,
        appliance_activity_stats=activity_stats,
    )

    per_home: dict[str, dict[str, PecanStreetSingleHomeDataset]] = {}
    combined: dict[str, PecanStreetMultiHomeDataset] = {}
    for home_id in home_ids:
        per_home[home_id] = {}
        df = frames[home_id]
        for split in ("train", "val", "test"):
            if not (df[split_col].astype(str).str.lower() == split).any():
                continue
            per_home[home_id][split] = PecanStreetSingleHomeDataset(
                df=df,
                split=split,  # type: ignore[arg-type]
                bundle=bundle,
                input_window=input_window,
                horizon=horizon,
                target_mode=target_mode,
                stride=stride,
                availability_cols=availability_cols,
                drop_unavailable_windows=drop_unavailable_windows,
                return_timestamps=return_timestamps,
            )

    for split in ("train", "val", "test"):
        split_home_ids = [
            home_id for home_id in home_ids if split in per_home[home_id]
        ]
        if split_home_ids:
            combined[split] = PecanStreetMultiHomeDataset(
                datasets=[per_home[home_id][split] for home_id in split_home_ids],
                home_ids=split_home_ids,
            )

    if "train" not in combined or "val" not in combined:
        raise ValueError("Every multi-home run requires source train and val splits.")
    return combined, bundle, per_home


def build_multi_home_dataloaders(
    csv_paths: Iterable[str | Path],
    batch_size: int = 128,
    num_workers: int = 0,
    balanced_train: bool = True,
    seed: int = 42,
    pin_memory: bool = True,
    **dataset_kwargs: Any,
) -> tuple[
    dict[str, DataLoader],
    PecanStreetDataBundle,
    dict[str, dict[str, DataLoader]],
]:
    """Build pooled and per-home loaders under a source-only LOHO protocol."""

    datasets, bundle, per_home_datasets = build_multi_home_datasets(
        csv_paths=csv_paths,
        **dataset_kwargs,
    )
    loaders: dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        worker_kwargs: dict[str, Any] = {}
        if num_workers > 0:
            worker_kwargs.update(persistent_workers=True, prefetch_factor=2)
        if split == "train" and balanced_train:
            loaders[split] = DataLoader(
                dataset,
                batch_sampler=BalancedHomeBatchSampler(
                    dataset=dataset,
                    batch_size=batch_size,
                    seed=seed,
                    drop_last=True,
                ),
                num_workers=num_workers,
                pin_memory=pin_memory,
                **worker_kwargs,
            )
        else:
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=(split == "train"),
                **worker_kwargs,
            )

    per_home_loaders: dict[str, dict[str, DataLoader]] = {}
    for home_id, split_datasets in per_home_datasets.items():
        per_home_loaders[home_id] = {}
        for split, dataset in split_datasets.items():
            worker_kwargs = {}
            if num_workers > 0:
                worker_kwargs.update(persistent_workers=True, prefetch_factor=2)
            per_home_loaders[home_id][split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
                **worker_kwargs,
            )
    return loaders, bundle, per_home_loaders
