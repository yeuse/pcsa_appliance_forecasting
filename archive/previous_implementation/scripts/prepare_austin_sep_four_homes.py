"""Prepare four comparable Austin homes for September 2018.

Homes 7951 and 3039 are read from the one-minute Dataport export. Homes 8386
and 8565 are read from the sorted one-second Q3 export and aggregated to one-minute
means. The common labels are the four appliance channels already reported in
the manuscript: air conditioner, refrigerator, dishwasher and microwave.

Negative real-power readings are invalid observations and are omitted before
aggregation. A negative grid reading invalidates the containing minute because
it represents net export. Negative appliance-circuit seconds are treated as
meter noise; the minute remains usable only when at least 50 nonnegative
seconds remain. No negative reading is clipped to zero or treated as demand.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


MINUTE_HOMES = ["7951", "3039"]
SECOND_HOMES = ["8386", "8565"]
HOMES = [*MINUTE_HOMES, *SECOND_HOMES]
APPLIANCES = ["air1", "refrigerator1", "dishwasher1", "microwave1"]
POWER_COLUMNS = ["grid", *APPLIANCES]
THRESHOLDS_KW = {
    "air1": 0.50,
    "refrigerator1": 0.05,
    "dishwasher1": 0.05,
    "microwave1": 0.10,
}
START = pd.Timestamp("2018-09-01 00:00:00")
END_EXCLUSIVE = pd.Timestamp("2018-10-01 00:00:00")
FULL_INDEX = pd.date_range(START, END_EXCLUSIVE, freq="min", inclusive="left")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minute-csv", type=Path, required=True)
    parser.add_argument("--second-csv", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/austin_2018_sep_4homes"),
    )
    parser.add_argument(
        "--minimum-valid-seconds",
        type=int,
        default=50,
        help="Minimum nonnegative one-second readings required per output minute.",
    )
    parser.add_argument("--input-window", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=30)
    return parser.parse_args()


def read_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path, skiprows=[1], dtype=str).fillna("")
    metadata["dataid"] = metadata["dataid"].astype(str)
    selected = metadata[metadata["dataid"].isin(HOMES)].set_index("dataid")
    missing = [home for home in HOMES if home not in selected.index]
    if missing:
        raise ValueError(f"Homes missing from metadata: {missing}")
    return selected


def record_id(line: bytes) -> int:
    comma = line.find(b",")
    if comma <= 0:
        return -1
    try:
        return int(line[:comma])
    except ValueError:
        return -1


def lower_bound_dataid(path: Path, target: int, header_bytes: int) -> int:
    """Return the first byte position whose sorted dataid is >= target."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        low, high = header_bytes, size
        while high - low > 8192:
            midpoint = (low + high) // 2
            handle.seek(midpoint)
            handle.readline()
            position = handle.tell()
            line = handle.readline()
            if not line:
                high = midpoint
            elif record_id(line) < target:
                low = handle.tell()
            else:
                high = position

        handle.seek(low)
        if low > header_bytes:
            handle.readline()
        while True:
            position = handle.tell()
            line = handle.readline()
            if not line or record_id(line) >= target:
                return position


def flush_second_minute(
    output: list[dict[str, object]],
    minute_key: str | None,
    sums: np.ndarray,
    counts: np.ndarray,
    negatives: np.ndarray,
    source_rows: int,
    minimum_valid_seconds: int,
) -> None:
    if minute_key is None:
        return
    row: dict[str, object] = {
        "minute": minute_key,
        "source_rows": source_rows,
    }
    for index, column in enumerate(POWER_COLUMNS):
        valid = counts[index] >= minimum_valid_seconds
        if column == "grid":
            valid = valid and negatives[index] == 0
        row[column] = float(sums[index] / counts[index]) if valid else np.nan
        row[f"{column}_available_seconds"] = int(counts[index])
        row[f"{column}_negative_excluded"] = int(negatives[index])
    output.append(row)


def aggregate_second_home(
    path: Path,
    dataid: str,
    minimum_valid_seconds: int,
) -> pd.DataFrame:
    with path.open("rb") as handle:
        header_line = handle.readline()
        header = header_line.decode("utf-8-sig").rstrip("\r\n").split(",")
        header_bytes = handle.tell()
    indices = {column: header.index(column) for column in ["localminute", *POWER_COLUMNS]}
    start = lower_bound_dataid(path, int(dataid), header_bytes)
    end = lower_bound_dataid(path, int(dataid) + 1, header_bytes)
    if start >= end:
        raise RuntimeError(f"Home {dataid} does not occur in {path}")

    output: list[dict[str, object]] = []
    current_minute: str | None = None
    sums = np.zeros(len(POWER_COLUMNS), dtype=np.float64)
    counts = np.zeros(len(POWER_COLUMNS), dtype=np.int64)
    negatives = np.zeros(len(POWER_COLUMNS), dtype=np.int64)
    source_rows = 0
    total_rows = 0

    with path.open("rb") as handle:
        handle.seek(start)
        while handle.tell() < end:
            line = handle.readline()
            if not line or record_id(line) != int(dataid):
                break
            fields = line.rstrip(b"\r\n").split(b",")
            timestamp = fields[indices["localminute"]].decode("ascii", errors="ignore")
            minute_key = timestamp[:16]
            if minute_key != current_minute:
                flush_second_minute(
                    output,
                    current_minute,
                    sums,
                    counts,
                    negatives,
                    source_rows,
                    minimum_valid_seconds,
                )
                current_minute = minute_key
                sums.fill(0.0)
                counts.fill(0)
                negatives.fill(0)
                source_rows = 0
            source_rows += 1
            total_rows += 1
            for power_index, column in enumerate(POWER_COLUMNS):
                raw = fields[indices[column]]
                if not raw:
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if not math.isfinite(value):
                    continue
                if value < 0.0:
                    negatives[power_index] += 1
                    continue
                sums[power_index] += value
                counts[power_index] += 1
            if total_rows % 1_000_000 == 0:
                print(f"home {dataid}: processed {total_rows:,} one-second rows", flush=True)

    flush_second_minute(
        output,
        current_minute,
        sums,
        counts,
        negatives,
        source_rows,
        minimum_valid_seconds,
    )
    frame = pd.DataFrame(output)
    frame["minute"] = pd.to_datetime(frame["minute"], errors="coerce")
    frame = frame.dropna(subset=["minute"])
    frame = frame[(frame["minute"] >= START) & (frame["minute"] < END_EXCLUSIVE)]
    return frame.sort_values("minute").drop_duplicates("minute", keep="last").set_index("minute")


def extract_minute_homes(path: Path, dataids: list[str]) -> dict[str, pd.DataFrame]:
    parts: dict[str, list[pd.DataFrame]] = {dataid: [] for dataid in dataids}
    usecols = ["dataid", "localminute", *POWER_COLUMNS]
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype={"dataid": "string", "localminute": "string"},
        chunksize=500_000,
    ):
        selected = chunk[
            chunk["dataid"].isin(dataids)
            & chunk["localminute"].str.slice(0, 7).eq("2018-09")
        ].copy()
        for dataid, home in selected.groupby("dataid", sort=False):
            parts[str(dataid)].append(home)

    outputs: dict[str, pd.DataFrame] = {}
    for dataid in dataids:
        if not parts[dataid]:
            raise RuntimeError(f"Home {dataid} has no September 2018 rows in {path}")
        frame = pd.concat(parts[dataid], ignore_index=True)
        frame["minute"] = pd.to_datetime(
            frame["localminute"].str.slice(0, 19),
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
        frame = frame.dropna(subset=["minute"]).sort_values("minute")
        output = frame.set_index("minute")[POWER_COLUMNS].copy()
        for column in POWER_COLUMNS:
            output[column] = pd.to_numeric(output[column], errors="coerce")
            negative = output[column].notna() & output[column].lt(0.0)
            output[f"{column}_available_seconds"] = (
                output[column].notna() & ~negative
            ).astype(np.int64)
            output[f"{column}_negative_excluded"] = negative.astype(np.int64)
            output.loc[negative, column] = np.nan
        output["source_rows"] = 1
        outputs[dataid] = output[~output.index.duplicated(keep="last")]
    return outputs


def split_labels(index: pd.DatetimeIndex) -> np.ndarray:
    labels = np.full(len(index), "train", dtype=object)
    labels[(index.day >= 21) & (index.day <= 25)] = "val"
    labels[index.day >= 26] = "test"
    return labels


def usable_windows(
    grid_available: np.ndarray,
    target_available: np.ndarray,
    input_window: int,
    horizon: int,
) -> int:
    total = len(grid_available) - input_window - horizon + 1
    if total <= 0:
        return 0
    grid_prefix = np.concatenate(([0], np.cumsum(grid_available, dtype=np.int64)))
    target_rows = target_available.all(axis=1)
    target_prefix = np.concatenate(([0], np.cumsum(target_rows, dtype=np.int64)))
    starts = np.arange(total)
    grid_window_ok = (
        grid_prefix[starts + input_window + horizon] - grid_prefix[starts]
        == input_window + horizon
    )
    target_start = starts + input_window
    future_ok = target_prefix[target_start + horizon] - target_prefix[target_start] == horizon
    return int(np.count_nonzero(grid_window_ok & future_ok))


def prepare_output(dataid: str, source: pd.DataFrame, metadata: pd.Series) -> pd.DataFrame:
    source = source.reindex(FULL_INDEX)
    output = pd.DataFrame(index=FULL_INDEX)
    output["dataid"] = dataid
    output["localminute"] = FULL_INDEX.strftime("%Y-%m-%d %H:%M:00-05")
    output["city"] = str(metadata.get("city", ""))
    output["state"] = str(metadata.get("state", ""))
    output["year"] = FULL_INDEX.year
    output["month"] = FULL_INDEX.month
    output["day"] = FULL_INDEX.day
    output["hour"] = FULL_INDEX.hour
    output["minute"] = FULL_INDEX.minute
    output["dayofweek"] = FULL_INDEX.dayofweek
    output["is_weekend"] = (FULL_INDEX.dayofweek >= 5).astype(np.int64)
    output["hour_sin"] = np.sin(2 * math.pi * FULL_INDEX.hour / 24)
    output["hour_cos"] = np.cos(2 * math.pi * FULL_INDEX.hour / 24)
    output["dow_sin"] = np.sin(2 * math.pi * FULL_INDEX.dayofweek / 7)
    output["dow_cos"] = np.cos(2 * math.pi * FULL_INDEX.dayofweek / 7)
    output["split"] = split_labels(FULL_INDEX)
    output["source_rows"] = source["source_rows"].fillna(0).astype(np.int64)

    for column in POWER_COLUMNS:
        value = pd.to_numeric(source[column], errors="coerce")
        seconds = pd.to_numeric(
            source[f"{column}_available_seconds"], errors="coerce"
        ).fillna(0).astype(np.int64)
        negative = pd.to_numeric(
            source[f"{column}_negative_excluded"], errors="coerce"
        ).fillna(0).astype(np.int64)
        valid = value.notna() & value.ge(0.0)
        output[column] = value.where(valid)
        output[f"{column}_available"] = valid.astype(np.int64)
        output[f"{column}_available_seconds"] = seconds
        output[f"{column}_negative_excluded"] = negative

    target_available = output[[f"{app}_available" for app in APPLIANCES]]
    output["available_target_count"] = target_available.sum(axis=1)
    output["all_targets_available"] = target_available.all(axis=1).astype(np.int64)
    output["model_row_available"] = (
        output["grid_available"].eq(1) & output["all_targets_available"].eq(1)
    ).astype(np.int64)
    return output.reset_index(drop=True)


def audit_output(
    frame: pd.DataFrame,
    input_window: int,
    horizon: int,
) -> dict[str, object]:
    dataid = str(frame["dataid"].iloc[0])
    audit: dict[str, object] = {
        "dataid": dataid,
        "minutes": len(frame),
        "grid_negative_readings_excluded": int(frame["grid_negative_excluded"].sum()),
        "grid_available_minutes": int(frame["grid_available"].sum()),
        "all_targets_available_minutes": int(frame["all_targets_available"].sum()),
    }
    split = frame["split"].to_numpy()
    for split_name in ("train", "val", "test"):
        mask = split == split_name
        audit[f"{split_name}_usable_windows"] = usable_windows(
            frame.loc[mask, "grid_available"].to_numpy(dtype=bool),
            frame.loc[mask, [f"{app}_available" for app in APPLIANCES]].to_numpy(dtype=bool),
            input_window,
            horizon,
        )
        for appliance in APPLIANCES:
            values = frame.loc[mask, appliance].to_numpy(dtype=float)
            available = frame.loc[mask, f"{appliance}_available"].to_numpy(dtype=bool)
            audit[f"{split_name}_{appliance}_active_minutes"] = int(
                np.count_nonzero(available & (values >= THRESHOLDS_KW[appliance]))
            )
    audit["all_appliances_active_all_splits"] = all(
        int(audit[f"{split_name}_{app}_active_minutes"]) > 0
        for split_name in ("train", "val", "test")
        for app in APPLIANCES
    )
    audit["minimum_split_usable_windows"] = min(
        int(audit[f"{split_name}_usable_windows"])
        for split_name in ("train", "val", "test")
    )
    return audit


def main() -> None:
    args = parse_args()
    if not 1 <= args.minimum_valid_seconds <= 60:
        raise ValueError("minimum-valid-seconds must be between 1 and 60")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_metadata(args.metadata_csv)

    sources = extract_minute_homes(args.minute_csv, MINUTE_HOMES)
    for dataid in SECOND_HOMES:
        sources[dataid] = aggregate_second_home(
            args.second_csv,
            dataid,
            minimum_valid_seconds=args.minimum_valid_seconds,
        )

    prepared_frames: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for dataid in HOMES:
        prepared = prepare_output(dataid, sources[dataid], metadata.loc[dataid])
        prepared.to_csv(
            args.output_dir / f"home_{dataid}_2018_sep_1min.csv",
            index=False,
        )
        prepared_frames.append(prepared)
        audits.append(audit_output(prepared, args.input_window, args.horizon))

    combined = pd.concat(prepared_frames, ignore_index=True)
    combined.to_csv(args.output_dir / "selected_4homes_2018_sep_1min.csv", index=False)
    audit = pd.DataFrame(audits)
    audit.to_csv(args.output_dir / "quality_audit.csv", index=False)
    protocol = {
        "homes": HOMES,
        "pv_metadata": {
            home: str(metadata.loc[home].get("pv", "")).strip().lower() == "yes"
            for home in HOMES
        },
        "location": "Austin, Texas",
        "period": "2018-09-01 through 2018-09-30",
        "appliances": APPLIANCES,
        "split": "days 1-20 train, 21-25 validation, 26-30 test",
        "input_window_minutes": args.input_window,
        "forecast_horizon_minutes": args.horizon,
        "minimum_valid_seconds_for_1s_sources": args.minimum_valid_seconds,
        "negative_power_policy": (
            "Every negative reading is excluded before aggregation. A negative grid "
            "second invalidates its minute because it indicates net export. For an "
            "appliance channel, the minute remains available only if at least the "
            "configured number of nonnegative seconds remain. Sliding windows "
            "containing an unavailable grid-history minute are excluded."
        ),
        "source_files": {
            **{home: str(args.minute_csv.resolve()) for home in MINUTE_HOMES},
            **{home: str(args.second_csv.resolve()) for home in SECOND_HOMES},
        },
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(audit.to_string(index=False))
    print(f"output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
