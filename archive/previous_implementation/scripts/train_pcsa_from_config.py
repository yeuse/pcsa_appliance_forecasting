"""Run the source training CLI from a saved experiment configuration."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_home7951  # noqa: E402


DERIVED_FIELDS = {
    "input_dim",
    "num_appliances",
    "nominal_rated_power_kw",
    "rated_power_cap_enabled",
    "use_amp",
    "use_tail_disaggregation_head",
    "initialization",
    "monitor_mode",
    "risk_checkpoint_start_epoch",
    "risk_mae_ceiling",
    "risk_monitor",
    "scheduler_monitor",
    "scheduler_start_epoch",
}


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a JSON object")
    return config


def flatten_training_fields(config: dict) -> dict:
    fields = {
        "csv_path": config["csv_path"],
        "appliances": config["appliance_cols"],
    }
    for section in ("data", "model", "optimization", "loss", "trainer"):
        values = config.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be a JSON object")
        for key, value in values.items():
            if key in fields and fields[key] != value:
                raise ValueError(f"Conflicting values for {key}")
            fields[key] = value
    return fields


def build_command(config: dict, python: str = sys.executable) -> list[str]:
    saved_argv = sys.argv
    try:
        sys.argv = ["train_home7951.py"]
        defaults = vars(train_home7951.parse_args())
    finally:
        sys.argv = saved_argv

    fields = flatten_training_fields(config)
    unsupported = set(fields) - set(defaults) - DERIVED_FIELDS
    if unsupported:
        raise ValueError(f"Unsupported training settings: {sorted(unsupported)}")

    if fields["num_appliances"] != len(fields["appliances"]):
        raise ValueError("num_appliances does not match appliance_cols")
    if config["input_cols"] != train_home7951.DEFAULT_INPUT_COLS:
        raise ValueError("input_cols differs from the training script's input schema")
    if fields["input_dim"] != len(config["input_cols"]):
        raise ValueError("input_dim does not match input_cols")
    if fields["use_tail_disaggregation_head"] != (
        fields["target_mode"] == "history_tail"
    ):
        raise ValueError("use_tail_disaggregation_head conflicts with target_mode")
    if fields["nominal_rated_power_kw"] != fields["rated_power_kw"]:
        raise ValueError("nominal_rated_power_kw differs from rated_power_kw")
    if fields["initialization"] is not None:
        raise ValueError("A non-null initialization record needs an explicit loader")
    if fields["monitor_mode"] != "min":
        raise ValueError("The source training CLI selects a minimum monitor")
    if fields["scheduler_monitor"] != fields["monitor"]:
        raise ValueError("scheduler_monitor differs from the selected monitor")
    if fields["scheduler_start_epoch"] != 1:
        raise ValueError("scheduler_start_epoch differs from the CLI schedule")
    if fields["risk_monitor"] is not None or fields["risk_mae_ceiling"] is not None:
        raise ValueError("Risk monitor settings require a separate risk-stage recipe")

    ablation = config.get("ablation", {})
    for key, value in {
        "state_conditioned_power": fields["state_conditioned_power"],
        "bridge_residual_enabled": fields["use_bridge_residual"],
        "rated_power_cap_enabled": fields["rated_power_cap_enabled"],
        "lambda_agg": fields["lambda_agg"],
        "lambda_ghost": fields["lambda_ghost"],
        "lambda_peak": fields["lambda_peak"],
    }.items():
        if key in ablation and ablation[key] != value:
            raise ValueError(f"ablation.{key} conflicts with the training settings")

    run_dir = Path(config["run_dir"])
    if Path(config["checkpoint_dir"]) != run_dir / "checkpoints":
        raise ValueError("checkpoint_dir does not match run_dir")
    if Path(config["result_dir"]) != run_dir / "results":
        raise ValueError("result_dir does not match run_dir")

    command = [python, str(ROOT / "scripts" / "train_home7951.py")]
    for key, value in fields.items():
        if key in DERIVED_FIELDS or value is None or value == defaults.get(key):
            continue
        if isinstance(value, bool):
            command.append(f"--{key}" if value else f"--no-{key}")
        elif isinstance(value, list):
            command.append(f"--{key}")
            command.extend(str(item) for item in value)
        else:
            command.extend((f"--{key}", str(value)))

    if not fields["rated_power_cap_enabled"]:
        command.append("--disable_rated_power_cap")
    if not fields["use_amp"]:
        command.append("--no_amp")
    command.extend(("--output_dir", str(run_dir.parent), "--run_name", run_dir.name))
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    command = build_command(config)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return

    for label, value in (
        ("processed source CSV", config["csv_path"]),
        ("initialization checkpoint", config["model"].get("init_checkpoint")),
    ):
        if value is None:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
