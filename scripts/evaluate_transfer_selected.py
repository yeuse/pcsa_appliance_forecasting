"""Read-only evaluation of validation-selected transfer checkpoints.

No optimizer, training, threshold search, cap fitting, or model selection.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import Subset
from data import build_single_home_datasets
from scripts.evaluate_transfer_history_and_forecast import (
    read_json, source_pisa_config_path, source_state_thresholds, validate_protocol,
)
from scripts.run_baseline_cross_home_transfer import build_models, apply_caps, load_checkpoint_strict
from scripts.run_pisa_cross_home_transfer import (
    build_source_model, apply_rated_power_caps,
    build_loaders,
)
from scripts.transfer_event_audit import evaluate_audited as evaluate


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_path(run, method, item, method_summary, config, key):
    selected = item["selected_model"]
    if method == "PISA" and selected == "two_stage_selected_checkpoint":
        return Path(item["selected_checkpoint"]).resolve()
    if selected == "few_shot_checkpoint":
        slug = "aggregate_tcn" if method == "Aggregate-to-appliance TCN" else "two_stage"
        return run / "checkpoints" / slug / f"fewshot_{key}" / "best.pt"
    if selected in {"source_zero_shot", "support_calibrated_source"}:
        return Path(method_summary["source_checkpoint"]).resolve()
    raise ValueError(f"Unsupported selection: {method}: {selected}")


def validate_mae(actual, expected, tolerance):
    if expected is None or not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
        raise ValueError("Missing or non-finite validation MAE")
    if abs(float(actual) - float(expected)) > tolerance:
        raise ValueError(f"Validation reproduction failed: {actual} vs stored {expected}; test NOT evaluated")


def make_model(job):
    config = copy.deepcopy(job["source_config"])
    if job["method"] == "PISA":
        transfer = job["config"]
        if transfer.get("enhanced_forecast_head", False):
            config.setdefault("model", {}).update(
                residual_tcn_use_future_context=True,
                residual_tcn_target_adapter_dim=int(transfer.get("target_adapter_dim", 32)),
                future_residual_fusion="learned_blend",
                future_base_blend_init=float(transfer.get("future_base_blend_init", 4.0)),
                target_output_calibration=True,
            )
        model = build_source_model(config)
    else:
        direct, two_stage = build_models(config)
        model = direct if job["method"] == "Aggregate-to-appliance TCN" else two_stage
    load_checkpoint_strict(model, job["checkpoint"])
    # Source-zero-shot must keep source caps. Other selections use the saved
    # support-only calibration; never estimate caps using validation/test data.
    if job["item"]["selected_model"] != "source_zero_shot":
        caps = job["item"]["power_cap_calibration"]["calibrated_caps_kw"]
        (apply_rated_power_caps if job["method"] == "PISA" else apply_caps)(model, caps)
    model.requires_grad_(False)
    return model


def prepare(args):
    plans = []
    for home in args.homes:
        pisa_run = args.outputs_root / "transfer_runs_two_stage" / f"7951_to_{home}_pisa_twostage_h120_f200_s42"
        pc = read_json(pisa_run / "config.json")
        ps = read_json(pisa_run / "results/transfer_summary.json")
        source_checkpoint = Path(pc["source_checkpoint"]).resolve()
        psc = read_json(source_pisa_config_path(pc, source_checkpoint))
        groups = [("PISA", pisa_run, pc, ps, psc, {"PISA": ps})]
        protocol = thresholds = None
        for label, directory, suffix in (
            ("baseline_original", "transfer_baselines", "baselines_s42"),
            ("baseline_joint", "transfer_baselines_joint", "baselines_joint_e200_val_s42"),
        ):
            run = args.outputs_root / directory / f"7951_to_{home}_{suffix}"
            config = read_json(run / "config.json")
            summary = read_json(run / "results/baseline_transfer_summary.json")
            source_config = read_json(Path(config["source_run_dir"]) / "config.json")
            protocol = validate_protocol(pc, config, ps, summary, psc, source_config)
            current_thresholds = source_state_thresholds(source_checkpoint, source_config, protocol[-1])
            if thresholds is not None and current_thresholds != thresholds:
                raise ValueError("Baseline state thresholds differ")
            thresholds = current_thresholds
            groups.append((label, run, config, summary, source_config, summary["methods"]))
        jobs = []
        for label, run, config, summary, source_config, methods in groups:
            for method, method_summary in methods.items():
                if method not in {"PISA", "Aggregate-to-appliance TCN", "Seq2Seq-NILM -> TCN"}:
                    raise ValueError(f"Unknown method: {method}")
                for fraction in args.fractions:
                    key = f"{fraction:.3f}"
                    item = method_summary["few_shot"][key]
                    checkpoint = selected_path(run, method, item, method_summary, config, key)
                    if not checkpoint.is_file():
                        raise FileNotFoundError(checkpoint)
                    if item["selected_model"] != "source_zero_shot":
                        caps = item["power_cap_calibration"]["calibrated_caps_kw"]
                        if len(caps) != len(protocol[-1]) or not all(math.isfinite(x) and x > 0 for x in caps):
                            raise ValueError("Invalid saved support caps")
                    if item.get("selected_validation_mae") is None:
                        raise ValueError("Missing selected validation MAE")
                    jobs.append(dict(group=label, method=method, config=config,
                                     source_config=source_config, item=item, key=key,
                                     checkpoint=checkpoint, run=str(run)))
        for path in protocol[:2]:
            if not path.is_file():
                raise FileNotFoundError(path)
        plans.append((home, protocol, thresholds, jobs))
    return plans


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs_root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--homes", nargs="+", default=["3039", "8386", "8565"])
    parser.add_argument("--fractions", nargs="+", type=float, default=[.01, .05, .1])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--validation_tolerance", type=float, default=5e-5)
    parser.add_argument("--smoke_windows", type=int, default=0,
                        help="Validation-only smoke check; never evaluates test or verifies full MAE")
    parser.add_argument("--preflight_only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.smoke_windows < 0:
        parser.error("Invalid batch/workers/smoke size")
    if not 0 <= args.validation_tolerance < 1 or not all(0 < f <= 1 for f in args.fractions):
        parser.error("Invalid tolerance/fractions")
    plans = prepare(args)
    print(f"Preflight OK: {sum(len(p[3]) for p in plans)} selected models", flush=True)
    if args.preflight_only:
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    rows = []
    comparison = []
    event_rows = []
    for home, protocol, thresholds, jobs in plans:
        source, target, window, horizon, stride, inputs, appliances = protocol
        common = dict(input_cols=inputs, appliance_cols=appliances, input_window=window,
                      horizon=horizon, stride=stride, target_mode="future",
                      state_thresholds=thresholds, drop_unavailable_windows=True)
        _, bundle = build_single_home_datasets(csv_path=source, **common)
        datasets, _ = build_single_home_datasets(csv_path=target, reference_bundle=bundle, **common)
        selected = {s: datasets[s] for s in ("val", "test")}
        if args.smoke_windows:
            selected = {"val": Subset(datasets["val"], range(min(args.smoke_windows, len(datasets["val"]))))}
        if any(len(d) == 0 for d in selected.values()):
            raise ValueError("Empty evaluation split")
        loaders = build_loaders(selected, args.batch_size, args.num_workers)
        data_hashes = {str(p): fingerprint(p) for p in (source, target)}
        for job in jobs:
            print(f"{home} {job['group']} {job['method']} {job['key']}", flush=True)
            model = make_model(job)
            use_amp = device.type == "cuda" and not job["config"].get("no_amp", False)
            results = {}
            for split, loader in loaders.items():
                metrics = evaluate(model, loader, device, thresholds, appliances, use_amp,
                                   trace_fusion=(split == "val" and job["method"] == "PISA"))
                if split == "val" and not args.smoke_windows:
                    validate_mae(metrics["flat"]["regression/macro_avg/MAE"],
                                 job["item"]["selected_validation_mae"], args.validation_tolerance)
                results[split] = metrics
            metadata = {k: job[k] for k in ("group", "method", "key", "run", "item", "config", "source_config")}
            metadata.update(home=home, checkpoint=str(job["checkpoint"]),
                            checkpoint_sha256=fingerprint(job["checkpoint"]), data_sha256=data_hashes,
                            split_windows={s: len(d) for s, d in selected.items()},
                            state_thresholds=thresholds, smoke_only=bool(args.smoke_windows),
                            input_window=window, horizon=horizon, stride=stride,
                            batch_size=args.batch_size, use_amp=use_amp, torch_version=torch.__version__,
                            evaluator_sha256=fingerprint(Path(__file__)))
            slug = "pisa" if job["method"] == "PISA" else ("aggregate_tcn" if job["method"].startswith("Aggregate") else "two_stage")
            save(args.output_dir / f"{home}_{job['group']}_{slug}_{job['key']}.json",
                 dict(protocol="selected_transfer_readonly_v2_event_audit", metadata=metadata, results=results))
            for split, metrics in results.items():
                audit = metrics.get("event_audit", {})
                for protocol_name, scores in audit.get("protocols", {}).items():
                    for matching in ("exact", "tolerance_2min"):
                        for appliance, values in scores[matching].items():
                            event_rows.append(dict(home=home, group=job["group"], method=job["method"],
                                fraction=job["key"], split=split, protocol=protocol_name,
                                historical_source=("not_used" if protocol_name.startswith("internal/") else
                                    "true_history_DIAGNOSTIC_ONLY" if "DIAGNOSTIC_ONLY" in protocol_name else
                                    audit["previous_state_source"]),
                                matching=matching, appliance=appliance,
                                **{key: values.get(key) for key in ("EventPrecision", "EventRecall", "EventF1", "EventTP", "EventFP", "EventFN")}))
                comparison.append(dict(home=home, group=job["group"], method=job["method"],
                                       fraction=job["key"], split=split,
                                       MAE_kW=metrics["flat"]["regression/macro_avg/MAE"],
                                       selected_model=job["item"]["selected_model"],
                                       selected_epoch=job["item"].get("selected_epoch")))
                for metric, value in metrics["flat"].items():
                    rows.append(dict(home=home, group=job["group"], method=job["method"],
                                     fraction=job["key"], split=split, metric=metric, value=value))
            with (args.output_dir / "metrics_long.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
                writer.writeheader()
                writer.writerows(comparison)
            if event_rows:
                with (args.output_dir / "event_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(event_rows[0]))
                    writer.writeheader()
                    writer.writerows(event_rows)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    save(args.output_dir / "COMPLETE.json", dict(smoke_only=bool(args.smoke_windows), models=sum(len(p[3]) for p in plans)))
    print(f"Complete: {args.output_dir}")


if __name__ == "__main__":
    main()
