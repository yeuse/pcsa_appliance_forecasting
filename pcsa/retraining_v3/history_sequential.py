"""Reuse historical pretraining, then fit base and residual separately (160+80+80)."""
import argparse
import copy
import csv
import json
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import torch
from .common import datasets, make_model, seed_all, read_protocol
from .run import train_stage, save_json, lock_run, manifest, hash_file, loader, move
from .history_pretrain import HistoryView, add_residual, tensor_hash, run as pretrain_run
from scripts.train_home7951 import configure_trainable_parameters

LEGACY_ENTRY_SHA256 = "1837df612092dc5e645764e1e2185f2407808e2f4b30a42ebc311d73e3457322"


def lock_with_validation_fix(out, c):
    """Allow only this known assertion-only repair; retain original provenance."""
    path = out / "manifest.json"
    if path.exists():
        old = read(path)
        new = manifest(c)
        if old != new:
            name = "retraining_v3/history_sequential.py"
            expected = copy.deepcopy(old)
            expected["code"][name] = new["code"][name]
            if (old["code"].get(name) != LEGACY_ENTRY_SHA256 or expected != new
                    or read(out / "protocol.json") != c):
                raise RuntimeError("Changes exceed the known validation-check fix; use a new output directory")
            if list(out.glob("seed_*/history_sequential/residual/best.pt")):
                raise RuntimeError("Residual training already started; automatic repair is limited to the failed initialization check")
            save_json(out / "manifest_before_validation_fix.json", old)
            with zipfile.ZipFile(out / "validation_fix_source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as z:
                from .common import ROOT
                for f in new["code"]:z.write(ROOT / f, f)
            save_json(out / "validation_fix.json", {
                "reason":"Replace bitwise output equality with zero-residual and numerical-tolerance checks",
                "training_protocol_changed":False,
                "old_entry_sha256":LEGACY_ENTRY_SHA256,"new_entry_sha256":new["code"][name]})
            save_json(path, new)
            print("Applied validation-only repair; completed base stages will be reused.", flush=True)
    lock_run(out, c)


def check_initial_residual(base, model, batch, folder):
    """Require identical base tensors and zero residuals; tolerate FP32 roundoff."""
    base.eval();model.eval()
    state = model.state_dict()
    weights_equal = all(torch.equal(v, state[k]) for k,v in base.state_dict().items())
    with torch.no_grad():
        b = base(batch);m = model(batch)
    diagnostics = {"base_tensors_equal":weights_equal,"atol_kw":1e-6,"rtol":1e-5}
    for key in ("future_tcn_power_residual_raw","future_tcn_state_residual_logits"):
        diagnostics[key+"_max_abs"] = float(m[key].abs().max())
    pairs = {"base_route":(b["y_power"],m["base_y_power"]),
             "zero_residual_route":(m["base_y_power"],m["y_power"]),
             "final_prediction":(b["y_power"],m["y_power"])}
    for name,(x,y) in pairs.items():
        diagnostics[name+"_max_abs_diff_kw"] = float((x-y).abs().max())
        diagnostics[name+"_close"] = bool(torch.allclose(x,y,atol=1e-6,rtol=1e-5))
    save_json(folder / "initial_residual_check.json", diagnostics)
    if (not weights_equal or any(diagnostics[k+"_max_abs"] != 0 for k in
            ("future_tcn_power_residual_raw","future_tcn_state_residual_logits"))
            or not all(diagnostics[n+"_close"] for n in pairs)):
        raise RuntimeError("Residual initialization failed; measured differences: " + json.dumps(diagnostics))
    print("Residual initialization verified: " + json.dumps(diagnostics),flush=True)
    return diagnostics


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(source, out, device, seeds, smoke=False):
    source, out = source.resolve(), out.resolve()
    if out == source or source in out.parents:
        raise ValueError("Output must be outside the previous run")
    original = read(source / "protocol.json")
    if original.get("experiment") != "pcsa_history_pretraining_v1":
        raise ValueError("Expected the completed history_pretrain_v1 experiment")
    if original.get("smoke_only", False) != smoke:
        raise ValueError("Cannot mix smoke and formal experiments")
    old_manifest = read(source / "manifest.json")
    current = manifest(original)
    if old_manifest["data"] != current["data"]:
        raise ValueError("Source data changed")
    required = ["retraining_v3/" + n for n in
                ("common.py", "run.py", "diagnostics.py", "history_pretrain.py")]
    for name, value in old_manifest["code"].items():
        if (name.startswith("src/") or name in required) and current["code"].get(name) != value:
            raise ValueError("Source implementation changed: " + name)
    provenance = {}
    for seed in seeds:
        folder = source / f"seed_{seed}/history_pretrain"
        checkpoint = folder / "history/best.pt"
        if not (folder / "history/summary.json").is_file():
            raise ValueError(f"Historical training incomplete for seed {seed}")
        provenance[str(seed)] = {"checkpoint": str(checkpoint), "sha256": hash_file(checkpoint)}
    c = copy.deepcopy(original)
    c.update(experiment="pcsa_history_sequential_v1", experiment_seeds=seeds,
             sequential_epochs={"history_reused":160,"base":80,"residual":80},
             reused_history=provenance,
             schedules={"history_sequential":["history:160 reused", "frozen_history_base:80", "frozen_base_residual:80"]},
             budget_note="320 maximum passes including reused history; new work at most 160 passes")
    lock_with_validation_fix(out, c)
    save_json(out / "run_mode.json", {"mode":"smoke" if smoke else "validation_only"})
    ds, bundle = datasets(c, c["source_home"])
    save_json(out / "preprocessing.json", bundle.scalers.state_dict())
    save_json(out / "label_budget.json", read(source / "label_budget.json"))
    rows = []
    for seed in seeds:
        folder = out / f"seed_{seed}/history_sequential"
        prior = source / f"seed_{seed}/history_pretrain"
        seed_all(seed)
        base = make_model(c, "pcsa", False)
        if tensor_hash(base.state_dict()) != read(prior / "initialization.json")["base_state_sha256"]:
            raise ValueError("Initial base weights differ from the previous experiment")
        saved = torch.load(provenance[str(seed)]["checkpoint"], map_location="cpu")
        if saved["protocol"] != original or saved["seed"] != seed:
            raise ValueError("Checkpoint seed/protocol mismatch")
        HistoryView(base).load_state_dict(saved["state_dict"], strict=True)
        history_hash = tensor_hash(HistoryView(base).state_dict())
        configure_trainable_parameters(base, "full")
        for name, p in base.named_parameters():
            if name.startswith(("encoder.","past_recon_head.")):p.requires_grad = False
        first = train_stage(base, ds, c, folder / "base", device, 80, seed, smoke=smoke)
        if tensor_hash(HistoryView(base).state_dict()) != history_hash:
            raise RuntimeError("History changed during base prediction training")
        model = add_residual(base, c, seed)
        configure_trainable_parameters(model, "future_residual_tcn")
        # Zero-initialized residuals must preserve the selected base forecast.
        base.eval();model.to(device).eval()
        batch = move(next(iter(loader(ds["val"], 16))), device)
        initial_check = check_initial_residual(base, model, batch, folder)
        second = train_stage(model, ds, c, folder / "residual", device, 80, seed, smoke=smoke)
        if tensor_hash(HistoryView(model).state_dict()) != history_hash:
            raise RuntimeError("History changed during residual training")
        old = read(prior / "history/summary.json")
        save_json(folder / "reused_history_summary.json", old)
        save_json(folder / "history_validation.json", read(prior / "history_validation.json"))
        save_json(folder / "checks.json", {"history_unchanged":True,
            "initial_residual_matches_base_within_tolerance":True,
            "initial_residual_check":initial_check,
            "base_frozen_during_residual": second["frozen_parameter_max_update"] == 0,
            "frozen_buffers_unchanged": first["frozen_buffers_unchanged"] and second["frozen_buffers_unchanged"]})
        summaries = [old, first, second]
        row = {"seed":seed,"arm":"history_sequential",
               "forecast_val_mae_kw":second["best_val_macro_mae_kw"],
               "base_val_mae_kw":first["best_val_macro_mae_kw"],
               "previous_joint_val_mae_kw":read(prior / "forecast/summary.json")["best_val_macro_mae_kw"],
               "original_control_val_mae_kw":read(source / f"seed_{seed}/control/forecast/summary.json")["best_val_macro_mae_kw"],
               "history_selected_epoch":old["selected_epoch"],
               "base_selected_epoch":first["selected_epoch"],
               "residual_selected_epoch":second["selected_epoch"]}
        for k in ("epochs_run","optimizer_steps","training_seconds","total_wall_seconds"):
            row["pipeline_"+k] = sum(s[k] for s in summaries)
            row["new_work_"+k] = first[k]+second[k]
        rows.append(row)
        save_json(out / "comparison.json", rows)
        with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(rows)
        print(json.dumps(row),flush=True)
        del model, base
    for k in ("original_control_val_mae_kw","previous_joint_val_mae_kw","forecast_val_mae_kw"):
        v=[r[k] for r in rows]
        print(f"{k}: {np.mean(v):.6f} +/- {np.std(v,ddof=1) if len(v)>1 else 0.:.6f} kW",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--device",default="cuda")
    p.add_argument("--seeds",type=int,nargs="+",default=[7,42,123])
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args();device=torch.device(a.device)
    if len(set(a.seeds))!=len(a.seeds):p.error("Duplicate seeds")
    if not a.smoke and device.type!="cuda":p.error("Formal experiment requires CUDA")
    if device.type=="cuda" and not torch.cuda.is_available():p.error("CUDA unavailable")
    torch.set_num_threads(2)
    if a.smoke:
        with tempfile.TemporaryDirectory(prefix="history_sequential_") as temp:
            source=Path(temp)/"source"
            c=read_protocol();c.update(experiment="pcsa_history_pretraining_v1",smoke_only=True,validation_only=True)
            pretrain_run(c,source,device,a.seeds,True)
            run(source,Path(temp)/"result",device,a.seeds,True)
    else:
        if a.source_run is None or a.output is None:p.error("--source-run and --output required")
        run(a.source_run,a.output,device,a.seeds)


if __name__=="__main__":main()
