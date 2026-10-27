"""Matched 320-pass PCSA training schedules, training/validation only."""
import argparse
import copy
import csv
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from .common import datasets, make_model, read_protocol, seed_all, objective, label_budget
from .run import train_stage, evaluate, loader, save_json, lock_run
from scripts.train_home7951 import configure_trainable_parameters


def tensor_hash(state):
    h = hashlib.sha256()
    for name, t in sorted(state.items()):
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class HistoryView(torch.nn.Module):
    """Only existing encoder/head, supervising exactly the residual TCN history input.

    PCSA's historical power is ungated; its historical state is a separate input.
    This view does not add parameters or a refiner and does not run future routes.
    """
    def __init__(self, model):
        super().__init__()
        if model.household_film is not None:
            raise ValueError("This experiment assumes the canonical no-FiLM architecture")
        self.encoder = model.encoder
        self.past_recon_head = model.past_recon_head
        self.constraint = model.constraint
        self.cap = model._cap_amplitude

    def forward(self, batch):
        out = self.past_recon_head(self.encoder(batch["x_hist"]))
        out["past_power"] = self.cap(out["past_power"], self.constraint.rated_power)
        out["history_power_is_ungated"] = True
        return out


def add_residual(base, c, seed):
    seed_all(seed)
    model = make_model(c, "pcsa", True)
    missing, extra = model.load_state_dict(base.state_dict(), strict=False)
    if extra or any(not n.startswith("refined_history_residual_tcn.") for n in missing):
        raise RuntimeError("Invalid base initialization")
    return model


def run(c, out, device, seeds, smoke):
    lock_run(out, c)
    save_json(out / "run_mode.json", {"mode": "smoke" if smoke else "validation_only"})
    ds, bundle = datasets(c, c["source_home"])
    save_json(out / "preprocessing.json", bundle.scalers.state_dict())
    save_json(out / "label_budget.json", {
        split: label_budget(ds[split], range(len(ds[split])), history=True, future=True)
        for split in ("train", "val")})
    rows = []
    for seed in seeds:
        hashes = []
        for arm in ("control", "history_pretrain"):
            folder = out / f"seed_{seed}" / arm
            seed_all(seed)
            base = make_model(c, "pcsa", False)
            hashes.append(tensor_hash(base.state_dict()))
            save_json(folder / "initialization.json", {"base_state_sha256": hashes[-1], "seed": seed})
            if arm == "control":
                configure_trainable_parameters(base, "full")
                first = train_stage(base, ds, c, folder / "base", device, 160, seed,
                                    history_aux=True, smoke=smoke)
                model = add_residual(base, c, seed)
                configure_trainable_parameters(model, "future_residual_tcn")
                second = train_stage(model, ds, c, folder / "forecast", device, 160, seed, smoke=smoke)
            else:
                view = HistoryView(base)
                first = train_stage(view, ds, c, folder / "history", device, 160, seed,
                                    history_only=True, smoke=smoke)
                model = add_residual(base, c, seed)
                configure_trainable_parameters(model, "full")
                for name, p in model.named_parameters():
                    if name.startswith(("encoder.", "past_recon_head.")):
                        p.requires_grad = False
                before = tensor_hash(HistoryView(model).state_dict())
                second = train_stage(model, ds, c, folder / "forecast", device, 160, seed, smoke=smoke)
                if tensor_hash(HistoryView(model).state_dict()) != before:
                    raise RuntimeError("Historical extractor changed during forecasting")
            model.to(device).eval()
            ids = np.linspace(0, len(ds["val"])-1, min(64,len(ds["val"])),dtype=int) if smoke else None
            val_loader = loader(ds["val"], 16 if smoke else 512, ids=ids)
            view = HistoryView(model).to(device)
            view.eval()
            with torch.no_grad():
                batch = next(iter(val_loader))
                inputs = {k: batch[k].to(device) for k in ("x_hist", "x_future_time")}
                actual = model(inputs)
                direct = view(inputs)
                if not torch.equal(actual["future_tcn_input_power"], direct["past_power"]):
                    raise RuntimeError("Pretraining history differs from actual forecasting input")
                if not torch.equal(actual["future_tcn_input_p_on"], direct["past_p_on"]):
                    raise RuntimeError("Pretraining states differ from actual forecasting input")
            hist = evaluate(view, val_loader, device, c, True,
                            objective(c, bundle.appliance_scales).to(device), bundle.appliance_scales)
            save_json(folder / "history_validation.json", hist)
            forecast = json.loads((folder / "forecast/selected_validation.json").read_text())
            row = {"seed": seed, "arm": arm, "forecast_val_mae_kw": forecast["macro_mae_kw"],
                   "history_val_mae_kw": hist["macro_mae_kw"],
                   "stage1_selected_epoch": first["selected_epoch"],
                   "stage2_selected_epoch": second["selected_epoch"],
                   "total_epochs_run": first["epochs_run"] + second["epochs_run"],
                   "total_optimizer_steps": first["optimizer_steps"] + second["optimizer_steps"],
                   "total_training_seconds": first["training_seconds"] + second["training_seconds"],
                   "total_wall_seconds": first["total_wall_seconds"] + second["total_wall_seconds"]}
            save_json(folder / "checks.json", {
                "history_view_equals_forecast_input": True,
                "forecast_frozen_parameters_unchanged": second["frozen_parameter_max_update"] == 0,
                "forecast_frozen_buffers_unchanged": second["frozen_buffers_unchanged"],
                "stage1_trainable_parameters": first["trainable_parameters"],
                "stage2_trainable_parameters": second["trainable_parameters"]})
            rows.append(row)
            save_json(out / "comparison.json", rows)
            with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row));writer.writeheader();writer.writerows(rows)
            print(json.dumps(row), flush=True)
            del view, model, base
        if hashes[0] != hashes[1]:raise RuntimeError("Unmatched initial weights")
    for arm in ("control", "history_pretrain"):
        values = [r["forecast_val_mae_kw"] for r in rows if r["arm"] == arm]
        print(f"{arm}: validation MAE {np.mean(values):.6f} +/- "
              f"{np.std(values, ddof=1) if len(values)>1 else 0.:.6f} kW", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", type=int, nargs="+", default=[7,42,123])
    p.add_argument("--smoke", action="store_true", help="Temporary CPU validation; auto-deletes weights")
    a = p.parse_args()
    device = torch.device(a.device)
    if not a.smoke and device.type != "cuda":raise RuntimeError("Full training requires CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    if not a.smoke and a.output is None:p.error("--output is required")
    if len(set(a.seeds)) != len(a.seeds):p.error("Duplicate seeds")
    torch.set_num_threads(2)
    c = copy.deepcopy(read_protocol())
    c.update(validation_only=True, experiment="pcsa_history_pretraining_v1",
             experiment_seeds=a.seeds, smoke_only=a.smoke,
             schedules={"control": ["base_future_with_history_aux:160", "frozen_base_residual:160"],
                        "history_pretrain": ["encoder_history_head:160", "frozen_history_all_forecast_routes:160"]},
             history_pretrain_loss="masked scaled L1 of capped history + 0.1 state BCE",
             history_pretrain_selection="minimum validation historical macro MAE in kW",
             budget_note="320 maximum passes per arm; actual passes, steps and compute may differ")
    if a.smoke:
        with tempfile.TemporaryDirectory(prefix="pcsa_history_pretrain_") as temp:
            run(c, Path(temp), device, a.seeds, True)
    else:run(c, a.output.resolve(), device, a.seeds, False)


if __name__ == "__main__":main()
