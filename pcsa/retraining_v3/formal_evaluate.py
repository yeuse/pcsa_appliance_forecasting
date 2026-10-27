"""Frozen source checkpoint evaluation and fixed-protocol target adaptation."""
import argparse
import copy
import csv
import json
from pathlib import Path
import numpy as np
import torch
from .common import read_protocol, datasets, make_model, objective, label_budget
from .run import (save_json, manifest, hash_file, lock_run, evaluate, loader,
                  benchmark, transfer)


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def validate_run(root, canonical, smoke):
    c=read(root/"protocol.json")
    for k,v in canonical.items():
        if c.get(k)!=v:raise ValueError(f"Training protocol differs at {root}: {k}")
    if c.get("balanced_on_off_power_loss",False) or c.get("diagnostic_variant","control")!="control":
        raise ValueError("Only the fixed control loss is supported")
    m=read(root/"manifest.json");now=manifest(c)
    if m["data"]!=now["data"]:raise ValueError("Source data changed: "+str(root))
    for k,v in m["code"].items():
        if (k.startswith("src/") or k in ("retraining_v3/common.py","retraining_v3/run.py")) and now["code"].get(k)!=v:
            raise ValueError("Model/training implementation changed: "+k)
    mode=read(root/"run_mode.json")["mode"]
    if (mode=="smoke")!=smoke:raise ValueError("Smoke/formal source mismatch")
    return c


def preflight(pcsa_root, baseline_root, seeds, smoke):
    canonical=read_protocol()
    pc=validate_run(pcsa_root,canonical,smoke)
    bc=validate_run(baseline_root,canonical,smoke)
    if pc.get("experiment")!="pcsa_history_sequential_v1":
        raise ValueError("Expected the selected sequential PCSA experiment")
    if pc.get("sequential_epochs")!={"history_reused":160,"base":80,"residual":80}:
        raise ValueError("PCSA schedule differs from the fixed 160+80+80 schedule")
    entries={}
    for seed in seeds:
        folder=pcsa_root/f"seed_{seed}/history_sequential"
        checks=read(folder/"checks.json")
        if not all(checks.get(k) is True for k in ("history_unchanged","base_frozen_during_residual","frozen_buffers_unchanged")):
            raise ValueError("PCSA freezing checks not passed")
        for method in ("pcsa","aggregate","two_stage"):
            root=folder/"residual" if method=="pcsa" else baseline_root/f"seed_{seed}/source/{method}"
            cp=root/"best.pt";summary=read(root/"summary.json")
            saved=torch.load(cp,map_location="cpu")
            expected=pc if method=="pcsa" else bc
            if (saved["protocol"]!=expected or saved["seed"]!=seed or summary["smoke_only"]!=smoke
                or saved["epoch"]!=summary["selected_epoch"]
                or saved["val_macro_mae_kw"]!=summary["best_val_macro_mae_kw"]):
                raise ValueError("Checkpoint selection/seed/protocol mismatch: "+str(cp))
            if method=="pcsa":
                budget=[read(folder/"reused_history_summary.json"),read(folder/"base/summary.json"),summary]
            elif method=="two_stage":
                budget=[read(baseline_root/f"seed_{seed}/source/nilm/summary.json"),summary]
            else:budget=[summary]
            entries[f"{seed}/{method}"]={"path":str(cp),"sha256":hash_file(cp),
                "selected_epoch":saved["epoch"],"original_training_protocol":expected,
                "stage_summaries":budget}
    return canonical,entries


def summarize(out,seeds,smoke):
    suffix="smoke_validation" if smoke else "test"
    rows=[]
    for seed in seeds:
        root=out/f"seed_{seed}"
        for method in ("pcsa","aggregate","two_stage"):
            p=root/f"source/{method}/{suffix}.json"
            if p.exists():rows.append({"seed":seed,"setting":"source","method":method,"mae_kw":read(p)["macro_mae_kw"]})
            for p in sorted(root.glob(f"target_*/{method}/**/*{suffix}.json")):
                home=p.parts[-3] if p.parent.name==method else p.parts[-4]
                setting=home+"/"+(p.parent.name+"/" if p.parent.name!=method else "")+p.stem.replace("_"+suffix,"")
                rows.append({"seed":seed,"setting":setting,"method":method,"mae_kw":read(p)["macro_mae_kw"]})
    save_json(out/"results_by_seed.json",rows)
    if rows:
        with (out/"results_by_seed.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups=[]
    for setting,method in sorted({(r["setting"],r["method"]) for r in rows}):
        values=[r["mae_kw"] for r in rows if r["setting"]==setting and r["method"]==method]
        row={"setting":setting,"method":method,"n_seeds":len(values),"mean_mae_kw":float(np.mean(values)),
             "sample_sd_kw":float(np.std(values,ddof=1)) if len(values)>1 else None}
        groups.append(row)
        sd=f"{row['sample_sd_kw']:.6f}" if row['sample_sd_kw'] is not None else "NA"
        print(f"{setting:44s} {method:12s} {row['mean_mae_kw']:.6f} +/- {sd} kW (n={len(values)})",flush=True)
    save_json(out/"results_summary.json",groups)
    if groups:
        with (out/"results_summary.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(groups[0]));w.writeheader();w.writerows(groups)


def run(pcsa_root,baseline_root,out,seeds,device,phase,smoke=False):
    pcsa_root,baseline_root,out=pcsa_root.resolve(),baseline_root.resolve(),out.resolve()
    for root in (pcsa_root,baseline_root):
        if out==root or root in out.parents:raise ValueError("Use a separate output directory")
    canonical,entries=preflight(pcsa_root,baseline_root,seeds,smoke)
    c=copy.deepcopy(canonical)
    c.update(experiment="pcsa_fixed_formal_evaluation_v1",validation_only=smoke,smoke_only=smoke,
             evaluation_seeds=seeds,source_epochs={**canonical["source_epochs"],"pcsa_history":160,"pcsa_base":80,"pcsa_residual":80},
             method_description="Two-stage framework: history pretraining, then forecasting with sequential base/residual optimization",
             checkpoint_provenance={k:{x:e[x] for x in ("path","sha256","selected_epoch")} for k,e in entries.items()})
    lock_run(out,c)
    save_json(out/"run_mode.json",{"mode":"smoke" if smoke else "full"})
    save_json(out/"source_training_provenance.json",entries)
    ds,bundle=datasets(c,c["source_home"])
    save_json(out/"source_label_budget.json",{s:label_budget(ds[s],range(len(ds[s])),True,True) for s in ("train","val")})
    split="val" if smoke else "test";suffix="smoke_validation" if smoke else "test"
    ids=np.linspace(0,len(ds[split])-1,min(64,len(ds[split])),dtype=int) if smoke else None
    for seed in seeds:
        for method in ("pcsa","aggregate","two_stage"):
            e=entries[f"{seed}/{method}"];folder=out/f"seed_{seed}/source/{method}"
            saved=torch.load(e["path"],map_location="cpu")
            model=make_model(c,method).to(device);model.load_state_dict(saved["state_dict"],strict=True)
            # Evaluation envelope only: original training protocol/cost are retained separately.
            checkpoint=folder/"best.pt"
            if checkpoint.exists():
                imported=torch.load(checkpoint,map_location="cpu")
                if (imported.get("imported_checkpoint_sha256")!=e["sha256"] or imported["protocol"]!=c
                    or not all(torch.equal(v,imported["state_dict"][k]) for k,v in saved["state_dict"].items())):
                    raise ValueError("Imported evaluation checkpoint changed")
            else:
                folder.mkdir(parents=True,exist_ok=True)
                torch.save({**saved,"protocol":c,"original_training_protocol":saved["protocol"],
                            "imported_checkpoint_sha256":e["sha256"]},checkpoint)
            if phase in ("source","suite") and not (folder/f"{suffix}.json").exists():
                loss=objective(c,bundle.appliance_scales).to(device)
                save_json(folder/f"{suffix}.json",evaluate(model,loader(ds[split],16 if smoke else 512,ids=ids),device,c,loss_fn=loss,scales=bundle.appliance_scales))
            if phase in ("source","suite") and not (folder/"compute.json").exists():
                save_json(folder/"compute.json",benchmark(model,ds["val"],device))
            del model
        if phase in ("transfer","suite"):transfer(c,out,seed,device,smoke)
        summarize(out,seeds,smoke)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pcsa-run",type=Path,required=True)
    p.add_argument("--baseline-run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[7,42,123])
    p.add_argument("--phase",choices=["source","transfer","suite"],default="suite")
    p.add_argument("--device",default="cuda")
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args();device=torch.device(a.device)
    if len(set(a.seeds))!=len(a.seeds):p.error("Duplicate seeds")
    if not a.smoke and device.type!="cuda":p.error("Formal evaluation requires CUDA")
    if device.type=="cuda" and not torch.cuda.is_available():p.error("CUDA unavailable")
    torch.set_num_threads(2)
    run(a.pcsa_run,a.baseline_run,a.output,a.seeds,device,a.phase,a.smoke)


if __name__=="__main__":main()
