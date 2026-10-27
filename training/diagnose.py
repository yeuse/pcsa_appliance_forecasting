"""Read existing checkpoints; evaluate TRAIN/VAL only, never mutate the source run."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .common import datasets,make_model,objective,label_budget,seed_all
from .run import evaluate,loader,save_json,hash_file,lock_run

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--scope",choices=["source","all"],default="source")
    p.add_argument("--seeds",type=int,nargs="+",default=[7,42,123])
    p.add_argument("--max-windows",type=int,default=None)
    a=p.parse_args();root=a.source_run.resolve();out=a.output.resolve()
    if out==root or root in out.parents:raise ValueError("Diagnostics must be outside the original run")
    if a.max_windows is not None and a.max_windows<1:raise ValueError("max-windows must be positive")
    c=json.loads((root/"protocol.json").read_text());device=torch.device(a.device)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(2)
    checkpoints=[]
    for seed in a.seeds:
        for name in ["pcsa_base","pcsa","nilm","two_stage","aggregate"]:
            checkpoints.append(root/f"seed_{seed}"/"source"/name/"best.pt")
        if a.scope=="all":
            for home in c["target_homes"]:
                for method in ["pcsa","aggregate","two_stage"]:
                    for frac in c["support_fractions"]:
                        checkpoints.append(root/f"seed_{seed}"/f"target_{home}"/method/f"support_{frac:.3f}"/"best.pt")
    if any(not p.exists() for p in checkpoints):raise FileNotFoundError("Missing checkpoint in requested scope")
    provenance={"source_run":str(root),"scope":a.scope,"seeds":a.seeds,"max_windows":a.max_windows,
        "source_run_mode":json.loads((root/"run_mode.json").read_text()),
        "checkpoint_sha256":{str(p.relative_to(root)):hash_file(p) for p in checkpoints},
        "source_manifest_sha256":hash_file(root/"manifest.json"),"splits":["train","val"]}
    if (out/"provenance.json").exists() and json.loads((out/"provenance.json").read_text())!=provenance:
        raise RuntimeError("Changed diagnostic inputs; use a new output directory")
    lock_run(out,c)
    original=json.loads((root/"manifest.json").read_text())
    current=json.loads((out/"manifest.json").read_text())
    if original["data"]!=current["data"]:raise RuntimeError("Source data hashes differ")
    for name,h in original["code"].items():
        if name.startswith("src/") and current["code"].get(name)!=h:
            raise RuntimeError("Source model/data/loss implementation changed: "+name)
    save_json(out/"provenance.json",provenance)
    source_ds,bundle=datasets(c,c["source_home"]);cache={c["source_home"]:source_ds}
    for cp in checkpoints:
        rel=cp.relative_to(root);parts=rel.parts;seed=int(parts[0].split("_")[1]);is_source=parts[1]=="source"
        name=parts[2];method="pcsa" if name=="pcsa_base" else name
        home=c["source_home"] if is_source else int(parts[1].split("_")[1])
        if home not in cache:cache[home]=datasets(c,home,bundle)[0]
        ds=cache[home];seed_all(seed);model=make_model(c,method,residual=name!="pcsa_base").to(device)
        state=torch.load(cp,map_location=device,weights_only=False)
        if state["seed"]!=seed or state["protocol"]!=c:raise RuntimeError("Checkpoint seed/protocol mismatch")
        model.load_state_dict(state["state_dict"],strict=True)
        history=name=="nilm";history_aux=name=="pcsa_base" or (not is_source and method!="aggregate")
        scales=ds["train"].bundle.appliance_scales;loss=objective(c,scales,history_aux).to(device)
        folder=out/rel.parent
        for split in ["train","val"]:
            ids=list(range(len(ds[split])))
            if split=="train" and not is_source:
                fraction=float(parts[3].removeprefix("support_"));ids=ids[:max(1,round(len(ids)*fraction))]
            if a.max_windows is not None:
                ids=[ids[i] for i in np.linspace(0,len(ids)-1,min(len(ids),a.max_windows),dtype=int)]
            result=evaluate(model,loader(ds[split],256,ids=ids),device,c,history,loss,scales)
            result["checkpoint_epoch"]=state["epoch"]
            result["labels"]=label_budget(ds[split],ids,history or (history_aux and split=="train"),future=not history)
            result["diagnostic_subset_only"]=a.max_windows is not None
            save_json(folder/(split+"_diagnostics.json"),result)
        print(str(rel.parent)+" TRAIN/VAL diagnostics complete",flush=True)

if __name__=="__main__":main()
