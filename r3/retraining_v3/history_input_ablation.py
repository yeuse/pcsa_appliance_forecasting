"""Frozen history input comparison; common predictor, TRAIN/VAL only."""
import argparse,copy,hashlib,json,csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,Subset
from .common import datasets,make_model,seed_all,label_budget
from .run import train_stage,save_json,lock_run,hash_file,manifest
from src.models.baselines import ApplianceHistoryTCNForecaster

ARMS=("pcsa_history","nilm_history","oracle_history")

class CachedHistory(Dataset):
    def __init__(self,base,ids,power,state):
        self.base=base;self.ids=list(ids);self.power=power;self.prob=state
        self.indices=np.asarray(base.indices)[self.ids].tolist()
    def __len__(self):return len(self.ids)
    def __getattr__(self,name):return getattr(self.base,name)
    def __getitem__(self,index):
        b=dict(self.base[self.ids[index]])
        b["appliance_history_power"]=self.power[index]
        b["appliance_history_state"]=self.prob[index]
        return b

class CommonPredictor(torch.nn.Module):
    def __init__(self,c):
        super().__init__()
        self.forecaster=ApplianceHistoryTCNForecaster(4,c["horizon"],hidden_dim=64,num_layers=3,
            rated_power=torch.tensor(c["caps"]))
        self.forecaster.power_cap_mode=c["power_cap_mode"]
        self.forecaster.state_gate_floor=c["state_gate_floor"]
    def forward(self,b):
        # Explicit keys prevent fallback to ground-truth histories in learned arms.
        return self.forecaster({k:b[k] for k in
            ["appliance_history_power","appliance_history_state","x_future_time"]})

def complete_history_ids(ds,smoke):
    bad=(ds.target_available<.5).any(axis=1).astype(np.int64)
    prefix=np.concatenate(([0],np.cumsum(bad)))
    starts=np.asarray(ds.indices,dtype=np.int64)
    ids=np.flatnonzero(prefix[starts+ds.input_window]-prefix[starts]==0).tolist()
    if not ids:raise RuntimeError("No complete-history windows")
    if smoke:ids=[ids[i] for i in np.linspace(0,len(ids)-1,min(64,len(ids)),dtype=int)]
    return ids

def tensor_hash(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def cache_history(ds,ids,extractor,arm,device):
    powers=[];probs=[];sums={};counts={}
    scales=torch.as_tensor(ds.bundle.appliance_scales,device=device).view(1,-1,1)
    if extractor is not None:extractor.eval()
    for b in DataLoader(Subset(ds,ids),batch_size=128,shuffle=False,num_workers=0):
        true=b["y_hist_power"].to(device);state=b["y_hist_state"].to(device)
        if not bool((b["hist_mask"]>.5).all()):raise RuntimeError("Incomplete oracle history")
        if arm=="oracle_history":power=true;prob=state
        else:
            inputs={k:b[k].to(device) for k in ["x_hist","x_future_time"]}
            o=extractor(inputs)
            if arm=="pcsa_history":power=o["future_tcn_input_power"];prob=o["future_tcn_input_p_on"]
            else:power=o["past_power"];prob=o["past_p_on"]
        if not torch.isfinite(power).all() or not torch.isfinite(prob).all():raise RuntimeError("Invalid cached history")
        powers.append(power.cpu());probs.append(prob.cpu())
        for group,w in {"all":torch.ones_like(state),"on":state,"off":1-state}.items():
            n=w.sum((0,2)).double().cpu();counts[group]=counts.get(group,torch.zeros_like(n))+n
            values={"mae_kw":(power-true).abs(),"scaled_mae":(power-true).abs()/scales,
                "prediction_mean_kw":power,"target_mean_kw":true,"p_on_mean":prob}
            for key,value in values.items():
                x=(value*w).sum((0,2)).double().cpu();idx=(group,key)
                sums[idx]=sums.get(idx,torch.zeros_like(x))+x
    stats={}
    for group,n in counts.items():
        stats[group]={"positions":n.tolist()}
        for (g,key),v in sums.items():
            if g==group:stats[group][key]=[float(x/z) if z>0 else None for x,z in zip(v,n)]
    return CachedHistory(ds,ids,torch.cat(powers),torch.cat(probs)),stats

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--seeds",type=int,nargs="+",default=[7,42,123])
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args();root=a.source_run.resolve();out=a.output.resolve();device=torch.device(a.device)
    if out==root or root in out.parents:raise ValueError("Use an output outside the source run")
    if device.type!="cuda" and not a.smoke:raise RuntimeError("CPU supports --smoke only")
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(2)
    source_c=json.loads((root/"protocol.json").read_text())
    original=json.loads((root/"manifest.json").read_text());current=manifest(source_c)
    if original["data"]!=current["data"]:raise RuntimeError("Source data hashes differ")
    for name,h in original["code"].items():
        if name.startswith("src/") and current["code"].get(name)!=h:raise RuntimeError("Source implementation differs: "+name)
    mode=json.loads((root/"run_mode.json").read_text())["mode"]
    if mode=="smoke" and not a.smoke:raise RuntimeError("Full experiment cannot use smoke source checkpoints")
    cps={}
    for seed in a.seeds:
        for method in ["pcsa","nilm"]:
            cp=root/f"seed_{seed}"/"source"/method/"best.pt"
            if not cp.exists():raise FileNotFoundError(cp)
            cps[str(cp.relative_to(root))]=hash_file(cp)
    c=copy.deepcopy(source_c)
    c.update(validation_only=True,history_input_comparison={"version":1,"smoke":a.smoke,
        "source_run":str(root),"source_checkpoint_sha256":cps,"source_mode":mode,
        "seeds":a.seeds,"arms":list(ARMS),"history_channels":"power and state probability",
        "window_rule":"original future/grid rules plus all appliance history labels available",
        "oracle":"privileged diagnostic reference; not a deployable method or guaranteed bound"})
    lock_run(out,c);save_json(out/"run_mode.json",{"mode":"smoke" if a.smoke else "validation"})
    ds,_=datasets(source_c,source_c["source_home"])
    ids={s:complete_history_ids(ds[s],a.smoke) for s in ["train","val"]}
    save_json(out/"window_audit.json",{s:{"original_windows":len(ds[s]),"used_windows":len(ids[s]),
        "labels":label_budget(ds[s],ids[s],True),"smoke_subset":a.smoke} for s in ids})
    comparison=[]
    for seed in a.seeds:
        seed_all(seed);initial=CommonPredictor(c).state_dict();initial_hash=tensor_hash(initial)
        for arm in ARMS:
            print(f"seed={seed} arm={arm}: caching fixed TRAIN/VAL histories",flush=True)
            folder=out/f"seed_{seed}"/arm;extractor=None
            if arm!="oracle_history":
                method="pcsa" if arm=="pcsa_history" else "nilm"
                cp=root/f"seed_{seed}"/"source"/method/"best.pt"
                saved=torch.load(cp,map_location=device,weights_only=False)
                if saved["protocol"]!=source_c or saved["seed"]!=seed:raise RuntimeError("Source checkpoint mismatch")
                extractor=make_model(source_c,method).to(device)
                extractor.load_state_dict(saved["state_dict"],strict=True)
                for param in extractor.parameters():param.requires_grad_(False)
                frozen_hash=tensor_hash(extractor.state_dict())
            cached={}
            for split in ["train","val"]:
                cached[split],stats=cache_history(ds[split],ids[split],extractor,arm,device)
                save_json(folder/(split+"_history_quality.json"),stats)
            if extractor is not None:
                if frozen_hash!=tensor_hash(extractor.state_dict()):raise RuntimeError("Frozen extractor changed")
                del extractor
            model=CommonPredictor(c);model.load_state_dict(initial)
            if tensor_hash(model.state_dict())!=initial_hash:raise RuntimeError("Predictor initialization mismatch")
            save_json(folder/"input_provenance.json",{"arm":arm,"initial_predictor_sha256":initial_hash,
                "history_power_sha256":{s:tensor_hash({"power":cached[s].power,"state":cached[s].prob}) for s in cached},
                "oracle_uses_ground_truth_history":arm=="oracle_history","extractor_frozen":True,
                "max_epochs":c["source_epochs"]["two_stage_forecast"],"source_pretraining_budget_excluded":True})
            summary=train_stage(model,cached,c,folder,device,c["source_epochs"]["two_stage_forecast"],seed,smoke=a.smoke)
            result=json.loads((folder/"selected_validation.json").read_text())
            row={"seed":seed,"arm":arm,"macro_mae_kw":result["macro_mae_kw"],
                "selected_epoch":summary["selected_epoch"],"epochs_run":summary["epochs_run"]}
            for j,name in enumerate(c["appliances"]):
                row[name+"_mae_kw"]=result["mae_kw"][j]
                row[name+"_on_mae_kw"]=result["on_mae_kw"][j]
                row[name+"_off_mae_kw"]=result["off_mae_kw"][j]
            comparison.append(row)
            save_json(out/"comparison.json",comparison)
            print(f"seed={seed} arm={arm}: selected validation MAE={result['macro_mae_kw']:.6f}",flush=True)
            del model,cached
    with (out/"comparison.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
    for arm in ARMS:
        vals=[r["macro_mae_kw"] for r in comparison if r["arm"]==arm]
        print(f"{arm}: mean={np.mean(vals):.6f}, sample_sd={np.std(vals,ddof=1) if len(vals)>1 else float('nan'):.6f}")

if __name__=="__main__":main()
