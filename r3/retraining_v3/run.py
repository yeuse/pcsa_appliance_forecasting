"""Canonical entry point: python -m retraining_v3.run --phase smoke|audit|source|transfer|suite."""
import argparse, copy, hashlib, json, platform, time, zipfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .diagnostics import Diagnostics, loss_parts
from .common import (ROOT,read_protocol,seed_all,make_model,datasets,objective,
    history_objective,label_budget,support_caps,apply_caps,train_with_frozen_features)
from scripts.train_home7951 import configure_trainable_parameters
from scripts.run_pisa_cross_home_transfer import configure_transfer_parameters
from scripts.run_baseline_cross_home_transfer import configure_direct_heads,configure_two_stage_heads

def save_json(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def frozen_buffers_unchanged(model, snapshots):
    # Resolve aliases directly: named_buffers() deduplicates shared tensors.
    return all(torch.equal(value, model.get_buffer(name).detach().cpu())
               for name, value in snapshots.items())

def hash_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def manifest(c):
    files=list((ROOT/"src").rglob("*.py"))+[p for p in (ROOT/"retraining_v3").iterdir() if p.is_file()]
    files += [ROOT/"scripts"/n for n in ["train_home7951.py","run_pisa_cross_home_transfer.py","run_baseline_cross_home_transfer.py","prepare_austin_sep_four_homes.py"]]
    return {"code":{str(p.relative_to(ROOT)).replace("\\","/"):hash_file(p) for p in sorted(files)},
        "data":{str(h):hash_file(ROOT/c["data_dir"]/f"home_{h}_2018_sep_1min.csv") for h in [c["source_home"],*c["target_homes"]]}}

def lock_run(out,c):
    m=manifest(c); file=out/"manifest.json"
    if (out/"protocol.json").exists() and json.loads((out/"protocol.json").read_text())!=c:
        raise RuntimeError("Protocol/variant changed. Choose a NEW output directory.")
    if file.exists() and json.loads(file.read_text()) != m:
        raise RuntimeError("Code/data changed. Choose a NEW output directory; never mix protocols.")
    out.mkdir(parents=True,exist_ok=True); save_json(file,m); save_json(out/"protocol.json",c)
    save_json(out/"environment.json",{"python":platform.python_version(),"torch":torch.__version__,
        "cuda":torch.version.cuda,"device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})
    if not (out/"source_snapshot.zip").exists():
        with zipfile.ZipFile(out/"source_snapshot.zip","w",zipfile.ZIP_DEFLATED) as z:
            for f in m["code"]: z.write(ROOT/f,f)

def loader(ds,batch,shuffle=False,ids=None):
    return DataLoader(ds if ids is None else Subset(ds,list(ids)),batch_size=batch,
        shuffle=shuffle,num_workers=0,drop_last=False)

def move(b,device): return {k:v.to(device) if torch.is_tensor(v) else v for k,v in b.items()}

def group(name):
    if "refined_history_residual_tcn" in name:
        if "gate_logit" in name: return "residual_gates"
        if "residual_heads" in name: return "residual_heads"
        return "residual_features"
    if "history_recon_refiner" in name:return "unused_history_refiner"
    if "past_recon_head" in name:return "history_head"
    if "encoder" in name:return "encoder"
    return "other"

def gradient_summary(model):
    d={}
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        t=d.setdefault(group(n),{"tensors":0,"none":0,"nonzero":0,"norm_sum":0.})
        t["tensors"]+=1
        if p.grad is None:t["none"]+=1
        else:
            if not bool(torch.isfinite(p.grad).all()):raise RuntimeError(f"Nonfinite gradient: {n}")
            t["nonzero"]+=int(bool(p.grad.abs().sum()>0));t["norm_sum"]+=float(p.grad.norm())
    return d

@torch.no_grad()
def evaluate(model,dl,device,c,history=False,loss_fn=None,scales=None):
    model.eval(); err=torch.zeros(4,device=device); count=torch.zeros(4,device=device)
    onerr=err.clone(); offerr=err.clone(); offpower=err.clone(); onc=count.clone(); offc=count.clone(); sat=count.clone(); low_gate=count.clone()
    wall=time.perf_counter();diagnostics=Diagnostics(c,history)
    for batch in dl:
        b=move(batch,device); o=model(b)
        parts=loss_parts(o,b,c,scales,loss_fn,history)[1] if loss_fn is not None else None
        diagnostics.update(o,b,parts)
        pred=o["past_power"] if history else o["y_power"]
        y=b["y_hist_power"] if history else b["y_power"]
        mask=b["hist_mask"] if history else b["target_mask"]
        state=b["y_hist_state"] if history else b["y_state"]
        if not bool(torch.isfinite(pred).all()) or bool((pred<0).any()):raise RuntimeError("Invalid prediction")
        e=(pred-y).abs(); err+=(e*mask).sum((0,2));count+=mask.sum((0,2))
        onerr+=(e*mask*state).sum((0,2));onc+=(mask*state).sum((0,2))
        offerr+=(e*mask*(1-state)).sum((0,2));offc+=(mask*(1-state)).sum((0,2))
        offpower+=(pred*mask*(1-state)).sum((0,2))
        cap=torch.tensor(c["caps"],device=device).view(1,-1,1)
        if hasattr(model,"constraint"):cap=model.constraint.rated_power
        elif hasattr(model,"forecaster"):cap=model.forecaster.rated_power
        elif hasattr(model,"rated_power"):cap=model.rated_power
        if bool((pred>cap+1e-5).any()):raise RuntimeError("Output cap violated")
        amp=o.get("amplitude_power",pred)
        sat+=((amp>=.99*cap)*mask).sum((0,2))
        gate=o.get("p_on",o.get("past_p_on"))
        if gate is not None:low_gate+=((gate<.05)*mask).sum((0,2))
    if device.type=="cuda":torch.cuda.synchronize()
    mae=err/count.clamp_min(1)
    def masked_values(a,b):return [float(x/y) if y>0 else None for x,y in zip(a.cpu(),b.cpu())]
    return {"macro_mae_kw":float(mae.mean()),"mae_kw":mae.cpu().tolist(),
        "on_mae_kw":masked_values(onerr,onc),"off_mae_kw":masked_values(offerr,offc),
        "off_mean_predicted_kw":masked_values(offpower,offc),
        "low_state_probability_fraction":masked_values(low_gate,count),
        "near_cap_fraction":masked_values(sat,count),"forecast_positions":count.cpu().tolist(),
        "evaluation_wall_seconds":time.perf_counter()-wall,"diagnostics":diagnostics.result()}

def train_stage(model,ds,c,folder,device,epochs,seed,history_only=False,history_aux=False,smoke=False,ids=None,lr=None):
    folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/"best.pt"
    if (folder/"summary.json").exists():
        saved=torch.load(checkpoint,map_location=device)
        summary=json.loads((folder/"summary.json").read_text())
        if saved["protocol"]!=c or saved["seed"]!=seed or summary["smoke_only"]!=smoke:
            raise RuntimeError("Checkpoint protocol/seed/mode mismatch")
        model.to(device).load_state_dict(saved["state_dict"])
        return summary
    seed_all(seed);model.to(device)
    batch=16 if smoke else (c["transfer_batch_size"] if ids is not None else c["source_batch_size"])
    tids=list(ids) if ids is not None else list(range(len(ds["train"])))
    vids=list(range(len(ds["val"])))
    if smoke:
        tids=[tids[i] for i in np.linspace(0,len(tids)-1,min(64,len(tids)),dtype=int)]
        vids=[vids[i] for i in np.linspace(0,len(vids)-1,min(64,len(vids)),dtype=int)]
        epochs=3
    tl=loader(ds["train"],batch,True,tids);vl=loader(ds["val"],batch,False,vids)
    scales=ds["train"].bundle.appliance_scales
    loss_fn=objective(c,scales,history_aux).to(device)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=lr or c["lr"],weight_decay=c["weight_decay"])
    before={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    frozen_buffers={}
    for prefix,module in model.named_modules():
        ps=list(module.parameters())
        if ps and not any(p.requires_grad for p in ps):
            for name,b in module.named_buffers():
                frozen_buffers[(prefix+"."+name).strip(".")]=b.detach().cpu().clone()
    initial=evaluate(model,vl,device,c,history_only,loss_fn,scales)
    save_json(folder/"initial_validation.json",initial)
    best=initial["macro_mae_kw"];best_epoch=0;bad=0;steps=0;records=[];grad_records=[]
    def persist():torch.save({"state_dict":model.state_dict(),"protocol":c,"seed":seed,
        "epoch":best_epoch,"val_macro_mae_kw":best},checkpoint)
    persist();start=time.perf_counter();train_seconds=0.
    for epoch in range(1,epochs+1):
        train_with_frozen_features(model);loss_sum=0.;samples=0;ep_start=time.perf_counter()
        train_diagnostics=Diagnostics(c,history_only)
        for batch_data in tl:
            b=move(batch_data,device); opt.zero_grad(set_to_none=True);o=model(b)
            loss,parts=loss_parts(o,b,c,scales,loss_fn,history_only)
            train_diagnostics.update(o,b,parts)
            if not bool(torch.isfinite(loss)):raise RuntimeError("Nonfinite loss")
            loss.backward()
            if steps<3:grad_records.append({"step":steps, "groups":gradient_summary(model)})
            torch.nn.utils.clip_grad_norm_(params,c["grad_clip"],error_if_nonfinite=True)
            opt.step(); steps+=1; size=b["x_hist"].shape[0];samples+=size;loss_sum+=float(loss.detach())*size
        if device.type=="cuda":torch.cuda.synchronize()
        train_seconds+=time.perf_counter()-ep_start
        val=evaluate(model,vl,device,c,history_only,loss_fn,scales)
        if val["macro_mae_kw"]<best-c["early_stop_min_delta"]:
            best=val["macro_mae_kw"];best_epoch=epoch;bad=0;persist()
        else:bad+=1
        row={"epoch":epoch,"loss":loss_sum/samples,"val":val,"optimizer_steps":steps,"samples_this_epoch":samples,"train_diagnostics":train_diagnostics.result()};records.append(row)
        save_json(folder/"history.json",records)
        print(f"{folder.name} epoch={epoch} val_MAE={val['macro_mae_kw']:.6f} steps={steps}",flush=True)
        if not smoke and epoch>=c["min_epochs"] and bad>=c["patience"]:break
    delta={};frozen_max=0.
    for n,p in model.named_parameters():
        diff=float((p.detach().cpu()-before[n]).abs().max())
        if p.requires_grad:delta[group(n)]=max(delta.get(group(n),0.),diff)
        else:frozen_max=max(frozen_max,diff)
    if frozen_max!=0:raise RuntimeError("Frozen parameters changed")
    if not frozen_buffers_unchanged(model,frozen_buffers):
        raise RuntimeError("Frozen running statistics/buffers changed")
    summary={"smoke_only":smoke,"epochs_run":len(records),"selected_epoch":best_epoch,
        "best_val_macro_mae_kw":best,"optimizer_steps":steps,"training_seconds":train_seconds,
        "total_wall_seconds":time.perf_counter()-start,"trainable_parameters":sum(p.numel() for p in params),
        "total_parameters":sum(p.numel() for p in model.parameters()),"parameter_max_updates":delta,
        "frozen_parameter_max_update":frozen_max,"frozen_buffers_unchanged":True,"first_steps_gradients":grad_records,
        "train_labels":label_budget(ds["train"],tids,history_only or history_aux,future=not history_only),
        "validation_labels":label_budget(ds["val"],vids,history_only,future=not history_only),
        "selection":"validation MAE, including initial candidate; test unused"}
    model.load_state_dict(torch.load(checkpoint,map_location=device)["state_dict"])
    save_json(folder/"selected_train.json",evaluate(model,loader(ds["train"],batch,ids=tids),device,c,history_only,loss_fn,scales))
    save_json(folder/"selected_validation.json",evaluate(model,vl,device,c,history_only,loss_fn,scales))
    save_json(folder/"summary.json",summary); return summary

@torch.no_grad()
def benchmark(model,ds,device):
    b=move(next(iter(loader(ds,1))),device);model.eval()
    for _ in range(5):model(b)
    if device.type=="cuda":torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device)
    baseline=torch.cuda.memory_allocated(device) if device.type=="cuda" else None
    times=[]
    for _ in range(20):
        start=time.perf_counter();model(b)
        if device.type=="cuda":torch.cuda.synchronize()
        times.append(1000*(time.perf_counter()-start))
    return {"batch_size":1,"warmup":5,"repetitions":20,"precision":"float32",
        "forward_latency_ms_mean":float(np.mean(times)),"forward_latency_ms_median":float(np.median(times)),
        "incremental_peak_cuda_bytes":torch.cuda.max_memory_allocated(device)-baseline if baseline is not None else None,
        "parameter_count":sum(p.numel() for p in model.parameters()),
        "model_tensor_bytes":sum(t.numel()*t.element_size() for t in list(model.parameters())+list(model.buffers())),
        "device":str(device),"includes_data_loading":False}

def source(c,out,seed,device,smoke=False):
    ds,bundle=datasets(c,c["source_home"]);basepath=out/f"seed_{seed}"/"source"
    save_json(basepath/"preprocessing.json",bundle.scalers.state_dict())
    seed_all(seed);base=make_model(c,"pcsa",False)
    configure_trainable_parameters(base,"full")
    train_stage(base,ds,c,basepath/"pcsa_base",device,c["source_epochs"]["pcsa_base"],seed,history_aux=True,smoke=smoke)
    seed_all(seed);pcsa=make_model(c,"pcsa",True)
    missing,extra=pcsa.load_state_dict(base.state_dict(),strict=False)
    if extra or any(not n.startswith("refined_history_residual_tcn.") for n in missing):raise RuntimeError("Invalid base initialization")
    configure_trainable_parameters(pcsa,"future_residual_tcn")
    train_stage(pcsa,ds,c,basepath/"pcsa",device,c["source_epochs"]["pcsa_residual"],seed,smoke=smoke)
    seed_all(seed);nilm=make_model(c,"nilm")
    train_stage(nilm,ds,c,basepath/"nilm",device,c["source_epochs"]["nilm"],seed,history_only=True,smoke=smoke)
    seed_all(seed);two=make_model(c,"two_stage");two.nilm_model.load_state_dict(nilm.state_dict())
    train_stage(two,ds,c,basepath/"two_stage",device,c["source_epochs"]["two_stage_forecast"],seed,smoke=smoke)
    seed_all(seed);agg=make_model(c,"aggregate")
    train_stage(agg,ds,c,basepath/"aggregate",device,c["source_epochs"]["aggregate"],seed,smoke=smoke)
    split="val" if smoke or c.get("validation_only",False) else "test"
    for name,m in [("pcsa",pcsa),("aggregate",agg),("two_stage",two)]:
        ids=np.linspace(0,len(ds[split])-1,min(64,len(ds[split])),dtype=int) if smoke else None
        save_json(basepath/name/("smoke_validation.json" if smoke else "validation.json" if c.get("validation_only",False) else "test.json"),evaluate(m.to(device),loader(ds[split],64 if smoke else 512,ids=ids),device,c))
        save_json(basepath/name/"compute.json",benchmark(m,ds["val"],device))
    return ds,bundle

def transfer(c,out,seed,device,smoke=False):
    _,bundle=datasets(c,c["source_home"])
    for home in (c["target_homes"][:1] if smoke else c["target_homes"]):
        ds,_=datasets(c,home,bundle)
        for method in ["pcsa","aggregate","two_stage"]:
            m=make_model(c,method).to(device)
            cp=out/f"seed_{seed}"/"source"/method/"best.pt"
            if not cp.exists():raise RuntimeError(f"Run source first: {cp}")
            state=torch.load(cp,map_location=device)
            if state["protocol"]!=c or state["seed"]!=seed:raise RuntimeError("Source protocol/seed mismatch")
            m.load_state_dict(state["state_dict"])
            root=out/f"seed_{seed}"/f"target_{home}"/method
            vids=np.linspace(0,len(ds["val"])-1,min(64,len(ds["val"])),dtype=int) if smoke else None
            vl=loader(ds["val"],64 if smoke else 512,ids=vids)
            test=vl if smoke or c.get("validation_only",False) else loader(ds["test"],512)
            suffix="smoke_validation" if smoke else "validation" if c.get("validation_only",False) else "test"
            save_json(root/f"zero_shot_{suffix}.json",evaluate(m,test,device,c))
            for frac in (c["support_fractions"][:1] if smoke else c["support_fractions"]):
                ids=list(range(max(1,round(len(ds["train"])*frac))))
                caps=support_caps(c,ds["train"],ids);candidate=copy.deepcopy(m);apply_caps(candidate,caps)
                folder=root/f"support_{frac:.3f}"
                save_json(folder/"budget.json",{"smoke_only":smoke,"scope":"planned support and validation; actual optimizer subset in summary.json",
                    "training_fraction_only":frac,"caps_kw":caps,
                    "training":label_budget(ds["train"],ids,method!="aggregate"),
                    "calibration":label_budget(ds["train"],ids,False),
                    "validation":label_budget(ds["val"],range(len(ds["val"])),False),
                    "calibration_labels_subset_of_training":True,
                    "validation_additional_to_support":True})
                cap_val=evaluate(candidate,vl,device,c)
                save_json(folder/f"cap_only_{suffix}.json",evaluate(candidate,test,device,c))
                if method=="pcsa":configure_transfer_parameters(candidate,"history_forecast_heads")
                elif method=="aggregate":configure_direct_heads(candidate)
                else:configure_two_stage_heads(candidate)
                train_stage(candidate,ds,c,folder,device,c["transfer_epochs"],seed,history_aux=method!="aggregate",ids=ids,lr=c["transfer_lr"],smoke=smoke)
                # Initial cap-only checkpoint participates in the same validation selection.
                # train_stage already wrote validation with full loss diagnostics.
                if suffix!="validation":
                    save_json(folder/f"selected_{suffix}.json",evaluate(candidate,test,device,c))
                save_json(folder/"cap_only_validation.json",cap_val)

def audit(c,out):
    from scripts.prepare_austin_sep_four_homes import usable_windows
    result={};_,ref=datasets(c,c["source_home"])
    for home in [c["source_home"],*c["target_homes"]]:
        ds,_=datasets(c,home,None if home==c["source_home"] else ref);r={}
        for split,d in ds.items():
            count=usable_windows(d.mains_available>.5,d.target_available>.5,c["input_window"],c["horizon"],False)
            if count!=len(d):raise RuntimeError("Preparation/loader window mismatch")
            r[split]={"windows":len(d),"grid_unavailable_minutes":int((d.mains_available<.5).sum()),
                "grid_exclusion_fraction":float((d.mains_available<.5).mean()),
                "appliance_unavailable_fraction":(d.target_available<.5).mean(0).tolist(),
                "labels":label_budget(d,range(len(d)),True)}
        r["supports"]={str(f):{"training":label_budget(ds["train"],range(max(1,round(len(ds["train"])*f))),True),
            "caps":support_caps(c,ds["train"],range(max(1,round(len(ds["train"])*f))))} for f in c["support_fractions"]}
        result[str(home)]=r
    save_json(out/"data_audit.json",result)
    print("Data audit complete: preparation and loader window counts agree",flush=True)

def variant_protocol(c,variant):
    c=copy.deepcopy(c);c["validation_only"]=True;c["diagnostic_variant"]=variant
    if variant=="balanced_power":c["balanced_on_off_power_loss"]=True
    elif variant=="no_amp_on":c["lambda_amp_on"]=0.
    elif variant=="old_residual_gate":c["residual_gate_logit"]=-3.
    elif variant!="control":raise ValueError(variant)
    return c

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase",choices=["audit","smoke","source","transfer","suite"],required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seeds",type=int,nargs="+",default=None)
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--validation-only",action="store_true")
    ap.add_argument("--variant",choices=["control","balanced_power","no_amp_on","old_residual_gate"],default="control")
    args=ap.parse_args();c=read_protocol();out=args.output.resolve();device=torch.device(args.device)
    if args.variant!="control" and not args.validation_only:
        raise RuntimeError("Single-factor variants require --validation-only")
    if args.validation_only:
        c=variant_protocol(c,args.variant)
    if args.phase not in {"smoke","audit"} and device.type!="cuda":
        raise RuntimeError("Full experiments require --device cuda; CPU is reserved for smoke/audit")
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(2);lock_run(out,c)
    # Prevent smoke checkpoints from ever feeding a full experiment.
    mode="smoke" if args.phase=="smoke" else "validation" if args.validation_only else "full"
    marker=out/"run_mode.json"
    if marker.exists() and json.loads(marker.read_text())["mode"]!=mode:raise RuntimeError("Do not mix smoke/full directories")
    save_json(marker,{"mode":mode})
    if args.phase in {"audit","smoke","suite"}:audit(c,out)
    seeds=args.seeds or ([7] if args.phase=="smoke" else c["seeds"])
    for seed in seeds:
        if args.phase in {"smoke","source","suite"}:source(c,out,seed,device,args.phase=="smoke")
        if args.phase in {"transfer","suite","smoke"}:transfer(c,out,seed,device,args.phase=="smoke")

if __name__=="__main__":main()
