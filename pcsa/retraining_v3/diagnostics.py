"""Detached, mask-aware diagnostics. Counts are overlapping forecast positions."""
import torch
from .common import history_objective

class Diagnostics:
    def __init__(self, c, history=False):
        self.c=c; self.history=history; self.sums={}; self.counts={}
        self.losses={}; self.loss_samples=0; self.history_ungated=False

    @torch.no_grad()
    def update(self, out, batch, parts=None):
        hist=self.history
        y=batch["y_hist_power" if hist else "y_power"]
        state=batch["y_hist_state" if hist else "y_state"]
        mask=batch["hist_mask" if hist else "target_mask"]
        pred=out["past_power" if hist else "y_power"]
        prob=out.get("past_p_on" if hist else "p_on")
        amp=None if hist else out.get("amplitude_power")
        gate=None if prob is None else self.c["state_gate_floor"]+(1-self.c["state_gate_floor"])*prob
        # History output in this protocol is exactly smooth amplitude * soft gate.
        if hist and out.get("history_power_is_ungated",False):
            self.history_ungated=True;gate=None;amp=pred
        elif hist and gate is not None:amp=pred/gate.clamp_min(1e-8)
        values={"target_mean_kw":y,"prediction_mean_kw":pred,"mae_kw":(pred-y).abs(),"bias_kw":pred-y}
        if prob is not None:
            values.update(p_on_mean=prob,
                p_on_below_005_fraction=(prob<.05).float(),p_on_above_05_fraction=(prob>=.5).float())
            if gate is not None:values["soft_gate_mean"]=gate
        if amp is not None:
            values.update(amplitude_mean_kw=amp,amplitude_mae_kw=(amp-y).abs(),
                gate_suppression_kw=amp-pred)
        base=None if hist else out.get("base_y_power")
        if base is not None:
            values.update(base_prediction_mean_kw=base,base_mae_kw=(base-y).abs(),
                residual_signed_kw=pred-base,residual_absolute_kw=(pred-base).abs(),
                residual_mae_gain_kw=(base-y).abs()-(pred-y).abs())
            if "base_p_on" in out:values["base_p_on_mean"]=out["base_p_on"]
        for name,w in {"all":mask,"on":mask*state,"off":mask*(1-state)}.items():
            count=w.sum((0,2)).double().cpu()
            self.counts[name]=self.counts.get(name,torch.zeros_like(count))+count
            for key,v in values.items():
                z=(v.detach()*w).sum((0,2)).double().cpu()
                if not torch.isfinite(z).all():raise RuntimeError("Nonfinite diagnostic: "+key)
                index=(name,key);self.sums[index]=self.sums.get(index,torch.zeros_like(z))+z
        if parts is not None:
            n=int(y.shape[0]);self.loss_samples+=n
            for key,value in parts.items():
                value=float(value.detach()) if torch.is_tensor(value) else float(value)
                self.losses[key]=self.losses.get(key,0.)+value*n

    def result(self):
        groups={}
        for name,count in self.counts.items():
            row={"positions":count.tolist()}
            for (group,key),total in self.sums.items():
                if group==name:row[key]=[float(x/n) if n>0 else None for x,n in zip(total,count)]
            groups[name]=row
        return {"appliances":self.c["appliances"],"groups":groups,
            "loss_batch_means":{k:v/self.loss_samples for k,v in self.losses.items()},
            "loss_aggregation":"sample-weighted batch means; masked/ON losses are not global position means",
            "counts":"overlapping valid positions, not unique timestamps or independent events",
            "history_amplitude_derived_from_soft_gate":self.history and not self.history_ungated,
            "residual_gain_sign":"positive means final prediction improves absolute error over base"}

def loss_parts(out,batch,c,scales,loss_fn,history=False):
    if history:
        total,raw=history_objective(out,batch,scales,return_parts=True)
        return total,{"raw/history_power":raw["power"],"raw/history_state":raw["state"],
            "weighted/history_power":raw["power"],"weighted/history_state":raw["weighted_state"],"total":total.detach()}
    total,raw=loss_fn(out,batch)
    weights={"power":1.,"state":c["lambda_state"],"ghost":c["lambda_ghost"],
        "amp_on":c["lambda_amp_on"],"recon_power":loss_fn.cfg.lambda_recon_power,
        "recon_state":loss_fn.cfg.lambda_recon_state}
    parts={"raw/"+k[5:]:v for k,v in raw.items() if k.startswith("loss_") and k!="loss_total"}
    parts.update({"weighted/"+k:raw["loss_"+k]*v for k,v in weights.items()})
    parts["total"]=total.detach()
    return total,parts
