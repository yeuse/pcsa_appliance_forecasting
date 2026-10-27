"""One audited model/data/loss definition for all training commands."""
from pathlib import Path
import json, hashlib, random
import numpy as np
import torch
from src.models.pisa import build_pisa_model_from_config
from src.models.baselines import (AggregateToApplianceTCN, HistoricalNILMSeq2Seq,
    ApplianceHistoryTCNForecaster, TwoStageNILMForecastPipeline)
from src.data.pecanstreet_dataset import build_single_home_datasets
from src.training.losses import PISALoss, PISALossConfig
ROOT = Path(__file__).resolve().parents[1]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def read_protocol():
    c = json.loads((ROOT / "training/protocol.json").read_text())
    if c["lambda_agg"] != 0 or c["window_rule"] != "history_only":
        raise ValueError("This protocol requires net-grid-safe loss and history-only grid windows")
    if c["on_weight_actual"] != 1 + c["active_power_weight"]:
        raise ValueError("Incorrect ON weight declaration")
    return c

def model_config(c, residual=True):
    return {"data": {"appliance_cols": c["appliances"], "rated_power_kw": c["caps"]},
        "model": {"input_dim":6, "num_appliances":4, "horizon":c["horizon"],
        "d_model":128, "n_heads":4, "encoder_layers":3,
        "use_bridge_residual":True, "state_conditioned_power":True,
        "state_gate_floor":c["state_gate_floor"], "state_power_blend":1.,
        "power_cap_mode":c["power_cap_mode"], "use_history_recon_refiner":False,
        "use_refined_history_residual_tcn":residual,
        "residual_tcn_hidden_dim":64, "residual_tcn_num_layers":3,
        "residual_tcn_history_source":"base", "residual_tcn_shared":False,
        "future_residual_fusion":"gated_residual", "future_residual_zero_init":True,
        "future_residual_gate_init":c["residual_gate_logit"]}}

def make_model(c, method, residual=True):
    cap = torch.tensor(c["caps"])
    if method == "pcsa": return build_pisa_model_from_config(model_config(c, residual))
    if method == "aggregate":
        m = AggregateToApplianceTCN(6, 4, c["horizon"], hidden_dim=64, num_layers=3, rated_power=cap)
    else:
        n = HistoricalNILMSeq2Seq(6,4,hidden_dim=64,num_layers=2,rated_power=cap)
        if method == "nilm": m = n
        elif method == "two_stage":
            f = ApplianceHistoryTCNForecaster(4,c["horizon"],hidden_dim=64,num_layers=3,rated_power=cap)
            m = TwoStageNILMForecastPipeline(n,f)
        else: raise ValueError(method)
    for child in m.modules():
        if hasattr(child, "rated_power"):
            child.power_cap_mode = c["power_cap_mode"]
            child.state_gate_floor = c["state_gate_floor"]
    return m

def datasets(c, home, reference=None):
    return build_single_home_datasets(ROOT/c["data_dir"]/f"home_{home}_2018_sep_1min.csv",
        input_window=c["input_window"], horizon=c["horizon"],
        state_thresholds=dict(zip(c["appliances"],c["thresholds"])),
        reference_bundle=reference, require_future_mains=False)

def objective(c, scales, history=False):
    kw = {name:0. for name in PISALossConfig.__dataclass_fields__ if name.startswith("lambda_")}
    kw.update(lambda_power=1.,lambda_state=c["lambda_state"],lambda_ghost=c["lambda_ghost"],
        lambda_amp_on=c["lambda_amp_on"], active_power_weight=c["active_power_weight"],
        aggregate_mode="none",active_recon_weight=0.,
        balanced_on_off_power_loss=c.get("balanced_on_off_power_loss",False))
    if history:
        kw.update(lambda_recon_power=c["lambda_history_power"],lambda_recon_state=c["lambda_history_state"])
    return PISALoss(scales, PISALossConfig(**kw), state_alpha=.5)

def history_objective(out,batch,scales,return_parts=False):
    mask=batch["hist_mask"]
    scale=torch.as_tensor(scales,device=mask.device).view(1,-1,1)
    err=(out["past_power"]/scale-batch["y_hist_power_scaled"]).abs()
    power=(err*mask).sum()/mask.sum().clamp_min(1)
    state=torch.nn.functional.binary_cross_entropy_with_logits(out["past_state_logits"],batch["y_hist_state"],reduction="none")
    state_sum=(state*mask).sum();denom=mask.sum().clamp_min(1)
    state=state_sum/denom
    weighted_state=.1*state_sum/denom  # Preserve the original floating-point operation order.
    total=power+weighted_state
    return (total,{"power":power.detach(),"state":state.detach(),"weighted_state":weighted_state.detach()}) if return_parts else total

def label_budget(ds, ids, history=True, future=True):
    starts=np.asarray(ds.indices,dtype=np.int64)[list(ids)]
    rows=np.unique((starts[:,None]+ds.input_window+np.arange(ds.horizon)).ravel()) if future else np.array([],dtype=int)
    hist=np.unique((starts[:,None]+np.arange(ds.input_window)).ravel()) if history else np.array([],dtype=int)
    coverage=np.union1d(rows,hist)
    union=coverage[(ds.target_available[coverage]>.5).any(axis=1)]
    times=ds.timestamps
    return {"windows":len(starts),"unique_future_timestamps":len(rows),
        "unique_history_timestamps":len(hist),"window_coverage_timestamps":len(coverage),
        "unique_label_timestamps_union":len(union),
        "valid_appliance_labels_by_channel":ds.target_available[union].sum(0).astype(int).tolist(),
        "first_label_time":str(times[union[0]]),"last_label_time":str(times[union[-1]]),
        "label_span_minutes":int(union[-1]-union[0]+1),
        "elapsed_from_split_start_minutes":int(union[-1]+1),
        "indices_sha256":hashlib.sha256(starts.astype("<i8").tobytes()).hexdigest()}

def support_caps(c, ds, ids):
    starts=np.asarray(ds.indices)[list(ids)]
    rows=np.unique((starts[:,None]+ds.input_window+np.arange(ds.horizon)).ravel())
    caps=np.array(c["caps"],dtype=float)
    for j in range(4):
        vals=ds.y_power[rows,j]; valid=(ds.target_available[rows,j]>.5)&(vals>0)
        if valid.any(): caps[j]=max(caps[j],float(np.quantile(vals[valid],c["cap_quantile"]))*(1+c["cap_margin"]))
    return caps.tolist()

def apply_caps(model,caps):
    # Preserve source TCN input normalization: only output constraints are recalibrated.
    with torch.no_grad():
        if hasattr(model,"constraint"):
            model.constraint.rated_power.copy_(torch.tensor(caps,device=model.constraint.rated_power.device).view(1,-1,1))
        else:
            for child in model.modules():
                if hasattr(child,"rated_power"):
                    child.rated_power.copy_(torch.tensor(caps,device=child.rated_power.device).view(1,-1,1))


def train_with_frozen_features(model):
    """Enable trainable routes while freezing dropout/BN in frozen subtrees."""
    model.train()
    for module in model.modules():
        parameters=list(module.parameters())
        if parameters and not any(p.requires_grad for p in parameters):
            module.eval()
