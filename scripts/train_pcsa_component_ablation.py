"""Source PCSA ablations; frozen historical pretraining, validation-selected forecasts.
No residual: history160 + base80 (max240); no future state gate:160+80+80.
State loss and historical state inputs remain present in the no-gate arm.
"""
import argparse,copy,json,sys,csv,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--history-run',type=Path)
    ap.add_argument('--formal-run',type=Path)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--seeds',type=int,nargs='+',default=[7,42,123])
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    import torch,numpy as np
    from training.common import ROOT as PROJECT,read_protocol,datasets,make_model,seed_all,objective
    from training.run import train_stage,save_json,lock_run,manifest,hash_file,evaluate,loader,move
    from training.history_pretrain import HistoryView,tensor_hash
    from training.history_sequential import check_initial_residual
    from scripts.train_home7951 import configure_trainable_parameters
    read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
    device=torch.device(a.device);torch.set_num_threads(2)
    if len(set(a.seeds))!=len(a.seeds):ap.error('Duplicate seeds')
    if device.type=='cuda' and not torch.cuda.is_available():ap.error('CUDA unavailable')
    if not a.smoke and (not all([a.history_run,a.formal_run,a.output]) or device.type!='cuda'):
        ap.error('Formal run requires --history-run --formal-run --output and CUDA')
    c=read_protocol();ds,bundle=datasets(c,c['source_home'])
    tmp=tempfile.TemporaryDirectory(prefix='pcsa_ablation_smoke_') if a.smoke else None
    out=Path(tmp.name) if a.smoke else a.output.resolve()
    history=None;formal=None;provenance={}
    if not a.smoke:
        history=a.history_run.resolve();formal=a.formal_run.resolve()
        for parent in [history,formal]:
            if out==parent or parent in out.parents:raise ValueError('Use separate output directory')
        hc=read(history/'protocol.json');fc=read(formal/'protocol.json')
        if hc.get('experiment')!='pcsa_history_pretraining_v1' or hc.get('smoke_only'):
            raise ValueError('Expected completed full historical-pretraining run')
        if fc.get('experiment')!='pcsa_fixed_formal_evaluation_v1' or fc.get('smoke_only'):
            raise ValueError('Expected fixed full formal evaluation')
        for name,value in c.items():
            if hc.get(name)!=value:raise ValueError('Historical protocol mismatch: '+name)
        now=manifest(c)
        for root in [history,formal]:
            old=read(root/'manifest.json')
            if old['data']!=now['data']:raise ValueError('Data changed: '+str(root))
            for name,value in old['code'].items():
                if name.startswith('src/') or name in ['training/common.py','training/run.py','training/diagnostics.py','training/history_pretrain.py','scripts/train_home7951.py']:
                    if now['code'].get(name)!=value:raise ValueError('Frozen code changed: '+name)
        for seed in a.seeds:
            cp=history/f'seed_{seed}/history_pretrain/history/best.pt'
            summ=read(cp.parent/'summary.json')
            saved=torch.load(cp,map_location='cpu',weights_only=False)
            if saved['seed']!=seed or saved['protocol']!=hc or saved['epoch']!=summ['selected_epoch'] or summ['smoke_only']:
                raise ValueError('Historical checkpoint mismatch')
            # Verify the formal PCSA was actually trained from this same historical checkpoint.
            entry=read(formal/'source_training_provenance.json')[f'{seed}/pcsa']
            if entry['original_training_protocol']['reused_history'][str(seed)]['sha256']!=hash_file(cp):
                raise ValueError('History differs from formal PCSA initializer')
            provenance[str(seed)]={'history_sha256':hash_file(cp),'history_path':str(cp),
                'full_reference_test_sha256':hash_file(formal/f'seed_{seed}/source/pcsa/test.json')}
    c=copy.deepcopy(c);c.update(experiment='pcsa_source_component_ablation_v1',smoke_only=a.smoke,
        experiment_seeds=a.seeds,entry_sha256=hash_file(__file__),history_provenance=provenance,
        schedules={'no_residual':[160,80,0],'no_future_state_gate':[160,80,80]},
        scope='Source7951 only; same future losses; no_future_state_gate removes output gating only, not state supervision or residual fusion gates',
        budget_note='No-residual has 240 max total epochs; full and no-gate have 320. Not equal-compute architecture isolation.')
    lock_run(out,c);save_json(out/'run_mode.json',{'mode':'smoke' if a.smoke else 'full'})
    (out/'entry_snapshot.py').write_text(Path(__file__).read_text(encoding='utf8'),encoding='utf8')
    results=[];trained=[]
    for seed in a.seeds:
        for arm in ['no_residual','no_future_state_gate']:
            ac=copy.deepcopy(c);ac['ablation_arm']=arm
            folder=out/f'seed_{seed}'/arm
            seed_all(seed);base=make_model(ac,'pcsa',False)
            if a.smoke:
                hs=train_stage(HistoryView(base),ds,ac,folder/'history_smoke',device,160,seed,history_only=True,smoke=True)
            else:
                prior=history/f'seed_{seed}/history_pretrain'
                if tensor_hash(base.state_dict())!=read(prior/'initialization.json')['base_state_sha256']:
                    raise ValueError('Initial base differs from original pipeline')
                saved=torch.load(prior/'history/best.pt',map_location='cpu',weights_only=False)
                HistoryView(base).load_state_dict(saved['state_dict'],strict=True)
                hs=read(prior/'history/summary.json')
            base.state_conditioned_power=(arm!='no_future_state_gate')
            hhash=tensor_hash(HistoryView(base).state_dict())
            configure_trainable_parameters(base,'full')
            for n,p in base.named_parameters():
                if n.startswith(('encoder.','past_recon_head.')):p.requires_grad=False
            bs=train_stage(base,ds,ac,folder/'base',device,80,seed,smoke=a.smoke)
            if tensor_hash(HistoryView(base).state_dict())!=hhash:raise RuntimeError('History changed')
            model=base;stage_summaries=[hs,bs];new_summaries=[bs]
            if arm=='no_future_state_gate':
                seed_all(seed);model=make_model(ac,'pcsa',True)
                missing,extra=model.load_state_dict(base.state_dict(),strict=False)
                if extra or any(not n.startswith('refined_history_residual_tcn.') for n in missing):raise RuntimeError('Residual initialization keys')
                model.state_conditioned_power=False
                configure_trainable_parameters(model,'future_residual_tcn')
                base.eval();model.to(device).eval()
                check_initial_residual(base,model,move(next(iter(loader(ds['val'],16))),device),folder)
                rs=train_stage(model,ds,ac,folder/'residual',device,80,seed,smoke=a.smoke)
                stage_summaries.append(rs);new_summaries.append(rs)
            if tensor_hash(HistoryView(model).state_dict())!=hhash:raise RuntimeError('History changed')
            model.eval();batch=move(next(iter(loader(ds['val'],4))),device)
            with torch.no_grad():
                prediction=model({k:batch[k] for k in ['x_hist','x_future_time']})
                if arm=='no_future_state_gate' and not torch.equal(prediction['y_power'],prediction['amplitude_power']):
                    raise RuntimeError('Output state gate was not removed')
                if arm=='no_residual' and model.refined_history_residual_tcn is not None:raise RuntimeError('Residual still exists')
            save_json(folder/'checks.json',{'history_unchanged':True,'arm_verified':True,
                'frozen_parameters_unchanged':all(s['frozen_parameter_max_update']==0 for s in new_summaries),
                'frozen_buffers_unchanged':all(s['frozen_buffers_unchanged'] for s in new_summaries)})
            row={'seed':seed,'arm':arm,'val_mae_kw':stage_summaries[-1]['best_val_macro_mae_kw']}
            for key in ['epochs_run','optimizer_steps','training_seconds','total_wall_seconds']:
                row['pipeline_'+key]=sum(s[key] for s in stage_summaries)
                row['new_work_'+key]=sum(s[key] for s in new_summaries)
            save_json(folder/'budget.json',row);save_json(folder/'reused_history_summary.json',hs)
            trained.append((seed,arm,ac,folder,row));del model,base
    # Test evaluation occurs only after all specified variants have completed validation selection.
    for seed,arm,ac,folder,row in trained:
        model=make_model(ac,'pcsa',arm!='no_residual');model.state_conditioned_power=(arm!='no_future_state_gate')
        cp=folder/('base' if arm=='no_residual' else 'residual')/'best.pt'
        saved=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(saved['state_dict'],strict=True)
        model.to(device).eval();split='val' if a.smoke else 'test'
        ids=list(range(min(16,len(ds[split])))) if a.smoke else None
        ev=evaluate(model,loader(ds[split],16 if a.smoke else 512,ids=ids),device,ac,
                    loss_fn=objective(ac,bundle.appliance_scales).to(device),scales=bundle.appliance_scales)
        save_json(folder/('smoke_validation.json' if a.smoke else 'test.json'),ev)
        row[split+'_mae_kw']=ev['macro_mae_kw'];results.append(row);del model
    if not a.smoke:
        for seed in a.seeds:
            ev=read(formal/f'seed_{seed}/source/pcsa/test.json')
            entry=read(formal/'source_training_provenance.json')[f'{seed}/pcsa']
            ref={'seed':seed,'arm':'full_reference','test_mae_kw':ev['macro_mae_kw']}
            for key in ['epochs_run','optimizer_steps','training_seconds','total_wall_seconds']:ref['pipeline_'+key]=sum(s[key] for s in entry['stage_summaries'])
            results.append(ref)
    save_json(out/'comparison.json',results)
    keys=list(dict.fromkeys(k for row in results for k in row))
    with (out/'comparison.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(results)
    metric='val_mae_kw' if a.smoke else 'test_mae_kw'
    summary=[]
    for arm in sorted({r['arm'] for r in results}):
        v=[r[metric] for r in results if r['arm']==arm]
        summary.append({'arm':arm,'n_seeds':len(v),'mean_mae_kw':float(np.mean(v)),'sample_sd_kw':float(np.std(v,ddof=1)) if len(v)>1 else None})
    save_json(out/'results_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
    if tmp:tmp.cleanup();print('Smoke passed; temporary weights removed.')
    else:print('Saved:',out,flush=True)

if __name__=='__main__':main()
