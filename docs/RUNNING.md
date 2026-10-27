# Running PCSA

Run commands from the repository root. The model and data implementations are preserved; experiment imports and package paths have been renamed for public use. See [provenance notes](PROVENANCE.md).

## Final method and evidence

Home 7951 is the source, with targets 3039, 8386 and 8565; seeds are 7, 42 and 123. PCSA has two conceptual stages: historical reconstruction pretraining, followed by forecasting with sequential base then residual optimization (maximum 160 + 80 + 80 epochs). Each seed uses its own historical checkpoint. The old shared seed-42 initialization is not used in these results.

- Source and target tests: `records/formal_results_summary.csv` and `formal_results_by_seed.csv`.
- Per-appliance ON/OFF errors, OFF predicted power, cap-only controls, cap saturation, target label budgets, parameter/frozen-buffer checks and compute: `records/formal_evaluation_records.zip`.
- Training-schedule evidence: `records/history_pretrain_records.zip` and `history_sequential_records.zip`.
- Retrained component ablations: `records/ablation_comparison.csv` and `component_ablation_records.zip`.
- Source pipeline provenance and costs: `records/formal_source_training_provenance.json`.
- Code and prepared-data SHA256 hashes: `records/formal_manifest.json`. Original absolute server paths in provenance are historical metadata, not portable command arguments.

## Environment and data

Install a CUDA PyTorch environment with numpy, pandas, scipy, scikit-learn, PyYAML and tqdm. The recorded environment is in `records/formal_environment.json`; select a PyTorch build appropriate for your CUDA driver. Raw Dataport data are subject to provider access terms and are not distributed. Model weights are not included in this release; the records are not an inference-ready checkpoint bundle.

Prepare the four CSV files at `data/austin_2018_sep_4homes/home_{home}_2018_sep_1min.csv`, with home equal to 7951, 3039, 8386 and 8565. Use `scripts/prepare_austin_sep_four_homes.py --help` for the raw-input arguments and explicitly set `--window-rule history_only`. The released manifest contains the hashes of the prepared CSVs used for the reported results. Do not substitute a different extract while claiming identical reproduction.

## Commands for the recorded workflow

```bash
# From repository root
# Physical GPU 0 is exposed as logical cuda:0
export CUDA_VISIBLE_DEVICES=0
python -m training.run --phase audit --output outputs/source --device cpu
python -m training.run --phase source --output outputs/source --device cuda:0 --seeds 7 42 123
python -m training.history_pretrain --output outputs/history --device cuda:0 --seeds 7 42 123
python -m training.history_sequential --source-run outputs/history --output outputs/forecast --device cuda:0 --seeds 7 42 123
python -m training.formal_evaluate --pcsa-run outputs/forecast --baseline-run outputs/source --output outputs/evaluation --device cuda:0 --seeds 7 42 123
python scripts/train_pcsa_component_ablation.py --history-run outputs/history --formal-run outputs/evaluation --output outputs/ablation --device cuda:0 --seeds 7 42 123
```

The first source run provides the baselines and an earlier PCSA control; the final PCSA uses `outputs/forecast`, not that earlier control. The historical comparison runner also trains control and joint-forecast arms. Those exploratory costs must not be confused with the reported final pipeline cost. Completed stages from the same package and manifest can be reused; interrupted stages restart rather than restoring an exact optimizer step. Use fresh output paths when code or data change.

## Interpretation and limitations

Support fractions are chronological fractions of retained target training windows, excluding additional target-validation labels. History labels are used by PCSA and NILM–TCN but not Aggregate. Equal maximum epochs do not imply equal computation or supervision. No-residual ablation has a 240-epoch maximum versus 320 for full PCSA and no-future-state-gate.

Source caps [3, 0.5, 2, 2.5] kW were empirical choices after inspecting the source month, including eventual test dates. Earlier tests were inspected during development. This is a retrospective fixed-protocol comparison, not a new untouched holdout. Net-grid imports are not gross demand; negative intervals and missing labels affect coverage. Three seeds provide descriptive variability, not a significance guarantee. ON forecasting and event recognition remain limited; no operational HEMS benefit is established.

The no-future-state-gate variant removes output conditioning only; state supervision, historical states and residual fusion gates remain. It has slightly lower overall MAE than full PCSA. These outcomes are retained without selecting only favorable variants.

OpenAI Codex assisted with code inspection/revision, diagnostics and documentation. Authors remain responsible for scientific verification. Model, loss and data logic are preserved from the run snapshot; package imports and entry paths are documented in the migration map.

Existing result records keep original run identifiers and paths for traceability. They are evidence from the reported runs, not paths to use for new training. Do not mix checkpoints from the original package with new output directories; the original code remains accessible through its recorded Git commit.
