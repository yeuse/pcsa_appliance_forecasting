# Reproducibility status and required artifacts

## Available now

Executable source, a config-driven source training entry point, dependency
declaration, synthetic tests, recovered source and transfer configs, scalar
result provenance, and validation-only fusion diagnostics. Three recorded
configs define the seed-7 source chain. The base fit and auxiliary
history-refiner pass belong to conceptual Stage I; residual-TCN fitting is
Stage II. Their input and output paths are repository-relative.
Model/checkpoint hashes are retained in the scalar results. Private repository
prefixes are removed from public configs.

## Not yet available in this candidate

1. Raw or processed meter data. Obtain access through the official Pecan Street
   Dataport under the provider's license; no measurements are redistributed.
2. Pretrained weights. They are optional for retraining the documented seed-7
   source chain, which generates its own checkpoints.
3. The original server environment lockfile, CUDA/driver/GPU details and a
   clean real-data reproduction of the reported numerical scores.
4. The recorded seed-123 source configuration and the full three-seed source
   evaluation artifact set.
5. A corrected three-home event suite: currently only 8565/10% is audited.

Do not state that these missing items are completed in the manuscript response.

## Source training from the saved configs

From the repository root, after preparing an authorized Home-7951 CSV:

```bash
python scripts/train_pcsa_from_config.py --config configs/observed/pcsa_source_base_seed42_config.json
python scripts/train_pcsa_from_config.py --config configs/observed/pcsa_source_history_seed42_config.json
python scripts/train_pcsa_from_config.py --config configs/observed/pcsa_source_seed7_config.json
```

The first configuration has no initialization checkpoint. The second points
to the first run's `checkpoints/best.pt`, and the third points to the second
run's `checkpoints/best.pt`. The runner uses the stored hyperparameters and
checks required local files before invoking the original training script.
`--dry-run` prints the command without training. The historical-refiner config
records a correction of `model.state_gate_floor` from 0.05 to the base model's
0.01, which is preserved in the portable copy. The recorded final stage uses
seed 7 and base reconstructed history for its TCN input.

## Existing-artifact layout

```text
data/austin_2018_sep_4homes/home_{7951,3039,8386,8565}_2018_sep_1min.csv
outputs/runs/7951_tcn64x3_basehist_event0_e160_s42/
  config.json
  results/data_info.json
  checkpoints/best.pt
outputs/runs/7951_baselines_64x3_poweronly_s42/
  config.json
  checkpoints/aggregate_tcn/best.pt
  checkpoints/two_stage_forecaster/best.pt
outputs/transfer_runs_two_stage/7951_to_<home>_pisa_twostage_h120_f200_s42/
outputs/transfer_baselines/7951_to_<home>_baselines_s42/
outputs/transfer_baselines_joint/7951_to_<home>_baselines_joint_e200_val_s42/
```

Each transfer run needs `config.json`, the appropriate results/transfer summary,
and selected weights. PISA uses `checkpoints/fewshot_<fraction>/selected.pt`.
Baselines use `checkpoints/{aggregate_tcn,two_stage}/fewshot_<fraction>/best.pt`,
unless selection points to source weights. In particular joint Seq2Seq at
8565/1% selects the support-calibrated source model.

If transferring artifacts to a new machine, make a separate working copy of the
configs/summaries. Replace only the original repository prefix in path fields
with your new repository root or repository-relative paths. Keep weight bytes,
selection labels, caps and scalar metrics unchanged. Existing evaluation code
fails on missing paths rather than searching for a different checkpoint.
Observed public configs alone do NOT replace the required selection summaries.

## Budgets are not matched

PISA: history stage max 120 epochs, forecast stage max 200; both 3e-5 learning
rate, patience 25, minimum 40 epochs. Original baselines: max 80, patience 15,
minimum 25. Joint baselines: max 200, patience 25, minimum 40. PISA historical
power supervision weight is 1.0; joint baseline weight is 0.2. Do not report
these experiments as strictly equal-budget or architecture-only ablations.
