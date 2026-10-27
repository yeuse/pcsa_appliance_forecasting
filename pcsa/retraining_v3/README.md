# PCSA R3 reproducible retraining protocol

This is the ONLY entry point for the new experiments. Old outputs and the R2 manuscript belong to the old protocol and must not be relabelled as R3 results. `public_release/pisa` is an archive, not the R3 training source.

## Commands

From the project root, with a CUDA-enabled PyTorch environment:

```bash
python -m retraining_v3.run --phase audit --output outputs/retrain_v3_full --device cpu
python -m retraining_v3.run --phase suite --output outputs/retrain_v3_full --device cuda --seeds 7 42 123
```

For local CPU verification ONLY:

```bash
python -m retraining_v3.run --phase smoke --output outputs/retrain_v3_smoke --device cpu
```

Source and transfer may be run separately with `--phase source` and `--phase transfer` using the same full output directory. Complete stages resume from their validation-selected checkpoints. Interrupted stages restart; the runner does not claim exact mid-epoch resumption. An existing directory with different code or data hashes is rejected. Smoke and full results cannot share a directory. Test labels are never used in smoke mode.

Use the existing GPU-server CUDA/PyTorch installation. Required Python packages: torch, numpy, pandas, scikit-learn, scipy, tqdm. Run the tests there before scheduling the full experiment; CPU smoke does not validate CUDA numerics or GPU memory use. FP32 is intentional for the first auditable protocol, not an efficiency claim.

## Changes from R2

- Independent base and residual training for EACH seed, with no imported R2 checkpoint.
- History-refiner disabled. The base reconstruction feeds the residual TCN.
- Smooth output amplitude `a = C * softplus(z) / hypot(softplus(z), C)` then state gate `y = a * (0.01 + 0.99 * sigmoid(u))`.
- The smooth map has a positive analytic derivative for finite positive amplitude. Finite-precision extreme saturation is still possible. This is a changed forward model, not a straight-through gradient trick.
- Residual gate logit 0 (gate 0.5); residual output heads remain zero-initialized. Initial full forecast equals the base forecast. Feature/gate gradients start after the residual head first updates.
- ON-only pre-state-gate amplitude loss weight 0.05, applied equally to forecast methods. This supervises amplitude when the state gate is low; it does not require amplitude to be zero in OFF states.
- Common future prediction losses: power 1, state 0.1, ghost 0.05, ON amplitude 0.05. Aggregate net-grid upper-bound loss is OFF for all methods. Main power ON weight is **2.5**, OFF weight 1 (parameter `active_power_weight=1.5` means an addition to 1).
- PCSA base and joint-history transfer also supervise history (power 0.2, state 0.1). NILM pretraining uses scaled historical L1 + 0.1 masked BCE. These differences are declared, not called identical supervision.
- Historical net-grid readings and all future appliance targets must be valid. Future grid availability is NOT used to select windows. Negative grid intervals remain excluded from the available historical input, so the estimand remains limited to valid import-history windows; it is not whole-day gross-demand forecasting.

## Data preparation

To rebuild the extract, use `scripts/prepare_austin_sep_four_homes.py` with the existing raw-source arguments and **`--window-rule history_only`**. The default old script mode remains `history_and_future` for archival compatibility. The new runner explicitly selects history-only and checks preparation-count equality. Existing minute CSV values need not be overwritten to change the window rule.

Minute alignment, split dates and minute-channel availability remain unchanged. For second-resolution sources at least 50 valid readings are required; a negative grid second invalidates that minute. Original raw timestamps, PV and storage metadata still require author verification. The archived PV metadata is not sufficient to prove that a missing/false entry means no PV. Do not equate net grid with gross household demand.

## Budgets and selection

| Method | Source stages / max epochs | Total max sample passes |
|---|---|---|
| PCSA | base 160, residual 160 | 320 |
| Aggregate TCN | direct 320 | 320 |
| Seq2Seq-NILM-TCN | NILM 160, forecast 160 | 320 |

Source batch 512, AdamW learning rate 1e-4. Transfer maximum 200 epochs, batch 256, learning rate 3e-5 for every method. Weight decay 1e-4; clipping 1; no scheduler; no dropped final batches. Minimum 20 epochs and patience 30 per stage; strict lower validation MAE selects a checkpoint, with epoch-zero candidate included. Different early stops mean actual passes can differ. Equal maximum sample passes are NOT equal FLOPs, time, trainable capacity or hyperparameter-search cost. Every stage records actual updates, samples, training time, total wall time and parameter counts.

Target support = earliest 1%, 5%, 10% of valid target TRAIN windows. Separate full target validation selects checkpoints. Cap calibration uses unique future-label rows in the support, quantile 0.995 and margin 5%, with source caps as lower bounds. Source TCN input scaling remains fixed. Cap-only source and adapted model test results are separate; adaptation selection includes the cap-only initial candidate.

`budget.json` counts unique target timestamps and valid appliance labels. Historical reconstruction labels are included for PCSA and NILM-TCN. Calibration labels overlap the training labels and are not added twice. Validation labels are additional, disjoint by date, and must be disclosed. Label time-span and elapsed time from the start of the split are both reported. Percent training windows must never be described as percent TOTAL target labels.

## Files and integrity

Each output root contains protocol, source/data SHA256 hashes, source snapshot and environment. Each stage contains best.pt, history.json and summary.json (early gradients, actual changes, frozen-parameter check). Every source stage uses the same seed-specific pipeline; no shared seed-42 base checkpoint. A failure never silently falls back to another seed.

Smoke also exercises the first target home at 1% support for all three methods, using validation only. Smoke uses 64 evenly spaced TRAIN windows and 64 VALIDATION windows, batch 16, three epochs per stage. The subset is chosen by index, not by errors. It validates implementation, not accuracy or convergence. Do not include smoke metrics in the paper. Full experiments are the only source of replacement Tables 4-6.

Frozen temporal feature extractors stay in evaluation mode during head-only adaptation, including Aggregate TCN. Both frozen parameters and running-statistic buffers are checked. Completed-stage recovery checks checkpoint seed/protocol/mode. Each source model writes a batch-one forward benchmark with five warm-up and twenty timed runs, excluding data loading; CUDA timings are synchronized. CPU timings are not GPU efficiency evidence.
