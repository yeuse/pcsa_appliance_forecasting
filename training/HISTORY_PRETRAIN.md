# PCSA history pretraining schedule comparison

Run from the repository root on the GPU server:

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m training.history_pretrain --output outputs/history_pretrain_v1 --device cuda --seeds 7 42 123
```

Upload `training/history_pretrain.py` and the updated `training/diagnostics.py`.
The other existing source files remain required. Do not change files while running;
the manifest locks code/data. Use a new output directory if either changes.

Both arms retain the original PCSA inference architecture, source 7951, full original
train/validation windows, seeds, initialization, batch size 512, AdamW learning rate
1e-4, gradient clipping, and early stopping (minimum 20 epochs, patience 30,
initial checkpoint eligible). Test data are not evaluated or used for selection.
Missing historical labels are masked; complete-history filtering is NOT used.

* Control: original 160-epoch base future training with historical auxiliary loss,
  then 160-epoch residual training with the base frozen.
* History pretrain: 160 epochs training only the existing encoder and historical
  reconstruction head, then freeze both and train ALL remaining forecast routes
  together for up to 160 epochs. The original base forecast route must be trained
  here because it was not trained in stage one. No new refiner is added.

Thus the treatment is a training-schedule comparison (supervision order, selection,
and freezing), not an isolated loss-weight change. Maximum total budget is 320
passes per arm; actual early-stopped passes, optimizer steps and compute differ.
Both arms rerun from scratch; no historical checkpoint is silently reused.

Historical pretraining uses masked scaled L1 plus 0.1 state BCE. The supervised
power is exactly the smoothly capped power that the residual TCN consumes. It is
NOT multiplied by the state gate; probabilities are a separate input. Selection
uses historical validation macro MAE in kW. Forecast training retains control
future losses and selects on future validation macro MAE. ON/OFF diagnostics are
reported, not used to choose a favorable checkpoint after the experiment.

Results: `comparison.json` / `comparison.csv`; each seed/arm contains stage
`history.json`, `summary.json`, `best.pt`, selected train/validation diagnostics,
`history_validation.json`, and `checks.json`. The root label budget includes
the union of history/future labels separately for training and validation.
Stage summaries record actual updates, gradients, compute, and frozen tensors.

Only completed stages resume. An interrupted stage starts again from its initial
state, not the interrupted optimizer step. Old experimental output directories
are not modified. Updated diagnostics changes the manifest; do not use this code
to resume unrelated unfinished runs under their previous manifest.

Local temporary smoke validation (auto-removes all generated weights/results):

```bash
python -B -m training.history_pretrain --smoke --device cpu --seeds 7
```

Smoke uses 64 sampled windows per split and three epochs per stage. These metrics
are only execution checks, not scientific results.
