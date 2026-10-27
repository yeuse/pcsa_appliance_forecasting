# History pretraining followed by sequential prediction training

Upload history_sequential.py to the existing retraining_v3 folder. Keep the previous
history_pretrain.py, diagnostics.py and all canonical source files unchanged.

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m retraining_v3.history_sequential --source-run outputs/history_pretrain_v1 --output outputs/history_sequential_v1 --device cuda --seeds 7 42 123
```

This is three optimization stages overall: reused historical pretraining (maximum
160 epochs), base prediction with encoder/history head frozen (maximum 80), then
residual prediction with the entire base frozen (maximum 80). The prediction
portion has two steps. Total maximum budget remains 320 passes, including the
reused historical training. Actual compute is reported, not claimed equal.

Uses the original full train/validation windows, missing-history masks, seeds,
initial base weights, loss settings, batch size and early stopping. No test
selection or evaluation. Historical best.pt files for each seed must remain in
outputs/history_pretrain_v1/seed_SEED/history_pretrain/history/ on the server.
Source code/data hashes and checkpoint seed/protocol are verified. The old run
is read-only. Both new stages include their initial state as a selection candidate.

comparison.csv compares the new result to the saved original control and joint
forecast training results. pipeline_* includes historical pretraining cost;
new_work_* counts only the two new stages. Each seed has base/ and residual/
with histories, selected train/validation diagnostics, summaries and best weights.
The historical validation report is reused only after verifying the historical
weights and buffers remain unchanged. checks.json verifies freezing and exact
initial base/residual prediction equality.

Completed stages can be skipped on rerun with identical code/data/protocol.
Interrupted stages restart from their initial state; no optimizer-step resume.
Do not edit code during a run. Local smoke testing uses temporary files and
removes all generated weights when done:

```bash
python -B -m retraining_v3.history_sequential --smoke --device cpu --seeds 7
```
