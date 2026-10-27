# Training and evaluation

Run all commands from the repository root. See [the execution guide](../docs/RUNNING.md) for the complete paper workflow.

- `common.py` and `protocol.json`: shared model, data, loss and budget definitions.
- `run.py`: data audit, source baselines and the earlier PCSA control.
- `history_pretrain.py`: history pretraining and schedule comparison.
- `history_sequential.py`: base forecasting followed by frozen-base residual training.
- `formal_evaluate.py`: evaluation of the selected source checkpoints and target adaptation.
- `diagnose.py`, `diagnostics.py`: conditional error and module diagnostics.
- `history_input_ablation.py`: controlled comparison of reconstructed history inputs.

The earlier PCSA control is not the final PCSA model. Use the full sequence in the execution guide. Use fresh output directories for this package layout; existing source snapshots and checkpoints retain their original provenance.
