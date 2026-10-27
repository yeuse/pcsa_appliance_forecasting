# PCSA implementation guide

The current implementation is organized at the repository root. Run all commands from that root. Previous code is isolated under `archive/previous_implementation/`.

| Component | Implementation |
|---|---|
| PCSA model and forecast routes | [`pisa.py`](../src/models/pisa.py) |
| Encoders, decoders and residual blocks | [`blocks.py`](../src/models/blocks.py) |
| Smooth output constraints | [`power_output.py`](../src/models/power_output.py) |
| Data loading and temporal windows | [`pecanstreet_dataset.py`](../src/data/pecanstreet_dataset.py) |
| Training objectives | [`losses.py`](../src/training/losses.py) |
| Shared experiment definitions | [`common.py`](../training/common.py) |
| Historical pretraining | [`history_pretrain.py`](../training/history_pretrain.py) |
| Sequential base/residual training | [`history_sequential.py`](../training/history_sequential.py) |
| Formal source/target evaluation | [`formal_evaluate.py`](../training/formal_evaluate.py) |
| Component ablations | [`train_pcsa_component_ablation.py`](../scripts/train_pcsa_component_ablation.py) |

**[Complete environment, data and execution instructions](RUNNING.md)**

**[Three-seed results and provenance](../records)**

The public layout renames the experiment package and updates imports and commands. Model and data implementations remain unchanged; original records and the migration map document provenance.
