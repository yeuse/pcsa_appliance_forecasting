# PCSA implementation guide

The current evaluated implementation is stored in the versioned `pcsa/` directory. Run its commands from that directory so Python imports the corresponding models and loaders. The root-level source and scripts describe an earlier implementation.

| Component | Implementation |
|---|---|
| PCSA model and forecast routes | [`pisa.py`](../pcsa/src/models/pisa.py) |
| Encoders, decoders and residual blocks | [`blocks.py`](../pcsa/src/models/blocks.py) |
| Smooth output constraints | [`power_output.py`](../pcsa/src/models/power_output.py) |
| Data loading and temporal windows | [`pecanstreet_dataset.py`](../pcsa/src/data/pecanstreet_dataset.py) |
| Training objectives | [`losses.py`](../pcsa/src/training/losses.py) |
| Shared experiment definitions | [`common.py`](../pcsa/retraining_v3/common.py) |
| Historical pretraining | [`history_pretrain.py`](../pcsa/retraining_v3/history_pretrain.py) |
| Sequential base/residual training | [`history_sequential.py`](../pcsa/retraining_v3/history_sequential.py) |
| Formal source/target evaluation | [`formal_evaluate.py`](../pcsa/retraining_v3/formal_evaluate.py) |
| Component ablations | [`train_pcsa_component_ablation.py`](../pcsa/scripts/train_pcsa_component_ablation.py) |

**[Complete environment, data and execution instructions](../pcsa/README.md)**

**[Three-seed results and provenance](../pcsa/records)**

The frozen code is preserved exactly as executed. Its intermediate experiment README files provide historical context; follow the complete execution sequence linked above for the final evaluated forecasting model.
