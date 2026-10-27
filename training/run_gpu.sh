#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python -m training.run --phase audit --output outputs/source --device cpu
python -m training.run --phase source --output outputs/source --device cuda:0 --seeds 7 42 123
python -m training.history_pretrain --output outputs/history --device cuda:0 --seeds 7 42 123
python -m training.history_sequential --source-run outputs/history --output outputs/forecast --device cuda:0 --seeds 7 42 123
python -m training.formal_evaluate --pcsa-run outputs/forecast --baseline-run outputs/source --output outputs/evaluation --device cuda:0 --seeds 7 42 123
python scripts/train_pcsa_component_ablation.py --history-run outputs/history --formal-run outputs/evaluation --output outputs/ablation --device cuda:0 --seeds 7 42 123
