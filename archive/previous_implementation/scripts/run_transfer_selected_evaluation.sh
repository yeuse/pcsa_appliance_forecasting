#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
output="outputs/transfer_selected_eval_$(date +%Y%m%d_%H%M%S)"
python -u scripts/evaluate_transfer_selected.py \
  --output_dir "$output" --device cuda --batch_size 1024 --num_workers 8 "$@"
