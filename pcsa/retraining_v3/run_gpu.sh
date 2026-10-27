#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m retraining_v3.run --phase suite --output outputs/retrain_v3_full --device cuda --seeds 7 42 123
