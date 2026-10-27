#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for variant in control balanced_power no_amp_on old_residual_gate; do
  python -m training.run --phase source --validation-only --variant "$variant" --output "outputs/diagnostic_ofat_v1/$variant" --device cuda --seeds 7 42 123
done
