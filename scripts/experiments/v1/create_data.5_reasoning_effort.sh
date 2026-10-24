#!/bin/bash
# Reasoning-effort experiment: draw every data factor at random (--vary none).
# EXAMPLES_PER_VALUE is the total example count. All efforts share this dataset.
# Rerun to resume, or increase EXAMPLES_PER_VALUE to append examples.
# Shared settings (inputs, context length, seed, pool, workers) are in config.sh.
#
#   bash scripts/experiments/v1/create_data.5_reasoning_effort.sh
#   EXAMPLES_PER_VALUE=1000 NUM_WORKERS=8 bash scripts/experiments/v1/create_data.5_reasoning_effort.sh

set -euo pipefail

################ Hyperparameters ####################
VARY="none"
EXAMPLES_PER_VALUE="${EXAMPLES_PER_VALUE:-500}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-500}"  # New-seed attempts per slot (same config).
#####################################################

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
create_data "$VARY" "$EXAMPLES_PER_VALUE" "$MAX_ATTEMPTS"
