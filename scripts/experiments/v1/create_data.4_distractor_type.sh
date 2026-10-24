#!/bin/bash
# Experiment 4: distractor type (code, essays, books, disjoint-tree, wikipedia).
# Every other factor is drawn at random for each example. Rerun the same command
# to resume; a larger EXAMPLES_PER_VALUE appends to the experiment. Shared
# settings (inputs, context length, seed, pool, workers) are in config.sh.
#
#   bash scripts/experiments/v1/create_data.4_distractor_type.sh
#   EXAMPLES_PER_VALUE=300 NUM_WORKERS=8 bash scripts/experiments/v1/create_data.4_distractor_type.sh

set -euo pipefail

################ Hyperparameters ####################
VARY="distractor-type"
EXAMPLES_PER_VALUE="${EXAMPLES_PER_VALUE:-500}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-500}"  # New-seed attempts per slot (same config).
#####################################################

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
create_data "$VARY" "$EXAMPLES_PER_VALUE" "$MAX_ATTEMPTS"
