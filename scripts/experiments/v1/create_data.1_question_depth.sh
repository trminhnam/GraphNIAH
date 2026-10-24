#!/bin/bash
# Experiment 1: question depth (4, 8, 16, 32 gold-path papers).
# Every other factor is drawn at random for each example. Rerun the same command
# to resume; a larger EXAMPLES_PER_VALUE appends to the experiment. Shared
# settings (inputs, context length, seed, pool, workers) are in config.sh.
#
#   bash scripts/experiments/v1/create_data.1_question_depth.sh
#   EXAMPLES_PER_VALUE=300 NUM_WORKERS=8 bash scripts/experiments/v1/create_data.1_question_depth.sh

set -euo pipefail

################ Hyperparameters ####################
VARY="question-depth"
EXAMPLES_PER_VALUE="${EXAMPLES_PER_VALUE:-500}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-500}"  # New-seed attempts per slot (same config).
#####################################################

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
create_data "$VARY" "$EXAMPLES_PER_VALUE" "$MAX_ATTEMPTS"
