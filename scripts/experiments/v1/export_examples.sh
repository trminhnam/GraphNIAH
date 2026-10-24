#!/bin/bash
# Export the examples of every v1 experiment (not the growth-parameter search)
# to one JSONL file: qid, reachability task, system and user prompts, answer,
# config, and the experiments that use each example. Run it after the
# create_data scripts; rerun to export again (the file is replaced).
#
#   bash scripts/experiments/v1/export_examples.sh
#   EXPORT_DIR=./output/other_folder bash scripts/experiments/v1/export_examples.sh

set -euo pipefail

################ Hyperparameters ####################
EXPORT_DIR="${EXPORT_DIR:-./output/huggingface_dataset}"
#####################################################

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
EXPORT_FILE="$EXPORT_DIR/examples.jsonl"

print_header "EXPORT EXAMPLES"
printf '  Experiments : %s (growth_search skipped)\n' "$OUTPUT_ROOT"
printf '  Output      : %s\n\n' "$EXPORT_FILE"
started=$SECONDS
"$PYTHON_BIN" endpoints/export_examples.py --root "$OUTPUT_ROOT" --output "$EXPORT_FILE"
printf '\nEXPORT EXAMPLES finished in %s\n' "$(format_duration $((SECONDS - started)))"
