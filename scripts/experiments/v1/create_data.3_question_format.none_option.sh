#!/bin/bash
# Append None to only the multiple-choice examples each model already answered.
# Wrong answers are included; failed requests and other question formats are not.
# Defaults to all seven models and low effort; MODEL_SPEC / REASONING_EFFORT override.
# Conversion appends examples to POOL and preserves original records and outputs.
# Run this stage once, sequentially, before parallel model inference.
# Rerun after more original examples finish to extend each converted experiment.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.none_option.sh"

print_header "CONVERT MULTIPLE CHOICE | append None, preserve labels"
EXPERIMENT="$SOURCE_EXPERIMENT_DIR/experiment.jsonl"
if [[ ! -s "$EXPERIMENT" ]]; then
    printf '[skip] no source experiment -> %s\n' "$EXPERIMENT"
    exit 0
fi
for MODEL_SLUG in "${MODEL_SLUGS[@]}"; do
    PREDICTIONS="$SOURCE_EXPERIMENT_DIR/inference/reasoning_effort=${REASONING_EFFORT}/model=${MODEL_SLUG}.json"
    if [[ ! -f "$PREDICTIONS" ]]; then
        printf '[skip] no original predictions -> %s\n' "$PREDICTIONS"
        continue
    fi
    "$PYTHON_BIN" convert_reachability_multiple_none-option.py \
        --input "$EXPERIMENT" \
        --pool "$POOL" \
        --predictions "$PREDICTIONS" \
        --output "$(none_option_dir "$MODEL_SLUG")/experiment.jsonl" \
        --none-option-conversion-mode append
done
