#!/bin/bash
# Evaluate converted multiple-choice results separately from original results.
# Defaults to all seven models and low effort; MODEL_SPEC / REASONING_EFFORT override.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.none_option.sh"

print_header "EVALUATION | multiple choice with appended None"
for MODEL_SLUG in "${MODEL_SLUGS[@]}"; do
    CONVERTED_DIR="$(none_option_dir "$MODEL_SLUG")"
    PREDICTIONS="$CONVERTED_DIR/inference.json"
    if [[ ! -f "$PREDICTIONS" ]]; then
        printf '[skip] no converted predictions -> %s\n' "$PREDICTIONS"
        continue
    fi
    "$PYTHON_BIN" run_eval.py \
        --predictions "$PREDICTIONS" \
        --vary question-format \
        --output "$CONVERTED_DIR/eval.json"
done
