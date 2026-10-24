#!/bin/bash
# Score every experiment x model prediction file (run_eval.py).
# Evaluate all eight models by default; MODEL_SPEC overrides this list.
# Experiments come from config.sh (VARY_SPEC overrides).
# Match inference's effort folders: low/medium/high for reasoning_effort,
# REASONING_EFFORT (default low) for every other experiment.
#
#   bash scripts/experiments/v1/eval.sh
#   MODEL_SPEC=gpt-5.6-sol VARY_SPEC=node-count bash scripts/experiments/v1/eval.sh

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

if [[ -z "${MODEL_SPEC:-}" ]]; then
    MODEL_SLUGS=(
        gpt-5.6-luna
        gpt-5.6-sol
        gemini-3.8-flash
        gemini-3.6-flash
        gemini-3.1-pro-preview
        qwen3.5-35b-a3b
        qwen3.5-122b-a10b
        qwen3.5-397b-a17b
    )
fi

print_header "EVALUATION"
printf '  Experiments    : %s\n' "${VARY_VALUES[*]}"
printf '  Models         : %s\n' "${MODEL_SLUGS[*]}"
printf '  Root           : %s\n' "$(growth_dir)"

RUNS=0
for MODEL_SLUG in "${MODEL_SLUGS[@]}"; do
    for VARY in "${VARY_VALUES[@]}"; do
        EXPERIMENT_DIR="$(experiment_dir "$VARY")"
        EVAL_VARY="$VARY"
        if [[ "$VARY" == reasoning_effort || "$VARY" == context_length ]]; then
            EVAL_VARY=none
        fi
        read -r -a EFFORTS <<< "$(reasoning_efforts "$VARY")"
        for EFFORT in "${EFFORTS[@]}"; do
            PREDICTIONS="$EXPERIMENT_DIR/inference/reasoning_effort=${EFFORT}/model=${MODEL_SLUG}.json"
            if [[ ! -f "$PREDICTIONS" ]]; then
                printf '\n[skip] no predictions -> %s\n' "$PREDICTIONS"
                continue
            fi
            RUNS=$((RUNS + 1))
            printf '\n'
            "$PYTHON_BIN" run_eval.py \
                --predictions "$PREDICTIONS" \
                --vary "$EVAL_VARY" \
                --output "$EXPERIMENT_DIR/eval/reasoning_effort=${EFFORT}/model=${MODEL_SLUG}.json"
        done
    done
done

printf '\nEVALUATION finished: %d report(s)\n' "$RUNS"
