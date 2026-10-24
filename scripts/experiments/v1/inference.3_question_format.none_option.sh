#!/bin/bash
# Run after create_data.3_question_format.none_option.sh finishes.
# Each model receives only its converted multiple-choice examples.
# No MAX_SAMPLES or dimension filters: the converted experiment defines the subset.
# Use the same request settings as the original run; defaults match inference.sh.
# All seven models are visited by default; MODEL_SPEC overrides the list.
# Each model uses its own inference pool unless INFERENCE_POOL is explicitly set.

set -euo pipefail
INFERENCE_POOL_OVERRIDE="${INFERENCE_POOL:-}"
source "$(dirname "${BASH_SOURCE[0]}")/config.none_option.sh"

MAX_WORKERS="${MAX_WORKERS:-16}"
TEMPERATURE="${TEMPERATURE:-1}"
TOP_P="${TOP_P:-0.95}"
FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0}"

print_header "INFERENCE | multiple choice with appended None"
for MODEL_SLUG in "${MODEL_SLUGS[@]}"; do
    CONVERTED_DIR="$(none_option_dir "$MODEL_SLUG")"
    EXPERIMENT="$CONVERTED_DIR/experiment.jsonl"
    if [[ ! -s "$EXPERIMENT" ]]; then
        printf '[skip] no converted examples -> %s\n' "$EXPERIMENT"
        continue
    fi
    MODEL_LIMIT="$(model_context_limit "$MODEL_SLUG")"
    if (( CONTEXT_LENGTH > MODEL_LIMIT )); then
        printf '[skip] %s accepts %s tokens, fewer than %s\n' \
            "$MODEL_SLUG" "$MODEL_LIMIT" "$CONTEXT_LENGTH"
        continue
    fi
    MODEL_POOL="${INFERENCE_POOL_OVERRIDE:-$OUTPUT_ROOT/inference_pool.${MODEL_SLUG}.jsonl}"
    printf '\n[Run] model=%s reasoning_effort=%s workers=%s\n' \
        "$MODEL_SLUG" "$REASONING_EFFORT" "$MAX_WORKERS"
    "$PYTHON_BIN" run_inference.py \
        --pool "$POOL" \
        --inference-pool "$MODEL_POOL" \
        --experiment "$EXPERIMENT" \
        --output "$CONVERTED_DIR/inference.json" \
        --model "$MODEL_SLUG" \
        --reasoning-effort "$REASONING_EFFORT" \
        --temperature "$TEMPERATURE" \
        --top-p "$TOP_P" \
        --frequency-penalty "$FREQUENCY_PENALTY" \
        --presence-penalty "$PRESENCE_PENALTY" \
        --max-workers "$MAX_WORKERS"
done
