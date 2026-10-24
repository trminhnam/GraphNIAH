#!/bin/bash
# Run the question-depth experiment with gemini-3.1-pro-preview on Flex only.
# Select question depths 16 and 32.
# The registered slug gemini-3.1-pro-preview resolves to gemini/gemini-3.1-pro-preview.
# Other inference settings are passed to the shared launcher.
# Retry unanswered examples, stopping early when every selected example is answered.
# Override MAX_TRIALS (default 20) and TRIAL_WAIT_SECONDS (default 60) as needed.
#
#   bash scripts/experiments/v1/inference.1_question_depth.gemini-3.1-pro-preview.sh

set -euo pipefail

MAX_TRIALS="${MAX_TRIALS:-20}"
TRIAL_WAIT_SECONDS="${TRIAL_WAIT_SECONDS:-60}"  # Pause between trials.

export VARY_SPEC="question-depth"
export QUESTION_DEPTHS="4 8 16 32"
export MODEL_SPEC="gemini-3.1-pro-preview"
export INFERENCE_POOL="${INFERENCE_POOL:-${OUTPUT_ROOT:-./output/experiments.v1}/inference_pool.gemini-3.1-pro-preview.jsonl}"
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "$SCRIPT_DIR/config.sh"
OUTPUT="$(experiment_dir "$VARY_SPEC")/inference/reasoning_effort=${REASONING_EFFORT}/model=${MODEL_SPEC}.json"

for (( TRIAL = 1; TRIAL <= MAX_TRIALS; TRIAL++ )); do
    printf '\n[TRIAL %d/%d] %s\n' "$TRIAL" "$MAX_TRIALS" "$(date '+%Y-%m-%d %H:%M:%S')"
    bash "$SCRIPT_DIR/inference.sh"
    # run_inference.py exits 0 even when requests failed, so read its output.
    if [[ ! -f "$OUTPUT" ]]; then
        printf '[TRIAL %d/%d] no output at %s; stopping.\n' "$TRIAL" "$MAX_TRIALS" "$OUTPUT" >&2
        exit 1
    fi
    UNANSWERED="$("$PYTHON_BIN" -c \
        'import orjson, sys; m = orjson.loads(open(sys.argv[1], "rb").read())["metadata"]; print(m["selected_examples"] - m["selected_answered_examples"])' \
        "$OUTPUT")"
    if (( UNANSWERED == 0 )); then
        printf '\n[TRIAL %d/%d] every selected example is answered.\n' "$TRIAL" "$MAX_TRIALS"
        exit 0
    fi
    printf '\n[TRIAL %d/%d] %d selected examples have no answer yet.\n' \
        "$TRIAL" "$MAX_TRIALS" "$UNANSWERED"
    if (( TRIAL < MAX_TRIALS )); then
        printf 'Waiting %ss before the next trial.\n' "$TRIAL_WAIT_SECONDS"
        sleep "$TRIAL_WAIT_SECONDS"
    fi
done

printf '\n%d selected examples still have no answer after %d trials. Output: %s\n' \
    "$UNANSWERED" "$MAX_TRIALS" "$OUTPUT" >&2
exit 1
