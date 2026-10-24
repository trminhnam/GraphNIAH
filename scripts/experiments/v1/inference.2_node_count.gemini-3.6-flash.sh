#!/bin/bash
# Run the node-count experiment with gemini-3.6-flash.
# Other inference settings are passed to the shared launcher.
#
# Gemini stays on the flex tier, so a busy flex tier (HTTP 503) leaves failed
# examples. Each trial reruns the launcher, which sends only the examples that
# have no answer yet. Trials stop early once every selected example is answered.
#
#   bash scripts/experiments/v1/inference.2_node_count.gemini-3.6-flash.sh
#   MAX_TRIALS=5 TRIAL_WAIT_SECONDS=300 bash scripts/experiments/v1/inference.2_node_count.gemini-3.6-flash.sh

set -euo pipefail

MAX_TRIALS="${MAX_TRIALS:-20}"
TRIAL_WAIT_SECONDS="${TRIAL_WAIT_SECONDS:-60}"  # Pause between trials.

export VARY_SPEC="node-count"
export MODEL_SPEC="gemini-3.6-flash"
export INFERENCE_POOL="${INFERENCE_POOL:-${OUTPUT_ROOT:-./output/experiments.v1}/inference_pool.gemini-3.6-flash.jsonl}"
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
        'import json, sys; m = json.load(open(sys.argv[1]))["metadata"]; print(m["selected_examples"] - m["selected_answered_examples"])' \
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
