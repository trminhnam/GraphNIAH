#!/bin/bash
# Send 100 examples per studied value to every model (run_inference.py).
# Experiments and models come from config.sh (VARY_SPEC / MODEL_SPEC override).
# Rerun the same command to resume: saved answers are kept, failed requests are
# sent again, and examples appended to an experiment are picked up.
# Only reasoning_effort runs low, medium, and high on the same dataset. Other
# experiments use REASONING_EFFORT (default low). Outputs are separate per effort.
# All experiments reuse INFERENCE_POOL; run these launchers one at a time.
# QUESTION_DEPTHS="16 32" optionally selects depths before limiting examples.
# COMPONENT_NODE_COUNTS="200 250" optionally selects component sizes.
#
#   bash scripts/experiments/v1/inference.sh
#   MODEL_SPEC=gpt-5.6-terra VARY_SPEC=question-depth MAX_SAMPLES=100 bash scripts/experiments/v1/inference.sh

set -euo pipefail

################ Hyperparameters ####################
# Examples per studied value; reasoning_effort uses the same N examples per effort.
# Set MAX_SAMPLES= to use all available examples.
MAX_SAMPLES="${MAX_SAMPLES-100}"
MAX_WORKERS="${MAX_WORKERS:-16}"  # Concurrent requests per model.
TEMPERATURE="${TEMPERATURE:-1}"
TOP_P="${TOP_P:-0.95}"
FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0}"
#####################################################

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

print_header "INFERENCE"
printf '  Experiments    : %s\n' "${VARY_VALUES[*]}"
printf '  Models         : %s\n' "${MODEL_SLUGS[*]}"
printf '  Examples/value : %s\n' "${MAX_SAMPLES:-all}"
printf '  Settings       : temperature=%s top_p=%s default_reasoning=%s workers=%s\n' \
    "$TEMPERATURE" "$TOP_P" "$REASONING_EFFORT" "$MAX_WORKERS"
printf '  Root           : %s\n' "$(growth_dir)"
printf '  Inference pool : %s\n' "$INFERENCE_POOL"

SESSION_START=$SECONDS
RUNS=0
for MODEL_SLUG in "${MODEL_SLUGS[@]}"; do
    MODEL_LIMIT="$(model_context_limit "$MODEL_SLUG")"
    if (( CONTEXT_LENGTH > MODEL_LIMIT )); then
        printf '\n[skip] %s accepts %s tokens, fewer than %s\n' \
            "$MODEL_SLUG" "$MODEL_LIMIT" "$CONTEXT_LENGTH"
        continue
    fi
    for VARY in "${VARY_VALUES[@]}"; do
        EXPERIMENT_DIR="$(experiment_dir "$VARY")"
        EXPERIMENT="$EXPERIMENT_DIR/experiment.jsonl"
        if [[ ! -s "$EXPERIMENT" ]]; then
            printf '\n[skip] no experiment yet -> %s\n' "$EXPERIMENT"
            continue
        fi
        read -r -a EFFORTS <<< "$(reasoning_efforts "$VARY")"
        for EFFORT in "${EFFORTS[@]}"; do
            RUNS=$((RUNS + 1))
            printf '\n[Run %d] model=%s vary=%s reasoning_effort=%s\n' \
                "$RUNS" "$MODEL_SLUG" "$VARY" "$EFFORT"
            ARGS=(
                --pool "$POOL"
                --inference-pool "$INFERENCE_POOL"
                --experiment "$EXPERIMENT"
                --output "$EXPERIMENT_DIR/inference/reasoning_effort=${EFFORT}/model=${MODEL_SLUG}.json"
                --model "$MODEL_SLUG"
                --temperature "$TEMPERATURE"
                --top-p "$TOP_P"
                --frequency-penalty "$FREQUENCY_PENALTY"
                --presence-penalty "$PRESENCE_PENALTY"
                --reasoning-effort "$EFFORT"
                --max-workers "$MAX_WORKERS"
            )
            if [[ -n "$MAX_SAMPLES" ]]; then
                ARGS+=(--max-samples "$MAX_SAMPLES")
            fi
            if [[ -n "${QUESTION_DEPTHS:-}" ]]; then
                read -r -a DEPTHS <<< "$QUESTION_DEPTHS"
                ARGS+=(--question-depths "${DEPTHS[@]}")
            fi
            if [[ -n "${COMPONENT_NODE_COUNTS:-}" ]]; then
                read -r -a NODE_COUNTS <<< "$COMPONENT_NODE_COUNTS"
                ARGS+=(--component-node-counts "${NODE_COUNTS[@]}")
            fi
            "$PYTHON_BIN" run_inference.py "${ARGS[@]}"
        done
    done
done

printf '\nINFERENCE finished: %d run(s) in %s\n' \
    "$RUNS" "$(format_duration $((SECONDS - SESSION_START)))"
