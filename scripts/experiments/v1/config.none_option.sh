#!/bin/bash
# Shared paths and model list for the experiment-3 appended-None comparison.
# Source this file from the stage scripts; run them from the repository root.

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

if [[ -z "${MODEL_SPEC:-}" ]]; then
    MODEL_SLUGS=(
        gpt-5.6-luna
        gpt-5.6-sol
        gemini-3.8-flash
        gemini-3.6-flash
        qwen3.5-35b-a3b
        qwen3.5-122b-a10b
        qwen3.5-397b-a17b
    )
fi
SOURCE_EXPERIMENT_DIR="$(experiment_dir question-format)"

none_option_dir() {
    printf '%s/none-option/reasoning_effort=%s/model=%s' \
        "$SOURCE_EXPERIMENT_DIR" "$REASONING_EFFORT" "$1"
}
