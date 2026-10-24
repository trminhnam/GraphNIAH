#!/bin/bash
# Run question depths 16 and 32 with qwen3.5-397b-a17b.
# Other inference settings are passed to the shared launcher.
#
#   bash scripts/experiments/v1/inference.1_question_depth.qwen3.5-397b-a17b.sh

set -euo pipefail

export VARY_SPEC="question-depth"
export MODEL_SPEC="qwen3.5-397b-a17b"
export INFERENCE_POOL="${INFERENCE_POOL:-${OUTPUT_ROOT:-./output/experiments.v1}/inference_pool.qwen3.5-397b-a17b.jsonl}"
export QUESTION_DEPTHS="4 8 16 32"
exec bash "$(dirname "${BASH_SOURCE[0]}")/inference.sh"
