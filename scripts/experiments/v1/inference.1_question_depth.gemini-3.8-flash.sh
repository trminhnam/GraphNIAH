#!/bin/bash
# Run the question-depth experiment with gemini-3.8-flash.
# Other inference settings are passed to the shared launcher.
#
#   bash scripts/experiments/v1/inference.1_question_depth.gemini-3.8-flash.sh

set -euo pipefail

export VARY_SPEC="question-depth"
export MODEL_SPEC="gemini-3.8-flash"
export INFERENCE_POOL="${INFERENCE_POOL:-${OUTPUT_ROOT:-./output/experiments.v1}/inference_pool.gemini-3.8-flash.jsonl}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/inference.sh"
