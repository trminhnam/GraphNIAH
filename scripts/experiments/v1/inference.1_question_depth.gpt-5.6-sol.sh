#!/bin/bash
# Run question depths 16 and 32 with gpt-5.6-sol.
# Other inference settings are passed to the shared launcher.
#
#   bash scripts/experiments/v1/inference.1_question_depth.gpt-5.6-sol.sh

set -euo pipefail

export VARY_SPEC="question-depth"
export MODEL_SPEC="gpt-5.6-sol"
export INFERENCE_POOL="${INFERENCE_POOL:-${OUTPUT_ROOT:-./output/experiments.v1}/inference_pool.gpt-5.6-sol.jsonl}"
export QUESTION_DEPTHS="8 16 32"
exec bash "$(dirname "${BASH_SOURCE[0]}")/inference.sh"
