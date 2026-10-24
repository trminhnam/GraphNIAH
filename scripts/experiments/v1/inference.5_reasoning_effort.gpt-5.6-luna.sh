#!/bin/bash
# Compare low, medium, and high reasoning effort with gpt-5.6-luna.
# Other inference settings are passed to the shared launcher.
#
#   bash scripts/experiments/v1/inference.5_reasoning_effort.gpt-5.6-luna.sh

set -euo pipefail

export VARY_SPEC="reasoning_effort"
export MODEL_SPEC="gpt-5.6-luna"
export INFERENCE_POOL="${INFERENCE_POOL:-${OUTPUT_ROOT:-./output/experiments.v1}/inference_pool.gpt-5.6-luna.jsonl}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/inference.sh"
