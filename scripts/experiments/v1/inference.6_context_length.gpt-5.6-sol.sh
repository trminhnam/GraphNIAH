#!/bin/bash
# Run the context-length experiment with gpt-5.6-sol and its own inference pool.
# Other inference settings pass to the shared context-length launcher.
#
#   bash scripts/experiments/v1/inference.6_context_length.gpt-5.6-sol.sh

set -euo pipefail

export MODEL_SPEC="gpt-5.6-sol"
export INFERENCE_POOL="${INFERENCE_POOL:-${OUTPUT_ROOT:-./output/experiments.v1}/inference_pool.gpt-5.6-sol.jsonl}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/inference.6_context_length.sh"
