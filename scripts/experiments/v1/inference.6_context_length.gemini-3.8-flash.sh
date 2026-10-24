#!/bin/bash
# Run the context-length experiment with gemini-3.8-flash and its own inference pool.
# Other inference settings pass to the shared context-length launcher.
#
#   bash scripts/experiments/v1/inference.6_context_length.gemini-3.8-flash.sh

set -euo pipefail

export MODEL_SPEC="gemini-3.8-flash"
export INFERENCE_POOL="${INFERENCE_POOL:-${OUTPUT_ROOT:-./output/experiments.v1}/inference_pool.gemini-3.8-flash.jsonl}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/inference.6_context_length.sh"
