#!/bin/bash
# Compare low, medium, and high reasoning effort with qwen3.5-122b-a10b.
# Other inference settings are passed to the shared launcher.
#
#   bash scripts/experiments/v1/inference.5_reasoning_effort.qwen3.5-122b-a10b.sh

set -euo pipefail

export VARY_SPEC="reasoning_effort"
export MODEL_SPEC="qwen3.5-122b-a10b"
export INFERENCE_POOL="${INFERENCE_POOL:-${OUTPUT_ROOT:-./output/experiments.v1}/inference_pool.qwen3.5-122b-a10b.jsonl}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/inference.sh"
