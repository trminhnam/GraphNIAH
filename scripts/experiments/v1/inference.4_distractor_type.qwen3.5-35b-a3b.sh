#!/bin/bash
# Run the distractor-type experiment with qwen3.5-35b-a3b.
# Other inference settings are passed to the shared launcher.
#
#   bash scripts/experiments/v1/inference.4_distractor_type.qwen3.5-35b-a3b.sh

set -euo pipefail

export VARY_SPEC="distractor-type"
export MODEL_SPEC="qwen3.5-35b-a3b"
export INFERENCE_POOL="${INFERENCE_POOL:-${OUTPUT_ROOT:-./output/experiments.v1}/inference_pool.qwen3.5-35b-a3b.jsonl}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/inference.sh"
