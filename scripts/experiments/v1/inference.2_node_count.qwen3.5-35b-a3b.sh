#!/bin/bash
# Run the node-count experiment with qwen3.5-35b-a3b.
# Other inference settings are passed to the shared launcher.
#
#   bash scripts/experiments/v1/inference.2_node_count.qwen3.5-35b-a3b.sh

set -euo pipefail

export VARY_SPEC="node-count"
export MODEL_SPEC="qwen3.5-35b-a3b"
export INFERENCE_POOL="${INFERENCE_POOL:-${OUTPUT_ROOT:-./output/experiments.v1}/inference_pool.qwen3.5-35b-a3b.jsonl}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/inference.sh"
