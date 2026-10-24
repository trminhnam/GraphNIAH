#!/bin/bash
# Run component sizes 200 and 250 with qwen3.5-397b-a17b.
# Other inference settings are passed to the shared launcher.
#
#   bash scripts/experiments/v1/inference.2_node_count.qwen3.5-397b-a17b.sh

set -euo pipefail

export VARY_SPEC="node-count"
export MODEL_SPEC="qwen3.5-397b-a17b"
export COMPONENT_NODE_COUNTS="100 150 200 250"
export INFERENCE_POOL="${INFERENCE_POOL:-${OUTPUT_ROOT:-./output/experiments.v1}/inference_pool.qwen3.5-397b-a17b.jsonl}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/inference.sh"
