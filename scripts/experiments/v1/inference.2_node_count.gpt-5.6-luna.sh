#!/bin/bash
# Run the node-count experiment with gpt-5.6-luna.
# Other inference settings are passed to the shared launcher.
#
#   bash scripts/experiments/v1/inference.2_node_count.gpt-5.6-luna.sh

set -euo pipefail

export VARY_SPEC="node-count"
export MODEL_SPEC="gpt-5.6-luna"
export INFERENCE_POOL="${INFERENCE_POOL:-${OUTPUT_ROOT:-./output/experiments.v1}/inference_pool.gpt-5.6-luna.jsonl}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/inference.sh"
