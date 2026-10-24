#!/bin/bash
# Run component sizes 200 and 250 with gpt-5.6-sol.
# Other inference settings are passed to the shared launcher.
#
#   bash scripts/experiments/v1/inference.2_node_count.gpt-5.6-sol.sh

set -euo pipefail

export VARY_SPEC="node-count"
export MODEL_SPEC="gpt-5.6-sol"
export COMPONENT_NODE_COUNTS="200 250"
export INFERENCE_POOL="${INFERENCE_POOL:-${OUTPUT_ROOT:-./output/experiments.v1}/inference_pool.gpt-5.6-sol.jsonl}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/inference.sh"
