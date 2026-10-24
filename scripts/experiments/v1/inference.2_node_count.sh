#!/bin/bash
# Run the node-count experiment with REASONING_EFFORT (default low).
# MODEL_SPEC and other inference settings are passed to the shared launcher.
#
#   bash scripts/experiments/v1/inference.2_node_count.sh

set -euo pipefail

export VARY_SPEC="node-count"
exec bash "$(dirname "${BASH_SOURCE[0]}")/inference.sh"
