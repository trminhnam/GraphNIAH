#!/bin/bash
# Evaluate the node-count experiment with REASONING_EFFORT (default low).
# Evaluate all eight models by default; MODEL_SPEC overrides the model list.
#
#   bash scripts/experiments/v1/eval.2_node_count.sh

set -euo pipefail

export VARY_SPEC="node-count"
exec bash "$(dirname "${BASH_SOURCE[0]}")/eval.sh"
