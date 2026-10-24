#!/bin/bash
# Run the distractor-type experiment with REASONING_EFFORT (default low).
# MODEL_SPEC and other inference settings are passed to the shared launcher.
#
#   bash scripts/experiments/v1/inference.4_distractor_type.sh

set -euo pipefail

export VARY_SPEC="distractor-type"
exec bash "$(dirname "${BASH_SOURCE[0]}")/inference.sh"
