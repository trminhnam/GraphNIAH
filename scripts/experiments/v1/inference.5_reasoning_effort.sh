#!/bin/bash
# Compare low, medium, and high reasoning effort on the same randomized dataset.
# MODEL_SPEC and other inference settings are passed to the shared launcher.
#
#   bash scripts/experiments/v1/inference.5_reasoning_effort.sh

set -euo pipefail

export VARY_SPEC="reasoning_effort"
exec bash "$(dirname "${BASH_SOURCE[0]}")/inference.sh"
