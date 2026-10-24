#!/bin/bash
# Evaluate low, medium, and high reasoning effort on the same randomized dataset.
# Evaluate all eight models by default; MODEL_SPEC overrides the model list.
#
#   bash scripts/experiments/v1/eval.5_reasoning_effort.sh

set -euo pipefail

export VARY_SPEC="reasoning_effort"
exec bash "$(dirname "${BASH_SOURCE[0]}")/eval.sh"
