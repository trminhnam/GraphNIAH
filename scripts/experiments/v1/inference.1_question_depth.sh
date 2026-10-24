#!/bin/bash
# Run the question-depth experiment with REASONING_EFFORT (default low).
# MODEL_SPEC and other inference settings are passed to the shared launcher.
#
#   bash scripts/experiments/v1/inference.1_question_depth.sh

set -euo pipefail

export VARY_SPEC="question-depth"
exec bash "$(dirname "${BASH_SOURCE[0]}")/inference.sh"
