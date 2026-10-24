#!/bin/bash
# Run the question-format experiment with REASONING_EFFORT (default low).
# MODEL_SPEC and other inference settings are passed to the shared launcher.
#
#   bash scripts/experiments/v1/inference.3_question_format.sh

set -euo pipefail

export VARY_SPEC="question-format"
exec bash "$(dirname "${BASH_SOURCE[0]}")/inference.sh"
