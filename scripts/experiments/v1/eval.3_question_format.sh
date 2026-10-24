#!/bin/bash
# Evaluate the question-format experiment with REASONING_EFFORT (default low).
# Evaluate all eight models by default; MODEL_SPEC overrides the model list.
#
#   bash scripts/experiments/v1/eval.3_question_format.sh

set -euo pipefail

export VARY_SPEC="question-format"
exec bash "$(dirname "${BASH_SOURCE[0]}")/eval.sh"
