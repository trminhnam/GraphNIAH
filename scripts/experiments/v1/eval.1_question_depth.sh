#!/bin/bash
# Evaluate the question-depth experiment with REASONING_EFFORT (default low).
# Evaluate all eight models by default; MODEL_SPEC overrides the model list.
#
#   bash scripts/experiments/v1/eval.1_question_depth.sh

set -euo pipefail

export VARY_SPEC="question-depth"
exec bash "$(dirname "${BASH_SOURCE[0]}")/eval.sh"
