#!/bin/bash
# Evaluate the distractor-type experiment with REASONING_EFFORT (default low).
# Evaluate all eight models by default; MODEL_SPEC overrides the model list.
#
#   bash scripts/experiments/v1/eval.4_distractor_type.sh

set -euo pipefail

export VARY_SPEC="distractor-type"
exec bash "$(dirname "${BASH_SOURCE[0]}")/eval.sh"
