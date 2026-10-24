#!/bin/bash
# Evaluate 500k and 922k at one reasoning effort (default low).
# Evaluate all eight models by default; MODEL_SPEC overrides the model list.
# At plotting time, use the existing 200k vary=reasoning_effort/eval report
# for the same model, effort, settings, and example count. It is not rewritten.
#
#   bash scripts/experiments/v1/eval.6_context_length.sh

set -euo pipefail

printf '200k: reuse the existing reasoning_effort evaluation at plotting time.\n'
for CONTEXT_LENGTH in 500000 922000; do
    CONTEXT_LENGTH="$CONTEXT_LENGTH" VARY_SPEC=context_length \
        bash "$(dirname "${BASH_SOURCE[0]}")/eval.sh"
done
