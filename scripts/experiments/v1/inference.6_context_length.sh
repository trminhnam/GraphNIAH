#!/bin/bash
# Run 500k and 922k with one reasoning effort (default low).
# Default to 50 examples per context length; MAX_SAMPLES overrides this.
# At plotting time, reuse the 200k reasoning_effort result for the same model,
# effort, settings, and example count. This script does not rerun 200k.
# MODEL_SPEC, INFERENCE_POOL, and other settings pass to the shared launcher.
# Models exceeding their configured context limit are skipped there.
#
#   bash scripts/experiments/v1/inference.6_context_length.sh

set -euo pipefail

export MAX_SAMPLES="${MAX_SAMPLES-50}"

printf '200k: reuse the existing reasoning_effort result at plotting time.\n'
for CONTEXT_LENGTH in 500000 922000; do
    CONTEXT_LENGTH="$CONTEXT_LENGTH" VARY_SPEC=context_length \
        bash "$(dirname "${BASH_SOURCE[0]}")/inference.sh"
done
