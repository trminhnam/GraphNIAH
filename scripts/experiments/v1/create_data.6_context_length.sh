#!/bin/bash
# Context-length experiment: draw every data factor at random (--vary none).
# Build only 500k and 922k. At plotting time, reuse the existing 200k data and
# results under vary=reasoning_effort with the same settings and example count.
# EXAMPLES_PER_VALUE is the total example count at each context length.
#
#   bash scripts/experiments/v1/create_data.6_context_length.sh

set -euo pipefail

EXAMPLES_PER_VALUE="${EXAMPLES_PER_VALUE:-500}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-500}"

printf '200k: reuse the existing reasoning_effort experiment at plotting time.\n'
for CONTEXT_LENGTH in 500000 922000; do
    (
        source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
        create_data context_length "$EXAMPLES_PER_VALUE" "$MAX_ATTEMPTS"
    )
done
