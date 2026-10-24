#!/bin/bash
# Growth-parameter search, step 2: measure paper degrees in every alpha's
# examples and write one file per question format and alpha under
# <growth search root>/question_format=<format>/graph_statistics/.
#
#   bash scripts/experiments/v1/growth_search.2_degrees.sh

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

print_header "GROWTH SEARCH | degree distributions"
printf '  Alpha values   : %s\n' "${GROWTH_ALPHA_VALUES[*]}"
printf '  Root           : %s\n\n' "$GROWTH_SETTINGS_DIR"

"$PYTHON_BIN" endpoints/calculate_sample_paper_degree_distribution_v2.py \
    --search-root "$GROWTH_SETTINGS_DIR" \
    --parameters alpha \
    --vary "$GROWTH_VARY" \
    --alpha "${GROWTH_ALPHA_VALUES[@]}"
