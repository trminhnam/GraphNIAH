#!/bin/bash
# Growth-parameter search, step 3: pick the alpha whose in-degree distribution
# is closest to the real citation graph (KL, degree cutoffs K,
# bootstrap). Check bootstrap_selection_frequency in each summary_by_K.csv: if
# the winner is not stable at the cutoffs you care about, grow only the top 2-3
# alphas with step 1 and run steps 2 and 3 again.
#
#   bash scripts/experiments/v1/growth_search.3_search.sh
#   SEARCH_WORKERS=8 BOOTSTRAP_REPEATS=500 bash scripts/experiments/v1/growth_search.3_search.sh
#   POWER_LAW_FIT=1 bash scripts/experiments/v1/growth_search.3_search.sh

set -euo pipefail

################ Hyperparameters ####################
SEARCH_WORKERS="${SEARCH_WORKERS:-4}"
BOOTSTRAP_REPEATS="${BOOTSTRAP_REPEATS:-500}"
# 1: draw the power law fitted to the generated examples on the log-log panels.
POWER_LAW_FIT="${POWER_LAW_FIT:-0}"
# Bootstrap repeats of the power-law analysis in each K folder (0 skips it).
POWER_LAW_REPEATS="${POWER_LAW_REPEATS:-1000}"
#####################################################

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
# The path category's suffix keeps cs and all searches in separate folders.
SEARCH_OUTPUT_DIR="${SEARCH_OUTPUT_DIR:-./output/graph_statistics/alpha_in_degree_search.v1${PATH_CATEGORY_SUFFIX}}"

print_header "GROWTH SEARCH | in-degree search"
printf '  Alpha values   : %s\n' "${GROWTH_ALPHA_VALUES[*]}"
printf '  Root           : %s\n' "$GROWTH_SETTINGS_DIR"
printf '  Output         : %s\n' "$SEARCH_OUTPUT_DIR"
printf '  Global files   : %s\n\n' "${GLOBAL_DISTRIBUTIONS[*]}"
for GLOBAL_DISTRIBUTION in "${GLOBAL_DISTRIBUTIONS[@]}"; do
    if [[ ! -f "$GLOBAL_DISTRIBUTION" ]]; then
        printf 'Missing global degree distribution: %s\n' "$GLOBAL_DISTRIBUTION" >&2
        printf 'Make it with: CUTOFF_DATE=%s bash scripts/db_utils/calculate_bulk_paper_degree_distribution.sh\n' \
            "$CUTOFF_DATE" >&2
        exit 1
    fi
done

if [[ "$POWER_LAW_FIT" == 1 ]]; then
    POWER_LAW_OPTION=--power-law-fit
else
    POWER_LAW_OPTION=--no-power-law-fit
fi

"$PYTHON_BIN" endpoints/search_alpha_in_degree_distributions_v1.py \
    --experiment-root "$GROWTH_SETTINGS_DIR" \
    --global-distribution "${GLOBAL_DISTRIBUTIONS[@]}" \
    --degree-cutoffs 0 1 \
    --methods kl \
    --output-dir "$SEARCH_OUTPUT_DIR" \
    --alpha "${GROWTH_ALPHA_VALUES[@]}" \
    --bootstrap-repeats "$BOOTSTRAP_REPEATS" \
    "$POWER_LAW_OPTION" \
    --power-law-repeats "$POWER_LAW_REPEATS" \
    --workers "$SEARCH_WORKERS"
