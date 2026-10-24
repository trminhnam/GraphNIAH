#!/bin/bash
# Growth-parameter search, step 1: for every alpha of the grid in config.sh,
# build components-only examples (no distractors or prompts) for the
# question-format experiment, so the three formats are balanced and depth and
# node count are drawn as in the real experiments. Every alpha uses the same
# seed, so all alphas get the same planned examples and differ only in growth.
#
# One create_growth_search.v1.py run builds every alpha: its main process and
# workers start once, and a free worker moves on to the next alpha. Each alpha
# still gets its own pool and experiment files (the same files one
# create_dataset.v1.py --components-only run per alpha writes), so jobs with
# different alphas can run at the same time:
#
#   ALPHA_SPEC=0.3 bash scripts/experiments/v1/growth_search.1_create_data.sh
#
# Rerun to resume. To grow only the best alphas later (e.g. to 5,000 examples):
#
#   ALPHA_SPEC=0.3 GROWTH_EXAMPLES_PER_VALUE=1667 bash scripts/experiments/v1/growth_search.1_create_data.sh

set -euo pipefail

################ Hyperparameters ####################
MAX_ATTEMPTS="${MAX_ATTEMPTS:-500}"  # New-seed attempts per slot (same config).
#####################################################

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
check_inputs

CANDIDATES=${#GROWTH_ALPHA_VALUES[@]}
print_header "GROWTH SEARCH | create data"
printf '  Alpha values   : %s\n' "${GROWTH_ALPHA_VALUES[*]}"
printf '  Candidates     : %s, each %s examples per format (vary=%s)\n' \
    "$CANDIDATES" "$GROWTH_EXAMPLES_PER_VALUE" "$GROWTH_VARY"
printf '  Input paths    : %s\n' "${PATHS_FILES[*]}"
printf '  Graph DB       : %s\n' "$GRAPH_DB"
printf '  Context length : %s tokens\n' "$CONTEXT_LENGTH"
printf '  Seed           : %s\n' "$RANDOM_SEED"
printf '  Cutoff date    : %s\n' "$CUTOFF_DATE"
printf '  Workers        : %s\n' "$NUM_WORKERS"
printf '  Max attempts   : %s per slot\n' "$MAX_ATTEMPTS"
printf '  Root           : %s\n\n' "$GROWTH_SETTINGS_DIR"

STARTED=$SECONDS
mkdir -p "$GROWTH_SETTINGS_DIR"
# Load tokenizers from the local Hugging Face cache only: no update check
# from every worker. To download a missing one, run with HF_HUB_OFFLINE=0.
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" "$PYTHON_BIN" create_growth_search.v1.py \
    --paths "${PATHS_FILES[@]}" \
    --graph-db "$GRAPH_DB" \
    --search-root "$GROWTH_SETTINGS_DIR" \
    --alpha "${GROWTH_ALPHA_VALUES[@]}" \
    --vary "$GROWTH_VARY" \
    --examples-per-value "$GROWTH_EXAMPLES_PER_VALUE" \
    --context-length "$CONTEXT_LENGTH" \
    --random-seed "$RANDOM_SEED" \
    --cutoff-date "$CUTOFF_DATE" \
    --max-attempts "$MAX_ATTEMPTS" \
    --workers "$NUM_WORKERS"

printf '\nGROWTH SEARCH | create data finished: %d alpha(s) in %s\n' \
    "$CANDIDATES" "$(format_duration $((SECONDS - STARTED)))"
