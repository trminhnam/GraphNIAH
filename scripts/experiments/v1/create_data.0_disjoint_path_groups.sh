#!/bin/bash
# Step 0, before any create_data.N_*.sh: for each input paths file (one per
# question depth k, paths of 3k - 2 papers), check every path once, find the
# conflicting pairs, and store every disjoint group of 2, 4, 8, and 16 paths when there
# are at most GROUP_LIMIT (else only the conflicts, for draw-and-retry). Writes
# disjoint_path_groups.json next to each sampled_paths.json. Rerun it whenever
# the paths, the graph DB, or CUTOFF_DATE change; create_dataset.v1.py refuses
# a cache built from other inputs. The caches sit next to the paths files and
# are shared with scripts/experiments/v2, so running either step 0 is enough.
#
#   bash scripts/experiments/v1/create_data.0_disjoint_path_groups.sh
#   NUM_WORKERS=16 bash scripts/experiments/v1/create_data.0_disjoint_path_groups.sh

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

for path in "${PATHS_FILES[@]}" "$GRAPH_DB"; do
    if [[ ! -f "$path" ]]; then
        printf 'Missing input: %s\n' "$path" >&2
        exit 1
    fi
done

print_header "DISJOINT PATH GROUPS"
printf '  Input paths    : %s\n' "${PATHS_FILES[*]}"
printf '  Graph DB       : %s\n' "$GRAPH_DB"
printf '  Cutoff date    : %s\n' "$CUTOFF_DATE"
printf '  Group limit    : %s\n' "$GROUP_LIMIT"
printf '  Workers        : %s\n' "$NUM_WORKERS"

STARTED=$SECONDS
"$PYTHON_BIN" endpoints/precompute_disjoint_path_groups.py \
    --paths "${PATHS_FILES[@]}" \
    --graph-db "$GRAPH_DB" \
    --cutoff-date "$CUTOFF_DATE" \
    --group-limit "$GROUP_LIMIT" \
    --workers "$NUM_WORKERS"
printf '\nDISJOINT PATH GROUPS finished in %s\n' "$(format_duration $((SECONDS - STARTED)))"
