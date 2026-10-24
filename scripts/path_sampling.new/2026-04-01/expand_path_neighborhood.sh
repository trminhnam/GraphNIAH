#!/bin/bash
set -eo pipefail

# Run with the project's niah environment activated.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi
set -u
source "$SCRIPT_DIR/path_sampling_settings.sh"
export PYTHONUNBUFFERED=1

PATH_FILES=()
for TRAVERSAL_MODE in "${TRAVERSAL_MODES[@]}"; do
    for HOP_COUNT in "${HOP_COUNTS[@]}"; do
        PATH_FILE="$DATA_DIR/hops=$HOP_COUNT/$TRAVERSAL_MODE/sampled_paths.json"
        if [[ ! -f "$PATH_FILE" ]]; then
            echo "Skipping missing sampled paths file: $PATH_FILE" >&2
            continue
        fi
        PATH_FILES+=("$PATH_FILE")
    done
done

if [[ ${#PATH_FILES[@]} -eq 0 ]]; then
    echo "No sampled paths files found; nothing to expand."
    exit 0
fi

# Level 0 is every sampled path paper; expand levels 0-19 and store level 20.
# One invocation deduplicates every frontier and reuses one Spark session.
# Reruns resume the database checkpoint; pass --restart to rebuild it.
python -u endpoints/expand_path_neighborhood.py \
    "${PATH_FILES[@]}" \
    --db-path "$SHARED_DB_PATH" \
    --max-level 20 \
    --bulk-batch-api-group-size 10000 \
    --bulk-title-search-batch-size 10000 \
    --bulk-edge-budget 500000 \
    --seed-missing-papers \
    --cutoff-date "$CUTOFF_DATE" \
    "$@"
