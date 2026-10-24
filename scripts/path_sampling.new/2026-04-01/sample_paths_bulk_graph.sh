#!/bin/bash
# Sample papers-only paths with construct_paths_from_bulk_graph.py: no seed
# papers, one pass over the citation graph among the papers published on or
# after CUTOFF_DATE (titles with MathML are excluded there). PATH_CATEGORY in
# path_sampling_settings.sh picks the fields of the path papers (cs or all)
# and their output folder and graph DB, which expand_path_neighborhood.sh in
# this folder reads unchanged. sample_paths.sh uses the same folders but does
# not filter fields.
#
# Rerunning extends this script's own output. A sampled_paths.json written by
# construct_paths_from_semantic_scholar.py at the same path is not extended:
# the run stops with a "different settings" error, so move that file away first.
set -eo pipefail

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

RANDOM_SEED="${RANDOM_SEED:-1}"
TARGET_NUM_PATHS=10000
# Each paper may appear in at most this many paths, so paths overlap less.
# A smaller value allows fewer paths; the log warns when the target is missed.
MAX_PATHS_PER_PAPER=1

for TRAVERSAL_MODE in "${TRAVERSAL_MODES[@]}"; do
    if [[ "$TRAVERSAL_MODE" != "papers-only" ]]; then
        echo "Skipping traversal mode $TRAVERSAL_MODE: construct_paths_from_bulk_graph.py samples papers-only paths only." >&2
        continue
    fi
    for HOP_COUNT in "${HOP_COUNTS[@]}"; do
        DATA_SOURCE_DIR="$DATA_DIR/hops=$HOP_COUNT/$TRAVERSAL_MODE"
        mkdir -p "$DATA_SOURCE_DIR"
        echo "SAMPLED_PATHS_FILE: $DATA_SOURCE_DIR/sampled_paths.json"

        python -u construct_paths_from_bulk_graph.py \
            --num-hops "$HOP_COUNT" \
            --target-num-paths "$TARGET_NUM_PATHS" \
            --start-date "$CUTOFF_DATE" \
            --category "$PATH_CATEGORY" \
            --output-dir "$DATA_SOURCE_DIR" \
            --db-path "$SHARED_DB_PATH" \
            --random-seed "$RANDOM_SEED" \
            --max-paths-per-paper "$MAX_PATHS_PER_PAPER"

        echo ""
        echo ""
    done
done
