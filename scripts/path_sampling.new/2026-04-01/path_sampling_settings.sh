#!/bin/bash
# Shared output locations and sampling grid for sampling and expansion.
# Earliest publication date for sampled path papers and expanded neighbors.
CUTOFF_DATE="${CUTOFF_DATE:-2026-04-01}"
# Fields of study of the path papers (sample_paths_bulk_graph.sh --category):
# cs = Computer Science papers only; all = any field. A category other than
# all gets its own .<category> data folder and graph DB, so it never mixes
# with the all-field data.
PATH_CATEGORY="${PATH_CATEGORY:-all}"
if [[ "$PATH_CATEGORY" == all ]]; then
    PATH_CATEGORY_SUFFIX=""
else
    PATH_CATEGORY_SUFFIX=".${PATH_CATEGORY}"
fi
DATA_DIR="${DATA_DIR:-./output/data.${CUTOFF_DATE}${PATH_CATEGORY_SUFFIX}}"
SHARED_DB_PATH="${SHARED_DB_PATH:-./resources/semantic_scholar_graph.${CUTOFF_DATE}${PATH_CATEGORY_SUFFIX}/semantic_scholar_paths.db}"
# construct_paths_from_semantic_scholar.py expects path length in nodes.
HOP_COUNTS=(10 22 46 94) # 4, 8, 16, 32
TRAVERSAL_MODES=(papers-only)
