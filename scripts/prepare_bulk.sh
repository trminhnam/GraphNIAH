#!/bin/bash
# Download one bulk release and build the tables used by the v1 pipeline.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

# This is the release in the source workflow, not a claim about a frozen dataset.
RELEASE_ID="${RELEASE_ID:-2026-10-06}"
CPUS="${CPUS:-16}"
DRIVER_MEMORY="${DRIVER_MEMORY:-96g}"
python endpoints/download_semantic_scholar_datasets.py \
    --release-id "$RELEASE_ID" --datasets papers paper-ids citations \
    --num-workers "${DOWNLOAD_WORKERS:-4}"
python build_semantic_scholar_bulk_api_index.py \
    --resume-existing \
    --spark-master "local[$CPUS]" --driver-memory "$DRIVER_MEMORY" \
    --local-dir "${SPARK_LOCAL_DIR:-/tmp/benchmark-spark}" \
    --shuffle-partitions 512 --num-buckets 2048 \
    --input-max-partition-bytes 256m \
    --skip-table external_ids --skip-table authors --skip-table author_papers \
    --skip-table abstracts --skip-table tldrs --skip-table publication_venues \
    --skip-table embeddings --skip-table s2orc_text
