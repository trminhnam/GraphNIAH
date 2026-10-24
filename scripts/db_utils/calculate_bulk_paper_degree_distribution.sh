#!/bin/bash
# Exact degree histograms over the Semantic Scholar bulk index.
#       scripts/db_utils/calculate_bulk_paper_degree_distribution.sh
set -euo pipefail

CUTOFF_DATE="${CUTOFF_DATE:-2026-04-01}"
INDEX_ROOT="${INDEX_ROOT:-resources/semantic_scholar_bulk/index}"
OUTPUT_DIR="${OUTPUT_DIR:-output/graph_statistics/bulk_paper_degree_distribution}"
# Broadcast (map-side) filtering avoids shuffling the whole citation table.
# 4096 MiB is half of Spark's 8 GiB broadcast table limit, and 32M IDs x 128 B
# (the code's conservative estimate) is about 3.8 GiB, well within the 512g driver.
BROADCAST_MAX_ELIGIBLE_IDS="${BROADCAST_MAX_ELIGIBLE_IDS:-32000000}"
BROADCAST_MAX_MIB="${BROADCAST_MAX_MIB:-4096}"

# Spark spills and shuffle files must land on node-local disk, never on /home.
SPARK_LOCAL_DIR="${SLURM_TMPDIR:-/tmp/s2-spark-${SLURM_JOB_ID:-manual}}"
CPUS="${SLURM_CPUS_PER_TASK:-64}"

df -h "$(dirname "${SPARK_LOCAL_DIR}")" || true

# One run per category writes both strict-reference-count files, because both
# modes share the same edge scan and degree calculation.
for CATEGORY in cs all; do
    echo "[degree-distribution] category=${CATEGORY} strict-reference-count=0,1 cutoff=${CUTOFF_DATE} cpus=${CPUS} local-dir=${SPARK_LOCAL_DIR} output-dir=${OUTPUT_DIR}"

    python endpoints/calculate_bulk_paper_degree_distribution.py \
        --index-root "${INDEX_ROOT}" \
        --cutoff-date "${CUTOFF_DATE}" \
        --category "${CATEGORY}" \
        --strict-reference-count 0 1 \
        --output-dir "${OUTPUT_DIR}" \
        --broadcast-max-eligible-ids "${BROADCAST_MAX_ELIGIBLE_IDS}" \
        --broadcast-max-mib "${BROADCAST_MAX_MIB}" \
        --spark-master "local[${CPUS}]" \
        --driver-memory 512g \
        --driver-max-result-size 32g \
        --shuffle-partitions 1024 \
        --input-max-partition-bytes 256m \
        --local-dir "${SPARK_LOCAL_DIR}"
done
