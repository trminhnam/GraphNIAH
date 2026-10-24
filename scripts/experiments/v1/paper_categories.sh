#!/bin/bash
# Category sets of the v1 papers: the Semantic Scholar fields of study of the
# papers on the sampled paths (sampled_paths) and in the growth search's
# examples, all alphas combined (generated_examples). Run after
# growth_search.1_create_data.sh; the bulk index must be on this machine.
# Writes <name>.<cutoff>.json to OUTPUT_DIR and prints each category's share.
#

set -euo pipefail

################ Hyperparameters ####################
INDEX_ROOT="${INDEX_ROOT:-resources/semantic_scholar_bulk/index}"
DRIVER_MEMORY="${DRIVER_MEMORY:-64g}"
#####################################################

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
# The path category's suffix keeps cs and all reports in separate folders.
OUTPUT_DIR="${OUTPUT_DIR:-./output/graph_statistics/paper_category_sets${PATH_CATEGORY_SUFFIX}}"
check_inputs

# Spark spills must land on node-local disk, never on /home.
SPARK_LOCAL_DIR="${SLURM_TMPDIR:-/tmp/s2-spark-${SLURM_JOB_ID:-manual}}"
# A shell attached with `srun --overlap --pty bash` has no SLURM_CPUS_PER_TASK.
CPUS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-8}}"

print_header "PAPER CATEGORIES | sampled_paths, generated_examples"
printf '  Input paths    : %s\n' "${PATHS_FILES[*]}"
printf '  Graph DB       : %s\n' "$GRAPH_DB"
printf '  Alpha values   : %s\n' "${GROWTH_ALPHA_VALUES[*]}"
printf '  Root           : %s\n' "$GROWTH_SETTINGS_DIR"
printf '  Cutoff date    : %s\n' "$CUTOFF_DATE"
printf '  Bulk index     : %s\n' "$INDEX_ROOT"
printf '  Spark          : local[%s], driver memory %s\n' "$CPUS" "$DRIVER_MEMORY"
printf '  Output         : %s\n\n' "$OUTPUT_DIR"

"$PYTHON_BIN" endpoints/calculate_paper_category_sets.py \
    --paths "${PATHS_FILES[@]}" \
    --search-root "$GROWTH_SETTINGS_DIR" \
    --alpha "${GROWTH_ALPHA_VALUES[@]}" \
    --vary "$GROWTH_VARY" \
    --graph-db "$GRAPH_DB" \
    --cutoff-date "$CUTOFF_DATE" \
    --index-root "$INDEX_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --spark-master "local[${CPUS}]" \
    --driver-memory "$DRIVER_MEMORY" \
    --local-dir "$SPARK_LOCAL_DIR"
