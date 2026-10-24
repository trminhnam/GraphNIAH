#!/bin/bash
# Shared settings and stage functions for the v1 experiment scripts
# (create_dataset.v1.py: v8 alpha-based tree growth).
# The other scripts source this file; run them from the repository root.
#
# Inputs: one paths file per question depth k (paths of 3k - 2 papers), each
# with its disjoint_path_groups.json from create_data.0_disjoint_path_groups.sh
# (the same files and caches as scripts/experiments/v2):
#
#   $DATA_DIR/hops=<10|22|46|94>/papers-only/sampled_paths.json
#   $DATA_DIR/hops=<10|22|46|94>/papers-only/disjoint_path_groups.json
#
# Output layout (one folder per experiment; per-example factors such as format,
# depth, node count, and distractor type live inside each record's config):
#
#   $OUTPUT_ROOT/pool.jsonl                     shared by every experiment
#   $OUTPUT_ROOT/inference_pool.jsonl           shared model responses
#   $OUTPUT_ROOT/cutoff_date=2026-04-01/category=cs/hops=10-22-46-94/context_length=200k/seed=1/alpha=0.2/vary=<factor>/
#       experiment.jsonl (+ .settings.json)    create_data.N_<factor>.sh
#       inference/reasoning_effort=<effort>/model=<slug>.json
#       eval/reasoning_effort=<effort>/model=<slug>.json
# The reasoning_effort experiment uses --vary none for data creation and
# evaluation, and the same dataset for low, medium, and high reasoning effort.
#
# Growth-parameter search (growth_search.N_*.sh), one pool per alpha:
#
#   $GROWTH_SEARCH_ROOT/cutoff_date=2026-04-01/category=cs/hops=10-22-46-94/context_length=200k/seed=1/
#       alpha=<alpha>/pool.jsonl
#       alpha=<alpha>/vary=question-format/experiment.jsonl
#       question_format=<format>/graph_statistics/   degree files for the search
#
# Every setting below can be overridden with an environment variable of the
# same name, e.g. CONTEXT_LENGTH=100000 RANDOM_SEED=2 bash <script>.

if [[ ! -f create_dataset.v1.py ]]; then
    printf 'Run this script from the repository root.\n' >&2
    exit 1
fi
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

################ Shared settings ####################
# Sampled paths and graph DB from scripts/path_sampling.new/2026-04-01
# (path_sampling_settings.sh); every paper is published on or after CUTOFF_DATE.
CUTOFF_DATE="${CUTOFF_DATE:-2026-04-01}"
# Fields of study of the path papers (PATH_CATEGORY in path_sampling_settings.sh):
# cs = Computer Science only; all = any field. A category other than all has
# its own .<category> data folder and graph DB, and a category=<category>
# level in the output folders, so its outputs never mix with the all-field ones.
PATH_CATEGORY="${PATH_CATEGORY:-all}"
if [[ "$PATH_CATEGORY" == all ]]; then
    PATH_CATEGORY_SUFFIX=""
    PATH_CATEGORY_LEVEL=""
else
    PATH_CATEGORY_SUFFIX=".${PATH_CATEGORY}"
    PATH_CATEGORY_LEVEL="/category=${PATH_CATEGORY}"
fi
DATA_DIR="${DATA_DIR:-./output/data.${CUTOFF_DATE}${PATH_CATEGORY_SUFFIX}}"
GRAPH_DB="${GRAPH_DB:-./resources/semantic_scholar_graph.${CUTOFF_DATE}${PATH_CATEGORY_SUFFIX}/semantic_scholar_paths.db}"
# Input path lengths in papers, one file each: 3k - 2 for question depths
# k = 4, 8, 16, 32 (the same HOP_COUNTS as path_sampling_settings.sh).
HOP_COUNTS_SPEC="${HOP_COUNTS_SPEC:-10 22 46 94}"
# Store every disjoint group of paths when there are at most this many;
# above it, keep only the conflicting pairs and draw-and-retry.
GROUP_LIMIT="${GROUP_LIMIT:-10000000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-200000}"
RANDOM_SEED="${RANDOM_SEED:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output/experiments.v1}"
POOL="${POOL:-$OUTPUT_ROOT/pool.jsonl}"
INFERENCE_POOL="${INFERENCE_POOL:-$OUTPUT_ROOT/inference_pool.jsonl}"
# Growth parameter of every component; set it to the search's winner.
ALPHA="${ALPHA:-0.1}"
# Optional distractor locations; create_dataset.v1.py has defaults for each.
CODE_DIR="${CODE_DIR:-}"
ESSAYS_DIR="${ESSAYS_DIR:-}"
BOOKS_DIR="${BOOKS_DIR:-}"
WIKIPEDIA_INDEX="${WIKIPEDIA_INDEX:-}"

# Experiments that inference.sh and eval.sh visit (VARY_SPEC="a,b" overrides).
# context_length is run separately by the stage-6 scripts at 500k and 922k;
# its 200k plotting point comes from reasoning_effort at the matching effort.
VARY_VALUES=(
    question-depth
    node-count
    question-format
    distractor-type
    reasoning_effort
)
# The other four experiments use this single reasoning effort.
REASONING_EFFORT="${REASONING_EFFORT:-low}"
# Growth-parameter search: the alpha grid and examples per alpha. ALPHA_SPEC
# ("a,b") overrides the grid, e.g. to run one alpha per job. Each alpha has
# its own pool, so they can run at the same time.
GROWTH_SEARCH_ROOT="${GROWTH_SEARCH_ROOT:-./output/experiments.v1/growth_search}"
GROWTH_ALPHA_VALUES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
GROWTH_VARY="question-format"
# Per question format, so 333 gives 999 examples per alpha.
GROWTH_EXAMPLES_PER_VALUE="${GROWTH_EXAMPLES_PER_VALUE:-500}"
# Real citation-graph degree distributions to compare with, made for the same
# cutoff by CUTOFF_DATE=<date> bash scripts/db_utils/calculate_bulk_paper_degree_distribution.sh
GLOBAL_DISTRIBUTION_DIR="${GLOBAL_DISTRIBUTION_DIR:-./output/graph_statistics/bulk_paper_degree_distribution}"
GLOBAL_DISTRIBUTIONS=("$GLOBAL_DISTRIBUTION_DIR/all.${CUTOFF_DATE}.strict-reference-count-0.json")

# Models for inference.sh and eval.sh (MODEL_SPEC="a,b" overrides).
MODEL_SLUGS=(
    # "gpt-5.6-sol"
    # "gpt-5.6-terra"
    "gpt-5.6-luna"
    # "qwen3.5-397b-a17b"
    "qwen3.5-35b-a3b"
    # "qwen3.5-122b-a10b"
    # "deepseek-v4-pro"
    "gemini-3.8-flash"
    "gemini-3.6-flash"
    # "gemma-4-31b"
)
#####################################################

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED=1
# These runs only read the graph DB. On Lustre (/scratch), SQLite locking
# costs network round trips, so skip it and the shared prepared cache.
# Never write the graph DB while they run.
export GRAPH_DB_IMMUTABLE="${GRAPH_DB_IMMUTABLE:-1}"
export DISABLE_PREPARED_GRAPH_CACHE="${DISABLE_PREPARED_GRAPH_CACHE:-1}"

if [[ -n "${VARY_SPEC:-}" ]]; then
    IFS=',' read -r -a VARY_VALUES <<< "$VARY_SPEC"
fi
if [[ -n "${MODEL_SPEC:-}" ]]; then
    IFS=',' read -r -a MODEL_SLUGS <<< "$MODEL_SPEC"
fi
if [[ -n "${ALPHA_SPEC:-}" ]]; then
    IFS=',' read -r -a GROWTH_ALPHA_VALUES <<< "$ALPHA_SPEC"
fi
read -r -a HOP_COUNTS <<< "$HOP_COUNTS_SPEC"
HOPS_SLUG="$(IFS='-'; printf '%s' "${HOP_COUNTS[*]}")"
PATHS_FILES=()
for HOP_COUNT in "${HOP_COUNTS[@]}"; do
    PATHS_FILES+=("$DATA_DIR/hops=${HOP_COUNT}/papers-only/sampled_paths.json")
done
if (( CONTEXT_LENGTH % 1000 == 0 )); then
    CONTEXT_LENGTH_LABEL="$((CONTEXT_LENGTH / 1000))k"
else
    CONTEXT_LENGTH_LABEL="$CONTEXT_LENGTH"
fi
SETTINGS_PATH="cutoff_date=${CUTOFF_DATE}${PATH_CATEGORY_LEVEL}/hops=${HOPS_SLUG}/context_length=${CONTEXT_LENGTH_LABEL}/seed=${RANDOM_SEED}"
SETTINGS_DIR="$OUTPUT_ROOT/$SETTINGS_PATH"
GROWTH_SETTINGS_DIR="$GROWTH_SEARCH_ROOT/$SETTINGS_PATH"

# Worker processes for create_dataset.v1.py. Each opens its own database
# connection and tokenizers (about 0.6 GB in a short local run; the default
# leaves room for caches). Under SLURM, fit the job's memory and CPUs.
MEMORY_PER_WORKER_MB="${MEMORY_PER_WORKER_MB:-4096}"
if [[ -z "${NUM_WORKERS:-}" ]]; then
    if [[ -n "${SLURM_MEM_PER_NODE:-}" ]]; then
        NUM_WORKERS=$(( SLURM_MEM_PER_NODE / MEMORY_PER_WORKER_MB ))
        # A shell attached with `srun --jobid=... --overlap --pty bash` has no
        # SLURM_CPUS_PER_TASK, so fall back to the job's CPUs on this node.
        CPU_LIMIT="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-}}"
        if [[ -n "$CPU_LIMIT" ]] && (( CPU_LIMIT < NUM_WORKERS )); then
            NUM_WORKERS="$CPU_LIMIT"
        fi
        (( NUM_WORKERS >= 1 )) || NUM_WORKERS=1
    else
        NUM_WORKERS=4
    fi
fi

RULE="$(printf '=%.0s' $(seq 1 90))"

# The folder of the current growth setting (ALPHA), and of one experiment in it.
growth_dir() {
    printf '%s/alpha=%s' "$SETTINGS_DIR" "$ALPHA"
}

experiment_dir() {
    local experiment="$1"
    if [[ "$experiment" == none ]]; then experiment=reasoning_effort; fi
    printf '%s/vary=%s' "$(growth_dir)" "$experiment"
}

# reasoning_efforts <experiment>: space-separated values for inference/eval.
reasoning_efforts() {
    if [[ "$1" == reasoning_effort || "$1" == none ]]; then
        printf 'low medium high'
    else
        printf '%s' "$REASONING_EFFORT"
    fi
}

format_duration() {
    local total="$1"
    printf '%dh%02dm%02ds' $((total / 3600)) $((total % 3600 / 60)) $((total % 60))
}

print_header() {
    printf '\n%s\n %s | %s\n%s\n' "$RULE" "$1" "$(date '+%Y-%m-%d %H:%M:%S')" "$RULE"
}

# model_context_limit <model slug>: the model's context window in tokens.
model_context_limit() {
    "$PYTHON_BIN" -c \
        'from src.llm.model_info import MODEL_MAX_LENGTHS; import sys; print(MODEL_MAX_LENGTHS[sys.argv[1]])' \
        "$1"
}

# check_inputs: stop unless every paths file, its disjoint path groups, and the
# graph DB exist.
check_inputs() {
    local path
    for path in "${PATHS_FILES[@]}" "$GRAPH_DB"; do
        if [[ ! -f "$path" ]]; then
            printf 'Missing input: %s\n' "$path" >&2
            exit 1
        fi
    done
    for path in "${PATHS_FILES[@]}"; do
        if [[ ! -f "$(dirname "$path")/disjoint_path_groups.json" ]]; then
            printf 'Missing disjoint path groups next to %s\n' "$path" >&2
            printf 'Build them with: bash scripts/experiments/v1/create_data.0_disjoint_path_groups.sh\n' >&2
            exit 1
        fi
    done
}

# create_data <experiment> <examples per value (total for none)> <max attempts> [extra arguments...]
# Builds (or resumes, or extends) one experiment with the current ALPHA. Only
# one create_data script can write to a pool at a time, so run them one after
# another, or give each its own POOL.
create_data() {
    local vary="$1" examples_per_value="$2" max_attempts="$3"
    local extra_args=("${@:4}")
    local output started=$SECONDS
    output="$(experiment_dir "$vary")/experiment.jsonl"
    if [[ "$vary" == context_length ]]; then vary=none; fi
    check_inputs
    local distractor_args=()
    if [[ -n "$CODE_DIR" ]]; then distractor_args+=(--code-dir "$CODE_DIR"); fi
    if [[ -n "$ESSAYS_DIR" ]]; then distractor_args+=(--essays-dir "$ESSAYS_DIR"); fi
    if [[ -n "$BOOKS_DIR" ]]; then distractor_args+=(--books-dir "$BOOKS_DIR"); fi
    if [[ -n "$WIKIPEDIA_INDEX" ]]; then
        distractor_args+=(--wikipedia-index "$WIKIPEDIA_INDEX")
    fi

    print_header "CREATE DATA | vary=$vary"
    if [[ "$vary" == none ]]; then
        printf '  Total examples : %s\n' "$examples_per_value"
    else
        printf '  Examples/value : %s\n' "$examples_per_value"
    fi
    printf '  Input paths    : %s\n' "${PATHS_FILES[*]}"
    printf '  Graph DB       : %s\n' "$GRAPH_DB"
    printf '  Context length : %s tokens\n' "$CONTEXT_LENGTH"
    printf '  Seed           : %s\n' "$RANDOM_SEED"
    printf '  Cutoff date    : %s\n' "$CUTOFF_DATE"
    printf '  Growth         : alpha=%s\n' "$ALPHA"
    printf '  Extra options  : %s\n' "${extra_args[*]:-none}"
    printf '  Workers        : %s\n' "$NUM_WORKERS"
    printf '  Max attempts   : %s per slot\n' "$max_attempts"
    printf '  Pool           : %s\n' "$POOL"
    printf '  Experiment     : %s\n\n' "$output"
    mkdir -p "$(dirname "$output")" "$(dirname "$POOL")"
    # Load tokenizers from the local Hugging Face cache only: no update check
    # from every worker. To download a missing one, run with HF_HUB_OFFLINE=0.
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" "$PYTHON_BIN" create_dataset.v1.py \
        --paths "${PATHS_FILES[@]}" \
        --graph-db "$GRAPH_DB" \
        --vary "$vary" \
        --examples-per-value "$examples_per_value" \
        --pool "$POOL" \
        --output "$output" \
        --context-length "$CONTEXT_LENGTH" \
        --random-seed "$RANDOM_SEED" \
        --cutoff-date "$CUTOFF_DATE" \
        --alpha "$ALPHA" \
        --max-attempts "$max_attempts" \
        --workers "$NUM_WORKERS" \
        ${distractor_args[@]+"${distractor_args[@]}"} \
        ${extra_args[@]+"${extra_args[@]}"}
    printf '\nCREATE DATA | vary=%s finished in %s\n' \
        "$vary" "$(format_duration $((SECONDS - started)))"
}
