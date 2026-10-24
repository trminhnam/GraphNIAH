#!/bin/bash
# Merge model-specific pools, then copy the result back after all jobs finish.
# Create missing pools for all seven models so future runs can reuse the cache.
# OUTPUT_ROOT and INFERENCE_POOL override the defaults in config.sh.
# Activate the niah environment before running this script.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

shopt -s nullglob
SOURCE_POOLS=()
# Include future model pools too, even if they are not listed below.
for SOURCE_POOL in "$OUTPUT_ROOT"/inference_pool.*.jsonl; do
    if [[ -f "$SOURCE_POOL" && ! "$SOURCE_POOL" -ef "$INFERENCE_POOL" ]]; then
        SOURCE_POOLS+=("$SOURCE_POOL")
    fi
done

for MODEL in \
    gpt-5.6-luna gpt-5.6-sol gemini-3.8-flash gemini-3.6-flash \
    qwen3.5-35b-a3b qwen3.5-122b-a10b qwen3.5-397b-a17b; do
    SOURCE_POOL="$OUTPUT_ROOT/inference_pool.${MODEL}.jsonl"
    if [[ ! -e "$SOURCE_POOL" && "$SOURCE_POOL" != "$INFERENCE_POOL" ]]; then
        SOURCE_POOLS+=("$SOURCE_POOL")
    fi
done

"$PYTHON_BIN" merge_inference_pools.py \
    --sources "${SOURCE_POOLS[@]}" \
    --target "$INFERENCE_POOL" \
    --copy-back
