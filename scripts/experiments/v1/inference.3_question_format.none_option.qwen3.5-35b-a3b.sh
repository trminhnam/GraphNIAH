#!/bin/bash
# Infer converted multiple-choice examples with qwen3.5-35b-a3b.
# First run create_data.3_question_format.none_option.sh to prepare selections.
# Model launchers can then run in parallel, each with its own response pool.

set -euo pipefail
export MODEL_SPEC="qwen3.5-35b-a3b"
export INFERENCE_POOL="${INFERENCE_POOL:-${OUTPUT_ROOT:-./output/experiments.v1}/inference_pool.qwen3.5-35b-a3b.jsonl}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/inference.3_question_format.none_option.sh"
