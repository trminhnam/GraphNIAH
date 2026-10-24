#!/bin/bash
# Infer converted multiple-choice examples with gemini-3.8-flash.
# First run create_data.3_question_format.none_option.sh to prepare selections.
# Model launchers can then run in parallel, each with its own response pool.

set -euo pipefail
export MODEL_SPEC="gemini-3.8-flash"
export INFERENCE_POOL="${INFERENCE_POOL:-${OUTPUT_ROOT:-./output/experiments.v1}/inference_pool.gemini-3.8-flash.jsonl}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/inference.3_question_format.none_option.sh"
