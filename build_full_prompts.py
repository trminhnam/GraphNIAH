#!/usr/bin/env python3
"""
build_full_prompts.py - Assemble final user messages from generated questions.

This script consumes the JSON output from ``create_needles.py`` or
``create_distracting_content.py`` and rewrites each question entry into:

- ``qid``
- ``user_message``
- ``answer``
- ``answer_type`` (forwarded for downstream evaluation)

By default the written ``user_message`` keeps the existing prompt layout:

    # Question

    <question>

    # Context
    ###
    <context item 1>

    <context item 2>

With ``--question-last``, the context comes first and the question is appended
at the end under ``# Question``.

The default system prompt and format reminder are imported from ``src/prompts``.
With ``--target-context-length``, the script packs distractors against every
Hugging Face tokenizer listed in ``resources/llm_models.yaml`` whose
``hf_model_id`` is defined and whose declared native context length covers the
requested target, plus local tiktoken-based OpenAI counters for OpenAI models
whose declared native context length covers the target. The written prompt is
a single shared prompt that fits the relevant benchmark tokenizer set.

Arguments
---------

--input
  Required path to the JSON produced by ``create_needles.py`` or
  ``create_distracting_content.py``.

--output
  Required path for the rewritten full-prompt JSON.

--question-last
  Optional flag to place the question after the context instead of before it.

--target-context-length
  Optional shared token budget for the final chat prompt. When set, the script
  counts tokens using the shared ``system + user`` chat prompt and trims
  distractors until the prompt fits every loaded benchmark tokenizer.

--traversal-mode
  Reachability graph mode. ``mixed`` and ``alternating`` include paper and
  author nodes; ``papers-only`` includes only paper nodes.

--system-prompt-path
  Optional path to override the default system prompt used during token
  counting.

--format-reminder-path
  Optional path to override the default format reminder appended to every
  built ``user_message``.

--needle-order
  Controls the relative order of needles after shuffling with distractors.
  Supported values: ``random``, ``forward``, ``backward``.

--random-seed
  Optional random seed for deterministic context shuffling.

--context-item-prefix
  Prefix inserted before the first context item. Default: ``###\\n\\n``.

--context-item-separator
  Separator inserted between consecutive context items. Default:
  ``\\n\\n``.

--context-tail-margin
  Headroom reserved below ``--target-context-length`` during prompt packing.

--partial-negative-tree-packing / --no-partial-negative-tree-packing
  Allow a prefix of complete rendered blocks from the next negative tree to
  use the remaining token budget. Enabled by default. The ``--no-`` form keeps
  every negative tree atomic.

--distracting-source-dir
  Optional override directory for resolving ``wiki_bm25:`` and ``wiki_dense:``
  distractor references.

--indent
  JSON indentation width for the written output file.

Usage
-----

Basic rewrite:

python build_full_prompts.py \
    --input output/citation_graph_20/questions/questions.w_needles.json \
    --output output/citation_graph_20/questions/questions.w_full_prompts.json

Shared-tokenizer packing:

python build_full_prompts.py \
    --input output/citation_graph_20/questions/questions.with_distractors.json \
    --output output/citation_graph_20/questions/questions.w_full_prompts.json \
    --target-context-length 100000 \
    --system-prompt-path prompts/system.txt \
    --format-reminder-path prompts/format_reminder.txt \
    --needle-order forward \
    --random-seed 7
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import orjson
from queue import Empty, Queue
import random
import re
import statistics
import threading
import uuid
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from tqdm.auto import tqdm

from create_needles import build_metadata_needle
from src.graph.readers import DBReader
from src.logging.console import format_display_path, log_line
from src.graph.resolution import resolve_db_path
from src.prompts import (
    DEFAULT_REACHABILITY_TRAVERSAL_MODE,
    FORMAT_REMINDER,
    REACHABILITY_COUNTING_FORMAT_REMINDER,
    REACHABILITY_FORMAT_REMINDER,
    REACHABILITY_MULTIPLE_FORMAT_REMINDER,
    REACHABILITY_TRAVERSAL_MODES,
    SYSTEM_PROMPT,
    build_reachability_counting_system_prompt,
    build_reachability_multiple_system_prompt,
    build_reachability_system_prompt,
)
from src.questions.rerender import rerender_questions_enabled
from src.retrieval import BM25Retriever, DenseRetriever
from src.setup.pipeline_parameters import (
    DISTRACTING_CONTENT_TYPE_PARAMETER_FIELD,
    NEEDLE_TYPE_PARAMETER_FIELD,
    NEEDLE_ORDER_PARAMETER_FIELD as PIPELINE_NEEDLE_ORDER_PARAMETER_FIELD,
    PARAMETERS_FIELD,
    QUESTION_TYPE_PARAMETER_FIELD,
    TARGET_CONTEXT_LENGTH_PARAMETER_FIELD as PIPELINE_TARGET_CONTEXT_LENGTH_PARAMETER_FIELD,
    TRAVERSAL_MODE_PARAMETER_FIELD,
    get_parameters_from_payload,
    merge_parameters,
)
from src.setup.constants import (
    DB_PATH_FIELD,
    DEFAULT_WIKIPEDIA_DISTRACTING_DIR,
    DISTRACTING_CONTENT_FIELD,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V4,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V5,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V6,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V7,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V8,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V10,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V11,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V9,
    DISTRACTING_CONTENT_TYPE_PAPER_RELATED,
    DISTRACTING_CONTENT_TYPE_REPEAT_NEEDLES,
    GENERATED_AT_FIELD,
    GRAPH_JSON_FIELD,
    METADATA_FIELD,
    NODE_PAPER,
    PAPER_INDEX_ITEM_PREFIX,
    PAPER_NEGATIVE_TREE_ITEM_PREFIX,
    PAPER_RELATED_INDEX_LEVEL_ITEM_PREFIX,
    PAPER_RELATED_INDEX_ITEM_PREFIX,
    PATH_ITEM_PREFIX,
    QUESTIONS_FIELD,
    REPEAT_ITEM_PREFIX,
    SOURCE_QUESTIONS_FIELD,
    TOTAL_QUESTIONS_FIELD,
    WIKI_BM25_ITEM_PREFIX,
    WIKI_DENSE_ITEM_PREFIX,
    nid_db_id,
    paper_nid,
)
from src.utils.preprocessing import remove_references_section
from src.utils.resume import format_resume_mismatches

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - exercised only when deps are missing.
    AutoTokenizer = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only when deps are missing.
    OpenAI = None

try:
    import tiktoken
except ImportError:  # pragma: no cover - exercised only when deps are missing.
    tiktoken = None


CLI_NAME = "build_full_prompts"
FULL_PROMPTS_FIELD = "full_prompts"
QUESTION_FIELD = "question"
ANSWER_FIELD = "answer"
QUESTION_ID_FIELD = "qid"
NEEDLES_FIELD = "needles"
USER_MESSAGE_FIELD = "user_message"
DISTRACTOR_ITEMS_FIELD = "distractor_items"
DISTRACTOR_TOKENS_FIELD = "distractor_tokens"
NEEDLE_ITEMS_FIELD = "needle_items"
NEEDLE_TOKENS_FIELD = "needle_tokens"
PROMPT_TOKENS_FIELD = "prompt_tokens"
PROMPT_WITHOUT_DISTRACTORS_TOKENS_FIELD = "prompt_without_distractors_tokens"
CONTEXT_STATS_FIELDS = (
    NEEDLE_ITEMS_FIELD,
    NEEDLE_TOKENS_FIELD,
    DISTRACTOR_ITEMS_FIELD,
    DISTRACTOR_TOKENS_FIELD,
    PROMPT_WITHOUT_DISTRACTORS_TOKENS_FIELD,
)
SYSTEM_MESSAGE_FIELD = "system_prompt"
NO_SYSTEM_MESSAGE_FIELD = "no_system_message"
DISTRACTING_CONTENT_ITEMS_FIELD = "distracting_content_items"
QUESTION_LAST_FIELD = "question_last"
SYNTHESIZED_QIDS_FIELD = "synthesized_qids"
SOURCE_FINGERPRINT_FIELD = "source_fingerprint"
SKIPPED_QIDS_FIELD = "skipped_qids"
TARGET_CONTEXT_LENGTH_FIELD = "target_context_length"
TOKENIZER_MODEL_SLUGS_FIELD = "tokenizer_model_slugs"
NEEDLE_ORDER_FIELD = "needle_order"
RANDOM_SEED_FIELD = "random_seed"
CONTEXT_ITEM_PREFIX_FIELD = "context_item_prefix"
CONTEXT_ITEM_SEPARATOR_FIELD = "context_item_separator"
CONTEXT_TAIL_MARGIN_FIELD = "context_tail_margin"
PARTIAL_NEGATIVE_TREE_PACKING_FIELD = "partial_negative_tree_packing"
MAX_SAMPLES_FIELD = "max_samples"
MUTABLE_CHECKPOINT_RUN_SETTING_FIELDS = frozenset(
    {MAX_SAMPLES_FIELD, TOKENIZER_MODEL_SLUGS_FIELD}
)
QUESTION_TYPE_FIELD = "question_type"
TRAVERSAL_MODE_FIELD = "traversal_mode"

QUESTION_TYPE_FACTOID = "factoid"
QUESTION_TYPE_COUNTING = "counting"
QUESTION_TYPE_REACHABILITY_YESNO = "reachability-yesno"
QUESTION_TYPE_REACHABILITY_MULTIPLE = "reachability-multiple"
QUESTION_TYPE_REACHABILITY_COUNTING = "reachability-counting"
REACHABILITY_QUESTION_TYPES = {
    QUESTION_TYPE_REACHABILITY_COUNTING,
    QUESTION_TYPE_REACHABILITY_MULTIPLE,
    QUESTION_TYPE_REACHABILITY_YESNO,
}
REACHABILITY_ANSWER_TYPES = {
    "REACHABILITY_COUNTING",
    "REACHABILITY_MULTIPLE",
    "REACHABILITY_YESNO",
}
SUPPORTED_QUESTION_TYPES = {
    QUESTION_TYPE_COUNTING,
    QUESTION_TYPE_FACTOID,
    QUESTION_TYPE_REACHABILITY_COUNTING,
    QUESTION_TYPE_REACHABILITY_MULTIPLE,
    QUESTION_TYPE_REACHABILITY_YESNO,
}
DEFAULT_QUESTION_TYPE = QUESTION_TYPE_FACTOID
DEFAULT_NUM_WORKERS = 1
PAPER_DISTRACTOR_SQLITE_CACHE_SIZE_KIB = 1048576
PAPER_DISTRACTOR_SQLITE_MMAP_SIZE_BYTES = 2147418112

DEFAULT_NEEDLE_ORDER = "forward"
DEFAULT_CONTEXT_ITEM_PREFIX = ""
DEFAULT_CONTEXT_ITEM_SEPARATOR = "\n\n###\n\n"
DEFAULT_CONTEXT_TAIL_MARGIN = 100
SUPPORTED_NEEDLE_ORDERS = {"random", "forward", "backward"}
WIKI_BM25_CACHE_SUBDIR = "bm25"
WIKI_DENSE_CACHE_SUBDIR = "dense"
OPENAI_API_MODEL_PREFIX = "openai/"
OPENAI_TIKTOKEN_FALLBACK_ENCODING = "o200k_base"
LONG_CONTEXT_HF_FALLBACK_THRESHOLD = 500000
PAPER_RELATED_INDEX_LEVEL_ITEM_RE = re.compile(
    rf"^{PAPER_RELATED_INDEX_LEVEL_ITEM_PREFIX}(?P<level>[1-9]\d*):(?P<value>.*)$"
)

INVALID_INPUT_DATA_ERROR = "Input file must be a JSON object with a questions list."
INVALID_QUESTION_ENTRY_ERROR = "Each question entry in the input file must be a JSON object."
INVALID_METADATA_ERROR = "Input metadata must be a JSON object."
INVALID_QUESTION_TEXT_ERROR = (
    "Question entry at index {index} must contain a non-empty 'question' string."
)
INVALID_ANSWER_ERROR = "Question entry at index {index} must contain a non-empty 'answer' value."
INVALID_NEEDLES_ERROR = "Question entry at index {index} must contain a 'needles' list of strings."
INVALID_DISTRACTING_CONTENT_ERROR = (
    "Question entry at index {index} must contain a "
    "'distracting_content_items' list of strings when present."
)
INVALID_NEEDLE_ORDER_ERROR = (
    "Unsupported needle order: {needle_order}. " "Expected one of: backward, forward, random."
)
REPEAT_NEEDLES_RANDOM_ORDER_REQUIRED_ERROR = "repeat-needles requires --needle-order=random."
REPEAT_NEEDLES_TARGET_CONTEXT_REQUIRED_ERROR = "repeat-needles requires --target-context-length."
INVALID_TARGET_CONTEXT_LENGTH_ERROR = (
    "target_context_length must be a positive integer when provided."
)
INVALID_CONTEXT_TAIL_MARGIN_ERROR = "context_tail_margin must be greater than or equal to 0."
INVALID_MAX_SAMPLES_ERROR = "max_samples must be greater than or equal to 0 when provided."
INVALID_NUM_WORKERS_ERROR = "num_workers must be greater than or equal to 1."
INVALID_QUESTION_TYPE_ERROR = (
    "Unsupported question type: {question_type}. Expected one of: "
    "counting, factoid, reachability-counting, reachability-multiple, "
    "reachability-yesno."
)
QUESTION_TYPE_MISMATCH_ERROR = (
    "Question entry at index {index} declares question_type={record_type!r}, "
    "but --question-type is {question_type!r}."
)
TARGET_CONTEXT_REQUIRES_SYSTEM_PROMPT_ERROR = (
    "system_prompt is required when target_context_length is set."
)
NO_SYSTEM_MESSAGE_REQUIRES_SYSTEM_PROMPT_ERROR = (
    "system_prompt is required when no_system_message is enabled."
)
TARGET_CONTEXT_REQUIRES_FORMAT_REMINDER_ERROR = (
    "format_reminder is required when target_context_length is set."
)
NO_TOKENIZERS_AVAILABLE_ERROR = "No prompt-packing tokenizer or counter could be loaded."
TOKENIZER_LOAD_WARNING = (
    "Skipping Hugging Face tokenizer for prompt packing: "
    "{slug} ({hf_model_id}) -> {error_type}: {error}"
)
TOKENIZER_REMAINING_MESSAGE = "Prompt-packing tokenizers: {slugs}"
TOKENIZER_DUPLICATES_MESSAGE = " ({count} duplicate(s) collapsed)"
TOKENIZER_LONG_CONTEXT_FALLBACK_WARNING = (
    "No Hugging Face prompt-packing tokenizer declares native_context_length >= "
    "{target_context_length}; using local Hugging Face tokenizers below the "
    "target as long-context packing proxies."
)
OPENAI_FAKE_TOKENIZER_LOAD_WARNING = (
    "Skipping fake OpenAI tokenizer for prompt packing: "
    "{slug} ({model_name}) -> {error_type}: {error}"
)
OPENAI_COUNTER_IMPORT_ERROR = "`openai` is required to load OpenAI input-token counters."
OPENAI_COUNTER_API_UNAVAILABLE_ERROR = (
    "Installed `openai` package does not support responses.input_tokens.count."
)
UNSUPPORTED_DISTRACTOR_ITEM_ERROR = (
    "Unsupported distracting content item at question index {index}: {item}"
)
WIKI_RETRIEVER_REQUIRED_ERROR = (
    "Question entry at index {index} needs a {retriever_name} to resolve {item!r}."
)
WIKI_PARAGRAPH_NOT_FOUND_ERROR = (
    "Question entry at index {index} references missing Wikipedia distractor {item!r}."
)
INVALID_PAPER_INDEX_DISTRACTOR_ERROR = (
    "Question entry at index {index} references invalid paper_index distractor {item!r}."
)
PAPER_INDEX_DISTRACTOR_DB_REQUIRED_ERROR = (
    "Question entry at index {index} needs a SQLite DB path in metadata to resolve {item!r}."
)
PAPER_INDEX_DISTRACTOR_NOT_FOUND_ERROR = (
    "Question entry at index {index} references missing paper_index distractor {item!r}."
)
PROMPT_SKIPPED_WARNING = (
    "SKIP index={index} qid={qid} reason=needles-exceed-target{details} "
    "target_context_length={target_context_length:,}"
)
PROMPT_SKIPPED_DETAIL_FIELDS = (
    (NEEDLE_ITEMS_FIELD, "needles"),
    (PROMPT_TOKENS_FIELD, "prompt_tokens"),
)
INVALID_ANSWER_SKIP_WARNING = "SKIP index={index} qid={qid} reason=invalid-answer: {error}"
INCOMPATIBLE_CHECKPOINT_ERROR = (
    "Existing checkpoint at {out_path} was built with different full-prompt "
    "settings. Remove the file or choose a different --output path."
)


def build_incompatible_checkpoint_error(
    out_path: Path,
    saved: Any,
    current: Any,
) -> ValueError:
    """Return a checkpoint error that includes exact saved/current differences."""
    mismatches = format_resume_mismatches(saved, current)
    return ValueError(
        f"{INCOMPATIBLE_CHECKPOINT_ERROR.format(out_path=out_path)}\n"
        f"Mismatches:\n{mismatches}"
    )


def validate_questions_data(input_data: Any) -> list[dict[str, Any]]:
    """Return the validated question list from one pipeline JSON payload."""
    if not isinstance(input_data, dict):
        raise ValueError(INVALID_INPUT_DATA_ERROR)

    questions = input_data.get(QUESTIONS_FIELD)
    if not isinstance(questions, list):
        raise ValueError(INVALID_INPUT_DATA_ERROR)

    for question in questions:
        if not isinstance(question, dict):
            raise ValueError(INVALID_QUESTION_ENTRY_ERROR)

    return questions


def validate_question_text(question: dict[str, Any], index: int) -> str:
    """Return one validated question string."""
    question_text = question.get(QUESTION_FIELD)
    if not isinstance(question_text, str) or not question_text.strip():
        raise ValueError(INVALID_QUESTION_TEXT_ERROR.format(index=index))
    return question_text


def validate_answer_value(
    question: dict[str, Any],
    index: int,
) -> str | list[str]:
    """Return one validated scalar answer or list of alternate answers."""
    answer = question.get(ANSWER_FIELD)
    if answer is None:
        raise ValueError(INVALID_ANSWER_ERROR.format(index=index))

    if isinstance(answer, list):
        answer_values = [
            str(value).strip() for value in answer if value is not None and str(value).strip()
        ]
        if not answer_values:
            raise ValueError(INVALID_ANSWER_ERROR.format(index=index))
        return answer_values

    answer_text = str(answer).strip()
    if not answer_text:
        raise ValueError(INVALID_ANSWER_ERROR.format(index=index))
    return answer_text


def validate_needles(question: dict[str, Any], index: int) -> list[str]:
    """Return the validated needles list."""
    needles = question.get(NEEDLES_FIELD)
    if not isinstance(needles, list):
        raise ValueError(INVALID_NEEDLES_ERROR.format(index=index))
    if any(not isinstance(needle, str) for needle in needles):
        raise ValueError(INVALID_NEEDLES_ERROR.format(index=index))
    return needles


def validate_distracting_content(
    question: dict[str, Any],
    index: int,
) -> list[str]:
    """Return the validated distractor-item list."""
    distracting_content_items = question.get(DISTRACTING_CONTENT_ITEMS_FIELD)
    if distracting_content_items is None:
        return []
    if not isinstance(distracting_content_items, list):
        raise ValueError(INVALID_DISTRACTING_CONTENT_ERROR.format(index=index))
    if any(not isinstance(item, str) for item in distracting_content_items):
        raise ValueError(INVALID_DISTRACTING_CONTENT_ERROR.format(index=index))
    return distracting_content_items


def validate_max_samples(max_samples: int | None) -> int | None:
    """Return one validated max-samples value."""
    if max_samples is None:
        return None
    if max_samples < 0:
        raise ValueError(INVALID_MAX_SAMPLES_ERROR)
    return max_samples


def validate_num_workers(num_workers: int) -> int:
    """Return one valid prompt-worker count."""
    if num_workers < 1:
        raise ValueError(INVALID_NUM_WORKERS_ERROR)
    return num_workers


def validate_question_type(question_type: str) -> str:
    """Return one supported question-type slug."""
    normalized = str(question_type or "").strip()
    if normalized not in SUPPORTED_QUESTION_TYPES:
        raise ValueError(INVALID_QUESTION_TYPE_ERROR.format(question_type=question_type))
    return normalized


def resolve_question_type_prompts(
    *,
    question_type: str,
    needle_type: str | None,
    traversal_mode: str,
    system_prompt: str | None,
    format_reminder: str | None,
) -> tuple[str | None, str | None]:
    """Resolve default prompt text while preserving explicit overrides."""
    normalized = validate_question_type(question_type)
    if normalized not in REACHABILITY_QUESTION_TYPES:
        return system_prompt, format_reminder

    normalized_traversal_mode = str(traversal_mode or "").strip()
    if normalized_traversal_mode not in REACHABILITY_TRAVERSAL_MODES:
        supported = ", ".join(sorted(REACHABILITY_TRAVERSAL_MODES))
        raise ValueError(
            f"Unsupported reachability traversal mode: {traversal_mode!r}. "
            f"Expected one of: {supported}."
        )
    if system_prompt is SYSTEM_PROMPT:
        prompt_builders = {
            QUESTION_TYPE_REACHABILITY_COUNTING: (build_reachability_counting_system_prompt),
            QUESTION_TYPE_REACHABILITY_MULTIPLE: (build_reachability_multiple_system_prompt),
            QUESTION_TYPE_REACHABILITY_YESNO: build_reachability_system_prompt,
        }
        prompt_builder = prompt_builders[normalized]
        system_prompt = prompt_builder(str(needle_type or ""), normalized_traversal_mode)
    if format_reminder is FORMAT_REMINDER:
        format_reminders = {
            QUESTION_TYPE_REACHABILITY_COUNTING: (REACHABILITY_COUNTING_FORMAT_REMINDER),
            QUESTION_TYPE_REACHABILITY_MULTIPLE: (REACHABILITY_MULTIPLE_FORMAT_REMINDER),
            QUESTION_TYPE_REACHABILITY_YESNO: REACHABILITY_FORMAT_REMINDER,
        }
        format_reminder = format_reminders[normalized]
    return system_prompt, format_reminder


def build_context_text(
    context_items: list[str],
    *,
    context_item_prefix: str,
    context_item_separator: str,
) -> str:
    """Render the context block from one ordered list of context items."""
    if not context_items:
        return "# Context"

    return "# Context\n\n" f"{context_item_prefix}" f"{context_item_separator.join(context_items)}"


def build_user_message(
    question_text: str,
    context_text: str,
    question_last: bool,
    *,
    instruction_prompt: str | None = None,
) -> str:
    """Return the final user-role message."""
    if question_last:
        user_message = f"{context_text}\n\n# Question\n\n{question_text}"
    else:
        user_message = f"# Question\n\n{question_text}\n\n{context_text}"
    if instruction_prompt is None:
        return user_message
    return f"# Instruction\n{instruction_prompt.strip()}\n\n{user_message}"


def append_format_reminder(
    user_message: str,
    format_reminder: str | None,
) -> str:
    """Append the shared format reminder once to one user message."""
    trimmed_message = user_message.rstrip()
    if not format_reminder:
        return trimmed_message
    if trimmed_message.endswith(format_reminder):
        return trimmed_message
    return f"{trimmed_message}\n\n{format_reminder}"


def read_optional_text_file(path: str | Path | None) -> str | None:
    """Return file contents for one optional text path."""
    if path is None:
        return None
    return Path(path).read_text(encoding="utf-8")


def resolve_wiki_cache_dir(
    distracting_source_dir: str | Path | None,
    cache_subdir: str,
) -> Path:
    """Return the Wikipedia cache dir for one retriever family."""
    if distracting_source_dir is None:
        source_dir = Path(DEFAULT_WIKIPEDIA_DISTRACTING_DIR)
    else:
        source_dir = Path(distracting_source_dir)

    if source_dir.name == cache_subdir:
        return source_dir

    if source_dir.name in {WIKI_BM25_CACHE_SUBDIR, WIKI_DENSE_CACHE_SUBDIR}:
        return source_dir.parent / cache_subdir

    return source_dir / cache_subdir


def is_paper_distracting_content_type(distracting_content_type: Any) -> bool:
    """Return whether one distracting-content type uses paper text files."""
    return isinstance(distracting_content_type, str) and distracting_content_type.startswith(
        "paper"
    )


def filter_paper_distractor_headers(raw_text: str) -> str:
    """Remove markdown-style header lines from one paper distractor text."""
    return "\n".join(line for line in raw_text.split("\n") if not line.lstrip().startswith("#"))


def split_distracting_content_paragraphs(raw_text: str) -> tuple[str, ...]:
    """Return one text blob split into non-empty paragraphs."""
    return tuple(paragraph.strip() for paragraph in raw_text.split("\n\n") if paragraph.strip())


def parse_paper_related_index_item(item: str, index: int) -> tuple[str, int] | None:
    """Return ``(raw paper id, level)`` for a paper-related distractor item."""
    if item.startswith(PAPER_RELATED_INDEX_ITEM_PREFIX):
        raw_value = item[len(PAPER_RELATED_INDEX_ITEM_PREFIX) :].strip()
        level = 1
    else:
        match = PAPER_RELATED_INDEX_LEVEL_ITEM_RE.match(item)
        if match is None:
            return None
        raw_value = match.group("value").strip()
        level = int(match.group("level"))

    if not raw_value:
        raise ValueError(INVALID_PAPER_INDEX_DISTRACTOR_ERROR.format(index=index, item=item))
    return raw_value, level


def is_paper_related_index_item(item: str) -> bool:
    """Return whether one item is a paper-related distractor reference."""
    return (
        item.startswith(PAPER_RELATED_INDEX_ITEM_PREFIX)
        or PAPER_RELATED_INDEX_LEVEL_ITEM_RE.match(item) is not None
    )


def is_paper_negative_tree_item(item: str) -> bool:
    """Return whether one item references a complete negative tree."""
    return item.startswith(PAPER_NEGATIVE_TREE_ITEM_PREFIX)


def parse_paper_negative_tree_item(item: str, index: int) -> int:
    """Return the stored negative-tree index from one serialized item."""
    raw_value = item[len(PAPER_NEGATIVE_TREE_ITEM_PREFIX) :].strip()
    try:
        tree_index = int(raw_value)
    except ValueError as error:
        raise ValueError(
            INVALID_PAPER_INDEX_DISTRACTOR_ERROR.format(index=index, item=item)
        ) from error
    if tree_index < 0 or str(tree_index) != raw_value:
        raise ValueError(INVALID_PAPER_INDEX_DISTRACTOR_ERROR.format(index=index, item=item))
    return tree_index


def paper_related_index_item_level(item: str, index: int) -> int:
    """Return the descriptor-sampling level for one paper-related item."""
    parsed_item = parse_paper_related_index_item(item, index)
    if parsed_item is None:
        raise ValueError(INVALID_PAPER_INDEX_DISTRACTOR_ERROR.format(index=index, item=item))
    return parsed_item[1]


def load_path_distracting_content_paragraphs(
    path: Path,
    *,
    distracting_content_type: str | None,
    path_paragraph_cache: dict[tuple[str, bool], tuple[str, ...]] | None,
) -> tuple[str, ...]:
    """Return cleaned/split paragraphs for one path-based distractor item."""
    use_paper_cleaning = is_paper_distracting_content_type(distracting_content_type)
    cache_key = (str(path), use_paper_cleaning)
    if path_paragraph_cache is not None:
        cached_paragraphs = path_paragraph_cache.get(cache_key)
        if cached_paragraphs is not None:
            return cached_paragraphs

    raw_text = path.read_text(encoding="utf-8")
    if use_paper_cleaning:
        raw_text = remove_references_section(raw_text)
        raw_text = filter_paper_distractor_headers(raw_text)

    paragraphs = split_distracting_content_paragraphs(raw_text)
    if path_paragraph_cache is not None:
        path_paragraph_cache[cache_key] = paragraphs
    return paragraphs


def parse_paper_index_item(item: str, index: int) -> str:
    """Return a graph paper node id from one paper_index distractor item."""
    if item.startswith(PAPER_INDEX_ITEM_PREFIX):
        raw_value = item[len(PAPER_INDEX_ITEM_PREFIX) :].strip()
    else:
        parsed_related_item = parse_paper_related_index_item(item, index)
        raw_value = parsed_related_item[0] if parsed_related_item is not None else ""
    if raw_value.startswith(f"{NODE_PAPER}:"):
        try:
            nid_db_id(raw_value)
        except (IndexError, ValueError) as exc:
            raise ValueError(
                INVALID_PAPER_INDEX_DISTRACTOR_ERROR.format(index=index, item=item)
            ) from exc
        return raw_value

    if not raw_value:
        raise ValueError(INVALID_PAPER_INDEX_DISTRACTOR_ERROR.format(index=index, item=item))
    return paper_nid(raw_value)


def load_paper_index_distracting_content_paragraphs(
    item: str,
    index: int,
    *,
    question: dict[str, Any] | None = None,
    paper_distractor_db: DBReader | None,
    paper_distractor_cache: dict[str, tuple[str, ...]] | None,
) -> tuple[str, ...]:
    """Return descriptive metadata text for one paper_index distractor item."""
    del question
    if paper_distractor_cache is not None and item in paper_distractor_cache:
        return paper_distractor_cache[item]

    if paper_distractor_db is None:
        raise ValueError(PAPER_INDEX_DISTRACTOR_DB_REQUIRED_ERROR.format(index=index, item=item))

    node_id = parse_paper_index_item(item, index)
    if paper_distractor_db.get_paper(nid_db_id(node_id)) is None:
        raise ValueError(PAPER_INDEX_DISTRACTOR_NOT_FOUND_ERROR.format(index=index, item=item))

    paragraphs = (
        build_metadata_needle(
            node_id,
            paper_distractor_db,
            {},
            {},
        ),
    )
    if paper_distractor_cache is not None:
        paper_distractor_cache[item] = paragraphs
    return paragraphs


def get_metadata_path(metadata: Any, field_name: str) -> Path | None:
    """Return one existing path from a metadata object."""
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(field_name)
    if not value:
        return None
    path = Path(str(value))
    if path.exists():
        return path
    return None


def resolve_paper_distractor_db_path(input_metadata: Any) -> Path | None:
    """Resolve the DB path needed to render paper_index distractors."""
    if not isinstance(input_metadata, dict):
        return None

    distracting_metadata = input_metadata.get(DISTRACTING_CONTENT_FIELD, {})
    metadata_candidates = [
        metadata
        for metadata in (distracting_metadata, input_metadata)
        if isinstance(metadata, dict)
    ]

    for metadata in metadata_candidates:
        db_path = get_metadata_path(metadata, DB_PATH_FIELD)
        if db_path is not None:
            return db_path

    for metadata in metadata_candidates:
        graph_path = get_metadata_path(metadata, GRAPH_JSON_FIELD)
        if graph_path is None:
            continue

        meta_db_path = ""
        raw_meta_db_path = metadata.get(DB_PATH_FIELD)
        if raw_meta_db_path:
            meta_db_path = str(raw_meta_db_path)
        try:
            resolved_db, _ = resolve_db_path(
                graph_path,
                None,
                meta_db_path=meta_db_path,
            )
        except FileNotFoundError:
            continue
        return resolved_db

    return None


def build_repeated_distractor_prefix(
    distracting_content_paragraphs: list[str],
    count: int,
) -> list[str]:
    """Return the first ``count`` items from the repeating distractor stream."""
    if not distracting_content_paragraphs or count <= 0:
        return []

    full_cycles, remainder = divmod(count, len(distracting_content_paragraphs))
    return distracting_content_paragraphs * full_cycles + distracting_content_paragraphs[:remainder]


def find_max_fitting_distractor_count(
    distracting_content_paragraphs: list[str],
    *,
    max_count: int,
    fits_distractor_count: Callable[[int], bool],
) -> int:
    """Return the largest repeating distractor-prefix count that still fits."""
    if not distracting_content_paragraphs or max_count <= 0:
        return 0

    fit_cache: dict[int, bool] = {}

    def count_fits(distractor_count: int) -> bool:
        cached_result = fit_cache.get(distractor_count)
        if cached_result is not None:
            return cached_result

        fit_result = fits_distractor_count(distractor_count)
        fit_cache[distractor_count] = fit_result
        return fit_result

    low = 0
    high = 1
    while high <= max_count and count_fits(high):
        low = high
        if high == max_count:
            return low
        high = min(high * 2, max_count)
        if high == low:
            return low

    while low < high:
        mid = (low + high + 1) // 2
        if count_fits(mid):
            low = mid
        else:
            high = mid - 1

    return low


def normalize_openai_input_token_counter_model_name(api_model_name: str) -> str:
    """Return the OpenAI Responses API model name for one configured model."""
    model_name = api_model_name.strip()
    if model_name.startswith(OPENAI_API_MODEL_PREFIX):
        return model_name[len(OPENAI_API_MODEL_PREFIX) :]
    return model_name


def should_use_fake_openai_tokenizer(
    target_context_length: int | None,
) -> bool:
    """Return whether prompt packing should include local OpenAI token counts."""
    return target_context_length is not None


class FakeOpenAITokenizer:
    """Local tiktoken-based counter used as an OpenAI prompt-packing tokenizer."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encoding = self.load_encoding(model_name)
        encoding_name = getattr(self.encoding, "name", OPENAI_TIKTOKEN_FALLBACK_ENCODING)
        self.prompt_packing_identity = ("fake_openai_tokenizer", encoding_name)

    @staticmethod
    def load_encoding(model_name: str) -> Any:
        if tiktoken is None:
            return CharacterFallbackEncoding()

        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            try:
                return tiktoken.get_encoding(OPENAI_TIKTOKEN_FALLBACK_ENCODING)
            except Exception:
                return CharacterFallbackEncoding()
        except Exception:
            return CharacterFallbackEncoding()

    def encode(self, text: str, add_special_tokens: bool = False) -> Any:
        del add_special_tokens
        if isinstance(self.encoding, CharacterFallbackEncoding):
            return self.encoding.encode(text)
        try:
            return self.encoding.encode(text, disallowed_special=())
        except TypeError:
            return self.encoding.encode(text)

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        token_count = 3
        for message in messages:
            token_count += 3
            token_count += len(self.encode(str(message.get("role", ""))))
            token_count += len(self.encode(str(message.get("content", ""))))
            if "name" in message:
                token_count += 1
                token_count += len(self.encode(str(message["name"])))
        return token_count


class CharacterFallbackEncoding:
    """Offline fallback that conservatively treats each character as one token."""

    name = "character_fallback"

    def encode(self, text: str) -> range:
        return range(len(text))


class OpenAIInputTokenCounter:
    """Count chat input tokens through the OpenAI Responses API."""

    def __init__(self, model_name: str) -> None:
        if OpenAI is None:
            raise ImportError(OPENAI_COUNTER_IMPORT_ERROR)

        self.model_name = model_name
        self.client = None
        self.prompt_packing_identity = ("openai_input_tokens_count", model_name)

    def get_client(self) -> Any:
        """Return the cached OpenAI client."""
        if self.client is None:
            self.client = OpenAI()
        return self.client

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        """Return the counted input tokens for one chat transcript."""
        client = self.get_client()
        try:
            response = client.responses.input_tokens.count(
                model=self.model_name,
                input=messages,
            )
        except AttributeError as exc:
            raise RuntimeError(OPENAI_COUNTER_API_UNAVAILABLE_ERROR) from exc

        return int(response.input_tokens)


def load_prompt_packing_tokenizers(
    target_context_length: int | None = None,
    *,
    only_slugs: Sequence[str] | None = None,
) -> list[tuple[str, Any]]:
    """Load every relevant Hugging Face tokenizer used for shared packing.

    ``only_slugs`` loads just those models' tokenizers, e.g. the distinct ones an
    earlier full call kept, so a worker process skips loading the duplicates.
    """
    use_fake_openai_tokenizer = should_use_fake_openai_tokenizer(target_context_length)

    if AutoTokenizer is None and not use_fake_openai_tokenizer:
        raise ImportError("`transformers` is required to load prompt-packing tokenizers.")

    from src.llm.model_info import DEFAULT_LLM_MANAGER

    model_items = list(DEFAULT_LLM_MANAGER.models.items())
    strict_hf_available = any(
        bool(getattr(model_info, "hf_model_id", ""))
        and (
            target_context_length is None
            or int(getattr(model_info, "native_context_length", 0)) >= target_context_length
        )
        for _slug, model_info in model_items
    )
    fallback_hf_available = any(
        bool(getattr(model_info, "hf_model_id", "")) for _slug, model_info in model_items
    )
    use_long_context_hf_fallback = (
        target_context_length is not None
        and target_context_length >= LONG_CONTEXT_HF_FALLBACK_THRESHOLD
        and not strict_hf_available
        and fallback_hf_available
    )
    if use_long_context_hf_fallback:
        log_line(
            TOKENIZER_LONG_CONTEXT_FALLBACK_WARNING.format(
                target_context_length=target_context_length,
            )
        )

    tokenizers: list[tuple[str, Any]] = []
    for slug, model_info in model_items:
        # The fallback choice above still sees every model, as in a full call.
        if only_slugs is not None and slug not in only_slugs:
            continue
        hf_model_id = getattr(model_info, "hf_model_id", "")
        api_model_name = getattr(model_info, "api_model_name", None)
        provider = getattr(model_info, "provider", None)
        native_context_length = int(getattr(model_info, "native_context_length", 0))

        if hf_model_id:
            if AutoTokenizer is None:
                continue
            if (
                target_context_length is not None
                and native_context_length < target_context_length
                and not use_long_context_hf_fallback
            ):
                continue
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    hf_model_id,
                    trust_remote_code=True,
                )
            except Exception as exc:
                log_line(
                    TOKENIZER_LOAD_WARNING.format(
                        slug=slug,
                        hf_model_id=hf_model_id,
                        error_type=type(exc).__name__,
                        error=exc,
                    )
                )
                continue
            tokenizers.append((slug, tokenizer))
            continue

        if target_context_length is not None and native_context_length < target_context_length:
            continue
        if provider != "OpenAI" or not use_fake_openai_tokenizer:
            continue
        if not isinstance(api_model_name, str) or not api_model_name.strip():
            continue

        openai_model_name = normalize_openai_input_token_counter_model_name(api_model_name)
        try:
            tokenizers.append((slug, FakeOpenAITokenizer(openai_model_name)))
        except Exception as exc:
            log_line(
                OPENAI_FAKE_TOKENIZER_LOAD_WARNING.format(
                    slug=slug,
                    model_name=openai_model_name,
                    error_type=type(exc).__name__,
                    error=exc,
                )
            )
            continue

    tokenizers, duplicate_count = deduplicate_prompt_packing_tokenizers(tokenizers)

    if not tokenizers:
        raise ValueError(NO_TOKENIZERS_AVAILABLE_ERROR)

    duplicates_text = (
        TOKENIZER_DUPLICATES_MESSAGE.format(count=duplicate_count) if duplicate_count else ""
    )
    log_line(
        TOKENIZER_REMAINING_MESSAGE.format(slugs=", ".join(slug for slug, _tokenizer in tokenizers))
        + duplicates_text
    )

    return tokenizers


def same_tokenizer(
    tokenizer1: AutoTokenizer,
    tokenizer2: AutoTokenizer,
) -> bool:
    """Return whether two tokenizers are identical."""
    identity1 = getattr(tokenizer1, "prompt_packing_identity", None)
    identity2 = getattr(tokenizer2, "prompt_packing_identity", None)
    if identity1 is not None or identity2 is not None:
        return identity1 is not None and identity1 == identity2

    if tokenizer1.__class__.__name__ != tokenizer2.__class__.__name__:
        return False

    if tokenizer1.vocab_size != tokenizer2.vocab_size:
        return False

    if tokenizer1.get_vocab() != tokenizer2.get_vocab():
        return False

    return all(
        tokenizer1.convert_ids_to_tokens(index) == tokenizer2.convert_ids_to_tokens(index)
        for index in range(tokenizer1.vocab_size)
    )


def deduplicate_prompt_packing_tokenizers(
    tokenizers: list[tuple[str, Any]],
) -> tuple[list[tuple[str, Any]], int]:
    """Return the unique prompt-packing tokenizers and how many were dropped.

    Every benchmark model family shares a handful of tokenizers, so listing each
    dropped duplicate buried the pipeline log; the caller reports one count.
    """
    unique_tokenizers: list[tuple[str, Any]] = []
    duplicate_count = 0
    for slug, tokenizer in tokenizers:
        if any(
            same_tokenizer(existing_tokenizer, tokenizer)
            for _existing_slug, existing_tokenizer in unique_tokenizers
        ):
            duplicate_count += 1
            continue
        unique_tokenizers.append((slug, tokenizer))
    return unique_tokenizers, duplicate_count


def render_messages_for_token_count(messages: list[dict[str, str]]) -> str:
    """Render chat messages into plain text for tokenizer fallback counting."""
    return "".join(f"<{message['role']}>{message['content']}" for message in messages)


def count_plain_encoded_tokens(
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
) -> int:
    """Count tokens by encoding a plain rendered message transcript."""
    rendered_messages = render_messages_for_token_count(messages)
    encode_kwargs: dict[str, Any] = {"add_special_tokens": True}
    if max_tokens is not None:
        encode_kwargs.update({"truncation": True, "max_length": max_tokens})
    try:
        token_ids = tokenizer.encode(rendered_messages, **encode_kwargs)
    except TypeError:
        token_ids = tokenizer.encode(rendered_messages)
    except AttributeError:
        call_kwargs = dict(encode_kwargs)
        try:
            encoded = tokenizer(rendered_messages, **call_kwargs)
        except TypeError:
            encoded = tokenizer(rendered_messages)
        if isinstance(encoded, dict):
            token_ids = encoded["input_ids"]
        else:
            token_ids = encoded.input_ids
    token_count = len(token_ids)
    if max_tokens is not None:
        return min(token_count, max_tokens)
    return token_count


def chat_template_token_ids(result: Any) -> Any:
    """Return token ids from ``apply_chat_template(tokenize=True)``.

    transformers 5 returns a BatchEncoding (``input_ids``, ``attention_mask``)
    instead of a list, so ``len()`` of the raw result would always be 2.
    """
    if isinstance(result, Mapping):
        return result["input_ids"]
    return result


def measure_tokenized_chat_length(
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
) -> int:
    """Return the tokenized prompt length for one tokenizer."""
    count_messages = getattr(tokenizer, "count_messages", None)
    if callable(count_messages):
        token_count = int(count_messages(messages))
        if max_tokens is not None:
            return min(token_count, max_tokens)
        return token_count

    prompt_packing_mode = getattr(tokenizer, "prompt_packing_mode", None)
    if prompt_packing_mode == "plain_encode":
        return count_plain_encoded_tokens(tokenizer, messages, max_tokens)

    template_kwargs: dict[str, Any] = {"tokenize": True}
    if max_tokens is not None:
        template_kwargs.update({"truncation": True, "max_length": max_tokens})
    try:
        token_ids = tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        try:
            token_ids = tokenizer.apply_chat_template(messages, tokenize=True)
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            setattr(tokenizer, "prompt_packing_mode", "plain_encode")
            return count_plain_encoded_tokens(tokenizer, messages, max_tokens)
        token_count = len(chat_template_token_ids(token_ids))
        setattr(tokenizer, "prompt_packing_mode", "chat_template")
        if max_tokens is not None:
            return min(token_count, max_tokens)
        return token_count
    except (AttributeError, NotImplementedError, ValueError):
        setattr(tokenizer, "prompt_packing_mode", "plain_encode")
        return count_plain_encoded_tokens(tokenizer, messages, max_tokens)

    setattr(tokenizer, "prompt_packing_mode", "chat_template")
    token_count = len(chat_template_token_ids(token_ids))
    if max_tokens is not None:
        return min(token_count, max_tokens)
    return token_count


def build_user_prompt_for_question(
    question: dict[str, Any],
    index: int,
    *,
    distracting_content_type: str | None,
    question_last: bool,
    target_context_length: int | None,
    system_prompt: str | None,
    format_reminder: str | None,
    needle_order: str,
    random_seed: int | None,
    context_item_prefix: str,
    context_item_separator: str,
    context_tail_margin: int,
    partial_negative_tree_packing: bool,
    tokenizers: list[tuple[str, Any]] | None,
    wiki_bm25_retriever: Any,
    wiki_dense_retriever: Any,
    paper_distractor_db: DBReader | None,
    no_system_message: bool = False,
    path_paragraph_cache: dict[tuple[str, bool], tuple[str, ...]] | None = None,
    paper_distractor_cache: dict[str, tuple[str, ...]] | None = None,
    paper_index_distractor_loader: Callable[..., tuple[str, ...]] | None = None,
    context_stats: dict[str, int] | None = None,
    distractor_item_loader: Callable[[str, int], tuple[str, ...]] | None = None,
    fragment_length_cache: dict[tuple[int, str], int | None] | None = None,
) -> str | None:
    """Build one packed user prompt or return ``None`` when it cannot fit."""
    question_text = validate_question_text(question, index)
    needles = validate_needles(question, index)
    distracting_content_items = validate_distracting_content(question, index)
    atomic_distractors = (
        question.get(QUESTION_TYPE_FIELD) in REACHABILITY_QUESTION_TYPES
        or question.get("answer_type") in REACHABILITY_ANSWER_TYPES
    )

    if needle_order not in SUPPORTED_NEEDLE_ORDERS:
        raise ValueError(INVALID_NEEDLE_ORDER_ERROR.format(needle_order=needle_order))
    if no_system_message and system_prompt is None:
        raise ValueError(NO_SYSTEM_MESSAGE_REQUIRES_SYSTEM_PROMPT_ERROR)

    if target_context_length is not None:
        if target_context_length <= 0:
            raise ValueError(INVALID_TARGET_CONTEXT_LENGTH_ERROR)
        if context_tail_margin < 0:
            raise ValueError(INVALID_CONTEXT_TAIL_MARGIN_ERROR)
        if system_prompt is None:
            raise ValueError(TARGET_CONTEXT_REQUIRES_SYSTEM_PROMPT_ERROR)
        if format_reminder is None:
            raise ValueError(TARGET_CONTEXT_REQUIRES_FORMAT_REMINDER_ERROR)
        if not tokenizers:
            raise ValueError(NO_TOKENIZERS_AVAILABLE_ERROR)
    elif tokenizers is None:
        tokenizers = []

    prompt_length_cache: dict[tuple[str, int | None], int] = {}
    prompt_overhead_cache: dict[int, int] = {}
    if fragment_length_cache is None:
        fragment_length_cache = {}

    # Packing probes use cached fragment counts; final fit checks still use
    # full rendered chat tokenization.
    needle_texts = set(needles)
    final_context_items: list[str] = []

    def build_final_user_message(context_items: list[str]) -> str:
        nonlocal final_context_items
        final_context_items = list(context_items)
        if context_stats is not None:
            distractor_items = sum(1 for item in context_items if item not in needle_texts)
            context_stats[DISTRACTOR_ITEMS_FIELD] = distractor_items
            context_stats[NEEDLE_ITEMS_FIELD] = len(context_items) - distractor_items
        context_text = build_context_text(
            context_items,
            context_item_prefix=context_item_prefix,
            context_item_separator=context_item_separator,
        )
        return append_format_reminder(
            build_user_message(
                question_text,
                context_text,
                question_last,
                instruction_prompt=(system_prompt if no_system_message else None),
            ),
            format_reminder,
        )

    def build_messages(user_message: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt is not None and not no_system_message:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        return messages

    def count_text_fragment_tokens(tokenizer: Any, text: str) -> int | None:
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            try:
                return len(encode(text, add_special_tokens=False))
            except TypeError:
                return len(encode(text))

        if not callable(tokenizer):
            return None

        try:
            encoded = tokenizer(text, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer(text)

        if isinstance(encoded, dict):
            return len(encoded["input_ids"])
        return len(encoded.input_ids)

    def tokenizer_fragment_length(
        tokenizer_index: int,
        tokenizer: Any,
        text: str,
    ) -> int | None:
        cache_key = (tokenizer_index, text)
        if cache_key in fragment_length_cache:
            return fragment_length_cache[cache_key]

        token_count = count_text_fragment_tokens(tokenizer, text)
        fragment_length_cache[cache_key] = token_count
        return token_count

    def tokenizer_items_length(
        tokenizer_index: int,
        tokenizer: Any,
        items: list[str],
    ) -> int | None:
        """Return the summed length of some context items for one tokenizer."""
        total = 0
        for item in items:
            token_length = tokenizer_fragment_length(tokenizer_index, tokenizer, item)
            if token_length is None:
                return None
            total += token_length
        return total

    def context_token_split(
        needle_items: list[str],
        distractor_items: list[str],
    ) -> tuple[int, int] | None:
        """Return the (needle, distractor) token split of one packed context.

        Packing enforces the budget with whichever tokenizer counts the prompt
        longest, so the split is reported for that same tokenizer. Taking the
        longest count per item instead would add up to more tokens than any real
        prompt holds.
        """
        splits: list[tuple[int, int]] = []
        for tokenizer_index, (_slug, tokenizer) in enumerate(tokenizers or []):
            needle_tokens = tokenizer_items_length(
                tokenizer_index,
                tokenizer,
                needle_items,
            )
            distractor_tokens = tokenizer_items_length(
                tokenizer_index,
                tokenizer,
                distractor_items,
            )
            if needle_tokens is None or distractor_tokens is None:
                continue
            splits.append((needle_tokens, distractor_tokens))
        if not splits:
            return None
        return max(splits, key=sum)

    def finalize_user_message(user_message: str | None) -> str | None:
        """Record needle/distractor token totals for the returned prompt.

        Packing only ever compares whole prompts, so the per-item split is
        measured once here, on the context items the returned prompt kept.
        """
        if user_message is None or context_stats is None or not tokenizers:
            return user_message

        split = context_token_split(
            [item for item in final_context_items if item in needle_texts],
            [item for item in final_context_items if item not in needle_texts],
        )
        if split is None:
            return user_message
        context_stats[NEEDLE_TOKENS_FIELD] = split[0]
        context_stats[DISTRACTOR_TOKENS_FIELD] = split[1]
        return user_message

    def record_needles_only_stats(prompt_tokens: int) -> None:
        """Record how far the needle set alone overshot the target length.

        The prompt is measured in full so the skip warning can report the
        amount by which it exceeded the target.
        """
        if context_stats is None:
            return
        context_stats[NEEDLE_ITEMS_FIELD] = len(needles)
        context_stats[PROMPT_TOKENS_FIELD] = prompt_tokens

    def user_message_fragments(context_items: list[str]) -> list[str]:
        if context_items:
            context_fragments = ["# Context\n\n", context_item_prefix]
            for item_index, item in enumerate(context_items):
                if item_index > 0:
                    context_fragments.append(context_item_separator)
                context_fragments.append(item)
        else:
            context_fragments = ["# Context"]

        if question_last:
            fragments = context_fragments + ["\n\n# Question\n\n", question_text]
        else:
            fragments = ["# Question\n\n", question_text, "\n\n"] + context_fragments

        if no_system_message:
            fragments = [
                "# Instruction\n",
                str(system_prompt or "").strip(),
                "\n\n",
                *fragments,
            ]

        if format_reminder:
            fragments.extend(["\n\n", format_reminder])
        return fragments

    def tokenizer_prompt_overhead(tokenizer_index: int, tokenizer: Any) -> int:
        cached_overhead = prompt_overhead_cache.get(tokenizer_index)
        if cached_overhead is not None:
            return cached_overhead

        overhead = measure_tokenized_chat_length(tokenizer, build_messages(""))
        prompt_overhead_cache[tokenizer_index] = overhead
        return overhead

    def estimate_tokenizer_prompt_length(
        tokenizer_index: int,
        tokenizer: Any,
        fragments: list[str],
    ) -> int | None:
        prompt_length = 0
        for fragment in fragments:
            fragment_length = tokenizer_fragment_length(
                tokenizer_index,
                tokenizer,
                fragment,
            )
            if fragment_length is None:
                return None
            prompt_length += fragment_length
        prompt_length += tokenizer_prompt_overhead(tokenizer_index, tokenizer)
        return prompt_length

    def estimated_max_prompt_length(context_items: list[str]) -> int | None:
        if not tokenizers:
            return 0

        fragments = user_message_fragments(context_items)
        estimated_lengths: list[int] = []
        for tokenizer_index, (_slug, tokenizer) in enumerate(tokenizers):
            tokenizer_length = estimate_tokenizer_prompt_length(
                tokenizer_index,
                tokenizer,
                fragments,
            )
            if tokenizer_length is not None:
                estimated_lengths.append(tokenizer_length)

        if not estimated_lengths:
            return None
        return max(estimated_lengths)

    def max_prompt_length(
        context_items: list[str],
        max_tokens: int | None = None,
    ) -> int:
        if not tokenizers:
            return 0
        user_message = build_final_user_message(context_items)
        cache_key = (user_message, max_tokens)
        cached_length = prompt_length_cache.get(cache_key)
        if cached_length is not None:
            return cached_length

        messages = build_messages(user_message)
        prompt_length = 0
        for _slug, tokenizer in tokenizers:
            tokenizer_length = measure_tokenized_chat_length(
                tokenizer,
                messages,
                max_tokens,
            )
            prompt_length = max(prompt_length, tokenizer_length)
            if max_tokens is not None and prompt_length >= max_tokens:
                break
        prompt_length_cache[cache_key] = prompt_length
        return prompt_length

    def fits_target_length(context_items: list[str], target_length: int) -> bool:
        return max_prompt_length(context_items, target_length + 1) <= target_length

    def fast_fits_target_length(context_items: list[str], target_length: int) -> bool:
        estimated_length = estimated_max_prompt_length(context_items)
        if estimated_length is not None:
            return estimated_length <= target_length
        return fits_target_length(context_items, target_length)

    if target_context_length is not None:
        prompt_without_distractors_tokens = max_prompt_length(
            list(needles),
            target_context_length + 1,
        )
        if prompt_without_distractors_tokens > target_context_length:
            prompt_without_distractors_tokens = max_prompt_length(list(needles))
            record_needles_only_stats(prompt_without_distractors_tokens)
            return None
        if context_stats is not None:
            context_stats[PROMPT_WITHOUT_DISTRACTORS_TOKENS_FIELD] = (
                prompt_without_distractors_tokens
            )

    def truncate_text_to_fit_target(
        base_context_items: list[str],
        tail_text: str,
        target_length: int,
    ) -> str:
        low = 0
        high = len(tail_text)

        while low < high:
            mid = (low + high + 1) // 2
            truncated_tail = tail_text[:mid].rstrip()
            if not truncated_tail:
                high = mid - 1
                continue

            if fast_fits_target_length(
                base_context_items + [truncated_tail],
                target_length,
            ):
                low = mid
            else:
                high = mid - 1

        return tail_text[:low].rstrip()

    def load_distracting_content_paragraphs(item: str) -> tuple[str, ...]:
        if distractor_item_loader is not None:
            return distractor_item_loader(item, index)
        if item.startswith(PATH_ITEM_PREFIX):
            path = Path(item[len(PATH_ITEM_PREFIX) :])
            return load_path_distracting_content_paragraphs(
                path,
                distracting_content_type=distracting_content_type,
                path_paragraph_cache=path_paragraph_cache,
            )

        if (
            item.startswith(PAPER_INDEX_ITEM_PREFIX)
            or is_paper_related_index_item(item)
            or is_paper_negative_tree_item(item)
        ):
            paper_index_loader = (
                paper_index_distractor_loader or load_paper_index_distracting_content_paragraphs
            )
            return paper_index_loader(
                item,
                index,
                question=question,
                paper_distractor_db=paper_distractor_db,
                paper_distractor_cache=paper_distractor_cache,
            )

        if item.startswith(WIKI_BM25_ITEM_PREFIX):
            if wiki_bm25_retriever is None:
                raise ValueError(
                    WIKI_RETRIEVER_REQUIRED_ERROR.format(
                        index=index,
                        retriever_name="BM25Retriever",
                        item=item,
                    )
                )
            paragraph = wiki_bm25_retriever.get_document_by_id(item[len(WIKI_BM25_ITEM_PREFIX) :])
            if paragraph is None:
                raise ValueError(WIKI_PARAGRAPH_NOT_FOUND_ERROR.format(index=index, item=item))
            return split_distracting_content_paragraphs(str(paragraph.content))

        if item.startswith(WIKI_DENSE_ITEM_PREFIX):
            if wiki_dense_retriever is None:
                raise ValueError(
                    WIKI_RETRIEVER_REQUIRED_ERROR.format(
                        index=index,
                        retriever_name="DenseRetriever",
                        item=item,
                    )
                )
            paragraph = wiki_dense_retriever.get_document_by_id(item[len(WIKI_DENSE_ITEM_PREFIX) :])
            if paragraph is None:
                raise ValueError(WIKI_PARAGRAPH_NOT_FOUND_ERROR.format(index=index, item=item))
            return split_distracting_content_paragraphs(str(paragraph.content))

        if item.startswith(REPEAT_ITEM_PREFIX):
            return split_distracting_content_paragraphs(item[len(REPEAT_ITEM_PREFIX) :])

        raise ValueError(UNSUPPORTED_DISTRACTOR_ITEM_ERROR.format(index=index, item=item))

    def context_texts(context_entries: list[dict[str, Any]]) -> list[str]:
        return [str(entry["text"]) for entry in context_entries]

    def apply_needle_order(
        context_entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if needle_order == "random":
            return context_entries

        ordered_needles = [entry for entry in context_entries if entry["kind"] == "needle"]
        ordered_needles.sort(
            key=lambda entry: int(entry["order"]),
            reverse=needle_order == "backward",
        )
        needle_iter = iter(ordered_needles)
        return [
            next(needle_iter) if entry["kind"] == "needle" else entry for entry in context_entries
        ]

    def trim_entries_to_target(
        context_entries: list[dict[str, Any]],
        removable_kinds: set[str],
        target_length: int,
        removal_priority: Callable[[dict[str, Any]], tuple[int, ...]] | None = None,
        fits_context: Callable[[list[str], int], bool] | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        context_fits = fits_context or fast_fits_target_length
        uses_fast_fit = fits_context is None or fits_context is fast_fits_target_length
        trimmed_entries = list(context_entries)
        if context_fits(context_texts(trimmed_entries), target_length):
            return trimmed_entries, True

        removable_indexes: list[int] = []
        for entry_index, entry in enumerate(trimmed_entries):
            if str(entry["kind"]) in removable_kinds:
                removable_indexes.append(entry_index)

        if not removable_indexes:
            return trimmed_entries, False

        def priority_key(entry_index: int) -> tuple[int, ...]:
            entry = trimmed_entries[entry_index]
            entry_priority = (
                removal_priority(entry) if removal_priority is not None else (entry_index,)
            )
            return (*entry_priority, entry_index)

        removable_indexes.sort(key=priority_key, reverse=True)

        def entries_after_removing(remove_count: int) -> list[dict[str, Any]]:
            removed_indexes = set(removable_indexes[:remove_count])
            return [
                entry
                for entry_index, entry in enumerate(trimmed_entries)
                if entry_index not in removed_indexes
            ]

        all_removed_entries = entries_after_removing(len(removable_indexes))
        if not context_fits(context_texts(all_removed_entries), target_length):
            if uses_fast_fit and fits_target_length(
                context_texts(all_removed_entries),
                target_length,
            ):
                return all_removed_entries, True
            return all_removed_entries, False

        low = 1
        high = len(removable_indexes)
        while low < high:
            mid = (low + high) // 2
            candidate_entries = entries_after_removing(mid)
            if context_fits(context_texts(candidate_entries), target_length):
                high = mid
            else:
                low = mid + 1

        trimmed_entries = entries_after_removing(low)
        reinsert_index = removable_indexes[low - 1]
        removed_for_reinsert = set(removable_indexes[:low])
        removed_for_reinsert.remove(reinsert_index)
        base_entries = [
            entry
            for entry_index, entry in enumerate(context_entries)
            if entry_index not in removed_for_reinsert and entry_index != reinsert_index
        ]
        reinsert_entry = context_entries[reinsert_index]
        if not atomic_distractors:
            truncated_text = truncate_text_to_fit_target(
                context_texts(base_entries),
                str(reinsert_entry["text"]),
                target_length,
            )
            if truncated_text and truncated_text != reinsert_entry["text"]:
                candidate_entries = []
                for entry_index, entry in enumerate(context_entries):
                    if entry_index in removed_for_reinsert:
                        continue
                    if entry_index == reinsert_index:
                        candidate_entries.append({**entry, "text": truncated_text})
                    else:
                        candidate_entries.append(entry)
                if context_fits(context_texts(candidate_entries), target_length):
                    return candidate_entries, True

        return trimmed_entries, True

    def paper_related_removal_priority(entry: dict[str, Any]) -> tuple[int, int]:
        return int(entry.get("related_level") or 1), int(entry.get("order") or 0)

    def build_repeat_needles_user_prompt() -> str | None:
        if target_context_length is None:
            raise ValueError(REPEAT_NEEDLES_TARGET_CONTEXT_REQUIRED_ERROR)
        if needle_order != "random":
            raise ValueError(REPEAT_NEEDLES_RANDOM_ORDER_REQUIRED_ERROR)
        if not needles:
            return build_final_user_message([])

        shuffle_seed = random.Random(random_seed).getrandbits(64)

        def shuffled_needle_rounds(round_count: int) -> list[str]:
            context_items = list(needles) * round_count
            random.Random(shuffle_seed).shuffle(context_items)
            return context_items

        packing_target_length = max(
            target_context_length - context_tail_margin,
            0,
        )
        selected_round_count = 1
        if fast_fits_target_length(
            shuffled_needle_rounds(1),
            packing_target_length,
        ):
            selected_round_count = max(
                1,
                find_max_fitting_distractor_count(
                    needles,
                    max_count=max(packing_target_length, 1),
                    fits_distractor_count=lambda round_count: fast_fits_target_length(
                        shuffled_needle_rounds(round_count),
                        packing_target_length,
                    ),
                ),
            )

        selected_round_count = find_max_fitting_distractor_count(
            needles,
            max_count=selected_round_count,
            fits_distractor_count=lambda round_count: fits_target_length(
                shuffled_needle_rounds(round_count),
                target_context_length,
            ),
        )
        if selected_round_count <= 0:
            return None
        return build_final_user_message(shuffled_needle_rounds(selected_round_count))

    def build_paper_negative_tree_user_prompt() -> str | None:
        """Pack complete negative trees and, when enabled, one partial tree."""
        tree_items = [
            item for item in distracting_content_items if is_paper_negative_tree_item(item)
        ]
        group_rng = random.Random(random_seed)
        group_rng.shuffle(tree_items)

        entry_rng = random.Random(random_seed)
        needle_entries = [
            {
                "kind": "needle",
                "text": needle,
                "order": needle_index,
                "group_index": None,
                "random_order": entry_rng.random(),
            }
            for needle_index, needle in enumerate(needles)
        ]
        distractor_entries_by_group: list[list[dict[str, Any]]] = []
        next_tree_item_index = 0
        ordered_entries: list[dict[str, Any]] = []
        ordered_entries_group_count = -1
        context_cache: dict[tuple[int, int], list[str]] = {}

        def ensure_tree_groups(group_count: int) -> bool:
            nonlocal next_tree_item_index
            while len(distractor_entries_by_group) < group_count and next_tree_item_index < len(
                tree_items
            ):
                item = tree_items[next_tree_item_index]
                next_tree_item_index += 1
                paragraphs = load_distracting_content_paragraphs(item)
                if not paragraphs:
                    continue
                group_index = len(distractor_entries_by_group)
                distractor_entries_by_group.append(
                    [
                        {
                            "kind": "distractor",
                            "text": paragraph,
                            "group_index": group_index,
                            "block_index": block_index,
                            "random_order": entry_rng.random(),
                        }
                        for block_index, paragraph in enumerate(paragraphs)
                    ]
                )
            return len(distractor_entries_by_group) >= group_count

        def context_for_selection(
            group_count: int,
            partial_block_count: int = 0,
        ) -> list[str]:
            """Return one cached context selection in deterministic prompt order."""
            nonlocal ordered_entries
            nonlocal ordered_entries_group_count

            required_group_count = group_count + int(partial_block_count > 0)
            if not ensure_tree_groups(required_group_count):
                return []

            cache_key = (group_count, partial_block_count)
            cached_context = context_cache.get(cache_key)
            if cached_context is not None:
                return cached_context

            loaded_group_count = len(distractor_entries_by_group)
            if ordered_entries_group_count != loaded_group_count:
                ordered_entries = list(needle_entries)
                for group_entries in distractor_entries_by_group:
                    ordered_entries.extend(group_entries)
                ordered_entries.sort(key=lambda entry: float(entry["random_order"]))
                ordered_entries_group_count = loaded_group_count

            entries = [
                entry
                for entry in ordered_entries
                if entry["kind"] == "needle"
                or int(entry["group_index"]) < group_count
                or (
                    int(entry["group_index"]) == group_count
                    and int(entry["block_index"]) < partial_block_count
                )
            ]
            entries = apply_needle_order(entries)
            context = [str(entry["text"]) for entry in entries]
            context_cache[cache_key] = context
            return context

        def context_for_group_count(group_count: int) -> list[str]:
            return context_for_selection(group_count)

        selected_group_count = len(tree_items)
        selected_partial_block_count = 0
        if target_context_length is not None:
            if partial_negative_tree_packing:
                selected_group_count = find_max_fitting_distractor_count(
                    tree_items,
                    max_count=len(tree_items),
                    fits_distractor_count=lambda group_count: (
                        ensure_tree_groups(group_count)
                        and fits_target_length(
                            context_for_group_count(group_count),
                            target_context_length,
                        )
                    ),
                )
                if ensure_tree_groups(selected_group_count + 1):
                    next_group = distractor_entries_by_group[selected_group_count]
                    selected_partial_block_count = find_max_fitting_distractor_count(
                        [str(entry["text"]) for entry in next_group],
                        max_count=len(next_group),
                        fits_distractor_count=lambda block_count: fits_target_length(
                            context_for_selection(selected_group_count, block_count),
                            target_context_length,
                        ),
                    )
            else:
                packing_target_length = max(
                    target_context_length - context_tail_margin,
                    0,
                )
                selected_group_count = find_max_fitting_distractor_count(
                    tree_items,
                    max_count=len(tree_items),
                    fits_distractor_count=lambda group_count: (
                        ensure_tree_groups(group_count)
                        and fast_fits_target_length(
                            context_for_group_count(group_count),
                            packing_target_length,
                        )
                    ),
                )
                while selected_group_count > 0 and not fits_target_length(
                    context_for_group_count(selected_group_count),
                    target_context_length,
                ):
                    selected_group_count -= 1
        else:
            ensure_tree_groups(len(tree_items))
            selected_group_count = len(distractor_entries_by_group)

        context_items = context_for_selection(
            selected_group_count,
            selected_partial_block_count,
        )
        if target_context_length is not None and not fits_target_length(
            context_items,
            target_context_length,
        ):
            return None
        if context_stats is not None:
            context_stats["complete_distractor_trees"] = selected_group_count
            context_stats["partial_distractor_tree_blocks"] = selected_partial_block_count
        return build_final_user_message(context_items)

    def build_paper_related_user_prompt() -> str | None:
        needle_entries = [
            {"kind": "needle", "text": needle, "order": needle_index}
            for needle_index, needle in enumerate(needles)
        ]
        paper_related_items: list[str] = []
        paper_remaining_items: list[str] = []

        for item in distracting_content_items:
            if is_paper_related_index_item(item):
                paper_related_items.append(item)
            else:
                paper_remaining_items.append(item)

        def load_paper_related_entries() -> list[dict[str, Any]]:
            entries: list[dict[str, Any]] = []
            for item in paper_related_items:
                related_level = paper_related_index_item_level(item, index)
                for paragraph in load_distracting_content_paragraphs(item):
                    entries.append(
                        {
                            "kind": "paper_related",
                            "text": paragraph,
                            "order": len(entries),
                            "related_level": related_level,
                        }
                    )
            return entries

        paper_remaining_entry_cache: list[dict[str, Any]] = []
        next_paper_remaining_item_index = 0

        def ensure_paper_remaining_entries(entry_count: int) -> bool:
            nonlocal next_paper_remaining_item_index
            attempts_without_output = 0
            while len(paper_remaining_entry_cache) < entry_count:
                if next_paper_remaining_item_index >= len(paper_remaining_items):
                    if (
                        not atomic_distractors
                        or target_context_length is None
                        or not paper_remaining_items
                    ):
                        break
                    next_paper_remaining_item_index = 0
                item = paper_remaining_items[next_paper_remaining_item_index]
                next_paper_remaining_item_index += 1
                previous_count = len(paper_remaining_entry_cache)
                for paragraph in load_distracting_content_paragraphs(item):
                    paper_remaining_entry_cache.append(
                        {
                            "kind": "paper_remaining",
                            "text": paragraph,
                            "order": len(paper_remaining_entry_cache),
                            "related_level": None,
                        }
                    )
                if len(paper_remaining_entry_cache) == previous_count:
                    attempts_without_output += 1
                    if attempts_without_output >= len(paper_remaining_items):
                        break
                else:
                    attempts_without_output = 0
            return len(paper_remaining_entry_cache) >= entry_count

        def load_all_paper_remaining_entries() -> list[dict[str, Any]]:
            ensure_paper_remaining_entries(len(paper_remaining_items))
            while next_paper_remaining_item_index < len(paper_remaining_items):
                ensure_paper_remaining_entries(len(paper_remaining_entry_cache) + 1)
            return list(paper_remaining_entry_cache)

        def get_paper_remaining_entries(entry_count: int) -> list[dict[str, Any]]:
            ensure_paper_remaining_entries(entry_count)
            return paper_remaining_entry_cache[:entry_count]

        def get_paper_remaining_entry(entry_index: int) -> dict[str, Any] | None:
            ensure_paper_remaining_entries(entry_index + 1)
            if entry_index >= len(paper_remaining_entry_cache):
                return None
            return paper_remaining_entry_cache[entry_index]

        paper_related_entries = load_paper_related_entries()

        rng = random.Random(random_seed)
        rng.shuffle(paper_related_entries)
        context_entries = list(needle_entries) + paper_related_entries
        rng.shuffle(context_entries)
        context_entries = apply_needle_order(context_entries)

        def build_shuffled_paper_related_prompt(
            entries: list[dict[str, Any]],
        ) -> str | None:
            final_entries = list(entries)
            rng.shuffle(final_entries)
            final_entries = apply_needle_order(final_entries)

            if target_context_length is not None and not fits_target_length(
                context_texts(final_entries),
                target_context_length,
            ):
                final_entries, entries_fit = trim_entries_to_target(
                    final_entries,
                    {"paper_remaining"},
                    target_context_length,
                    fits_context=fits_target_length,
                )
                if not entries_fit:
                    final_entries, entries_fit = trim_entries_to_target(
                        final_entries,
                        {"paper_related"},
                        target_context_length,
                        paper_related_removal_priority,
                        fits_context=fits_target_length,
                    )
                if not entries_fit:
                    return None

            return build_final_user_message(context_texts(final_entries))

        if target_context_length is None:
            paper_remaining_entries = load_all_paper_remaining_entries()
            return build_shuffled_paper_related_prompt(context_entries + paper_remaining_entries)

        context_entries, entries_fit = trim_entries_to_target(
            context_entries,
            {"paper_related"},
            target_context_length,
            paper_related_removal_priority,
        )
        if not entries_fit:
            return None

        packing_target_length = max(target_context_length - context_tail_margin, 0)
        paper_remaining_entries: list[dict[str, Any]] = []
        if paper_remaining_items and fast_fits_target_length(
            context_texts(context_entries),
            packing_target_length,
        ):
            ensure_paper_remaining_entries(1)
            paper_remaining_entries = get_paper_remaining_entries(1)
        if paper_remaining_entries:

            def build_context_items_for_remaining_count(
                remaining_count: int,
            ) -> list[str] | None:
                if not ensure_paper_remaining_entries(remaining_count):
                    return None
                return context_texts(context_entries + get_paper_remaining_entries(remaining_count))

            def remaining_count_fits(remaining_count: int) -> bool:
                context_items = build_context_items_for_remaining_count(remaining_count)
                if context_items is None:
                    return False
                return fast_fits_target_length(context_items, packing_target_length)

            selected_remaining_count = find_max_fitting_distractor_count(
                paper_remaining_items,
                max_count=packing_target_length,
                fits_distractor_count=remaining_count_fits,
            )
            context_entries = context_entries + get_paper_remaining_entries(
                selected_remaining_count
            )

            if fast_fits_target_length(
                context_texts(context_entries),
                packing_target_length,
            ):
                tail_entry = get_paper_remaining_entry(selected_remaining_count)
                if tail_entry is not None and not atomic_distractors:
                    tail_text = truncate_text_to_fit_target(
                        context_texts(context_entries),
                        str(tail_entry["text"]),
                        packing_target_length,
                    )
                    if tail_text:
                        context_entries.append({**tail_entry, "text": tail_text})

        context_entries, entries_fit = trim_entries_to_target(
            context_entries,
            {"paper_remaining"},
            target_context_length,
            fits_context=fits_target_length,
        )
        if not entries_fit:
            context_entries, entries_fit = trim_entries_to_target(
                context_entries,
                {"paper_related"},
                target_context_length,
                paper_related_removal_priority,
                fits_context=fits_target_length,
            )
        if not entries_fit:
            return None

        return build_shuffled_paper_related_prompt(context_entries)

    if distracting_content_type == DISTRACTING_CONTENT_TYPE_REPEAT_NEEDLES:
        return finalize_user_message(build_repeat_needles_user_prompt())

    if distracting_content_type in {
        DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE,
        DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V4,
        DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V5,
        DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V6,
        DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V7,
        DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V8,
        DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V10,
        DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V11,
        DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V9,
    }:
        return finalize_user_message(build_paper_negative_tree_user_prompt())

    if distracting_content_type == DISTRACTING_CONTENT_TYPE_PAPER_RELATED:
        return finalize_user_message(build_paper_related_user_prompt())

    distracting_content_paragraphs: list[str] = []
    for item in distracting_content_items:
        distracting_content_paragraphs.extend(load_distracting_content_paragraphs(item))

    selected_distracting_content_paragraphs = list(distracting_content_paragraphs)
    tail_distracting_content_paragraph = ""

    if target_context_length is not None:
        assert tokenizers is not None
        packing_target_length = max(
            target_context_length - context_tail_margin,
            0,
        )

        selected_distracting_content_paragraphs = []
        if distracting_content_paragraphs:

            def build_context_items_for_distractor_count(
                distractor_count: int,
            ) -> list[str]:
                return list(needles) + build_repeated_distractor_prefix(
                    distracting_content_paragraphs,
                    distractor_count,
                )

            selected_distractor_count = find_max_fitting_distractor_count(
                distracting_content_paragraphs,
                max_count=packing_target_length,
                fits_distractor_count=lambda distractor_count: fast_fits_target_length(
                    build_context_items_for_distractor_count(distractor_count),
                    packing_target_length,
                ),
            )
            selected_distracting_content_paragraphs = build_repeated_distractor_prefix(
                distracting_content_paragraphs,
                selected_distractor_count,
            )

        base_context_items = list(needles) + list(selected_distracting_content_paragraphs)
        if (
            distracting_content_paragraphs
            and not atomic_distractors
            and fast_fits_target_length(
                base_context_items,
                packing_target_length,
            )
        ):
            candidate_tail = distracting_content_paragraphs[
                len(selected_distracting_content_paragraphs) % len(distracting_content_paragraphs)
            ]
            tail_distracting_content_paragraph = truncate_text_to_fit_target(
                base_context_items,
                candidate_tail,
                packing_target_length,
            )

    context_entries = [
        {"kind": "needle", "text": needle, "order": needle_index}
        for needle_index, needle in enumerate(needles)
    ] + [
        {
            "kind": "distractor",
            "text": paragraph,
            "order": paragraph_index,
        }
        for paragraph_index, paragraph in enumerate(selected_distracting_content_paragraphs)
    ]

    rng = random.Random(random_seed)
    rng.shuffle(context_entries)
    context_entries = apply_needle_order(context_entries)

    if tail_distracting_content_paragraph:
        context_entries.append(
            {
                "kind": "distractor",
                "text": tail_distracting_content_paragraph,
                "order": len(selected_distracting_content_paragraphs),
            }
        )

    if target_context_length is not None:
        assert tokenizers is not None
        while True:
            current_context_items = [entry["text"] for entry in context_entries]
            current_max_prompt_length = max_prompt_length(current_context_items)
            if current_max_prompt_length <= target_context_length:
                break

            if context_entries and context_entries[-1]["kind"] == "distractor":
                truncated_tail = truncate_text_to_fit_target(
                    [entry["text"] for entry in context_entries[:-1]],
                    context_entries[-1]["text"],
                    target_context_length,
                )
                if truncated_tail and truncated_tail != context_entries[-1]["text"]:
                    context_entries[-1]["text"] = truncated_tail
                    continue

                context_entries.pop()
                continue

            last_distractor_index = None
            for entry_index in range(len(context_entries) - 1, -1, -1):
                if context_entries[entry_index]["kind"] == "distractor":
                    last_distractor_index = entry_index
                    break

            if last_distractor_index is None:
                return None

            del context_entries[last_distractor_index]

    return finalize_user_message(
        build_final_user_message([entry["text"] for entry in context_entries])
    )


def build_fallback_qid(question: dict[str, Any], index: int) -> str:
    """Return a deterministic qid when upstream data omitted one."""
    qid_basis = orjson.dumps({
            "index": index,
            "question": question.get(QUESTION_FIELD),
            "answer": question.get(ANSWER_FIELD),
            "needles": question.get(NEEDLES_FIELD),
        }, option=orjson.OPT_SORT_KEYS).decode("utf-8")
    return str(uuid.uuid5(uuid.NAMESPACE_URL, qid_basis))


def resolve_question_id(question: dict[str, Any], index: int) -> tuple[str, bool]:
    """Return ``(qid, synthesized)`` for one question entry."""
    qid = question.get(QUESTION_ID_FIELD)
    if qid is None:
        return build_fallback_qid(question, index), True

    qid_text = str(qid).strip()
    if not qid_text:
        return build_fallback_qid(question, index), True
    return qid_text, False


def build_question_plan(
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the ordered qid plan for one prompt-build run."""
    question_plan: list[dict[str, Any]] = []
    for index, question in enumerate(questions):
        qid, synthesized = resolve_question_id(question, index)
        question_plan.append(
            {
                "index": index,
                "question": question,
                QUESTION_ID_FIELD: qid,
                "synthesized": synthesized,
            }
        )
    return question_plan


def normalize_checkpoint_run_settings(
    run_settings: dict[str, Any],
) -> dict[str, Any]:
    """Return the checkpoint settings that must stay fixed across resumes."""
    normalized = {
        field_name: value
        for field_name, value in run_settings.items()
        if field_name not in MUTABLE_CHECKPOINT_RUN_SETTING_FIELDS
    }
    normalized[QUESTION_TYPE_FIELD] = normalized.get(QUESTION_TYPE_FIELD) or DEFAULT_QUESTION_TYPE
    normalized[NO_SYSTEM_MESSAGE_FIELD] = bool(normalized.get(NO_SYSTEM_MESSAGE_FIELD, False))
    normalized[PARTIAL_NEGATIVE_TREE_PACKING_FIELD] = bool(
        normalized.get(PARTIAL_NEGATIVE_TREE_PACKING_FIELD, False)
    )
    return normalized


def process_question(
    question: dict[str, Any],
    index: int,
    *,
    question_type: str,
    distracting_content_type: str | None,
    question_last: bool,
    target_context_length: int | None,
    system_prompt: str | None,
    format_reminder: str | None,
    needle_order: str,
    random_seed: int | None,
    context_item_prefix: str,
    context_item_separator: str,
    context_tail_margin: int,
    partial_negative_tree_packing: bool,
    tokenizers: list[tuple[str, Any]] | None,
    wiki_bm25_retriever: Any,
    wiki_dense_retriever: Any,
    paper_distractor_db: DBReader | None,
    no_system_message: bool = False,
    path_paragraph_cache: dict[tuple[str, bool], tuple[str, ...]] | None = None,
    paper_distractor_cache: dict[str, tuple[str, ...]] | None = None,
    paper_index_distractor_loader: Callable[..., tuple[str, ...]] | None = None,
    context_stats: dict[str, int] | None = None,
    distractor_item_loader: Callable[[str, int], tuple[str, ...]] | None = None,
    fragment_length_cache: dict[tuple[int, str], int | None] | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    """Return one full-prompt record plus whether its qid was synthesized."""
    answer_value = validate_answer_value(question, index)
    qid, synthesized = resolve_question_id(question, index)

    if context_stats is None:
        context_stats = {}
    user_message = build_user_prompt_for_question(
        question,
        index,
        distracting_content_type=distracting_content_type,
        question_last=question_last,
        target_context_length=target_context_length,
        system_prompt=system_prompt,
        format_reminder=format_reminder,
        needle_order=needle_order,
        random_seed=random_seed,
        context_item_prefix=context_item_prefix,
        context_item_separator=context_item_separator,
        context_tail_margin=context_tail_margin,
        partial_negative_tree_packing=partial_negative_tree_packing,
        tokenizers=tokenizers,
        wiki_bm25_retriever=wiki_bm25_retriever,
        wiki_dense_retriever=wiki_dense_retriever,
        paper_distractor_db=paper_distractor_db,
        no_system_message=no_system_message,
        path_paragraph_cache=path_paragraph_cache,
        paper_distractor_cache=paper_distractor_cache,
        paper_index_distractor_loader=paper_index_distractor_loader,
        context_stats=context_stats,
        **({"distractor_item_loader": distractor_item_loader}
           if distractor_item_loader is not None else {}),
        **({"fragment_length_cache": fragment_length_cache}
           if fragment_length_cache is not None else {}),
    )
    if user_message is None:
        return None, synthesized

    output_question = {
        QUESTION_ID_FIELD: qid,
        USER_MESSAGE_FIELD: user_message,
        ANSWER_FIELD: answer_value,
        "answer_type": question.get("answer_type"),
    }
    for stats_field in CONTEXT_STATS_FIELDS:
        if stats_field in context_stats:
            output_question[stats_field] = context_stats[stats_field]
    output_question[SOURCE_FINGERPRINT_FIELD] = prompt_source_fingerprint(
        question,
        system_prompt=system_prompt,
        format_reminder=format_reminder,
    )
    record_question_type = question_type
    if question.get(QUESTION_TYPE_FIELD) is not None or question_type != DEFAULT_QUESTION_TYPE:
        output_question[QUESTION_TYPE_FIELD] = record_question_type
    if no_system_message:
        output_question[NO_SYSTEM_MESSAGE_FIELD] = True
    elif record_question_type in REACHABILITY_QUESTION_TYPES and system_prompt is not None:
        output_question[SYSTEM_MESSAGE_FIELD] = system_prompt
    return output_question, synthesized


def format_prompt_skipped_warning(
    *,
    index: int,
    qid: str,
    target_context_length: int | None,
    context_stats: dict[str, int],
) -> str:
    """Return the one-line reason why one question could not be packed."""
    details = "".join(
        f" {label}={context_stats[field]:,}"
        for field, label in PROMPT_SKIPPED_DETAIL_FIELDS
        if isinstance(context_stats.get(field), int)
    )
    return PROMPT_SKIPPED_WARNING.format(
        index=index,
        qid=qid,
        details=details,
        target_context_length=target_context_length or 0,
    )


def prompt_source_fingerprint(
    question: dict[str, Any],
    *,
    system_prompt: str | None,
    format_reminder: str | None,
) -> str:
    """Return a hash of everything in one input record that shapes its prompt.

    ``RE_RENDER_QUESTIONS=1`` rebuilds a saved prompt when this value changes,
    which covers a re-rendered question, normalized needles, and distractors
    that were rebuilt after the record was first packed.
    """
    payload = orjson.dumps(
        {
            "question": question,
            "system_prompt": system_prompt,
            "format_reminder": format_reminder,
        },
        option=orjson.OPT_SORT_KEYS,
    )
    return hashlib.sha256(payload).hexdigest()


def current_checkpoint_records(
    existing_questions: list[dict[str, Any]],
    existing_metadata: dict[str, Any] | None,
    questions: list[dict[str, Any]],
    *,
    system_prompt: str | None,
    format_reminder: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Return only the saved prompts whose input record is still unchanged."""
    fingerprints = {
        qid: prompt_source_fingerprint(
            question,
            system_prompt=system_prompt,
            format_reminder=format_reminder,
        )
        for question in questions
        if (qid := str(question.get(QUESTION_ID_FIELD) or "").strip())
    }
    kept = [
        question
        for question in existing_questions
        if (saved_fingerprint := question.get(SOURCE_FINGERPRINT_FIELD))
        and saved_fingerprint
        == fingerprints.get(str(question.get(QUESTION_ID_FIELD) or "").strip())
    ]
    log_line(
        f"[{CLI_NAME}] re-rendered questions: keeping {len(kept)} of "
        f"{len(existing_questions)} saved prompt(s); rebuilding the rest"
    )
    if not kept:
        return [], None
    metadata = dict(existing_metadata or {})
    full_prompts_metadata = metadata.get(FULL_PROMPTS_FIELD)
    if isinstance(full_prompts_metadata, dict):
        metadata[FULL_PROMPTS_FIELD] = {
            **full_prompts_metadata,
            SKIPPED_QIDS_FIELD: [],
        }
    return kept, metadata


def build_full_prompts_run_settings(
    *,
    question_last: bool,
    target_context_length: int | None,
    tokenizers: list[tuple[str, Any]] | None,
    needle_order: str,
    random_seed: int | None,
    context_item_prefix: str,
    context_item_separator: str,
    context_tail_margin: int,
    max_samples: int | None,
    partial_negative_tree_packing: bool = True,
    no_system_message: bool = False,
    question_type: str = DEFAULT_QUESTION_TYPE,
    traversal_mode: str = DEFAULT_REACHABILITY_TRAVERSAL_MODE,
) -> dict[str, Any]:
    """Return the run settings stored in full-prompts metadata."""
    tokenizer_model_slugs = None
    if tokenizers is not None:
        tokenizer_model_slugs = [slug for slug, _tokenizer in tokenizers]

    settings = {
        QUESTION_TYPE_FIELD: question_type,
        QUESTION_LAST_FIELD: question_last,
        TARGET_CONTEXT_LENGTH_FIELD: target_context_length,
        TOKENIZER_MODEL_SLUGS_FIELD: tokenizer_model_slugs,
        NEEDLE_ORDER_FIELD: needle_order,
        RANDOM_SEED_FIELD: random_seed,
        CONTEXT_ITEM_PREFIX_FIELD: context_item_prefix,
        CONTEXT_ITEM_SEPARATOR_FIELD: context_item_separator,
        CONTEXT_TAIL_MARGIN_FIELD: context_tail_margin,
        PARTIAL_NEGATIVE_TREE_PACKING_FIELD: partial_negative_tree_packing,
        MAX_SAMPLES_FIELD: max_samples,
        NO_SYSTEM_MESSAGE_FIELD: no_system_message,
    }
    if question_type in REACHABILITY_QUESTION_TYPES:
        settings[TRAVERSAL_MODE_FIELD] = traversal_mode
    return settings


def build_base_full_prompts_metadata(
    input_path: Path,
    *,
    run_settings: dict[str, Any],
    total_questions: int,
    synthesized_qids: int,
    skipped_qids: list[str] | None = None,
) -> dict[str, Any]:
    """Return metadata describing this full-prompt build."""
    full_prompts_metadata = {
        **run_settings,
        TOTAL_QUESTIONS_FIELD: total_questions,
        SYNTHESIZED_QIDS_FIELD: synthesized_qids,
    }
    if skipped_qids:
        full_prompts_metadata[SKIPPED_QIDS_FIELD] = skipped_qids
    return {
        SOURCE_QUESTIONS_FIELD: str(input_path),
        FULL_PROMPTS_FIELD: full_prompts_metadata,
    }


def add_full_prompts_metadata(
    full_prompts_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return the minimal metadata block for one full-prompt pass."""
    return full_prompts_metadata


def load_existing_output_questions(
    out_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Load existing output questions plus metadata from one checkpoint file."""
    if not out_path.exists():
        return [], None
    try:
        data = orjson.loads(out_path.read_text(encoding="utf-8"))
    except orjson.JSONDecodeError:
        return [], None
    if not isinstance(data, dict):
        return [], None

    questions = data.get(QUESTIONS_FIELD)
    if not isinstance(questions, list):
        return [], None
    if any(not isinstance(question, dict) for question in questions):
        return [], None

    metadata = data.get(METADATA_FIELD)
    if metadata is not None and not isinstance(metadata, dict):
        metadata = None

    return questions, metadata


def load_checkpoint_skipped_qids(
    *,
    out_path: Path,
    existing_metadata: dict[str, Any],
) -> list[str]:
    """Return skipped qids recorded in one checkpoint metadata block."""
    full_prompts_metadata = existing_metadata.get(FULL_PROMPTS_FIELD)
    if not isinstance(full_prompts_metadata, dict):
        raise build_incompatible_checkpoint_error(
            out_path,
            {"metadata.full_prompts": full_prompts_metadata},
            {"metadata.full_prompts": "<object>"},
        )

    raw_skipped_qids = full_prompts_metadata.get(SKIPPED_QIDS_FIELD)
    if raw_skipped_qids is None:
        return []
    if not isinstance(raw_skipped_qids, list):
        raise build_incompatible_checkpoint_error(
            out_path,
            {"metadata.full_prompts.skipped_qids": raw_skipped_qids},
            {
                "metadata.full_prompts.skipped_qids": (
                    "<list of non-empty unique strings>"
                )
            },
        )

    skipped_qids: list[str] = []
    for index, qid in enumerate(raw_skipped_qids):
        if not isinstance(qid, str):
            field_name = f"metadata.full_prompts.skipped_qids[{index}]"
            raise build_incompatible_checkpoint_error(
                out_path,
                {field_name: qid},
                {field_name: "<non-empty string>"},
            )
        qid_text = qid.strip()
        if not qid_text:
            field_name = f"metadata.full_prompts.skipped_qids[{index}]"
            raise build_incompatible_checkpoint_error(
                out_path,
                {field_name: qid},
                {field_name: "<non-empty string>"},
            )
        skipped_qids.append(qid_text)

    if len(set(skipped_qids)) != len(skipped_qids):
        raise build_incompatible_checkpoint_error(
            out_path,
            {"metadata.full_prompts.skipped_qids": skipped_qids},
            {"metadata.full_prompts.skipped_qids": list(dict.fromkeys(skipped_qids))},
        )

    return skipped_qids


def ensure_checkpoint_matches_current_run(
    *,
    out_path: Path,
    input_path: Path,
    existing_metadata: dict[str, Any] | None,
    existing_questions: list[dict[str, Any]],
    question_plan: list[dict[str, Any]],
    run_settings: dict[str, Any],
) -> None:
    """Raise when an existing checkpoint does not match the current run."""
    current_qids = [plan[QUESTION_ID_FIELD] for plan in question_plan]
    if not isinstance(existing_metadata, dict):
        raise build_incompatible_checkpoint_error(
            out_path,
            {"metadata": existing_metadata},
            {"metadata": "<object>"},
        )

    saved_input_path = existing_metadata.get(SOURCE_QUESTIONS_FIELD)
    if saved_input_path != str(input_path):
        raise build_incompatible_checkpoint_error(
            out_path,
            {SOURCE_QUESTIONS_FIELD: saved_input_path},
            {SOURCE_QUESTIONS_FIELD: str(input_path)},
        )

    full_prompts_metadata = existing_metadata.get(FULL_PROMPTS_FIELD)
    if not isinstance(full_prompts_metadata, dict):
        raise build_incompatible_checkpoint_error(
            out_path,
            {"metadata.full_prompts": full_prompts_metadata},
            {"metadata.full_prompts": "<object>"},
        )

    existing_settings = {
        field_name: full_prompts_metadata.get(field_name) for field_name in run_settings
    }
    normalized_existing_settings = normalize_checkpoint_run_settings(existing_settings)
    normalized_run_settings = normalize_checkpoint_run_settings(run_settings)
    if normalized_existing_settings != normalized_run_settings:
        raise build_incompatible_checkpoint_error(
            out_path,
            normalized_existing_settings,
            normalized_run_settings,
        )

    existing_skipped_qids = load_checkpoint_skipped_qids(
        out_path=out_path,
        existing_metadata=existing_metadata,
    )
    existing_qids = [
        qid
        for question in existing_questions
        if (qid := str(question.get(QUESTION_ID_FIELD) or "").strip())
    ]
    if not existing_qids and not existing_skipped_qids:
        raise build_incompatible_checkpoint_error(
            out_path,
            {"checkpoint_qids": []},
            {"checkpoint_qids": "<at least one built or skipped qid>"},
        )

    current_qid_set = set(current_qids)
    existing_qid_set = set(existing_qids)
    skipped_qid_set = set(existing_skipped_qids)
    if len(existing_qid_set) != len(existing_qids):
        raise build_incompatible_checkpoint_error(
            out_path,
            {"checkpoint_qids": existing_qids},
            {"checkpoint_qids": list(dict.fromkeys(existing_qids))},
        )
    overlapping_qids = sorted(existing_qid_set.intersection(skipped_qid_set))
    if overlapping_qids:
        raise build_incompatible_checkpoint_error(
            out_path,
            {"built_and_skipped_qids": overlapping_qids},
            {"built_and_skipped_qids": []},
        )

    unexpected_qids = sorted(
        existing_qid_set.union(skipped_qid_set).difference(current_qid_set)
    )
    if unexpected_qids:
        raise build_incompatible_checkpoint_error(
            out_path,
            {"checkpoint_qids_not_in_current_input": unexpected_qids},
            {"checkpoint_qids_not_in_current_input": []},
        )


def build_output_payload(
    *,
    input_path: Path,
    input_parameters: dict[str, Any],
    run_settings: dict[str, Any],
    question_plan: list[dict[str, Any]],
    output_by_qid: dict[str, dict[str, Any]],
    skipped_qids: set[str] | None,
    has_distracting_content: bool,
    needle_order: str,
    target_context_length: int | None,
) -> dict[str, Any]:
    """Return the serialized output payload for the current build state."""
    skipped_qid_set = skipped_qids or set()
    ordered_questions = [
        output_by_qid[plan[QUESTION_ID_FIELD]]
        for plan in question_plan
        if plan[QUESTION_ID_FIELD] in output_by_qid
    ]
    ordered_skipped_qids = [
        plan[QUESTION_ID_FIELD]
        for plan in question_plan
        if plan[QUESTION_ID_FIELD] in skipped_qid_set
    ]
    synthesized_qids = sum(
        1
        for plan in question_plan
        if plan["synthesized"] and plan[QUESTION_ID_FIELD] in output_by_qid
    )
    full_prompts_metadata = build_base_full_prompts_metadata(
        input_path,
        run_settings=run_settings,
        total_questions=len(ordered_questions),
        synthesized_qids=synthesized_qids,
        skipped_qids=ordered_skipped_qids,
    )
    output_parameters = merge_parameters(
        input_parameters,
        **{
            PIPELINE_NEEDLE_ORDER_PARAMETER_FIELD: (
                needle_order if has_distracting_content else None
            ),
            PIPELINE_TARGET_CONTEXT_LENGTH_PARAMETER_FIELD: (
                target_context_length if has_distracting_content else None
            ),
        },
    )

    return {
        METADATA_FIELD: add_full_prompts_metadata(full_prompts_metadata),
        PARAMETERS_FIELD: output_parameters,
        QUESTIONS_FIELD: ordered_questions,
    }


def write_output_payload(
    out_path: Path,
    payload: dict[str, Any],
    *,
    indent: int,
) -> None:
    """Write one output payload atomically."""
    tmp_path = out_path.with_name(f"{out_path.name}.tmp")
    tmp_path.write_text(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8"),
        encoding="utf-8",
    )
    tmp_path.replace(out_path)


PromptQuestionResult = tuple[
    dict[str, Any] | None,
    str | None,
    dict[str, int],
]
PromptPlanWorker = Callable[[dict[str, Any]], PromptQuestionResult]
PromptPlanWorkerFactory = Callable[
    [int],
    tuple[PromptPlanWorker, Callable[[], None]],
]


def run_stateful_prompt_workers(
    plans: list[dict[str, Any]],
    worker_factory: PromptPlanWorkerFactory,
    *,
    num_workers: int,
    on_result: Callable[[int, PromptQuestionResult], bool],
) -> None:
    """Process plans on reusable worker contexts until ``on_result`` stops.

    Each long-lived worker takes its next queued plan immediately after it
    finishes the current plan, independently of the other workers.
    """
    if not plans:
        return

    effective_workers = min(validate_num_workers(num_workers), len(plans))
    if effective_workers == 1:
        worker, close_worker = worker_factory(1)
        try:
            for position, plan in enumerate(plans):
                if not on_result(position, worker(plan)):
                    break
        finally:
            close_worker()
        return

    tasks: Queue[tuple[int, dict[str, Any]]] = Queue()
    for position, plan in enumerate(plans):
        tasks.put((position, plan))

    completions: Queue[tuple[int | None, PromptQuestionResult | None, BaseException | None]] = (
        Queue()
    )
    stop_event = threading.Event()

    def consume_plans() -> None:
        """Consume queued plans with resources owned by this worker thread."""
        try:
            worker, close_worker = worker_factory(effective_workers)
        except BaseException as error:
            stop_event.set()
            completions.put((None, None, error))
            return

        try:
            while not stop_event.is_set():
                try:
                    position, plan = tasks.get_nowait()
                except Empty:
                    return
                try:
                    result = worker(plan)
                except BaseException as error:
                    stop_event.set()
                    completions.put((None, None, error))
                    return
                completions.put((position, result, None))
        finally:
            close_worker()

    first_error: BaseException | None = None
    completed = 0
    with ThreadPoolExecutor(
        max_workers=effective_workers,
        thread_name_prefix="full-prompt",
    ) as executor:
        futures = [executor.submit(consume_plans) for _worker_index in range(effective_workers)]
        while completed < len(plans):
            position, result, worker_error = completions.get()
            if worker_error is not None:
                first_error = worker_error
                stop_event.set()
                break
            assert position is not None
            assert result is not None
            completed += 1
            try:
                should_continue = on_result(position, result)
            except BaseException as error:
                first_error = error
                stop_event.set()
                break
            if not should_continue:
                stop_event.set()
                break

        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as error:
                if first_error is None:
                    first_error = error

    if first_error is not None:
        raise first_error


def build_full_prompts(
    input_path: Path,
    *,
    question_last: bool = False,
    question_type: str = DEFAULT_QUESTION_TYPE,
    traversal_mode: str = DEFAULT_REACHABILITY_TRAVERSAL_MODE,
    no_system_message: bool = False,
    target_context_length: int | None = None,
    system_prompt: str | None = SYSTEM_PROMPT,
    format_reminder: str | None = FORMAT_REMINDER,
    needle_order: str = DEFAULT_NEEDLE_ORDER,
    random_seed: int | None = None,
    context_item_prefix: str = DEFAULT_CONTEXT_ITEM_PREFIX,
    context_item_separator: str = DEFAULT_CONTEXT_ITEM_SEPARATOR,
    context_tail_margin: int = DEFAULT_CONTEXT_TAIL_MARGIN,
    partial_negative_tree_packing: bool = True,
    distracting_source_dir: str | Path | None = None,
    tokenizers: list[tuple[str, Any]] | None = None,
    max_samples: int | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    checkpoint_path: Path | None = None,
    checkpoint_indent: int = 2,
    paper_distractor_db_path: str | Path | None = None,
    paper_index_distractor_loader: Callable[..., tuple[str, ...]] | None = None,
    paper_index_distractor_loader_factory: (
        Callable[[], Callable[..., tuple[str, ...]]] | None
    ) = None,
) -> dict[str, Any]:
    """Load one pipeline JSON file and build full prompt records."""
    with open(input_path, encoding="utf-8") as fh:
        input_data = orjson.loads(fh.read())

    questions = validate_questions_data(input_data)
    max_samples = validate_max_samples(max_samples)
    num_workers = validate_num_workers(num_workers)
    question_type = validate_question_type(question_type)
    for index, question in enumerate(questions):
        record_type = question.get(QUESTION_TYPE_FIELD)
        if record_type is not None and record_type != question_type:
            raise ValueError(
                QUESTION_TYPE_MISMATCH_ERROR.format(
                    index=index,
                    record_type=record_type,
                    question_type=question_type,
                )
            )
    input_metadata = input_data.get(METADATA_FIELD, {})
    input_parameters = get_parameters_from_payload(input_data)
    needle_type = input_parameters.get(NEEDLE_TYPE_PARAMETER_FIELD)
    if not isinstance(needle_type, str):
        needle_type = None
    system_prompt, format_reminder = resolve_question_type_prompts(
        question_type=question_type,
        needle_type=needle_type,
        traversal_mode=traversal_mode,
        system_prompt=system_prompt,
        format_reminder=format_reminder,
    )
    if question_type != DEFAULT_QUESTION_TYPE or QUESTION_TYPE_PARAMETER_FIELD in input_parameters:
        input_parameters = merge_parameters(
            input_parameters,
            **{QUESTION_TYPE_PARAMETER_FIELD: question_type},
        )
    if question_type in REACHABILITY_QUESTION_TYPES:
        input_parameters = merge_parameters(
            input_parameters,
            **{TRAVERSAL_MODE_PARAMETER_FIELD: traversal_mode},
        )

    if target_context_length is not None and tokenizers is None:
        tokenizers = load_prompt_packing_tokenizers(target_context_length)

    run_settings = build_full_prompts_run_settings(
        question_type=question_type,
        question_last=question_last,
        target_context_length=target_context_length,
        tokenizers=tokenizers,
        needle_order=needle_order,
        random_seed=random_seed,
        context_item_prefix=context_item_prefix,
        context_item_separator=context_item_separator,
        context_tail_margin=context_tail_margin,
        partial_negative_tree_packing=partial_negative_tree_packing,
        max_samples=max_samples,
        no_system_message=no_system_message,
        traversal_mode=traversal_mode,
    )
    question_plan = build_question_plan(questions)
    question_id_order = {plan[QUESTION_ID_FIELD]: plan["index"] for plan in question_plan}
    output_by_qid: dict[str, dict[str, Any]] = {}
    skipped_qid_set: set[str] = set()
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        existing_questions, existing_metadata = load_existing_output_questions(checkpoint_path)
        if rerender_questions_enabled() and (
            existing_questions or existing_metadata is not None
        ):
            existing_questions, existing_metadata = current_checkpoint_records(
                existing_questions,
                existing_metadata,
                questions,
                system_prompt=system_prompt,
                format_reminder=format_reminder,
            )
        if existing_questions or existing_metadata is not None:
            ensure_checkpoint_matches_current_run(
                out_path=checkpoint_path,
                input_path=input_path,
                existing_metadata=existing_metadata,
                existing_questions=existing_questions,
                question_plan=question_plan,
                run_settings=run_settings,
            )
            assert existing_metadata is not None
            skipped_qid_set = set(
                load_checkpoint_skipped_qids(
                    out_path=checkpoint_path,
                    existing_metadata=existing_metadata,
                )
            )
            output_by_qid = {
                qid: question
                for question in existing_questions
                if (qid := str(question.get(QUESTION_ID_FIELD) or "").strip())
                and qid in question_id_order
                and (
                    target_context_length is None
                    or isinstance(
                        question.get(PROMPT_WITHOUT_DISTRACTORS_TOKENS_FIELD),
                        int,
                    )
                )
            }
            if max_samples is not None and len(output_by_qid) > max_samples:
                # raise ValueError(
                #     INCOMPATIBLE_CHECKPOINT_ERROR.format(out_path=checkpoint_path)
                # )
                # Print warning message
                log_line(
                    f"Warning: max_samples is set to {max_samples} but checkpoint contains {len(output_by_qid)} built question(s)."
                )
            if output_by_qid or skipped_qid_set:
                log_line(
                    "Checkpoint found -- loaded "
                    f"{len(output_by_qid)} built question(s) and "
                    f"{len(skipped_qid_set)} skipped question(s)."
                )

    all_distracting_items = [
        item
        for question in questions
        for item in question.get(DISTRACTING_CONTENT_ITEMS_FIELD, [])
        if isinstance(item, str)
    ]
    needs_wiki_bm25 = any(item.startswith(WIKI_BM25_ITEM_PREFIX) for item in all_distracting_items)
    needs_wiki_dense = any(
        item.startswith(WIKI_DENSE_ITEM_PREFIX) for item in all_distracting_items
    )
    needs_paper_index = any(
        item.startswith(PAPER_INDEX_ITEM_PREFIX)
        or is_paper_related_index_item(item)
        or is_paper_negative_tree_item(item)
        for item in all_distracting_items
    )
    distracting_metadata = (
        input_metadata.get(DISTRACTING_CONTENT_FIELD, {})
        if isinstance(input_metadata, dict)
        else {}
    )
    resolved_distracting_source_dir = distracting_source_dir
    if resolved_distracting_source_dir is None and isinstance(distracting_metadata, dict):
        resolved_distracting_source_dir = distracting_metadata.get("source_dir")

    wiki_bm25_cache_dir = (
        resolve_wiki_cache_dir(
            resolved_distracting_source_dir,
            WIKI_BM25_CACHE_SUBDIR,
        )
        if needs_wiki_bm25
        else None
    )
    wiki_dense_cache_dir = (
        resolve_wiki_cache_dir(
            resolved_distracting_source_dir,
            WIKI_DENSE_CACHE_SUBDIR,
        )
        if needs_wiki_dense
        else None
    )
    resolved_paper_distractor_db_path = None
    if needs_paper_index:
        resolved_paper_distractor_db_path = (
            Path(paper_distractor_db_path)
            if paper_distractor_db_path is not None
            else resolve_paper_distractor_db_path(input_metadata)
        )

    distracting_content_type = input_parameters.get(DISTRACTING_CONTENT_TYPE_PARAMETER_FIELD)
    if not isinstance(distracting_content_type, str):
        distracting_content_type = None
    has_distracting_content = input_parameters.get(
        DISTRACTING_CONTENT_TYPE_PARAMETER_FIELD
    ) is not None or bool(distracting_metadata)

    pending_plans = [
        plan
        for plan in question_plan
        if plan[QUESTION_ID_FIELD] not in output_by_qid
        and plan[QUESTION_ID_FIELD] not in skipped_qid_set
    ]
    skipped_count = len(question_plan) - len(pending_plans)
    if checkpoint_path is not None and skipped_count:
        log_line(f"Resume: {skipped_count}/{len(question_plan)} question(s) already built.")

    tokenizer_copy_lock = threading.Lock()

    def make_prompt_worker(
        effective_workers: int,
    ) -> tuple[PromptPlanWorker, Callable[[], None]]:
        """Create prompt resources that remain owned by one worker thread."""
        wiki_bm25_retriever = (
            BM25Retriever(cache_dir=wiki_bm25_cache_dir)
            if wiki_bm25_cache_dir is not None
            else None
        )
        wiki_dense_retriever = (
            DenseRetriever(cache_dir=wiki_dense_cache_dir)
            if wiki_dense_cache_dir is not None
            else None
        )
        paper_distractor_db = None
        if resolved_paper_distractor_db_path is not None:
            paper_distractor_db = DBReader(
                resolved_paper_distractor_db_path,
                sqlite_cache_size_kib=max(
                    1,
                    PAPER_DISTRACTOR_SQLITE_CACHE_SIZE_KIB // effective_workers,
                ),
                sqlite_mmap_size_bytes=max(
                    1,
                    PAPER_DISTRACTOR_SQLITE_MMAP_SIZE_BYTES // effective_workers,
                ),
            )
        path_paragraph_cache: dict[tuple[str, bool], tuple[str, ...]] = {}
        paper_distractor_cache: dict[str, tuple[str, ...]] = {}
        try:
            if effective_workers > 1 and tokenizers:
                with tokenizer_copy_lock:
                    worker_tokenizers = copy.deepcopy(tokenizers)
            else:
                worker_tokenizers = tokenizers
            worker_paper_index_loader = (
                paper_index_distractor_loader_factory()
                if paper_index_distractor_loader_factory is not None
                else paper_index_distractor_loader
            )
        except BaseException:
            if paper_distractor_db is not None:
                paper_distractor_db.close()
            raise

        def process_plan(plan: dict[str, Any]) -> PromptQuestionResult:
            index = int(plan["index"])
            skip_warning = None
            context_stats: dict[str, int] = {}
            try:
                output_question, _synthesized = process_question(
                    plan["question"],
                    index,
                    question_type=question_type,
                    distracting_content_type=distracting_content_type,
                    question_last=question_last,
                    target_context_length=target_context_length,
                    system_prompt=system_prompt,
                    format_reminder=format_reminder,
                    needle_order=needle_order,
                    random_seed=random_seed,
                    context_item_prefix=context_item_prefix,
                    context_item_separator=context_item_separator,
                    context_tail_margin=context_tail_margin,
                    partial_negative_tree_packing=partial_negative_tree_packing,
                    tokenizers=worker_tokenizers,
                    wiki_bm25_retriever=wiki_bm25_retriever,
                    wiki_dense_retriever=wiki_dense_retriever,
                    paper_distractor_db=paper_distractor_db,
                    no_system_message=no_system_message,
                    path_paragraph_cache=path_paragraph_cache,
                    paper_distractor_cache=paper_distractor_cache,
                    paper_index_distractor_loader=worker_paper_index_loader,
                    context_stats=context_stats,
                )
            except ValueError as error:
                if str(error) != INVALID_ANSWER_ERROR.format(index=index):
                    raise
                output_question = None
                skip_warning = INVALID_ANSWER_SKIP_WARNING.format(
                    error=error,
                    index=index,
                    qid=plan[QUESTION_ID_FIELD],
                )
            return output_question, skip_warning, context_stats

        def close_worker() -> None:
            if paper_distractor_db is not None:
                paper_distractor_db.close()

        return process_plan, close_worker

    def write_checkpoint() -> None:
        if checkpoint_path is None:
            return
        write_output_payload(
            checkpoint_path,
            build_output_payload(
                input_path=input_path,
                input_parameters=input_parameters,
                run_settings=run_settings,
                question_plan=question_plan,
                output_by_qid=output_by_qid,
                skipped_qids=skipped_qid_set,
                has_distracting_content=has_distracting_content,
                needle_order=needle_order,
                target_context_length=target_context_length,
            ),
            indent=checkpoint_indent,
        )

    completed_results: dict[int, PromptQuestionResult] = {}
    next_commit_position = 0

    with tqdm(
        total=len(question_plan),
        initial=skipped_count,
        desc="Building full prompts",
        unit="question",
    ) as progress_bar:

        def commit_completed_result(
            position: int,
            result: PromptQuestionResult,
        ) -> bool:
            nonlocal next_commit_position
            completed_results[position] = result
            while next_commit_position in completed_results:
                if max_samples is not None and len(output_by_qid) >= max_samples:
                    return False
                plan = pending_plans[next_commit_position]
                output_question, skip_warning, context_stats = completed_results.pop(
                    next_commit_position
                )
                index = int(plan["index"])
                if output_question is None:
                    skipped_qid_set.add(plan[QUESTION_ID_FIELD])
                    progress_bar.write(
                        skip_warning
                        or format_prompt_skipped_warning(
                            index=index,
                            qid=plan[QUESTION_ID_FIELD],
                            target_context_length=target_context_length,
                            context_stats=context_stats,
                        )
                    )
                else:
                    output_by_qid[plan[QUESTION_ID_FIELD]] = output_question
                next_commit_position += 1
                progress_bar.update(1)
                write_checkpoint()
                if max_samples is not None and len(output_by_qid) >= max_samples:
                    return False
            return True

        if max_samples is None or len(output_by_qid) < max_samples:
            run_stateful_prompt_workers(
                pending_plans,
                make_prompt_worker,
                num_workers=num_workers,
                on_result=commit_completed_result,
            )

    result = build_output_payload(
        input_path=input_path,
        input_parameters=input_parameters,
        run_settings=run_settings,
        question_plan=question_plan,
        output_by_qid=output_by_qid,
        skipped_qids=skipped_qid_set,
        has_distracting_content=has_distracting_content,
        needle_order=needle_order,
        target_context_length=target_context_length,
    )
    if checkpoint_path is not None:
        write_output_payload(
            checkpoint_path,
            result,
            indent=checkpoint_indent,
        )

    return result


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="JSON",
        help="Input JSON from create_needles.py or create_distracting_content.py.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Path to write the output JSON file.",
    )
    parser.add_argument(
        "--question-last",
        action="store_true",
        help="Place the question after the context instead of before it.",
    )
    parser.add_argument(
        "--question-type",
        choices=sorted(SUPPORTED_QUESTION_TYPES),
        default=DEFAULT_QUESTION_TYPE,
        help=(
            "Prompt family to use. Existing data defaults to factoid; "
            "reachability types select their needle-specific graph prompts."
        ),
    )
    parser.add_argument(
        "--traversal-mode",
        choices=sorted(REACHABILITY_TRAVERSAL_MODES),
        default=DEFAULT_REACHABILITY_TRAVERSAL_MODE,
        help="Graph semantics for reachability prompts.",
    )
    parser.add_argument(
        "--target-context-length",
        type=int,
        default=None,
        help=(
            "Optional shared prompt-token budget. When set, prompt packing uses "
            "every configured Hugging Face tokenizer whose declared native "
            "context length covers that target, plus the provided system "
            "prompt and format reminder."
        ),
    )
    parser.add_argument(
        "--system-prompt-path",
        default=None,
        metavar="PATH",
        help="Optional path to the system prompt text used for token counting.",
    )
    parser.add_argument(
        "--format-reminder-path",
        default=None,
        metavar="PATH",
        help=(
            "Optional path to the format reminder text appended during token "
            "counting. The written user_message still excludes this suffix."
        ),
    )
    parser.add_argument(
        "--needle-order",
        choices=sorted(SUPPORTED_NEEDLE_ORDERS),
        default=DEFAULT_NEEDLE_ORDER,
        help=(
            "Order constraint for needles after shuffling with distractors "
            f"(default: {DEFAULT_NEEDLE_ORDER})."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic context shuffling.",
    )
    parser.add_argument(
        "--max-samples",
        "--max_samples",
        dest="max_samples",
        type=int,
        default=None,
        help="Build up to N prompts, continuing past skipped questions.",
    )
    parser.add_argument(
        "--context-item-prefix",
        default=DEFAULT_CONTEXT_ITEM_PREFIX,
        help=("Prefix inserted before the first context item."),
    )
    parser.add_argument(
        "--context-item-separator",
        default=DEFAULT_CONTEXT_ITEM_SEPARATOR,
        help="Separator inserted between context items.",
    )
    parser.add_argument(
        "--context-tail-margin",
        type=int,
        default=DEFAULT_CONTEXT_TAIL_MARGIN,
        help="Headroom reserved below target_context_length during prompt packing.",
    )
    parser.add_argument(
        "--partial-negative-tree-packing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow a prefix of complete rendered blocks from the next negative "
            "tree to fill the remaining prompt budget (default: enabled)."
        ),
    )
    parser.add_argument(
        "--distracting-source-dir",
        default=None,
        metavar="PATH",
        help=("Optional override for resolving wiki_bm25:/wiki_dense: distractor " "items."),
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation width (default: 2).",
    )
    return parser


def summarize_context_item_usage(
    questions: list[dict[str, Any]] | None,
    count_field: str,
    token_field: str,
) -> dict[str, float] | None:
    """Return packed-item statistics for one kind of context item.

    The distractor pool is generated far larger than any context length can
    hold, so the packed count is the only record of how much of it a prompt
    actually uses. Needle counts are fixed per question, but they still decide
    how much of the budget is left, so both kinds are reported the same way.
    """
    counts: list[int] = []
    token_totals: list[int] = []
    for question in questions or []:
        if not isinstance(question, dict):
            continue
        count = question.get(count_field)
        if not isinstance(count, int):
            continue
        counts.append(count)
        token_total = question.get(token_field)
        if isinstance(token_total, int):
            token_totals.append(token_total)
    if not counts:
        return None

    usage: dict[str, float] = {
        "prompts": len(counts),
        "mean": statistics.fmean(counts),
        "std": statistics.pstdev(counts),
        "min": min(counts),
        "max": max(counts),
    }
    if len(token_totals) == len(counts) and sum(counts) > 0:
        usage["tokens_per_item"] = sum(token_totals) / sum(counts)
    return usage


def format_context_usage_line(
    label: str,
    item_name: str,
    usage: dict[str, float] | None,
) -> str:
    """Return one packed-item summary line, or an empty string."""
    if usage is None:
        return ""
    tokens_per_item = usage.get("tokens_per_item")
    tokens_text = (
        f" | {tokens_per_item:,.1f} tokens/{item_name}" if tokens_per_item is not None else ""
    )
    return (
        f"\n  {label:<15}: {usage['mean']:,.1f} +/- {usage['std']:,.1f} items/prompt"
        f" (min={usage['min']:,}, max={usage['max']:,}){tokens_text}"
    )


def format_measured_prompts_line(
    usages: list[tuple[str, dict[str, float] | None]],
) -> str:
    """Return the line naming how many prompts the item lines cover.

    Resumed output files can hold records written before a stats field existed,
    so the counts are reported per kind whenever they disagree.
    """
    counts = {label: int(usage["prompts"]) for label, usage in usages if usage is not None}
    if not counts:
        return ""
    if len(set(counts.values())) == 1:
        measured = f"{next(iter(counts.values())):,}"
    else:
        measured = ", ".join(f"{label.lower()}={count:,}" for label, count in counts.items())
    return f"\n  {'Measured':<15}: {measured} prompt(s)"


def print_summary(
    out_path: Path,
    metadata: dict[str, Any],
    parameters: dict[str, Any],
    questions: list[dict[str, Any]] | None = None,
    *,
    cli_name: str = CLI_NAME,
    extra_rows: Sequence[tuple[str, str]] = (),
) -> None:
    """Print a concise run summary under the name of the script that ran.

    ``extra_rows`` are ``(label, value)`` pairs a caller adds to this block
    instead of printing its own prefixed lines after it.
    """
    full_prompts_metadata = metadata[FULL_PROMPTS_FIELD]
    question_order = (
        "question-last" if full_prompts_metadata[QUESTION_LAST_FIELD] else "question-first"
    )
    skipped_qids = full_prompts_metadata.get(SKIPPED_QIDS_FIELD, [])
    skipped_count = len(skipped_qids) if isinstance(skipped_qids, list) else 0
    needle_usage = summarize_context_item_usage(
        questions,
        NEEDLE_ITEMS_FIELD,
        NEEDLE_TOKENS_FIELD,
    )
    distractor_usage = summarize_context_item_usage(
        questions,
        DISTRACTOR_ITEMS_FIELD,
        DISTRACTOR_TOKENS_FIELD,
    )
    usage_lines = (
        format_measured_prompts_line([("Needles", needle_usage), ("Distractors", distractor_usage)])
        + format_context_usage_line("Needles", "needle", needle_usage)
        + format_context_usage_line("Distractors", "distractor", distractor_usage)
    )
    log_line(
        f"\n[{cli_name}]"
        f"\n  Built          : {full_prompts_metadata[TOTAL_QUESTIONS_FIELD]} question(s)"
        f"\n  Skipped        : {skipped_count} question(s)"
        f"\n  Order          : {question_order}"
        f"\n  Needle Order   : {parameters.get(PIPELINE_NEEDLE_ORDER_PARAMETER_FIELD)}"
        f"\n  Target Length  : {parameters.get(PIPELINE_TARGET_CONTEXT_LENGTH_PARAMETER_FIELD)}"
        f"{usage_lines}"
        + "".join(f"\n  {label:<15}: {value}" for label, value in extra_rows)
        + f"\n  QIDs           : synthesized {full_prompts_metadata[SYNTHESIZED_QIDS_FIELD]}"
        f"\n  Source         : {format_display_path(metadata[SOURCE_QUESTIONS_FIELD])}"
        f"\n  Output         : {out_path}"
    )


def main() -> None:
    """Run the CLI entry point."""
    args = build_parser().parse_args()
    out_path = Path(args.output)

    system_prompt = SYSTEM_PROMPT
    if args.system_prompt_path is not None:
        system_prompt = read_optional_text_file(args.system_prompt_path)

    format_reminder = FORMAT_REMINDER
    if args.format_reminder_path is not None:
        format_reminder = read_optional_text_file(args.format_reminder_path)

    result = build_full_prompts(
        Path(args.input),
        question_last=args.question_last,
        question_type=args.question_type,
        traversal_mode=args.traversal_mode,
        target_context_length=args.target_context_length,
        system_prompt=system_prompt,
        format_reminder=format_reminder,
        needle_order=args.needle_order,
        random_seed=args.random_seed,
        max_samples=args.max_samples,
        context_item_prefix=args.context_item_prefix,
        context_item_separator=args.context_item_separator,
        context_tail_margin=args.context_tail_margin,
        partial_negative_tree_packing=args.partial_negative_tree_packing,
        distracting_source_dir=args.distracting_source_dir,
        checkpoint_path=out_path,
        checkpoint_indent=args.indent,
    )

    print_summary(
        out_path,
        result[METADATA_FIELD],
        result[PARAMETERS_FIELD],
        result[QUESTIONS_FIELD],
    )


if __name__ == "__main__":
    main()
