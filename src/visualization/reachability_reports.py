"""Bucket-report helpers shared by experiment 7 (reachability yes/no) and
experiment 8 (reachability multiple choice) notebooks.

Every function here is either byte-identical between the two experiments, or
differs only in a small number of explicit parameters (a field-name tuple, a
metric column, a callable). Logic that genuinely diverges between the two
question types -- path scanning (``gather_results``), construction
diagnostics, prompt-payload lookup, and per-label vs. per-option metrics --
stays in ``reachability_yesno.py`` / ``reachability_multiple.py`` instead,
each of which re-exports or thinly wraps the functions below so calling
notebooks see the same public API as before.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pandas as pd
from IPython.display import Image, display

from .artifacts import (
    format_bernoulli_ci_pct,
    format_ci_pct,
    format_mean_std,
    model_slug_matches_family,
    print_figure_path,
    print_table_path,
    stable_content_hash,
    summarize_prompt_fill_payload,
    write_bucket_family_grid_figure_artifacts,
    write_bucket_figure_artifacts,
    write_table_artifacts,
)

# Populated by each experiment's own ``configure()``; shared so that CFG-reading
# functions below need no extra parameter, and ``reachability_multiple.CFG``
# stays accessible to notebooks that read it directly (e.g. for chance
# accuracy). Only one experiment's config is ever loaded per Python process in
# practice -- each notebook is its own kernel -- so sharing one namespace is a
# safe simplification, not a live risk.
CFG = SimpleNamespace()

FAMILY_FILTERS = [
    ("gpt-5.6", "GPT family"),
    ("qwen3.5-moe", "Qwen3.5 MoE family"),
    ("qwen3.5-non-moe", "Qwen3.5 non-MoE family"),
    ("deepseek-v4", "DeepSeek family"),
    ("gemini", "Gemini family"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Path/label helpers
# ─────────────────────────────────────────────────────────────────────────────


def sub_experiment_tables_dir(sub_tag: str, question_hops: str | None = None) -> Path:
    base = CFG.tables_dir / f"sub_experiment={sub_tag}"
    return base if question_hops is None else base / f"question_hops={question_hops}"


def sub_experiment_figures_dir(sub_tag: str, question_hops: str | None = None) -> Path:
    base = CFG.figures_dir / f"sub_experiment={sub_tag}"
    return base if question_hops is None else base / f"question_hops={question_hops}"


def sub_experiment_label(
    sub_tag: str, stem: str, question_hops: str | None = None, *, experiment_tag: str
) -> str:
    if question_hops is None:
        return f"tab:experiment_{experiment_tag}_{sub_tag}_{stem}"
    return f"tab:experiment_{experiment_tag}_{sub_tag}_question_hops_{question_hops}_{stem}"


def as_bucket_axis(frame: pd.DataFrame, axis_field: str) -> pd.DataFrame:
    """Return a view of ``frame`` with ``bucket_label`` aliased to ``axis_field``.

    Lets the bucket-axis table/figure builders below be reused unmodified for a
    different swept axis (e.g. ``question_hops`` in the qhops sub-experiment,
    where bucket is fixed to one value) by relabeling which column they read as
    the category axis.
    """
    if axis_field == "bucket_label":
        return frame
    return frame.assign(bucket_label=frame[axis_field].astype(str))


def order_model_slugs(results: pd.DataFrame, preferred_model_order: list[str]) -> list[str]:
    available = sorted(results["model_slug"].unique()) if not results.empty else []
    ordered = [slug for slug in preferred_model_order if slug in available]
    ordered += [slug for slug in available if slug not in ordered]
    return ordered


# ─────────────────────────────────────────────────────────────────────────────
# Generic artifact-path scanning
# ─────────────────────────────────────────────────────────────────────────────
#
# Order-independent replacement for hand-built glob f-strings: a path is
# accepted as long as every required segment is present and every present
# segment (required or not) matches its configured allowed values, so a new
# path segment (e.g. exp7 v7's ``alpha=...``) added between two existing ones
# is tolerated automatically rather than silently rejecting every path.


def get_condition_layout_rank(
    values: dict[str, str], expansion_segment_keys: frozenset[str]
) -> int | None:
    """Return the rank for the merged condition layout.

    ``expansion_segment_keys`` names segments belonging to a legacy fallback
    layout (e.g. exp8's expansion-depth re-runs); a path carrying any of them
    is deprioritized in favour of the current layout at the same logical
    condition. Pass an empty set when no such fallback layout exists.
    """
    if expansion_segment_keys.intersection(values):
        return None
    return 0


def parse_condition_path(
    path: Path,
    root: Path,
    filename: str,
    required_keys: frozenset[str],
    segment_filters: dict[str, set[str]],
    expansion_segment_keys: frozenset[str],
) -> dict[str, str] | None:
    """Return the ``key=value`` path segments of one artifact, or ``None``.

    Rejects the path when a required segment is missing or when any present
    segment has a configured value that does not match this sweep.
    """
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return None
    if not parts or parts[-1] != filename:
        return None

    values = {}
    for segment in parts[:-1]:
        if "=" not in segment:
            return None
        key, value = segment.split("=", 1)
        values[key] = value

    if not required_keys.issubset(values):
        return None
    for key, allowed in segment_filters.items():
        if key in values and values[key] not in allowed:
            return None
    if get_condition_layout_rank(values, expansion_segment_keys) is None:
        return None
    return values


def select_preferred_artifacts(
    root: Path,
    filename: str,
    required_keys: frozenset[str],
    segment_filters: dict[str, set[str]],
    expansion_segment_keys: frozenset[str] = frozenset(),
) -> list[tuple[Path, dict[str, str]]]:
    """Return one artifact per condition using the scripts' fallback order."""
    selected = {}
    for path in sorted(root.rglob(filename)):
        values = parse_condition_path(
            path, root, filename, required_keys, segment_filters, expansion_segment_keys
        )
        if values is None:
            continue
        logical_condition = tuple(
            sorted((key, value) for key, value in values.items() if key not in expansion_segment_keys)
        )
        rank = get_condition_layout_rank(values, expansion_segment_keys)
        candidate = (rank, path, values)
        previous = selected.get(logical_condition)
        if previous is None or candidate[:2] < previous[:2]:
            selected[logical_condition] = candidate

    return [
        (path, values)
        for _rank, path, values in sorted(selected.values(), key=lambda candidate: str(candidate[1]))
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Shared metric helpers
# ─────────────────────────────────────────────────────────────────────────────


def get_bucket_columns(hop_results: pd.DataFrame, bucket_labels: list[str]) -> list[str]:
    """Return the observed buckets, in ``bucket_labels`` order."""
    observed = set(hop_results["bucket_label"].astype(str))
    return [label for label in bucket_labels if label in observed]


def get_hop_model_slugs(hop_results: pd.DataFrame, ordered_model_slugs: list[str]) -> list[str]:
    """Return the models present for one hop count, in the preferred order."""
    observed = set(hop_results["model_slug"])
    return [slug for slug in ordered_model_slugs if slug in observed]


def pool_counts(subset: pd.DataFrame, count_fields: tuple[str, ...]) -> dict[str, int]:
    """Pool scored counts for one model-bucket cell."""
    return {field: int(subset[field].sum()) for field in count_fields}


def get_cell_counts(
    hop_results: pd.DataFrame, model_slug: str, bucket_label: str, count_fields: tuple[str, ...]
) -> dict[str, int] | None:
    """Return pooled counts for one model-bucket cell, or ``None`` when empty."""
    subset = hop_results[
        (hop_results["model_slug"] == model_slug) & (hop_results["bucket_label"] == bucket_label)
    ]
    if subset.empty:
        return None
    return pool_counts(subset, count_fields)


def overall_accuracy_pct(counts: dict[str, int]) -> float | None:
    """Return accuracy over every scored item in one cell."""
    if not counts["total"]:
        return None
    return 100.0 * counts["correct"] / counts["total"]


def get_average_reasoning_tokens(report: dict) -> float | None:
    """Return the average reasoning-token count per response in an eval report.

    Prefer an explicit reasoning-token count when the provider reports one.
    Older inference artifacts only store ``output_tokens``; that value includes
    hidden reasoning tokens and is the common fallback used by the result
    notebooks.
    """
    summaries = [
        report.get("cost_and_tokens"),
        report.get("inference_metadata"),
    ]
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        value = summary.get("avg_reasoning_tokens")
        if value is not None:
            return float(value)
        total = summary.get("total_reasoning_tokens")
        samples = summary.get("samples_with_usage")
        if total is not None and samples:
            return float(total) / int(samples)

    explicit_values = []
    output_values = []
    for detail in report.get("details") or []:
        usage = detail.get("usage") or {}
        if not isinstance(usage, dict):
            continue
        reasoning_value = usage.get("reasoning_tokens")
        for details_key in ("completion_tokens_details", "output_tokens_details"):
            token_details = usage.get(details_key) or {}
            if reasoning_value is None and isinstance(token_details, dict):
                reasoning_value = token_details.get("reasoning_tokens")
        if reasoning_value is not None:
            explicit_values.append(float(reasoning_value))
        if usage.get("output_tokens") is not None:
            output_values.append(float(usage["output_tokens"]))

    if explicit_values:
        return sum(explicit_values) / len(explicit_values)

    for summary in summaries:
        if isinstance(summary, dict) and summary.get("avg_output_tokens") is not None:
            return float(summary["avg_output_tokens"])
    if output_values:
        return sum(output_values) / len(output_values)
    return None


def mean_reasoning_tokens(frame: pd.DataFrame) -> float | None:
    """Return reasoning tokens per question, weighted by each report's size."""
    if frame.empty or "reasoning_tokens" not in frame:
        return None

    values = pd.to_numeric(frame["reasoning_tokens"], errors="coerce")
    valid = values.notna()
    if not valid.any():
        return None

    if "total" in frame:
        weights = pd.to_numeric(frame["total"], errors="coerce")
        weighted = valid & weights.notna() & (weights > 0)
        if weighted.any():
            return float((values[weighted] * weights[weighted]).sum() / weights[weighted].sum())
    return float(values[valid].mean())


def build_bucket_matrix(
    hop_results: pd.DataFrame,
    value_fn,
    model_slugs: list[str],
    bucket_columns: list[str],
    model_names: dict[str, str],
    count_fields: tuple[str, ...],
) -> pd.DataFrame:
    """Return a Model x bucket matrix filled by ``value_fn(pooled_counts)``."""
    table = pd.DataFrame(
        index=[model_names[slug] for slug in model_slugs], columns=bucket_columns, dtype=float
    )
    for model_slug in model_slugs:
        for bucket_label in bucket_columns:
            counts = get_cell_counts(hop_results, model_slug, bucket_label, count_fields)
            if counts is None:
                continue
            table.loc[model_names[model_slug], bucket_label] = value_fn(counts)
    table.index.name = "Model"
    return table


def build_bucket_ci_matrix(
    hop_results: pd.DataFrame,
    value_fn: Callable[[dict[str, int]], float | None],
    ci_fn: Callable[[dict[str, int]], tuple[float, float]],
    model_slugs: list[str],
    bucket_columns: list[str],
    model_names: dict[str, str],
    count_fields: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return display and numeric matrices for a metric with per-cell 95% CIs."""
    values = build_bucket_matrix(
        hop_results,
        value_fn,
        model_slugs,
        bucket_columns,
        model_names,
        count_fields,
    )
    display_values = values.astype(object)
    for model_slug in model_slugs:
        for bucket_label in bucket_columns:
            counts = get_cell_counts(
                hop_results, model_slug, bucket_label, count_fields
            )
            if counts is None:
                continue
            estimate_pct = value_fn(counts)
            if estimate_pct is None:
                continue
            display_values.loc[model_names[model_slug], bucket_label] = format_ci_pct(
                estimate_pct, ci_fn(counts)
            )
    return display_values, values


def needle_token_cache_path(question_hops: str, bucket_label: str) -> Path:
    """Return the on-disk cache file for one (question_hops, bucket) group's tokenization."""
    return (
        CFG.experiment_root
        / "cache"
        / "needle_token_metrics"
        / f"question_hops={question_hops}"
        / f"bucket={bucket_label}.json"
    )


def add_figure_metric_columns(
    frame: pd.DataFrame, metric_values: dict[str, Callable]
) -> pd.DataFrame:
    """Attach the composition/position-robust metric columns used by the bucket figures."""
    for column, value_fn in metric_values.items():
        if frame.empty:
            frame[column] = pd.Series(dtype=float)
        else:
            frame[column] = frame.apply(value_fn, axis=1).astype(float)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Needle tokenization
# ─────────────────────────────────────────────────────────────────────────────


def needle_question_content_hash(row: dict, answer_field: str) -> str:
    """Hash the parts of a source question row that determine its cached metrics."""
    return stable_content_hash(
        {
            "question": row["question"],
            "needles": row["needles"],
            "question_type": row["question_type"],
            "needle_type": row["needle_type"],
            "traversal_mode": row["traversal_mode"],
            "visible_paper_count": row["visible_paper_count"],
            answer_field: row[answer_field],
            "realized_difficulty": row["realized_difficulty"],
        }
    )


def tokenize_needle_question(
    row: dict,
    packing_tokenizers: list[tuple[str, object]],
    context_item_prefix: str,
    context_item_separator: str,
    question_last: bool,
    target_context_token_count: int,
    answer_field: str,
) -> tuple[dict, list[dict]]:
    """Tokenize one source question -- the expensive per-question step being cached."""
    import build_full_prompts
    from .artifacts import count_text_tokens

    system_prompt, format_reminder = build_full_prompts.resolve_question_type_prompts(
        question_type=str(row["question_type"]),
        needle_type=str(row["needle_type"]),
        traversal_mode=str(row["traversal_mode"]),
        system_prompt=build_full_prompts.SYSTEM_PROMPT,
        format_reminder=build_full_prompts.FORMAT_REMINDER,
    )
    context_text = build_full_prompts.build_context_text(
        row["needles"],
        context_item_prefix=context_item_prefix,
        context_item_separator=context_item_separator,
    )
    user_message = build_full_prompts.append_format_reminder(
        build_full_prompts.build_user_message(row["question"], context_text, question_last),
        format_reminder,
    )
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    needle_only_prompt_tokens = max(
        build_full_prompts.measure_tokenized_chat_length(tokenizer, messages)
        for _slug, tokenizer in packing_tokenizers
    )
    needle_block_token_lengths = [
        max(count_text_tokens(tokenizer, needle) for _slug, tokenizer in packing_tokenizers)
        for needle in row["needles"]
    ]
    total_needle_block_tokens = sum(needle_block_token_lengths)
    visible_paper_count = int(row["visible_paper_count"])
    needle_question_row = {
        "question_hops": row["question_hops"],
        "bucket": row["bucket"],
        "qid": row["qid"],
        answer_field: row[answer_field],
        "realized_difficulty": row["realized_difficulty"],
        "needles_per_question": len(row["needles"]),
        "visible_papers_per_question": visible_paper_count,
        "tokens_per_visible_paper": (
            total_needle_block_tokens / visible_paper_count if visible_paper_count else float("nan")
        ),
        "context_tokens_left_after_needles": target_context_token_count - needle_only_prompt_tokens,
    }
    needle_length_rows = [
        {
            "question_hops": row["question_hops"],
            "bucket": row["bucket"],
            "needle_length_tokens": needle_length_tokens,
        }
        for needle_length_tokens in needle_block_token_lengths
    ]
    return needle_question_row, needle_length_rows


def compute_needle_question_metrics(
    needle_source_questions: pd.DataFrame,
    needle_prompt_fill: pd.DataFrame,
    prompt_packing_configurations: pd.DataFrame,
    answer_field: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tokenize every source question to derive per-question needle/token metrics.

    Tokenizing every needle in every question is the slow part of the needle
    statistics pipeline, so each question's result is cached to disk under
    ``needle_token_cache_path``, keyed by a content hash of that qid's own
    question/needles/packing-relevant fields. A later run (or a different
    sub-experiment covering the same (question_hops, bucket)) only retokenizes
    qids that are new or whose content actually changed -- e.g. growing a
    bucket's sample count only tokenizes the new qids, not the whole bucket.
    Tokenizers are loaded lazily, only if at least one qid needs it.
    """
    import build_full_prompts
    from .artifacts import read_json_cache, write_json_cache

    if needle_source_questions.empty or prompt_packing_configurations.empty:
        print("Skipping needle token measurements until both source needles and assembled prompts exist.")
        return pd.DataFrame(), pd.DataFrame()

    packing_columns = [
        "target_context_length",
        "tokenizer_model_slugs",
        "question_last",
        "context_item_prefix",
        "context_item_separator",
    ]
    unique_packing_configurations = prompt_packing_configurations[packing_columns].drop_duplicates()
    if len(unique_packing_configurations) != 1:
        raise ValueError(
            "Expected one prompt-packing configuration across selected hops, "
            f"found {len(unique_packing_configurations)}."
        )

    prompt_configuration = unique_packing_configurations.iloc[0]
    target_context_token_count = int(prompt_configuration["target_context_length"])
    metadata_tokenizer_slugs = list(prompt_configuration["tokenizer_model_slugs"])
    question_last = bool(prompt_configuration["question_last"])
    context_item_prefix = str(prompt_configuration["context_item_prefix"])
    context_item_separator = str(prompt_configuration["context_item_separator"])
    packing_config_hash = stable_content_hash(
        {
            "target_context_length": target_context_token_count,
            "tokenizer_model_slugs": metadata_tokenizer_slugs,
            "question_last": question_last,
            "context_item_prefix": context_item_prefix,
            "context_item_separator": context_item_separator,
        }
    )

    packing_tokenizers: list[tuple[str, object]] | None = None

    def loaded_packing_tokenizers() -> list[tuple[str, object]]:
        nonlocal packing_tokenizers
        if packing_tokenizers is None:
            loaded_tokenizers = build_full_prompts.load_prompt_packing_tokenizers(target_context_token_count)
            tokenizers_by_slug = dict(loaded_tokenizers)
            missing_tokenizers = [
                slug for slug in metadata_tokenizer_slugs if slug not in tokenizers_by_slug
            ]
            if missing_tokenizers:
                raise ValueError(f"Could not load prompt-packing tokenizers: {missing_tokenizers}")
            packing_tokenizers = [(slug, tokenizers_by_slug[slug]) for slug in metadata_tokenizer_slugs]
        return packing_tokenizers

    needle_question_rows: list[dict] = []
    needle_length_rows: list[dict] = []
    cache_hits = 0
    cache_misses = 0

    for (question_hops, bucket_label), group in needle_source_questions.groupby(
        ["question_hops", "bucket"], sort=False
    ):
        cache_path = needle_token_cache_path(question_hops, bucket_label)
        cached = read_json_cache(cache_path)
        cached_entries = (
            cached.get("entries", {})
            if cached is not None and cached.get("packing_config_hash") == packing_config_hash
            else {}
        )

        updated_entries: dict[str, dict] = {}
        for row in group.to_dict("records"):
            qid = str(row["qid"])
            content_hash = needle_question_content_hash(row, answer_field)
            cached_entry = cached_entries.get(qid)
            if cached_entry is not None and cached_entry.get("content_hash") == content_hash:
                cache_hits += 1
                needle_question_row = cached_entry["needle_question_row"]
                needle_length_rows_for_qid = cached_entry["needle_length_rows"]
            else:
                cache_misses += 1
                needle_question_row, needle_length_rows_for_qid = tokenize_needle_question(
                    row,
                    loaded_packing_tokenizers(),
                    context_item_prefix,
                    context_item_separator,
                    question_last,
                    target_context_token_count,
                    answer_field,
                )
            updated_entries[qid] = {
                "content_hash": content_hash,
                "needle_question_row": needle_question_row,
                "needle_length_rows": needle_length_rows_for_qid,
            }
            needle_question_rows.append(needle_question_row)
            needle_length_rows.extend(needle_length_rows_for_qid)

        # Rewriting every group (even on a pure cache hit) keeps the file
        # pruned of qids that no longer exist in the source, at negligible
        # cost compared to tokenization.
        write_json_cache(
            cache_path, {"packing_config_hash": packing_config_hash, "entries": updated_entries}
        )

    if cache_hits or cache_misses:
        print(f"Needle token cache: {cache_hits} question(s) reused, {cache_misses} question(s) (re)tokenized.")

    needle_question_metrics = pd.DataFrame(needle_question_rows)
    if not needle_prompt_fill.empty:
        needle_question_metrics = needle_question_metrics.merge(
            needle_prompt_fill[["question_hops", "bucket", "qid", "distractors_used"]],
            on=["question_hops", "bucket", "qid"],
            how="left",
            validate="one_to_one",
        )
    else:
        needle_question_metrics["distractors_used"] = pd.NA
    needle_length_metrics = pd.DataFrame(needle_length_rows)
    return needle_question_metrics, needle_length_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Report: needle and prompt-packing statistics
# ─────────────────────────────────────────────────────────────────────────────


def report_prompt_packing_configuration(
    prompt_packing_configurations: pd.DataFrame,
    sub_tag: str,
    *,
    question_type_label: str,
    experiment_number: str,
) -> dict[str, object] | None:
    """Display the prompt-packing configuration as a single row.

    ``prompt_packing_configurations`` carries one row per question_hops value,
    but the configuration itself (tokenizers, context length, ...) never
    varies across them within a sub-experiment -- that invariant is exactly
    what ``compute_needle_question_metrics`` asserts. Drop the question_hops
    column before de-duplicating so the table collapses to one row instead of
    repeating the same values once per question_hops.
    """
    if prompt_packing_configurations.empty:
        print(f"[{sub_tag}] Skipping the prompt-packing configuration table.")
        return None

    config_columns = [
        column for column in prompt_packing_configurations.columns if column != "question_hops"
    ]
    unique_configs = prompt_packing_configurations[config_columns].drop_duplicates()
    if len(unique_configs) != 1:
        raise ValueError(
            "Expected one prompt-packing configuration across selected hops, "
            f"found {len(unique_configs)}."
        )

    prompt_configuration_table = unique_configs.reset_index(drop=True)
    prompt_configuration_table["tokenizer_model_slugs"] = prompt_configuration_table[
        "tokenizer_model_slugs"
    ].map(", ".join)
    prompt_configuration_table["context_item_prefix"] = prompt_configuration_table[
        "context_item_prefix"
    ].map(repr)
    prompt_configuration_table["context_item_separator"] = prompt_configuration_table[
        "context_item_separator"
    ].map(repr)
    prompt_configuration_table.index = ["Config"]
    artifacts = write_table_artifacts(
        prompt_configuration_table,
        CFG.tables_dir / f"sub_experiment={sub_tag}",
        "table_prompt_packing_configuration",
        caption=(
            f"Prompt-packing configuration used by the selected format-version {CFG.target_format_version} "
            f"{question_type_label}."
        ),
        label=f"tab:experiment_{experiment_number}_{sub_tag}_prompt_packing_configuration",
        table_environment="table*",
        bold_header=True,
    )
    display(prompt_configuration_table)
    print_table_path(artifacts)
    return artifacts


def report_needle_statistics(
    needle_question_metrics: pd.DataFrame,
    needle_length_metrics: pd.DataFrame,
    question_hops_values: list[str],
    bucket_labels: list[str],
    sub_tag: str,
    row_axis: str = "bucket",
    caption_context: str = "",
    *,
    count_column_label: str,
    count_column_fn: Callable[[pd.DataFrame], str],
    experiment_number: str,
) -> dict[str, object]:
    if needle_question_metrics.empty:
        print(f"[{sub_tag}] No needle statistics table written; no measurements were taken.")
        return {}

    row_name = "Question hops" if row_axis == "question_hops" else "Bucket"
    summary_rows = []
    missing_prompt_values: set[str] = set()
    for question_hops in question_hops_values:
        hop_question_metrics = needle_question_metrics[
            needle_question_metrics["question_hops"] == question_hops
        ]
        hop_needle_metrics = needle_length_metrics[
            needle_length_metrics["question_hops"] == question_hops
        ]
        for bucket_label in bucket_labels:
            bucket_questions = hop_question_metrics[hop_question_metrics["bucket"] == bucket_label]
            if bucket_questions.empty:
                continue
            bucket_needles = hop_needle_metrics[hop_needle_metrics["bucket"] == bucket_label]
            row_value = question_hops if row_axis == "question_hops" else bucket_label
            summary_rows.append(
                {
                    row_name: row_value,
                    "Sample count": len(bucket_questions),
                    count_column_label: count_column_fn(bucket_questions),
                    "Realized difficulty": format_mean_std(bucket_questions["realized_difficulty"]),
                    "Distractors used": format_mean_std(bucket_questions["distractors_used"]),
                    "Context tokens left after needles": format_mean_std(
                        bucket_questions["context_tokens_left_after_needles"]
                    ),
                    "Needles per question": format_mean_std(bucket_questions["needles_per_question"]),
                    "Visible papers per question": format_mean_std(
                        bucket_questions["visible_papers_per_question"]
                    ),
                    "Tokens per visible paper": format_mean_std(
                        bucket_questions["tokens_per_visible_paper"]
                    ),
                    "Needle length tokens": format_mean_std(bucket_needles["needle_length_tokens"]),
                }
            )
            if not bucket_questions["distractors_used"].notna().any():
                missing_prompt_values.add(str(row_value))

    if not summary_rows:
        print(f"[{sub_tag}] no needle statistics found, skipping")
        return {}
    if missing_prompt_values:
        print(f"[{sub_tag}] missing prompt output for {row_name.lower()}(s): " + ", ".join(sorted(missing_prompt_values)))

    needle_statistics_table = pd.DataFrame(summary_rows).set_index(row_name)
    tables_dir = sub_experiment_tables_dir(sub_tag)
    artifacts = write_table_artifacts(
        needle_statistics_table,
        tables_dir,
        "table_needle_statistics_by_search_bucket",
        caption=(
            f"Needle and prompt-packing statistics by {row_name.lower()} for "
            f"experiment {experiment_number}{caption_context}."
        ),
        label=sub_experiment_label(sub_tag, "needle_statistics", experiment_tag=experiment_number),
        table_environment="table*",
        bold_header=True,
    )
    print(f"[{sub_tag}]")
    display(needle_statistics_table)
    print_table_path(artifacts)

    return artifacts


def report_prompt_fill(
    question_hops_values: list[str],
    bucket_labels: list[str],
    distracting_content_type: str,
    context_length: str,
    sub_tag: str,
    row_axis: str = "bucket",
    caption_context: str = "",
    *,
    find_payload_path: Callable[[str, str, str, str], Path | None],
    experiment_number: str,
) -> dict[str, object]:
    row_name = "Question hops" if row_axis == "question_hops" else "Bucket"
    fill_rows = []
    for question_hops in question_hops_values:
        for bucket_label in bucket_labels:
            prompts_path = find_payload_path(
                question_hops, bucket_label, distracting_content_type, context_length
            )
            if prompts_path is None:
                continue
            summary = summarize_prompt_fill_payload(prompts_path)
            if summary is None:
                continue
            row_value = question_hops if row_axis == "question_hops" else bucket_label
            fill_rows.append({row_name: row_value, **summary})

    if not fill_rows:
        print(f"[{sub_tag}] no prompt payloads found, skipping fill table")
        return {}

    prompt_fill = pd.DataFrame(fill_rows).set_index(row_name).round(2)
    tables_dir = sub_experiment_tables_dir(sub_tag)
    artifacts = write_table_artifacts(
        prompt_fill,
        tables_dir,
        "table_prompt_fill_by_search_bucket",
        caption=(
            f"Assembled prompt size by {row_name.lower()} for experiment "
            f"{experiment_number}{caption_context}. Context length must be held constant across "
            "conditions for accuracy to be comparable."
        ),
        label=sub_experiment_label(sub_tag, "prompt_fill", experiment_tag=experiment_number),
        table_environment="table*",
        bold_header=True,
    )

    print(f"[{sub_tag}]")
    display(prompt_fill)
    print_table_path(artifacts)

    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Report: figures
# ─────────────────────────────────────────────────────────────────────────────


def report_family_figures(
    sub_results: pd.DataFrame,
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    column_values: list[str],
    sub_tag: str,
    axis_field: str = "bucket_label",
    axis_label: str = "Expected search-step bucket",
    title_prefix: str = "",
    *,
    metric_column: str,
    metric_label: str,
    family_stem_prefix: str,
    chance_accuracy_pct: float,
    metric_ci: dict,
) -> dict[str, object]:
    ignored_models: list[str] = []
    if sub_results.empty:
        print(f"[{sub_tag}] No per-family figures written; no eval reports matched.")
        return {}

    hop_results = as_bucket_axis(sub_results, axis_field)
    model_slugs = [slug for slug in ordered_model_slugs if slug in set(hop_results["model_slug"])]
    figures_dir = sub_experiment_figures_dir(sub_tag)
    chance_baseline = {value: chance_accuracy_pct for value in column_values}

    artifacts: dict[str, object] = {}
    for family_key, family_label in FAMILY_FILTERS:
        family_slugs = [
            slug
            for slug in model_slugs
            if model_slug_matches_family(slug, family_key) and slug not in ignored_models
        ]
        if not family_slugs:
            print(f"[{sub_tag}][{family_label}] no models with data, skipping")
            continue
        family_results = hop_results[hop_results["model_slug"].isin(family_slugs)]
        family_stem = f"{family_stem_prefix}_{family_key.replace('.', '_').replace('-', '_')}"
        family_artifacts = write_bucket_figure_artifacts(
            family_results,
            figures_dir,
            family_stem,
            bucket_labels=column_values,
            ordered_model_slugs=family_slugs,
            model_names=model_names,
            num_columns_span=2,
            title=f"{title_prefix}{family_label} — [{sub_tag}]",
            error_bar=True,
            metric_ci=metric_ci,
            value_column=metric_column,
            y_label=metric_label,
            x_axis_label=axis_label,
            baseline_by_bucket=chance_baseline,
            baseline_name=f"Chance ({chance_accuracy_pct:.0f}%)",
        )
        artifacts[family_key] = family_artifacts
        print(f"[{sub_tag}][{family_label}] models: {family_slugs}")
        print_figure_path(family_artifacts)
        display(Image(filename=str(family_artifacts["spans"][2]["png_path"])))

    return artifacts


def report_family_grid(
    sub_results: pd.DataFrame,
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    column_values: list[str],
    sub_tag: str,
    axis_field: str = "bucket_label",
    axis_label: str = "Expected search-step bucket",
    title: str = "Experiment",
) -> dict[str, object]:
    if sub_results.empty:
        print(f"[{sub_tag}] No unified family-grid figure written; no eval reports matched.")
        return {}

    hop_results = as_bucket_axis(sub_results, axis_field)
    figures_dir = sub_experiment_figures_dir(sub_tag)
    artifacts = write_bucket_family_grid_figure_artifacts(
        hop_results,
        figures_dir,
        "accuracy_by_search_bucket_family_grid",
        bucket_labels=column_values,
        ordered_model_slugs=ordered_model_slugs,
        model_names=model_names,
        title=f"{title} — [{sub_tag}]",
        shared_legend=False,
        error_bar=True,
        x_axis_label=axis_label,
    )
    print(f"[{sub_tag}]")
    print_figure_path(artifacts)
    display(Image(filename=str(artifacts["png_path"])))

    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Report: distractor-type comparison
# ─────────────────────────────────────────────────────────────────────────────


def format_distractor_type(distractor_type: str) -> str:
    """Return a readable label for a distractor-type slug."""
    return distractor_type.replace("-", " ").capitalize()


def report_distractor_comparison(
    sub_results: pd.DataFrame,
    preferred_model_order: list[str],
    question_hops_values: list[str],
    bucket_labels: list[str],
    reference_distractor_type: str,
    sub_tag: str,
    *,
    count_fields: tuple[str, ...],
    experiment_number: str,
) -> dict[str, dict[str, object]]:
    """Compare pooled accuracy across distractor types and context lengths for one sub-experiment.

    ``sub_results`` must already be filtered to this sub-experiment's rows
    (which distractor types, buckets, context lengths, and question-hops
    values are in scope). The reference distractor type is pooled across all
    context lengths into a single column; every other distractor type gets
    one column per context length actually observed, ordered by increasing
    token count. Columns with no data for any model/bucket are dropped.
    """
    import math

    from .artifacts import get_context_length_token_map, get_model_display_names

    artifacts_by_hop: dict[str, dict[str, object]] = {}
    if sub_results.empty:
        print(f"[{sub_tag}] No eval reports matched the distractor comparison sweep.")
        return artifacts_by_hop

    comparison_distractor_types = sorted(sub_results["distracting_content_type"].unique())
    other_distractor_types = [
        distractor_type
        for distractor_type in comparison_distractor_types
        if distractor_type != reference_distractor_type
    ]
    distractor_display_names = {
        distractor_type: format_distractor_type(distractor_type)
        for distractor_type in comparison_distractor_types
    }
    context_length_tokens = get_context_length_token_map()
    comparison_context_lengths = sorted(
        sub_results["context_length"].unique(),
        key=lambda context_length: context_length_tokens.get(context_length, math.inf),
    )
    # The reference type may have no eval reports yet; the per-hop loop below then skips every row.
    reference_column = f"{format_distractor_type(reference_distractor_type)} accuracy (%)"
    comparison_columns = [
        f"{distractor_display_names[distractor_type]} ({context_length}) accuracy (%)"
        for distractor_type in other_distractor_types
        for context_length in comparison_context_lengths
    ]
    print(
        f"[{sub_tag}] Discovered distractor types: {comparison_distractor_types}, "
        f"context lengths: {comparison_context_lengths}"
    )

    available_model_slugs = sorted(sub_results["model_slug"].unique())
    ordered_model_slugs = [slug for slug in preferred_model_order if slug in available_model_slugs]
    ordered_model_slugs += [slug for slug in available_model_slugs if slug not in ordered_model_slugs]
    model_names = get_model_display_names(ordered_model_slugs) if ordered_model_slugs else {}

    def format_accuracy(condition_results: pd.DataFrame) -> str | None:
        if condition_results.empty:
            return None
        counts = pool_counts(condition_results, count_fields)
        return format_bernoulli_ci_pct(counts["correct"], counts["total"])

    for question_hops in question_hops_values:
        hop_results = sub_results[sub_results["question_hops"] == question_hops].copy()
        if hop_results.empty:
            print(f"[{sub_tag}][question_hops={question_hops}] no matching eval reports, skipping comparison")
            continue

        observed_model_slugs = set(hop_results["model_slug"])
        hop_model_slugs = [slug for slug in ordered_model_slugs if slug in observed_model_slugs]
        rows = []
        for model_slug in hop_model_slugs:
            for bucket_label in get_bucket_columns(hop_results, bucket_labels):
                base_results = hop_results[
                    (hop_results["model_slug"] == model_slug) & (hop_results["bucket_label"] == bucket_label)
                ]
                row = {"Model / bucket": f"{model_names[model_slug]} / {bucket_label}"}

                reference_results = base_results[
                    base_results["distracting_content_type"] == reference_distractor_type
                ]
                reference_present = not reference_results.empty
                row[reference_column] = format_accuracy(reference_results)

                any_compared_present = False
                for distractor_type in other_distractor_types:
                    display_name = distractor_display_names[distractor_type]
                    for context_length in comparison_context_lengths:
                        column_name = f"{display_name} ({context_length}) accuracy (%)"
                        condition_results = base_results[
                            (base_results["distracting_content_type"] == distractor_type)
                            & (base_results["context_length"] == context_length)
                        ]
                        if condition_results.empty:
                            row[column_name] = None
                            continue
                        any_compared_present = True
                        row[column_name] = format_accuracy(condition_results)

                if not reference_present or not any_compared_present:
                    continue
                rows.append(row)

        if not rows:
            print(
                f"[{sub_tag}][question_hops={question_hops}] no model/bucket has both "
                "needles-only and a distractor condition, skipping table"
            )
            continue

        comparison_table = pd.DataFrame(rows, columns=["Model / bucket", reference_column, *comparison_columns])
        comparison_table = comparison_table.set_index("Model / bucket").round(2)
        comparison_table = comparison_table.dropna(axis=1, how="all")
        display_table = comparison_table.astype(object).where(comparison_table.notna(), "-")

        hop_tables_dir = sub_experiment_tables_dir(sub_tag, question_hops)
        artifacts = write_table_artifacts(
            display_table,
            hop_tables_dir,
            "table_distractor_type_comparison",
            caption=(
                "Accuracy (\\%) with 95\\% Wilson score confidence intervals for needles-only "
                "and all discovered distractor "
                f"conditions (by context length) in experiment {experiment_number} v6, "
                f"{question_hops} question hops. Entries are estimate [lower, upper]."
            ),
            label=sub_experiment_label(
                sub_tag, "distractor_comparison", question_hops, experiment_tag=experiment_number
            ),
            table_environment="table*",
            bold_header=True,
        )
        artifacts_by_hop[question_hops] = artifacts

        print(f"[{sub_tag}][question_hops={question_hops}] compared distractors: " + ", ".join(comparison_distractor_types))
        display(display_table)
        print_table_path(artifacts)

    return artifacts_by_hop
