"""Notebook helpers for the ``experiment_7_reachability_yesno.vN.multiple``
notebooks -- the converted-to-multiple-choice variant of experiment 7.

``scripts/experiments/experiment_7_reachability_yesno.vN.multiple/convert_data.sh``
doesn't generate fresh source data: it converts experiment 7's already-built
``prompts/full_prompts.json`` into a 2-option ``reachability-multiple``
framing (``num_options=1`` -- the target paper -- plus ``none_option=1``),
under its own path schema (``conversion=.../conversion_format_version=.../
source_question_format_version=...`` instead of a plain ``format_version=``
segment, and no ``source/`` tree at all -- there's nothing to run
construction-diagnostics or needle-statistics against).

Because ``eval.py`` is run with ``--question-type reachability-multiple``, it
produces the exact same ``option_metrics`` shape experiment 8 gets, so with
``CFG.option_keys = ["a", "b"]`` (target vs. none) once :func:`configure`
runs, almost all of experiment 8's option-based reporting
(:mod:`reachability_multiple`) applies unmodified and is re-exported below.
Only :func:`configure`, :func:`gather_results`, and
:func:`find_prompt_payload_path` need their own path-scheme-aware logic; the
three report functions that hardcode "experiment 8" in their captions/labels
get a thin wrapper here passing ``experiment_number="7.multiple"`` instead.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import reachability_multiple as _exp8
from .artifacts import load_json
from .reachability_multiple import (
    CFG,
    FIGURE_METRIC_CI,
    FIGURE_METRIC_VALUES,
    add_figure_metric_columns,
    count_fields,
    get_option_counts,
    get_prediction_counts,
    macro_ci_bounds_pct,
    macro_option_accuracy_pct,
    option_recall_pct,
    order_model_slugs,
    position_baseline_by_bucket,
    position_baseline_pct,
    prediction_entropy_ratio,
    report_bucket_figures,
    report_family_figures,
    report_family_grid,
)
from .reachability_reports import (
    get_average_reasoning_tokens,
    report_distractor_comparison as _report_distractor_comparison,
    select_preferred_artifacts,
)

# Segments that must be present for each artifact kind. No ``source/`` tree
# exists for this variant, so there is no ``SOURCE_SEGMENT_KEYS`` -- only
# prompts and eval outputs.
PROMPT_SEGMENT_KEYS = {
    "hops",
    "question_type",
    "conversion",
    "conversion_format_version",
    "source_question_format_version",
    "source_no_construction",
    "num_options",
    "none_option",
    "traversal_mode",
    "needle_type",
    "max_author_answer_index",
    "num_children_per_node",
    "question_hops",
    "reuse_growth_session_across_buckets",
    "cutoff_date",
    "bucket",
    "distracting_content_type",
    "needle_order",
    "context_length",
}
MODEL_OUTPUT_SEGMENT_KEYS = PROMPT_SEGMENT_KEYS | {"model"}


def configure(
    *,
    experiment_root: Path,
    tables_dir: Path,
    figures_dir: Path,
    target_input_hops: str,
    target_conversion: str,
    target_conversion_format_version: str,
    target_source_question_format_version: str,
    target_source_no_construction: str,
    target_num_options: str,
    target_none_option: str,
    target_max_author_answer_index: str,
    target_num_children_per_node: str,
    target_reuse_growth_session_across_buckets: str,
    target_cutoff_date: str,
    target_traversal_mode: str,
    target_needle_type: str,
    target_needle_order: str,
    target_alpha: str | None = None,
) -> None:
    """Set the fixed path-selector hyperparameters every function reads from ``CFG``.

    Mirrors :func:`reachability_multiple.configure`'s option-count bookkeeping
    (``option_keys``/``option_labels``/``chance_accuracy_pct``), since the
    downstream reporting functions reused from that module depend on it.
    ``target_alpha`` is only meaningful for v7+ source data (whose
    distractor-tree scaling exponent survives the yesno->multiple
    conversion); leave it ``None`` (the default) for v5/v6-sourced roots.
    """
    num_target_options = int(target_num_options)
    none_option = int(target_none_option)
    if none_option not in (0, 1):
        raise ValueError("target_none_option must be either 0 or 1.")
    num_options = num_target_options + none_option
    option_keys = [chr(ord("a") + index) for index in range(num_options)]
    CFG.__dict__.clear()
    CFG.__dict__.update(
        experiment_root=experiment_root,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        target_input_hops=target_input_hops,
        target_question_type="reachability-multiple",
        target_conversion=target_conversion,
        target_conversion_format_version=target_conversion_format_version,
        target_source_question_format_version=target_source_question_format_version,
        target_source_no_construction=target_source_no_construction,
        target_num_options=target_num_options,
        target_none_option=target_none_option,
        target_max_author_answer_index=target_max_author_answer_index,
        target_num_children_per_node=target_num_children_per_node,
        target_reuse_growth_session_across_buckets=target_reuse_growth_session_across_buckets,
        target_cutoff_date=target_cutoff_date,
        target_traversal_mode=target_traversal_mode,
        target_needle_type=target_needle_type,
        target_needle_order=target_needle_order,
        target_alpha=target_alpha,
        num_options=num_options,
        num_target_options=num_target_options,
        none_option=none_option,
        option_keys=option_keys,
        option_labels=[key.upper() for key in option_keys],
        chance_accuracy_pct=100.0 / num_options,
    )


def build_segment_filters(
    bucket_labels: list[str],
    distracting_content_types: list[str],
    question_hops_values: list[str],
    context_lengths: list[str],
) -> dict[str, set[str]]:
    """Return the ``key -> allowed values`` filter for one gather/scan pass."""
    return {
        "hops": {CFG.target_input_hops},
        "question_type": {CFG.target_question_type},
        "conversion": {CFG.target_conversion},
        "conversion_format_version": {CFG.target_conversion_format_version},
        "source_question_format_version": {CFG.target_source_question_format_version},
        "source_no_construction": {CFG.target_source_no_construction},
        "num_options": {CFG.target_num_options},
        "none_option": {CFG.target_none_option},
        "traversal_mode": {CFG.target_traversal_mode},
        "needle_type": {CFG.target_needle_type},
        "max_author_answer_index": {CFG.target_max_author_answer_index},
        "num_children_per_node": {CFG.target_num_children_per_node},
        "question_hops": set(question_hops_values),
        "reuse_growth_session_across_buckets": {
            CFG.target_reuse_growth_session_across_buckets
        },
        "cutoff_date": {CFG.target_cutoff_date},
        "bucket": set(bucket_labels),
        "distracting_content_type": set(distracting_content_types),
        "needle_order": {CFG.target_needle_order},
        "context_length": set(context_lengths),
        **(
            {"alpha": {CFG.target_alpha}}
            if getattr(CFG, "target_alpha", None) is not None
            else {}
        ),
    }


def find_prompt_payload_path(
    question_hops: str, bucket_label: str, distracting_content_type: str, context_length: str
) -> Path | None:
    """Locate the assembled (converted) prompt payload backing one evaluated bucket."""
    root = CFG.experiment_root / "prompts"
    segment_filters = build_segment_filters(
        [bucket_label], [distracting_content_type], [question_hops], [context_length]
    )
    matches = [
        path
        for path, values in select_preferred_artifacts(
            root, "full_prompts.json", PROMPT_SEGMENT_KEYS, segment_filters
        )
        if values["question_hops"] == question_hops and values["bucket"] == bucket_label
    ]
    return matches[0] if matches else None


def gather_results(
    bucket_ranges: dict[str, tuple[int, int]],
    bucket_sort_keys: dict[str, int],
    question_hops_values: list[str],
    distracting_content_types: list[str],
    context_lengths: list[str],
) -> pd.DataFrame:
    """Gather one eval report per condition, across every sub-experiment."""
    segment_filters = build_segment_filters(
        list(bucket_ranges), distracting_content_types, question_hops_values, context_lengths
    )
    records = []
    for eval_path, values in select_preferred_artifacts(
        CFG.experiment_root / "eval", "eval.json", MODEL_OUTPUT_SEGMENT_KEYS, segment_filters
    ):
        bucket_label = values["bucket"]
        bucket_min, bucket_max = bucket_ranges[bucket_label]
        report = load_json(eval_path)
        option_counts = get_option_counts(report)
        prediction_counts = get_prediction_counts(report)

        record = {
            "model_slug": values["model"],
            "conversion": values["conversion"],
            "source_question_format_version": values["source_question_format_version"],
            "question_hops": values["question_hops"],
            "num_options": values["num_options"],
            "none_option": values["none_option"],
            "displayed_num_options": CFG.num_options,
            "bucket_label": bucket_label,
            "bucket_min": bucket_min,
            "bucket_max": bucket_max,
            "bucket_sort_key": bucket_sort_keys[bucket_label],
            "bucket_interval": f"[{bucket_min}, {bucket_max})",
            "context_length": values["context_length"],
            "traversal_mode": values["traversal_mode"],
            "needle_type": values["needle_type"],
            "distracting_content_type": values["distracting_content_type"],
            "needle_order": values["needle_order"],
            "alpha": values.get("alpha", ""),
            "accuracy_pct": 100.0 * float(report["accuracy"]),
            "correct": report.get("correct"),
            "total": report.get("total"),
            "wrong_answer": report.get("wrong_answer"),
            "no_answer_box": report.get("no_answer_box"),
            "unparseable": report.get("unparseable"),
            "format_mismatch": report.get("format_mismatch"),
            "format_mismatch_rate": report.get("format_mismatch_rate"),
            "reasoning_tokens": get_average_reasoning_tokens(report),
            "eval_path": str(eval_path),
        }
        for key in CFG.option_keys:
            record[f"option_{key}_total"] = option_counts[key]["total"]
            record[f"option_{key}_correct"] = option_counts[key]["correct"]
            record[f"predicted_{key}"] = prediction_counts[key]
        record["predicted_unparsed"] = prediction_counts["unparsed"]
        records.append(record)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Report: accuracy tables (thin wrappers -- fix the "experiment 8" captions)
# ─────────────────────────────────────────────────────────────────────────────


def report_accuracy_by_bucket(
    sub_results: pd.DataFrame,
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    column_values: list[str],
    sub_tag: str,
    axis_field: str = "bucket_label",
    axis_label: str = "Expected search-step bucket",
    caption_context: str = "",
) -> dict[str, object]:
    return _exp8.report_accuracy_by_bucket(
        sub_results, ordered_model_slugs, model_names, column_values, sub_tag,
        axis_field=axis_field, axis_label=axis_label, caption_context=caption_context,
        experiment_number="7.multiple",
        include_none_prediction_rate=False,
    )


def report_answer_distribution(
    sub_results: pd.DataFrame,
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    sub_tag: str,
    axis_field: str = "bucket_label",
    axis_label: str = "Expected search-step bucket",
    group_field: str | None = None,
    group_values: list[str] | None = None,
    group_label: str = "Group",
    caption_context: str = "",
) -> dict[str, object]:
    return _exp8.report_answer_distribution(
        sub_results, ordered_model_slugs, model_names, sub_tag,
        axis_field=axis_field, axis_label=axis_label, group_field=group_field,
        group_values=group_values, group_label=group_label, caption_context=caption_context,
        experiment_number="7.multiple",
    )


def report_macro_accuracy_by_bucket(
    sub_results: pd.DataFrame,
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    column_values: list[str],
    sub_tag: str,
    axis_field: str = "bucket_label",
    axis_label: str = "Expected search-step bucket",
    caption_context: str = "",
) -> dict[str, object]:
    return _exp8.report_macro_accuracy_by_bucket(
        sub_results, ordered_model_slugs, model_names, column_values, sub_tag,
        axis_field=axis_field, axis_label=axis_label, caption_context=caption_context,
        experiment_number="7.multiple",
    )


def report_distractor_comparison(
    sub_results: pd.DataFrame,
    preferred_model_order: list[str],
    question_hops_values: list[str],
    bucket_labels: list[str],
    reference_distractor_type: str,
    sub_tag: str,
) -> dict[str, dict[str, object]]:
    return _report_distractor_comparison(
        sub_results, preferred_model_order, question_hops_values, bucket_labels,
        reference_distractor_type, sub_tag,
        count_fields=count_fields(),
        experiment_number="7.multiple",
    )
