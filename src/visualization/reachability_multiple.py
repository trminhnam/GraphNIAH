"""Notebook helpers for ``experiment_8_reachability_multiple.v6.ipynb``.

Everything specific to experiment 8 (reachability multiple choice) -- path
parsing, correct-option-position metric helpers, construction diagnostics,
and figure specs -- lives here. Logic shared with experiment 7 (reachability
yes/no) lives in ``reachability_reports.py`` instead; the functions below
that only differ from their experiment-7 counterpart by a few fixed values
are thin wrappers around the shared versions, so this module's public
surface is unchanged from before the split.

The notebook calls :func:`configure` once (from its hyperparameter cell) with
every fixed path-selector hyperparameter, then :func:`gather_results` once to
build the master results table. Each sub-experiment section filters that
table with ``filter_results_by_subexperiment`` and passes the filtered rows
plus its own ``question_hops``/``bucket_label`` axis values into the
``report_*`` functions below.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

from .artifacts import (
    format_bernoulli_ci_pct,
    format_ci_pct,
    load_json,
    mean_ci_bounds_pct,
    print_figure_path,
    print_table_path,
    wilson_ci_bounds_pct,
    write_bucket_figure_artifacts,
    write_spanner_table_artifacts,
    write_table_artifacts,
)
from .reachability_reports import (
    CFG,
    FAMILY_FILTERS,
    add_figure_metric_columns as _add_figure_metric_columns,
    as_bucket_axis,
    build_bucket_ci_matrix,
    compute_needle_question_metrics as _compute_needle_question_metrics,
    get_average_reasoning_tokens,
    get_bucket_columns,
    get_cell_counts,
    get_hop_model_slugs,
    mean_reasoning_tokens,
    order_model_slugs,
    pool_counts,
    report_distractor_comparison as _report_distractor_comparison,
    report_family_figures as _report_family_figures,
    report_family_grid as _report_family_grid,
    report_needle_statistics as _report_needle_statistics,
    report_prompt_fill as _report_prompt_fill,
    report_prompt_packing_configuration as _report_prompt_packing_configuration,
    select_preferred_artifacts as _select_preferred_artifacts,
    sub_experiment_figures_dir,
    sub_experiment_label,
    sub_experiment_tables_dir,
)
from src.questions.reachability_multiple import (
    NONE_OPTION_TEXT,
    graph_invariant_holds as multiple_choice_graph_invariant_holds,
)
from src.questions.reachability_yesno import visible_graph_from_metadata

# Segments that must be present for each artifact kind.
SOURCE_SEGMENT_KEYS = {
    "hops",
    "question_type",
    "format_version",
    "construction",
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
}
PROMPT_SEGMENT_KEYS = SOURCE_SEGMENT_KEYS | {
    "distracting_content_type",
    "needle_order",
    "context_length",
}
MODEL_OUTPUT_SEGMENT_KEYS = PROMPT_SEGMENT_KEYS | {"model"}
EXPANSION_SEGMENT_KEYS = {"branch_expansion_depth", "max_expanded_children_per_paper"}


def configure(
    *,
    experiment_root: Path,
    tables_dir: Path,
    figures_dir: Path,
    target_input_hops: str,
    target_question_type: str,
    target_format_version: str,
    target_construction: str,
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
    target_theta: str | None = None,
    target_discount: str | None = None,
    target_none_option_conversion_mode: str | None = None,
    target_convert_num_options: str | None = None,
) -> None:
    """Set the fixed path-selector hyperparameters every function reads from ``CFG``.

    ``target_alpha`` is only meaningful for v7+ experiment roots, which sweep
    a distractor-tree scaling exponent absent from earlier versions; leave it
    ``None`` (the default) for any experiment root that doesn't have it.
    ``target_theta``/``target_discount`` work the same way for v10 roots,
    which sweep the CRP tree-expansion parameters.
    ``target_none_option_conversion_mode`` optionally selects one converted
    None-option result set without changing existing experiment notebooks.
    ``target_convert_num_options`` optionally selects one reduced-option set.
    """
    num_target_options = int(target_num_options)
    none_option = int(target_none_option)
    if none_option not in (0, 1):
        raise ValueError("target_none_option must be either 0 or 1.")
    num_options = num_target_options + none_option
    # eval.py keys option_metrics by the lowercase canonical letter.
    option_keys = [chr(ord("a") + index) for index in range(num_options)]
    CFG.__dict__.clear()
    CFG.__dict__.update(
        experiment_root=experiment_root,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        target_input_hops=target_input_hops,
        target_question_type=target_question_type,
        target_format_version=target_format_version,
        target_construction=target_construction,
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
        target_theta=target_theta,
        target_discount=target_discount,
        target_none_option_conversion_mode=target_none_option_conversion_mode,
        target_convert_num_options=target_convert_num_options,
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
    """Return the ``key -> allowed values`` filter for one gather/scan pass.

    Every axis besides ``bucket``/``distracting_content_type``/``question_hops``/
    ``context_length`` is fixed to the configured ``CFG.target_*`` value; those
    four are passed in explicitly so callers can broaden them across every
    sub-experiment (the master gather) or narrow them to one sub-experiment (a
    comparison scan).
    """
    return {
        "hops": {CFG.target_input_hops},
        "question_type": {CFG.target_question_type},
        "format_version": {CFG.target_format_version},
        "construction": {CFG.target_construction},
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
        **{
            key: {getattr(CFG, f"target_{key}")}
            for key in ("theta", "discount")
            if getattr(CFG, f"target_{key}", None) is not None
        },
        **(
            {"none_option_conversion_mode": {CFG.target_none_option_conversion_mode}}
            if getattr(CFG, "target_none_option_conversion_mode", None) is not None
            else {}
        ),
        **(
            {"convert_num_options": {CFG.target_convert_num_options}}
            if getattr(CFG, "target_convert_num_options", None) is not None
            else {}
        ),
    }


def select_preferred_artifacts(
    root: Path, filename: str, required_keys: set[str], segment_filters: dict[str, set[str]]
) -> list[tuple[Path, dict[str, str]]]:
    """Return one artifact per condition using the scripts' fallback order."""
    return _select_preferred_artifacts(
        root, filename, required_keys, segment_filters, expansion_segment_keys=EXPANSION_SEGMENT_KEYS
    )


def get_option_counts(report: dict) -> dict[str, dict[str, int]]:
    """Return per-correct-option-position scored counts for one eval report."""
    counts = {key: {"total": 0, "correct": 0} for key in CFG.option_keys}
    metrics = report.get("option_metrics")
    if isinstance(metrics, dict):
        for key in CFG.option_keys:
            option_metrics = metrics.get(key)
            if not isinstance(option_metrics, dict):
                continue
            counts[key]["total"] = int(option_metrics.get("total") or 0)
            counts[key]["correct"] = int(option_metrics.get("correct") or 0)
        return counts

    # Backward compatibility for eval files written before option_metrics.
    for detail in report.get("details") or []:
        canonical_values = detail.get("ground_truth_canonical") or []
        if isinstance(canonical_values, str):
            canonical_values = [canonical_values]
        key = next(
            (
                str(value).casefold()
                for value in canonical_values
                if str(value).casefold() in counts
            ),
            None,
        )
        if key is None:
            continue
        counts[key]["total"] += 1
        if detail.get("correct") is True:
            counts[key]["correct"] += 1
    return counts


def get_prediction_counts(report: dict) -> dict[str, int]:
    """Return how often the model answered each option position.

    Reveals position bias, which pooled accuracy hides: a model that always
    answers ``A`` scores whatever share of items happen to have ``A`` correct.
    ``unparsed`` counts responses from which no option letter could be read.
    """
    counts = {key: 0 for key in CFG.option_keys}
    counts["unparsed"] = 0
    for detail in report.get("details") or []:
        option = str(detail.get("extracted_option") or "").casefold()
        if option in counts and option != "unparsed":
            counts[option] += 1
        else:
            counts["unparsed"] += 1
    return counts


def resolve_report_artifact_path(path_value: str) -> Path:
    """Resolve an artifact path stored in an eval report."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    project_root = CFG.experiment_root.parents[2]
    return project_root / path


def get_none_option_label(question: dict) -> str | None:
    """Return the sample-specific label for the None option."""
    conversion = question.get("none_option_conversion") or {}
    if isinstance(conversion, dict) and conversion.get("none_option"):
        return str(conversion["none_option"]).strip().casefold()

    multiple_choice = question.get("multiple_choice") or {}
    if isinstance(multiple_choice, dict):
        for option in multiple_choice.get("options") or []:
            if isinstance(option, dict) and option.get("option_kind") == "none":
                return str(option.get("label") or "").strip().casefold() or None

    rendered_question = str(question.get("user_message") or question.get("question") or "")
    match = re.search(
        rf"(?m)^\s*([A-Z])\s*[.)]\s*{re.escape(NONE_OPTION_TEXT)}\s*$",
        rendered_question,
    )
    return match.group(1).casefold() if match else None


@lru_cache(maxsize=None)
def load_none_option_labels(groundtruth_path: Path) -> dict[str, str | None]:
    """Load the per-sample None labels once for all models sharing a prompt file."""
    questions = load_json(groundtruth_path).get("questions") or []
    return {
        str(question.get("qid")): get_none_option_label(question)
        for question in questions
        if question.get("qid") is not None
    }


def get_none_prediction_rate(report: dict) -> float | None:
    """Return the share of all samples where the model selected None."""
    if not getattr(CFG, "none_option", 0):
        return None

    groundtruth_path_value = report.get("groundtruth_path")
    if not groundtruth_path_value:
        raise ValueError("Cannot calculate None prediction rate without groundtruth_path.")
    groundtruth_path = resolve_report_artifact_path(str(groundtruth_path_value))
    none_labels = load_none_option_labels(groundtruth_path)

    details = report.get("details") or []
    if not details:
        return None
    missing_qids = [
        str(detail.get("qid"))
        for detail in details
        if not none_labels.get(str(detail.get("qid")))
    ]
    if missing_qids:
        raise ValueError(
            f"Could not identify the None option for {len(missing_qids)} sample(s) "
            f"in {groundtruth_path}."
        )

    none_predictions = sum(
        str(detail.get("extracted_option") or "").strip().casefold()
        == none_labels[str(detail.get("qid"))]
        for detail in details
    )
    return none_predictions / len(details)


def mean_none_prediction_rate(frame: pd.DataFrame) -> float | None:
    """Return the sample-weighted None prediction rate across result rows."""
    if frame.empty or "none_prediction_rate" not in frame:
        return None
    values = pd.to_numeric(frame["none_prediction_rate"], errors="coerce")
    weights = pd.to_numeric(frame["total"], errors="coerce")
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return None
    return float((values[valid] * weights[valid]).sum() / weights[valid].sum())


def gather_results(
    bucket_ranges: dict[str, tuple[int, int]],
    bucket_sort_keys: dict[str, int],
    question_hops_values: list[str],
    distracting_content_types: list[str],
    context_lengths: list[str],
) -> pd.DataFrame:
    """Gather one eval report per condition, across every sub-experiment.

    ``bucket_ranges``/``question_hops_values``/``distracting_content_types``/
    ``context_lengths`` should cover the union used by any sub-experiment, so
    a sub-experiment can later select its own rows with
    ``filter_results_by_subexperiment`` instead of re-scanning disk.
    ``theta`` and ``discount`` are not filtered here for the same reason, so
    each sub-experiment must select its own values of them.
    """
    segment_filters = build_segment_filters(
        list(bucket_ranges), distracting_content_types, question_hops_values, context_lengths
    )
    for key in ("theta", "discount"):  # not fixed here -- every value is kept
        segment_filters.pop(key, None)
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
            "format_version": values["format_version"],
            "question_hops": values["question_hops"],
            "construction": values["construction"],
            "num_options": values["num_options"],
            "none_option": values["none_option"],
            "none_option_conversion_mode": values.get(
                "none_option_conversion_mode", ""
            ),
            "convert_num_options": values.get("convert_num_options", ""),
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
            "theta": values.get("theta", ""),
            "discount": values.get("discount", ""),
            "accuracy_pct": 100.0 * float(report["accuracy"]),
            "correct": report.get("correct"),
            "total": report.get("total"),
            "wrong_answer": report.get("wrong_answer"),
            "no_answer_box": report.get("no_answer_box"),
            "unparseable": report.get("unparseable"),
            # eval.py records these only for --question-type reachability-multiple.
            "format_mismatch": report.get("format_mismatch"),
            "format_mismatch_rate": report.get("format_mismatch_rate"),
            "none_prediction_rate": get_none_prediction_rate(report),
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


def count_fields() -> tuple[str, ...]:
    return (
        "total",
        "correct",
        *(f"option_{key}_total" for key in CFG.option_keys),
        *(f"option_{key}_correct" for key in CFG.option_keys),
        *(f"predicted_{key}" for key in CFG.option_keys),
        "predicted_unparsed",
    )


def add_figure_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the position-robust metric columns used by the bucket figures."""
    return _add_figure_metric_columns(frame, FIGURE_METRIC_VALUES)


# ─────────────────────────────────────────────────────────────────────────────
# Shared metric helpers (correct-option-position split)
# ─────────────────────────────────────────────────────────────────────────────


def option_recall_pct(counts: dict[str, int], key: str) -> float | None:
    """Return accuracy on items whose correct answer sits at one position."""
    option_total = counts[f"option_{key}_total"]
    if not option_total:
        return None
    return 100.0 * counts[f"option_{key}_correct"] / option_total


def macro_option_accuracy_pct(counts: dict[str, int]) -> float | None:
    """Return the unweighted mean accuracy across correct-option positions.

    Insensitive to how the correct answer is distributed over positions, so a
    model that favours one letter cannot be flattered by a position skew.
    Undefined unless every position is represented.
    """
    recalls = [option_recall_pct(counts, key) for key in CFG.option_keys]
    if any(recall is None for recall in recalls):
        return None
    return sum(recalls) / len(recalls)


def position_baseline_pct(counts: dict[str, int]) -> float | None:
    """Return the accuracy of always answering the most frequent position.

    The constructor balances positions, so this sits just above chance; it is
    reported to make the "constant answer" floor explicit rather than assumed.
    """
    option_totals = [counts[f"option_{key}_total"] for key in CFG.option_keys]
    labelled_total = sum(option_totals)
    if not labelled_total:
        return None
    return 100.0 * max(option_totals) / labelled_total


def prediction_entropy_ratio(counts: dict[str, int]) -> float | None:
    """Return the model's answer-distribution entropy over positions, 0-1.

    ``1.0`` means the model spread its answers uniformly across the option
    letters; ``0.0`` means it answered the same letter every time. Unparsed
    responses are excluded so formatting failures do not read as balance.
    """
    import math

    predicted = [counts[f"predicted_{key}"] for key in CFG.option_keys]
    predicted_total = sum(predicted)
    if not predicted_total or CFG.num_options < 2:
        return None
    entropy = 0.0
    for value in predicted:
        if value:
            share = value / predicted_total
            entropy -= share * math.log(share)
    return entropy / math.log(CFG.num_options)


def macro_ci_bounds_pct(counts: dict[str, int]) -> tuple[float, float]:
    """Return the 95% CI bounds for macro accuracy, or ``(nan, nan)``.

    Macro accuracy averages ``NUM_OPTIONS`` independent per-position accuracies,
    so their Wilson intervals are combined with :func:`mean_ci_bounds_pct` (MOVER).
    """
    return mean_ci_bounds_pct(
        [option_recall_pct(counts, key) for key in CFG.option_keys],
        [
            wilson_ci_bounds_pct(
                counts[f"option_{key}_correct"], counts[f"option_{key}_total"]
            )
            for key in CFG.option_keys
        ],
    )


FIGURE_METRIC_CI = {
    "accuracy_pct": lambda counts: wilson_ci_bounds_pct(counts["correct"], counts["total"]),
    "macro_option_accuracy_pct": macro_ci_bounds_pct,
}

FIGURE_METRIC_VALUES = {
    "macro_option_accuracy_pct": macro_option_accuracy_pct,
    "prediction_entropy_ratio": prediction_entropy_ratio,
}


def position_baseline_by_bucket(
    hop_results: pd.DataFrame, bucket_labels: list[str]
) -> dict[str, float]:
    """Return the always-most-frequent-position baseline for each bucket.

    Every model scores the same prompt file, so the position mix is a property
    of the bucket; take the widest per-model view in case a run was partial.
    """
    baselines = {}
    total_columns = [f"option_{key}_total" for key in CFG.option_keys]
    for bucket_label in get_bucket_columns(hop_results, bucket_labels):
        bucket_rows = hop_results[hop_results["bucket_label"] == bucket_label]
        per_model = bucket_rows.groupby("model_slug")[total_columns].sum()
        if per_model.empty:
            continue
        option_totals = [int(per_model[column].max()) for column in total_columns]
        labelled_total = sum(option_totals)
        if labelled_total:
            baselines[bucket_label] = 100.0 * max(option_totals) / labelled_total
    return baselines


# ─────────────────────────────────────────────────────────────────────────────
# Report: accuracy tables
# ─────────────────────────────────────────────────────────────────────────────


def compute_position_composition(
    hop_results: pd.DataFrame, axis_values: list[str]
) -> dict[str, dict[str, float]]:
    """Return the correct-option-position composition per axis value, plus "Overall".

    Every model scores the same prompt file, so the position mix is a
    property of the axis value (or of the whole sweep, for "Overall"), not of
    any one model; take the widest per-model view in case a run was partial.
    """
    total_columns = [f"option_{key}_total" for key in CFG.option_keys]
    composition: dict[str, dict[str, float]] = {}
    for axis_value in [*axis_values, None]:
        subset = (
            hop_results if axis_value is None else hop_results[hop_results["bucket_label"] == axis_value]
        )
        per_model = subset.groupby("model_slug")[total_columns].sum()
        if per_model.empty:
            continue
        option_totals = [int(per_model[column].max()) for column in total_columns]
        labelled_total = sum(option_totals)
        if not labelled_total:
            continue
        key = "Overall" if axis_value is None else axis_value
        composition[key] = {"max_position_share_pct": 100.0 * max(option_totals) / labelled_total}
    return composition


def report_accuracy_by_bucket(
    sub_results: pd.DataFrame,
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    column_values: list[str],
    sub_tag: str,
    axis_field: str = "bucket_label",
    axis_label: str = "Expected search-step bucket",
    caption_context: str = "",
    experiment_number: str = "8",
    include_none_prediction_rate: bool = True,
) -> dict[str, object]:
    """Accuracy, correct-option-position composition, and per-position recall.

    One row per (axis value, model), plus an "Overall" axis value per model
    pooling every value in scope. Merges what used to be three separate
    tables (a bucket-columns accuracy spanner, a pooled option-position table,
    and a position-balance table) and replaces the standalone overall-accuracy
    summary.
    """
    if sub_results.empty:
        print(f"[{sub_tag}] No eval reports matched this sub-experiment's sweep yet.")
        return {}

    hop_results = as_bucket_axis(sub_results, axis_field)
    model_slugs = get_hop_model_slugs(hop_results, ordered_model_slugs)
    axis_values = get_bucket_columns(hop_results, column_values)
    composition = compute_position_composition(hop_results, axis_values)

    def counts_to_row(
        axis_value: str,
        model_slug: str,
        counts: dict[str, int],
        result_rows: pd.DataFrame,
    ) -> dict[str, object]:
        comp = composition.get(axis_value)
        row: dict[str, object] = {
            axis_label: axis_value,
            "Model": model_names[model_slug],
            "Total questions": counts["total"],
        }
        for key, label in zip(CFG.option_keys, CFG.option_labels):
            row[f"{label} questions"] = counts[f"option_{key}_total"]
        row["Max position share (%)"] = comp["max_position_share_pct"] if comp else None
        for key, label in zip(CFG.option_keys, CFG.option_labels):
            row[f"{label} recall, % [95% CI]"] = format_ci_pct(
                option_recall_pct(counts, key),
                wilson_ci_bounds_pct(
                    counts[f"option_{key}_correct"], counts[f"option_{key}_total"]
                ),
            )
        row["Macro accuracy, % [95% CI]"] = format_ci_pct(
            macro_option_accuracy_pct(counts), macro_ci_bounds_pct(counts)
        )
        row["Total accuracy, % [95% CI]"] = format_bernoulli_ci_pct(
            counts["correct"], counts["total"]
        )
        if include_none_prediction_rate:
            none_prediction_rate = mean_none_prediction_rate(result_rows)
            row["None Prediction Rate"] = (
                none_prediction_rate if none_prediction_rate is not None else "N/A"
            )
        row["Reasoning tokens"] = mean_reasoning_tokens(result_rows)
        return row

    rows = []
    for axis_value in axis_values:
        for model_slug in model_slugs:
            counts = get_cell_counts(hop_results, model_slug, axis_value, count_fields())
            if counts is None:
                continue
            result_rows = hop_results[
                (hop_results["model_slug"] == model_slug)
                & (hop_results["bucket_label"] == axis_value)
            ]
            rows.append(counts_to_row(axis_value, model_slug, counts, result_rows))

    for model_slug in model_slugs:
        model_results = hop_results[hop_results["model_slug"] == model_slug]
        if model_results.empty:
            continue
        rows.append(
            counts_to_row(
                "Overall",
                model_slug,
                pool_counts(model_results, count_fields()),
                model_results,
            )
        )

    table = pd.DataFrame(rows).set_index([axis_label, "Model"]).round(2)
    tables_dir = sub_experiment_tables_dir(sub_tag)
    tables_dir.mkdir(parents=True, exist_ok=True)
    sub_results.to_csv(tables_dir / "accuracy_long.csv", index=False)

    artifacts = write_table_artifacts(
        table,
        tables_dir,
        "table_accuracy_by_search_bucket",
        caption=(
            "Accuracy, correct-option-position composition, and per-position recall by "
            f"{axis_label.lower()} for experiment {experiment_number} (reachability multiple choice, "
            f"{CFG.num_options} options){caption_context}, plus an Overall row pooling every "
            f"value in scope. Chance is {CFG.chance_accuracy_pct:.0f}\\%; Max position share is "
            "the accuracy of always answering the most frequent option letter at that value; "
            "Accuracy and recall entries are estimate [lower, upper] using 95\\% Wilson "
            "score confidence intervals; macro accuracy combines the per-position Wilson "
            "intervals (MOVER). None Prediction Rate is the 0--1 share of all samples for "
            "which the model selected the sample-specific "
            "None option, and is N/A when the setting has no None option. Reasoning tokens is "
            "the mean per response; older artifacts use output tokens, which include hidden "
            "reasoning tokens."
        ),
        label=sub_experiment_label(sub_tag, "search_bucket_accuracy", experiment_tag=experiment_number),
        table_environment="table*",
        bold_header=True,
    )

    print(f"[{sub_tag}]")
    display(table)
    print_table_path(artifacts)

    return artifacts


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
    experiment_number: str = "8",
) -> dict[str, object]:
    """Distribution of predicted option letters, pooled over axis values.

    Unlike ``report_accuracy_by_bucket`` this is about what the model
    predicted, not the ground-truth composition, so it is kept separate.
    """
    if sub_results.empty:
        print(f"[{sub_tag}] No answer-distribution table written; no eval reports matched.")
        return {}

    hop_results = as_bucket_axis(sub_results, axis_field)
    hop_model_slugs = get_hop_model_slugs(hop_results, ordered_model_slugs)
    tables_dir = sub_experiment_tables_dir(sub_tag)

    groups = group_values if group_field is not None else [None]
    distribution_rows = []
    for group_value in groups:
        group_results = (
            hop_results if group_field is None else hop_results[hop_results[group_field] == group_value]
        )
        for model_slug in hop_model_slugs:
            model_results = group_results[group_results["model_slug"] == model_slug]
            counts = pool_counts(model_results, count_fields())
            answered = sum(counts[f"predicted_{key}"] for key in CFG.option_keys)
            answered_total = answered + counts["predicted_unparsed"]
            if not answered_total:
                continue
            row = {}
            if group_field is not None:
                row[group_label] = group_value
            row["Model"] = model_names[model_slug]
            row["Responses"] = answered_total
            for key, label in zip(CFG.option_keys, CFG.option_labels):
                row[f"{label} (%)"] = 100.0 * counts[f"predicted_{key}"] / answered_total
            row["Unparsed (%)"] = 100.0 * counts["predicted_unparsed"] / answered_total
            row["Answer entropy (0-1)"] = prediction_entropy_ratio(counts)
            distribution_rows.append(row)

    if not distribution_rows:
        print(f"[{sub_tag}] no answer-distribution rows found, skipping")
        return {}

    index_cols = [group_label, "Model"] if group_field is not None else ["Model"]
    answer_distribution = pd.DataFrame(distribution_rows).set_index(index_cols).round(2)
    artifacts = write_table_artifacts(
        answer_distribution,
        tables_dir,
        "table_answer_distribution_by_option",
        caption=(
            f"Distribution of predicted option letters for experiment {experiment_number}{caption_context}, "
            f"pooled over {axis_label.lower()} values. An entropy ratio near "
            "0 means the model answered nearly the same letter every "
            "time; 1 means it spread answers uniformly."
        ),
        label=sub_experiment_label(sub_tag, "answer_distribution", experiment_tag=experiment_number),
        table_environment="table*",
        bold_header=True,
    )

    print(f"[{sub_tag}] predicted answer distribution")
    display(answer_distribution)
    print_table_path(artifacts)

    return artifacts


def report_macro_accuracy_by_bucket(
    sub_results: pd.DataFrame,
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    column_values: list[str],
    sub_tag: str,
    axis_field: str = "bucket_label",
    axis_label: str = "Expected search-step bucket",
    caption_context: str = "",
    experiment_number: str = "8",
) -> dict[str, object]:
    """Macro accuracy (unweighted mean over option positions) as a Model x axis matrix.

    Kept as its own compact wide table (unlike ``report_accuracy_by_bucket``)
    since it is a single derived metric, not raw composition/count detail.
    """
    if sub_results.empty:
        print(f"[{sub_tag}] No macro-accuracy table written; no eval reports matched.")
        return {}

    hop_results = as_bucket_axis(sub_results, axis_field)
    hop_model_slugs = get_hop_model_slugs(hop_results, ordered_model_slugs)
    table_columns = get_bucket_columns(hop_results, column_values)
    tables_dir = sub_experiment_tables_dir(sub_tag)

    table, numeric_table = build_bucket_ci_matrix(
        hop_results,
        macro_option_accuracy_pct,
        macro_ci_bounds_pct,
        hop_model_slugs,
        table_columns,
        model_names,
        count_fields(),
    )
    table["Avg"] = numeric_table[table_columns].mean(axis=1, skipna=True).round(1)

    artifacts = write_spanner_table_artifacts(
        table,
        tables_dir,
        "table_macro_accuracy_by_search_bucket",
        caption=(
            "Macro accuracy (\\%), the unweighted mean accuracy across correct-option "
            f"positions, by {axis_label.lower()} for experiment {experiment_number} (reachability multiple "
            f"choice){caption_context}. Grouped entries are estimate [lower, upper] using "
            "95\\% confidence intervals that combine the per-position Wilson score "
            "intervals (MOVER); Avg is the unweighted mean point estimate."
        ),
        label=sub_experiment_label(sub_tag, "macro_accuracy_by_search_bucket", experiment_tag=experiment_number),
        spanner_label=axis_label,
        grouped_columns=table_columns,
        trailing_columns=["Avg"],
        table_environment="table*",
    )

    print(f"[{sub_tag}] macro accuracy over option positions (%)")
    display(table)
    print_table_path(artifacts)

    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Report: construction diagnostics
# ─────────────────────────────────────────────────────────────────────────────


def find_source_payload_path(
    question_hops: str, bucket_label: str, filename: str
) -> Path | None:
    """Locate one constructor output backing an evaluated bucket.

    Uses the same current-first, legacy-fallback condition order as the scripts.
    """
    root = CFG.experiment_root / "source"
    segment_filters = build_segment_filters([bucket_label], [], [question_hops], [])
    matches = [
        path
        for path, values in select_preferred_artifacts(root, filename, SOURCE_SEGMENT_KEYS, segment_filters)
        if values["question_hops"] == question_hops and values["bucket"] == bucket_label
    ]
    return matches[0] if matches else None


def summarize_source_bucket(questions_path: Path) -> dict[str, object] | None:
    """Recompute option costs, invariants, and path reuse for one bucket."""
    payload = load_json(questions_path)
    questions = payload.get("questions") or []
    if not questions:
        return None

    correct_costs: list[float] = []
    wrong_costs: list[float] = []
    correct_components: list[int] = []
    wrong_components: list[int] = []
    position_counter: Counter = Counter()
    none_position_counter: Counter = Counter()
    wrong_path_counter: Counter = Counter()
    correct_path_ids: set[str] = set()
    target_rooted_cheaper = 0
    invariant_ok = 0
    none_correct = 0
    target_correct = 0
    scored = 0

    for question in questions:
        multiple_choice = question.get("multiple_choice") or {}
        reachability = question.get("reachability") or {}
        options = multiple_choice.get("options") or []
        correct_option = str(multiple_choice.get("correct_option") or "").upper()
        metadata = reachability.get("visible_graph")
        if not options or not correct_option or not isinstance(metadata, dict):
            continue

        scored += 1
        position_counter[correct_option] += 1
        none_option_labels = [
            str(option.get("label") or "").upper()
            for option in options
            if isinstance(option, dict) and option.get("option_kind") == "none"
        ]
        none_position_counter.update(none_option_labels)
        if correct_option in none_option_labels:
            none_correct += 1
            path_ids = list(question.get("path_ids") or [])
            if path_ids:
                correct_path_ids.add(str(path_ids[0]))
        else:
            target_correct += 1

        for option in options:
            if not isinstance(option, dict):
                continue
            label = str(option.get("label") or "").upper()
            complexity = option.get("complexity")
            component_size = option.get("component_size")
            path_id = str(option.get("path_id") or "")
            if label == correct_option:
                if complexity is not None:
                    correct_costs.append(float(complexity))
                if component_size is not None:
                    correct_components.append(int(component_size))
                if path_id:
                    correct_path_ids.add(path_id)
                continue
            if complexity is not None:
                wrong_costs.append(float(complexity))
            if component_size is not None:
                wrong_components.append(int(component_size))
            if path_id:
                wrong_path_counter[path_id] += 1

        complexity_metadata = reachability.get("complexity") or {}
        source_work = complexity_metadata.get("source_endpoint_work")
        target_work = complexity_metadata.get("target_endpoint_work")
        if source_work is not None and target_work is not None:
            target_rooted_cheaper += float(target_work) < float(source_work)

        visible = visible_graph_from_metadata(metadata)
        option_pairs = [
            (str(option.get("label") or ""), str(option.get("node_id") or ""))
            for option in options
            if isinstance(option, dict) and option.get("option_kind") != "none"
        ]
        if multiple_choice_graph_invariant_holds(
            visible,
            source_node_id=str(reachability.get("source_node_id") or ""),
            options=option_pairs,
            correct_option=correct_option,
            none_option_label=(none_option_labels[0] if none_option_labels else None),
        ):
            invariant_ok += 1

    if not scored:
        return None

    def median_or_none(values: list) -> float | None:
        return float(statistics.median(values)) if values else None

    summary = {"Questions": scored, "Invariant OK": invariant_ok}
    for label in CFG.option_labels:
        summary[label] = int(position_counter.get(label, 0))
    if CFG.none_option:
        summary["None correct"] = none_correct
        summary["Target correct"] = target_correct
        summary["None correct (%)"] = 100.0 * none_correct / scored
        for label in CFG.option_labels:
            summary[f"None@{label}"] = int(none_position_counter.get(label, 0))
    summary.update(
        {
            "Correct cost (med)": median_or_none(correct_costs),
            "Wrong cost (med)": median_or_none(wrong_costs),
            "Correct component (med)": median_or_none(correct_components),
            "Wrong component (med)": median_or_none(wrong_components),
            "Target-rooted cheaper (%)": 100.0 * target_rooted_cheaper / scored,
            "Distinct correct paths": len(correct_path_ids),
            "Distinct wrong paths": len(wrong_path_counter),
            "Wrong-path reuse (max)": max(wrong_path_counter.values()) if wrong_path_counter else 0,
        }
    )
    return summary


def report_construction_diagnostics(
    question_hops_values: list[str],
    bucket_labels: list[str],
    sub_tag: str,
    row_axis: str = "bucket",
    caption_context: str = "",
) -> dict[str, object]:
    """Recompute construction diagnostics, one combined table per sub-experiment.

    ``row_axis`` selects which swept axis becomes the table's row label:
    ``"bucket"`` (bucket_labels vary, question_hops fixed to one value -- the
    complexity sub-experiment) or ``"question_hops"`` (the reverse -- qhops).
    """
    row_name = "Question hops" if row_axis == "question_hops" else "Bucket"
    diagnostic_rows = []
    for question_hops in question_hops_values:
        for bucket_label in bucket_labels:
            questions_path = find_source_payload_path(question_hops, bucket_label, "questions.json")
            if questions_path is None:
                continue
            summary = summarize_source_bucket(questions_path)
            if summary is None:
                continue
            row_value = question_hops if row_axis == "question_hops" else bucket_label
            diagnostic_rows.append({row_name: row_value, **summary})

    if not diagnostic_rows:
        print(f"[{sub_tag}] no constructor outputs found, skipping diagnostics")
        return {}

    construction_diagnostics = pd.DataFrame(diagnostic_rows).set_index(row_name).round(2)
    tables_dir = sub_experiment_tables_dir(sub_tag)
    artifacts = write_table_artifacts(
        construction_diagnostics,
        tables_dir,
        "table_construction_diagnostics_by_search_bucket",
        caption=(
            f"Construction diagnostics per {row_name.lower()} for "
            f"experiment 8{caption_context}. \\emph{{Invariant OK}} counts "
            "questions whose stored visible graph still makes exactly one option "
            "reachable; \\emph{Wrong cost} is capped by the correct tree's "
            "component size."
        ),
        label=sub_experiment_label(sub_tag, "construction_diagnostics", experiment_tag="8"),
        table_environment="table*",
        bold_header=True,
    )

    print(f"[{sub_tag}]")
    display(construction_diagnostics)
    if (construction_diagnostics["Invariant OK"] != construction_diagnostics["Questions"]).any():
        print(
            f"[{sub_tag}] WARNING: at least one {row_name.lower()} has "
            "questions whose stored graph does not make exactly one option reachable."
        )
    print_table_path(artifacts)

    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Report: needle and prompt-packing statistics
# ─────────────────────────────────────────────────────────────────────────────


def find_prompt_payload_path(
    question_hops: str, bucket_label: str, distracting_content_type: str, context_length: str
) -> Path | None:
    """Locate the assembled prompt payload backing one evaluated bucket."""
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


def load_needle_statistics_inputs(
    question_hops_values: list[str],
    bucket_labels: list[str],
    distracting_content_type: str,
    context_length: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load source questions, prompt fill, and packing configurations.

    Returns empty frames rather than raising when a stage has not been run yet,
    so the notebook still executes on a partially generated experiment.
    """
    source_rows = []
    fill_rows = []
    configuration_rows = []

    for question_hops in question_hops_values:
        for bucket_label in bucket_labels:
            questions_path = find_source_payload_path(question_hops, bucket_label, "needles.json")
            if questions_path is not None:
                source_payload = load_json(questions_path)
                for question in source_payload.get("questions") or []:
                    needles = question.get("needles") or []
                    if not isinstance(needles, list) or any(
                        not isinstance(needle, str) for needle in needles
                    ):
                        raise ValueError(
                            f"Invalid needles for qid={question.get('qid')} in {questions_path}"
                        )
                    multiple_choice = question.get("multiple_choice") or {}
                    reachability = question.get("reachability") or {}
                    visible_graph = reachability.get("visible_graph") or {}
                    expanded_paper_ids = set(visible_graph.get("expanded_paper_ids") or [])
                    terminal_paper_ids = set(
                        (visible_graph.get("terminal_reference_nodes") or {}).keys()
                    )
                    source_rows.append(
                        {
                            "question_hops": question_hops,
                            "bucket": bucket_label,
                            "qid": question.get("qid"),
                            "question": question.get("question"),
                            "question_type": question.get("question_type"),
                            "needle_type": CFG.target_needle_type,
                            "traversal_mode": CFG.target_traversal_mode,
                            "needles": needles,
                            "correct_option": multiple_choice.get("correct_option"),
                            "num_options": multiple_choice.get("num_options"),
                            "num_target_options": multiple_choice.get(
                                "num_target_options"
                            ),
                            "none_option": multiple_choice.get("none_option"),
                            "realized_difficulty": (reachability.get("complexity") or {}).get("value"),
                            "visible_paper_count": len(expanded_paper_ids | terminal_paper_ids),
                        }
                    )

            prompts_path = find_prompt_payload_path(
                question_hops, bucket_label, distracting_content_type, context_length
            )
            if prompts_path is None:
                continue
            prompt_payload = load_json(prompts_path)
            metadata = prompt_payload.get("metadata") or {}
            full_prompt_metadata = metadata.get("full_prompts") or {}
            configuration_rows.append(
                {
                    "question_hops": question_hops,
                    "target_context_length": full_prompt_metadata.get("target_context_length"),
                    "tokenizer_model_slugs": tuple(
                        full_prompt_metadata.get("tokenizer_model_slugs") or []
                    ),
                    "question_last": full_prompt_metadata.get("question_last", False),
                    "context_item_prefix": full_prompt_metadata.get("context_item_prefix", ""),
                    "context_item_separator": full_prompt_metadata.get(
                        "context_item_separator", "\n\n"
                    ),
                }
            )
            prompt_fill_metadata = metadata.get("prompt_fill") or {}
            for fill in prompt_fill_metadata.get("per_question") or []:
                fill_rows.append(
                    {
                        "question_hops": question_hops,
                        "bucket": bucket_label,
                        "qid": fill.get("qid"),
                        "distractors_used": fill.get("distractor_blocks"),
                    }
                )

    if not source_rows:
        print("No matching source needles were found -- run the experiment 8 v6 data-generation scripts first.")
    if not configuration_rows:
        print(
            "No matching full prompts were found -- needle statistics will omit "
            "prompt-packing columns until the prompt stage has run."
        )
    return (
        pd.DataFrame(source_rows),
        pd.DataFrame(fill_rows),
        pd.DataFrame(configuration_rows).drop_duplicates().reset_index(drop=True),
    )


def report_prompt_packing_configuration(
    prompt_packing_configurations: pd.DataFrame, sub_tag: str
) -> dict[str, object] | None:
    return _report_prompt_packing_configuration(
        prompt_packing_configurations,
        sub_tag,
        question_type_label="reachability multiple-choice experiment",
        experiment_number="8",
    )


def compute_needle_question_metrics(
    needle_source_questions: pd.DataFrame,
    needle_prompt_fill: pd.DataFrame,
    prompt_packing_configurations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _compute_needle_question_metrics(
        needle_source_questions,
        needle_prompt_fill,
        prompt_packing_configurations,
        answer_field="correct_option",
    )


def report_needle_statistics(
    needle_question_metrics: pd.DataFrame,
    needle_length_metrics: pd.DataFrame,
    question_hops_values: list[str],
    bucket_labels: list[str],
    sub_tag: str,
    row_axis: str = "bucket",
    caption_context: str = "",
) -> dict[str, object]:
    return _report_needle_statistics(
        needle_question_metrics,
        needle_length_metrics,
        question_hops_values,
        bucket_labels,
        sub_tag,
        row_axis=row_axis,
        caption_context=caption_context,
        count_column_label="Correct-option count",
        count_column_fn=lambda bucket_questions: ", ".join(
            f"{label}={int((bucket_questions['correct_option'] == label).sum())}"
            for label in CFG.option_labels
        ),
        experiment_number="8",
    )


def report_prompt_fill(
    question_hops_values: list[str],
    bucket_labels: list[str],
    distracting_content_type: str,
    context_length: str,
    sub_tag: str,
    row_axis: str = "bucket",
    caption_context: str = "",
) -> dict[str, object]:
    return _report_prompt_fill(
        question_hops_values,
        bucket_labels,
        distracting_content_type,
        context_length,
        sub_tag,
        row_axis=row_axis,
        caption_context=caption_context,
        find_payload_path=find_prompt_payload_path,
        experiment_number="8",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report: figures
# ─────────────────────────────────────────────────────────────────────────────


def bucket_figure_specs() -> list[tuple[str, dict, str]]:
    return [
        (
            "accuracy_by_search_bucket",
            {"value_column": "accuracy_pct", "y_label": "Accuracy (%)"},
            f"headline — pooled accuracy against the {CFG.chance_accuracy_pct:.0f}% random-guess floor",
        ),
        (
            "macro_accuracy_by_search_bucket",
            {
                "value_column": "macro_option_accuracy_pct",
                "y_label": "Macro accuracy over option positions (%)",
            },
            "below the pooled figure means the model favours particular option letters",
        ),
    ]


def report_bucket_figures(
    sub_results: pd.DataFrame,
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    column_values: list[str],
    sub_tag: str,
    axis_field: str = "bucket_label",
    axis_label: str = "Expected search-step bucket",
    title: str = "Experiment 8 Reachability Multiple Choice — Papers-only",
) -> dict[str, object]:
    if sub_results.empty:
        print(f"[{sub_tag}] No bucket figure written; no eval reports matched.")
        return {}

    hop_results = as_bucket_axis(sub_results, axis_field)
    figures_dir = sub_experiment_figures_dir(sub_tag)
    chance_baseline = {value: CFG.chance_accuracy_pct for value in column_values}
    constant_answer_baseline = position_baseline_by_bucket(hop_results, column_values)

    artifacts: dict[str, object] = {}
    for stem, figure_kwargs, note in bucket_figure_specs():
        figure_kwargs = dict(figure_kwargs)
        if stem == "accuracy_by_search_bucket":
            figure_kwargs["baseline_by_bucket"] = constant_answer_baseline
            figure_kwargs["baseline_name"] = "Always-most-frequent-option baseline"
        else:
            figure_kwargs["baseline_by_bucket"] = chance_baseline
            figure_kwargs["baseline_name"] = f"Chance ({CFG.chance_accuracy_pct:.0f}%)"

        figure_artifacts = write_bucket_figure_artifacts(
            hop_results,
            figures_dir,
            stem,
            bucket_labels=column_values,
            ordered_model_slugs=ordered_model_slugs,
            model_names=model_names,
            num_columns_span=2,
            title=f"{title}, {CFG.num_options} options — [{sub_tag}]",
            error_bar=figure_kwargs.pop("error_bar", True),
            metric_ci=FIGURE_METRIC_CI,
            x_axis_label=axis_label,
            **figure_kwargs,
        )
        artifacts[stem] = figure_artifacts

        print(f"[{sub_tag}][{stem}] {note}")
        print_figure_path(figure_artifacts)
        display(Image(filename=str(figure_artifacts["png_path"])))

    return artifacts


def report_family_figures(
    sub_results: pd.DataFrame,
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    column_values: list[str],
    sub_tag: str,
    axis_field: str = "bucket_label",
    axis_label: str = "Expected search-step bucket",
    title_prefix: str = "",
) -> dict[str, object]:
    return _report_family_figures(
        sub_results,
        ordered_model_slugs,
        model_names,
        column_values,
        sub_tag,
        axis_field=axis_field,
        axis_label=axis_label,
        title_prefix=title_prefix,
        metric_column="accuracy_pct",
        metric_label="Accuracy (%)",
        family_stem_prefix="accuracy_by_search_bucket",
        chance_accuracy_pct=CFG.chance_accuracy_pct,
        metric_ci=FIGURE_METRIC_CI,
    )


def report_family_grid(
    sub_results: pd.DataFrame,
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    column_values: list[str],
    sub_tag: str,
    axis_field: str = "bucket_label",
    axis_label: str = "Expected search-step bucket",
    title: str = "Experiment 8",
) -> dict[str, object]:
    return _report_family_grid(
        sub_results,
        ordered_model_slugs,
        model_names,
        column_values,
        sub_tag,
        axis_field=axis_field,
        axis_label=axis_label,
        title=title,
    )


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
) -> dict[str, dict[str, object]]:
    return _report_distractor_comparison(
        sub_results,
        preferred_model_order,
        question_hops_values,
        bucket_labels,
        reference_distractor_type,
        sub_tag,
        count_fields=count_fields(),
        experiment_number="8",
    )
