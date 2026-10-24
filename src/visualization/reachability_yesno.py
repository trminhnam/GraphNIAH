"""Notebook helpers for ``experiment_7_reachability_yesno.v6.ipynb``.

Everything specific to experiment 7 (reachability yes/no) -- path parsing,
YES/NO metric helpers, construction diagnostics, and figure specs -- lives
here. Logic shared with experiment 8 (reachability multiple choice) lives in
``reachability_reports.py`` instead; the functions below that only differ
from their experiment-8 counterpart by a few fixed values are thin wrappers
around the shared versions, so this module's public surface is unchanged
from before the split.

The notebook calls :func:`configure` once (from its hyperparameter cell) with
every fixed path-selector hyperparameter, then :func:`gather_results` once to
build the master results table. Each sub-experiment section filters that
table with ``filter_results_by_subexperiment`` and passes the filtered rows
plus its own ``question_hops``/``bucket_label`` axis values into the
``report_*`` functions below.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import networkx as nx
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
    select_preferred_artifacts,
    sub_experiment_figures_dir,
    sub_experiment_label,
    sub_experiment_tables_dir,
)
from src.questions.reachability_yesno import (
    no_exhaustive_search_steps,
    visible_graph_from_metadata,
    yes_expected_search_steps,
)

# Segments that must be present for each artifact kind. ``alpha`` (added by
# v7's distractor-tree scaling sweep) and ``theta``/``discount`` (v10's CRP
# tree expansion) are deliberately not required here, so older paths without
# them parse with the same code -- they just become extra keys in ``values``
# that ``build_segment_filters`` can optionally pin via ``CFG.target_alpha``/
# ``CFG.target_theta``/``CFG.target_discount``.
SOURCE_SEGMENT_KEYS = {
    "hops",
    "question_type",
    "format_version",
    "no_construction",
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

COUNT_FIELDS = ("total", "correct", "yes_total", "yes_correct", "no_total", "no_correct")

BUCKET_FIGURE_SPECS = [
    (
        "balanced_accuracy_by_search_bucket",
        {
            "value_column": "balanced_accuracy_pct",
            "y_label": "Balanced accuracy (%)",
            "baseline_name": "Chance (50%)",
        },
        "headline — mean of YES and NO recall; 50% is a constant-label predictor",
    ),
    (
        "label_recall_by_search_bucket",
        {
            "value_column": "yes_recall_pct",
            "overlay_column": "no_recall_pct",
            "series_names": ("YES recall", "NO recall"),
            "y_label": "Per-label recall (%)",
            # Four overlapping bands in two colours is unreadable; the intervals
            # for these same numbers are in the balanced-accuracy figure.
            "error_bar": False,
        },
        "NO recall pinned at 100% with YES near 0 means the model always answers NO",
    ),
    (
        "accuracy_by_search_bucket",
        {
            "value_column": "accuracy_pct",
            "y_label": "Accuracy (%)",
            "baseline_name": "Always-majority baseline",
        },
        "pooled accuracy — only meaningful relative to the dashed majority baseline",
    ),
]

# Same as BUCKET_FIGURE_SPECS minus the YES/NO recall-lines figure: the qhops
# sub-experiment sweeps a single bucket, so the recall breakdown is redundant
# with the YES/NO columns already in the accuracy-by-question-hops table.
QHOPS_FIGURE_SPECS = [
    spec for spec in BUCKET_FIGURE_SPECS if spec[0] != "label_recall_by_search_bucket"
]


def configure(**kwargs) -> None:
    """Set the fixed path-selector hyperparameters every function reads from ``CFG``."""
    CFG.__dict__.clear()
    CFG.__dict__.update(kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Gather: parse eval.json paths and build the master results table
# ─────────────────────────────────────────────────────────────────────────────


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
    comparison scan). ``alpha`` is only included (pinned to
    ``CFG.target_alpha``) when that attribute is set -- absent for pre-v7
    experiment roots, which have no alpha segment to filter on. ``theta`` and
    ``discount`` work the same way for v10 roots.
    """
    return {
        "hops": {CFG.target_input_hops},
        "question_type": {CFG.target_question_type},
        "format_version": {CFG.target_format_version},
        "no_construction": {CFG.target_no_construction},
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
    }


def get_answer_type_counts(report: dict) -> dict[str, dict[str, int]]:
    counts = {
        "yes": {"total": 0, "correct": 0},
        "no": {"total": 0, "correct": 0},
    }
    metrics = report.get("answer_type_metrics")
    if isinstance(metrics, dict):
        for label in ("yes", "no"):
            label_metrics = metrics.get(label)
            if not isinstance(label_metrics, dict):
                continue
            counts[label]["total"] = int(label_metrics.get("total") or 0)
            counts[label]["correct"] = int(label_metrics.get("correct") or 0)
        return counts

    # Backward compatibility for eval files created before per-label metrics.
    for detail in report.get("details") or []:
        canonical_values = detail.get("ground_truth_canonical") or []
        if isinstance(canonical_values, str):
            canonical_values = [canonical_values]
        label = next(
            (
                str(value).casefold()
                for value in canonical_values
                if str(value).casefold() in {"yes", "no"}
            ),
            None,
        )
        if label is None:
            continue
        counts[label]["total"] += 1
        if detail.get("correct") is True:
            counts[label]["correct"] += 1
    return counts


def gather_results(
    bucket_ranges: dict[str, tuple[int, int]],
    bucket_sort_keys: dict[str, int],
    distracting_content_types: list[str],
    context_lengths: list[str],
) -> pd.DataFrame:
    """Gather every eval report for the fixed axes, across every sub-experiment.

    ``bucket_ranges``/``bucket_sort_keys`` should cover the union of buckets
    used by any sub-experiment, and ``distracting_content_types``/
    ``context_lengths`` the union of distractor types and context lengths, so
    a sub-experiment can later select its own rows with
    ``filter_results_by_subexperiment`` instead of re-scanning disk.
    ``question_hops``, ``theta``, and ``discount`` are not filtered here for
    the same reason, so each sub-experiment must select its own values of them.
    """
    segment_filters = build_segment_filters(
        list(bucket_ranges), distracting_content_types, [], context_lengths
    )
    for key in ("question_hops", "theta", "discount"):  # not fixed here -- every value is kept
        segment_filters.pop(key, None)

    records = []
    for eval_path, values in select_preferred_artifacts(
        CFG.experiment_root / "eval", "eval.json", MODEL_OUTPUT_SEGMENT_KEYS, segment_filters
    ):
        bucket_label = values["bucket"]
        bucket_min, bucket_max = bucket_ranges[bucket_label]
        report = load_json(eval_path)
        answer_type_counts = get_answer_type_counts(report)
        records.append(
            {
                "model_slug": values["model"],
                "format_version": values["format_version"],
                "question_hops": values["question_hops"],
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
                "yes_correct": answer_type_counts["yes"]["correct"],
                "yes_total": answer_type_counts["yes"]["total"],
                "no_correct": answer_type_counts["no"]["correct"],
                "no_total": answer_type_counts["no"]["total"],
                "wrong_answer": report.get("wrong_answer"),
                "no_answer_box": report.get("no_answer_box"),
                "unparseable": report.get("unparseable"),
                "reasoning_tokens": get_average_reasoning_tokens(report),
                "eval_path": str(eval_path),
            }
        )

    return pd.DataFrame(records)


def add_figure_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the composition-robust metric columns used by the bucket figures."""
    return _add_figure_metric_columns(frame, FIGURE_METRIC_VALUES)


# ─────────────────────────────────────────────────────────────────────────────
# Shared metric helpers (YES/NO split)
# ─────────────────────────────────────────────────────────────────────────────


def label_recall_pct(counts: dict[str, int], label: str) -> float | None:
    """Return per-label accuracy, or ``None`` when the label is absent."""
    label_total = counts[f"{label}_total"]
    if not label_total:
        return None
    return 100.0 * counts[f"{label}_correct"] / label_total


def balanced_accuracy_pct(counts: dict[str, int]) -> float | None:
    """Return the unweighted mean of YES and NO recall.

    Undefined when a bucket contains only one label, since a single recall is
    not comparable to a two-label mean.
    """
    recalls = [
        recall
        for recall in (label_recall_pct(counts, "yes"), label_recall_pct(counts, "no"))
        if recall is not None
    ]
    if len(recalls) < 2:
        return None
    return sum(recalls) / len(recalls)


def balanced_ci_bounds_pct(counts: dict[str, int]) -> tuple[float, float]:
    """Return the 95% CI bounds for balanced accuracy, or ``(nan, nan)``.

    Balanced accuracy averages two independent label recalls, so their Wilson
    intervals are combined with :func:`mean_ci_bounds_pct` (MOVER).
    """
    labels = ("yes", "no")
    return mean_ci_bounds_pct(
        [label_recall_pct(counts, label) for label in labels],
        [label_ci_bounds_pct(counts, label) for label in labels],
    )


def label_ci_bounds_pct(counts: dict[str, int], label: str) -> tuple[float, float]:
    """Return the 95% Wilson CI bounds for one label recall, or ``(nan, nan)``."""
    return wilson_ci_bounds_pct(counts[f"{label}_correct"], counts[f"{label}_total"])


# Plotted column -> CI bounds from the same pooled counts. Only columns
# listed here can be drawn with ``error_bar=True``.
FIGURE_METRIC_CI = {
    "accuracy_pct": lambda counts: wilson_ci_bounds_pct(counts["correct"], counts["total"]),
    "balanced_accuracy_pct": balanced_ci_bounds_pct,
    "yes_recall_pct": lambda counts: label_ci_bounds_pct(counts, "yes"),
    "no_recall_pct": lambda counts: label_ci_bounds_pct(counts, "no"),
}

# Derived columns the bucket figures plot instead of pooled accuracy, which is
# not comparable across buckets whose YES/NO mix differs.
FIGURE_METRIC_VALUES = {
    "balanced_accuracy_pct": balanced_accuracy_pct,
    "yes_recall_pct": lambda counts: label_recall_pct(counts, "yes"),
    "no_recall_pct": lambda counts: label_recall_pct(counts, "no"),
}


def majority_baseline_by_bucket(
    hop_results: pd.DataFrame, bucket_labels: list[str]
) -> dict[str, float]:
    """Return the always-majority-label baseline for each observed bucket.

    Every model scores the same prompt file, so the label mix is a property of
    the bucket; take the widest per-model view in case a run was partial.
    """
    baselines = {}
    for bucket_label in get_bucket_columns(hop_results, bucket_labels):
        bucket_rows = hop_results[hop_results["bucket_label"] == bucket_label]
        per_model = bucket_rows.groupby("model_slug")[["yes_total", "no_total"]].sum()
        if per_model.empty:
            continue
        yes_total = int(per_model["yes_total"].max())
        no_total = int(per_model["no_total"].max())
        if yes_total + no_total:
            baselines[bucket_label] = 100.0 * max(yes_total, no_total) / (yes_total + no_total)
    return baselines


# ─────────────────────────────────────────────────────────────────────────────
# Report: accuracy tables
# ─────────────────────────────────────────────────────────────────────────────


def compute_label_composition(
    hop_results: pd.DataFrame, axis_values: list[str]
) -> dict[str, dict[str, float]]:
    """Return the YES/NO composition for each axis value, plus an "Overall" entry.

    Every model scores the same prompt file, so the label mix is a property of
    the axis value (or of the whole sweep, for "Overall"), not of any one
    model; take the widest per-model view in case a run was partial.
    """
    composition: dict[str, dict[str, float]] = {}
    for axis_value in [*axis_values, None]:
        subset = (
            hop_results if axis_value is None else hop_results[hop_results["bucket_label"] == axis_value]
        )
        per_model = subset.groupby("model_slug")[["yes_total", "no_total"]].sum()
        if per_model.empty:
            continue
        yes_total = int(per_model["yes_total"].max())
        no_total = int(per_model["no_total"].max())
        labelled_total = yes_total + no_total
        if not labelled_total:
            continue
        key = "Overall" if axis_value is None else axis_value
        composition[key] = {
            "no_share_pct": 100.0 * no_total / labelled_total,
            "majority_baseline_pct": 100.0 * max(yes_total, no_total) / labelled_total,
        }
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
) -> dict[str, object]:
    """Accuracy, YES/NO composition, and per-label recall in one combined table.

    One row per (axis value, model), plus an "Overall" axis value per model
    pooling every value in scope. Merges what used to be three separate tables
    (a bucket-columns accuracy spanner, a pooled answer-type table, and a
    label-balance table) and replaces the standalone overall-accuracy summary.
    """
    if sub_results.empty:
        print(f"[{sub_tag}] No eval reports matched this sub-experiment's sweep yet.")
        return {}

    hop_results = as_bucket_axis(sub_results, axis_field)
    model_slugs = get_hop_model_slugs(hop_results, ordered_model_slugs)
    axis_values = get_bucket_columns(hop_results, column_values)
    composition = compute_label_composition(hop_results, axis_values)

    def counts_to_row(
        axis_value: str,
        model_slug: str,
        counts: dict[str, int],
        result_rows: pd.DataFrame,
    ) -> dict[str, object]:
        comp = composition.get(axis_value)
        return {
            axis_label: axis_value,
            "Model": model_names[model_slug],
            "Total questions": counts["total"],
            "YES questions": counts["yes_total"],
            "NO questions": counts["no_total"],
            "NO share (%)": comp["no_share_pct"] if comp else None,
            "Majority baseline (%)": comp["majority_baseline_pct"] if comp else None,
            "YES recall, % [95% CI]": format_ci_pct(
                label_recall_pct(counts, "yes"),
                label_ci_bounds_pct(counts, "yes"),
            ),
            "NO recall, % [95% CI]": format_ci_pct(
                label_recall_pct(counts, "no"),
                label_ci_bounds_pct(counts, "no"),
            ),
            "Total accuracy, % [95% CI]": format_bernoulli_ci_pct(
                counts["correct"], counts["total"]
            ),
            "Reasoning tokens": mean_reasoning_tokens(result_rows),
        }

    rows = []
    for axis_value in axis_values:
        for model_slug in model_slugs:
            counts = get_cell_counts(hop_results, model_slug, axis_value, COUNT_FIELDS)
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
                pool_counts(model_results, COUNT_FIELDS),
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
            f"Accuracy, YES/NO composition, and per-label recall by {axis_label.lower()} "
            f"for experiment 7 (reachability yes/no){caption_context}, plus an Overall row "
            "pooling every value in scope. The majority baseline is the accuracy of always "
            "answering the majority label at that value. Accuracy and recall entries are "
            "estimate [lower, upper] using 95\\% Wilson score confidence intervals. Reasoning tokens "
            "is the mean per response; older artifacts "
            "use output tokens, which include hidden reasoning tokens."
        ),
        label=sub_experiment_label(sub_tag, "search_bucket_accuracy", experiment_tag="7"),
        table_environment="table*",
        bold_header=True,
    )

    print(f"[{sub_tag}]")
    display(table)
    print_table_path(artifacts)

    return artifacts


def report_balanced_accuracy_by_bucket(
    sub_results: pd.DataFrame,
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    column_values: list[str],
    sub_tag: str,
    axis_field: str = "bucket_label",
    axis_label: str = "Expected search-step bucket",
    caption_context: str = "",
) -> dict[str, object]:
    """Balanced accuracy (mean of YES/NO recall) as a Model x axis-value matrix.

    Kept as its own compact wide table (unlike ``report_accuracy_by_bucket``)
    since it is a single derived metric, not raw composition/count detail.
    """
    if sub_results.empty:
        print(f"[{sub_tag}] No balanced-accuracy table written; no eval reports matched.")
        return {}

    hop_results = as_bucket_axis(sub_results, axis_field)
    hop_model_slugs = get_hop_model_slugs(hop_results, ordered_model_slugs)
    table_columns = get_bucket_columns(hop_results, column_values)
    tables_dir = sub_experiment_tables_dir(sub_tag)

    table, numeric_table = build_bucket_ci_matrix(
        hop_results,
        balanced_accuracy_pct,
        balanced_ci_bounds_pct,
        hop_model_slugs,
        table_columns,
        model_names,
        COUNT_FIELDS,
    )
    table["Avg"] = numeric_table[table_columns].mean(axis=1, skipna=True).round(1)

    artifacts = write_spanner_table_artifacts(
        table,
        tables_dir,
        "table_balanced_accuracy_by_search_bucket",
        caption=(
            "Balanced accuracy (\\%), the unweighted mean of YES and NO recall, by "
            f"{axis_label.lower()} for experiment 7 (reachability yes/no){caption_context}. "
            "Grouped entries are estimate [lower, upper] using 95\\% confidence intervals "
            "that combine the Wilson score intervals of YES and NO recall (MOVER); Avg is "
            "the unweighted mean point estimate."
        ),
        label=sub_experiment_label(sub_tag, "balanced_accuracy_by_search_bucket", experiment_tag="7"),
        spanner_label=axis_label,
        grouped_columns=table_columns,
        trailing_columns=["Avg"],
        table_environment="table*",
    )

    print(f"[{sub_tag}] balanced accuracy (%)")
    display(table)
    print_table_path(artifacts)

    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Report: construction diagnostics
# ─────────────────────────────────────────────────────────────────────────────


def find_source_payload_path(question_hops: str, bucket_label: str, filename: str) -> Path | None:
    """Locate one constructor output backing an evaluated bucket."""
    root = CFG.experiment_root / "source"
    segment_filters = build_segment_filters([bucket_label], [], [question_hops], [])
    matches = [
        path
        for path, values in select_preferred_artifacts(root, filename, SOURCE_SEGMENT_KEYS, segment_filters)
        if values["question_hops"] == question_hops and values["bucket"] == bucket_label
    ]
    return matches[0] if matches else None


def find_source_questions_path(question_hops: str, bucket_label: str) -> Path | None:
    """Locate the constructor output backing one evaluated bucket."""
    return find_source_payload_path(question_hops, bucket_label, "questions.json")


def summarize_source_bucket(
    questions_path: Path, target_rooted_easy_threshold: int
) -> dict[str, object] | None:
    """Recompute realized complexity and both solver directions for one bucket."""
    payload = load_json(questions_path)
    questions = payload.get("questions") or []
    if not questions:
        return None

    complexity_by_label: dict[str, list[float]] = {"YES": [], "NO": []}
    target_rooted_costs: list[int] = []
    cheapest_costs: list[int] = []

    for question in questions:
        reachability = question.get("reachability") or {}
        label = str(reachability.get("label") or "").upper()
        metadata = reachability.get("visible_graph")
        if label not in complexity_by_label or not isinstance(metadata, dict):
            continue

        visible = visible_graph_from_metadata(metadata)
        graph = visible.graph
        source_node_id = str(reachability.get("source_node_id") or "")
        target_node_id = str(reachability.get("target_node_id") or "")
        if source_node_id not in graph or target_node_id not in graph:
            continue

        if label == "YES":
            complexity = yes_expected_search_steps(visible, source_node_id, target_node_id)
        else:
            complexity = no_exhaustive_search_steps(visible, source_node_id, target_node_id)
            source_component = len(nx.node_connected_component(graph, source_node_id))
            target_component = len(nx.node_connected_component(graph, target_node_id))
            target_rooted_costs.append(target_component)
            cheapest_costs.append(min(source_component, target_component))

        if complexity is not None:
            complexity_by_label[label].append(float(complexity))

    yes_total = len(complexity_by_label["YES"])
    no_total = len(complexity_by_label["NO"])
    labelled_total = yes_total + no_total
    if labelled_total == 0:
        return None

    def median_or_none(values: list) -> float | None:
        return float(statistics.median(values)) if values else None

    easy_share = (
        100.0 * sum(cost < target_rooted_easy_threshold for cost in cheapest_costs) / len(cheapest_costs)
        if cheapest_costs
        else None
    )

    return {
        "Questions": labelled_total,
        "YES": yes_total,
        "NO": no_total,
        "NO share (%)": 100.0 * no_total / labelled_total,
        "YES cost (med)": median_or_none(complexity_by_label["YES"]),
        "NO cost (med)": median_or_none(complexity_by_label["NO"]),
        "NO target-rooted (med)": median_or_none(target_rooted_costs),
        f"NO cheapest < {target_rooted_easy_threshold:g} (%)": easy_share,
    }


def report_construction_diagnostics(
    question_hops_values: list[str],
    bucket_labels: list[str],
    bucket_ranges: dict[str, tuple[int, int]],
    sub_tag: str,
    row_axis: str = "bucket",
    caption_context: str = "",
) -> dict[str, object]:
    """Recompute construction diagnostics, one combined table per sub-experiment.

    ``row_axis`` selects which swept axis becomes the table's row label:
    ``"bucket"`` (bucket_labels vary, question_hops fixed to one value -- the
    complexity sub-experiment) or ``"question_hops"`` (the reverse -- qhops).
    """
    # A NO question is scored by the size of the SOURCE component, i.e. the cost
    # of a source-rooted exhaustive search. Reachability is symmetric, so a
    # solver may instead expand from the target; the cheaper of the two is the
    # real cost. This threshold is the top of this sub-experiment's easiest
    # bucket -- the share of NO items below it is the share whose nominal
    # difficulty does not survive that alternative strategy.
    target_rooted_easy_threshold = bucket_ranges[bucket_labels[0]][1]
    row_name = "Question hops" if row_axis == "question_hops" else "Bucket"

    diagnostic_rows = []
    for question_hops in question_hops_values:
        for bucket_label in bucket_labels:
            questions_path = find_source_questions_path(question_hops, bucket_label)
            if questions_path is None:
                continue
            summary = summarize_source_bucket(questions_path, target_rooted_easy_threshold)
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
            f"experiment 7{caption_context}. \\emph{{NO cost}} is the scored "
            "metric (source-rooted exhaustive search); \\emph{NO target-rooted} "
            "is the cost of expanding from the target instead."
        ),
        label=sub_experiment_label(sub_tag, "construction_diagnostics", experiment_tag="7"),
        table_environment="table*",
        bold_header=True,
    )

    print(f"[{sub_tag}]")
    display(construction_diagnostics)
    print_table_path(artifacts)

    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Report: needle and prompt-packing statistics
# ─────────────────────────────────────────────────────────────────────────────


def find_source_needles_path(question_hops: str, bucket_label: str) -> Path | None:
    """Locate one source payload for the selected format version."""
    return find_source_payload_path(question_hops, bucket_label, "needles.json")


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
    """Load source questions, prompt fill, and packing configurations."""
    source_rows = []
    fill_rows = []
    configuration_rows = []

    for question_hops in question_hops_values:
        for bucket_label in bucket_labels:
            questions_path = find_source_needles_path(question_hops, bucket_label)
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
                            "answer_label": reachability.get("label"),
                            "realized_difficulty": (reachability.get("complexity") or {}).get(
                                "value"
                            ),
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
        raise FileNotFoundError("No matching source needles were found.")
    if not configuration_rows:
        raise FileNotFoundError("No matching full prompts were found.")
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
        question_type_label="reachability yes/no experiment",
        experiment_number="7",
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
        answer_field="answer_label",
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
        count_column_label="YES/NO count",
        count_column_fn=lambda bucket_questions: (
            f"YES={int((bucket_questions['answer_label'] == 'YES').sum())}, "
            f"NO={int((bucket_questions['answer_label'] == 'NO').sum())}"
        ),
        experiment_number="7",
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
        experiment_number="7",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report: figures
# ─────────────────────────────────────────────────────────────────────────────


def report_bucket_figures(
    sub_results: pd.DataFrame,
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    column_values: list[str],
    sub_tag: str,
    axis_field: str = "bucket_label",
    axis_label: str = "Expected search-step bucket",
    figure_specs: list = BUCKET_FIGURE_SPECS,
    title: str = "Experiment 7 Reachability Yes/No — Papers-only",
) -> dict[str, object]:
    if sub_results.empty:
        print(f"[{sub_tag}] No bucket figure written; no eval reports matched.")
        return {}

    hop_results = as_bucket_axis(sub_results, axis_field)
    figures_dir = sub_experiment_figures_dir(sub_tag)
    chance_baseline = {value: 50.0 for value in column_values}
    majority_baseline = majority_baseline_by_bucket(hop_results, column_values)

    artifacts: dict[str, object] = {}
    for stem, figure_kwargs, note in figure_specs:
        figure_kwargs = dict(figure_kwargs)
        if stem == "balanced_accuracy_by_search_bucket":
            figure_kwargs["baseline_by_bucket"] = chance_baseline
        elif stem == "accuracy_by_search_bucket":
            figure_kwargs["baseline_by_bucket"] = majority_baseline

        figure_artifacts = write_bucket_figure_artifacts(
            hop_results,
            figures_dir,
            stem,
            bucket_labels=column_values,
            ordered_model_slugs=ordered_model_slugs,
            model_names=model_names,
            num_columns_span=2,
            title=f"{title} — [{sub_tag}]",
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
        metric_column="balanced_accuracy_pct",
        metric_label="Balanced accuracy (%)",
        family_stem_prefix="balanced_accuracy_by_search_bucket",
        chance_accuracy_pct=50.0,
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
    title: str = "Experiment 7",
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
        count_fields=COUNT_FIELDS,
        experiment_number="7",
    )
