"""Notebook helpers for ``experiment_9_reachability_counting.vN.ipynb``.

Everything specific to experiment 9 (reachability counting) -- path parsing,
numeric-answer metric helpers, construction diagnostics, and figure specs --
lives here. Logic shared with experiments 7/8 lives in
``reachability_reports.py`` instead.

Unlike experiments 7 and 8, the ground truth here is a free integer (e.g.
``answer: ["83"]``, "how many distinct papers are reachable"), scored by
``eval.py`` with an exact-match comparison and **no** per-label/per-option
breakdown in ``eval.json`` (no ``answer_type_metrics``/``option_metrics`` --
just ``total``/``correct``/``accuracy``). Beyond the pooled accuracy every
other experiment reports, this module also derives numeric-error diagnostics
(mean absolute error, exact-match rate, within-±1/±5 rates) from each eval
report's ``details`` array (present for every question type, not counting-
specific). Those are computed as raw *sums* per report, not pre-divided
percentages, so the same generic ``pool_counts``/``get_cell_counts``/
``build_bucket_matrix`` helpers from ``reachability_reports.py`` can pool
them correctly across buckets for an "Overall" row -- averaging pre-computed
per-report MAEs would silently misweight unequal sample sizes.

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

import pandas as pd
from IPython.display import Image, display

from .artifacts import (
    format_bernoulli_ci_pct,
    format_mean_std,
    load_json,
    print_figure_path,
    print_table_path,
    wilson_ci_bounds_pct,
    write_bucket_figure_artifacts,
    write_table_artifacts,
)
from .reachability_reports import (
    CFG,
    FAMILY_FILTERS,
    as_bucket_axis,
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

# Segments that must be present for each artifact kind. No
# ``max_author_answer_index`` and no ``num_options``/``none_option`` --
# confirmed absent from exp9's actual path structure (unlike exp7/exp8).
# ``alpha`` (added by v7's distractor-tree scaling sweep) is deliberately not
# required, so v5/v6 (no alpha segment) and v7 (has one) both parse with the
# same code. The same holds for v10's ``theta``/``discount`` segments.
SOURCE_SEGMENT_KEYS = {
    "hops",
    "question_type",
    "format_version",
    "construction",
    "traversal_mode",
    "needle_type",
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

COUNT_FIELDS = (
    "total",
    "correct",
    "parsed_count",
    "abs_error_sum",
    "exact_match_count",
    "within_1_count",
    "within_5_count",
)


def configure(**kwargs) -> None:
    """Set the fixed path-selector hyperparameters every function reads from ``CFG``."""
    CFG.__dict__.clear()
    CFG.__dict__.update(kwargs)


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
        "format_version": {CFG.target_format_version},
        "construction": {CFG.target_construction},
        "traversal_mode": {CFG.target_traversal_mode},
        "needle_type": {CFG.target_needle_type},
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


# ─────────────────────────────────────────────────────────────────────────────
# Gather: parse eval.json paths and build the master results table
# ─────────────────────────────────────────────────────────────────────────────


def get_counting_error_sums(report: dict) -> dict[str, int | float]:
    """Return raw sums (not percentages) over ``report["details"]`` for pooling.

    Only rows where the model's answer parsed *and* both it and the ground
    truth normalize to an integer are counted -- everything else (unparsed
    responses, non-numeric extractions) is excluded the same way ``correct``
    already is in the base ``total``/``correct`` counts.
    """
    parsed_count = 0
    abs_error_sum = 0.0
    exact_match_count = 0
    within_1_count = 0
    within_5_count = 0
    for detail in report.get("details") or []:
        if not detail.get("parsed"):
            continue
        extracted = detail.get("extracted_canonical")
        ground_truth_values = detail.get("ground_truth_canonical") or []
        if not extracted or not ground_truth_values:
            continue
        try:
            extracted_int = int(extracted)
            truth_int = int(ground_truth_values[0])
        except (TypeError, ValueError):
            continue
        parsed_count += 1
        error = abs(extracted_int - truth_int)
        abs_error_sum += error
        if error == 0:
            exact_match_count += 1
        if error <= 1:
            within_1_count += 1
        if error <= 5:
            within_5_count += 1
    return {
        "parsed_count": parsed_count,
        "abs_error_sum": abs_error_sum,
        "exact_match_count": exact_match_count,
        "within_1_count": within_1_count,
        "within_5_count": within_5_count,
    }


def mean_absolute_error(counts: dict[str, int | float]) -> float | None:
    """Return the pooled mean absolute error, or ``None`` when nothing parsed."""
    if not counts["parsed_count"]:
        return None
    return counts["abs_error_sum"] / counts["parsed_count"]


def exact_match_pct(counts: dict[str, int | float]) -> float | None:
    """Return the pooled exact-match rate among parsed numeric answers."""
    if not counts["parsed_count"]:
        return None
    return 100.0 * counts["exact_match_count"] / counts["parsed_count"]


def within_k_pct(counts: dict[str, int | float], key: str) -> float | None:
    """Return the pooled "within ±k" rate among parsed numeric answers."""
    if not counts["parsed_count"]:
        return None
    return 100.0 * counts[key] / counts["parsed_count"]


def gather_results(
    bucket_ranges: dict[str, tuple[int, int]],
    bucket_sort_keys: dict[str, int],
    distracting_content_types: list[str],
    context_lengths: list[str],
) -> pd.DataFrame:
    """Gather every eval report for the fixed axes, across every sub-experiment.

    ``question_hops``, ``theta``, and ``discount`` are not filtered here, so
    each sub-experiment must select its own values of them.
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
        error_sums = get_counting_error_sums(report)
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
                "wrong_answer": report.get("wrong_answer"),
                "no_answer_box": report.get("no_answer_box"),
                "unparseable": report.get("unparseable"),
                "reasoning_tokens": get_average_reasoning_tokens(report),
                "eval_path": str(eval_path),
                **error_sums,
            }
        )

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Report: accuracy tables
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
    """Accuracy plus numeric-error diagnostics, one row per (axis value, model).

    Unlike experiments 7/8, there is no ground-truth label/option composition
    to report -- the answer is a free integer -- so this table is just
    pooled accuracy plus the numeric-error breakdown from
    :func:`get_counting_error_sums`, with an "Overall" row per model pooling
    every value in scope.
    """
    if sub_results.empty:
        print(f"[{sub_tag}] No eval reports matched this sub-experiment's sweep yet.")
        return {}

    hop_results = as_bucket_axis(sub_results, axis_field)
    model_slugs = get_hop_model_slugs(hop_results, ordered_model_slugs)
    axis_values = get_bucket_columns(hop_results, column_values)

    def counts_to_row(
        axis_value: str,
        model_slug: str,
        counts: dict[str, int],
        result_rows: pd.DataFrame,
    ) -> dict[str, object]:
        return {
            axis_label: axis_value,
            "Model": model_names[model_slug],
            "Total questions": counts["total"],
            "Accuracy, % [95% CI]": format_bernoulli_ci_pct(
                counts["correct"], counts["total"]
            ),
            "Mean abs error": mean_absolute_error(counts),
            "Exact match (%)": exact_match_pct(counts),
            "Within ±1 (%)": within_k_pct(counts, "within_1_count"),
            "Within ±5 (%)": within_k_pct(counts, "within_5_count"),
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
            f"Accuracy and numeric-error diagnostics by {axis_label.lower()} for "
            f"experiment 9 (reachability counting){caption_context}, plus an Overall row "
            "pooling every value in scope. Mean abs error and the within-\\emph{k} rates "
            "are computed over parsed numeric answers only. Accuracy is estimate [lower, "
            "upper] using a 95\\% Wilson score confidence interval. Reasoning tokens is the mean "
            "per response; older artifacts "
            "use output tokens, which include hidden reasoning tokens."
        ),
        label=sub_experiment_label(sub_tag, "search_bucket_accuracy", experiment_tag="9"),
        table_environment="table*",
        bold_header=True,
    )

    print(f"[{sub_tag}]")
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


def summarize_source_bucket(questions_path: Path) -> dict[str, object] | None:
    """Recompute reachable-count and component-size stats for one bucket."""
    payload = load_json(questions_path)
    questions = payload.get("questions") or []
    if not questions:
        return None

    counts: list[float] = []
    complexities: list[float] = []
    for question in questions:
        reachability = question.get("reachability") or {}
        count_value = (reachability.get("count") or {}).get("value")
        complexity_value = (reachability.get("complexity") or {}).get("value")
        if count_value is not None:
            counts.append(float(count_value))
        if complexity_value is not None:
            complexities.append(float(complexity_value))

    if not counts:
        return None

    def median_or_none(values: list) -> float | None:
        return float(statistics.median(values)) if values else None

    return {
        "Questions": len(counts),
        "Reachable count (mean)": sum(counts) / len(counts),
        "Reachable count (med)": median_or_none(counts),
        "Component size (mean)": sum(complexities) / len(complexities) if complexities else None,
        "Component size (med)": median_or_none(complexities),
    }


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
            f"experiment 9{caption_context}. \\emph{{Reachable count}} is the ground-truth "
            "answer (papers reachable from the source, excluding the source itself); "
            "\\emph{{Component size}} is the bidirectional source-component size the bucket "
            "was targeted on."
        ),
        label=sub_experiment_label(sub_tag, "construction_diagnostics", experiment_tag="9"),
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
                            "answer": question.get("answer"),
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
        question_type_label="reachability counting experiment",
        experiment_number="9",
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
        answer_field="answer",
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
        count_column_label="Reachable count",
        count_column_fn=lambda bucket_questions: format_mean_std(
            bucket_questions["answer"].map(
                lambda value: float(value[0]) if isinstance(value, list) and value else None
            )
        ),
        experiment_number="9",
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
        experiment_number="9",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report: figures
# ─────────────────────────────────────────────────────────────────────────────

FIGURE_METRIC_CI = {
    "accuracy_pct": lambda counts: wilson_ci_bounds_pct(counts["correct"], counts["total"]),
}


def report_bucket_figures(
    sub_results: pd.DataFrame,
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    column_values: list[str],
    sub_tag: str,
    axis_field: str = "bucket_label",
    axis_label: str = "Expected search-step bucket",
    title: str = "Experiment 9 Reachability Counting — Papers-only",
) -> dict[str, object]:
    if sub_results.empty:
        print(f"[{sub_tag}] No bucket figure written; no eval reports matched.")
        return {}

    hop_results = as_bucket_axis(sub_results, axis_field)
    figures_dir = sub_experiment_figures_dir(sub_tag)

    figure_artifacts = write_bucket_figure_artifacts(
        hop_results,
        figures_dir,
        "accuracy_by_search_bucket",
        bucket_labels=column_values,
        ordered_model_slugs=ordered_model_slugs,
        model_names=model_names,
        num_columns_span=2,
        title=f"{title} — [{sub_tag}]",
        error_bar=True,
        metric_ci=FIGURE_METRIC_CI,
        x_axis_label=axis_label,
        value_column="accuracy_pct",
        y_label="Accuracy (%)",
    )

    print(f"[{sub_tag}][accuracy_by_search_bucket] headline — exact-match accuracy")
    print_figure_path(figure_artifacts)
    display(Image(filename=str(figure_artifacts["png_path"])))

    return {"accuracy_by_search_bucket": figure_artifacts}


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
        chance_accuracy_pct=0.0,
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
    title: str = "Experiment 9",
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
        experiment_number="9",
    )
