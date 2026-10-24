"""Run-level report over the full-prompt files one experiment produced.

A create-data run writes one ``full_prompts.json`` per condition, hours apart,
and each stage only reports on itself. This module reads the finished files back
and renders one table plus the warnings that need a human decision, so a long run
ends with a verdict instead of a wall of stage output.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import orjson
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

PROMPT_FILE_NAME = "full_prompts.json"
PROMPTS_DIR_NAME = "prompts"
PLOTS_DIR_NAME = "plots"
CONDITION_SEPARATOR = "="
METADATA_FIELD = "metadata"
QUESTIONS_FIELD = "questions"
FULL_PROMPTS_FIELD = "full_prompts"
TOTAL_QUESTIONS_FIELD = "total_questions"
SKIPPED_QIDS_FIELD = "skipped_qids"
USER_MESSAGE_FIELD = "user_message"
ANSWER_FIELD = "answer"
NEEDLE_ITEMS_FIELD = "needle_items"
NEEDLE_TOKENS_FIELD = "needle_tokens"
DISTRACTOR_ITEMS_FIELD = "distractor_items"
DISTRACTOR_TOKENS_FIELD = "distractor_tokens"

UNEVEN_FILL_RATIO = 0.5
MAX_LABELS_REPORTED = 4
MISSING_VALUE = "-"

TABLE_COLUMNS = (
    ("built", "built"),
    ("skip", "skipped"),
    ("needles/prompt", "needle_items_mean"),
    ("tok/needle", "tokens_per_needle"),
    ("distr/prompt", "distractor_items_mean"),
    ("tok/distr", "tokens_per_distractor"),
    ("median chars", "median_chars"),
    ("min/med", "min_fill_ratio"),
)

# One histogram per metric, built from the per-question values inside one full_prompts.json.
# (file name, x-axis label, item-count field, token field or None for a raw count histogram)
HISTOGRAM_METRICS = (
    ("needles_per_prompt.png", "needle items per prompt", NEEDLE_ITEMS_FIELD, None),
    ("tokens_per_needle.png", "tokens per needle item", NEEDLE_ITEMS_FIELD, NEEDLE_TOKENS_FIELD),
    ("distractors_per_prompt.png", "distractor items per prompt", DISTRACTOR_ITEMS_FIELD, None),
    ("tokens_per_distractor.png", "tokens per distractor item", DISTRACTOR_ITEMS_FIELD, DISTRACTOR_TOKENS_FIELD),
)


def find_prompt_files(root: str | Path) -> list[Path]:
    """Return every full-prompt file under one experiment root, sorted."""
    return sorted(Path(root).rglob(PROMPT_FILE_NAME))


def parse_condition(path: str | Path, root: str | Path) -> dict[str, str]:
    """Return the ``key=value`` path segments that name one condition."""
    path = Path(path)
    root = Path(root)
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path

    condition: dict[str, str] = {}
    for part in relative.parts:
        key, separator, value = part.partition(CONDITION_SEPARATOR)
        if separator and key and value:
            condition[key] = value
    return condition


def per_question_counts(questions: list[dict[str, Any]], field: str) -> list[int]:
    """Return the raw per-question values of one integer field."""
    return [
        question[field]
        for question in questions
        if isinstance(question, dict) and isinstance(question.get(field), int)
    ]


def per_question_ratios(
    questions: list[dict[str, Any]],
    count_field: str,
    token_field: str,
) -> list[float]:
    """Return the per-question token-per-item ratio, skipping questions with no items."""
    ratios = []
    for question in questions:
        if not isinstance(question, dict):
            continue
        count = question.get(count_field)
        tokens = question.get(token_field)
        if isinstance(count, int) and count > 0 and isinstance(tokens, int):
            ratios.append(tokens / count)
    return ratios


def mean_field(questions: list[dict[str, Any]], field: str) -> float | None:
    """Return the mean of one integer field across the questions that carry it."""
    values = per_question_counts(questions, field)
    if not values:
        return None
    return statistics.fmean(values)


def tokens_per_item(
    questions: list[dict[str, Any]],
    count_field: str,
    token_field: str,
) -> float | None:
    """Return the mean token size of one kind of context item."""
    items = 0
    tokens = 0
    for question in questions:
        if not isinstance(question, dict):
            continue
        count = question.get(count_field)
        token_total = question.get(token_field)
        if not isinstance(count, int) or not isinstance(token_total, int):
            continue
        items += count
        tokens += token_total
    if items <= 0:
        return None
    return tokens / items


def answer_labels(questions: list[dict[str, Any]]) -> dict[str, int]:
    """Return how many prompts carry each answer label."""
    labels: Counter[str] = Counter()
    for question in questions:
        if not isinstance(question, dict):
            continue
        answer = question.get(ANSWER_FIELD)
        if isinstance(answer, list):
            answer = answer[0] if answer else None
        if answer is None:
            continue
        labels[str(answer)] += 1
    if len(labels) > MAX_LABELS_REPORTED:
        return {}
    return dict(labels)


def summarize_prompt_file(path: str | Path, root: str | Path) -> dict[str, Any]:
    """Return the report row for one full-prompt file."""
    path = Path(path)
    data = orjson.loads(path.read_text(encoding="utf-8"))
    questions = data.get(QUESTIONS_FIELD) or []
    metadata = data.get(METADATA_FIELD) or {}
    full_prompts = metadata.get(FULL_PROMPTS_FIELD) or {}
    skipped_qids = full_prompts.get(SKIPPED_QIDS_FIELD) or []

    chars = sorted(
        len(str(question.get(USER_MESSAGE_FIELD) or ""))
        for question in questions
        if isinstance(question, dict)
    )
    median_chars = chars[len(chars) // 2] if chars else None
    min_fill_ratio = (
        chars[0] / median_chars if chars and median_chars else None
    )

    return {
        "path": path,
        "condition": parse_condition(path, root),
        "questions": questions,
        "built": full_prompts.get(TOTAL_QUESTIONS_FIELD, len(questions)),
        "skipped": len(skipped_qids) if isinstance(skipped_qids, list) else 0,
        "needle_items_mean": mean_field(questions, NEEDLE_ITEMS_FIELD),
        "distractor_items_mean": mean_field(questions, DISTRACTOR_ITEMS_FIELD),
        "tokens_per_needle": tokens_per_item(
            questions,
            NEEDLE_ITEMS_FIELD,
            NEEDLE_TOKENS_FIELD,
        ),
        "tokens_per_distractor": tokens_per_item(
            questions,
            DISTRACTOR_ITEMS_FIELD,
            DISTRACTOR_TOKENS_FIELD,
        ),
        "median_chars": median_chars,
        "min_fill_ratio": min_fill_ratio,
        "labels": answer_labels(questions),
    }


def parse_filter_argument(raw: str) -> tuple[str, set[str]]:
    """Return one ``key=value[,value...]`` --filter argument as (key, allowed values)."""
    key, separator, values = raw.partition(CONDITION_SEPARATOR)
    if not separator or not key:
        raise ValueError(f"Invalid --filter (expected key=value): {raw!r}")
    return key, {value for value in values.split(",") if value}


def matches_filters(condition: dict[str, str], filters: dict[str, set[str]]) -> bool:
    """Return whether one row's condition satisfies every key=value(s) filter."""
    return all(condition.get(key) in values for key, values in filters.items())


def plot_directory(path: Path, root: Path) -> Path:
    """Return the plots-tree directory that mirrors one prompt file's condition path."""
    relative_parts = path.relative_to(root).parent.parts
    if relative_parts and relative_parts[0] == PROMPTS_DIR_NAME:
        relative_parts = relative_parts[1:]
    return Path(root, PLOTS_DIR_NAME, *relative_parts)


def save_histograms(path: Path, root: Path, questions: list[dict[str, Any]]) -> Path | None:
    """Write one histogram PNG per context-item metric, mirrored under plots/."""
    plot_dir = plot_directory(path, root)
    written = False
    for file_name, label, count_field, token_field in HISTOGRAM_METRICS:
        values = (
            per_question_counts(questions, count_field)
            if token_field is None
            else per_question_ratios(questions, count_field, token_field)
        )
        if not values:
            continue
        if not written:
            plot_dir.mkdir(parents=True, exist_ok=True)
            written = True
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(values, bins="auto")
        ax.set_xlabel(label)
        ax.set_ylabel("prompts")
        ax.set_title(label)
        fig.tight_layout()
        fig.savefig(plot_dir / file_name, dpi=150)
        plt.close(fig)
    return plot_dir if written else None


def varying_condition_keys(rows: list[dict[str, Any]]) -> list[str]:
    """Return the condition keys that differ across the reported files."""
    keys: list[str] = []
    for row in rows:
        for key in row["condition"]:
            if key not in keys:
                keys.append(key)
    return [
        key
        for key in keys
        if len({row["condition"].get(key) for row in rows}) > 1
    ]


def shared_condition(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Return the condition keys every reported file has in common."""
    if not rows:
        return {}
    varying = set(varying_condition_keys(rows))
    shared = {
        key: value
        for key, value in rows[0]["condition"].items()
        if key not in varying
    }
    return {
        key: value
        for key, value in shared.items()
        if all(row["condition"].get(key) == value for row in rows)
    }


def format_cell(key: str, value: Any) -> str:
    """Return one table cell as display text."""
    if value is None:
        return MISSING_VALUE
    if key == "min_fill_ratio":
        return f"{value:.0%}"
    if isinstance(value, float):
        return f"{value:,.1f}"
    return f"{value:,}"


def format_report_table(rows: list[dict[str, Any]]) -> list[str]:
    """Return the report table as display lines."""
    if not rows:
        return []

    condition_keys = varying_condition_keys(rows)
    headers = [*condition_keys, *(header for header, _key in TABLE_COLUMNS)]
    table_rows = [
        [
            *(row["condition"].get(key, MISSING_VALUE) for key in condition_keys),
            *(format_cell(key, row[key]) for _header, key in TABLE_COLUMNS),
        ]
        for row in rows
    ]

    widths = [
        max(len(header), *(len(table_row[index]) for table_row in table_rows))
        for index, header in enumerate(headers)
    ]

    def format_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip()

    return [
        format_row(headers),
        "  ".join("-" * width for width in widths),
        *(format_row(table_row) for table_row in table_rows),
    ]


def condition_label(row: dict[str, Any], condition_keys: list[str]) -> str:
    """Return one short name for the condition a warning belongs to."""
    keys = condition_keys or list(row["condition"])
    parts = [f"{key}={row['condition'][key]}" for key in keys if key in row["condition"]]
    return " ".join(parts) or row["path"].name


def collect_warnings(rows: list[dict[str, Any]]) -> list[str]:
    """Return the findings that need a decision before running inference."""
    condition_keys = varying_condition_keys(rows)
    warnings: list[str] = []
    for row in rows:
        label = condition_label(row, condition_keys)
        if row["skipped"]:
            warnings.append(
                f"{label}: {row['skipped']} question(s) dropped "
                f"({row['built']} built)"
            )
        ratio = row["min_fill_ratio"]
        if ratio is not None and ratio < UNEVEN_FILL_RATIO:
            warnings.append(
                f"{label}: shortest prompt is {ratio:.0%} of the median "
                "-- context length is not constant"
            )
        labels = row["labels"]
        if len(labels) > 1 and len(set(labels.values())) > 1:
            balance = " ".join(
                f"{name}={count}" for name, count in sorted(labels.items())
            )
            warnings.append(f"{label}: answer balance is uneven ({balance})")
    return warnings


def build_report(root: str | Path, filters: dict[str, set[str]] | None = None) -> list[str]:
    """Return the full report for one experiment root as display lines."""
    root = Path(root)
    rows = [summarize_prompt_file(path, root) for path in find_prompt_files(root)]
    if not rows:
        return [f"[report] no {PROMPT_FILE_NAME} found under {root}"]

    if filters:
        rows = [row for row in rows if matches_filters(row["condition"], filters)]
        if not rows:
            return [f"[report] no {PROMPT_FILE_NAME} matched the given filters under {root}"]

    lines = [f"[report] {len(rows)} prompt file(s) under {root}"]
    plotted = [
        plot_dir
        for row in rows
        if (plot_dir := save_histograms(row["path"], root, row["questions"])) is not None
    ]
    if plotted:
        lines.append(f"  plots: {len(plotted)} file(s) of histograms under {root / PLOTS_DIR_NAME}")
    shared = shared_condition(rows)
    if shared:
        lines.append(
            "  shared: " + " ".join(f"{key}={value}" for key, value in shared.items())
        )
    lines.append("")
    lines.extend(format_report_table(rows))

    warnings = collect_warnings(rows)
    lines.append("")
    if warnings:
        lines.append(f"WARNINGS ({len(warnings)})")
        lines.extend(f"  - {warning}" for warning in warnings)
    else:
        lines.append("WARNINGS (0)")
    return lines
