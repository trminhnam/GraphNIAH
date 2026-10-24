#!/usr/bin/env python3
"""Score the run_inference.py output of one create_dataset.v1 experiment.

Each prediction record already carries its gold answer and the config it was
built with, so no separate ground-truth file is needed. Every record is scored
with eval.py's ``score_prediction``. Failed requests (no answer) are excluded
from accuracy and confidence intervals, and counted separately. Returned
answers that are wrong or unparseable still count as incorrect.

The report gives accuracy with a 95% Wilson score interval: overall, by the
studied factor (--vary), and by gold answer for yes/no and multiple choice.
Each studied value averages over answered examples with that value. The other
factors were drawn at random, so they get no breakdown of their own; each
scored record keeps its config for any further analysis.
With --vary none, no factor is studied and by_studied_value is empty.
Total response cost is the sum of known costs in the prediction records,
including cached responses. It excludes separate retry costs and is not the
new spending for the latest inference run. Missing prices are reported.

Example (activate the niah environment first)::

    python run_eval.py --vary question-depth \
        --predictions output/experiments.v1/.../vary=question-depth/inference/model=gpt-5.6-sol.json \
        --output output/experiments.v1/.../vary=question-depth/eval/model=gpt-5.6-sol.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import orjson

from eval import score_prediction
from run_inference import OUTPUT_SCHEMA_VERSION, write_output
from src.llm.accounting import aggregate_accounting
from src.logging.console import log_line
from src.questions.dataset_pool import portable_path
from src.visualization.artifacts import wilson_ci_bounds_pct

# --vary value -> the config field it studies.
VARY_KEYS = {
    "question-depth": "question_depth",
    "node-count": "component_node_count",
    "question-format": "question_format",
    "distractor-type": "distractor_type",
}
GOLD_LABEL_FORMATS = ("reachability-yesno", "reachability-multiple")


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [record for record in scored if not record["failed_request"]]
    total = len(scored)
    correct = sum(record["correct"] for record in scored)
    lower, upper = wilson_ci_bounds_pct(correct, total)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "ci95": [lower / 100, upper / 100] if total else None,
    }


def natural_key(value: str) -> tuple[int, Any]:
    """Sort numbers by value, before any other label."""
    return (0, int(value)) if value.isdigit() else (1, value)


def grouped(
    scored: list[dict[str, Any]], key_of: Any
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in scored:
        groups[key_of(record)].append(record)
    return {key: summarize(groups[key]) for key in sorted(groups, key=natural_key)}


def score_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for record in records:
        score = score_prediction(
            record.get("raw_answer"),
            record["answer"],
            str(record.get("answer_type") or ""),
            record.get("question_type"),
        )
        scored.append(
            {
                "qid": record["qid"],
                "question_type": record["question_type"],
                "config": record["config"],
                "gold": record["answer"],
                "extracted": score["extracted"],
                "correct": score["correct"],
                "match_type": score["match_type"],
                "parsed": score["parsed"],
                "answer_box_present": score["answer_box_present"],
                "format_mismatch": score["format_mismatch"],
                "failed_request": not record.get("raw_answer"),
            }
        )
    return scored


def evaluate(predictions: dict[str, Any], vary: str) -> dict[str, Any]:
    studied_key = None if vary == "none" else VARY_KEYS[vary]
    scored = score_records(predictions["records"])
    accounting = aggregate_accounting(predictions["records"])
    return {
        "vary": vary,
        "model": predictions["metadata"].get("model"),
        "cost": {
            "total_cost": accounting["total_total_cost"],
            "records_with_cost": accounting["records_with_cost"],
            "records_without_cost": accounting["records_without_cost"],
            "cost_complete": accounting["cost_complete"],
        },
        "overall": summarize(scored),
        "by_studied_value": grouped(
            scored, lambda record: str(record["config"][studied_key])
        )
        if studied_key is not None
        else {},
        "by_gold_answer": {
            question_format: grouped(
                [r for r in scored if r["question_type"] == question_format],
                lambda record: str(record["gold"][0]),
            )
            for question_format in GOLD_LABEL_FORMATS
            if any(r["question_type"] == question_format for r in scored)
        },
        "counts": {
            "failed_requests": sum(r["failed_request"] for r in scored),
            "unparseable": sum(
                not r["parsed"] for r in scored if not r["failed_request"]
            ),
            "format_mismatch": sum(r["format_mismatch"] for r in scored),
        },
        "records": scored,
    }


def format_row(label: str, stats: dict[str, Any]) -> str:
    label = label.removeprefix("reachability-")
    if not stats["total"]:
        return f"    {label:<14} n=0"
    lower, upper = stats["ci95"]
    return (
        f"    {label:<14} n={stats['total']:<5} accuracy={stats['accuracy']:.3f} "
        f"[{lower:.3f}, {upper:.3f}]"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--predictions", type=Path, required=True, help="run_inference.py output."
    )
    parser.add_argument(
        "--vary",
        choices=[*VARY_KEYS, "none"],
        required=True,
        help="The studied factor, or none when every factor was drawn at random.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictions = orjson.loads(args.predictions.read_bytes())
    if predictions.get("schema_version") != OUTPUT_SCHEMA_VERSION:
        raise ValueError(f"{args.predictions} is not a run_inference.py output.")
    report = {
        "predictions": portable_path(args.predictions),
        **evaluate(predictions, args.vary),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_output(args.output, report)
    log_line(f"[Eval] {report['model']} | vary {args.vary} | {args.predictions}")
    log_line(format_row("overall", report["overall"]))
    cost = report["cost"]
    total_cost = cost["total_cost"]
    cost_text = "unavailable" if total_cost is None else f"${total_cost:.6f}"
    if total_cost is not None and not cost["cost_complete"]:
        cost_text += " (partial)"
    log_line(
        f"  total response cost (USD): {cost_text} | "
        f"priced records={cost['records_with_cost']}/{len(report['records'])} "
        "(includes cached responses; excludes retries)"
    )
    if args.vary != "none":
        log_line(f"  by {args.vary}:")
    for value, stats in report["by_studied_value"].items():
        log_line(format_row(value, stats))
    counts = report["counts"]
    log_line(
        f"  failed requests={counts['failed_requests']} (excluded from accuracy) "
        f"unparseable={counts['unparseable']} format mismatch={counts['format_mismatch']}"
    )
    log_line(f"  report: {args.output}")


if __name__ == "__main__":
    main()
