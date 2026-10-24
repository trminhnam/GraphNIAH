#!/usr/bin/env python3
"""Cut a full-prompts file down to a fixed, label-balanced set of questions.

Reads a ``full_prompts.json`` written by ``build_full_prompts.py`` and keeps the
first ``--num-samples`` questions in the order ``inference.py`` sends them:
categorical answer labels (YES/NO, or option letters) are interleaved first,
then the prefix is kept. This is the same question set, in the same order, that
``inference.py --max-samples N`` picks from the full file, so every model run
on the cut file sees exactly the same examples.

The output keeps the input structure. Only the ``questions`` list is shorter,
and ``metadata.full_prompts.total_questions`` is updated to match it.

Example
-------

    python cut_full_prompts.py \\
        --input output/experiments/.../full_prompts.json \\
        --output output/experiments/.../full_prompts.100.json \\
        --num-samples 100
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import orjson

from src.setup.constants import METADATA_FIELD, QUESTIONS_FIELD, TOTAL_QUESTIONS_FIELD
from src.utils.stratify import format_label_counts, stratification_label, stratify_by_label

# Same key as build_full_prompts.FULL_PROMPTS_FIELD; importing that module is slow.
FULL_PROMPTS_FIELD = "full_prompts"
# Exit status when the input has fewer questions than --num-samples. The
# experiment inference.sh scripts skip such a file instead of stopping.
TOO_FEW_QUESTIONS_EXIT_CODE = 3


class TooFewQuestionsError(ValueError):
    """The input has fewer questions than the requested cut size."""


def select_samples(
    questions: list[dict[str, Any]],
    num_samples: int,
) -> list[dict[str, Any]]:
    """Return the first ``num_samples`` questions in inference.py's stratified order."""
    if num_samples < 1:
        raise ValueError(f"--num-samples must be at least 1, got {num_samples}.")
    if num_samples > len(questions):
        raise TooFewQuestionsError(
            f"Asked for {num_samples} question(s), but the input has only {len(questions)}."
        )
    return stratify_by_label(questions, stratification_label)[:num_samples]


def cut_payload(payload: dict[str, Any], num_samples: int) -> dict[str, Any]:
    """Return a copy of a full-prompts payload that keeps only the selected questions."""
    selected = select_samples(payload[QUESTIONS_FIELD], num_samples)
    cut = {**payload, QUESTIONS_FIELD: selected}
    metadata = payload.get(METADATA_FIELD)
    full_prompts_metadata = (
        metadata.get(FULL_PROMPTS_FIELD) if isinstance(metadata, dict) else None
    )
    if isinstance(full_prompts_metadata, dict) and TOTAL_QUESTIONS_FIELD in full_prompts_metadata:
        cut[METADATA_FIELD] = {
            **metadata,
            FULL_PROMPTS_FIELD: {**full_prompts_metadata, TOTAL_QUESTIONS_FIELD: len(selected)},
        }
    return cut


def describe_questions(questions: list[dict[str, Any]]) -> str:
    """Return the question count plus its label counts when every question has a label."""
    labels = [stratification_label(question) for question in questions]
    if any(label is None for label in labels):
        return f"{len(questions)} question(s), not stratified (no categorical labels)"
    return f"{len(questions)} question(s): {format_label_counts(Counter(labels))}"


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Keep the first N questions of a full_prompts.json in the stratified "
            "order used by inference.py, and write them in the same format."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="full_prompts.json written by build_full_prompts.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the cut file. Must differ from --input.",
    )
    parser.add_argument(
        "--num-samples",
        "--num_samples",
        dest="num_samples",
        type=int,
        required=True,
        help="Number of questions to keep.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = arg_parser().parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("--input and --output must be different files.")

    payload = orjson.loads(args.input.read_bytes())
    if not isinstance(payload, dict) or not isinstance(payload.get(QUESTIONS_FIELD), list):
        raise ValueError(
            f"{args.input}: expected a full_prompts.json object with a '{QUESTIONS_FIELD}' list."
        )

    try:
        cut = cut_payload(payload, args.num_samples)
    except TooFewQuestionsError as error:
        print(f"cut_full_prompts: {args.input}: {error}", file=sys.stderr)
        raise SystemExit(TOO_FEW_QUESTIONS_EXIT_CODE) from error
    except ValueError as error:
        raise SystemExit(f"cut_full_prompts: {args.input}: {error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        orjson.dumps(cut, option=orjson.OPT_INDENT_2).decode("utf-8"),
        encoding="utf-8",
    )
    print(f"Input : {describe_questions(payload[QUESTIONS_FIELD])}  ({args.input})")
    print(f"Output: {describe_questions(cut[QUESTIONS_FIELD])}  ({args.output})")


if __name__ == "__main__":
    main()
