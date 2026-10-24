"""Helpers for resuming question-processing scripts from JSON output."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import orjson

from src.setup.constants import QUESTIONS_FIELD

QUESTION_ID_FIELDS = ("question_id", "qid", "id")
QUESTION_RESUME_FIELDS = (*QUESTION_ID_FIELDS, "path_index")
MISSING_RESUME_VALUE = object()


def stable_resume_value(value: Any) -> str:
    """Return a deterministic string for one resume-key value."""
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS, default=str).decode("utf-8")


def format_resume_value(value: Any) -> str:
    """Return one saved/current value for a mismatch message."""
    if value is MISSING_RESUME_VALUE:
        return "<missing>"
    return stable_resume_value(value)


def collect_resume_mismatches(
    saved: Any,
    current: Any,
    *,
    path: str = "",
) -> list[str]:
    """Return readable leaf-level differences between saved and current values."""
    if isinstance(saved, Mapping) and isinstance(current, Mapping):
        mismatches: list[str] = []
        for key in sorted(set(saved) | set(current), key=str):
            child_path = f"{path}.{key}" if path else str(key)
            mismatches.extend(
                collect_resume_mismatches(
                    saved.get(key, MISSING_RESUME_VALUE),
                    current.get(key, MISSING_RESUME_VALUE),
                    path=child_path,
                )
            )
        return mismatches
    if saved == current:
        return []
    return [
        (
            f"{path or 'value'}: saved={format_resume_value(saved)}; "
            f"current={format_resume_value(current)}"
        )
    ]


def format_resume_mismatches(saved: Any, current: Any) -> str:
    """Format saved/current differences for an exception message."""
    return "\n".join(
        f"  - {mismatch}" for mismatch in collect_resume_mismatches(saved, current)
    )


def question_resume_key_candidates(question: dict[str, Any]) -> list[str]:
    """Return stable question keys ordered from strongest to weakest."""
    keys: list[str] = []

    for field_name in QUESTION_ID_FIELDS:
        if field_name not in question:
            continue
        value = question[field_name]
        if value is None or str(value).strip() == "":
            continue
        keys.append(f"{field_name}:{stable_resume_value(value)}")

    if question.get("path_index") is not None:
        keys.append(
            "path:"
            + stable_resume_value(
                [
                    question.get("path_index"),
                    question.get("answer_type"),
                    question.get("answer_id"),
                ]
            )
        )

    content_key = {
        "question": question.get("question"),
        "answer_type": question.get("answer_type"),
        "answer_id": question.get("answer_id"),
        "node_ids": question.get("node_ids"),
    }
    if any(value is not None for value in content_key.values()):
        keys.append(f"content:{stable_resume_value(content_key)}")

    return keys


def question_identity_key_candidates(question: dict[str, Any]) -> list[str]:
    """Return explicit question-id resume keys, if present."""
    keys: list[str] = []
    for field_name in QUESTION_ID_FIELDS:
        if field_name not in question:
            continue
        value = question[field_name]
        if value is None or str(value).strip() == "":
            continue
        keys.append(f"{field_name}:{stable_resume_value(value)}")
    return keys


def map_completed_question_records(
    questions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return completed question records keyed by every available resume key."""
    records: dict[str, dict[str, Any]] = {}
    for question in questions:
        for key in question_resume_key_candidates(question):
            records.setdefault(key, question)
    return records


def find_completed_question_record(
    question: dict[str, Any],
    completed_by_key: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the completed record matching one source question, if any."""
    key_candidates = question_identity_key_candidates(question)
    if not key_candidates:
        key_candidates = question_resume_key_candidates(question)
    for key in key_candidates:
        if key in completed_by_key:
            return completed_by_key[key]
    return None


def split_resume_questions(
    source_questions: list[dict[str, Any]],
    completed_questions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return completed output records and unprocessed source questions."""
    completed_by_key = map_completed_question_records(completed_questions)
    if not completed_by_key:
        return [], source_questions

    resumed_questions: list[dict[str, Any]] = []
    pending_questions: list[dict[str, Any]] = []
    for question in source_questions:
        completed = find_completed_question_record(question, completed_by_key)
        if completed is None:
            pending_questions.append(question)
            continue
        resumed_questions.append(completed)

    return resumed_questions, pending_questions


def merge_questions_by_source_order(
    source_questions: list[dict[str, Any]],
    completed_questions: list[dict[str, Any]],
    processed_questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge resumed and newly processed records in source-question order."""
    completed_by_key = map_completed_question_records(
        completed_questions + processed_questions
    )
    output_questions: list[dict[str, Any]] = []
    for question in source_questions:
        completed = find_completed_question_record(question, completed_by_key)
        if completed is not None:
            output_questions.append(completed)
    return output_questions


def carry_question_resume_fields(
    output_question: dict[str, Any],
    source_question: dict[str, Any],
) -> dict[str, Any]:
    """Copy source identifier fields onto a processed output question."""
    for field_name in QUESTION_RESUME_FIELDS:
        if field_name in source_question:
            output_question[field_name] = source_question[field_name]
    return output_question


def load_json_output(path: Path) -> dict[str, Any] | None:
    """Load an existing JSON output file if it exists."""
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        data = orjson.loads(handle.read())
    if not isinstance(data, dict):
        raise ValueError(f"Resume output must be a JSON object: {path}")
    return data


def get_output_questions(data: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return dict question records from a JSON output payload."""
    if data is None:
        return []
    questions = data.get(QUESTIONS_FIELD)
    if not isinstance(questions, list):
        return []
    return [question for question in questions if isinstance(question, dict)]


def write_json_output(path: Path, payload: dict[str, Any], indent: int) -> None:
    """Atomically write a JSON payload to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        handle.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8"))
    temp_path.replace(path)
