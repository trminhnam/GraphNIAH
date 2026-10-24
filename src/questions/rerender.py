"""Re-render stored reachability questions from their saved structure.

Set ``RE_RENDER_QUESTIONS=1`` to rebuild finished records instead of rerunning
the search. Every stored record keeps the titles and options it rendered, so a
re-render needs no graph search and no database:

- the question string is rebuilt with the current question templates,
- titles that still contain line breaks or repeated spaces are normalized the
  way the renderer normalizes them today, everywhere the record shows them.

Question ids, answers, needle papers and search trees stay unchanged, so the
distractors already built for a record remain valid.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping

from create_needles import normalize_title_whitespace
from src.questions.reachability_counting import (
    QUESTION_TYPE as COUNTING_QUESTION_TYPE,
    build_question_text as build_counting_question_text,
)
from src.questions.reachability_multiple import (
    QUESTION_TYPE as MULTIPLE_QUESTION_TYPE,
    build_question_text as build_multiple_question_text,
)
from src.questions.reachability_yesno import (
    QUESTION_TYPE as YESNO_QUESTION_TYPE,
    build_question_text as build_yesno_question_text,
)
from src.setup.constants import NODE_AUTHOR, nid_type

RE_RENDER_QUESTIONS_ENV = "RE_RENDER_QUESTIONS"
DISABLED_ENV_VALUES = frozenset({"", "0", "false", "no", "off"})
RERENDERABLE_QUESTION_TYPES = frozenset(
    {YESNO_QUESTION_TYPE, MULTIPLE_QUESTION_TYPE, COUNTING_QUESTION_TYPE}
)


def rerender_questions_enabled() -> bool:
    """Return whether ``RE_RENDER_QUESTIONS`` asks for a re-render."""
    value = os.environ.get(RE_RENDER_QUESTIONS_ENV, "")
    return value.strip().lower() not in DISABLED_ENV_VALUES


@dataclass(slots=True)
class RerenderStats:
    """Counts describing what one re-render pass changed."""

    total: int = 0
    normalized_titles: int = 0
    changed_questions: int = 0

    def summary(self) -> str:
        """Return a one-line summary for the console."""
        return (
            f"{self.total} record(s): {self.changed_questions} question(s) "
            f"rewritten, {self.normalized_titles} with normalized titles"
        )


def title_replacements(titles: Mapping[str, str]) -> dict[str, str]:
    """Return raw-to-normalized titles for titles rendered with odd spacing."""
    replacements: dict[str, str] = {}
    for raw_title in titles.values():
        raw = str(raw_title or "")
        normalized = normalize_title_whitespace(raw)
        if normalized and normalized != raw:
            replacements[raw] = normalized
    return replacements


def replace_in_value(value: Any, replacements: Mapping[str, str]) -> Any:
    """Return one record value with every raw title replaced by its normal form."""
    if isinstance(value, str):
        for raw in sorted(replacements, key=len, reverse=True):
            if raw in value:
                value = value.replace(raw, replacements[raw])
        return value
    if isinstance(value, dict):
        return {
            key: replace_in_value(item, replacements) for key, item in value.items()
        }
    if isinstance(value, list):
        return [replace_in_value(item, replacements) for item in value]
    return value


def node_display_label(node_id: str, visible_graph: Mapping[str, Any]) -> str:
    """Return the question label for one source or target node."""
    if nid_type(node_id) == NODE_AUTHOR:
        names = visible_graph.get("author_names_by_author") or {}
        return f"Author {names.get(node_id, node_id)}"
    titles = visible_graph.get("paper_titles_by_paper") or {}
    title = titles.get(node_id)
    if not title:
        raise ValueError(f"Record has no stored title for source/target {node_id!r}.")
    return f'Paper "{title}"'


def include_authorship_for(reachability: Mapping[str, Any]) -> bool:
    """Return whether one record's question names authorship relationships.

    Counting records store the flag under ``identity_rule``; the other types
    store it directly. A record without it cannot be re-rendered safely, since
    guessing would change which relationships the question asks about.
    """
    identity_rule = reachability.get("identity_rule") or {}
    if "author_identity_checks" in identity_rule:
        return bool(identity_rule["author_identity_checks"])
    if "author_identity_checks" in reachability:
        return bool(reachability["author_identity_checks"])
    raise ValueError("Record has no reachability.author_identity_checks flag.")


def build_question_for_record(record: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Return one record's current question text and the labels it names."""
    question_type = str(record.get("question_type") or "")
    reachability = record.get("reachability") or {}
    visible_graph = reachability.get("visible_graph") or {}
    source_node_id = str(reachability.get("source_node_id") or "")
    if not source_node_id:
        raise ValueError("Record has no reachability.source_node_id.")
    source_label = node_display_label(source_node_id, visible_graph)

    include_authorship = include_authorship_for(reachability)
    if question_type == COUNTING_QUESTION_TYPE:
        return (
            build_counting_question_text(
                source_label=source_label,
                source_type=str(
                    reachability.get("source_node_type") or nid_type(source_node_id)
                ),
                include_authorship=include_authorship,
            ),
            [source_label],
        )

    if question_type == MULTIPLE_QUESTION_TYPE:
        options = [
            (str(option.get("label") or ""), str(option.get("text") or ""))
            for option in (record.get("multiple_choice") or {}).get("options") or []
        ]
        if not options:
            raise ValueError("Multiple-choice record has no stored options.")
        return (
            build_multiple_question_text(
                source_label=source_label,
                options=options,
                include_authorship=include_authorship,
            ),
            [source_label, *(text for _label, text in options)],
        )

    target_node_id = str(reachability.get("target_node_id") or "")
    if not target_node_id:
        raise ValueError("Yes/no record has no reachability.target_node_id.")
    target_label = node_display_label(target_node_id, visible_graph)
    return (
        build_yesno_question_text(
            source_label=source_label,
            target_label=target_label,
            include_authorship=include_authorship,
        ),
        [source_label, target_label],
    )


def rerender_question_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return one normalized, re-rendered record."""
    if str(record.get("question_type") or "") not in RERENDERABLE_QUESTION_TYPES:
        return dict(record)

    visible_graph = (record.get("reachability") or {}).get("visible_graph") or {}
    replacements = title_replacements(
        visible_graph.get("paper_titles_by_paper") or {}
    )
    updated = replace_in_value(dict(record), replacements) if replacements else dict(record)

    question, labels = build_question_for_record(updated)
    previous_question = str(updated.get("question") or "")
    for label in labels:
        if label not in previous_question:
            raise ValueError(
                f"Re-rendered label {label!r} is missing from the stored question "
                f"of {updated.get('qid')!r}."
            )
    updated["question"] = question
    return updated


def rerender_needle_records(
    records: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], RerenderStats]:
    """Return re-rendered records keyed by qid, plus what changed."""
    stats = RerenderStats(total=len(records))
    output: dict[str, dict[str, Any]] = {}
    for qid, record in records.items():
        updated = rerender_question_record(record)
        if updated.get("needles") != record.get("needles"):
            stats.normalized_titles += 1
        if updated.get("question") != record.get("question"):
            stats.changed_questions += 1
        output[qid] = updated
    return output, stats
