"""Helpers for scholarly-graph reachability multiple-choice questions."""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx

from src.questions.reachability_yesno import VisibleReachabilityGraph

QUESTION_TYPE = "reachability-multiple"
ANSWER_TYPE = "REACHABILITY_MULTIPLE"
CONSTRUCTION = "disjoint-option-trees"
MIN_NUM_OPTIONS = 2
MAX_NUM_OPTIONS = 26
DEFAULT_NONE_OPTION = 0
SUPPORTED_NONE_OPTIONS = (0, 1)
NONE_OPTION_TEXT = "None of these targets is reachable from the source."
OPTION_LABELS = tuple(chr(ord("A") + index) for index in range(MAX_NUM_OPTIONS))
AUTHOR_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def option_labels(num_options: int) -> tuple[str, ...]:
    """Return validated uppercase labels for ``num_options`` choices."""
    if not MIN_NUM_OPTIONS <= num_options <= MAX_NUM_OPTIONS:
        raise ValueError(
            f"num_options must be between {MIN_NUM_OPTIONS} and "
            f"{MAX_NUM_OPTIONS}, inclusive."
        )
    return OPTION_LABELS[:num_options]


def displayed_num_options(num_options: int, none_option: int) -> int:
    """Validate and return the total number of displayed choices."""
    if none_option not in SUPPORTED_NONE_OPTIONS or isinstance(none_option, bool):
        raise ValueError("none_option must be either 0 or 1.")
    minimum = 1 if none_option else MIN_NUM_OPTIONS
    maximum = MAX_NUM_OPTIONS - none_option
    if (
        not isinstance(num_options, int)
        or isinstance(num_options, bool)
        or not minimum <= num_options <= maximum
    ):
        raise ValueError(
            f"num_options must be between {minimum} and {maximum}, inclusive, "
            f"when none_option={none_option}."
        )
    return num_options + none_option


def build_question_text(
    *,
    source_label: str,
    options: Sequence[tuple[str, str]],
    include_authorship: bool = True,
) -> str:
    """Render one reachability query and its labeled target options."""
    relationship_types = (
        "`cites`, `cited_by`, `author_of`, or `authored_by`"
        if include_authorship
        else "`cites` or `cited_by`"
    )
    option_text = "\n".join(f"{label}. {text}" for label, text in options)
    none_instruction = (
        ' If no listed target is reachable, select the "None of these targets is '
        'reachable from the source" option.'
        if any(text == NONE_OPTION_TEXT for _label, text in options)
        else ""
    )
    return (
        "Using only the relationships described in the context, which of the following is "
        f"reachable from {source_label} by following one or more "
        f"{relationship_types} relationships?{none_instruction}\n\n{option_text}"
    )


def deduplicate_answer_texts(values: Sequence[str]) -> list[str]:
    """Return non-empty answer strings once, preserving their order."""
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def author_name_answer_variants(name: str) -> list[str]:
    """Return conservative full-name, comma, and initials variants."""
    normalized = " ".join(str(name or "").split())
    if not normalized:
        return []
    variants = [normalized]
    if "," in normalized:
        return variants

    parts = normalized.split()
    if len(parts) < 2:
        return variants
    suffix_count = int(
        parts[-1].rstrip(".").casefold() in AUTHOR_SUFFIXES
        and len(parts) >= 3
    )
    family_start = len(parts) - 1 - suffix_count
    given_parts = parts[:family_start]
    family_parts = parts[family_start:]
    if not given_parts:
        return variants

    family = " ".join(family_parts)
    given = " ".join(given_parts)
    initials = " ".join(
        f"{character.upper()}."
        for part in given_parts
        if (character := next((value for value in part if value.isalpha()), ""))
    )
    variants.append(f"{family}, {given}")
    if initials:
        variants.append(f"{family}, {initials}")
    return deduplicate_answer_texts(variants)


def answer_variants(
    label: str,
    option_text: str,
    aliases: Sequence[str] = (),
) -> list[str]:
    """Return label-only, standalone-content, and labeled-content answers."""
    normalized_label = str(label or "").strip().upper()
    if normalized_label not in OPTION_LABELS:
        raise ValueError(f"Unsupported option label: {label!r}.")
    normalized_text = str(option_text or "").strip()
    if not normalized_text:
        raise ValueError("option_text must be non-empty.")
    answers = [
        normalized_label,
        f"{normalized_label}.",
        f"{normalized_label})",
    ]
    for text in deduplicate_answer_texts([normalized_text, *aliases]):
        answers.extend(
            [
                text,
                f"{normalized_label}. {text}",
                f"{normalized_label}) {text}",
            ]
        )
    return deduplicate_answer_texts(answers)


def reachable_option_labels(
    visible_graph: VisibleReachabilityGraph,
    *,
    source_node_id: str,
    options: Sequence[tuple[str, str]],
) -> tuple[str, ...]:
    """Return option labels reachable from ``source_node_id`` in order."""
    graph = visible_graph.graph
    if source_node_id not in graph:
        return ()
    reachable: list[str] = []
    for label, node_id in options:
        if node_id in graph and nx.has_path(graph, source_node_id, node_id):
            reachable.append(str(label).upper())
    return tuple(reachable)


def graph_invariant_holds(
    visible_graph: VisibleReachabilityGraph,
    *,
    source_node_id: str,
    options: Sequence[tuple[str, str]],
    correct_option: str,
    none_option_label: str | None = None,
) -> bool:
    """Return whether the declared target, or no displayed target, is reachable."""
    normalized_correct = str(correct_option or "").strip().upper()
    reachable = reachable_option_labels(
        visible_graph,
        source_node_id=source_node_id,
        options=options,
    )
    normalized_none_label = str(none_option_label or "").strip().upper()
    if normalized_none_label and normalized_correct == normalized_none_label:
        return reachable == ()
    return reachable == (normalized_correct,)
