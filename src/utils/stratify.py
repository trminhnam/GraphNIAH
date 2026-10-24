"""Interleave prompt items by their categorical answer label.

Shared by ``inference.py`` (to order and truncate its input) and
``cut_full_prompts.py`` (to save the same subset as a file). Kept free of heavy
imports so the cut script starts quickly.
"""

from collections.abc import Callable
from typing import Any

ANSWER_FIELD = "answer"

# Answer types whose ``answer[0]`` is a categorical label rather than free text.
STRATIFIED_ANSWER_TYPES = {"REACHABILITY_YESNO", "REACHABILITY_MULTIPLE"}
# Labels that have a conventional order; anything else (A, B, C, D) sorts.
LABEL_PRIORITY = ("YES", "NO")


def stratification_label(item: dict[str, Any]) -> str | None:
    """Return the categorical answer label used to interleave one input item.

    Only categorical answer types are interleaved; for those ``answer[0]`` is the
    label itself (``YES``/``NO``, or the correct option letter). Everything else
    returns ``None`` so free-form question types keep their input file order.
    """
    if str(item.get("answer_type") or "").strip() not in STRATIFIED_ANSWER_TYPES:
        return None
    answer = item.get(ANSWER_FIELD)
    if isinstance(answer, str):
        candidate = answer
    elif isinstance(answer, (list, tuple)) and answer:
        candidate = answer[0]
    else:
        return None
    return str(candidate or "").strip().upper() or None


def label_sort_key(label: str) -> tuple[int, str]:
    """Order YES before NO; every other label set falls back to sorting."""
    if label in LABEL_PRIORITY:
        return (LABEL_PRIORITY.index(label), label)
    return (len(LABEL_PRIORITY), label)


def stratify_by_label(
    entries: list[Any],
    label_of: Callable[[Any], str | None],
    completed_counts: dict[str, int] | None = None,
) -> list[Any]:
    """Interleave entries so consecutive ones rotate through the answer labels.

    Yields ``[YES, NO, YES, NO, ...]`` or ``[A, B, C, D, A, ...]`` by always
    taking from whichever label is furthest behind, counting labels already
    finished in ``completed_counts``. Each label keeps its input order, so a
    resumed run continues a label's queue where it stopped instead of restarting
    it, and a run interrupted at 5 YES / 4 NO resumes on NO. Entries are
    returned unchanged when any of them has no categorical label.
    """
    labels = [label_of(entry) for entry in entries]
    if any(label is None for label in labels):
        return list(entries)

    queues: dict[str, list[Any]] = {}
    for label, entry in zip(labels, entries):
        queues.setdefault(label, []).append(entry)
    counts = {label: 0 for label in queues}
    for label, count in (completed_counts or {}).items():
        if label in counts:
            counts[label] = count

    ordered: list[Any] = []
    while any(queues.values()):
        label = min(
            (label for label, queue in queues.items() if queue),
            key=lambda value: (counts[value], label_sort_key(value)),
        )
        ordered.append(queues[label].pop(0))
        counts[label] += 1
    return ordered


def format_label_counts(counts: dict[str, int]) -> str:
    """Return ``LABEL=count`` pairs in the canonical label order."""
    return ", ".join(
        f"{label}={count}"
        for label, count in sorted(counts.items(), key=lambda item: label_sort_key(item[0]))
    )
