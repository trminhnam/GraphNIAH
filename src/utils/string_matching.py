from __future__ import annotations

import re
from typing import List

from rapidfuzz import fuzz

from src.setup.constants import FUZZY_MATCHING_THRESHOLD

COMPACT_MIN_LENGTH = 4
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def compact_alnum_string(text: str) -> str:
    """Return lowercase ASCII letters and digits only."""
    return NON_ALNUM_RE.sub("", text.lower())


def score_string_match(
    query: str,
    candidate: str,
    min_compact_length: int = COMPACT_MIN_LENGTH,
) -> float:
    """Score two strings using lowercase matching plus compact fallback."""
    query_lower = query.lower().strip()
    candidate_lower = candidate.lower().strip()
    if not query_lower or not candidate_lower:
        return 0.0
    if query_lower == candidate_lower:
        return 1.0

    query_compact = compact_alnum_string(query)
    candidate_compact = compact_alnum_string(candidate)
    can_use_compact = (
        len(query_compact) >= min_compact_length
        and len(candidate_compact) >= min_compact_length
    )
    if can_use_compact and query_compact == candidate_compact:
        return 1.0

    lower_score = fuzz.ratio(query_lower, candidate_lower) / 100.0
    if not can_use_compact:
        return lower_score

    compact_score = fuzz.ratio(query_compact, candidate_compact) / 100.0
    return max(lower_score, compact_score)


def search_strings(
    query: str,
    candidates: List[str],
    threshold: float = FUZZY_MATCHING_THRESHOLD,
) -> int:
    """Return the best matching candidate index, or -1 when none pass."""
    best_index = -1
    best_score = 0.0
    for index, candidate in enumerate(candidates):
        score = score_string_match(query, candidate)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index if best_score >= threshold else -1
