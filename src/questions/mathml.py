"""Detect MathML markup in paper titles.

Papers whose stored title contains MathML markup (for example
``<mml:math ...><mml:mi>x</mml:mi></mml:math>``) render as unreadable context.
Existing sampled paths can be cleaned before question and distractor creation.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from src.setup.constants import nid_db_id

# A MathML element is an ``mml:``-prefixed tag or a ``math`` tag, possibly
# HTML-escaped. Plain comparisons such as "0 < x" do not match.
MATHML_MARKUP_PATTERN = re.compile(
    r"(?:<|&lt;)/?(?:mml:[a-z]|math(?:[\s>/]|&gt;))",
    re.IGNORECASE,
)


def title_has_mathml(title: Any) -> bool:
    """Return whether one title contains MathML markup."""
    return bool(MATHML_MARKUP_PATTERN.search(str(title or "")))


def stored_paper_titles(db: Any, db_ids: Sequence[str]) -> dict[str, Any]:
    """Return stored titles by database id, using whichever getter exists."""
    batch_loader = getattr(db, "get_paper_titles", None)
    if callable(batch_loader):
        return dict(batch_loader(list(db_ids)))
    title_loader = getattr(db, "get_paper_title", None)
    if callable(title_loader):
        return {db_id: title_loader(db_id) for db_id in db_ids}
    paper_loader = getattr(db, "get_paper", None)
    if callable(paper_loader):
        return {
            db_id: (paper_loader(db_id) or {}).get("title") for db_id in db_ids
        }
    return {}


def paper_ids_with_mathml_titles(
    db: Any,
    paper_ids: Sequence[str],
    *,
    cache: dict[str, bool] | None = None,
) -> frozenset[str]:
    """Read each title once when a construction-local cache is supplied."""
    normalized_ids = list(dict.fromkeys(str(paper_id) for paper_id in paper_ids))
    if not normalized_ids:
        return frozenset()
    checked = cache if cache is not None else {}
    missing_ids = [paper_id for paper_id in normalized_ids if paper_id not in checked]
    db_ids = [nid_db_id(paper_id) for paper_id in missing_ids]
    if not missing_ids:
        return frozenset(paper_id for paper_id in normalized_ids if checked[paper_id])
    titles_by_db_id = stored_paper_titles(db, db_ids)
    for paper_id, db_id in zip(missing_ids, db_ids):
        checked[paper_id] = title_has_mathml(titles_by_db_id.get(db_id))
    return frozenset(
        paper_id
        for paper_id in normalized_ids
        if checked[paper_id]
    )
