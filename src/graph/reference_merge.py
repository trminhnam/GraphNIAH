from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from loguru import logger

from src.setup.constants import preprocess_reference_author, preprocess_reference_title


def reference_year(ref: Any) -> str:
    """Return a sortable year-like string from a reference object."""
    for attr in ("year", "published", "date"):
        value = getattr(ref, attr, None)
        if value:
            return str(value)
    return ""


def apply_reference_batch(
    *,
    network: Any,
    paper_node: str,
    refs: Iterable[Any],
    depth: int,
    paper_label: str = "",
    resolve_authors: Callable[[Any, str], list[str]] | None = None,
    skip_author: Callable[[str], bool] | None = None,
) -> int:
    """Insert references for one paper in a rollback-safe DB batch.

    The helper snapshots all in-memory mutable state touched by upsert helpers.
    If the DB transaction fails, every in-memory side-effect is restored before
    re-raising the exception.

    Reference-list author strings are stored as pending mentions for the cited
    paper. They are not inserted into the canonical authors table until that
    cited paper is explored and its first-page author list is available.
    """
    refs_list = sorted(list(refs), key=reference_year, reverse=True)
    if not refs_list:
        return 0

    ref_cap = network.limits.max_refs or 0
    inserted = 0
    pending_citations: list[tuple[str, str]] = []
    mutation_journal = network.start_mutation_journal()

    try:
        with network.db.batch():
            for ref in refs_list:
                if ref_cap and inserted >= ref_cap:
                    break

                ref_title = preprocess_reference_title(
                    (getattr(ref, "title", "") or "")
                    or getattr(ref, "raw", "")[:80]
                )
                if not ref_title.strip():
                    continue

                ref_node = network.upsert_paper(title=ref_title, depth=depth + 1)
                if ref_node is None:
                    continue

                authors = (
                    resolve_authors(ref, ref_title)
                    if resolve_authors is not None
                    else list(getattr(ref, "authors", []) or [])
                )

                ref_author_pos = 0
                for author_name in authors:
                    author_name = preprocess_reference_author(author_name)
                    if not author_name:
                        continue
                    if skip_author is not None and skip_author(author_name):
                        continue
                    ref_author_pos += 1
                    network.add_pending_reference_author(
                        ref_node,
                        paper_node,
                        position=ref_author_pos,
                        name=author_name,
                    )

                pending_citations.append((paper_node, ref_node))
                inserted += 1

    except Exception:
        mutation_journal.rollback()
        logger.warning(
            "DB batch for {} rolled back — restored state.",
            paper_label or paper_node,
        )
        raise
    else:
        mutation_journal.commit()

    # DB committed — apply staged graph mutations
    for citing_node, cited_node in pending_citations:
        network.add_citation(citing_node, cited_node)

    return inserted
