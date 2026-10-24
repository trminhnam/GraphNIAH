"""Shared reference-count validation for Semantic Scholar responses."""

from typing import Any

from src.graph.storage import DatabaseManager


def store_reference_validation(
    db: DatabaseManager,
    paper_db_id: str,
    paper: dict[str, Any] | None,
    references: list[dict[str, Any]],
) -> bool:
    """Store Semantic Scholar's reference count and list-completeness flag.

    Returns false without changing the row when ``referenceCount`` is absent.
    The caller controls the surrounding transaction.
    """
    reference_count = paper.get("referenceCount") if paper else None
    return store_reference_validation_counts(db, paper_db_id, reference_count, len(references))


def reference_validation_columns(
    reference_count: int | None,
    observed_count: int,
    *,
    restricted: bool = False,
) -> dict[str, Any]:
    """Return the ``reference_count``/``valid_references`` column values.

    Returns an empty dict when ``reference_count`` is absent, so callers can
    merge the result into a larger paper update. ``restricted=True`` (a
    --bulk-filter-neighbors-by-cutoff fetch) omits ``valid_references``:
    ``observed_count`` there only reflects a deliberately filtered subset, so
    comparing it against the true ``reference_count`` would always report a
    false "invalid" mismatch.
    """
    if reference_count is None:
        return {}

    normalized_count = int(reference_count)
    if restricted:
        return {"reference_count": normalized_count}
    return {
        "reference_count": normalized_count,
        "valid_references": observed_count == normalized_count,
    }


def store_reference_validation_counts(
    db: DatabaseManager,
    paper_db_id: str,
    reference_count: int | None,
    observed_count: int,
) -> bool:
    """Store a reported reference count and the observed bulk edge count."""
    columns = reference_validation_columns(reference_count, observed_count)
    if not columns:
        return False

    db.update_paper(paper_db_id, **columns)
    return True
