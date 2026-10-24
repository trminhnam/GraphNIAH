"""Cutoff-aware breadth-first traversal from sampled path papers."""

from __future__ import annotations

import sqlite3
from datetime import date
from time import monotonic

from loguru import logger

from sample_paths import parse_published_date_value

SQL_ID_BATCH_SIZE = 900


def select_cached_bulk_misses(
    connection: sqlite3.Connection, paper_ids: list[str], index_created_at: str
) -> set[str]:
    """Read cached lookup failures for these papers and the current bulk index."""
    result = set()
    for start in range(0, len(paper_ids), SQL_ID_BATCH_SIZE):
        group = paper_ids[start : start + SQL_ID_BATCH_SIZE]
        placeholders = ",".join("?" for _ in group)
        rows = connection.execute(
            "SELECT paper_id FROM bulk_lookup_misses "
            f"WHERE index_created_at = ? AND paper_id IN ({placeholders})",
            [index_created_at, *group],
        )
        result.update(str(row["paper_id"]) for row in rows)
    return result


def collect_neighbor_metadata(
    connection: sqlite3.Connection,
    neighbor_ids: set[str],
    cutoff_date: date,
) -> dict[str, str]:
    """Load one bounded batch of candidate neighbors and apply the cutoff."""
    if not neighbor_ids:
        return {}
    placeholders = ",".join("?" for _ in neighbor_ids)
    rows = connection.execute(
        "SELECT id, semantic_scholar_id, published FROM papers "
        f"WHERE id IN ({placeholders})",
        list(neighbor_ids),
    )
    result = {}
    for row in rows:
        published = parse_published_date_value(row["published"])
        if published is not None and published >= cutoff_date:
            result[str(row["id"])] = str(row["semantic_scholar_id"] or "")
    return result


def collect_next_path_level(
    connection: sqlite3.Connection,
    source_ids: list[str],
    visited: set[str],
    cutoff_date: date,
) -> dict[str, str]:
    """Read both citation directions using indexed, bounded SQLite queries.

    Filter stored edges too: a fully explored paper may have older neighbors.
    Read IDs from the covering citation indexes first, skipping visited papers
    before loading metadata. Keep at most one metadata batch of candidate IDs.
    """
    result: dict[str, str] = {}
    candidates: set[str] = set()
    started = last_log = monotonic()
    logger.info("Reading stored neighbors | {} source papers", len(source_ids))
    for start in range(0, len(source_ids), SQL_ID_BATCH_SIZE):
        group = source_ids[start : start + SQL_ID_BATCH_SIZE]
        placeholders = ",".join("?" for _ in group)
        for source, target in (
            ("citing_paper_id", "cited_paper_id"),
            ("cited_paper_id", "citing_paper_id"),
        ):
            rows = connection.execute(
                f"SELECT {target} AS id FROM citations "
                f"WHERE {source} IN ({placeholders})",
                group,
            )
            for row in rows:
                paper_id = str(row["id"])
                if paper_id in visited or paper_id in result:
                    continue
                candidates.add(paper_id)
                if len(candidates) == SQL_ID_BATCH_SIZE:
                    result.update(
                        collect_neighbor_metadata(connection, candidates, cutoff_date)
                    )
                    candidates.clear()
        result.update(collect_neighbor_metadata(connection, candidates, cutoff_date))
        candidates.clear()
        now = monotonic()
        processed = start + len(group)
        if now - last_log >= 10 or processed == len(source_ids):
            logger.info(
                "Reading stored neighbors | {}/{} source papers | "
                "{} new papers after cutoff | {:.1f}s",
                processed,
                len(source_ids),
                len(result),
                now - started,
            )
            last_log = now
    return result
