"""Extract the citation graph among papers published in a date window.

Used by construct_paths_from_bulk_graph.py. A paper is eligible when:

- its publicationDate is a complete YYYY-MM-DD date inside the window
  (papers with a missing, partial, or invalid date are excluded);
- its title is not empty and has no MathML markup (the rule of
  src/questions/mathml.py), because question construction rejects such papers;
- with category "cs", its fieldsOfStudy include Computer Science.

Only citations between two eligible papers are kept, so an ineligible paper
can never be on a path. The filter is applied once, together with the date
cutoff, before any edge is read.

Spark work: one scan of the papers table (only the filter columns) and one
scan of the citation table's two id columns, filtered map-side by semi-joins
against the eligible ids when they fit the broadcast budget. Only the kept
edges reach Python. The edge list is cached as an .npz file and reused while
the index's createdAt, the window, the category, and the filter version stay
the same.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import orjson

from src.graph.bulk_degree_distribution import (
    BROADCAST_ESTIMATED_BYTES_PER_ID,
    broadcast_memory_budget,
)
from src.questions.mathml import MATHML_MARKUP_PATTERN

# Bump when the eligibility rule changes, so old caches are rebuilt.
WINDOW_GRAPH_CACHE_VERSION = 1
WINDOW_CATEGORIES = ("all", "cs")
# Java regex form of the MathML title rule (same pattern, case-insensitive).
MATHML_TITLE_SPARK_PATTERN = "(?i)" + MATHML_MARKUP_PATTERN.pattern


def eligible_window_papers(
    papers: Any,
    start_date: date,
    end_date: date | None,
    category: str,
) -> Any:
    """Return the distinct corpusIds of the papers that may be on a path."""
    from pyspark.sql import functions as F

    if category not in WINDOW_CATEGORIES:
        raise ValueError(f"category must be one of {WINDOW_CATEGORIES}")
    publication_date = F.trim(F.col("publicationDate"))
    day = F.expr("try_cast(trim(publicationDate) AS DATE)")
    title = F.trim(F.coalesce(F.col("title"), F.lit("")))
    eligible = (
        papers.filter(F.col("corpusId").isNotNull())
        .filter(publication_date.rlike(r"^\d{4}-\d{2}-\d{2}$"))
        .filter(day >= F.lit(start_date))
        .filter((title != "") & ~title.rlike(MATHML_TITLE_SPARK_PATTERN))
    )
    if end_date is not None:
        eligible = eligible.filter(day <= F.lit(end_date))
    if category == "cs":
        if "fieldsOfStudy" not in papers.columns:
            raise ValueError("The papers index needs fieldsOfStudy for category 'cs'.")
        eligible = eligible.filter(
            F.array_contains("fieldsOfStudy", "Computer Science")
        )
    return eligible.select("corpusId").distinct()


def extract_window_edges(
    papers: Any,
    citations: Any,
    start_date: date,
    end_date: date | None,
    category: str,
    *,
    broadcast_max_mib: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (citing, cited, number of eligible papers) for the window.

    Edges are the citation rows whose two papers are eligible, without
    self-citations. Repeated rows are left for the caller to merge.
    """
    from pyspark import StorageLevel
    from pyspark.sql import functions as F

    eligible = eligible_window_papers(papers, start_date, end_date, category).persist(
        StorageLevel.MEMORY_AND_DISK
    )
    try:
        num_eligible = eligible.count()
        if num_eligible == 0:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty.copy(), 0
        budget = broadcast_memory_budget(papers.sparkSession, broadcast_max_mib)
        broadcastable = num_eligible * BROADCAST_ESTIMATED_BYTES_PER_ID <= budget

        def id_filter(column_name: str) -> Any:
            ids = eligible.select(F.col("corpusId").alias(column_name))
            return F.broadcast(ids) if broadcastable else ids.hint("merge")

        edges = (
            citations.select("citingCorpusId", "citedCorpusId")
            .join(id_filter("citingCorpusId"), "citingCorpusId", "left_semi")
            .join(id_filter("citedCorpusId"), "citedCorpusId", "left_semi")
            .filter(F.col("citingCorpusId") != F.col("citedCorpusId"))
            .select("citingCorpusId", "citedCorpusId")
        )
        frame = edges.toPandas()
        return (
            frame["citingCorpusId"].to_numpy(dtype=np.int64),
            frame["citedCorpusId"].to_numpy(dtype=np.int64),
            num_eligible,
        )
    finally:
        eligible.unpersist()


def window_graph_cache_path(
    cache_dir: Path,
    category: str,
    start_date: date,
    end_date: date | None,
) -> Path:
    end = end_date.isoformat() if end_date is not None else "open"
    return cache_dir / f"{category}.{start_date.isoformat()}.{end}.npz"


def window_graph_cache_key(
    index_created_at: str,
    category: str,
    start_date: date,
    end_date: date | None,
) -> dict[str, Any]:
    """Settings that must match for a cached edge list to be reused."""
    return {
        "cache_version": WINDOW_GRAPH_CACHE_VERSION,
        "index_created_at": index_created_at,
        "category": category,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat() if end_date is not None else None,
    }


def load_window_graph_cache(
    path: Path,
    key: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    """Return the cached (citing, cited, metadata), or None when stale or absent."""
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        metadata = orjson.loads(str(data["metadata"]))
        if any(metadata.get(name) != value for name, value in key.items()):
            return None
        return data["citing"], data["cited"], metadata


def save_window_graph_cache(
    path: Path,
    citing: np.ndarray,
    cited: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Write the cache atomically, so an interrupted run leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz.tmp")
    try:
        with os.fdopen(handle, "wb") as output:
            np.savez(
                output,
                citing=np.asarray(citing, dtype=np.int64),
                cited=np.asarray(cited, dtype=np.int64),
                metadata=np.array(orjson.dumps(metadata).decode("utf-8")),
            )
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)
