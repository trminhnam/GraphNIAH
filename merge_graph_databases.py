#!/usr/bin/env python3
"""Merge per-machine citation graph databases into one canonical database.

Each machine runs ``construct_paths_from_semantic_scholar.py`` with its own hop
count and produces (a) a graph database and (b) one or more
``sampled_paths.json`` files whose ``node_id`` values reference that machine's
local UUIDs. This script folds those databases into a single canonical database
and rewrites the sampled paths so their node IDs point at the canonical rows.

The canonical database always wins: source rows may only fill fields that are
empty in the canonical row, never overwrite them.

Entity resolution runs in this order, so the merge is idempotent (re-running
maps every row to itself):

1. ``semantic_scholar_id``
2. ``arxiv_id`` (papers) / ``orcid`` (authors)
3. the source UUID, when that row already exists in the canonical database
4. papers only: normalized title + publication year + author names, when
   exactly one canonical paper matches (see "Title/author fallback" below)
5. otherwise insert, keeping the source UUID when it is free

Title/author fallback (papers only)
------------------------------------
When neither side has a matching ``semantic_scholar_id``/``arxiv_id`` and the
source UUID is not already claimed in canonical, a source paper is still
merged into an existing canonical paper if all of the following hold:

- their titles are equal after Unicode-aware normalization;
- neither side's ``semantic_scholar_id``/``arxiv_id`` contradicts the other
  (both present and different disqualifies the candidate);
- their publication years are compatible (blank on either side is a
  wildcard; both present and different disqualifies the candidate);
- their normalized author-name sets are compatible: one set must fully
  contain the other, so a source/canonical pair where one side lists fewer
  authors (e.g. an "et al." truncation) can still match, as long as every
  name that IS listed agrees.

If more than one canonical paper satisfies all of this, the match is
ambiguous: the source row is inserted as new instead of guessing, and a
warning is logged. Authors are never matched this way — only their existing
``semantic_scholar_id``/``orcid``/UUID resolution applies to them.

Arguments
---------
--canonical-db PATH
    Authoritative database that every source is merged into. Created with the
    standard schema if it does not exist yet.
--source DB [PATHS ...]
    A source database, optionally followed by the ``sampled_paths.json`` files
    (or directories to search for them) belonging to that database. Repeat once
    per machine. Paths files must be listed with the database whose UUIDs they
    reference.
--remap-out PATH
    Write source-specific old-to-new node ID mappings as JSON. A flat mapping
    is also included for IDs whose meaning is unambiguous across all sources.
--refresh-node-payloads
    After remapping, rebuild each step's node metadata from the canonical
    database. Use when canonical and source disagree on titles or authors.
--dry-run
    Run the whole merge against an isolated database copy, report what would
    change, and leave the canonical database and sampled paths untouched.

Example
-------
python merge_graph_databases.py \\
    --canonical-db resources/semantic_scholar_graph/semantic_scholar_paths.db \\
    --source incoming/machine-a/graph.db incoming/machine-a \\
    --source incoming/machine-b/graph.db incoming/machine-b/hops=10/sampled_paths.json
"""

from __future__ import annotations

import argparse
import orjson
import re
import sqlite3
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator

from loguru import logger

from src.graph.sampled_paths import apply_node_id_updates_to_walk_data
from src.graph.duplicate_comparison import compare_title_batch, normalize_title_batch
from src.graph.readers import DBReader, build_node_payload
from src.graph.storage import DatabaseManager, copy_sqlite_database, generate_uuid
from src.questions.reachability_counting import normalize_author_list
from src.questions.reachability_yesno import normalize_visible_identity
from src.setup import bootstrap_runtime
from src.setup.constants import (
    NODE_AUTHOR,
    NODE_PAPER,
    author_nid,
    is_uuid_text,
    nid_db_id,
    nid_type,
    paper_nid,
)
from src.utils.process_pool import batches, process_results

SAMPLED_PATHS_GLOB = "**/sampled_paths.json"

# Columns copied from a source row into an existing canonical row when the
# canonical value is empty. Uniquely indexed columns are handled separately.
PAPER_FILL_COLUMNS = (
    "title",
    "abstract",
    "published",
    "venue",
    "pdf_url",
    "text_path",
    "reference_count",
    "valid_references",
)
PAPER_UNIQUE_COLUMNS = ("semantic_scholar_id", "arxiv_id")
AUTHOR_FILL_COLUMNS = ("name",)
AUTHOR_UNIQUE_COLUMNS = ("semantic_scholar_id", "orcid")


@dataclass
class MergeStats:
    papers_matched: int = 0
    papers_matched_by_identity: int = 0
    papers_inserted: int = 0
    papers_uuid_reassigned: int = 0
    authors_matched: int = 0
    authors_inserted: int = 0
    authors_uuid_reassigned: int = 0
    fields_filled: int = 0
    unique_conflicts: int = 0
    citations_inserted: int = 0
    authorships_written: int = 0
    variants_inserted: int = 0
    pending_inserted: int = 0
    expansions_carried: int = 0

    def summary_lines(self) -> list[str]:
        return [
            f"papers    : {self.papers_matched} matched "
            f"({self.papers_matched_by_identity} by title/author identity), "
            f"{self.papers_inserted} inserted "
            f"({self.papers_uuid_reassigned} needed a new UUID)",
            f"authors   : {self.authors_matched} matched, {self.authors_inserted} inserted "
            f"({self.authors_uuid_reassigned} needed a new UUID)",
            f"edges     : {self.citations_inserted} citations, "
            f"{self.authorships_written} authorships, "
            f"{self.variants_inserted} name variants, "
            f"{self.pending_inserted} pending reference authors",
            f"metadata  : {self.fields_filled} empty fields filled, "
            f"{self.unique_conflicts} unique-column conflicts skipped",
            f"expansion : {self.expansions_carried} completed expansions carried over",
        ]


@dataclass
class SourceBundle:
    db_path: Path
    paths_files: list[Path] = field(default_factory=list)


def open_source(db_path: Path) -> sqlite3.Connection:
    """Open a source database read-only so a merge can never mutate it."""
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA query_only=ON")
    # Keep one stable view across papers, authors, and edges if the source is
    # still open on the machine where the merge runs.
    conn.execute("BEGIN")
    return conn


def iter_rows(conn: sqlite3.Connection, sql: str) -> Iterator[sqlite3.Row]:
    cursor = conn.execute(sql)
    while True:
        rows = cursor.fetchmany(2000)
        if not rows:
            return
        yield from rows


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def unique_owner(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    value: str,
) -> str | None:
    """Return the canonical row id already holding *value* in *column*."""
    if not value:
        return None
    row = conn.execute(
        f"SELECT id FROM {table} WHERE {column} = ? AND {column} != '' LIMIT 1",
        (value,),
    ).fetchone()
    return str(row["id"]) if row else None


def require_uuid_source_id(table: str, db_id: str) -> None:
    """Reject source rows whose primary key is not a UUID.

    Legacy incremental IDs would be copied verbatim into the canonical
    database and silently break the UUID graph store.
    """
    if not is_uuid_text(db_id):
        raise SystemExit(
            f"Source {table}.id is not a UUID: {db_id!r}. "
            "Legacy incremental IDs are no longer supported."
        )


def row_exists(conn: sqlite3.Connection, table: str, db_id: str) -> bool:
    row = conn.execute(f"SELECT 1 FROM {table} WHERE id = ? LIMIT 1", (db_id,)).fetchone()
    return row is not None


def fill_empty_fields(
    conn: sqlite3.Connection,
    table: str,
    canonical_id: str,
    source_row: sqlite3.Row,
    fill_columns: tuple[str, ...],
    unique_columns: tuple[str, ...],
    stats: MergeStats,
) -> None:
    """Copy source values into canonical columns that are still empty."""
    canonical_row = conn.execute(
        f"SELECT * FROM {table} WHERE id = ?",
        (canonical_id,),
    ).fetchone()
    if canonical_row is None:
        return

    updates: dict[str, Any] = {}
    source_columns = set(source_row.keys())
    canonical_columns = set(canonical_row.keys())
    for column in fill_columns:
        # Older graph databases may predate optional columns such as
        # reference_count/valid_references. Their absence must not make a
        # cross-database merge fail.
        if column not in source_columns or column not in canonical_columns:
            continue
        source_value = source_row[column]
        canonical_value = canonical_row[column]
        source_present = source_value is not None and (
            not isinstance(source_value, str) or bool(source_value.strip())
        )
        canonical_empty = canonical_value is None or (
            isinstance(canonical_value, str) and not canonical_value.strip()
        )
        if source_present and canonical_empty:
            updates[column] = source_value

    for column in unique_columns:
        source_value = str(source_row[column] or "").strip()
        canonical_value = str(canonical_row[column] or "").strip()
        if not source_value:
            continue
        if canonical_value:
            if canonical_value != source_value:
                logger.warning(
                    "Keeping canonical {}.{}={} for {}; source has {}",
                    table,
                    column,
                    canonical_value,
                    canonical_id,
                    source_value,
                )
                stats.unique_conflicts += 1
            continue
        owner = unique_owner(conn, table, column, source_value)
        if owner is not None and owner != canonical_id:
            # Canonical already stores this identifier on a different row.
            # Merging the two rows is out of scope; report and move on.
            logger.warning(
                "Skipping {}.{}={} for {}: already held by {}",
                table,
                column,
                source_value,
                canonical_id,
                owner,
            )
            stats.unique_conflicts += 1
            continue
        updates[column] = source_value

    if table == "papers":
        # Older graph databases may predate explored_after_cutoff; treat it
        # as absent (never cutoff-restricted) rather than fail the merge.
        has_cutoff_column = "explored_after_cutoff" in canonical_columns
        source_after_cutoff = (
            bool(source_row["explored_after_cutoff"])
            if "explored_after_cutoff" in source_columns
            else False
        )
        canonical_after_cutoff = (
            bool(canonical_row["explored_after_cutoff"]) if has_cutoff_column else False
        )
        if source_row["explored"] and not canonical_row["explored"]:
            updates["explored"] = 1
            if has_cutoff_column and source_after_cutoff:
                updates["explored_after_cutoff"] = 1
        elif (
            has_cutoff_column
            and canonical_row["explored"]
            and canonical_after_cutoff
            and source_row["explored"]
            and not source_after_cutoff
        ):
            # Canonical was only explored under --bulk-filter-neighbors-by-cutoff;
            # source has the same paper fully explored -- prefer that.
            updates["explored_after_cutoff"] = None
        source_depth = int(source_row["depth"] if source_row["depth"] is not None else -1)
        canonical_depth = int(canonical_row["depth"] if canonical_row["depth"] is not None else -1)
        if canonical_depth < 0 <= source_depth:
            updates["depth"] = source_depth

    if not updates:
        return

    assignments = ", ".join(f"{column} = ?" for column in updates)
    conn.execute(
        f"UPDATE {table} SET {assignments} WHERE id = ?",
        (*updates.values(), canonical_id),
    )
    stats.fields_filled += len(updates)


def uuid_match_is_compatible(
    conn: sqlite3.Connection,
    table: str,
    source_row: sqlite3.Row,
    identity_columns: tuple[str, ...],
) -> bool:
    """Check that a same-UUID canonical row is really the same entity.

    A shared UUID normally means both databases descend from one ancestor. But
    if the two rows carry *different* non-empty external identifiers they are
    distinct entities that collided, and merging them would fuse two papers.
    """
    canonical_row = conn.execute(
        f"SELECT * FROM {table} WHERE id = ?",
        (str(source_row["id"]),),
    ).fetchone()
    if canonical_row is None:
        return False
    for column in identity_columns:
        source_value = str(source_row[column] or "").strip()
        canonical_value = str(canonical_row[column] or "").strip()
        if source_value and canonical_value and source_value != canonical_value:
            return False
    return True


def resolve_entity(
    conn: sqlite3.Connection,
    table: str,
    source_row: sqlite3.Row,
    identity_columns: tuple[str, ...],
) -> str | None:
    """Find the canonical row for *source_row*, or None when it is new."""
    for column in identity_columns:
        value = str(source_row[column] or "").strip()
        owner = unique_owner(conn, table, column, value)
        if owner is not None:
            return owner

    source_id = str(source_row["id"])
    if row_exists(conn, table, source_id) and uuid_match_is_compatible(
        conn, table, source_row, identity_columns
    ):
        return source_id
    return None


def insert_entity(
    conn: sqlite3.Connection,
    table: str,
    source_row: sqlite3.Row,
    columns: tuple[str, ...],
) -> tuple[str, bool]:
    """Insert a new canonical row, reusing the source UUID when it is free.

    Returns the canonical id and whether a fresh UUID had to be minted.
    """
    source_id = str(source_row["id"])
    reassigned = row_exists(conn, table, source_id)
    canonical_id = generate_uuid() if reassigned else source_id

    values = [canonical_id]
    for column in columns:
        value = source_row[column]
        if column in {"arxiv_id", "semantic_scholar_id", "orcid"}:
            value = str(value or "").strip()
        # ``orcid`` is NULL-able and UNIQUE; empty strings would collide.
        if column == "orcid":
            value = str(value or "").strip() or None
        values.append(value)

    placeholders = ", ".join("?" * (len(columns) + 1))
    conn.execute(
        f"INSERT INTO {table} (id, {', '.join(columns)}) VALUES ({placeholders})",
        tuple(values),
    )
    return canonical_id, reassigned


PUBLISHED_YEAR_RE = re.compile(r"\d{4}")


def published_year(value: Any) -> str:
    """Return the leading 4-digit year in *value*, or '' when none is found."""
    match = PUBLISHED_YEAR_RE.match(str(value or "").strip())
    return match.group(0) if match else ""


def paper_author_names(conn: sqlite3.Connection, paper_id: str) -> tuple[str, ...]:
    """Return this paper's raw author names from *conn*'s own authorship table."""
    rows = conn.execute(
        """SELECT authors.name FROM authorship
           JOIN authors ON authors.id = authorship.author_id
           WHERE authorship.paper_id = ?""",
        (paper_id,),
    ).fetchall()
    return tuple(str(row["name"] or "") for row in rows)


def author_lists_match(
    source_authors: tuple[str, ...],
    candidate_authors: tuple[str, ...],
) -> bool:
    """True if one normalized author-name set fully contains the other.

    A subset (rather than exact-equality or a fuzzy overlap ratio) tolerates
    one side truncating a long author list (e.g. "et al.") while still
    requiring every name that IS listed to agree. Neither side may be empty:
    a paper with no recorded authors offers no corroborating evidence, and an
    empty set is trivially a subset of anything.
    """
    if not source_authors or not candidate_authors:
        return False
    source_set = set(source_authors)
    candidate_set = set(candidate_authors)
    return source_set <= candidate_set or candidate_set <= source_set


def identity_columns_conflict(row_a: Any, row_b: Any) -> bool:
    """True if both rows carry a non-blank semantic_scholar_id/arxiv_id that differ."""
    for column in PAPER_UNIQUE_COLUMNS:
        value_a = str(row_a[column] or "").strip()
        value_b = str(row_b[column] or "").strip()
        if value_a and value_b and value_a != value_b:
            return True
    return False


def published_years_conflict(row_a: Any, row_b: Any) -> bool:
    """True if both rows carry a publication year and the years differ."""
    year_a = published_year(row_a["published"])
    year_b = published_year(row_b["published"])
    return bool(year_a and year_b and year_a != year_b)


def paper_identity_is_compatible(
    conn: sqlite3.Connection,
    source_row: sqlite3.Row,
    canonical_id: str,
) -> bool:
    """Reject a title/author candidate contradicted by year or a differing id."""
    canonical_row = conn.execute(
        "SELECT * FROM papers WHERE id = ?", (canonical_id,)
    ).fetchone()
    if canonical_row is None:
        return False
    if identity_columns_conflict(source_row, canonical_row):
        return False
    return not published_years_conflict(source_row, canonical_row)


def resolve_paper_by_identity(
    canonical: sqlite3.Connection,
    source: sqlite3.Connection,
    source_row: sqlite3.Row,
) -> str | None:
    """Fall back to title + publication year + author-name matching.

    Tried only after ``resolve_entity`` fails to find a match via
    semantic_scholar_id/arxiv_id/UUID. Catches papers discovered
    independently on two machines through different APIs, so neither side
    recorded a shared external id. See the module docstring's "Title/author
    fallback" section for the exact matching rules.
    """
    title_key = normalize_visible_identity(source_row["title"])
    if not title_key:
        return None
    source_authors = normalize_author_list(
        paper_author_names(source, str(source_row["id"]))
    )
    if not source_authors:
        # No corroborating author list on the source side; title alone is
        # too weak a signal to safely fuse two graph entities.
        return None

    candidates = canonical.execute(
        "SELECT id FROM papers WHERE title != '' AND visible_title_key(title) = ?",
        (title_key,),
    ).fetchall()

    matches: list[str] = []
    for candidate in candidates:
        candidate_id = str(candidate["id"])
        if not paper_identity_is_compatible(canonical, source_row, candidate_id):
            continue
        candidate_authors = normalize_author_list(
            paper_author_names(canonical, candidate_id)
        )
        if author_lists_match(source_authors, candidate_authors):
            matches.append(candidate_id)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        logger.warning(
            "Title/author identity for {!r} matches {} canonical papers; "
            "skipping automatic merge",
            str(source_row["title"]),
            len(matches),
        )
    return None


@dataclass(frozen=True)
class DuplicatePaperGroup:
    """Papers within one database the merge's title/author fallback would fuse."""

    paper_ids: tuple[str, ...]
    title: str
    is_clique: bool  # True when every pair in the group is individually compatible


def inspect_duplicate_paper_group(
    conn: sqlite3.Connection,
    paper_ids: tuple[str, ...],
) -> DuplicatePaperGroup | None:
    """Inspect a small known group using the duplicate-audit safety rules.

    This is the targeted counterpart of ``find_duplicate_paper_groups``. It
    is used when expansion resolves one title-only row to an external ID that
    another row already owns. The rows are mergeable only when every pair has
    the same normalized title, compatible external IDs and publication years,
    and compatible non-empty author lists.
    """
    unique_ids = tuple(dict.fromkeys(paper_ids))
    if len(unique_ids) < 2:
        return None
    placeholders = ", ".join("?" for _ in unique_ids)
    rows = conn.execute(
        f"""SELECT id, title, published, semantic_scholar_id, arxiv_id
            FROM papers WHERE id IN ({placeholders})""",
        unique_ids,
    ).fetchall()
    if len(rows) != len(unique_ids):
        return None

    title_keys = {normalize_visible_identity(row["title"]) for row in rows}
    if len(title_keys) != 1 or not next(iter(title_keys)):
        return None

    authors_by_id = {
        str(row["id"]): normalize_author_list(
            paper_author_names(conn, str(row["id"]))
        )
        for row in rows
    }
    is_clique = all(
        not identity_columns_conflict(row_a, row_b)
        and not published_years_conflict(row_a, row_b)
        and author_lists_match(
            authors_by_id[str(row_a["id"])],
            authors_by_id[str(row_b["id"])],
        )
        for row_a, row_b in combinations(rows, 2)
    )
    return DuplicatePaperGroup(
        paper_ids=tuple(sorted(unique_ids)),
        title=str(rows[0]["title"]),
        is_clique=is_clique,
    )


def find_duplicate_paper_groups(
    conn: sqlite3.Connection,
    max_group_size: int = 50,
    *,
    num_workers: int = 1,
) -> tuple[list[DuplicatePaperGroup], list[tuple[str, int]]]:
    """Find papers in *conn* that ``resolve_paper_by_identity`` would treat as one paper.

    Applies the exact same checks the merge's title/author fallback uses when
    merging a second database in (matching normalized title, no contradicting
    semantic_scholar_id/arxiv_id/publication year, compatible author-name
    sets) but pairwise within a single already-existing database, instead of
    one incoming row against canonical.

    Only papers sharing a normalized title are ever compared, so the scan
    avoids materializing the whole ``papers``/``authorship`` tables: it first
    computes each paper's title key with one pass, then only pulls the
    columns and author names of papers whose title key collides with another
    paper's. On the ~8M-row production database this two-pass approach keeps
    the run to a few minutes; comparing every paper against every other
    paper's author list would not finish in reasonable time.

    A title shared by more than *max_group_size* papers is excluded from the
    (quadratic) pairwise check and reported separately instead: at that size
    it is almost always a parsing artifact (an organization name or
    boilerplate string that ended up in the ``title`` column, e.g. "OUP
    accepted manuscript" or "OpenAI") rather than the same paper appearing
    many times, and checking every pair would be prohibitively slow.

    *conn* must allow creating TEMP tables — do not pass a connection with
    ``PRAGMA query_only=ON`` (that blocks temp-table writes too). Opening the
    main database with the read-only ``mode=ro`` URI is enough to guarantee
    this function cannot modify it.

    ``num_workers`` defaults to 1 for existing callers. Larger values use spawned
    processes for title normalization and author/pair comparisons; only the
    caller's connection reads the candidate tables and writes TEMP tables.

    Returns ``(groups, oversized_titles)``, where ``oversized_titles`` lists
    ``(title, paper_count)`` for the titles skipped for being too large.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    conn.create_function("visible_title_key", 1, normalize_visible_identity)
    conn.executescript(
        """
        DROP TABLE IF EXISTS temp.paper_title_keys;
        DROP TABLE IF EXISTS temp.title_key_counts;
        DROP TABLE IF EXISTS temp.dup_candidate_ids;
        """
    )

    started = time.time()
    logger.info("Computing normalized title keys for every paper...")
    if num_workers == 1:
        conn.execute(
            """CREATE TEMP TABLE paper_title_keys AS
               SELECT id, visible_title_key(title) AS tkey
               FROM papers WHERE title != ''"""
        )
    else:
        conn.execute("CREATE TEMP TABLE paper_title_keys (id TEXT, tkey TEXT)")
        conn.execute("SAVEPOINT title_normalization")
        rows = conn.execute("SELECT id, title FROM papers WHERE title != ''")
        tasks = batches(((str(row[0]), str(row[1])) for row in rows), 4096)
        try:
            with process_results(normalize_title_batch, tasks, num_workers) as results:
                for normalized in results:
                    conn.executemany(
                        "INSERT INTO paper_title_keys VALUES (?, ?)", normalized
                    )
        finally:
            rows.close()
            conn.execute("RELEASE title_normalization")
    conn.execute("CREATE INDEX idx_paper_title_keys_tkey ON paper_title_keys(tkey)")
    logger.info("  done in {:.0f}s", time.time() - started)

    conn.execute(
        """CREATE TEMP TABLE title_key_counts AS
           SELECT tkey, COUNT(*) AS c FROM paper_title_keys GROUP BY tkey"""
    )
    conn.execute("CREATE INDEX idx_title_key_counts_tkey ON title_key_counts(tkey)")

    oversized_titles = [
        (str(row["title"]), int(row["c"]))
        for row in conn.execute(
            """SELECT papers.title AS title, tc.c AS c
               FROM title_key_counts tc
               JOIN paper_title_keys ptk ON ptk.tkey = tc.tkey
               JOIN papers ON papers.id = ptk.id
               WHERE tc.c > ?
               GROUP BY tc.tkey""",
            (max_group_size,),
        ).fetchall()
    ]

    conn.execute(
        """CREATE TEMP TABLE dup_candidate_ids AS
           SELECT ptk.id AS id, ptk.tkey AS tkey
           FROM paper_title_keys ptk
           JOIN title_key_counts tc ON tc.tkey = ptk.tkey
           WHERE tc.c > 1 AND tc.c <= ?""",
        (max_group_size,),
    )
    conn.execute("CREATE INDEX idx_dup_candidate_ids_id ON dup_candidate_ids(id)")

    candidate_rows = conn.execute(
        """SELECT papers.id, papers.title, papers.published,
                  papers.semantic_scholar_id, papers.arxiv_id, dci.tkey AS tkey
           FROM papers
           JOIN dup_candidate_ids dci ON dci.id = papers.id"""
    ).fetchall()
    logger.info(
        "{} candidate papers to check, from {} title collision group(s) "
        "({} oversized title(s) skipped)",
        len(candidate_rows),
        len({row["tkey"] for row in candidate_rows}),
        len(oversized_titles),
    )

    started = time.monotonic()
    raw_authors: dict[str, list[str]] = {}
    for row in conn.execute(
        """SELECT authorship.paper_id AS paper_id, authors.name AS name
           FROM authorship
           JOIN dup_candidate_ids dci ON dci.id = authorship.paper_id
           JOIN authors ON authors.id = authorship.author_id"""
    ):
        raw_authors.setdefault(str(row["paper_id"]), []).append(str(row["name"] or ""))
    logger.info("Candidate author reads: {:.1f}s", time.monotonic() - started)

    groups_by_tkey: dict[str, list[sqlite3.Row]] = {}
    for row in candidate_rows:
        groups_by_tkey.setdefault(str(row["tkey"]), []).append(row)

    started = time.monotonic()
    tasks = batches(
        (
            (
                [dict(row) for row in rows],
                {str(row["id"]): raw_authors.get(str(row["id"]), []) for row in rows},
            )
            for rows in groups_by_tkey.values()
        ),
        128,
    )
    groups = []
    with process_results(compare_title_batch, tasks, num_workers) as results:
        for batch_groups in results:
            groups.extend(batch_groups)
    logger.info(
        "Author normalization and pair comparisons: {:.1f}s", time.monotonic() - started
    )

    return groups, oversized_titles


@dataclass
class DedupeApplyStats:
    groups_merged: int = 0
    groups_skipped_not_clique: int = 0
    papers_removed: int = 0
    fields_filled: int = 0
    unique_conflicts: int = 0
    citation_edges_rewritten: int = 0
    citation_self_loops_dropped: int = 0
    citation_edges_collapsed: int = 0
    authorship_rows_merged: int = 0
    pending_reference_rows_merged: int = 0
    expansion_leases_carried: int = 0

    def summary_lines(self) -> list[str]:
        return [
            f"groups    : {self.groups_merged} merged, "
            f"{self.groups_skipped_not_clique} skipped (not a clique)",
            f"papers    : {self.papers_removed} removed, "
            f"{self.fields_filled} fields backfilled onto survivors, "
            f"{self.unique_conflicts} unique-column conflicts skipped",
            f"citations : {self.citation_edges_rewritten} rewritten, "
            f"{self.citation_self_loops_dropped} dropped (would have become a "
            f"self-citation), {self.citation_edges_collapsed} collapsed into an "
            f"edge that already existed",
            f"other     : {self.authorship_rows_merged} authorship rows, "
            f"{self.pending_reference_rows_merged} pending-reference rows, "
            f"{self.expansion_leases_carried} expansion leases carried over",
        ]


def pick_group_winner(conn: sqlite3.Connection, paper_ids: tuple[str, ...]) -> str:
    """Pick which paper in a duplicate group survives the merge.

    Prefers, in order: a row carrying ``semantic_scholar_id``, then
    ``arxiv_id`` (a clique group can have at most one distinct value for
    either, since both columns are schema-unique when non-blank — see
    ``idx_papers_semantic_scholar_id_unique``/``idx_papers_arxiv_id_unique``
    — so this is unambiguous), then the row with the most recorded authors,
    then an already-``explored`` row (its existing citation expansion is kept
    as authoritative), and finally the lexicographically smallest id, for
    full determinism.
    """
    placeholders = ", ".join("?" for _ in paper_ids)
    rows_by_id = {
        str(row["id"]): row
        for row in conn.execute(
            f"SELECT id, semantic_scholar_id, arxiv_id, explored FROM papers "
            f"WHERE id IN ({placeholders})",
            paper_ids,
        ).fetchall()
    }
    author_counts = {
        paper_id: len(paper_author_names(conn, paper_id)) for paper_id in paper_ids
    }

    def sort_key(paper_id: str) -> tuple[int, int, int, int, str]:
        row = rows_by_id[paper_id]
        return (
            0 if str(row["semantic_scholar_id"] or "").strip() else 1,
            0 if str(row["arxiv_id"] or "").strip() else 1,
            -author_counts[paper_id],
            0 if row["explored"] else 1,
            paper_id,
        )

    return min(paper_ids, key=sort_key)


def rewrite_citations_for_group(
    conn: sqlite3.Connection,
    loser_ids: list[str],
    winner_id: str,
    stats: DedupeApplyStats,
) -> None:
    """Point every citation edge touching a loser at *winner_id* instead.

    Both endpoints of every affected edge are remapped in one pass (not one
    loser at a time), so a citation between two losers in the same group
    correctly becomes a self-loop and is dropped, instead of leaving a
    dangling reference to a loser that a later step in the same group would
    otherwise delete out from under it.

    An edge whose *other* endpoint belongs to a different duplicate group
    (e.g. A/A' are one group, B/B' another, and A' cites B') only gets fully
    resolved once both groups have been processed: this call remaps its own
    loser's side and leaves the other side as-is, which is correct because a
    paper belongs to at most one group, so that other side is untouched here
    only to be resolved by that other group's own call later in the same
    ``apply_duplicate_paper_groups`` loop. Since every group is processed
    exactly once against the table's current (not a stale) state, the final
    result does not depend on which group happens to run first.
    """
    placeholders = ", ".join("?" for _ in loser_ids)
    edges = conn.execute(
        f"""SELECT citing_paper_id, cited_paper_id FROM citations
            WHERE citing_paper_id IN ({placeholders})
               OR cited_paper_id IN ({placeholders})""",
        (*loser_ids, *loser_ids),
    ).fetchall()
    loser_set = set(loser_ids)
    for edge in edges:
        citing = str(edge["citing_paper_id"])
        cited = str(edge["cited_paper_id"])
        new_citing = winner_id if citing in loser_set else citing
        new_cited = winner_id if cited in loser_set else cited
        stats.citation_edges_rewritten += 1
        if new_citing == new_cited:
            # The two endpoints were duplicates of each other; that is not a
            # real self-citation, so drop the edge instead of keeping a loop.
            stats.citation_self_loops_dropped += 1
            continue
        cursor = conn.execute(
            "INSERT OR IGNORE INTO citations (citing_paper_id, cited_paper_id) VALUES (?, ?)",
            (new_citing, new_cited),
        )
        if cursor.rowcount == 0:
            stats.citation_edges_collapsed += 1
    conn.execute(
        f"""DELETE FROM citations
            WHERE citing_paper_id IN ({placeholders})
               OR cited_paper_id IN ({placeholders})""",
        (*loser_ids, *loser_ids),
    )


def rewrite_authorship_for_group(
    conn: sqlite3.Connection,
    loser_ids: list[str],
    winner_id: str,
    stats: DedupeApplyStats,
) -> None:
    placeholders = ", ".join("?" for _ in loser_ids)
    rows = conn.execute(
        f"""SELECT paper_id, author_id, organization, email, position FROM authorship
            WHERE paper_id IN ({placeholders})""",
        loser_ids,
    ).fetchall()
    for row in rows:
        conn.execute(
            """INSERT INTO authorship (paper_id, author_id, organization, email, position)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(paper_id, author_id) DO UPDATE SET
                   organization = CASE WHEN authorship.organization = ''
                                       THEN excluded.organization
                                       ELSE authorship.organization END,
                   email        = CASE WHEN authorship.email = ''
                                       THEN excluded.email
                                       ELSE authorship.email END,
                   position     = CASE WHEN authorship.position = 0
                                       THEN excluded.position
                                       ELSE authorship.position END
               WHERE (authorship.organization = '' AND excluded.organization != '')
                  OR (authorship.email = '' AND excluded.email != '')
                  OR (authorship.position = 0 AND excluded.position != 0)""",
            (winner_id, row["author_id"], row["organization"], row["email"], row["position"]),
        )
        stats.authorship_rows_merged += 1
    conn.execute(
        f"DELETE FROM authorship WHERE paper_id IN ({placeholders})",
        loser_ids,
    )


def rewrite_pending_reference_authors_for_group(
    conn: sqlite3.Connection,
    loser_ids: list[str],
    winner_id: str,
    stats: DedupeApplyStats,
) -> None:
    if not table_exists(conn, "pending_reference_authors"):
        return
    placeholders = ", ".join("?" for _ in loser_ids)
    rows = conn.execute(
        f"""SELECT paper_id, source_paper_id, position, name FROM pending_reference_authors
            WHERE paper_id IN ({placeholders}) OR source_paper_id IN ({placeholders})""",
        (*loser_ids, *loser_ids),
    ).fetchall()
    loser_set = set(loser_ids)
    for row in rows:
        paper_id = winner_id if str(row["paper_id"]) in loser_set else str(row["paper_id"])
        source_paper_id = (
            winner_id
            if str(row["source_paper_id"]) in loser_set
            else str(row["source_paper_id"])
        )
        cursor = conn.execute(
            """INSERT OR IGNORE INTO pending_reference_authors
               (paper_id, source_paper_id, position, name) VALUES (?, ?, ?, ?)""",
            (paper_id, source_paper_id, row["position"], row["name"]),
        )
        stats.pending_reference_rows_merged += cursor.rowcount if cursor.rowcount > 0 else 0
    conn.execute(
        f"""DELETE FROM pending_reference_authors
            WHERE paper_id IN ({placeholders}) OR source_paper_id IN ({placeholders})""",
        (*loser_ids, *loser_ids),
    )


def rewrite_expansion_leases_for_group(
    conn: sqlite3.Connection,
    loser_ids: list[str],
    winner_id: str,
    stats: DedupeApplyStats,
) -> None:
    if not table_exists(conn, "expansion_leases"):
        return
    placeholders = ", ".join("?" for _ in loser_ids)
    rows = conn.execute(
        f"""SELECT state FROM expansion_leases
            WHERE entity_type = 'paper' AND entity_id IN ({placeholders})""",
        loser_ids,
    ).fetchall()
    if any(str(row["state"]) == "complete" for row in rows):
        cursor = conn.execute(
            """INSERT INTO expansion_leases (entity_type, entity_id, state, owner_id, lease_until)
               VALUES ('paper', ?, 'complete', '', 0)
               ON CONFLICT(entity_type, entity_id) DO UPDATE SET
                   state = 'complete',
                   owner_id = '',
                   lease_until = 0,
                   updated_at = CURRENT_TIMESTAMP
               WHERE expansion_leases.state != 'complete'""",
            (winner_id,),
        )
        stats.expansion_leases_carried += cursor.rowcount if cursor.rowcount > 0 else 0
    conn.execute(
        f"""DELETE FROM expansion_leases
            WHERE entity_type = 'paper' AND entity_id IN ({placeholders})""",
        loser_ids,
    )


def apply_duplicate_paper_groups(
    conn: sqlite3.Connection,
    groups: list[DuplicatePaperGroup],
    *,
    dry_run: bool = True,
    commit: bool = True,
) -> DedupeApplyStats:
    """Merge each *clique* duplicate group into one surviving paper row.

    Only groups with ``is_clique = True`` are merged; ambiguous (chain)
    groups are skipped and counted in ``groups_skipped_not_clique``, never
    merged automatically — see ``find_duplicate_paper_groups`` for why a
    chain group's members are not all mutually confirmed duplicates.

    For each group, ``pick_group_winner`` chooses the surviving row; every
    other member's blank fields are backfilled onto it (reusing
    ``fill_empty_fields``, the same function the cross-database merge uses),
    then every citation, authorship, pending-reference-author, and
    expansion-lease row referencing a loser is rewritten to reference the
    winner instead — reusing the same remap-and-``INSERT OR IGNORE`` pattern
    ``merge_citations``/``merge_authorship``/etc. use when merging two
    databases, so a citation two sources independently discovered collapses
    into one edge here too. The one addition specific to merging *within* one
    database: a citation between two members of the same group would become
    a self-loop after remapping, since both endpoints are the same paper, so
    it is dropped instead of kept — the equivalent case can't arise when
    merging separate databases. Finally the loser rows are deleted; foreign
    keys are relied on (the caller must open *conn* with
    ``PRAGMA foreign_keys=ON``, e.g. via ``configure_sqlite_connection``) to
    turn a bug that misses some reference into a loud error instead of an
    orphaned row.

    Pass ``commit=False, dry_run=False`` to join a caller-owned transaction.
    Runs inside one transaction. With ``dry_run=True`` (the default), every
    change is rolled back at the end and only the returned stats reflect
    what would have happened; nothing is written to *conn*.
    """
    stats = DedupeApplyStats()
    try:
        for group in groups:
            if not group.is_clique:
                stats.groups_skipped_not_clique += 1
                continue

            winner_id = pick_group_winner(conn, group.paper_ids)
            loser_ids = [pid for pid in group.paper_ids if pid != winner_id]

            for loser_id in loser_ids:
                loser_row = conn.execute(
                    "SELECT * FROM papers WHERE id = ?", (loser_id,)
                ).fetchone()
                # Release identifiers owned by this loser before backfilling.
                # Otherwise the unique-owner check skips them and deletion
                # below permanently loses (for example) its arXiv identifier.
                conn.execute(
                    "UPDATE papers SET semantic_scholar_id = '', arxiv_id = '' "
                    "WHERE id = ?",
                    (loser_id,),
                )
                fill_empty_fields(
                    conn,
                    "papers",
                    winner_id,
                    loser_row,
                    PAPER_FILL_COLUMNS,
                    PAPER_UNIQUE_COLUMNS,
                    stats,
                )

            rewrite_citations_for_group(conn, loser_ids, winner_id, stats)
            rewrite_authorship_for_group(conn, loser_ids, winner_id, stats)
            rewrite_pending_reference_authors_for_group(conn, loser_ids, winner_id, stats)
            rewrite_expansion_leases_for_group(conn, loser_ids, winner_id, stats)

            loser_placeholders = ", ".join("?" for _ in loser_ids)
            conn.execute(
                f"DELETE FROM papers WHERE id IN ({loser_placeholders})", loser_ids
            )

            stats.groups_merged += 1
            stats.papers_removed += len(loser_ids)

        if dry_run:
            conn.rollback()
        elif commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    return stats


def merge_papers(
    canonical: sqlite3.Connection,
    source: sqlite3.Connection,
    stats: MergeStats,
) -> dict[str, str]:
    """Merge the papers table, returning only the ids that changed."""
    remap: dict[str, str] = {}
    columns = (
        "arxiv_id",
        "semantic_scholar_id",
        "title",
        "abstract",
        "published",
        "venue",
        "pdf_url",
        "explored",
        "text_path",
        "depth",
    )
    if any(
        row["name"] == "explored_after_cutoff"
        for row in source.execute("PRAGMA table_info(papers)")
    ):
        columns += ("explored_after_cutoff",)
    for row in iter_rows(source, "SELECT * FROM papers"):
        source_id = str(row["id"])
        require_uuid_source_id("papers", source_id)
        canonical_id = resolve_entity(canonical, "papers", row, PAPER_UNIQUE_COLUMNS)
        matched_by_identity = False
        if canonical_id is None:
            canonical_id = resolve_paper_by_identity(canonical, source, row)
            matched_by_identity = canonical_id is not None
        if canonical_id is None:
            canonical_id, reassigned = insert_entity(canonical, "papers", row, columns)
            stats.papers_inserted += 1
            stats.papers_uuid_reassigned += int(reassigned)
        else:
            stats.papers_matched += 1
            stats.papers_matched_by_identity += int(matched_by_identity)
            fill_empty_fields(
                canonical,
                "papers",
                canonical_id,
                row,
                PAPER_FILL_COLUMNS,
                PAPER_UNIQUE_COLUMNS,
                stats,
            )
        if canonical_id != source_id:
            remap[source_id] = canonical_id
    return remap


def merge_authors(
    canonical: sqlite3.Connection,
    source: sqlite3.Connection,
    stats: MergeStats,
) -> dict[str, str]:
    """Merge the authors table, returning only the ids that changed."""
    remap: dict[str, str] = {}
    columns = ("name", "semantic_scholar_id", "orcid")
    for row in iter_rows(source, "SELECT * FROM authors"):
        source_id = str(row["id"])
        require_uuid_source_id("authors", source_id)
        canonical_id = resolve_entity(canonical, "authors", row, AUTHOR_UNIQUE_COLUMNS)
        if canonical_id is None:
            canonical_id, reassigned = insert_entity(canonical, "authors", row, columns)
            stats.authors_inserted += 1
            stats.authors_uuid_reassigned += int(reassigned)
        else:
            stats.authors_matched += 1
            fill_empty_fields(
                canonical,
                "authors",
                canonical_id,
                row,
                AUTHOR_FILL_COLUMNS,
                AUTHOR_UNIQUE_COLUMNS,
                stats,
            )
        if canonical_id != source_id:
            remap[source_id] = canonical_id
    return remap


def merge_citations(
    canonical: sqlite3.Connection,
    source: sqlite3.Connection,
    paper_remap: dict[str, str],
    stats: MergeStats,
) -> None:
    for row in iter_rows(source, "SELECT citing_paper_id, cited_paper_id FROM citations"):
        citing = paper_remap.get(str(row["citing_paper_id"]), str(row["citing_paper_id"]))
        cited = paper_remap.get(str(row["cited_paper_id"]), str(row["cited_paper_id"]))
        cursor = canonical.execute(
            "INSERT OR IGNORE INTO citations (citing_paper_id, cited_paper_id) VALUES (?, ?)",
            (citing, cited),
        )
        stats.citations_inserted += cursor.rowcount if cursor.rowcount > 0 else 0


def merge_authorship(
    canonical: sqlite3.Connection,
    source: sqlite3.Connection,
    paper_remap: dict[str, str],
    author_remap: dict[str, str],
    stats: MergeStats,
) -> None:
    query = "SELECT paper_id, author_id, organization, email, position FROM authorship"
    for row in iter_rows(source, query):
        paper_id = paper_remap.get(str(row["paper_id"]), str(row["paper_id"]))
        author_id = author_remap.get(str(row["author_id"]), str(row["author_id"]))
        cursor = canonical.execute(
            """INSERT INTO authorship (paper_id, author_id, organization, email, position)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(paper_id, author_id) DO UPDATE SET
                   organization = CASE WHEN authorship.organization = ''
                                       THEN excluded.organization
                                       ELSE authorship.organization END,
                   email        = CASE WHEN authorship.email = ''
                                       THEN excluded.email
                                       ELSE authorship.email END,
                   position     = CASE WHEN authorship.position = 0
                                       THEN excluded.position
                                       ELSE authorship.position END
               WHERE (authorship.organization = '' AND excluded.organization != '')
                  OR (authorship.email = '' AND excluded.email != '')
                  OR (authorship.position = 0 AND excluded.position != 0)""",
            (paper_id, author_id, row["organization"], row["email"], row["position"]),
        )
        stats.authorships_written += cursor.rowcount if cursor.rowcount > 0 else 0


def merge_author_name_variants(
    canonical: sqlite3.Connection,
    source: sqlite3.Connection,
    author_remap: dict[str, str],
    stats: MergeStats,
) -> None:
    if not table_exists(source, "author_name_variants"):
        return
    query = "SELECT author_id, variant_name, normalized_name FROM author_name_variants"
    for row in iter_rows(source, query):
        author_id = author_remap.get(str(row["author_id"]), str(row["author_id"]))
        cursor = canonical.execute(
            """INSERT OR IGNORE INTO author_name_variants
               (author_id, variant_name, normalized_name) VALUES (?, ?, ?)""",
            (author_id, row["variant_name"], row["normalized_name"]),
        )
        stats.variants_inserted += cursor.rowcount if cursor.rowcount > 0 else 0


def merge_pending_reference_authors(
    canonical: sqlite3.Connection,
    source: sqlite3.Connection,
    paper_remap: dict[str, str],
    stats: MergeStats,
) -> None:
    if not table_exists(source, "pending_reference_authors"):
        return
    query = "SELECT paper_id, source_paper_id, position, name FROM pending_reference_authors"
    for row in iter_rows(source, query):
        paper_id = paper_remap.get(str(row["paper_id"]), str(row["paper_id"]))
        source_paper_id = paper_remap.get(
            str(row["source_paper_id"]),
            str(row["source_paper_id"]),
        )
        cursor = canonical.execute(
            """INSERT OR IGNORE INTO pending_reference_authors
               (paper_id, source_paper_id, position, name) VALUES (?, ?, ?, ?)""",
            (paper_id, source_paper_id, row["position"], row["name"]),
        )
        stats.pending_inserted += cursor.rowcount if cursor.rowcount > 0 else 0


def merge_expansion_leases(
    canonical: sqlite3.Connection,
    source: sqlite3.Connection,
    paper_remap: dict[str, str],
    author_remap: dict[str, str],
    stats: MergeStats,
) -> None:
    """Carry over completed expansions so the canonical DB skips refetching.

    Only ``complete`` rows transfer. ``pending`` and ``expanding`` rows are
    another machine's live coordination state: importing them would hand the
    canonical database leases owned by a worker that is no longer running.
    """
    if not table_exists(source, "expansion_leases") or not table_exists(
        canonical, "expansion_leases"
    ):
        return
    query = (
        "SELECT entity_type, entity_id FROM expansion_leases WHERE state = 'complete'"
    )
    for row in iter_rows(source, query):
        entity_type = str(row["entity_type"])
        entity_id = str(row["entity_id"])
        if entity_type == "paper":
            entity_id = paper_remap.get(entity_id, entity_id)
        elif entity_type == "author":
            entity_id = author_remap.get(entity_id, entity_id)
        else:
            continue
        cursor = canonical.execute(
            """INSERT INTO expansion_leases
                   (entity_type, entity_id, state, owner_id, lease_until)
               VALUES (?, ?, 'complete', '', 0)
               ON CONFLICT(entity_type, entity_id) DO UPDATE SET
                   state = 'complete',
                   owner_id = '',
                   lease_until = 0,
                   updated_at = CURRENT_TIMESTAMP
               WHERE expansion_leases.state != 'complete'""",
            (entity_type, entity_id),
        )
        stats.expansions_carried += cursor.rowcount if cursor.rowcount > 0 else 0


def build_node_id_updates(
    paper_remap: dict[str, str],
    author_remap: dict[str, str],
) -> dict[str, str]:
    """Turn DB-id remaps into ``paper:<uuid>`` / ``author:<uuid>`` node updates."""
    updates = {paper_nid(old): paper_nid(new) for old, new in paper_remap.items()}
    updates.update({author_nid(old): author_nid(new) for old, new in author_remap.items()})
    return updates


def build_remap_payload(
    source_updates: list[tuple[Path, dict[str, str]]],
) -> dict[str, Any]:
    """Build a remap file without conflating source-local UUID namespaces."""
    destinations: dict[str, set[str]] = {}
    for _source_db, updates in source_updates:
        for old_id, new_id in updates.items():
            destinations.setdefault(old_id, set()).add(new_id)

    ambiguous = {
        old_id: sorted(new_ids)
        for old_id, new_ids in destinations.items()
        if len(new_ids) > 1
    }
    combined = {
        old_id: next(iter(new_ids))
        for old_id, new_ids in destinations.items()
        if len(new_ids) == 1
    }
    return {
        "node_id_updates": combined,
        "sources": [
            {
                "source_db": str(source_db),
                "node_id_updates": updates,
            }
            for source_db, updates in source_updates
        ],
        "ambiguous_node_ids": ambiguous,
    }


def refresh_node_payloads(walk_data: dict[str, Any], db_path: Path) -> int:
    """Rebuild each step's node metadata from the canonical database."""
    reader = DBReader(db_path, require_uuid_ids=False)
    refreshed = 0
    try:
        for path_entry in walk_data.get("paths", []):
            for step in path_entry.get("path", []):
                node = step.get("node") or {}
                node_id = str(node.get("node_id") or "")
                if not node_id:
                    continue
                step["node"] = build_node_payload(node_id, None, reader)
                refreshed += 1
    finally:
        reader.close()
    return refreshed


def walk_node_ids(walk_data: dict[str, Any]) -> set[str]:
    """Collect every node-ID reference written by the path sampler."""
    node_ids: set[str] = set()
    for path_entry in walk_data.get("paths", []):
        start_node = str(path_entry.get("start_node") or "")
        if start_node:
            node_ids.add(start_node)
        for step in path_entry.get("path", []):
            node_id = str((step.get("node") or {}).get("node_id") or "")
            if node_id:
                node_ids.add(node_id)
            edge = step.get("edge_taken")
            if not isinstance(edge, dict):
                continue
            for endpoint in ("from", "to"):
                endpoint_id = str(edge.get(endpoint) or "")
                if endpoint_id:
                    node_ids.add(endpoint_id)
    return node_ids


def count_unresolvable_nodes(walk_data: dict[str, Any], canonical_db: Path) -> int:
    """Count node IDs that do not exist in the canonical database after remapping."""
    reader = DBReader(canonical_db, require_uuid_ids=False)
    missing = 0
    try:
        for node_id in walk_node_ids(walk_data):
            try:
                node_type = nid_type(node_id)
                db_id = nid_db_id(node_id)
            except (IndexError, ValueError):
                missing += 1
                continue
            if not is_uuid_text(db_id):
                missing += 1
                continue
            if node_type == NODE_PAPER:
                found = reader.get_paper(db_id) is not None
            elif node_type == NODE_AUTHOR:
                found = reader.get_author(db_id) is not None
            else:
                found = False
            if not found:
                missing += 1
    finally:
        reader.close()
    return missing


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        handle.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2, default=str).decode("utf-8"))
        handle.write("\n")
    temp_path.replace(path)


def read_walk_data(paths_file: Path) -> dict[str, Any]:
    """Load one sampled-paths file and reject malformed top-level data."""
    try:
        walk_data = orjson.loads(paths_file.read_text(encoding="utf-8"))
    except (OSError, orjson.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read sampled paths file {paths_file}: {exc}") from exc
    if not isinstance(walk_data, dict) or not isinstance(walk_data.get("paths"), list):
        raise RuntimeError(
            f"Sampled paths file {paths_file} must contain a JSON object with a paths list"
        )
    for path_index, path_entry in enumerate(walk_data["paths"]):
        if not isinstance(path_entry, dict) or not isinstance(path_entry.get("path"), list):
            raise RuntimeError(
                f"Sampled paths file {paths_file} has an invalid path at index "
                f"{path_index}"
            )
        for step_index, step in enumerate(path_entry["path"]):
            if not isinstance(step, dict) or not isinstance(step.get("node"), dict):
                raise RuntimeError(
                    f"Sampled paths file {paths_file} has an invalid step at "
                    f"paths[{path_index}].path[{step_index}]"
                )
            edge = step.get("edge_taken")
            if edge is not None and not isinstance(edge, dict):
                raise RuntimeError(
                    f"Sampled paths file {paths_file} has an invalid edge at "
                    f"paths[{path_index}].path[{step_index}]"
                )
    return walk_data


def rewrite_sampled_paths(
    paths_file: Path,
    node_updates: dict[str, str],
    canonical_db: Path,
    *,
    refresh_payloads: bool,
    dry_run: bool,
    validation_db: Path | None = None,
) -> None:
    walk_data = read_walk_data(paths_file)
    changed = apply_node_id_updates_to_walk_data(walk_data, node_updates)

    metadata = walk_data.get("metadata")
    if isinstance(metadata, dict):
        for key in ("graph_db_path", "db_path", "working_db_path"):
            if metadata.get(key) != str(canonical_db):
                metadata[key] = str(canonical_db)
                changed = True

    missing = count_unresolvable_nodes(
        walk_data,
        validation_db if validation_db is not None else canonical_db,
    )
    if missing:
        raise RuntimeError(
            f"{paths_file}: {missing} node IDs do not resolve in the canonical "
            "database; the sampled paths file was not rewritten"
        )

    refreshed = 0
    if refresh_payloads and not dry_run:
        refreshed = refresh_node_payloads(walk_data, canonical_db)
        changed = changed or refreshed > 0

    if dry_run:
        logger.info("[dry-run] {} | would rewrite={} ", paths_file, changed)
        return

    if not changed:
        logger.info("{} | already canonical, left untouched", paths_file)
        return

    write_json_atomic(paths_file, walk_data)
    logger.info(
        "{} | rewritten | refreshed_nodes={}",
        paths_file,
        refreshed,
    )


def collect_paths_files(entries: list[str]) -> list[Path]:
    """Expand directories into the sampled_paths.json files they contain."""
    files: list[Path] = []
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            files.extend(sorted(path.glob(SAMPLED_PATHS_GLOB)))
        elif path.is_file():
            files.append(path)
        else:
            raise SystemExit(f"Sampled paths entry not found: {path}")
    return files


def parse_sources(raw_sources: list[list[str]]) -> list[SourceBundle]:
    bundles: list[SourceBundle] = []
    path_owners: dict[Path, Path] = {}
    for entry in raw_sources:
        db_path = Path(entry[0])
        if not db_path.is_file():
            raise SystemExit(f"Source database not found: {db_path}")
        paths_files = collect_paths_files(entry[1:])
        for paths_file in paths_files:
            resolved = paths_file.resolve()
            previous_owner = path_owners.get(resolved)
            if previous_owner is not None:
                raise SystemExit(
                    f"Sampled paths file {paths_file} was listed for both "
                    f"{previous_owner} and {db_path}"
                )
            path_owners[resolved] = db_path
        bundles.append(
            SourceBundle(db_path=db_path, paths_files=paths_files)
        )
    return bundles


def validate_paths_files(bundles: list[SourceBundle]) -> None:
    """Parse every sampled-paths file before the canonical DB is modified."""
    for bundle in bundles:
        for paths_file in bundle.paths_files:
            read_walk_data(paths_file)


def merge_source(
    canonical: DatabaseManager,
    bundle: SourceBundle,
    *,
    dry_run: bool,
) -> tuple[dict[str, str], MergeStats]:
    """Merge one source database inside a single transaction."""
    stats = MergeStats()
    canonical.conn.create_function("visible_title_key", 1, normalize_visible_identity)
    source = open_source(bundle.db_path)
    # Python's sqlite3 opens a transaction implicitly on the first write and
    # holds it until commit/rollback, so the whole source merges atomically.
    try:
        paper_remap = merge_papers(canonical.conn, source, stats)
        author_remap = merge_authors(canonical.conn, source, stats)
        merge_citations(canonical.conn, source, paper_remap, stats)
        merge_authorship(canonical.conn, source, paper_remap, author_remap, stats)
        merge_author_name_variants(canonical.conn, source, author_remap, stats)
        merge_pending_reference_authors(canonical.conn, source, paper_remap, stats)
        merge_expansion_leases(canonical.conn, source, paper_remap, author_remap, stats)
        if dry_run:
            canonical.conn.rollback()
        else:
            canonical.conn.commit()
    except Exception:
        canonical.conn.rollback()
        raise
    finally:
        source.close()
    return build_node_id_updates(paper_remap, author_remap), stats


def open_merge_database(
    canonical_db: Path,
    *,
    dry_run: bool,
) -> tuple[DatabaseManager, TemporaryDirectory[str] | None]:
    """Open the real canonical DB or an isolated copy used for a dry run."""
    if not dry_run:
        return DatabaseManager(canonical_db, validate_uuid_values=False), None

    workspace = TemporaryDirectory(prefix="merge-graph-dry-run-")
    working_db = Path(workspace.name) / "canonical.db"
    if canonical_db.is_file():
        copy_sqlite_database(canonical_db, working_db)
    return DatabaseManager(working_db, validate_uuid_values=False), workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--canonical-db",
        required=True,
        metavar="PATH",
        help="Authoritative database that every source is merged into.",
    )
    parser.add_argument(
        "--source",
        required=True,
        nargs="+",
        action="append",
        metavar="DB [PATHS ...]",
        help=(
            "A source database, optionally followed by sampled_paths.json files "
            "or directories to search. Repeat once per machine."
        ),
    )
    parser.add_argument(
        "--remap-out",
        default=None,
        metavar="PATH",
        help="Write source-specific and unambiguous combined node ID remaps as JSON.",
    )
    parser.add_argument(
        "--refresh-node-payloads",
        action="store_true",
        help="Rebuild step node metadata from the canonical database after remapping.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes using an isolated DB copy without touching outputs.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bootstrap_runtime(args.log_level)

    canonical_db = Path(args.canonical_db)
    bundles = parse_sources(args.source)
    validate_paths_files(bundles)

    for bundle in bundles:
        if bundle.db_path.resolve() == canonical_db.resolve():
            raise SystemExit(
                f"Source database is the canonical database: {bundle.db_path}"
            )

    # Skip the full-table UUID audit: it scans every row of every table, which
    # costs minutes on a multi-GB canonical DB. Source IDs are checked per row
    # as they stream through instead.
    canonical, dry_run_workspace = open_merge_database(
        canonical_db,
        dry_run=args.dry_run,
    )
    source_updates: list[tuple[Path, dict[str, str]]] = []

    try:
        for bundle in bundles:
            logger.info("Merging {} -> {}", bundle.db_path, canonical_db)
            # Dry runs merge into an isolated database, so each source sees the
            # changes planned for every preceding source.
            node_updates, stats = merge_source(canonical, bundle, dry_run=False)
            for line in stats.summary_lines():
                logger.info("  {}", line)
            logger.info("  node IDs remapped: {}", len(node_updates))
            source_updates.append((bundle.db_path, node_updates))

            for paths_file in bundle.paths_files:
                rewrite_sampled_paths(
                    paths_file,
                    node_updates,
                    canonical_db,
                    refresh_payloads=args.refresh_node_payloads,
                    dry_run=args.dry_run,
                    validation_db=canonical.db_path,
                )
    finally:
        canonical.close()
        if dry_run_workspace is not None:
            dry_run_workspace.cleanup()

    if args.remap_out and not args.dry_run:
        remap_path = Path(args.remap_out)
        remap_path.parent.mkdir(parents=True, exist_ok=True)
        remap_payload = build_remap_payload(source_updates)
        write_json_atomic(remap_path, remap_payload)
        ambiguous_count = len(remap_payload["ambiguous_node_ids"])
        logger.info(
            "Wrote remap | {} | unambiguous_entries={} | ambiguous_entries={}",
            remap_path,
            len(remap_payload["node_id_updates"]),
            ambiguous_count,
        )
        if ambiguous_count:
            logger.warning(
                "Some local node IDs map differently by source; use the sources "
                "section of the remap file for those IDs"
            )

    if args.dry_run:
        logger.warning("Dry run complete; no database or sampled paths were modified.")


if __name__ == "__main__":
    main()
