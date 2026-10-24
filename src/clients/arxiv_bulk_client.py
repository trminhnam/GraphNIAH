"""
arXiv bulk client — reads metadata from a local JSON snapshot.

Example commands:
```
python -m src.clients.arxiv_bulk_client
```

Instead of calling the arXiv Atom API, this client reads from the Kaggle /
arXiv OAI metadata snapshot (``arxiv-metadata-oai-snapshot.json``), a ~5 GB
JSONL file where every line is a JSON object representing one paper.

On first use the client builds a SQLite index next to the snapshot (or at a
custom path).  Subsequent instantiations reuse the cached index, making all
lookups sub-second even over millions of records.

All query results are ordered by published date (descending by default).

PDF / source downloads still fetch directly from arxiv.org (identical to
:class:`ArxivOnlineClient`).

Usage::

    client = ArxivBulkClient("resources/arxiv_metadata_dump/arxiv-metadata-oai-snapshot.json")

    # Fetch metadata for a single paper by arXiv ID
    paper = client.get_paper("2005.11401")

    # Fetch metadata for a batch of IDs
    papers = client.get_papers(["1706.03762", "2005.11401"])

    # Full-text search across title + abstract
    papers = client.search("retrieval augmented generation", max_results=200)

    # Filter by category and/or date range, optionally stream to JSONL
    papers = client.crawl(
        category="cs.CL",
        date_from="2024-01-01",
        date_to="2024-03-31",
        max_results=1000,
        output_path="data/cs_cl_2024_q1.jsonl",
    )

    # Downloads still go to arxiv.org
    client.download_pdf("2005.11401", output_dir="data/pdfs")
    client.download_source("2005.11401", output_dir="data/sources")
"""

from __future__ import annotations

import csv
import hashlib
import orjson
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import requests
from google.cloud import storage
from loguru import logger

# Re-export shared enums / data-model so callers can use either client
# interchangeably without changing imports.
from src.clients.arxiv_client import ArxivPaperMetadata, SortOrder
from src.utils.string_matching import score_string_match

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

StrPath = str | Path


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_ARXIV_BASE = "https://arxiv.org"
_HTTP_TIMEOUT_SHORT = (5, 30)
_HTTP_TIMEOUT_LONG = (5, 120)
_CHUNK_SIZE = 8_192

_VERSION_RE = re.compile(r"v\d+$")
_WHITESPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_DDL = """\
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id         TEXT PRIMARY KEY,
    title            TEXT NOT NULL DEFAULT '',
    abstract         TEXT NOT NULL DEFAULT '',
    authors_json     TEXT NOT NULL DEFAULT '[]',
    primary_category TEXT NOT NULL DEFAULT '',
    categories_json  TEXT NOT NULL DEFAULT '[]',
    categories_str   TEXT NOT NULL DEFAULT '',
    published        TEXT NOT NULL DEFAULT '',
    updated          TEXT NOT NULL DEFAULT '',
    published_date   TEXT NOT NULL DEFAULT '',
    doi              TEXT NOT NULL DEFAULT '',
    journal_ref      TEXT NOT NULL DEFAULT '',
    comment          TEXT NOT NULL DEFAULT '',
    pdf_url          TEXT NOT NULL DEFAULT '',
    entry_url        TEXT NOT NULL DEFAULT ''
);

-- Individual columns used in WHERE / ORDER BY
CREATE INDEX IF NOT EXISTS idx_primary_category ON papers(primary_category);
CREATE INDEX IF NOT EXISTS idx_published_date   ON papers(published_date);
-- Composite index for the common crawl pattern: category + date range
CREATE INDEX IF NOT EXISTS idx_cat_date ON papers(primary_category, published_date);

CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    arxiv_id UNINDEXED,
    title,
    abstract,
    content='papers',
    content_rowid='rowid'
);

-- Kept for post-index incremental inserts; dropped during bulk builds.
CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
    INSERT INTO papers_fts(rowid, arxiv_id, title, abstract)
    VALUES (new.rowid, new.arxiv_id, new.title, new.abstract);
END;

CREATE TABLE IF NOT EXISTS index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_INSERT_PAPER = """\
INSERT OR REPLACE INTO papers (
    arxiv_id, title, abstract, authors_json, primary_category,
    categories_json, categories_str, published, updated, published_date,
    doi, journal_ref, comment, pdf_url, entry_url
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


# ---------------------------------------------------------------------------
# Snapshot → ArxivPaperMetadata parsing
# ---------------------------------------------------------------------------


def _parse_authors(record: dict) -> list[str]:
    """Return author names from a snapshot record.

    Prefers the structured ``authors_parsed`` triples ``[last, first, suffix]``
    over the raw comma-separated ``authors`` string.
    """
    authors_parsed: list[list[str]] = record.get("authors_parsed") or []
    if authors_parsed:
        return [
            name
            for parts in authors_parsed
            if (name := " ".join(p for p in reversed(parts) if p).strip())
        ]
    raw: str = record.get("authors") or ""
    return [a.strip() for a in raw.split(",") if a.strip()]


def _parse_published(record: dict) -> str:
    """Return ISO-8601 published timestamp from ``versions[0].created``."""
    versions: list[dict] = record.get("versions") or []
    if not versions:
        return ""
    created_str: str = versions[0].get("created") or ""
    try:
        dt = datetime.strptime(created_str, "%a, %d %b %Y %H:%M:%S %Z")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except (ValueError, TypeError):
        return created_str  # keep raw value on parse failure


def _parse_updated(record: dict) -> str:
    """Return ISO-8601 updated timestamp from ``update_date`` (``YYYY-MM-DD``)."""
    update_date: str = record.get("update_date") or ""
    if not update_date:
        return ""
    try:
        dt = datetime.strptime(update_date, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except (ValueError, TypeError):
        return update_date


def _parse_snapshot_record(record: dict) -> ArxivPaperMetadata:
    """Convert a raw OAI snapshot JSON object to :class:`ArxivPaperMetadata`.

    Schema differences from the Atom feed:

    * ``authors`` is a plain string; ``authors_parsed`` gives structured triples.
    * ``categories`` is a space-separated string, not a list.
    * Dates come from ``versions[0].created`` and ``update_date``.
    * ``pdf_url`` is absent and is constructed from the paper ID.
    """
    arxiv_id: str = (record.get("id") or "").strip()
    categories = (record.get("categories") or "").split()
    abstract = (record.get("abstract") or "").strip()

    return ArxivPaperMetadata(
        arxiv_id=arxiv_id,
        title=_WHITESPACE_RE.sub(" ", record.get("title") or "").strip(),
        abstract=abstract,
        summary=abstract,
        authors=_parse_authors(record),
        primary_category=categories[0] if categories else "",
        categories=categories,
        published=_parse_published(record),
        updated=_parse_updated(record),
        pdf_url=f"{_ARXIV_BASE}/pdf/{arxiv_id}" if arxiv_id else "",
        doi=record.get("doi") or "",
        journal_ref=record.get("journal-ref") or "",
        comment=record.get("comments") or "",
        entry_url=f"{_ARXIV_BASE}/abs/{arxiv_id}" if arxiv_id else "",
    )


# ---------------------------------------------------------------------------
# SQLite row serialisation
# ---------------------------------------------------------------------------


def _paper_to_row(p: ArxivPaperMetadata) -> tuple:
    return (
        p.arxiv_id,
        p.title,
        p.abstract,
        orjson.dumps(p.authors).decode("utf-8"),
        p.primary_category,
        orjson.dumps(p.categories).decode("utf-8"),
        " ".join(p.categories),  # space-separated for LIKE category matching
        p.published,
        p.updated,
        p.published[:10] if p.published else "",  # YYYY-MM-DD for range queries
        p.doi,
        p.journal_ref,
        p.comment,
        p.pdf_url,
        p.entry_url,
    )


def _row_to_paper(row: sqlite3.Row) -> ArxivPaperMetadata:
    abstract = row["abstract"]
    return ArxivPaperMetadata(
        arxiv_id=row["arxiv_id"],
        title=row["title"],
        abstract=abstract,
        summary=abstract,
        authors=orjson.loads(row["authors_json"]),
        primary_category=row["primary_category"],
        categories=orjson.loads(row["categories_json"]),
        published=row["published"],
        updated=row["updated"],
        pdf_url=row["pdf_url"],
        doi=row["doi"],
        journal_ref=row["journal_ref"],
        comment=row["comment"],
        entry_url=row["entry_url"],
    )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _normalise_arxiv_id(raw: str) -> str:
    """Strip URL prefix and version suffix, returning a bare arXiv ID.

    Examples
    --------
    ``"https://arxiv.org/abs/2005.11401v3"`` → ``"2005.11401"``
    ``"2005.11401v2"`` → ``"2005.11401"``
    """
    aid = raw.split("/abs/")[-1].split("arxiv.org/")[-1]
    return _VERSION_RE.sub("", aid)


@contextmanager
def _optional_writer(output_path: Optional[StrPath]):
    """Open *output_path* for UTF-8 writing and yield the file handle.

    Yields ``None`` when *output_path* is ``None``, so callers always use the
    same ``with`` pattern regardless of whether output is requested.
    """
    if output_path is None:
        yield None
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yield fh


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class ArxivBulkClient:
    """arXiv metadata client backed by a local OAI snapshot file.

    On first instantiation the client streams the snapshot and builds a SQLite
    index (``<snapshot_path>.index.db`` by default).  This only happens once;
    subsequent instantiations reuse the cached index for sub-second lookups.

    All query results are ordered by ``published_date`` (newest first by
    default; pass ``sort_order=SortOrder.Ascending`` for oldest first).

    Parameters
    ----------
    snapshot_path:
        Path to the ``arxiv-metadata-oai-snapshot.json`` JSONL file.
    index_path:
        Optional path for the SQLite index.  Defaults to
        ``<snapshot_path>.index.db``.
    rebuild_index:
        Force a full re-index even when one already exists.
    index_batch_size:
        Records committed per SQLite transaction during indexing.  Larger
        values are faster but use more memory.
    """

    def __init__(
        self,
        snapshot_path: StrPath,
        index_path: Optional[StrPath] = None,
        *,
        rebuild_index: bool = False,
        index_batch_size: int = 10_000,
    ) -> None:
        self._snapshot_path = Path(snapshot_path)
        if not self._snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {self._snapshot_path}")

        self._index_path = (
            Path(index_path)
            if index_path is not None
            else self._snapshot_path.with_suffix(
                self._snapshot_path.suffix + ".index.db"
            )
        )
        self._hash_path = self._snapshot_path.parent / (
            self._snapshot_path.name + ".sha256"
        )
        self._index_batch_size = index_batch_size

        self._session = self._make_session()
        self._last_request_time: float = 0.0
        self._delay_seconds: float = 3.0
        self._throttle_lock = threading.Lock()

        self._db = self._open_db()
        if rebuild_index or self._snapshot_changed():
            try:
                self._build_index()
            except sqlite3.DatabaseError as exc:
                logger.warning(
                    "Index DB corrupted during build ({}); deleting and retrying.",
                    exc,
                )
                self._db.close()
                self._index_path.unlink(missing_ok=True)
                self._db = self._open_db()
                self._build_index()

    # ------------------------------------------------------------------
    # HTTP session
    # ------------------------------------------------------------------

    @staticmethod
    def _make_session() -> requests.Session:
        session = requests.Session()
        session.mount("https://", requests.adapters.HTTPAdapter(max_retries=5))
        session.headers["User-Agent"] = "Mozilla/5.0"
        return session

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _open_db(self) -> sqlite3.Connection:
        if self._index_path.exists():
            try:
                con = sqlite3.connect(str(self._index_path), check_same_thread=False)
                con.execute("SELECT 1")
                con.close()
            except sqlite3.DatabaseError as exc:
                logger.warning(
                    "Index DB at {} is malformed ({}); deleting and rebuilding.",
                    self._index_path.name,
                    exc,
                )
                self._index_path.unlink()

        try:
            con = sqlite3.connect(str(self._index_path), check_same_thread=False)
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            con.execute("PRAGMA cache_size=-64000")  # 64 MB page cache
            con.executescript(_DDL)
            con.commit()
        except sqlite3.DatabaseError as exc:
            logger.warning(
                "Index DB at {} is malformed ({}); deleting and recreating.",
                self._index_path.name,
                exc,
            )
            try:
                con.close()
            except Exception:
                pass
            self._index_path.unlink(missing_ok=True)
            con = sqlite3.connect(str(self._index_path), check_same_thread=False)
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            con.execute("PRAGMA cache_size=-64000")  # 64 MB page cache
            con.executescript(_DDL)
            con.commit()
        return con

    def _index_is_complete(self) -> bool:
        row = self._db.execute(
            "SELECT value FROM index_meta WHERE key = 'complete'"
        ).fetchone()
        return row is not None and row["value"] == "1"

    def _compute_snapshot_hash(self) -> str:
        """Return the SHA-256 hex digest of the snapshot file."""
        h = hashlib.sha256()
        with self._snapshot_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):  # 1 MB chunks
                h.update(chunk)
        return h.hexdigest()

    def _snapshot_changed(self) -> bool:
        """Return True when the snapshot hash differs from the saved hash (or is absent)."""
        if not self._hash_path.exists():
            logger.info(
                "No snapshot hash found at {} — will build index.", self._hash_path.name
            )
            return True
        saved = self._hash_path.read_text().strip()
        logger.info("Verifying snapshot hash …")
        current = self._compute_snapshot_hash()
        if current != saved:
            logger.info(
                "Snapshot hash changed ({} → {}) — rebuilding index.",
                saved[:12],
                current[:12],
            )
            return True
        if not self._index_is_complete():
            logger.info("Index incomplete — rebuilding.")
            return True
        return False

    def _save_snapshot_hash(self) -> None:
        """Write the current snapshot SHA-256 to the hash file."""
        digest = self._compute_snapshot_hash()
        self._hash_path.write_text(digest)
        logger.info("Snapshot hash saved → {}", self._hash_path.name)

    def _commit_batch(self, batch: list[tuple]) -> None:
        """Insert *batch* into ``papers`` and flush to disk."""
        self._db.executemany(_INSERT_PAPER, batch)
        self._db.commit()

    def _build_index(self) -> None:
        """Stream the snapshot, populate ``papers``, then rebuild the FTS index.

        The FTS trigger is dropped before bulk loading and recreated afterward.
        Rebuilding FTS5 in one shot via ``INSERT INTO papers_fts(papers_fts)
        VALUES('rebuild')`` is orders of magnitude faster than per-row trigger
        inserts over ~3 M records.
        """
        size_gb = self._snapshot_path.stat().st_size / 1e9
        logger.info(
            "Building index from {} ({:.1f} GB) → {}",
            self._snapshot_path.name,
            size_gb,
            self._index_path.name,
        )

        # Clear previous state
        self._db.executescript(
            """\
            DROP TRIGGER IF EXISTS papers_ai;
            DELETE FROM papers_fts;
            DELETE FROM papers;
            DELETE FROM index_meta;
        """
        )
        self._db.commit()

        # Extra PRAGMAs for bulk-write performance (reset after build)
        self._db.execute("PRAGMA temp_store=MEMORY")
        self._db.execute("PRAGMA mmap_size=268435456")  # 256 MB

        batch: list[tuple] = []
        total = 0
        errors = 0
        t0 = time.monotonic()

        with self._snapshot_path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    batch.append(
                        _paper_to_row(_parse_snapshot_record(orjson.loads(raw_line)))
                    )
                except Exception as exc:
                    errors += 1
                    logger.debug("Skipping malformed record: {}", exc)
                    continue

                if len(batch) >= self._index_batch_size:
                    self._commit_batch(batch)
                    total += len(batch)
                    batch.clear()
                    logger.info(
                        "Indexed {:,} papers  ({:.0f}/s) …",
                        total,
                        total / (time.monotonic() - t0),
                    )

        if batch:
            self._commit_batch(batch)
            total += len(batch)

        # Rebuild the FTS index in one shot — much faster than per-row triggers
        logger.info("Rebuilding FTS index for {:,} papers …", total)
        self._db.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")

        # Restore the incremental-insert trigger and persist metadata
        self._db.executescript(
            """\
            CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
                INSERT INTO papers_fts(rowid, arxiv_id, title, abstract)
                VALUES (new.rowid, new.arxiv_id, new.title, new.abstract);
            END;
        """
        )
        self._db.executemany(
            "INSERT OR REPLACE INTO index_meta(key, value) VALUES (?, ?)",
            [("complete", "1"), ("total_papers", str(total))],
        )
        self._db.commit()

        elapsed = time.monotonic() - t0
        logger.info(
            "Index complete: {:,} papers in {:.1f} s  ({} parse errors)",
            total,
            elapsed,
            errors,
        )

        self._save_snapshot_hash()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def _fetch_by_ids(self, arxiv_ids: list[str]) -> list[ArxivPaperMetadata]:
        """Return papers for *arxiv_ids*, ordered by ``published_date DESC``."""
        if not arxiv_ids:
            return []
        placeholders = ",".join("?" * len(arxiv_ids))
        rows = self._db.execute(
            f"SELECT * FROM papers WHERE arxiv_id IN ({placeholders})"
            " ORDER BY published_date DESC",
            arxiv_ids,
        ).fetchall()
        return [_row_to_paper(r) for r in rows]

    @staticmethod
    def _category_condition(category: str) -> tuple[str, list]:
        """Return ``(sql_fragment, params)`` that matches *category* in any position.

        Uses ``primary_category = ?`` as a fast index-backed path for the common
        case, falling back to a padded LIKE expression that correctly handles all
        positions within the space-separated ``categories_str`` column:

            ``" cs.CL cs.LG "  LIKE  "% cs.CL %"``  →  True
        """
        condition = "(primary_category = ?" " OR ' ' || categories_str || ' ' LIKE ?)"
        return condition, [category, f"% {category} %"]

    def _query_papers(
        self,
        *,
        query: Optional[str],
        category: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
        sort_order: SortOrder,
        max_results: int,
    ) -> list[ArxivPaperMetadata]:
        """Build and execute a filtered SQL query, always ordering by published_date."""
        direction = "DESC" if sort_order == SortOrder.Descending else "ASC"
        conditions: list[str] = []
        params: list = []

        if category:
            cond, cparams = self._category_condition(category)
            conditions.append(cond)
            params.extend(cparams)

        if date_from:
            conditions.append("published_date >= ?")
            params.append(date_from)

        if date_to:
            conditions.append("published_date <= ?")
            params.append(date_to)

        where = " AND ".join(conditions) if conditions else "1=1"
        order_limit = f"ORDER BY published_date {direction} LIMIT ?"

        if query and query.strip():
            sql = (
                "SELECT papers.* FROM papers "
                "JOIN papers_fts ON papers.rowid = papers_fts.rowid "
                f"WHERE papers_fts MATCH ? AND {where} {order_limit}"
            )
            rows = self._db.execute(
                sql, [query.strip(), *params, max_results]
            ).fetchall()
        else:
            sql = f"SELECT * FROM papers WHERE {where} {order_limit}"
            rows = self._db.execute(sql, [*params, max_results]).fetchall()

        return [_row_to_paper(r) for r in rows]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        max_results: int = 100,
        sort_order: SortOrder = SortOrder.Descending,
    ) -> list[ArxivPaperMetadata]:
        """Full-text search across paper titles and abstracts via SQLite FTS5.

        Supports the standard FTS5 query syntax (``AND``, ``OR``, ``NOT``,
        phrase matching with ``"…"``).  Results are ordered by published date.

        Parameters
        ----------
        query:
            Search terms, e.g. ``"retrieval augmented generation"`` or
            ``"transformer AND vision"``.
        max_results:
            Maximum number of results to return.
        sort_order:
            ``SortOrder.Descending`` (newest first, default) or ``Ascending``.
        """
        if not query.strip():
            raise ValueError("query must not be empty")

        # Convert "ti:" to "title:" to adapt from arxiv.org search syntax to SQLite FTS5 syntax
        exact_match_title = False
        query_phrase = None
        if query.startswith('ti:"'):
            # query = query.replace('ti:"', 'title:"')
            # query phrase is between the first and last double quotes
            query_phrase = query.split('"')[1:2][0]
            query = f'title:"{query_phrase}"'
            exact_match_title = True
        elif query.startswith('title:"'):
            query_phrase = query.split('"')[1:2][0]
            exact_match_title = True

        papers = self._query_papers(
            query=query,
            category=None,
            date_from=None,
            date_to=None,
            sort_order=sort_order,
            max_results=max_results,
        )
        logger.info("search({!r}) → {} papers", query, len(papers))

        # Iterate over the papers, sort by edit distance from the query title
        if exact_match_title:
            return sorted(
                papers,
                key=lambda x: score_string_match(query_phrase, x.title),
                reverse=True,
            )
        else:
            return papers

    def get_paper(self, arxiv_id: str) -> ArxivPaperMetadata:
        """Return metadata for a single paper by its arXiv ID.

        Parameters
        ----------
        arxiv_id:
            Bare ID (``"2005.11401"``, ``"2005.11401v4"``) or a full URL.

        Raises
        ------
        ValueError
            If the paper is not found in the snapshot.
        """
        bare_id = _normalise_arxiv_id(arxiv_id)
        row = self._db.execute(
            "SELECT * FROM papers WHERE arxiv_id = ?", (bare_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"No paper found for id={arxiv_id!r} in snapshot")
        return _row_to_paper(row)

    def get_papers(self, arxiv_ids: list[str]) -> list[ArxivPaperMetadata]:
        """Return metadata for a batch of arXiv IDs, ordered by published date.

        Missing IDs are silently skipped.
        """
        bare_ids = [_normalise_arxiv_id(raw) for raw in arxiv_ids]
        papers = self._fetch_by_ids(bare_ids)
        logger.info("get_papers() → {} / {} found", len(papers), len(arxiv_ids))
        return papers

    def crawl(
        self,
        *,
        category: Optional[str] = None,
        query: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        max_results: int = 1000,
        sort_order: SortOrder = SortOrder.Descending,
        output_path: Optional[StrPath] = None,
    ) -> list[ArxivPaperMetadata]:
        """Retrieve papers matching a category and/or date range.

        At least one filter must be provided.  Results are ordered by
        published date.

        Parameters
        ----------
        category:
            arXiv category, e.g. ``"cs.CL"``, ``"cs.LG"``, ``"stat.ML"``.
        query:
            Free-text filter applied via FTS5.
        date_from:
            Earliest submission date (``"YYYY-MM-DD"``).
        date_to:
            Latest submission date (``"YYYY-MM-DD"``).
        max_results:
            Maximum number of papers to return.
        sort_order:
            ``SortOrder.Descending`` (newest first, default) or ``Ascending``.
        output_path:
            If provided, results are also streamed to this JSONL file.
        """
        if not any([category, query, date_from, date_to]):
            raise ValueError(
                "Provide at least one of: category, query, date_from, date_to"
            )
        logger.info(
            "crawl(category={!r}, query={!r}, date_from={}, date_to={}, max_results={})",
            category,
            query,
            date_from,
            date_to,
            max_results,
        )

        papers = self._query_papers(
            query=query,
            category=category,
            date_from=date_from,
            date_to=date_to,
            sort_order=sort_order,
            max_results=max_results,
        )
        with _optional_writer(output_path) as fh:
            if fh is not None:
                for paper in papers:
                    fh.write(paper.model_dump_json() + "\n")

        logger.info("crawl() → {} papers", len(papers))
        return papers

    # ------------------------------------------------------------------
    # Index statistics
    # ------------------------------------------------------------------

    def index_stats(self) -> dict:
        """Return basic statistics about the loaded index."""
        count = self._db.execute("SELECT COUNT(*) AS n FROM papers").fetchone()["n"]
        meta = {
            r["key"]: r["value"]
            for r in self._db.execute("SELECT key, value FROM index_meta").fetchall()
        }
        return {
            "snapshot_path": str(self._snapshot_path),
            "index_path": str(self._index_path),
            "total_indexed": count,
            **meta,
        }

    # ------------------------------------------------------------------
    # Source / PDF downloads
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        """Block until ``_delay_seconds`` have elapsed since the last request.

        Thread-safe: concurrent callers share one schedule, so requests from a
        shared client instance are queued instead of sleeping independently and
        then bursting together.
        """
        with self._throttle_lock:
            now = time.monotonic()
            earliest = self._last_request_time + self._delay_seconds
            wait = earliest - now
            self._last_request_time = max(now, earliest)
        if wait > 0:
            logger.debug("Rate-limit: sleeping {:.2f} s", wait)
            time.sleep(wait)

    def _stream_to_file(self, resp: requests.Response, dest: Path) -> None:
        """Write a streaming *resp* body to *dest* in chunks."""
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                if chunk:
                    fh.write(chunk)

    def check_source_available(self, arxiv_id: str) -> bool:
        """Return ``True`` if a LaTeX source tarball is available for *arxiv_id*."""
        self._throttle()
        try:
            resp = self._session.get(
                f"{_ARXIV_BASE}/format/{arxiv_id}", timeout=_HTTP_TIMEOUT_SHORT
            )
            resp.raise_for_status()
            return "Download source" in resp.text
        except requests.exceptions.RequestException as exc:
            logger.error("Error checking source availability for {}: {}", arxiv_id, exc)
            return False

    def download_source(
        self,
        arxiv_id: str,
        output_dir: StrPath = ".",
        *,
        skip_if_exists: bool = True,
    ) -> Optional[Path]:
        """Download the LaTeX source tarball for *arxiv_id*.

        Saved as ``<output_dir>/arXiv-<arxiv_id>.tar.gz``.
        Returns ``None`` if the source is unavailable.
        """
        dest = Path(output_dir) / f"arXiv-{arxiv_id.replace('/', '_')}.tar.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)

        if skip_if_exists and dest.exists():
            logger.info("Source already exists, skipping: {}", dest)
            return dest

        self._throttle()
        try:
            resp = self._session.get(
                f"{_ARXIV_BASE}/e-print/{arxiv_id}",
                timeout=_HTTP_TIMEOUT_LONG,
                stream=True,
            )
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "application/pdf" in content_type or "text/html" in content_type:
                logger.warning(
                    "Source not available for {} (Content-Type: {}), skipping.",
                    arxiv_id,
                    content_type,
                )
                return None

            self._stream_to_file(resp, dest)
            logger.info("Saved source tarball → {}", dest)
            return dest
        except requests.exceptions.RequestException as exc:
            logger.error("Error downloading source for {}: {}", arxiv_id, exc)
            dest.unlink(missing_ok=True)
            raise

    def download_pdf(
        self,
        arxiv_id: str,
        output_dir: StrPath = ".",
        pdf_download_from: Literal["arxiv", "gcs"] = "arxiv",
    ) -> Optional[Path]:
        """Download the PDF for *arxiv_id* from arxiv.org or GCS."""
        if pdf_download_from == "arxiv":
            self._throttle()
            try:
                resp = self._session.get(
                    f"{_ARXIV_BASE}/pdf/{arxiv_id}",
                    timeout=_HTTP_TIMEOUT_LONG,
                    stream=True,
                )
                resp.raise_for_status()
            except requests.exceptions.RequestException as exc:
                logger.error("Error downloading PDF for {}: {}", arxiv_id, exc)
                return None

            dest = Path(output_dir) / f"arXiv-{arxiv_id}.pdf"
            dest.parent.mkdir(parents=True, exist_ok=True)
            self._stream_to_file(resp, dest)
            logger.info("Saved PDF → {}", dest)
            return dest

        else:
            # Extract folder prefix from arxiv_id (e.g. "2601" from "2601.03210")
            folder = arxiv_id.split(".")[0]
            gcs_prefix = f"arxiv/arxiv/pdf/{folder}/"
            bucket_name = "arxiv-dataset"

            client = storage.Client.create_anonymous_client()

            # List all blobs in the folder matching this arxiv_id
            all_blobs = client.list_blobs(bucket_name, prefix=gcs_prefix)

            # Filter blobs that match this arxiv_id (versioned or unversioned)
            # e.g. matches: 2601.03210.pdf, 2601.03210v1.pdf, 2601.03210v3.pdf
            pattern = re.compile(rf"{re.escape(arxiv_id)}(v(\d+))?\.pdf$")

            matching_blobs = []
            for blob in all_blobs:
                filename = blob.name.split("/")[-1]
                match = pattern.match(filename)
                if match:
                    version = int(match.group(2)) if match.group(2) else 0
                    matching_blobs.append((version, blob))

            if not matching_blobs:
                logger.error("No PDF found in GCS for arxiv_id: {}", arxiv_id)
                return None

            # Pick the blob with the highest version number (0 = unversioned fallback)
            _, latest_blob = max(matching_blobs, key=lambda x: x[0])

            dest = Path(output_dir) / f"arXiv-{arxiv_id}.pdf"
            dest.parent.mkdir(parents=True, exist_ok=True)

            try:
                latest_blob.download_to_filename(str(dest))
                logger.info(
                    "Saved PDF → {} (from gs://{}/{})",
                    dest,
                    bucket_name,
                    latest_blob.name,
                )
            except Exception as exc:
                logger.error("Error downloading PDF for {} from GCS: {}", arxiv_id, exc)
                return None

            return dest

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_result_schema() -> type[ArxivPaperMetadata]:
        return ArxivPaperMetadata

    @staticmethod
    def save_jsonl(papers: list[ArxivPaperMetadata], path: StrPath) -> None:
        """Write *papers* to a JSONL file (one JSON object per line)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            fh.writelines(p.model_dump_json() + "\n" for p in papers)
        logger.info("Saved {} papers → {}", len(papers), path)

    @staticmethod
    def load_jsonl(path: StrPath) -> list[ArxivPaperMetadata]:
        """Load papers previously saved with :meth:`save_jsonl`."""
        with Path(path).open(encoding="utf-8") as fh:
            return [
                ArxivPaperMetadata.model_validate(orjson.loads(line))
                for line in fh
                if line.strip()
            ]

    @staticmethod
    def save_csv(papers: list[ArxivPaperMetadata], path: StrPath) -> None:
        """Write *papers* to a CSV file (list fields serialised as JSON strings)."""
        if not papers:
            logger.warning("save_csv: no papers to write")
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [p.model_dump() for p in papers]
        for row in rows:
            row["authors"] = orjson.dumps(row["authors"]).decode("utf-8")
            row["categories"] = orjson.dumps(row["categories"]).decode("utf-8")
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved {} papers → {}", len(papers), path)

    def close(self) -> None:
        """Close the SQLite index connection. Idempotent."""
        db = getattr(self, "_db", None)
        if db is None:
            return
        try:
            db.close()
        except sqlite3.Error:
            pass
        self._db = None

    def __enter__(self) -> "ArxivBulkClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        stats = self.index_stats()
        return (
            f"ArxivBulkClient("
            f"snapshot={self._snapshot_path.name!r}, "
            f"index={self._index_path.name!r}, "
            f"total_indexed={stats['total_indexed']:,})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _snapshot = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "resources/arxiv_metadata_dump/arxiv-metadata-oai-snapshot.json"
    )

    client = ArxivBulkClient(_snapshot)
    print(client)
    print("Index stats:", client.index_stats())

    # paper = client.get_paper("2601.03210")
    # print("\n--- get_paper('1706.03762') ---")
    # print(paper)

    # title = "Language Models are Few-Shot Learners"
    title = "Attention is all you need"
    # title = "Language-Guided Structure-Aware Network for Camouflaged Object Detection"
    candidates = client.search(f'ti:"{title}"', max_results=100)
    print(f"\n--- search('{title}') → {len(candidates)} papers ---")
    # match = next(
    #     (
    #         p
    #         for p in candidates
    #         if p.title.strip().lower() == "attention is all you need"
    #     ),
    #     None,
    # )
    # print(match)
    for i, c in enumerate(candidates):
        if c.title.strip().lower() == title.lower():
            print(f"[MATCH] {i} {c.arxiv_id} {c.title}")
            break
        else:
            print(f"[NOT MATCH] {i} {c.arxiv_id} {c.title}")

# papers = client.get_papers(["1706.03762", "2005.11401"])
# print(f"\n--- get_papers() → {len(papers)} papers ---")

# results = client.search("disentanglement learning", max_results=10)
# print(f"\n--- search() → {len(results)} papers ---")
# for p in results[:10]:
#     print(f"  {p.arxiv_id}  {p.title[:80]}")

# crawled = client.crawl(
#     # category="cs.CL",
#     date_from="2026-01-01",
#     date_to="2026-01-31",
#     max_results=30_000,
#     output_path="data/test_2026_01_bulk.jsonl",
# )
# print(f"\n--- crawl(2026-Q1) → {len(crawled)} papers ---")

# if crawled:
#     client.download_pdf(crawled[0].arxiv_id, output_dir="data/pdfs")
#     client.download_source(crawled[0].arxiv_id, output_dir="data/sources")
