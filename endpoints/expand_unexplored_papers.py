#!/usr/bin/env python3
"""Expand selected citation sides for recent unexplored papers.

Round-based bulk expansion over an existing Semantic Scholar graph database
(built by ``construct_paths_from_semantic_scholar.py``): repeatedly finds every
paper that is not yet explored (``papers.explored = 0``) and published on or
after ``--cutoff-date``, then fetches its references, citations, or both from
Semantic Scholar according to ``--expand-mode``. Only ``--expand-mode both``
marks a paper explored; a one-sided expansion leaves ``papers.explored = 0``.
Each expansion fetches up to 100,000 neighbors per selected side, for both
the online and bulk backends.

By default, every citing/cited paper is inserted regardless of its date.
With the bulk backend, ``--bulk-filter-neighbors-by-cutoff true`` also filters
neighbors on both selected sides before the per-source cap. Bulk results are
written in edge-count-based chunks; a single source is never split.

Each round snapshots its candidate list once (a fresh
``papers.explored = 0 AND published >= cutoff`` query) and processes exactly
that list; any new post-cutoff stub papers discovered during the round are
picked up by the *next* round's query, never added mid-round. The snapshot is
shuffled before dispatch, so ``--max-papers`` keeps a random sample of the
eligible papers instead of always the same lowest-id prefix. Pass
``--random-seed`` to make that shuffle reproducible.

Because one-sided expansions deliberately leave papers unexplored, those
modes remember which paper IDs were already dispatched during the current
invocation. This prevents the same papers from being selected forever while
still allowing newly discovered post-cutoff papers to be processed in later
rounds. A later invocation may update those papers again.

Concurrency
-----------
The bulk backend keeps one Spark session and one SQLite connection alive
across rounds. Spark handles parallel reads internally; ``--num-workers``
only controls the online backend.

For online expansion, ``--num-workers`` splits each round's candidate list
into that many contiguous chunks, one per worker process. Each worker opens its own
``SQLiteCitationNetwork``/``SemanticScholarClient`` and writes into the same
database file. This is safe because every paper upsert
(``SQLiteCitationNetwork.upsert_paper``) runs inside a SQLite ``BEGIN
IMMEDIATE`` transaction -- only one connection can be inside that
find-or-insert sequence at a time, so two workers discovering the same
citing/cited paper can never create two rows for it -- and citation-edge
inserts use ``ON CONFLICT ... DO NOTHING``, so a duplicate edge insert is a
no-op rather than an error. The same expansion-lease table
(``expansion_leases``) that ``construct_paths_from_semantic_scholar.py`` uses
also guards each paper here, so this script is also safe to run alongside
that script against the same database.

Because a single paper's selected expansion (its selected neighbors, edges,
validation metadata, and lease state) commits as one transaction, an
interrupted run never leaves a paper half-written. In ``both`` mode it is
simply still ``explored = 0`` and gets picked up again next time.

Credentials and proxies
-----------------------
API keys are loaded from
``resources/secrets/semantic_scholar_api_keys.txt``; if that file has no
usable lines, ``SEMANTIC_SCHOLAR_API_KEY`` is used as a fallback. Blank lines
and lines whose first non-whitespace character is ``#`` are ignored. Proxies
are loaded the same way from ``resources/secrets/proxies.txt`` using
``host:port:username:password`` lines, but proxies are optional: if that file
is missing or has no usable lines, every worker connects directly instead (no
proxy), and a warning is printed at startup. Each logical worker receives one
random API-key/proxy pair (or just an API key, when no proxies are
configured) and keeps it across every round. Selection is without
replacement within each list; only the empty list is refilled for its next
selection cycle.

Rate limiting is per worker, not coordinated across workers: aggregate
request rate is roughly ``--num-workers / --semantic-scholar-delay``. Keep
that under your account's Semantic Scholar rate limit (1 req/s
unauthenticated, 10 req/s with ``SEMANTIC_SCHOLAR_API_KEY`` set).

Usage
-----
python endpoints/expand_unexplored_papers.py --cutoff-date 2025-09-01 --num-workers 8

# Update references only; papers remain unexplored
python endpoints/expand_unexplored_papers.py --cutoff-date 2025-09-01 \
    --expand-mode references

# Update citations only; papers remain unexplored
python endpoints/expand_unexplored_papers.py --cutoff-date 2025-09-01 \
    --expand-mode citations

# Bounded test run: at most 2 rounds, at most 50 candidate papers total
python endpoints/expand_unexplored_papers.py --cutoff-date 2025-09-01 \\
    --num-workers 4 --max-rounds 2 --max-papers 50

# Use the local Semantic Scholar bulk index (no API key or proxy required)
python endpoints/expand_unexplored_papers.py --cutoff-date 2025-09-01 \\
    --semantic-scholar-backend bulk --num-workers 2 --max-rounds 2 --batch-api true
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import queue
import random
import sqlite3
import sys
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote

import orjson
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from construct_paths_from_semantic_scholar import (
    BULK_TITLE_MIN_MATCH_SCORE,
    BULK_TITLE_SEARCH_BATCH_SIZE,
    BULK_TITLE_SEARCH_FIELDS,
    RATE_LIMIT_DEFER_WAIT_SECONDS,
    clean_semantic_scholar_id,
    positive_int,
    refresh_current_paper,
    title_match_score,
    upsert_semantic_paper,
)
from merge_graph_databases import (
    apply_duplicate_paper_groups,
    inspect_duplicate_paper_group,
    pick_group_winner,
)
from sample_paths import parse_cutoff_date_input, parse_published_date_value
from src.clients.semantic_scholar_bulk_client import (
    DEFAULT_DATA_ROOT as DEFAULT_BULK_DATA_ROOT,
)
from src.clients.semantic_scholar_bulk_client import (
    DEFAULT_INDEX_ROOT as DEFAULT_BULK_INDEX_ROOT,
)
from src.clients.semantic_scholar_bulk_client import (
    DEFAULT_PAPER_CACHE_BYTES,
    SemanticScholarBulkClient,
)
from src.clients.semantic_scholar_client import (
    SemanticScholarClient,
    SemanticScholarRateLimitError,
    load_dotenv_if_available,
)
from src.graph.db_network import SQLiteCitationNetwork
from src.graph.storage import DatabaseManager
from src.setup import bootstrap_runtime
from src.setup.constants import paper_nid

DEFAULT_DB_PATH = Path("resources") / "semantic_scholar_graph" / "semantic_scholar_paths.db"
DEFAULT_PROXY_PATH = Path("resources") / "secrets" / "proxies.txt"
DEFAULT_API_KEY_PATH = Path("resources") / "secrets" / "semantic_scholar_api_keys.txt"
DEFAULT_SEMANTIC_SCHOLAR_DELAY_SECONDS = 2.5
MAX_NEIGHBORS_PER_SIDE = 100_000
PROGRESS_LOG_EVERY = 100
# Papers per get_papers_with_citations_and_references_batch() call in --batch-api
# mode. Kept well under the API's 500-id ceiling: each held expansion lease is
# only good for EXPANSION_LEASE_SECONDS (db_network.py), and a large batch's
# internal retries/pagination fallbacks can take a while, so a smaller group
# bounds how many leases are at risk if a group's batch call is slow or fails.
DEFAULT_BATCH_API_GROUP_SIZE = 100
# Source groups bound Spark preparation; edge chunks bound Python collection.
# A source stays intact even when its edges exceed the chunk budget.
BULK_BATCH_API_GROUP_SIZE = 500
BULK_EDGE_BUDGET = 100_000
# BULK_TITLE_SEARCH_BATCH_SIZE (keep it smaller than the expansion group),
# BULK_TITLE_MIN_MATCH_SCORE and BULK_TITLE_SEARCH_FIELDS come from
# construct_paths_from_semantic_scholar.py, so both scripts match titles alike.
EXPAND_MODE_REFERENCES = "references"
EXPAND_MODE_CITATIONS = "citations"
EXPAND_MODE_BOTH = "both"
EXPAND_MODES = (EXPAND_MODE_REFERENCES, EXPAND_MODE_CITATIONS, EXPAND_MODE_BOTH)
SEMANTIC_SCHOLAR_BACKEND_ONLINE = "online"
SEMANTIC_SCHOLAR_BACKEND_BULK = "bulk"
SEMANTIC_SCHOLAR_BACKENDS = (
    SEMANTIC_SCHOLAR_BACKEND_ONLINE,
    SEMANTIC_SCHOLAR_BACKEND_BULK,
)

# Per-retry "Semantic Scholar returned HTTP 429 ..." warning is expected under
# this script's aggressive polling and fires on nearly every request, drowning
# out real signal. Filtered here only -- other entry points that share
# SemanticScholarClient (e.g. construct_paths_from_semantic_scholar.py) still
# see it.
_HTTP_429_RETRY_MESSAGE = "Semantic Scholar returned HTTP 429 for"


def log_filter(record: dict) -> bool:
    return _HTTP_429_RETRY_MESSAGE not in record["message"]


@dataclass(frozen=True, repr=False)
class WorkerCredentials:
    api_key: str
    proxy_url: str


@dataclass
class WorkerResult:
    worker_index: int
    expanded: int = 0
    merged: int = 0
    already_explored: int = 0
    not_found: int = 0
    duplicate_conflicts: int = 0
    failed: int = 0
    not_found_ids: list[str] = field(default_factory=list)
    duplicate_conflict_ids: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Worker credentials
# ─────────────────────────────────────────────────────────────────────────────


def load_semantic_scholar_api_keys(api_key_path: Path) -> list[str]:
    """Load API keys from non-empty, non-comment file lines."""
    api_keys: list[str] = []
    if api_key_path.is_file():
        api_keys = [
            line.strip()
            for line in api_key_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if api_keys:
        return api_keys

    environment_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if environment_api_key:
        return [environment_api_key]
    raise ValueError(
        f"No Semantic Scholar API keys found in {api_key_path} or " "SEMANTIC_SCHOLAR_API_KEY"
    )


def parse_proxy_line(raw_proxy: str, *, line_number: int) -> str:
    """Convert host:port:username:password into a requests proxy URL."""
    parts = [part.strip() for part in raw_proxy.split(":", 3)]
    if len(parts) != 4 or any(not part for part in parts):
        raise ValueError(
            f"Invalid proxy on line {line_number}: expected host:port:username:password"
        )

    host, port, username, password = parts
    try:
        port_number = int(port)
    except ValueError:
        raise ValueError(f"Invalid proxy on line {line_number}: port must be an integer") from None
    if not 1 <= port_number <= 65535:
        raise ValueError(f"Invalid proxy on line {line_number}: port must be between 1 and 65535")

    encoded_username = quote(username, safe="")
    encoded_password = quote(password, safe="")
    return f"http://{encoded_username}:{encoded_password}@{host}:{port_number}/"


def load_proxy_urls(proxy_path: Path) -> list[str]:
    """Load proxies from non-empty, non-comment file lines.

    Proxies are optional: returns an empty list (rather than raising) when
    *proxy_path* does not exist or has no usable lines. Callers should treat
    an empty result as "no proxy configured, connect directly".
    """
    if not proxy_path.is_file():
        return []

    return [
        parse_proxy_line(line.strip(), line_number=line_number)
        for line_number, line in enumerate(
            proxy_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        )
        if line.strip() and not line.lstrip().startswith("#")
    ]


def create_semantic_scholar_client(
    backend: str,
    *,
    api_key: str,
    delay_seconds: float,
    proxy_url: str,
    bulk_data_root: Path,
    bulk_index_root: Path,
) -> SemanticScholarClient | SemanticScholarBulkClient:
    """Create the selected online or local Semantic Scholar client."""
    if backend == SEMANTIC_SCHOLAR_BACKEND_BULK:
        return SemanticScholarBulkClient(
            data_root=bulk_data_root,
            index_root=bulk_index_root,
            use_index=True,
            paper_cache_max_bytes=DEFAULT_PAPER_CACHE_BYTES,
            cache_title_index=True,
        )
    return SemanticScholarClient(
        api_key=api_key,
        delay_seconds=delay_seconds,
        proxy_url=proxy_url,
    )


def close_semantic_scholar_client(
    client: SemanticScholarClient | SemanticScholarBulkClient,
    backend: str,
) -> None:
    """Close the selected client and stop its local Spark session when needed."""
    if backend == SEMANTIC_SCHOLAR_BACKEND_BULK:
        client.close(stop_spark=True)
        return
    client.close()


def take_random_item(
    full_items: list[str],
    available_items: list[str],
    rng: random.Random,
) -> str:
    """Draw without replacement, refilling only this pool when empty."""
    if not full_items:
        raise ValueError("Cannot draw from an empty item list")
    if not available_items:
        available_items.extend(full_items)
    return available_items.pop(rng.randrange(len(available_items)))


def assign_worker_credentials(
    num_workers: int,
    api_keys: list[str],
    proxy_urls: list[str],
    *,
    rng: random.Random | None = None,
) -> list[WorkerCredentials]:
    """Assign one fixed API-key/proxy pair to each logical worker.

    *proxy_urls* may be empty -- every worker then gets ``proxy_url=""``
    (direct connection, no proxy) instead of drawing from an empty pool.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be positive")

    selector = rng or random.SystemRandom()
    available_api_keys: list[str] = []
    available_proxy_urls: list[str] = []
    credentials: list[WorkerCredentials] = []
    for _ in range(num_workers):
        proxy_url = (
            take_random_item(proxy_urls, available_proxy_urls, selector) if proxy_urls else ""
        )
        credentials.append(
            WorkerCredentials(
                api_key=take_random_item(api_keys, available_api_keys, selector),
                proxy_url=proxy_url,
            )
        )
    return credentials


# ─────────────────────────────────────────────────────────────────────────────
# Candidate selection
# ─────────────────────────────────────────────────────────────────────────────


def get_candidate_papers(
    db_path: Path,
    cutoff_date: date,
    *,
    exclude_bulk_misses_for_index: str | None = None,
    include_cutoff_tainted: bool = False,
) -> list[str]:
    """Return unexplored paper ids published on/after *cutoff_date*, sorted for determinism.

    Opens a read-only connection so this never competes for the write lock
    with worker processes or any other pipeline using the same database.
    ``exclude_bulk_misses_for_index``, when given the current bulk index's
    metadata "createdAt", skips papers already known (from an earlier bulk
    run against that same index build) to have no match in it -- see
    ``bulk_lookup_misses`` in ``src/graph/storage.py``. ``include_cutoff_tainted``
    also returns explored=1 papers whose only completion was under
    --bulk-filter-neighbors-by-cutoff (explored_after_cutoff=1), so a run that
    wants a full exploration re-selects them instead of treating them as done
    -- pass this whenever the current run is itself not cutoff-restricted, to
    match acquire_expansion()'s require_full_exploration.
    """
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        explored_clause = (
            "(explored = 0 OR explored_after_cutoff = 1)"
            if include_cutoff_tainted
            else "explored = 0"
        )
        query = f"SELECT id, published FROM papers WHERE {explored_clause} AND published >= ?"
        params: tuple[str, ...] = (cutoff_date.isoformat(),)
        if exclude_bulk_misses_for_index is not None:
            query += (
                " AND id NOT IN (SELECT paper_id FROM bulk_lookup_misses "
                "WHERE index_created_at = ?)"
            )
            params += (exclude_bulk_misses_for_index,)
        rows = connection.execute(query, params).fetchall()
    finally:
        connection.close()

    candidates: list[str] = []
    for row in rows:
        published_date = parse_published_date_value(row["published"])
        if published_date is not None and published_date >= cutoff_date:
            candidates.append(str(row["id"]))
    return sorted(candidates)


def read_bulk_index_created_at(bulk_index_root: Path) -> str:
    """Return the local bulk index's build timestamp, used to scope bulk_lookup_misses.

    Empty when the index's metadata predates this field (or is a test stub):
    every bulk_lookup_misses row then shares that same "" index_created_at,
    so the cache still behaves consistently, just without stale-index scoping.
    """
    metadata = orjson.loads((bulk_index_root / "metadata.json").read_bytes())
    return str(metadata.get("createdAt") or "")


def record_bulk_lookup_misses(
    db: DatabaseManager, paper_ids: list[str], index_created_at: str
) -> None:
    """Remember papers with no match in the current bulk index so later runs skip them.

    Upserts by paper_id, so a paper re-checked against a rebuilt index simply
    overwrites its old (now stale) index_created_at with the current one.
    """
    if not paper_ids:
        return
    db.conn.executemany(
        "INSERT INTO bulk_lookup_misses (paper_id, index_created_at, checked_at) "
        "VALUES (?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(paper_id) DO UPDATE SET "
        "index_created_at = excluded.index_created_at, checked_at = excluded.checked_at",
        [(paper_id, index_created_at) for paper_id in paper_ids],
    )
    db.flush()


def split_into_chunks(items: list[str], num_chunks: int) -> list[list[str]]:
    """Split *items* into up to *num_chunks* contiguous, near-equal chunks."""
    if not items:
        return []
    num_chunks = max(1, min(num_chunks, len(items)))
    chunk_size, remainder = divmod(len(items), num_chunks)
    chunks: list[list[str]] = []
    start = 0
    for index in range(num_chunks):
        size = chunk_size + (1 if index < remainder else 0)
        chunks.append(items[start : start + size])
        start += size
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Per-paper expansion (mirrors construct_paths_from_semantic_scholar.py's
# fetch_current_paper_expansion + expand_paper_node + ingest_paper_expansion,
# without the path-sampling BuildContext they depend on)
# ─────────────────────────────────────────────────────────────────────────────


def open_network(db_path: Path, owner_id: str) -> SQLiteCitationNetwork:
    db = DatabaseManager(db_path)
    return SQLiteCitationNetwork(db, owner_id=owner_id)


def merge_resolved_duplicate_paper(
    network: SQLiteCitationNetwork,
    paper_db_id: str,
    semantic_scholar_id: str,
    *,
    allow_merge: bool = True,
    apply_merge: Callable | None = None,
) -> str | None:
    """Merge a title-resolved duplicate using the corpus-audit rules.

    Returns ``"merged"`` after a safe merge, ``"duplicate_conflict"`` when
    another row owns the ID but the pair is not an unambiguous duplicate, and
    ``None`` when no other row owns the ID.
    """
    owner = network.db.get_paper_by_semantic_scholar_id(semantic_scholar_id)
    if owner is None or str(owner["id"]) == paper_db_id:
        return None

    owner_id = str(owner["id"])
    if not allow_merge:
        logger.warning(
            "Paper {} resolves to Semantic Scholar ID {} owned by {}; "
            "keeping both UUIDs because sampled paths may reference them",
            paper_db_id,
            semantic_scholar_id,
            owner_id,
        )
        return "duplicate_conflict"
    group = inspect_duplicate_paper_group(
        network.db.conn,
        (paper_db_id, owner_id),
    )
    if group is None or not group.is_clique:
        logger.warning(
            "Paper {} resolved to Semantic Scholar ID {} already owned by {}, "
            "but the duplicate-audit title/year/author checks are inconclusive; "
            "leaving both rows unchanged",
            paper_db_id,
            semantic_scholar_id,
            owner_id,
        )
        return "duplicate_conflict"

    winner_id = pick_group_winner(network.db.conn, group.paper_ids)
    if winner_id != owner_id:
        raise RuntimeError(
            "Duplicate merge did not choose the Semantic Scholar ID owner: "
            f"expected={owner_id}, chosen={winner_id}"
        )
    if apply_merge is not None:
        if not apply_merge(group, winner_id):
            return "duplicate_conflict"
    else:
        stats = apply_duplicate_paper_groups(
            network.db.conn,
            [group],
            dry_run=False,
        )
        if stats.groups_merged != 1:
            raise RuntimeError(f"Duplicate merge was not applied for paper {paper_db_id}")
    logger.info(
        "Merged duplicate paper {} into {} for Semantic Scholar ID {}",
        paper_db_id,
        owner_id,
        semantic_scholar_id,
    )
    return "merged"


def ingest_expanded_paper(
    network: SQLiteCitationNetwork,
    node_id: str,
    paper: dict[str, Any],
    citations: list[dict[str, Any]],
    references: list[dict[str, Any]],
    *,
    expand_mode: str = EXPAND_MODE_BOTH,
    after_cutoff: bool = False,
) -> None:
    """Atomically store the selected sides of one resolved paper expansion.

    ``after_cutoff=True`` records (via mark_explored) that citations and
    references were restricted to --bulk-filter-neighbors-by-cutoff, so the
    stored edges are deliberately incomplete for anything but a
    cutoff-restricted consumer.
    """
    include_citations = expand_mode in {EXPAND_MODE_CITATIONS, EXPAND_MODE_BOTH}
    include_references = expand_mode in {EXPAND_MODE_REFERENCES, EXPAND_MODE_BOTH}
    with network.expansion_batch():
        refresh_current_paper(
            network,
            node_id,
            paper,
            references if include_references else None,
            after_cutoff=after_cutoff,
        )
        if include_citations:
            for item in citations:
                citing_paper = item.get("citingPaper") or {}
                citing_node = upsert_semantic_paper(network, citing_paper)
                if citing_node is not None:
                    network.add_citation(citing_node, node_id)
        if include_references:
            for item in references:
                cited_paper = item.get("citedPaper") or {}
                cited_node = upsert_semantic_paper(network, cited_paper)
                if cited_node is not None:
                    network.add_citation(node_id, cited_node)

        if expand_mode == EXPAND_MODE_BOTH:
            network.mark_explored(node_id, after_cutoff=after_cutoff)
            network.complete_expansion(node_id)
        else:
            network.release_expansion(node_id)


def expand_one_paper(
    network: SQLiteCitationNetwork,
    client: SemanticScholarClient | SemanticScholarBulkClient,
    paper_db_id: str,
    *,
    expand_mode: str = EXPAND_MODE_BOTH,
) -> str:
    """Fetch and store the selected citation sides for one paper.

    Returns "expanded", "already_explored", "not_found", or
    "no_semantic_scholar_id".

    Every path other than success releases the expansion lease acquired
    below (via the ``finally`` block) rather than leaving it claimed until
    it times out (``EXPANSION_LEASE_SECONDS``, 30 minutes) -- a plain
    ``return`` does not trigger an ``except`` block, so a bare
    try/except around the whole function would miss exactly these cases.
    """
    node_id = paper_nid(paper_db_id)
    # The online backend never restricts neighbors by cutoff date, so it
    # always wants a full exploration -- a paper only completed under
    # --bulk-filter-neighbors-by-cutoff should be reopened here, not skipped.
    if network.acquire_expansion(node_id, require_full_exploration=True) == "complete":
        return "already_explored"

    completed = False
    try:
        row = network.db.get_paper(paper_db_id)
        if row is None:
            return "not_found"

        semantic_scholar_id = clean_semantic_scholar_id(
            row["semantic_scholar_id"],
            node_id=node_id,
            entity_label="Paper",
        )
        if not semantic_scholar_id:
            if not row["title"]:
                return "not_found"

            matched_paper = client.search_by_title(str(row["title"]))
            if not matched_paper or not matched_paper.get("paperId"):
                return "not_found"

            semantic_scholar_id = clean_semantic_scholar_id(
                matched_paper.get("paperId"),
                node_id=node_id,
                entity_label="Resolved paper",
            )
            if not semantic_scholar_id:
                return "no_semantic_scholar_id"

        paper, citations, references = client.get_paper_with_citations_and_references(
            semantic_scholar_id,
            max_results=MAX_NEIGHBORS_PER_SIDE,
            include_citations=expand_mode in {EXPAND_MODE_CITATIONS, EXPAND_MODE_BOTH},
            include_references=expand_mode
            in {EXPAND_MODE_REFERENCES, EXPAND_MODE_BOTH},
        )
        if not paper or not paper.get("paperId"):
            return "not_found"

        resolved_semantic_scholar_id = clean_semantic_scholar_id(
            paper.get("paperId"),
            node_id=node_id,
            entity_label="Resolved paper",
        )
        if not resolved_semantic_scholar_id:
            return "no_semantic_scholar_id"

        ingest_expanded_paper(
            network,
            node_id,
            paper,
            citations,
            references,
            expand_mode=expand_mode,
        )
        completed = True
        return "expanded"
    finally:
        if not completed:
            network.release_expansion(node_id)


def expand_paper_group_via_batch_api(
    network: SQLiteCitationNetwork,
    client: SemanticScholarClient | SemanticScholarBulkClient,
    paper_db_ids: list[str],
    *,
    expand_mode: str = EXPAND_MODE_BOTH,
    bulk_title_search_batch_size: int = BULK_TITLE_SEARCH_BATCH_SIZE,
    bulk_title_min_match_score: float = BULK_TITLE_MIN_MATCH_SCORE,
    fields: list[str] | None = None,
    allow_duplicate_merges: bool = True,
    bulk_edge_budget: int | None = None,
    bulk_min_neighbor_publication_date: str | None = None,
    ingest_paper: Callable | None = None,
    apply_duplicate_merge: Callable | None = None,
    require_full_exploration: bool | None = None,
) -> list[tuple[str, str]]:
    """Batch-expand one group of papers (<= DEFAULT_BATCH_API_GROUP_SIZE) using
    get_papers_with_citations_and_references_batch() instead of one request
    per paper.

    Returns one (paper_db_id, status) pair per input ID. In addition to the
    statuses from expand_one_paper(), "merged" means a title-only duplicate
    was safely merged into the row that owns the resolved Semantic Scholar
    ID, "duplicate_conflict" means the duplicate checks were inconclusive,
    and "failed" means a request or ingest failed.

    A paper without a stored Semantic Scholar ID needs a title search first.
    The online API has no batch title-search endpoint, so there it is one
    search_by_title() call per paper; the bulk client searches titles in
    bounded search_by_titles_batch() calls. Once a title resolves an ID, the
    paper joins the same expansion batch call as everyone else.

    If a title-only row resolves to an ID already owned by another row, the
    duplicate-audit title, year, external-ID, and author rules decide whether
    the rows can be merged. Ambiguous pairs are left unchanged.
    Set ``allow_duplicate_merges=False`` to preserve UUIDs referenced by paths.
    ``fields`` selects metadata for both source and neighbor papers.
    ``bulk_edge_budget`` enables bounded subgraph chunks; ``ingest_paper``
    optionally supplies a writer that retains committed neighbor UUIDs.
    ``apply_duplicate_merge`` can persist UUID remaps with each approved merge.
    ``bulk_min_neighbor_publication_date`` (bulk backend only) drops
    citation/reference neighbors published before that date on both sides,
    before the ``MAX_NEIGHBORS_PER_SIDE`` cap is applied.
    ``require_full_exploration=True`` also reopens cutoff-restricted
    completions when their original cutoff is unknown to the caller.

    Every lease acquired here is either completed for a two-sided expansion
    or released for a one-sided expansion before this function returns --
    including on an unexpected exception or KeyboardInterrupt, via the outer
    try/finally -- so nothing is ever left claimed.
    """
    if bulk_title_search_batch_size < 1:
        raise ValueError("bulk_title_search_batch_size must be positive")
    if not 0.0 <= bulk_title_min_match_score <= 100.0:
        raise ValueError("bulk_title_min_match_score must be between 0 and 100")
    statuses: dict[str, str] = {}
    node_id_by_db_id = {paper_db_id: paper_nid(paper_db_id) for paper_db_id in paper_db_ids}
    claimed: set[str] = set()
    # No cutoff filter this call means a full exploration is wanted, so a
    # paper only completed under an earlier --bulk-filter-neighbors-by-cutoff
    # run should be reopened rather than skipped as already-done.
    if require_full_exploration is None:
        require_full_exploration = bulk_min_neighbor_publication_date is None

    def release(paper_db_id: str) -> None:
        if paper_db_id in claimed:
            network.release_expansion(node_id_by_db_id[paper_db_id])
            claimed.discard(paper_db_id)

    try:
        # Step 1: claim a lease for every paper up front; skip ones already done.
        pending: list[str] = []
        for paper_db_id in paper_db_ids:
            if (
                network.acquire_expansion(
                    node_id_by_db_id[paper_db_id],
                    require_full_exploration=require_full_exploration,
                )
                == "complete"
            ):
                statuses[paper_db_id] = "already_explored"
                continue
            claimed.add(paper_db_id)
            pending.append(paper_db_id)

        # Step 2: resolve a Semantic Scholar ID for each claimed paper -- from
        # the stored row, or via title search.
        stored_id_by_db_id: dict[str, str] = {}
        title_by_db_id: dict[str, str] = {}
        for paper_db_id in pending:
            node_id = node_id_by_db_id[paper_db_id]
            row = network.db.get_paper(paper_db_id)
            if row is None:
                statuses[paper_db_id] = "not_found"
                release(paper_db_id)
                continue

            semantic_scholar_id = clean_semantic_scholar_id(
                row["semantic_scholar_id"], node_id=node_id, entity_label="Paper"
            )
            if semantic_scholar_id:
                stored_id_by_db_id[paper_db_id] = semantic_scholar_id
            elif row["title"]:
                title_by_db_id[paper_db_id] = str(row["title"])
            else:
                statuses[paper_db_id] = "not_found"
                release(paper_db_id)

        # The bulk client searches titles in bounded groups; the online API
        # has no batch title search, so it goes one title at a time.
        matched_paper_by_db_id: dict[str, dict[str, Any] | None] = {}
        search_by_titles_batch = getattr(client, "search_by_titles_batch", None)
        if search_by_titles_batch is not None and title_by_db_id:
            title_items = list(title_by_db_id.items())
            for start in range(0, len(title_items), bulk_title_search_batch_size):
                title_group = title_items[start : start + bulk_title_search_batch_size]
                try:
                    title_matches = search_by_titles_batch(
                        [title for _paper_db_id, title in title_group],
                        fields=BULK_TITLE_SEARCH_FIELDS,
                    )
                    if len(title_matches) != len(title_group):
                        raise RuntimeError(
                            "Bulk title search returned a different number of results: "
                            f"requested={len(title_group)}, returned={len(title_matches)}"
                        )
                    for (paper_db_id, _title), matched_paper in zip(
                        title_group,
                        title_matches,
                    ):
                        if matched_paper is not None and float(
                            matched_paper.get("matchScore") or 0.0
                        ) < float(bulk_title_min_match_score):
                            logger.debug(
                                "Rejecting weak title match for {}: score={} < {}",
                                paper_db_id,
                                matched_paper.get("matchScore"),
                                bulk_title_min_match_score,
                            )
                            matched_paper = None
                        matched_paper_by_db_id[paper_db_id] = matched_paper
                except Exception as exc:
                    logger.warning(
                        "Title search failed for {} paper(s); leaving unexplored for retry: {}",
                        len(title_group),
                        exc,
                    )
                    for paper_db_id, _title in title_group:
                        statuses[paper_db_id] = "failed"
                        release(paper_db_id)
        else:
            for paper_db_id, title in title_by_db_id.items():
                try:
                    matched_paper_by_db_id[paper_db_id] = client.search_by_title(title)
                except SemanticScholarRateLimitError:
                    logger.warning(
                        "Rate limited searching title for {}; waiting {:.0f}s before continuing",
                        paper_db_id,
                        RATE_LIMIT_DEFER_WAIT_SECONDS,
                    )
                    time.sleep(RATE_LIMIT_DEFER_WAIT_SECONDS)
                    statuses[paper_db_id] = "failed"
                    release(paper_db_id)
                except Exception as exc:
                    logger.warning(
                        "Title search failed for {}; leaving unexplored for retry: {}",
                        paper_db_id,
                        exc,
                    )
                    statuses[paper_db_id] = "failed"
                    release(paper_db_id)

        # Keep the claim order: it decides which row wins when two rows
        # resolve to the same Semantic Scholar paper (see docstring).
        semantic_scholar_id_by_db_id: dict[str, str] = {}
        for paper_db_id in pending:
            if paper_db_id in stored_id_by_db_id:
                semantic_scholar_id_by_db_id[paper_db_id] = stored_id_by_db_id[paper_db_id]
                continue
            if paper_db_id not in matched_paper_by_db_id:
                continue  # already given a status above
            node_id = node_id_by_db_id[paper_db_id]
            matched_paper = matched_paper_by_db_id[paper_db_id]

            if not matched_paper or not matched_paper.get("paperId"):
                statuses[paper_db_id] = "not_found"
                release(paper_db_id)
                continue

            resolved_id = clean_semantic_scholar_id(
                matched_paper.get("paperId"),
                node_id=node_id,
                entity_label="Resolved paper",
            )
            if not resolved_id:
                statuses[paper_db_id] = "no_semantic_scholar_id"
                release(paper_db_id)
                continue

            if paper_db_id in title_by_db_id:
                duplicate_status = merge_resolved_duplicate_paper(
                    network,
                    paper_db_id,
                    resolved_id,
                    allow_merge=allow_duplicate_merges,
                    apply_merge=apply_duplicate_merge,
                )
                if duplicate_status is not None:
                    statuses[paper_db_id] = duplicate_status
                    if duplicate_status == "merged":
                        # The shared merge logic deleted the loser row and its
                        # expansion lease inside the same transaction.
                        claimed.discard(paper_db_id)
                    else:
                        release(paper_db_id)
                    continue

            semantic_scholar_id_by_db_id[paper_db_id] = resolved_id

        # Step 3: one combined-fetch batch call for everyone with a usable ID.
        if semantic_scholar_id_by_db_id:
            db_ids_by_semantic_scholar_id: dict[str, list[str]] = {}
            for paper_db_id, sid in semantic_scholar_id_by_db_id.items():
                db_ids_by_semantic_scholar_id.setdefault(sid, []).append(paper_db_id)
            ready_ids = list(db_ids_by_semantic_scholar_id.keys())

            def fetch_batches():
                if bulk_edge_budget is not None:
                    yield from client.iter_expansion_subgraph_batches(
                        ready_ids,
                        fields=fields,
                        max_results=MAX_NEIGHBORS_PER_SIDE,
                        edge_budget=bulk_edge_budget,
                        include_citations=expand_mode
                        in {EXPAND_MODE_CITATIONS, EXPAND_MODE_BOTH},
                        include_references=expand_mode
                        in {EXPAND_MODE_REFERENCES, EXPAND_MODE_BOTH},
                        min_publication_date=bulk_min_neighbor_publication_date,
                    )
                else:
                    result = client.get_papers_with_citations_and_references_batch(
                        ready_ids,
                        max_results=MAX_NEIGHBORS_PER_SIDE,
                        include_citations=expand_mode
                        in {EXPAND_MODE_CITATIONS, EXPAND_MODE_BOTH},
                        include_references=expand_mode
                        in {EXPAND_MODE_REFERENCES, EXPAND_MODE_BOTH},
                        **({"fields": fields} if fields is not None else {}),
                        **(
                            {
                                "min_publication_date": bulk_min_neighbor_publication_date
                            }
                            if bulk_min_neighbor_publication_date is not None
                            else {}
                        ),
                    )
                    yield (
                        {sid: result.get(sid, (None, [], [])) for sid in ready_ids}
                        if result is not None
                        else {}
                    )

            batches = fetch_batches()
            try:
                while True:
                    try:
                        batch_result = next(batches)
                    except StopIteration:
                        break
                    except Exception as exc:  # noqa: BLE001 -- retain completed chunks and release remaining leases
                        if isinstance(exc, SemanticScholarRateLimitError):
                            time.sleep(RATE_LIMIT_DEFER_WAIT_SECONDS)
                        logger.warning(
                            "Batch fetch failed; unfinished papers remain retryable: {}",
                            exc,
                        )
                        break
                    for semantic_scholar_id in batch_result.keys():  # noqa: SIM118 -- also accepts ExpansionSubgraphBatch
                        paper_db_ids_for_id = db_ids_by_semantic_scholar_id[
                            semantic_scholar_id
                        ]
                        paper, citations, references = batch_result.get(
                            semantic_scholar_id, (None, [], [])
                        )

                        for paper_db_id in paper_db_ids_for_id:
                            node_id = node_id_by_db_id[paper_db_id]

                            if paper is None or not paper.get("paperId"):
                                statuses[paper_db_id] = "not_found"
                                release(paper_db_id)
                                continue

                            resolved_id = clean_semantic_scholar_id(
                                paper.get("paperId"),
                                node_id=node_id,
                                entity_label="Resolved paper",
                            )
                            if not resolved_id:
                                statuses[paper_db_id] = "no_semantic_scholar_id"
                                release(paper_db_id)
                                continue

                            # Another title-only row in this same batch may have
                            # just stored the resolved ID. Re-run the conservative
                            # duplicate check before ingesting so this row does not
                            # fail forever on the unique Semantic Scholar ID index.
                            if (
                                paper_db_id in title_by_db_id
                                or not allow_duplicate_merges
                                or apply_duplicate_merge is not None
                            ):
                                duplicate_status = merge_resolved_duplicate_paper(
                                    network,
                                    paper_db_id,
                                    resolved_id,
                                    allow_merge=allow_duplicate_merges,
                                    apply_merge=apply_duplicate_merge,
                                )
                                if duplicate_status is not None:
                                    statuses[paper_db_id] = duplicate_status
                                    if duplicate_status == "merged":
                                        claimed.discard(paper_db_id)
                                    else:
                                        release(paper_db_id)
                                    continue

                            # Re-verify (and thereby renew) the lease right before
                            # writing: the batch call above may have taken long
                            # enough (retries, pagination fallbacks for a hub
                            # paper) that our Step-1 lease already expired and
                            # another process took over and finished this paper
                            # in the meantime.
                            if (
                                network.acquire_expansion(
                                    node_id,
                                    require_full_exploration=require_full_exploration,
                                )
                                == "complete"
                            ):
                                statuses[paper_db_id] = "already_explored"
                                claimed.discard(paper_db_id)
                                continue

                            try:
                                if ingest_paper is None:
                                    ingest_expanded_paper(
                                        network,
                                        node_id,
                                        paper,
                                        citations,
                                        references,
                                        expand_mode=expand_mode,
                                        after_cutoff=bool(
                                            bulk_min_neighbor_publication_date
                                        ),
                                    )
                                else:
                                    ingest_paper(node_id, paper, citations, references)
                            except Exception as exc:  # noqa: BLE001 -- one failed paper must not discard the batch
                                logger.warning(
                                    "Ingest failed for {}; leaving unexplored for retry: {}",
                                    paper_db_id,
                                    exc,
                                )
                                statuses[paper_db_id] = "failed"
                                release(paper_db_id)
                                continue

                            statuses[paper_db_id] = "expanded"
                            claimed.discard(
                                paper_db_id
                            )  # completed or already released

            finally:
                batches.close()

        return [
            (paper_db_id, statuses.get(paper_db_id, "failed"))
            for paper_db_id in paper_db_ids
        ]
    finally:
        for paper_db_id in list(claimed):
            release(paper_db_id)


# ─────────────────────────────────────────────────────────────────────────────
# Worker process entry point
# ─────────────────────────────────────────────────────────────────────────────


def batch_api_group_size_for(
    semantic_scholar_backend: str,
    bulk_group_size: int = BULK_BATCH_API_GROUP_SIZE,
) -> int:
    if semantic_scholar_backend == SEMANTIC_SCHOLAR_BACKEND_BULK:
        return bulk_group_size
    return DEFAULT_BATCH_API_GROUP_SIZE


def expand_paper_batch(
    paper_db_ids: list[str],
    *,
    db_path: Path,
    delay_seconds: float,
    api_key: str,
    proxy_url: str,
    worker_index: int,
    log_level: str,
    progress_queue: Any,
    batch_api: bool = False,
    expand_mode: str = EXPAND_MODE_BOTH,
    semantic_scholar_backend: str = SEMANTIC_SCHOLAR_BACKEND_ONLINE,
    bulk_data_root: Path = DEFAULT_BULK_DATA_ROOT,
    bulk_index_root: Path = DEFAULT_BULK_INDEX_ROOT,
    bulk_batch_api_group_size: int = BULK_BATCH_API_GROUP_SIZE,
    bulk_title_search_batch_size: int = BULK_TITLE_SEARCH_BATCH_SIZE,
    bulk_title_min_match_score: float = BULK_TITLE_MIN_MATCH_SCORE,
) -> WorkerResult:
    bootstrap_runtime(log_level, setup_litellm_runtime=False, log_filter=log_filter)
    owner_id = f"expand_unexplored_papers:{os.getpid()}:{uuid.uuid4()}"
    network = open_network(db_path, owner_id)
    client = create_semantic_scholar_client(
        semantic_scholar_backend,
        api_key=api_key,
        delay_seconds=delay_seconds,
        proxy_url=proxy_url,
        bulk_data_root=bulk_data_root,
        bulk_index_root=bulk_index_root,
    )
    try:
        return process_paper_batch(
            paper_db_ids,
            network=network,
            client=client,
            worker_index=worker_index,
            progress=progress_queue.put,
            batch_api=batch_api,
            expand_mode=expand_mode,
            semantic_scholar_backend=semantic_scholar_backend,
            bulk_batch_api_group_size=bulk_batch_api_group_size,
            bulk_title_search_batch_size=bulk_title_search_batch_size,
            bulk_title_min_match_score=bulk_title_min_match_score,
        )
    finally:
        try:
            close_semantic_scholar_client(client, semantic_scholar_backend)
        finally:
            network.db.close()


def process_paper_batch(
    paper_db_ids: list[str],
    *,
    network: SQLiteCitationNetwork,
    client: SemanticScholarClient | SemanticScholarBulkClient,
    worker_index: int,
    progress: Callable[[int], None],
    batch_api: bool,
    expand_mode: str,
    semantic_scholar_backend: str,
    bulk_batch_api_group_size: int,
    bulk_title_search_batch_size: int,
    bulk_title_min_match_score: float,
    bulk_min_neighbor_publication_date: str | None = None,
    bulk_edge_budget: int = BULK_EDGE_BUDGET,
) -> WorkerResult:
    """Expand a fixed candidate list using an already-open client and database."""
    result = WorkerResult(worker_index=worker_index)

    logger.info(
        "[worker {}] Starting | {} paper(s) assigned | batch_api={} | expand_mode={}",
        worker_index,
        len(paper_db_ids),
        batch_api,
        expand_mode,
    )

    def record(position: int, paper_db_id: str, status: str) -> None:
        if status == "expanded":
            result.expanded += 1
        elif status == "merged":
            result.merged += 1
        elif status == "already_explored":
            result.already_explored += 1
        elif status == "duplicate_conflict":
            result.duplicate_conflicts += 1
            result.duplicate_conflict_ids.append(paper_db_id)
        elif status == "failed":
            result.failed += 1
        else:
            result.not_found += 1
            result.not_found_ids.append(paper_db_id)

        progress(1)
        if position % PROGRESS_LOG_EVERY == 0 or position == len(paper_db_ids):
            logger.info(
                "[worker {}] {}/{} done | expanded={} merged={} already_explored={} "
                "not_found={} duplicate_conflicts={} failed={}",
                worker_index,
                position,
                len(paper_db_ids),
                result.expanded,
                result.merged,
                result.already_explored,
                result.not_found,
                result.duplicate_conflicts,
                result.failed,
            )

    use_bulk = semantic_scholar_backend == SEMANTIC_SCHOLAR_BACKEND_BULK
    if batch_api or use_bulk:
        position = 0
        group_size = batch_api_group_size_for(
            semantic_scholar_backend,
            bulk_batch_api_group_size,
        )
        if not batch_api:
            group_size = 1
        for start in range(0, len(paper_db_ids), group_size):
            group = paper_db_ids[start : start + group_size]
            try:
                group_results = expand_paper_group_via_batch_api(
                    network,
                    client,
                    group,
                    expand_mode=expand_mode,
                    bulk_title_search_batch_size=bulk_title_search_batch_size,
                    bulk_title_min_match_score=bulk_title_min_match_score,
                    bulk_min_neighbor_publication_date=bulk_min_neighbor_publication_date,
                    bulk_edge_budget=bulk_edge_budget if use_bulk else None,
                )
            except SemanticScholarRateLimitError:
                logger.warning(
                    "[worker {}] Rate limited on a group of {} paper(s); "
                    "waiting {:.0f}s before continuing",
                    worker_index,
                    len(group),
                    RATE_LIMIT_DEFER_WAIT_SECONDS,
                )
                time.sleep(RATE_LIMIT_DEFER_WAIT_SECONDS)
                group_results = [(paper_db_id, "failed") for paper_db_id in group]
            except Exception as exc:
                # expand_paper_group_via_batch_api() already releases every
                # lease it claimed before propagating; this is a second,
                # outer safety net so one unexpected error in a group
                # doesn't crash the whole worker and lose every other
                # group still assigned to it.
                logger.warning(
                    "[worker {}] Batch group failed unexpectedly; leaving "
                    "unexplored for retry: {}",
                    worker_index,
                    exc,
                )
                group_results = [(paper_db_id, "failed") for paper_db_id in group]

            for paper_db_id, status in group_results:
                position += 1
                record(position, paper_db_id, status)
    else:
        for position, paper_db_id in enumerate(paper_db_ids, start=1):
            try:
                status = expand_one_paper(
                    network,
                    client,
                    paper_db_id,
                    expand_mode=expand_mode,
                )
            except SemanticScholarRateLimitError:
                logger.warning(
                    "[worker {}] Rate limited on {}; waiting {:.0f}s before continuing",
                    worker_index,
                    paper_db_id,
                    RATE_LIMIT_DEFER_WAIT_SECONDS,
                )
                time.sleep(RATE_LIMIT_DEFER_WAIT_SECONDS)
                status = "failed"
            except Exception as exc:
                logger.warning(
                    "[worker {}] Expansion failed for {}; leaving unexplored for retry: {}",
                    worker_index,
                    paper_db_id,
                    exc,
                )
                status = "failed"
            record(position, paper_db_id, status)

    logger.info(
        "[worker {}] Finished | expanded={} merged={} already_explored={} not_found={} "
        "duplicate_conflicts={} failed={}",
        worker_index,
        result.expanded,
        result.merged,
        result.already_explored,
        result.not_found,
        result.duplicate_conflicts,
        result.failed,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Round orchestration
# ─────────────────────────────────────────────────────────────────────────────


def round_made_no_progress(results: list[WorkerResult]) -> bool:
    """True when a round neither expanded nor merged anything.

    get_candidate_papers() only excludes explored=1 rows, so every
    not_found/failed paper is selected again next round (except non-retryable
    papers in bulk mode -- see bulk_non_retryable in main()): if a whole round
    ends this way, retrying would repeat forever rather than eventually
    succeeding.

    already_explored does NOT count as progress here, even though it looks
    like it should. construct_paths_from_semantic_scholar.py's
    expand_paper_node() can call complete_expansion() without ever calling
    mark_explored() (when Semantic Scholar can't resolve the paper at all --
    see its "Could not resolve"/"no valid Semantic Scholar ID" branches).
    Since the two scripts share the same expansion_leases table, such a
    paper is stuck forever with lease=complete but papers.explored=0: it
    keeps getting selected as a candidate here, acquire_expansion() reports
    it "complete" every time, and it would count as already_explored every
    round without this ever actually resolving it. Only an expansion or safe
    duplicate merge proves a round genuinely finished new work.
    """
    return sum(r.expanded + r.merged for r in results) == 0


def drain_progress_queue(progress_queue: Any, bar: tqdm) -> None:
    drained = 0
    while True:
        try:
            drained += progress_queue.get_nowait()
        except queue.Empty:
            break
    if drained:
        bar.update(drained)


def run_round(
    candidates: list[str],
    *,
    db_path: Path,
    worker_credentials: list[WorkerCredentials],
    delay_seconds: float,
    round_index: int,
    log_level: str,
    batch_api: bool = False,
    expand_mode: str = EXPAND_MODE_BOTH,
    semantic_scholar_backend: str = SEMANTIC_SCHOLAR_BACKEND_ONLINE,
    bulk_data_root: Path = DEFAULT_BULK_DATA_ROOT,
    bulk_index_root: Path = DEFAULT_BULK_INDEX_ROOT,
    bulk_batch_api_group_size: int = BULK_BATCH_API_GROUP_SIZE,
    bulk_title_search_batch_size: int = BULK_TITLE_SEARCH_BATCH_SIZE,
    bulk_title_min_match_score: float = BULK_TITLE_MIN_MATCH_SCORE,
    bulk_min_neighbor_publication_date: str | None = None,
    bulk_edge_budget: int = BULK_EDGE_BUDGET,
    bulk_network: SQLiteCitationNetwork | None = None,
    bulk_client: SemanticScholarBulkClient | None = None,
) -> list[WorkerResult]:
    if semantic_scholar_backend == SEMANTIC_SCHOLAR_BACKEND_BULK:
        if bulk_network is None or bulk_client is None:
            raise ValueError("Bulk expansion requires a persistent client and database")
        with tqdm(
            total=len(candidates), desc=f"round {round_index}", unit="paper"
        ) as bar:
            return [
                process_paper_batch(
                    candidates,
                    network=bulk_network,
                    client=bulk_client,
                    worker_index=0,
                    progress=bar.update,
                    batch_api=batch_api,
                    expand_mode=expand_mode,
                    semantic_scholar_backend=semantic_scholar_backend,
                    bulk_batch_api_group_size=bulk_batch_api_group_size,
                    bulk_title_search_batch_size=bulk_title_search_batch_size,
                    bulk_title_min_match_score=bulk_title_min_match_score,
                    bulk_min_neighbor_publication_date=bulk_min_neighbor_publication_date,
                    bulk_edge_budget=bulk_edge_budget,
                )
            ]

    chunks = split_into_chunks(candidates, len(worker_credentials))
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    try:
        with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
            futures: list[Future] = [
                executor.submit(
                    expand_paper_batch,
                    chunk,
                    db_path=db_path,
                    delay_seconds=delay_seconds,
                    api_key=worker_credentials[worker_index].api_key,
                    proxy_url=worker_credentials[worker_index].proxy_url,
                    worker_index=worker_index,
                    log_level=log_level,
                    progress_queue=progress_queue,
                    batch_api=batch_api,
                    expand_mode=expand_mode,
                    semantic_scholar_backend=semantic_scholar_backend,
                    bulk_data_root=bulk_data_root,
                    bulk_index_root=bulk_index_root,
                    bulk_batch_api_group_size=bulk_batch_api_group_size,
                    bulk_title_search_batch_size=bulk_title_search_batch_size,
                    bulk_title_min_match_score=bulk_title_min_match_score,
                )
                for worker_index, chunk in enumerate(chunks)
            ]

            with tqdm(total=len(candidates), desc=f"round {round_index}", unit="paper") as bar:
                while not all(future.done() for future in futures):
                    drain_progress_queue(progress_queue, bar)
                    time.sleep(0.5)
                drain_progress_queue(progress_queue, bar)

            return [future.result() for future in futures]
    finally:
        manager.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_bool_flag(raw: str) -> bool:
    """Parse a case-insensitive 0/1/true/false/yes/no CLI value."""
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError(
        f"invalid boolean value: {raw!r} (expected one of 0, 1, true, false, yes, no)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="SQLite database to expand.",
    )
    parser.add_argument(
        "--semantic-scholar-backend",
        choices=SEMANTIC_SCHOLAR_BACKENDS,
        default=SEMANTIC_SCHOLAR_BACKEND_ONLINE,
        help="Use the online API or the local indexed bulk datasets.",
    )
    parser.add_argument(
        "--bulk-data-root",
        type=Path,
        default=DEFAULT_BULK_DATA_ROOT,
        help="Raw Semantic Scholar bulk dataset root used by the bulk backend.",
    )
    parser.add_argument(
        "--bulk-index-root",
        type=Path,
        default=DEFAULT_BULK_INDEX_ROOT,
        help="Built Semantic Scholar bulk index used by the bulk backend.",
    )
    parser.add_argument(
        "--cutoff-date",
        required=True,
        help="Only expand currently-unexplored papers published on/after this date. "
        "Accepts YYYY, YYYY-MM, or YYYY-MM-DD.",
    )
    parser.add_argument(
        "--num-workers",
        type=positive_int,
        default=1,
        help="Online worker processes per round. Bulk mode uses one persistent Spark "
        "session with internal parallelism and one database writer.",
    )
    parser.add_argument(
        "--semantic-scholar-delay",
        type=float,
        default=DEFAULT_SEMANTIC_SCHOLAR_DELAY_SECONDS,
        help="Minimum delay between Semantic Scholar requests, per worker. Aggregate request "
        "rate is roughly --num-workers / --semantic-scholar-delay; keep that under your "
        "account's rate limit (1 req/s unauthenticated, 10 req/s with SEMANTIC_SCHOLAR_API_KEY). "
        "Ignored by the bulk backend.",
    )
    parser.add_argument(
        "--max-rounds",
        type=positive_int,
        default=None,
        help="Stop after this many rounds. Default: run until no eligible work remains.",
    )
    parser.add_argument(
        "--max-papers",
        type=positive_int,
        default=None,
        help="Stop once this many candidate papers have been dispatched across all rounds "
        "combined (counts attempts, not just successful expansions). Default: unbounded.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Seed for the per-round candidate shuffle, for reproducible runs. "
        "Default: a different shuffle each run.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--batch-api",
        type=parse_bool_flag,
        default=None,
        metavar="{0,1,true,false,yes,no}",
        help="Expand papers in groups instead of one paper at a time. "
        f"Groups contain up to {DEFAULT_BATCH_API_GROUP_SIZE} papers online "
        f"({BULK_BATCH_API_GROUP_SIZE} by default with the bulk backend). "
        "Bulk results always use edge-count-based chunks and honor the cutoff filter. "
        "Default: true with the bulk backend, false online.",
    )
    parser.add_argument(
        "--bulk-batch-api-group-size",
        type=positive_int,
        default=BULK_BATCH_API_GROUP_SIZE,
        help="Papers per combined expansion call with the bulk backend. Smaller values "
        "reduce driver memory; ignored by the online backend.",
    )
    parser.add_argument(
        "--bulk-edge-budget",
        type=positive_int,
        default=BULK_EDGE_BUDGET,
        help="Target edges per bulk metadata/write chunk; a single paper is never split. "
        "Ignored by the online backend.",
    )
    parser.add_argument(
        "--bulk-title-search-batch-size",
        type=positive_int,
        default=BULK_TITLE_SEARCH_BATCH_SIZE,
        help="Unresolved titles per bulk Spark title-search query. Smaller values bound "
        "fuzzy-match intermediates; ignored by the online backend.",
    )
    parser.add_argument(
        "--bulk-title-min-match-score",
        type=title_match_score,
        default=BULK_TITLE_MIN_MATCH_SCORE,
        help="Reject bulk fuzzy title matches below this 0-100 score. Exact normalized "
        "title matches score 100. Ignored by the online backend.",
    )
    parser.add_argument(
        "--bulk-filter-neighbors-by-cutoff",
        type=parse_bool_flag,
        default=False,
        metavar="{0,1,true,false,yes,no}",
        help="Only keep citation/reference neighbors published on or after --cutoff-date "
        "when expanding with the bulk backend, instead of storing every neighbor "
        "regardless of date. Applies to both sides with the same fixed date; since "
        "references are almost always older than their source paper, this can drop "
        "most of the references side. A neighbor whose publication date can't be "
        "determined is also dropped. Default: false (keep all neighbors, current "
        "behavior). Ignored by the online backend.",
    )
    parser.add_argument(
        "--expand-mode",
        choices=EXPAND_MODES,
        default=EXPAND_MODE_BOTH,
        help="Which citation sides to fetch and store. Only 'both' marks papers explored. "
        "Default: both.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bootstrap_runtime(args.log_level, setup_litellm_runtime=False, log_filter=log_filter)

    cutoff_date = parse_cutoff_date_input(args.cutoff_date)
    db_path = Path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # Apply schema migrations before the first read-only candidate query.
    with DatabaseManager(db_path, validate_uuid_values=False):
        pass

    bulk_data_root = Path(args.bulk_data_root)
    bulk_index_root = Path(args.bulk_index_root)
    bulk_index_created_at: str | None = None
    if args.semantic_scholar_backend == SEMANTIC_SCHOLAR_BACKEND_BULK:
        metadata_path = bulk_index_root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Semantic Scholar bulk index not found: {metadata_path}"
            )
        bulk_index_created_at = read_bulk_index_created_at(bulk_index_root)
        api_keys: list[str] = []
        proxy_urls: list[str] = []
        worker_credentials = [WorkerCredentials("", "")]
        if args.num_workers > 1:
            logger.warning(
                "Bulk mode uses one persistent Spark session and database writer; "
                "--num-workers={} applies only to the online backend.",
                args.num_workers,
            )
    else:
        load_dotenv_if_available()
        api_keys = load_semantic_scholar_api_keys(DEFAULT_API_KEY_PATH)
        proxy_urls = load_proxy_urls(DEFAULT_PROXY_PATH)
        if not proxy_urls:
            logger.warning(
                "No proxies configured ({} is missing or has no usable lines); "
                "every worker will connect directly to Semantic Scholar (no proxy).",
                DEFAULT_PROXY_PATH,
            )
        worker_credentials = assign_worker_credentials(
            args.num_workers,
            api_keys,
            proxy_urls,
        )

    use_bulk = args.semantic_scholar_backend == SEMANTIC_SCHOLAR_BACKEND_BULK
    batch_api = args.batch_api if args.batch_api is not None else use_bulk
    bulk_min_neighbor_publication_date = (
        cutoff_date.isoformat()
        if use_bulk and args.bulk_filter_neighbors_by_cutoff
        else None
    )
    if use_bulk:
        source_settings = (
            f"index={bulk_index_root} | "
            f"title_batch_size={args.bulk_title_search_batch_size} | "
            f"title_min_score={args.bulk_title_min_match_score:g} | "
            f"filter_neighbors_by_cutoff={args.bulk_filter_neighbors_by_cutoff} | "
            f"edge_budget={args.bulk_edge_budget}"
        )
    else:
        source_settings = (
            f"api_keys={len(api_keys)} | proxies={len(proxy_urls)} | "
            f"delay={args.semantic_scholar_delay}s/worker "
            f"(~{args.num_workers / args.semantic_scholar_delay:.2f} req/s aggregate)"
        )
    logger.info(
        "Starting unexplored-paper expansion | backend={} | db={} | cutoff={} | "
        "workers={} | {} | max_rounds={} | max_papers={} | batch_api={} "
        "(group size {}) | expand_mode={}",
        args.semantic_scholar_backend,
        db_path,
        cutoff_date.isoformat(),
        len(worker_credentials),
        source_settings,
        args.max_rounds if args.max_rounds is not None else "unlimited",
        args.max_papers if args.max_papers is not None else "unlimited",
        batch_api,
        batch_api_group_size_for(
            args.semantic_scholar_backend,
            args.bulk_batch_api_group_size,
        ) if batch_api else 1,
        args.expand_mode,
    )

    run_started = time.time()
    candidate_rng = random.Random(args.random_seed)
    round_index = 0
    total_dispatched = 0
    one_sided_dispatched: set[str] = set()
    # The local bulk index never changes during a run, so a paper it could not
    # find stays not found: skip it in later rounds (a later invocation retries it).
    bulk_non_retryable: set[str] = set()
    bulk_network = None
    bulk_client = None
    resources = ExitStack()

    try:
        while True:
            if args.max_rounds is not None and round_index >= args.max_rounds:
                logger.info("Reached --max-rounds={}; stopping.", args.max_rounds)
                break

            candidates = get_candidate_papers(
                db_path,
                cutoff_date,
                exclude_bulk_misses_for_index=bulk_index_created_at,
                include_cutoff_tainted=bulk_min_neighbor_publication_date is None,
            )
            if args.expand_mode != EXPAND_MODE_BOTH:
                candidates = [
                    paper_db_id
                    for paper_db_id in candidates
                    if paper_db_id not in one_sided_dispatched
                ]
            if bulk_non_retryable:
                candidate_count = len(candidates)
                candidates = [
                    paper_db_id
                    for paper_db_id in candidates
                    if paper_db_id not in bulk_non_retryable
                ]
                logger.info(
                    "Skipping {} non-retryable bulk paper(s) from an earlier round",
                    candidate_count - len(candidates),
                )
            # Shuffle so --max-papers takes a random sample of the eligible
            # papers rather than always the same lowest-id prefix.
            candidate_rng.shuffle(candidates)
            if not candidates:
                if args.expand_mode == EXPAND_MODE_BOTH:
                    logger.info(
                        "No unexplored papers on/after {} remain; stopping.",
                        cutoff_date.isoformat(),
                    )
                else:
                    logger.info(
                        "No unprocessed papers on/after {} remain for this {}-only "
                        "invocation; stopping. Papers remain unexplored and may be "
                        "updated again by a later invocation.",
                        cutoff_date.isoformat(),
                        args.expand_mode,
                    )
                break

            if args.max_papers is not None:
                remaining_budget = args.max_papers - total_dispatched
                if remaining_budget <= 0:
                    logger.info("Reached --max-papers={}; stopping.", args.max_papers)
                    break
                if len(candidates) > remaining_budget:
                    candidates = candidates[:remaining_budget]

            round_index += 1
            logger.info(
                "Round {} | {} candidate paper(s) | {} worker(s)",
                round_index,
                len(candidates),
                min(len(worker_credentials), len(candidates)),
            )

            round_started = time.time()
            if use_bulk and bulk_client is None:
                owner_id = f"expand_unexplored_papers:{os.getpid()}:{uuid.uuid4()}"
                bulk_network = open_network(db_path, owner_id)
                resources.callback(bulk_network.db.close)
                bulk_client = create_semantic_scholar_client(
                    SEMANTIC_SCHOLAR_BACKEND_BULK,
                    api_key="",
                    delay_seconds=0,
                    proxy_url="",
                    bulk_data_root=bulk_data_root,
                    bulk_index_root=bulk_index_root,
                )
                resources.callback(
                    close_semantic_scholar_client,
                    bulk_client,
                    SEMANTIC_SCHOLAR_BACKEND_BULK,
                )
            results = run_round(
                candidates,
                db_path=db_path,
                worker_credentials=worker_credentials,
                delay_seconds=args.semantic_scholar_delay,
                round_index=round_index,
                log_level=args.log_level,
                batch_api=batch_api,
                expand_mode=args.expand_mode,
                semantic_scholar_backend=args.semantic_scholar_backend,
                bulk_data_root=bulk_data_root,
                bulk_index_root=bulk_index_root,
                bulk_batch_api_group_size=args.bulk_batch_api_group_size,
                bulk_title_search_batch_size=args.bulk_title_search_batch_size,
                bulk_title_min_match_score=args.bulk_title_min_match_score,
                bulk_min_neighbor_publication_date=bulk_min_neighbor_publication_date,
                bulk_edge_budget=args.bulk_edge_budget if use_bulk else BULK_EDGE_BUDGET,
                bulk_network=bulk_network,
                bulk_client=bulk_client,
            )
            total_dispatched += len(candidates)
            if args.expand_mode != EXPAND_MODE_BOTH:
                one_sided_dispatched.update(candidates)
            if use_bulk:
                for result in results:
                    bulk_non_retryable.update(result.not_found_ids)
                    bulk_non_retryable.update(result.duplicate_conflict_ids)
                    record_bulk_lookup_misses(
                        bulk_network.db, result.not_found_ids, bulk_index_created_at
                    )

            expanded = sum(result.expanded for result in results)
            merged = sum(result.merged for result in results)
            already_explored = sum(result.already_explored for result in results)
            not_found = sum(result.not_found for result in results)
            duplicate_conflicts = sum(result.duplicate_conflicts for result in results)
            failed = sum(result.failed for result in results)
            logger.info(
                "Round {} done in {:.1f}s | expanded={} merged={} already_explored={} "
                "not_found={} duplicate_conflicts={} failed={} | cumulative dispatched={}",
                round_index,
                time.time() - round_started,
                expanded,
                merged,
                already_explored,
                not_found,
                duplicate_conflicts,
                failed,
                total_dispatched,
            )

            if round_made_no_progress(results):
                logger.warning(
                    "Round {} made no progress (0 expanded, 0 merged, {} already_explored, "
                    "{} not_found, {} duplicate_conflicts, {} failed) -- stopping to "
                    "avoid retrying the same "
                    "unresolvable papers forever. Investigate the failures (see "
                    "warnings above), or fix/remove the offending rows, then rerun.",
                    round_index,
                    already_explored,
                    not_found,
                    duplicate_conflicts,
                    failed,
                )
                break
    except KeyboardInterrupt:
        logger.warning(
            "Interrupted after {} round(s), {} paper(s) dispatched. Every paper's expansion "
            "commits atomically, so it is safe to rerun this exact command to resume.",
            round_index,
            total_dispatched,
        )
        raise SystemExit(130)
    finally:
        resources.close()

    logger.info(
        "Finished after {} round(s); {} paper(s) dispatched in total | {:.1f}s elapsed",
        round_index,
        total_dispatched,
        time.time() - run_started,
    )


if __name__ == "__main__":
    main()
