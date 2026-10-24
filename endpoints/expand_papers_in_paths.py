#!/usr/bin/env python3
"""Expand unexplored papers from one or more sampled-path files using bulk data.

Papers are deduplicated across all input files before batched Spark queries.
One Spark session, a bounded metadata cache, and one SQLite writer serve the
whole invocation. Metadata chunks are sized by actual edge counts. Both
citation sides are fetched, with up to 100,000
neighbors per side. Only papers present in the input paths are expanded;
new neighbors are stored without recursively expanding them.

An expansion commits its paper metadata, neighbors, authorship, edges,
reference validation, explored flag, and completed lease atomically.
Failed papers remain unexplored and can be retried by rerunning the command.
Audit-confirmed duplicates are merged and supplied path files are remapped.

``--seed-missing-papers`` creates a missing ``--db-path`` database and inserts
absent path nodes under their saved UUIDs. The opt-in ``--drop-disproved-paths``
removes paths whose papers or edges are missing from the bulk data after
expansion (see src/graph/path_seeding.py).

Example:
    python endpoints/expand_papers_in_paths.py first.json second.json \\
        --db-path resources/semantic_scholar_graph/semantic_scholar_paths.db
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
import uuid
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

import orjson
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from endpoints.expand_unexplored_papers import (
    BULK_BATCH_API_GROUP_SIZE,
    BULK_TITLE_MIN_MATCH_SCORE,
    BULK_TITLE_SEARCH_BATCH_SIZE,
    DEFAULT_BULK_DATA_ROOT,
    DEFAULT_BULK_INDEX_ROOT,
    DEFAULT_PAPER_CACHE_BYTES,
    expand_paper_group_via_batch_api,
    open_network,
    positive_int,
    record_bulk_lookup_misses,
    title_match_score,
)
from src.clients.semantic_scholar_bulk_client import SemanticScholarBulkClient
from src.graph.db_network import SQLiteCitationNetwork
from src.graph.path_duplicate_merger import PathDuplicateMerger
from src.graph.path_expansion_levels import select_cached_bulk_misses
from src.graph.path_expansion_writer import PathExpansionWriter
from src.graph.path_seeding import (
    MISSING_IN_BULK_STATUSES,
    PathChecker,
    drop_unsupported_paths_in_files,
    seed_path_files,
)
from src.graph.sampled_paths import collect_path_papers
from src.setup import bootstrap_runtime

# Fetch graph metadata without requiring or retrieving abstract text.
PAPER_FIELDS = [
    "paperId",
    "externalIds",
    "title",
    "authors",
    "year",
    "publicationDate",
    "venue",
    "openAccessPdf",
]
SQL_ID_BATCH_SIZE = 900
REQUIRED_INDEX_TABLES = (
    "papers",
    "paper_ids",
    "citations_by_cited",
    "citations_by_citing",
    "title_index",
    "title_tokens",
)


def select_unexplored_path_papers(
    network: SQLiteCitationNetwork,
    papers: dict[str, str],
) -> tuple[list[str], int, list[str]]:
    """Check identities and expansion state with bounded bulk SQLite reads."""
    paper_ids = list(papers)
    pending = []
    already_explored = 0
    issues = []
    for start in range(0, len(paper_ids), SQL_ID_BATCH_SIZE):
        group = paper_ids[start : start + SQL_ID_BATCH_SIZE]
        placeholders = ",".join("?" for _ in group)
        rows = {
            row["id"]: row
            for row in network.db.conn.execute(
                f"SELECT id, semantic_scholar_id, explored, explored_after_cutoff "
                f"FROM papers WHERE id IN ({placeholders})",
                group,
            )
        }
        for paper_id in group:
            row = rows.get(paper_id)
            if row is None:
                issues.append(f"paper:{paper_id} is missing from the database")
                continue
            stored_id = str(row["semantic_scholar_id"] or "")
            if papers[paper_id] and papers[paper_id] != stored_id:
                issues.append(
                    f"paper:{paper_id} has conflicting path/database Semantic Scholar IDs"
                )
                continue
            if row["explored"] and not row["explored_after_cutoff"]:
                already_explored += 1
            else:
                pending.append(paper_id)
    return pending, already_explored, issues


def retry_completed_unexplored_leases(
    network: SQLiteCitationNetwork,
    paper_ids: list[str],
) -> None:
    """Retry old failed lookups that the sampler marked complete but unexplored.

    Recheck explored inside the write transaction, and leave active leases
    owned by other processes untouched.
    """
    for start in range(0, len(paper_ids), SQL_ID_BATCH_SIZE):
        group = paper_ids[start : start + SQL_ID_BATCH_SIZE]
        placeholders = ",".join("?" for _ in group)
        with network.expansion_batch():
            network.db.conn.execute(
                "UPDATE expansion_leases SET state = 'pending', owner_id = '', "
                "lease_until = 0, updated_at = CURRENT_TIMESTAMP "
                "WHERE entity_type = 'paper' AND state = 'complete' "
                "AND entity_id IN (SELECT id FROM papers WHERE explored = 0 "
                f"AND id IN ({placeholders}))",
                group,
            )


def validate_bulk_index(index_root: Path) -> None:
    """Require complete indexes so batches cannot fall back to raw JSON scans."""
    metadata_path = index_root / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"Bulk index metadata not found: {metadata_path}")
    metadata = orjson.loads(metadata_path.read_bytes())
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid bulk index metadata: {metadata_path}")  # noqa: TRY004
    tables = set(metadata.get("tables") or [])
    missing = [
        table
        for table in REQUIRED_INDEX_TABLES
        if table not in tables or not (index_root / table / "_SUCCESS").is_file()
    ]
    if missing:
        raise ValueError(
            f"Build the missing/incomplete bulk index tables before expanding: {', '.join(missing)}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths_json", type=Path, nargs="+", help="Sampled paths JSON files."
    )
    parser.add_argument(
        "--db-path", type=Path, help="Override the database recorded in the path files."
    )
    parser.add_argument("--bulk-data-root", type=Path, default=DEFAULT_BULK_DATA_ROOT)
    parser.add_argument("--bulk-index-root", type=Path, default=DEFAULT_BULK_INDEX_ROOT)
    parser.add_argument(
        "--bulk-batch-api-group-size",
        type=positive_int,
        default=BULK_BATCH_API_GROUP_SIZE,
        help="Papers per expansion batch; reduce this for papers with many neighbors.",
    )
    parser.add_argument(
        "--bulk-edge-budget",
        type=positive_int,
        default=100_000,
        help="Target edges per metadata/write chunk; a single paper is never split.",
    )
    parser.add_argument(
        "--bulk-title-search-batch-size",
        type=positive_int,
        default=BULK_TITLE_SEARCH_BATCH_SIZE,
    )
    parser.add_argument(
        "--bulk-title-min-match-score",
        type=title_match_score,
        default=BULK_TITLE_MIN_MATCH_SCORE,
    )
    parser.add_argument(
        "--seed-missing-papers",
        action="store_true",
        help=(
            "Create --db-path if it does not exist and insert path papers/authors "
            "absent from the database under their saved UUIDs. Paths are never removed."
        ),
    )
    parser.add_argument(
        "--drop-disproved-paths",
        action="store_true",
        help=(
            "After expansion, remove paths whose papers or edges are missing from the "
            "bulk data, saving them to sampled_paths.dropped.jsonl beside each file."
        ),
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    return parser


def expand_pending_path_papers(
    network: SQLiteCitationNetwork,
    client: SemanticScholarBulkClient,
    writer: PathExpansionWriter,
    merger: PathDuplicateMerger,
    pending: list[str],
    args: argparse.Namespace,
    *,
    cutoff_date: str | None = None,
    attempted: set[str] | None = None,
    bulk_index_created_at: str | None = None,
    on_results: Callable[[list[tuple[str, str]]], None] | None = None,
) -> Counter[str]:
    """Expand a deduplicated frontier, including winners of duplicate merges.

    Reopen cutoff-restricted completions: the database does not record which
    cutoff produced them, so they cannot prove completeness for this run.
    """
    counts: Counter[str] = Counter()
    queue = list(pending)
    scheduled = set(queue)
    if attempted is None:
        attempted = set()
    with tqdm(total=len(pending), desc="expand path papers", unit="paper") as progress:
        start = 0
        while start < len(queue):
            queued = queue[start : start + args.bulk_batch_api_group_size]
            start += len(queued)
            group = list(
                dict.fromkeys(
                    merger.remaps.get(paper_id, paper_id) for paper_id in queued
                )
            )
            group = [paper_id for paper_id in group if paper_id not in attempted]
            if bulk_index_created_at is not None:
                cached = select_cached_bulk_misses(
                    network.db.conn, group, bulk_index_created_at
                )
                counts["cached_not_found"] += len(cached)
                if cached and on_results is not None:
                    on_results([(paper_id, "cached_not_found") for paper_id in cached])
                attempted.update(cached)
                group = [paper_id for paper_id in group if paper_id not in cached]
            if not group:
                progress.update(len(queued))
                continue
            attempted.update(group)
            retry_completed_unexplored_leases(network, group)
            started = time.perf_counter()
            write_before = writer.elapsed_seconds
            results = expand_paper_group_via_batch_api(
                network,
                client,
                group,
                bulk_title_search_batch_size=args.bulk_title_search_batch_size,
                bulk_title_min_match_score=args.bulk_title_min_match_score,
                fields=PAPER_FIELDS,
                apply_duplicate_merge=merger.apply,
                bulk_edge_budget=args.bulk_edge_budget,
                ingest_paper=writer.ingest,
                bulk_min_neighbor_publication_date=cutoff_date,
                require_full_exploration=True,
            )
            if bulk_index_created_at is not None:
                record_bulk_lookup_misses(
                    network.db,
                    [
                        paper_id
                        for paper_id, status in results
                        if status in {"not_found", "no_semantic_scholar_id"}
                    ],
                    bulk_index_created_at,
                )
            if on_results is not None:
                on_results(results)
            write_seconds = writer.elapsed_seconds - write_before
            logger.info(
                "Batch | papers={} | fetch/coordination={:.2f}s | write={:.2f}s | neighbor_cache_hits={}",
                len(group),
                time.perf_counter() - started - write_seconds,
                write_seconds,
                writer.cache_hits,
            )
            for paper_id, status in results:
                counts[status] += 1
                if status == "merged":
                    winner = merger.remaps[paper_id]
                    if winner not in scheduled:
                        queue.append(winner)
                        scheduled.add(winner)
                        progress.total += 1
                if status not in {"expanded", "already_explored", "merged"}:
                    logger.warning("paper:{}: {}", paper_id, status)
            progress.update(len(queued))
    return counts


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bootstrap_runtime(args.log_level, setup_litellm_runtime=False)
    seed = args.seed_missing_papers
    drop = args.drop_disproved_paths
    try:
        db_path, papers, path_count = collect_path_papers(
            args.paths_json, args.db_path, allow_missing_db=seed
        )
        if seed and not db_path.exists():
            logger.info("Creating a new graph database at {}", db_path)
        with ExitStack() as resources:
            owner_id = f"expand_papers_in_paths:{os.getpid()}:{uuid.uuid4()}"
            network = open_network(db_path, owner_id)
            resources.callback(network.db.close)
            writer = PathExpansionWriter(network)
            merger = PathDuplicateMerger(writer, args.paths_json)
            merger.prepare()
            resources.callback(merger.rewrite_paths)
            if seed:
                seed_path_files(network, args.paths_json)
                merger.load_remaps()
            papers = merger.canonical_papers(papers)
            pending, already_explored, issues = select_unexplored_path_papers(
                network, papers
            )
            logger.info(
                "{} path(s) | {} unique paper(s) | {} already explored | {} to expand | db={}",
                path_count,
                len(papers),
                already_explored,
                len(pending),
                db_path,
            )
            for issue in issues:
                logger.error(issue)
            if not pending and not drop:
                return 1 if issues else 0

            counts: Counter[str] = Counter()
            missing_in_bulk: set[str] = set()

            def record_missing(results: list[tuple[str, str]]) -> None:
                missing_in_bulk.update(
                    paper_id
                    for paper_id, status in results
                    if status in MISSING_IN_BULK_STATUSES
                )

            if pending:
                validate_bulk_index(args.bulk_index_root)
                client = SemanticScholarBulkClient(
                    data_root=args.bulk_data_root,
                    index_root=args.bulk_index_root,
                    use_index=True,
                    paper_cache_max_bytes=DEFAULT_PAPER_CACHE_BYTES,
                )
                resources.callback(client.close, stop_spark=True)
                counts = expand_pending_path_papers(
                    network,
                    client,
                    writer,
                    merger,
                    pending,
                    args,
                    on_results=record_missing,
                )

            if drop:
                drop_unsupported_paths_in_files(
                    args.paths_json,
                    PathChecker(network.db.conn, merger.remaps, missing_in_bulk),
                )
                # Only papers in the kept paths still need to be explored.
                _, papers, _ = collect_path_papers(args.paths_json, db_path)
                selected = merger.canonical_papers(papers)
            else:
                selected = merger.canonical_papers(
                    {paper_id: papers[paper_id] for paper_id in pending}
                )
            remaining, _, final_issues = select_unexplored_path_papers(
                network, selected
            )
            logger.info(
                "Finished | expanded={} | already_explored={} | merged={} | not_found={} | "
                "duplicate_conflicts={} | failed={} | remaining_unexplored={}",
                counts["expanded"],
                already_explored + counts["already_explored"],
                counts["merged"],
                counts["not_found"] + counts["no_semantic_scholar_id"],
                counts["duplicate_conflict"],
                counts["failed"],
                len(remaining),
            )
            for issue in final_issues:
                logger.error(issue)
            if drop:
                # Issues for papers whose paths were dropped no longer apply.
                return 1 if final_issues or remaining else 0
            return 1 if issues or final_issues or remaining else 0
    except KeyboardInterrupt:
        logger.warning(
            "Interrupted. Rerun the same command to retry unexplored papers."
        )
        return 130
    except (ValueError, OSError, sqlite3.Error) as exc:
        logger.error("Expansion stopped: {}", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
