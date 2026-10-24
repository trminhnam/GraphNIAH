#!/usr/bin/env python3
"""Expand sampled-path neighborhoods through a fixed number of citation levels.

Path papers are level 0. Both references and citations are children. With
--max-level 4, expand levels 0 through 3 and store level 4 without expanding
it. Neighbors older than --cutoff-date (default: 2025-09-01) or without a
known date are excluded at every level. Path-file cutoff metadata is not used.

Uses the local bulk index, one Spark session and one atomic SQLite writer.
Fully explored papers reuse stored edges. Checkpoints resume level frontiers
and committed expansions for the same settings. Use --restart to start over.
Successful filtered expansions set explored=1 and explored_after_cutoff=1.
Duplicate merges also update the supplied sampled-path files.
Unresolved papers do not block later levels: traversal uses available stored
edges. Exit status 1 reports unresolved entries after traversal finishes.
--seed-missing-papers and --drop-disproved-paths work as in
expand_papers_in_paths.py; with the latter, paths are checked and dropped
after level 0, so later levels start from the kept paths only.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import uuid
from contextlib import ExitStack
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from endpoints.expand_papers_in_paths import (
    build_parser as build_path_parser,
)
from endpoints.expand_papers_in_paths import (
    expand_pending_path_papers,
    select_unexplored_path_papers,
    validate_bulk_index,
)
from endpoints.expand_unexplored_papers import (
    DEFAULT_PAPER_CACHE_BYTES,
    open_network,
    positive_int,
    read_bulk_index_created_at,
)
from sample_paths import parse_cutoff_date_input
from src.clients.semantic_scholar_bulk_client import SemanticScholarBulkClient
from src.graph.path_duplicate_merger import PathDuplicateMerger
from src.graph.path_expansion_checkpoint import PathExpansionCheckpoint
from src.graph.path_expansion_levels import (
    collect_next_path_level,
    select_cached_bulk_misses,
)
from src.graph.path_expansion_writer import PathExpansionWriter
from src.graph.path_seeding import (
    MISSING_IN_BULK_STATUSES,
    PathChecker,
    drop_unsupported_paths_in_files,
    seed_path_files,
)
from src.graph.sampled_paths import collect_path_papers
from src.setup import bootstrap_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = build_path_parser()
    parser.description = __doc__
    parser.add_argument(
        "--cutoff-date",
        default="2025-09-01",
        help="Minimum neighbor publication date (YYYY, YYYY-MM, or YYYY-MM-DD). "
        "Default: 2025-09-01; independent of path-file metadata.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard this run's traversal checkpoint; retain graph data and lookup-miss cache.",
    )
    parser.add_argument(
        "--max-level",
        type=positive_int,
        default=4,
        help="Outer stored level; expand levels 0 through max-level minus 1 (default: 4).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bootstrap_runtime(args.log_level, setup_litellm_runtime=False)
    seed = args.seed_missing_papers
    drop = args.drop_disproved_paths
    try:
        cutoff = parse_cutoff_date_input(args.cutoff_date)
        db_path, frontier, path_count = collect_path_papers(
            args.paths_json, args.db_path, allow_missing_db=seed
        )
        if seed and not db_path.exists():
            logger.info("Creating a new graph database at {}", db_path)
        with ExitStack() as resources:
            network = open_network(
                db_path, f"expand_path_neighborhood:{os.getpid()}:{uuid.uuid4()}"
            )
            resources.callback(network.db.close)
            writer = PathExpansionWriter(network, after_cutoff=True)
            merger = PathDuplicateMerger(writer, args.paths_json)
            merger.prepare()
            resources.callback(merger.rewrite_paths)
            if seed:
                seed_path_files(network, args.paths_json)
                merger.load_remaps()
            client = None
            bulk_index_created_at = (
                read_bulk_index_created_at(args.bulk_index_root)
                if (args.bulk_index_root / "metadata.json").is_file()
                else None
            )
            frontier = merger.canonical_papers(frontier)
            logger.info(
                "{} paths | {} unique seeds | cutoff={} | outer level={} | db={}",
                path_count,
                len(frontier),
                cutoff,
                args.max_level,
                db_path,
            )
            checkpoint = PathExpansionCheckpoint(network)
            checkpoint.prepare(
                {
                    "version": 1,
                    "paths": sorted({str(path.resolve()) for path in args.paths_json}),
                    "cutoff": cutoff.isoformat(),
                    "max_level": args.max_level,
                    "bulk_index_root": str(args.bulk_index_root.resolve()),
                    "bulk_data_root": str(args.bulk_data_root.resolve()),
                    "index_created_at": bulk_index_created_at,
                    "title_min_match_score": args.bulk_title_min_match_score,
                },
                frontier,
                merger.remaps,
                args.restart,
            )
            start_level, saved_frontier, visited, attempted, incomplete, finished = (
                checkpoint.load()
            )

            def drop_disproved_paths() -> dict[str, str]:
                """Drop disproved paths once level 0 is done; return the kept seeds."""
                nonlocal incomplete
                drop_unsupported_paths_in_files(
                    args.paths_json,
                    PathChecker(
                        network.db.conn,
                        merger.remaps,
                        checkpoint.papers_with_status(MISSING_IN_BULK_STATUSES),
                        cutoff,
                    ),
                )
                _, kept, _ = collect_path_papers(args.paths_json, db_path)
                kept = merger.canonical_papers(kept)
                visited.update(checkpoint.replace_seeds(kept, merger.remaps))
                incomplete = checkpoint.has_unresolved_results()
                return kept

            # Preserve seed identity checks against the supplied path files.
            if start_level != 0:
                frontier = saved_frontier
            writer.on_success = checkpoint.record_success
            logger.info(
                "Checkpoint {} | level {} | {} papers | {} recorded attempts | finished={}",
                checkpoint.run_id[:12],
                start_level,
                len(frontier),
                len(attempted),
                finished,
            )
            # Level 0 finished in an earlier run, possibly without this flag.
            if drop and start_level > 0:
                drop_disproved_paths()
            if finished:
                logger.info(
                    "Saved traversal is finished; use --restart to rebuild the neighborhood"
                )
                return 1 if incomplete else 0
            for level in range(start_level, args.max_level):
                frontier = merger.canonical_papers(frontier)
                logger.info(
                    "Level {} | checking exploration state for {} papers",
                    level,
                    len(frontier),
                )
                pending, already_explored, issues = select_unexplored_path_papers(
                    network, frontier
                )
                for issue in issues:
                    logger.error(issue)
                # A wrong seed identity makes its entire neighborhood ambiguous.
                if issues:
                    return 1
                recorded = sum(paper_id in attempted for paper_id in pending)
                pending = [
                    paper_id for paper_id in pending if paper_id not in attempted
                ]
                cached = set()
                if pending:
                    validate_bulk_index(args.bulk_index_root)
                    if bulk_index_created_at is None:
                        bulk_index_created_at = read_bulk_index_created_at(
                            args.bulk_index_root
                        )
                    cached = select_cached_bulk_misses(
                        network.db.conn, pending, bulk_index_created_at
                    )
                    attempted.update(cached)
                    pending = [
                        paper_id for paper_id in pending if paper_id not in cached
                    ]
                    if cached:
                        incomplete = True
                        checkpoint.record_results(
                            [(paper_id, "cached_not_found") for paper_id in cached]
                        )
                logger.info(
                    "Level {} | {} papers | {} fully explored | "
                    "{} checkpoint attempts skipped | {} cached lookup misses skipped | {} to expand",
                    level,
                    len(frontier),
                    already_explored,
                    recorded,
                    len(cached),
                    len(pending),
                )
                if pending:
                    if client is None:
                        client = SemanticScholarBulkClient(
                            data_root=args.bulk_data_root,
                            index_root=args.bulk_index_root,
                            use_index=True,
                            paper_cache_max_bytes=DEFAULT_PAPER_CACHE_BYTES,
                            cache_title_index=True,
                        )
                        resources.callback(client.close, stop_spark=True)
                    counts = expand_pending_path_papers(
                        network,
                        client,
                        writer,
                        merger,
                        pending,
                        args,
                        cutoff_date=cutoff.isoformat(),
                        attempted=attempted,
                        bulk_index_created_at=bulk_index_created_at,
                        on_results=checkpoint.record_results,
                    )
                    logger.info("Level {} results: {}", level, dict(counts))
                    if any(
                        count
                        for status, count in counts.items()
                        if status not in {"expanded", "already_explored", "merged"}
                    ):
                        incomplete = True
                        logger.warning(
                            "Level {} has unresolved papers; continuing to later levels "
                            "using available stored edges",
                            level,
                        )
                if drop and level == 0:
                    frontier = drop_disproved_paths()
                frontier = merger.canonical_papers(frontier)
                visited = {merger.remaps.get(pid, pid) for pid in visited}
                visited.update(frontier)
                # Include stored edges even when no fetch was needed this level.
                logger.info("Level {} | collecting the next level", level)
                frontier = collect_next_path_level(
                    network.db.conn, list(frontier), visited, cutoff
                )
                checkpoint.save_level(
                    level + 1,
                    frontier,
                    finished=not frontier or level + 1 == args.max_level,
                )
                logger.info(
                    "Level {} | {} new papers stored | checkpoint saved",
                    level + 1,
                    len(frontier),
                )
                if not frontier:
                    logger.info(
                        "Stopping: no unvisited neighbors on/after {} at level {}",
                        cutoff,
                        level + 1,
                    )
                    break
            logger.info(
                "Finished; boundary papers retain their existing exploration state"
            )
            if incomplete:
                logger.warning(
                    "Some papers remain unresolved; their missing edges may limit "
                    "the neighborhood. Use --restart to retry temporary failures; cached lookup "
                    "misses retry when the bulk index build timestamp changes."
                )
            return 1 if incomplete else 0
    except KeyboardInterrupt:
        logger.warning(
            "Interrupted; committed expansions are retained. Rerun to continue."
        )
        return 130
    except (ValueError, OSError, sqlite3.Error) as exc:
        logger.error("Expansion stopped: {}", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
