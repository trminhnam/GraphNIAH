"""Precompute usable input paths and their disjoint groups for create_dataset.v2.

Each --paths file holds papers-only paths of exactly 3k - 2 papers (one file per
question depth k). For each file: check every path once (layout, explored and
cutoff checks, starting trees, citation edges), find the conflicting pairs
(shared paper or title, or a citation edge between them), and store every
disjoint group of 2, 4, 8, and 16 paths when there are at most --group-limit of
them; above the limit only the conflicting pairs are kept and groups are drawn
and redrawn on conflict. Both ways give a uniform draw over all disjoint
groups. The cache goes next to the paths file as disjoint_path_groups.json.
Path checks and group enumeration run in --workers processes.

Run after scripts/path_sampling.new/2026-04-01/expand_path_neighborhood.sh
(activate the niah environment first)::

    python endpoints/precompute_disjoint_path_groups.py \
        --paths output/data.2026-04-01/hops={10,22,46,94}/papers-only/sampled_paths.json \
        --graph-db resources/semantic_scholar_graph.2026-04-01/semantic_scholar_paths.db \
        --cutoff-date 2026-04-01 --workers 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.logging.console import log_line, progress_bar
from src.questions import disjoint_path_groups as groups
from src.questions.dataset_pool import portable_path

# One per worker process (and in this process when there is one worker).
WORKER_CHECKER: dict[str, Any] = {}


def start_path_worker(graph_db: Path, cutoff_date: str, only_explored: int) -> None:
    """Open this worker's own read-only database connection."""
    WORKER_CHECKER["current"] = groups.open_graph_checker(
        graph_db, cutoff_date, only_explored
    )


def check_chunk(task: tuple[int, list[dict[str, Any]]]) -> list[tuple[str, Any]]:
    depth, entries = task
    checker = WORKER_CHECKER["current"]
    return groups.check_paths(
        entries, depth, context=checker.context, facts=checker.facts, cutoff=checker.cutoff
    )


def precompute(args: argparse.Namespace, paths_file: Path, db_path: Path) -> None:
    depth, entries = groups.load_depth_paths(paths_file)
    log_line(
        f"\n[{paths_file}] {len(entries)} paths of {3 * depth - 2} papers "
        f"(question depth {depth})"
    )
    chunk_size = max(1, len(entries) // (4 * args.workers))
    tasks = [
        (depth, entries[start : start + chunk_size])
        for start in range(0, len(entries), chunk_size)
    ]
    checks: list[tuple[str, Any]] = []
    with progress_bar(total=len(entries), desc="Checking paths", unit="path") as bar:
        for results in groups.map_in_workers(
            check_chunk,
            tasks,
            args.workers,
            start_path_worker,
            (args.graph_db, args.cutoff_date, args.only_explored),
        ):
            checks.extend(results)
            bar.update(len(results))
    with progress_bar(
        total=len(groups.GROUP_SIZES) * groups.ENUMERATION_CHUNKS,
        desc="Enumerating groups",
        unit="chunk",
    ) as bar:
        payload = groups.write_cache(
            paths_file,
            depth=depth,
            path_count=len(entries),
            checks=checks,
            database=portable_path(db_path),
            cutoff_date=args.cutoff_date,
            only_explored=args.only_explored,
            paths_name=portable_path(paths_file),
            limit=args.group_limit,
            workers=args.workers,
            on_chunk=bar.update,
        )
    usable = len(payload["usable_path_ids"])
    rejected = ", ".join(f"{reason} {count}" for reason, count in payload["rejected"].items())
    log_line(f"  usable paths: {usable} of {len(entries)} (rejected: {rejected or 'none'})")
    log_line(f"  conflicting pairs: {len(payload['conflicting_pairs']) // 2:,}")
    for size, info in payload["groups"].items():
        if info["method"] == "stored":
            log_line(f"  groups of {size}: {info['count']:,} disjoint, all stored")
        else:
            log_line(
                f"  groups of {size}: more than {args.group_limit:,} (about "
                f"{info['estimated_count']:,.0f}); draw-and-retry"
            )
    log_line(f"  cache: {groups.cache_path_for(paths_file)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--paths", type=Path, nargs="+", required=True)
    parser.add_argument("--graph-db", type=Path, required=True)
    parser.add_argument("--cutoff-date", required=True)
    parser.add_argument("--only-explored", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--group-limit",
        type=int,
        default=groups.GROUP_LIMIT,
        help="Store every disjoint group when there are at most this many.",
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1 or args.group_limit < 1:
        parser.error("--workers and --group-limit must be positive.")
    checker = groups.open_graph_checker(args.graph_db, args.cutoff_date, args.only_explored)
    db_path = checker.db_path
    checker.close()
    for paths_file in args.paths:
        precompute(args, paths_file, db_path)


if __name__ == "__main__":
    main()
