#!/usr/bin/env python3
"""Construct simple paths from seed papers using Semantic Scholar.

The target length is ``--num-hops`` nodes, including the seed node. The script
samples uniformly from the current node's candidate pool and backtracks when a
branch cannot reach the target length. It writes lightweight traversal
checkpoints, plus periodic graph/database snapshots that can be resumed with a
larger target length.

Arguments
---------
seed_titles
    Optional positional seed paper titles. Quote titles that contain spaces.
    Positional titles keep their provided order.
--input-file PATH
    Text file with one paper title per line, or a .csv file with the columns
    semantic_scholar_id and title (the output of
    endpoints/export_s2_bulk_newest_papers.py). A CSV seed with an id is
    fetched by that id instead of a title search, and is skipped when the
    client has no paper with that id; a CSV seed with an empty id is found by
    title. The title is the seed's key, so only the first CSV row of each title
    is used. Titles loaded from the file are shuffled before traversal. On resume, checkpoint paths are continued first,
    then already-used starting titles are filtered from the current shuffled file
    order and unused titles are explored.
--extend-from-output PATH
    Use paths from an earlier sampled_paths.json as new starting paths. Each
    selected path is extended from one endpoint until it reaches --num-hops.
    Existing paths in --output-dir keep their path IDs and list positions.
    This option cannot be combined with seed titles or --input-file.
--num-hops N
    Required target number of nodes in each path, including the start node.
--paths-per-seed N
    Number of independently sampled paths to construct for each seed.
    Default: 1.
--target-num-paths N
    Stop after this many complete paths have been discovered. By default, the
    script attempts paths_per_seed for every input paper.
--start-date DATE
    Inclusive minimum publication date for paper travel candidates.
    Accepts YYYY, YYYY-MM, or YYYY-MM-DD.
--end-date DATE
    Inclusive maximum publication date for paper travel candidates.
    Accepts YYYY, YYYY-MM, or YYYY-MM-DD.
--traversal-mode {mixed,papers-only,papers-forward,papers-backward,alternating}
    Node-type transitions allowed when selecting the next travel candidate.
    ``mixed`` preserves the current behavior, ``papers-only`` excludes authors,
    ``papers-forward`` follows papers that cite the current paper,
    ``papers-backward`` follows references cited by the current paper, and
    ``alternating`` enforces paper-author-paper transitions. Default: mixed.
--output-dir PATH
    Output root for traversal states and sampled_paths.json.
    The checkpoints/ directory stores only traversal .state.json files.
    Extension runs reuse sampled_paths.json as a persistent path list: short
    paths are extended in place and new paths are appended.
    If sampled_paths.json already exists in the output directory, its path count
    is loaded on resume for completed-path reporting.
    If checkpoints already exist and --load-graph / --resume-checkpoint are
    omitted, the newest checkpoint is resumed automatically.
    Default: output/semantic_scholar_paths.
--graph-dir PATH
    Directory for Semantic Scholar graph database snapshots.
    Default: resources/semantic_scholar_graph.
--output PATH
    Paths JSON path, rewritten once after each finished path.
    Default: <output-dir>/sampled_paths.json.
--load-graph PATH
    Existing graph database snapshot, legacy graph JSON, graphs directory, or
    output root to continue from.
--resume-checkpoint PATH
    Traversal .state.json checkpoint to resume. The checkpoint supplies seed
    titles, date bounds, visited/tried nodes, expanded nodes, and RNG state.
    Saved paths shorter than the requested --num-hops are continued.
    Extension runs save only their active path job; remaining work is rebuilt
    from the destination output and the current command's input.
--db-path PATH
    Working SQLite DB path. Default: <graph-dir>/semantic_scholar_paths.db.
--graph-backend {networkx,sqlite}
    ``sqlite`` queries the working database for every neighborhood, coordinates
    expansions across processes, and skips per-worker graph snapshots. Default:
    networkx.
--checkpoint-every N
    Save a lightweight traversal checkpoint after every N successful moves. Use
    0 to disable mid-path checkpoints. Default: 5.
--max-checkpoints N
    Maximum checkpoint sets to keep. Use 0 to keep all. Default: 2.
--random-seed INT
    Random seed for reproducible input order, endpoint order, and travel choices.
    An interrupted active extension restores its saved travel RNG state first.
--semantic-scholar-delay SECONDS
    Minimum delay between online Semantic Scholar requests. Ignored in bulk
    mode. Default: 2.0.
--semantic-scholar-backend {online,bulk}
    Use the online Semantic Scholar API or the local indexed bulk datasets.
    Default: online.
--bulk-data-root PATH
    Raw Semantic Scholar bulk dataset root used by the bulk backend.
--bulk-index-root PATH
    Built Semantic Scholar bulk index used by the bulk backend.
--bulk-title-search-batch-size N
    Upcoming seed titles resolved together in one bulk Spark title-search
    query; the online backend searches titles one at a time. Also the number of
    upcoming CSV seed ids fetched together by id, with either backend (at most
    500 online). Matches are cached for the run. Default: 250.
--bulk-title-min-match-score SCORE
    Bulk backend only: reject title matches (seeds and papers without a
    Semantic Scholar ID) scoring below this 0-100 value. Exact normalized
    title matches score 100. Default: 90.
--bulk-seed-timeout-minutes MINUTES
    Bulk backend only: skip a seed when its path is not complete after this many
    minutes of path search (title search is not counted). The unfinished path is
    dropped and the seed is not retried on resume. The time is checked between
    search steps, so one slow expansion can run over the limit. 0 disables the
    limit. Default: 0 (no limit).
--no-tqdm-bar {0,1}
    Set to 1 to disable tqdm progress bars. Informational log messages are still
    printed. Default: 0.
--log-level {DEBUG,INFO,WARNING,ERROR}
    Console log level. Default: INFO.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import os
import random
import re
import signal
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import orjson
import requests
from loguru import logger
from tqdm import tqdm

from sample_paths import build_steps_from_trace
from src.clients.semantic_scholar_bulk_client import (
    DEFAULT_DATA_ROOT as DEFAULT_BULK_DATA_ROOT,
    DEFAULT_INDEX_ROOT as DEFAULT_BULK_INDEX_ROOT,
    SemanticScholarBulkClient,
)
from src.clients.semantic_scholar_client import (
    SemanticScholarClient,
    SemanticScholarRateLimitError,
)
from src.graph.checkpoints import build_graph_stem
from src.graph.db_network import SQLiteCitationNetwork
from src.graph.network import CitationNetwork
from src.graph.readers import DBReader
from src.graph.reference_validation import reference_validation_columns
from src.graph.storage import DatabaseManager, copy_sqlite_database
from src.setup import bootstrap_runtime
from src.setup.constants import (
    EDGE_AUTHORSHIP,
    EDGE_CITED_BY,
    EDGE_CITING,
    NODE_AUTHOR,
    NODE_PAPER,
    author_nid,
    nid_db_id,
    nid_type,
    normalize_text,
)
from src.setup.pipeline_parameters import (
    DISTRACTING_CONTENT_TYPE_PARAMETER_FIELD,
    NEEDLE_ORDER_PARAMETER_FIELD,
    NEEDLE_TYPE_PARAMETER_FIELD,
    NUM_HOPS_PARAMETER_FIELD,
    PARAMETERS_FIELD,
    TARGET_CONTEXT_LENGTH_PARAMETER_FIELD,
    merge_parameters,
)

DEFAULT_OUTPUT_DIR = Path("output") / "semantic_scholar_paths"
DEFAULT_GRAPH_DIR = Path("resources") / "semantic_scholar_graph"
DEFAULT_SHARED_DB_NAME = "semantic_scholar_paths.db"
DEFAULT_PATHS_JSON_NAME = "sampled_paths.json"
SEMANTIC_GRAPH_VERSION_KEEP_COUNT = 1
# Default number of completed paths between full graph snapshots. Larger values
# speed up long runs; resume relies on the working DB and per-path state
# checkpoints between snapshots. Override with --graph-snapshot-every.
GRAPH_SNAPSHOT_COMPLETED_PATH_INTERVAL = 10
SEMANTIC_GRAPH_ARTIFACT_SUFFIXES = (
    ".json.tmp",
    ".db-shm",
    ".db-wal",
    ".json",
    ".db",
    ".gexf",
)
SEMANTIC_GRAPH_VERSION_RE = re.compile(r"^(semantic_scholar_paths_\d{8}_\d{6})(?:_\d{3}_.+)?$")
RATE_LIMIT_DEFER_WAIT_SECONDS = 30.0
PATH_COMPLETE = "complete"
PATH_EXHAUSTED = "exhausted"
PATH_DEFERRED = "deferred"
PATH_TIMED_OUT = "timed_out"
TRAVERSAL_MODE_MIXED = "mixed"
TRAVERSAL_MODE_PAPERS_ONLY = "papers-only"
TRAVERSAL_MODE_PAPERS_FORWARD = "papers-forward"
TRAVERSAL_MODE_PAPERS_BACKWARD = "papers-backward"
TRAVERSAL_MODE_ALTERNATING = "alternating"
TRAVERSAL_MODES = (
    TRAVERSAL_MODE_MIXED,
    TRAVERSAL_MODE_PAPERS_ONLY,
    TRAVERSAL_MODE_PAPERS_FORWARD,
    TRAVERSAL_MODE_PAPERS_BACKWARD,
    TRAVERSAL_MODE_ALTERNATING,
)
STATE_VERSION = 1
PERSISTENT_STATE_VERSION = 1
GRAPH_BACKEND_NETWORKX = "networkx"
GRAPH_BACKEND_SQLITE = "sqlite"
GRAPH_BACKENDS = (GRAPH_BACKEND_NETWORKX, GRAPH_BACKEND_SQLITE)
SEMANTIC_SCHOLAR_BACKEND_ONLINE = "online"
SEMANTIC_SCHOLAR_BACKEND_BULK = "bulk"
SEMANTIC_SCHOLAR_BACKENDS = (
    SEMANTIC_SCHOLAR_BACKEND_ONLINE,
    SEMANTIC_SCHOLAR_BACKEND_BULK,
)
# Bulk-backend title search settings, shared with
# endpoints/expand_unexplored_papers.py.
# Titles per Spark title-search query: fuzzy matching can consider up to
# search_candidate_limit papers per title, so this bounds Spark's
# queryIndex/corpusId intermediate and its broadcast join.
BULK_TITLE_SEARCH_BATCH_SIZE = 250
# Lowest accepted title-match score (exact normalized titles score 100).
BULK_TITLE_MIN_MATCH_SCORE = 90.0
# A title search that only resolves an ID needs just paperId: the expansion
# call right after it fetches the full paper anyway.
BULK_TITLE_SEARCH_FIELDS = ["paperId"]
# Columns of a CSV --input-file, as written by
# endpoints/export_s2_bulk_newest_papers.py.
SEED_CSV_COLUMNS = ("semantic_scholar_id", "title")
# Most ids the online API accepts in one /paper/batch request.
ONLINE_PAPER_BATCH_LIMIT = 500
# Minutes of path search per seed before the seed is skipped (0 disables).
BULK_SEED_TIMEOUT_MINUTES = 0.0
RUN_MODE_PERSISTENT = "persistent_paths"
PERSISTENT_PATH_STORE_VERSION = 1
PATH_STATUS_ACTIVE = "active"
PATH_STATUS_COMPLETE = "complete"
PATH_STATUS_INCOMPLETE = "incomplete"
ENDPOINT_FIRST = "first"
ENDPOINT_LAST = "last"


class RawDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


@dataclass(frozen=True)
class TravelCandidate:
    node_id: str
    edge: tuple[str, str, dict[str, Any]]


@dataclass
class TravelFrame:
    node_id: str
    incoming_edge: tuple[str, str, dict[str, Any]] | None = None
    tried_nodes: set[str] = field(default_factory=set)


@dataclass
class PathRecord:
    path_id: str
    path_index: int
    trace: list[tuple[str, tuple[str, str, dict[str, Any]] | None]]
    status: str = PATH_STATUS_INCOMPLETE
    source_path_id: str | None = None
    extended_from_endpoint: str | None = None
    original_node_count: int | None = None
    seed_title: str | None = None


@dataclass
class ActiveExtensionJob:
    path_id: str
    path_index: int
    base_trace: list[tuple[str, tuple[str, str, dict[str, Any]] | None]]
    frames: list[TravelFrame]
    locked_frame_count: int
    endpoint_order: list[str]
    endpoint_attempt_index: int = 0
    traversal_mode: str = TRAVERSAL_MODE_MIXED


@dataclass
class BuildContext:
    network: CitationNetwork | SQLiteCitationNetwork
    client: SemanticScholarClient | SemanticScholarBulkClient
    graph_dir: Path
    checkpoint_dir: Path
    output_path: Path
    working_db_path: Path
    run_stem: str
    seed_titles: list[str]
    paths_per_seed: int
    target_num_paths: int | None
    target_num_nodes: int
    start_date: date | None
    end_date: date | None
    checkpoint_every: int
    max_checkpoints: int
    graph_snapshot_every: int
    expanded_nodes: set[str]
    rng: random.Random
    completed_traces: list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]]
    traversal_mode: str = TRAVERSAL_MODE_MIXED
    no_tqdm_bar: int = 0
    completed_path_ids: list[str] = field(default_factory=list)
    current_seed_index: int = 0
    current_repeat_index: int = 0
    checkpoint_count: int = 0
    venue_checked_nodes: set[str] = field(default_factory=set)
    last_graph_snapshot_path: Path | None = None
    graph_snapshot_node_ids: set[str] = field(default_factory=set)
    graph_snapshot_edge_keys: set[tuple[str, str]] = field(default_factory=set)
    last_graph_snapshot_complete_count: int = 0
    shutdown_checkpoint_path: Path | None = None
    # Cache of built sampled-path output entries, keyed by trace_key, so completed
    # paths are reconstructed from the DB only once instead of on every write.
    result_path_entry_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Bulk backend only. The local index never changes during a run, so a seed's
    # title match and a paper the index could not find are remembered per run.
    bulk_title_search_batch_size: int = BULK_TITLE_SEARCH_BATCH_SIZE
    bulk_title_min_match_score: float = BULK_TITLE_MIN_MATCH_SCORE
    seed_title_matches: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    bulk_not_found_nodes: set[str] = field(default_factory=set)
    # Seed title -> Semantic Scholar id from a CSV --input-file (either backend).
    # These seeds are fetched by id; their papers are cached in seed_title_matches.
    seed_semantic_scholar_ids: dict[str, str] = field(default_factory=dict)
    # 0 means no time limit per seed path.
    bulk_seed_timeout_seconds: float = 0.0
    # Deepest path length (in nodes) reached by the last construct_path() call.
    last_path_max_nodes: int = 0
    # Failed seeds (no path after full backtracking, or timed out): deepest path
    # length -> number of seeds. Saved in checkpoints so it continues on resume.
    failed_seed_max_nodes: dict[int, int] = field(default_factory=dict)


def uses_sqlite_backend(network: Any) -> bool:
    return bool(getattr(network, "is_sqlite_backend", False))


def uses_bulk_client(context: Any) -> bool:
    return isinstance(getattr(context, "client", None), SemanticScholarBulkClient)


def seed_path_deadline(context: Any) -> float | None:
    """Return the time.monotonic() deadline for a seed path starting now, or
    None when there is no limit (online backend, or the limit is 0)."""
    timeout_seconds = getattr(context, "bulk_seed_timeout_seconds", 0.0)
    if timeout_seconds <= 0 or not uses_bulk_client(context):
        return None
    return time.monotonic() + timeout_seconds


def record_failed_seed(context: Any) -> str:
    """Count the deepest path length of a failed seed and return a log summary,
    e.g. ``reached=31 | failed n=412: mean=27.3 median=28 min=3 max=47``."""
    reached = getattr(context, "last_path_max_nodes", 0)
    counts = getattr(context, "failed_seed_max_nodes", None)
    if counts is None:
        counts = {}
        context.failed_seed_max_nodes = counts
    counts[reached] = counts.get(reached, 0) + 1
    return f"reached={reached} | {format_failed_seed_stats(counts)}"


def failed_seed_max_nodes_from_state(state: dict[str, Any]) -> dict[int, int]:
    """Read the ``[deepest path length, seed count]`` pairs saved in a checkpoint."""
    return {
        int(nodes): int(count)
        for nodes, count in state.get("failed_seed_max_nodes") or []
    }


def format_failed_seed_stats(counts: dict[int, int]) -> str:
    """Summarize a non-empty ``{deepest path length: seed count}`` table."""
    ordered = sorted(counts.items())
    total = sum(count for _nodes, count in ordered)
    mean = sum(nodes * count for nodes, count in ordered) / total

    def value_at(position: int) -> int:
        seen = 0
        for nodes, count in ordered:
            seen += count
            if position < seen:
                return nodes
        raise IndexError(position)

    median = (value_at((total - 1) // 2) + value_at(total // 2)) / 2
    return (
        f"failed n={total}: mean={mean:.1f} median={median:g} "
        f"min={ordered[0][0]} max={ordered[-1][0]}"
    )


def is_not_found_error(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    return isinstance(exc, requests.HTTPError) and response is not None and response.status_code == 404


def drop_weak_title_match(
    paper: dict[str, Any] | None,
    min_match_score: float,
) -> dict[str, Any] | None:
    """Treat a bulk title match scoring below ``min_match_score`` as no match."""
    if paper is not None and float(paper.get("matchScore") or 0.0) < min_match_score:
        logger.debug(
            "Rejecting weak title match {}: score={} < {}",
            paper.get("paperId"),
            paper.get("matchScore"),
            min_match_score,
        )
        return None
    return paper


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return value


def nonnegative_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return value


def title_match_score(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError("title match score must be a number") from None
    if not 0.0 <= value <= 100.0:
        raise argparse.ArgumentTypeError("title match score must be between 0 and 100")
    return value


def parse_date_bound(raw: str, *, is_end: bool) -> date:
    text = raw.strip()
    if re.fullmatch(r"\d{4}", text):
        year = int(text)
        return date(year, 12, 31) if is_end else date(year, 1, 1)
    if re.fullmatch(r"\d{4}-\d{2}", text):
        year, month = (int(part) for part in text.split("-"))
        day = calendar.monthrange(year, month)[1] if is_end else 1
        return date(year, month, day)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise argparse.ArgumentTypeError("date must use YYYY, YYYY-MM, or YYYY-MM-DD")
    return datetime.strptime(text, "%Y-%m-%d").date()


def parse_start_date(raw: str) -> date:
    return parse_date_bound(raw, is_end=False)


def parse_end_date(raw: str) -> date:
    return parse_date_bound(raw, is_end=True)


def resolve_traversal_mode(
    requested_mode: str | None,
    saved_mode: str | None = None,
    *,
    saved_source: str = "saved run",
) -> str:
    if requested_mode is not None and requested_mode not in TRAVERSAL_MODES:
        raise SystemExit(f"Unsupported --traversal-mode: {requested_mode}")
    if saved_mode is not None and saved_mode not in TRAVERSAL_MODES:
        raise SystemExit(f"Unsupported traversal mode in {saved_source}: {saved_mode}")
    if requested_mode is not None and saved_mode is not None and requested_mode != saved_mode:
        raise SystemExit(
            f"--traversal-mode {requested_mode!r} does not match "
            f"{saved_source} mode {saved_mode!r}"
        )
    return requested_mode or saved_mode or TRAVERSAL_MODE_MIXED


def parse_paper_date(raw: str) -> date | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    if re.fullmatch(r"\d{4}", text):
        return date(int(text), 1, 1)
    if len(text) >= 10:
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def paper_allowed_by_date(
    network: CitationNetwork,
    node_id: str,
    start_date: date | None,
    end_date: date | None,
) -> bool:
    if nid_type(node_id) != NODE_PAPER:
        return True
    if start_date is None and end_date is None:
        return True
    row = network.db.get_paper(nid_db_id(node_id))
    published = parse_paper_date(row["published"] if row else "")
    if published is None:
        return False
    if start_date is not None and published < start_date:
        return False
    if end_date is not None and published > end_date:
        return False
    return True


def is_seed_csv(seed_path: Path) -> bool:
    return seed_path.suffix.lower() == ".csv"


def read_seed_csv_rows(seed_path: Path) -> tuple[list[tuple[str, str]], int]:
    """Return the (title, semantic_scholar_id) rows of a seed CSV, in file order,
    and the number of rows dropped because their title repeats an earlier row.

    The title is the seed's key, so only the first row of each title is kept.
    Rows without a title are skipped. The id may be empty.
    """
    with seed_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [column for column in SEED_CSV_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(
                f"Seed CSV {seed_path} must have the columns {', '.join(SEED_CSV_COLUMNS)}; "
                f"missing: {', '.join(missing)}"
            )
        rows: list[tuple[str, str]] = []
        seen_keys: set[str] = set()
        repeated = 0
        for row in reader:
            title = str(row["title"] or "").strip()
            key = seed_title_key(title)
            if not key:
                continue
            if key in seen_keys:
                repeated += 1
                continue
            seen_keys.add(key)
            rows.append((title, str(row["semantic_scholar_id"] or "").strip()))
    return rows, repeated


def read_seed_titles(args: argparse.Namespace, rng: random.Random) -> list[str]:
    titles = [title.strip() for title in args.seed_titles if title.strip()]
    if args.input_file:
        seed_path = Path(args.input_file)
        if is_seed_csv(seed_path):
            rows, repeated = read_seed_csv_rows(seed_path)
            file_titles = [title for title, _ in rows]
            if repeated:
                logger.info(
                    "Dropped {} seed CSV rows whose title repeats an earlier row: {}",
                    repeated,
                    seed_path,
                )
        else:
            file_titles = [
                line.strip()
                for line in seed_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        rng.shuffle(file_titles)
        titles.extend(file_titles)
    return titles


def read_seed_semantic_scholar_ids(args: argparse.Namespace) -> dict[str, str]:
    """Map each seed title of a CSV --input-file to its Semantic Scholar id.

    Text input files and CSV rows with an empty id give no entry.
    """
    input_file = getattr(args, "input_file", None)
    if not input_file or not is_seed_csv(Path(input_file)):
        return {}
    rows, _ = read_seed_csv_rows(Path(input_file))
    return {title: paper_id for title, paper_id in rows if paper_id}


def validate_input_mode(args: argparse.Namespace) -> None:
    if not getattr(args, "extend_from_output", None):
        return
    if getattr(args, "input_file", None) or any(
        str(title).strip() for title in getattr(args, "seed_titles", [])
    ):
        raise SystemExit(
            "--extend-from-output cannot be combined with seed titles or --input-file"
        )


def seed_title_key(title: str) -> str:
    return normalize_text(str(title or "").strip())


def add_seed_title_key(keys: set[str], title: str) -> None:
    key = seed_title_key(title)
    if key:
        keys.add(key)


def paper_node_title(network: CitationNetwork, node_id: str) -> str:
    if nid_type(node_id) != NODE_PAPER:
        return ""
    row = network.db.get_paper(nid_db_id(node_id))
    return str(row["title"] or "") if row else ""


def add_trace_start_title_key(
    keys: set[str],
    network: CitationNetwork,
    trace: list[tuple[str, tuple[str, str, dict[str, Any]] | None]],
) -> None:
    if trace:
        add_seed_title_key(keys, paper_node_title(network, trace[0][0]))


def add_frames_start_title_key(
    keys: set[str],
    network: CitationNetwork,
    frames: list[TravelFrame] | None,
) -> None:
    if frames:
        add_seed_title_key(keys, paper_node_title(network, frames[0].node_id))


def looks_like_local_node_id(value: str) -> bool:
    return bool(re.fullmatch(r"(?:paper|author):[0-9A-Za-z-]+", value))


def clean_semantic_scholar_id(
    raw_id: Any,
    *,
    node_id: str,
    entity_label: str,
) -> str:
    semantic_scholar_id = str(raw_id or "").strip()
    if not semantic_scholar_id:
        return ""
    if looks_like_local_node_id(semantic_scholar_id):
        logger.warning(
            "{} {} has local graph node ID where Semantic Scholar ID is required; ignoring id={}",
            entity_label,
            node_id,
            semantic_scholar_id,
        )
        return ""
    return semantic_scholar_id


def collect_used_seed_title_keys(
    network: CitationNetwork,
    *,
    state_seed_titles: list[str],
    current_seed_index: int,
    current_repeat_index: int,
    completed_traces: list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]],
    pending_frames: list[list[TravelFrame]],
    resume_frames: list[TravelFrame] | None,
) -> set[str]:
    keys: set[str] = set()
    used_count = max(0, min(current_seed_index, len(state_seed_titles)))
    for title in state_seed_titles[:used_count]:
        add_seed_title_key(keys, title)
    if current_repeat_index > 0 and current_seed_index < len(state_seed_titles):
        add_seed_title_key(keys, state_seed_titles[current_seed_index])

    for trace in completed_traces:
        add_trace_start_title_key(keys, network, trace)
    for frames in pending_frames:
        add_frames_start_title_key(keys, network, frames)
    add_frames_start_title_key(keys, network, resume_frames)
    return keys


def filter_unused_seed_titles(seed_titles: list[str], used_keys: set[str]) -> list[str]:
    return [title for title in seed_titles if seed_title_key(title) not in used_keys]


def normalize_paper_title(title: str) -> str:
    """Collapse embedded whitespace (including line breaks) in a paper title.

    Semantic Scholar sometimes returns titles with literal line breaks baked
    in (e.g. copied from a PDF's wrapped title). Collapsing them here, before
    the title is stored, keeps every downstream consumer of the DB title
    single-line.
    """
    return re.sub(r"\s+", " ", title).strip()


def paper_fields(paper: dict[str, Any]) -> dict[str, str]:
    external_ids = paper.get("externalIds") or {}
    open_access_pdf = paper.get("openAccessPdf") or {}
    published = str(paper.get("publicationDate") or "")
    if not published and paper.get("year"):
        published = f"{int(paper['year']):04d}-01-01"
    return {
        "arxiv_id": str(external_ids.get("ArXiv") or ""),
        "semantic_scholar_id": clean_semantic_scholar_id(
            paper.get("paperId"),
            node_id=str(paper.get("paperId") or ""),
            entity_label="Paper",
        ),
        "title": normalize_paper_title(str(paper.get("title") or "")),
        "abstract": str(paper.get("abstract") or ""),
        "published": published,
        "venue": str(paper.get("venue") or ""),
        "pdf_url": str(open_access_pdf.get("url") or ""),
    }


def upsert_semantic_author(
    network: CitationNetwork | SQLiteCitationNetwork,
    author: dict[str, Any],
) -> str | None:
    semantic_scholar_id = clean_semantic_scholar_id(
        author.get("authorId"),
        node_id=str(author.get("authorId") or ""),
        entity_label="Author",
    )
    name = str(author.get("name") or "").strip()
    if not name:
        return None

    if semantic_scholar_id and not uses_sqlite_backend(network):
        row = network.db.get_author_by_semantic_scholar_id(semantic_scholar_id)
        if row is not None:
            if not row["name"]:
                network.db.update_author(row["id"], name=name)
            network._author_index[normalize_text(name)] = str(row["id"])
            node_id = author_nid(str(row["id"]))
            if node_id not in network.G:
                network.G.add_node(node_id)
            return node_id

    return network.upsert_author(
        name=name,
        semantic_scholar_id=semantic_scholar_id,
    )


def attach_paper_authors(
    network: CitationNetwork | SQLiteCitationNetwork,
    paper_node: str,
    authors: list[dict[str, Any]],
) -> None:
    for position, author in enumerate(authors, start=1):
        author_node = upsert_semantic_author(network, author)
        if author_node is None:
            continue
        network.add_authorship(paper_node, author_node, position=position)


def upsert_semantic_paper(
    network: CitationNetwork | SQLiteCitationNetwork,
    paper: dict[str, Any],
    *,
    depth: int = -1,
    explored: bool = False,
) -> str | None:
    fields = paper_fields(paper)
    if not fields["title"] and not fields["semantic_scholar_id"]:
        return None

    if uses_sqlite_backend(network):
        paper_node = network.upsert_paper(
            **fields,
            explored=explored,
            text_path="",
            depth=depth,
        )
        if paper_node is not None:
            attach_paper_authors(network, paper_node, list(paper.get("authors") or []))
        return paper_node

    row = None
    if fields["semantic_scholar_id"]:
        row = network.db.get_paper_by_semantic_scholar_id(fields["semantic_scholar_id"])

    if row is not None:
        paper_node = network.update_existing_paper(
            str(row["id"]),
            **fields,
            explored=explored,
            text_path="",
            depth=depth,
        )
    else:
        paper_node = network.upsert_paper(
            **fields,
            explored=explored,
            text_path="",
            depth=depth,
        )

    if paper_node is not None:
        attach_paper_authors(network, paper_node, list(paper.get("authors") or []))
    return paper_node


def refresh_current_paper(
    network: CitationNetwork | SQLiteCitationNetwork,
    node_id: str,
    paper: dict[str, Any],
    references: list[dict[str, Any]] | None = None,
    *,
    after_cutoff: bool = False,
) -> None:
    """Refresh the expanded paper's row with one UPDATE.

    Pass ``references`` only when the reference side was fetched: the same
    UPDATE then also stores ``reference_count``/``valid_references``.
    ``after_cutoff=True`` (a --bulk-filter-neighbors-by-cutoff fetch) skips
    ``valid_references``: ``references`` there is a deliberately restricted
    subset, so it would always compare as "invalid" against the true count.
    """
    validation_columns = (
        reference_validation_columns(
            paper.get("referenceCount"), len(references), restricted=after_cutoff
        )
        if references is not None
        else {}
    )
    network.update_existing_paper(
        nid_db_id(node_id),
        **paper_fields(paper),
        explored=False,
        text_path="",
        **validation_columns,
    )
    attach_paper_authors(network, node_id, list(paper.get("authors") or []))


def fetch_current_paper_expansion(
    context: BuildContext,
    node_id: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    row = context.network.db.get_paper(nid_db_id(node_id))
    if row is None:
        return None, [], []

    semantic_scholar_id = clean_semantic_scholar_id(
        row["semantic_scholar_id"],
        node_id=node_id,
        entity_label="Paper",
    )
    if semantic_scholar_id:
        return context.client.get_paper_with_citations_and_references(semantic_scholar_id)

    if not row["title"]:
        return None, [], []

    if uses_bulk_client(context):
        paper = drop_weak_title_match(
            context.client.search_by_title(str(row["title"]), fields=BULK_TITLE_SEARCH_FIELDS),
            context.bulk_title_min_match_score,
        )
    else:
        paper = context.client.search_by_title(str(row["title"]))
    if not paper or not paper.get("paperId"):
        return paper, [], []

    resolved_id = clean_semantic_scholar_id(
        paper.get("paperId"),
        node_id=node_id,
        entity_label="Resolved paper",
    )
    if not resolved_id:
        return paper, [], []

    return context.client.get_paper_with_citations_and_references(resolved_id)


def expand_paper_node(context: BuildContext, node_id: str) -> None:
    sqlite_backend = uses_sqlite_backend(context.network)
    if not sqlite_backend and node_id in context.expanded_nodes:
        return

    try:
        if sqlite_backend and context.network.acquire_expansion(node_id) == "complete":
            return
        paper, citations, references = fetch_current_paper_expansion(context, node_id)
        if not paper or not paper.get("paperId"):
            logger.warning("Could not resolve Semantic Scholar paper for {}", node_id)
            if sqlite_backend:
                context.network.complete_expansion(node_id)
            context.expanded_nodes.add(node_id)
            return

        semantic_scholar_id = clean_semantic_scholar_id(
            paper.get("paperId"),
            node_id=node_id,
            entity_label="Resolved paper",
        )
        if not semantic_scholar_id:
            logger.warning(
                "Resolved paper for {} has no valid Semantic Scholar ID",
                node_id,
            )
            if sqlite_backend:
                context.network.complete_expansion(node_id)
            context.expanded_nodes.add(node_id)
            return

        if sqlite_backend:
            with context.network.expansion_batch():
                ingest_paper_expansion(context, node_id, paper, citations, references)
                context.network.complete_expansion(node_id)
        else:
            ingest_paper_expansion(context, node_id, paper, citations, references)
    except BaseException:
        if sqlite_backend:
            context.network.release_expansion(node_id)
        raise

    logger.debug(
        "Expanded paper {} | citations={} references={}",
        node_id,
        len(citations),
        len(references),
    )


def ingest_paper_expansion(
    context: BuildContext,
    node_id: str,
    paper: dict[str, Any],
    citations: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> None:
    refresh_current_paper(context.network, node_id, paper, references)
    for item in citations:
        citing_paper = item.get("citingPaper") or {}
        citing_node = upsert_semantic_paper(context.network, citing_paper)
        if citing_node is not None:
            context.network.add_citation(citing_node, node_id)
    for item in references:
        cited_paper = item.get("citedPaper") or {}
        cited_node = upsert_semantic_paper(context.network, cited_paper)
        if cited_node is not None:
            context.network.add_citation(node_id, cited_node)
    context.network.mark_explored(node_id)
    context.expanded_nodes.add(node_id)


def expand_author_node(context: BuildContext, node_id: str) -> None:
    sqlite_backend = uses_sqlite_backend(context.network)
    if not sqlite_backend and node_id in context.expanded_nodes:
        return

    try:
        if sqlite_backend and context.network.acquire_expansion(node_id) == "complete":
            return
        row = context.network.db.get_author(nid_db_id(node_id))
        semantic_scholar_id = (
            clean_semantic_scholar_id(
                row["semantic_scholar_id"],
                node_id=node_id,
                entity_label="Author",
            )
            if row
            else ""
        )
        if not semantic_scholar_id:
            logger.warning("Author {} has no Semantic Scholar ID; cannot expand", node_id)
            if sqlite_backend:
                context.network.complete_expansion(node_id)
            context.expanded_nodes.add(node_id)
            return

        papers = context.client.get_all_papers_of_author(semantic_scholar_id) or []
        if sqlite_backend:
            with context.network.expansion_batch():
                for paper in papers:
                    upsert_semantic_paper(context.network, paper)
                context.expanded_nodes.add(node_id)
                context.network.complete_expansion(node_id)
        else:
            for paper in papers:
                upsert_semantic_paper(context.network, paper)
            context.expanded_nodes.add(node_id)
    except BaseException:
        if sqlite_backend:
            context.network.release_expansion(node_id)
        raise
    logger.debug(
        "Expanded author {} | papers={}",
        node_id,
        len(papers),
    )


def expand_node(context: BuildContext, node_id: str) -> None:
    node_type = nid_type(node_id)
    if node_type == NODE_PAPER:
        expand_paper_node(context, node_id)
    elif node_type == NODE_AUTHOR:
        expand_author_node(context, node_id)


def add_candidate(
    candidates: dict[str, TravelCandidate],
    candidate: TravelCandidate,
) -> None:
    candidates.setdefault(candidate.node_id, candidate)


def collect_paper_candidates(
    context: BuildContext,
    node_id: str,
) -> list[TravelCandidate]:
    if uses_sqlite_backend(getattr(context, "network", None)):
        return collect_sqlite_candidates(context, node_id)
    graph = context.network.G
    candidates: dict[str, TravelCandidate] = {}

    for target in graph.successors(node_id):
        edge_attrs = dict(graph[node_id][target])
        edge_type = edge_attrs.get("edge_type", "")
        if edge_type == EDGE_CITING and paper_allowed_by_date(
            context.network,
            target,
            context.start_date,
            context.end_date,
        ):
            add_candidate(candidates, TravelCandidate(target, (node_id, target, edge_attrs)))
        elif edge_type == EDGE_AUTHORSHIP:
            add_candidate(candidates, TravelCandidate(target, (node_id, target, edge_attrs)))

    for source in graph.predecessors(node_id):
        edge_attrs = dict(graph[source][node_id])
        if edge_attrs.get("edge_type", "") != EDGE_CITING:
            continue
        if not paper_allowed_by_date(
            context.network,
            source,
            context.start_date,
            context.end_date,
        ):
            continue
        reverse_attrs = dict(edge_attrs)
        reverse_attrs["edge_type"] = EDGE_CITED_BY
        add_candidate(candidates, TravelCandidate(source, (node_id, source, reverse_attrs)))

    return list(candidates.values())


def collect_author_candidates(
    context: BuildContext,
    node_id: str,
) -> list[TravelCandidate]:
    if uses_sqlite_backend(context.network):
        return collect_sqlite_candidates(context, node_id)
    graph = context.network.G
    candidates: dict[str, TravelCandidate] = {}
    for source in graph.predecessors(node_id):
        edge_attrs = dict(graph[source][node_id])
        if edge_attrs.get("edge_type", "") != EDGE_AUTHORSHIP:
            continue
        if not paper_allowed_by_date(
            context.network,
            source,
            context.start_date,
            context.end_date,
        ):
            continue
        add_candidate(candidates, TravelCandidate(source, (node_id, source, edge_attrs)))
    return list(candidates.values())


def collect_sqlite_candidates(
    context: BuildContext,
    node_id: str,
) -> list[TravelCandidate]:
    candidates: dict[str, TravelCandidate] = {}
    for candidate_node_id, raw_attrs in context.network.list_neighbors(node_id):
        edge_attrs = dict(raw_attrs)
        published = str(edge_attrs.pop("published", ""))
        if nid_type(candidate_node_id) == NODE_PAPER:
            published_date = parse_paper_date(published)
            if published_date is None and (context.start_date or context.end_date):
                continue
            if context.start_date is not None and published_date < context.start_date:
                continue
            if context.end_date is not None and published_date > context.end_date:
                continue
        add_candidate(
            candidates,
            TravelCandidate(
                candidate_node_id,
                (node_id, candidate_node_id, edge_attrs),
            ),
        )
    return list(candidates.values())


def collect_candidates(
    context: BuildContext,
    node_id: str,
) -> list[TravelCandidate]:
    bulk_client = uses_bulk_client(context)
    if bulk_client and node_id in context.bulk_not_found_nodes:
        return []
    try:
        expand_node(context, node_id)
    except SemanticScholarRateLimitError:
        logger.warning(
            "Rate limited while expanding {}; deferring this branch without marking it explored",
            node_id,
        )
        raise
    except Exception as exc:
        logger.warning(
            "Expansion failed for {}; treating it as a dead end so traversal can backtrack: {}",
            node_id,
            exc,
        )
        if not uses_sqlite_backend(context.network):
            context.expanded_nodes.add(node_id)
        if bulk_client and is_not_found_error(exc):
            # Kept in memory only: another backend or a newer index may still find it.
            context.bulk_not_found_nodes.add(node_id)
        return []
    if nid_type(node_id) == NODE_AUTHOR:
        return collect_author_candidates(context, node_id)
    return collect_paper_candidates(context, node_id)


def candidate_allowed_from_current(
    current_node_id: str,
    candidate_node_id: str,
    traversal_mode: str = TRAVERSAL_MODE_MIXED,
    *,
    edge_type: str = "",
) -> bool:
    current_type = nid_type(current_node_id)
    candidate_type = nid_type(candidate_node_id)

    if traversal_mode == TRAVERSAL_MODE_PAPERS_ONLY:
        return current_type == NODE_PAPER and candidate_type == NODE_PAPER
    if traversal_mode == TRAVERSAL_MODE_PAPERS_FORWARD:
        return (
            current_type == NODE_PAPER
            and candidate_type == NODE_PAPER
            and edge_type == EDGE_CITED_BY
        )
    if traversal_mode == TRAVERSAL_MODE_PAPERS_BACKWARD:
        return (
            current_type == NODE_PAPER
            and candidate_type == NODE_PAPER
            and edge_type == EDGE_CITING
        )
    if traversal_mode == TRAVERSAL_MODE_ALTERNATING:
        return (current_type, candidate_type) in (
            (NODE_PAPER, NODE_AUTHOR),
            (NODE_AUTHOR, NODE_PAPER),
        )
    if traversal_mode != TRAVERSAL_MODE_MIXED:
        raise ValueError(f"Unsupported traversal mode: {traversal_mode}")
    if current_type == NODE_AUTHOR:
        return candidate_type == NODE_PAPER
    if current_type == NODE_PAPER:
        return candidate_type in (NODE_PAPER, NODE_AUTHOR)
    return False


def trace_from_frames(
    frames: list[TravelFrame],
) -> list[tuple[str, tuple[str, str, dict[str, Any]] | None]]:
    return [(frame.node_id, frame.incoming_edge) for frame in frames]


def frames_from_trace(
    trace: list[tuple[str, tuple[str, str, dict[str, Any]] | None]],
) -> list[TravelFrame]:
    frames = [
        TravelFrame(node_id=node_id, incoming_edge=incoming_edge)
        for node_id, incoming_edge in trace
    ]
    for index in range(len(frames) - 1):
        frames[index].tried_nodes.add(frames[index + 1].node_id)
    return frames


def reverse_traversal_mode(traversal_mode: str) -> str:
    if traversal_mode == TRAVERSAL_MODE_PAPERS_FORWARD:
        return TRAVERSAL_MODE_PAPERS_BACKWARD
    if traversal_mode == TRAVERSAL_MODE_PAPERS_BACKWARD:
        return TRAVERSAL_MODE_PAPERS_FORWARD
    return traversal_mode


def reverse_edge(
    edge: tuple[str, str, dict[str, Any]],
) -> tuple[str, str, dict[str, Any]]:
    source, target, raw_attrs = edge
    attrs = dict(raw_attrs)
    edge_type = str(attrs.get("edge_type") or "")
    if edge_type == EDGE_CITING:
        attrs["edge_type"] = EDGE_CITED_BY
    elif edge_type == EDGE_CITED_BY:
        attrs["edge_type"] = EDGE_CITING
    return target, source, attrs


def reverse_trace(
    trace: list[tuple[str, tuple[str, str, dict[str, Any]] | None]],
) -> list[tuple[str, tuple[str, str, dict[str, Any]] | None]]:
    if not trace:
        return []

    reversed_trace = [(trace[-1][0], None)]
    for index in range(len(trace) - 1, 0, -1):
        incoming_edge = trace[index][1]
        if incoming_edge is None:
            raise ValueError("non-start path node has no incoming edge")
        reversed_trace.append((trace[index - 1][0], reverse_edge(incoming_edge)))
    return reversed_trace


def active_extension_endpoint(job: ActiveExtensionJob) -> str:
    return job.endpoint_order[job.endpoint_attempt_index]


def active_extension_canonical_trace(
    job: ActiveExtensionJob,
) -> list[tuple[str, tuple[str, str, dict[str, Any]] | None]]:
    trace = trace_from_frames(job.frames)
    if active_extension_endpoint(job) == ENDPOINT_FIRST:
        return reverse_trace(trace)
    return trace


def extension_frames_for_endpoint(
    trace: list[tuple[str, tuple[str, str, dict[str, Any]] | None]],
    endpoint: str,
) -> list[TravelFrame]:
    if endpoint == ENDPOINT_FIRST:
        return frames_from_trace(reverse_trace(trace))
    if endpoint == ENDPOINT_LAST:
        return frames_from_trace(trace)
    raise ValueError(f"Unsupported endpoint: {endpoint}")


def extension_traversal_mode(traversal_mode: str, endpoint: str) -> str:
    if endpoint == ENDPOINT_FIRST:
        return reverse_traversal_mode(traversal_mode)
    return traversal_mode


def create_active_extension_job(
    record: PathRecord,
    rng: random.Random,
    traversal_mode: str,
) -> ActiveExtensionJob:
    endpoint_order = [ENDPOINT_LAST]
    if len(record.trace) > 1:
        first_endpoint = rng.choice([ENDPOINT_FIRST, ENDPOINT_LAST])
        second_endpoint = (
            ENDPOINT_LAST if first_endpoint == ENDPOINT_FIRST else ENDPOINT_FIRST
        )
        endpoint_order = [first_endpoint, second_endpoint]
    endpoint = endpoint_order[0]
    return ActiveExtensionJob(
        path_id=record.path_id,
        path_index=record.path_index,
        base_trace=record.trace,
        frames=extension_frames_for_endpoint(record.trace, endpoint),
        locked_frame_count=len(record.trace),
        endpoint_order=endpoint_order,
        traversal_mode=extension_traversal_mode(traversal_mode, endpoint),
    )


def advance_active_extension_endpoint(
    job: ActiveExtensionJob,
    traversal_mode: str,
) -> bool:
    next_index = job.endpoint_attempt_index + 1
    if next_index >= len(job.endpoint_order):
        return False
    job.endpoint_attempt_index = next_index
    endpoint = active_extension_endpoint(job)
    job.frames = extension_frames_for_endpoint(job.base_trace, endpoint)
    job.locked_frame_count = len(job.base_trace)
    job.traversal_mode = extension_traversal_mode(traversal_mode, endpoint)
    return True


def serialize_active_extension_job(job: ActiveExtensionJob) -> dict[str, Any]:
    return {
        "path_id": job.path_id,
        "path_index": job.path_index,
        "base_trace": serialize_trace(job.base_trace),
        "frames": serialize_frames(job.frames),
        "locked_frame_count": job.locked_frame_count,
        "endpoint_order": job.endpoint_order,
        "endpoint_attempt_index": job.endpoint_attempt_index,
        "traversal_mode": job.traversal_mode,
    }


def deserialize_active_extension_job(raw: dict[str, Any]) -> ActiveExtensionJob:
    endpoint_order = [str(value) for value in raw.get("endpoint_order") or []]
    if not endpoint_order or any(
        endpoint not in {ENDPOINT_FIRST, ENDPOINT_LAST} for endpoint in endpoint_order
    ):
        raise ValueError("active extension has invalid endpoint_order")
    frames = deserialize_frames(list(raw.get("frames") or []))
    base_trace = deserialize_trace(list(raw.get("base_trace") or []))
    locked_frame_count = int(raw.get("locked_frame_count") or 0)
    if not base_trace or not frames or locked_frame_count != len(base_trace):
        raise ValueError("active extension has invalid frames or locked boundary")
    endpoint_attempt_index = int(raw.get("endpoint_attempt_index") or 0)
    if endpoint_attempt_index < 0 or endpoint_attempt_index >= len(endpoint_order):
        raise ValueError("active extension has invalid endpoint attempt")
    return ActiveExtensionJob(
        path_id=str(raw.get("path_id") or ""),
        path_index=int(raw.get("path_index") or 0),
        base_trace=base_trace,
        frames=frames,
        locked_frame_count=locked_frame_count,
        endpoint_order=endpoint_order,
        endpoint_attempt_index=endpoint_attempt_index,
        traversal_mode=str(raw.get("traversal_mode") or TRAVERSAL_MODE_MIXED),
    )


def jsonable_random_state(value: Any) -> Any:
    if isinstance(value, tuple):
        return [jsonable_random_state(item) for item in value]
    if isinstance(value, list):
        return [jsonable_random_state(item) for item in value]
    return value


def random_state_from_json(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(random_state_from_json(item) for item in value)
    return value


def serialize_edge(
    edge: tuple[str, str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if edge is None:
        return None
    return {"from": edge[0], "to": edge[1], "attrs": edge[2]}


def deserialize_edge(
    raw: dict[str, Any] | None,
) -> tuple[str, str, dict[str, Any]] | None:
    if raw is None:
        return None
    return str(raw["from"]), str(raw["to"]), dict(raw.get("attrs") or {})


def serialize_frames(frames: list[TravelFrame]) -> list[dict[str, Any]]:
    return [
        {
            "node_id": frame.node_id,
            "incoming_edge": serialize_edge(frame.incoming_edge),
            "tried_nodes": sorted(frame.tried_nodes),
        }
        for frame in frames
    ]


def deserialize_frames(raw_frames: list[dict[str, Any]]) -> list[TravelFrame]:
    return [
        TravelFrame(
            node_id=str(raw["node_id"]),
            incoming_edge=deserialize_edge(raw.get("incoming_edge")),
            tried_nodes=set(raw.get("tried_nodes") or []),
        )
        for raw in raw_frames
    ]


def serialize_trace(
    trace: list[tuple[str, tuple[str, str, dict[str, Any]] | None]],
) -> list[dict[str, Any]]:
    return [{"node_id": node_id, "incoming_edge": serialize_edge(edge)} for node_id, edge in trace]


def deserialize_trace(
    raw_trace: list[dict[str, Any]],
) -> list[tuple[str, tuple[str, str, dict[str, Any]] | None]]:
    return [
        (str(item["node_id"]), deserialize_edge(item.get("incoming_edge"))) for item in raw_trace
    ]


def trace_equals_frames(
    trace: list[tuple[str, tuple[str, str, dict[str, Any]] | None]],
    frames: list[TravelFrame],
) -> bool:
    return serialize_trace(trace) == serialize_trace(trace_from_frames(frames))


def complete_traces_for_target(
    traces: list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]],
    target_num_nodes: int,
    target_num_paths: int | None = None,
) -> list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]]:
    complete = [trace for trace in traces if len(trace) >= target_num_nodes]
    if target_num_paths is not None:
        return complete[:target_num_paths]
    return complete


def complete_path_count(context: BuildContext) -> int:
    return len(
        complete_traces_for_target(
            context.completed_traces,
            context.target_num_nodes,
        )
    )


def generate_path_id() -> str:
    return str(uuid.uuid4())


def ensure_completed_path_ids(context: BuildContext) -> list[str]:
    path_ids = getattr(context, "completed_path_ids", None)
    if path_ids is None:
        path_ids = []
        context.completed_path_ids = path_ids
    while len(path_ids) < len(context.completed_traces):
        path_ids.append(generate_path_id())
    if len(path_ids) > len(context.completed_traces):
        del path_ids[len(context.completed_traces) :]
    return path_ids


def target_path_count(context: BuildContext) -> int:
    return context.target_num_paths or len(context.seed_titles) * context.paths_per_seed


def completed_path_log_label(context: BuildContext) -> str:
    complete_count = complete_path_count(context)
    if context.target_num_paths is None:
        return str(complete_count)
    return f"{complete_count}/{context.target_num_paths}"


def target_paths_reached(context: BuildContext) -> bool:
    return (
        context.target_num_paths is not None
        and complete_path_count(context) >= context.target_num_paths
    )


def trace_key(
    trace: list[tuple[str, tuple[str, str, dict[str, Any]] | None]],
) -> str:
    return orjson.dumps(serialize_trace(trace), option=orjson.OPT_SORT_KEYS).decode("utf-8")


def normalize_loaded_checkpoint_paths(
    completed_traces: list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]],
    resume_frames: list[TravelFrame] | None,
    target_num_nodes: int,
) -> tuple[
    list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]],
    list[list[TravelFrame]],
    list[TravelFrame] | None,
    int,
]:
    complete: list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]] = []
    complete_keys: set[str] = set()
    pending_by_key: dict[str, list[TravelFrame]] = {}
    resume_key = trace_key(trace_from_frames(resume_frames)) if resume_frames else None
    short_count = 0

    for trace in completed_traces:
        if len(trace) >= target_num_nodes:
            key = trace_key(trace)
            if key not in complete_keys:
                complete.append(trace)
                complete_keys.add(key)
            continue
        if not trace:
            continue
        short_count += 1
        key = trace_key(trace)
        if key != resume_key:
            pending_by_key.setdefault(key, frames_from_trace(trace))

    if resume_frames:
        resume_trace = trace_from_frames(resume_frames)
        if len(resume_frames) >= target_num_nodes:
            key = trace_key(resume_trace)
            if key not in complete_keys:
                complete.append(resume_trace)
            resume_frames = None
        elif len(resume_frames) == 0:
            resume_frames = None

    return complete, list(pending_by_key.values()), resume_frames, short_count


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        handle.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2, default=str).decode("utf-8"))
        handle.write("\n")
    temp_path.replace(path)


def save_graph_quiet(
    network: CitationNetwork | SQLiteCitationNetwork,
    output_dir: Path,
    *,
    stem: str,
    write_gexf: bool,
    write_json: bool = True,
) -> Path:
    from src.graph.persistence import save_graph

    logger.disable("src.graph.persistence")
    try:
        return save_graph(
            network,
            output_dir,
            stem=stem,
            write_gexf=write_gexf,
            write_json=write_json,
        )
    finally:
        logger.enable("src.graph.persistence")


def graph_edge_key(source: str, target: str) -> tuple[str, str]:
    return str(source), str(target)


def capture_graph_base(
    network: CitationNetwork | SQLiteCitationNetwork,
) -> tuple[set[str], set[tuple[str, str]]]:
    if uses_sqlite_backend(network):
        return set(), set()
    return (
        {str(node_id) for node_id in network.G.nodes()},
        {graph_edge_key(source, target) for source, target in network.G.edges()},
    )


def record_graph_snapshot_base(
    context: BuildContext,
    graph_json: Path | None,
    complete_count: int | None = None,
) -> None:
    context.last_graph_snapshot_path = graph_json
    (
        context.graph_snapshot_node_ids,
        context.graph_snapshot_edge_keys,
    ) = capture_graph_base(context.network)
    if complete_count is not None:
        context.last_graph_snapshot_complete_count = complete_count


def build_graph_delta_payload(context: BuildContext) -> dict[str, Any]:
    network = getattr(context, "network", None)
    base_graph = getattr(context, "last_graph_snapshot_path", None)
    if network is None or uses_sqlite_backend(network):
        return {
            "base_graph_db_path": str(base_graph) if base_graph else None,
            "nodes": [],
            "edges": [],
        }

    base_nodes = getattr(context, "graph_snapshot_node_ids", set())
    base_edges = getattr(context, "graph_snapshot_edge_keys", set())
    nodes = sorted(str(node_id) for node_id in network.G.nodes() if str(node_id) not in base_nodes)
    edges = [
        {
            "source": str(source),
            "target": str(target),
            "attrs": dict(attrs),
        }
        for source, target, attrs in network.G.edges(data=True)
        if graph_edge_key(source, target) not in base_edges
    ]
    edges.sort(key=lambda edge: (edge["source"], edge["target"]))
    return {
        "base_graph_db_path": str(base_graph) if base_graph else None,
        "nodes": nodes,
        "edges": edges,
    }


def save_graph_snapshot(
    context: BuildContext,
    *,
    write_gexf: bool = False,
    write_json: bool = True,
) -> Path:
    if uses_sqlite_backend(context.network):
        context.last_graph_snapshot_path = context.working_db_path
        context.last_graph_snapshot_complete_count = complete_path_count(context)
        return context.working_db_path
    working_db_path = context.working_db_path
    graph_json = save_graph_quiet(
        context.network,
        working_db_path.parent,
        stem=working_db_path.stem,
        write_gexf=write_gexf,
        write_json=write_json,
    )
    record_graph_snapshot_base(
        context,
        graph_json,
        complete_count=complete_path_count(context),
    )
    return graph_json


def maybe_save_completed_path_graph_snapshot(context: BuildContext) -> Path | None:
    if uses_sqlite_backend(context.network):
        return None
    interval = getattr(
        context,
        "graph_snapshot_every",
        GRAPH_SNAPSHOT_COMPLETED_PATH_INTERVAL,
    )
    if interval <= 0:
        return None

    complete_count = complete_path_count(context)
    last_count = getattr(context, "last_graph_snapshot_complete_count", 0)
    if complete_count <= 0 or complete_count - last_count < interval:
        return None

    graph_json = save_graph_snapshot(context, write_gexf=False, write_json=False)
    logger.info(
        "Saved graph snapshot after completed paths | completed={} | graph={}",
        complete_count,
        graph_json,
    )
    return graph_json


def resolve_final_graph_snapshot(
    context: BuildContext,
    *,
    target_was_reached_before_construction: bool,
) -> tuple[Path | None, bool]:
    if uses_sqlite_backend(getattr(context, "network", None)):
        return context.working_db_path, True
    if target_was_reached_before_construction and target_paths_reached(context):
        graph_json = getattr(context, "last_graph_snapshot_path", None)
        logger.info(
            "Skipping final graph snapshot because target paths were already complete"
            " | graph={}",
            graph_json if graph_json else "none",
        )
        return graph_json, True

    return save_graph_snapshot(context, write_gexf=False), False


def checkpoint_prune_key(path: Path) -> tuple[float, str]:
    try:
        return path.stat().st_mtime, path.name
    except FileNotFoundError:
        return 0.0, path.name


def collect_semantic_checkpoint_stems(context: BuildContext) -> list[str]:
    stems = {
        path.name.removesuffix(".state.json")
        for path in context.checkpoint_dir.glob(f"{context.run_stem}_*.state.json")
        if path.is_file() and not path.name.endswith(".tmp")
    }
    return sorted(
        stems,
        key=lambda stem: checkpoint_prune_key(context.checkpoint_dir / f"{stem}.state.json"),
        reverse=True,
    )


def prune_old_semantic_checkpoints(context: BuildContext) -> None:
    if context.max_checkpoints <= 0:
        return

    stale_stems = collect_semantic_checkpoint_stems(context)[context.max_checkpoints :]
    for stem in stale_stems:
        for suffix in (".state.json", ".paths.json"):
            path = context.checkpoint_dir / f"{stem}{suffix}"
            if path.exists():
                path.unlink()


def semantic_graph_artifact_stem(path: Path) -> str | None:
    for suffix in SEMANTIC_GRAPH_ARTIFACT_SUFFIXES:
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return None


def semantic_graph_version_stem(path: Path) -> str | None:
    artifact_stem = semantic_graph_artifact_stem(path)
    if artifact_stem is None:
        return None
    match = SEMANTIC_GRAPH_VERSION_RE.match(artifact_stem)
    return match.group(1) if match else None


def collect_semantic_graph_versions(
    graph_dir: Path,
) -> list[tuple[str, str, list[Path]]]:
    artifacts_by_version: dict[str, list[Path]] = {}
    for path in graph_dir.iterdir():
        if not path.is_file():
            continue
        version_stem = semantic_graph_version_stem(path)
        if version_stem is None:
            continue
        artifacts_by_version.setdefault(version_stem, []).append(path)

    versions = [
        (
            version_stem.removeprefix("semantic_scholar_paths_"),
            version_stem,
            artifacts,
        )
        for version_stem, artifacts in artifacts_by_version.items()
    ]
    return sorted(versions, key=lambda item: (item[0], item[1]), reverse=True)


def latest_semantic_graph_json(graph_dir: Path) -> Path | None:
    for _timestamp, version_stem, artifacts in collect_semantic_graph_versions(graph_dir):
        final_json = graph_dir / f"{version_stem}.json"
        if final_json.exists():
            return final_json
        json_paths = [path for path in artifacts if path.name.endswith(".json")]
        if json_paths:
            return max(
                json_paths,
                key=lambda path: semantic_graph_artifact_stem(path) or "",
            )
    return None


def prune_old_semantic_graph_versions(
    graph_dir: Path,
    *,
    max_versions: int = SEMANTIC_GRAPH_VERSION_KEEP_COUNT,
    keep_version_stems: set[str] | None = None,
) -> None:
    if max_versions <= 0 or not graph_dir.exists():
        return

    pruned_files: list[Path] = []
    protected_versions = set(keep_version_stems or [])
    versions = collect_semantic_graph_versions(graph_dir)
    stale_versions = versions[max_versions:]
    for _timestamp, version_stem, artifacts in stale_versions:
        if version_stem in protected_versions:
            continue
        for path in artifacts:
            if path.exists():
                path.unlink()
                pruned_files.append(path)

    kept_versions = versions[:max_versions]
    for _timestamp, version_stem, artifacts in kept_versions:
        has_root_artifact = any(
            semantic_graph_artifact_stem(path) == version_stem for path in artifacts
        )
        if not has_root_artifact:
            continue
        for path in artifacts:
            artifact_stem = semantic_graph_artifact_stem(path)
            if artifact_stem == version_stem:
                continue
            if path.exists():
                path.unlink()
                pruned_files.append(path)

    if pruned_files:
        logger.info(
            "Pruned old Semantic Scholar graph versions | files={}",
            len(pruned_files),
        )


def authorship_edge_db_ids(
    source_node_id: str,
    target_node_id: str,
) -> tuple[str, str] | None:
    """Return ``(paper_id, author_id)`` for an authorship edge direction."""
    source_type = nid_type(source_node_id)
    target_type = nid_type(target_node_id)
    if source_type == NODE_PAPER and target_type == NODE_AUTHOR:
        return nid_db_id(source_node_id), nid_db_id(target_node_id)
    if source_type == NODE_AUTHOR and target_type == NODE_PAPER:
        return nid_db_id(target_node_id), nid_db_id(source_node_id)
    return None


def canonical_authorship_position(
    db: DBReader,
    source_node_id: str,
    target_node_id: str,
) -> int | None:
    """Return the DB-backed authorship position for an edge, if present."""
    ids = authorship_edge_db_ids(source_node_id, target_node_id)
    conn = getattr(db, "conn", None)
    if ids is None or conn is None:
        return None
    paper_id, author_id = ids
    if hasattr(db, "resolve_paper_id"):
        paper_id = db.resolve_paper_id(paper_id)
    if hasattr(db, "resolve_author_id"):
        author_id = db.resolve_author_id(author_id)
    row = conn.execute(
        "SELECT position FROM authorship WHERE paper_id = ? AND author_id = ?",
        (paper_id, author_id),
    ).fetchone()
    if row is None:
        return None
    return int(row["position"])


def refresh_trace_authorship_positions(
    trace: list[tuple[str, tuple[str, str, dict[str, Any]] | None]],
    db: DBReader,
) -> list[tuple[str, tuple[str, str, dict[str, Any]] | None]]:
    """Refresh authorship edge positions from the current DB before serialization."""
    refreshed_trace: list[tuple[str, tuple[str, str, dict[str, Any]] | None]] = []
    for node_id, incoming_edge in trace:
        if incoming_edge is None:
            refreshed_trace.append((node_id, None))
            continue

        source_node_id, target_node_id, edge_attrs = incoming_edge
        refreshed_attrs = dict(edge_attrs)
        if refreshed_attrs.get("edge_type") == EDGE_AUTHORSHIP:
            position = canonical_authorship_position(
                db,
                source_node_id,
                target_node_id,
            )
            if position is not None:
                refreshed_attrs["position"] = position
        refreshed_trace.append((node_id, (source_node_id, target_node_id, refreshed_attrs)))
    return refreshed_trace


def build_path_entry(
    db: DBReader,
    trace: list[tuple[str, tuple[str, str, dict[str, Any]] | None]],
    path_id: str,
    index: int,
) -> dict[str, Any]:
    refreshed_trace = refresh_trace_authorship_positions(trace, db)
    steps = build_steps_from_trace(refreshed_trace, None, db)
    return {
        "path_id": path_id,
        "path_index": index,
        "start_node": refreshed_trace[0][0],
        "moves": max(len(steps) - 1, 0),
        "path": steps,
    }


def build_result_payload(
    graph_json: Path | None,
    traces: list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]],
    context: BuildContext,
    *,
    db_path: Path | None = None,
) -> dict[str, Any]:
    effective_db_path = db_path or (
        graph_json
        if graph_json is not None and graph_json.suffix.lower() == ".db"
        else (graph_json.with_suffix(".db") if graph_json is not None else context.working_db_path)
    )
    db = DBReader(
        effective_db_path,
        require_uuid_ids=not uses_sqlite_backend(getattr(context, "network", None)),
    )
    cache = getattr(context, "result_path_entry_cache", None)
    if cache is None:
        cache = {}
        context.result_path_entry_cache = cache
    try:
        paths = []
        complete_traces = complete_traces_for_target(
            traces,
            context.target_num_nodes,
        )
        path_ids = ensure_completed_path_ids(context)
        live_keys: set[str] = set()
        for index, trace in enumerate(complete_traces):
            if not trace:
                continue
            path_id = path_ids[index] if index < len(path_ids) else generate_path_id()
            key = trace_key(trace)
            live_keys.add(key)
            entry = cache.get(key)
            if entry is None:
                entry = build_path_entry(db, trace, path_id, index)
                cache[key] = entry
            else:
                # Reuse the DB-reconstructed steps; only the position/id can shift.
                entry["path_id"] = path_id
                entry["path_index"] = index
            paths.append(entry)
        for stale_key in set(cache) - live_keys:
            del cache[stale_key]
    finally:
        db.close()

    metadata: dict[str, Any] = {
        "graph_db_path": str(effective_db_path),
        "db_path": str(effective_db_path),
        "working_db_path": str(context.working_db_path),
        "seed_titles": context.seed_titles,
        "paths_per_seed": context.paths_per_seed,
        "target_num_paths": context.target_num_paths,
        "path_length": context.target_num_nodes,
        "requested_n": target_path_count(context),
        "n": len(paths),
        "complete_paths": len(paths),
        "sampling": "semantic_scholar_uniform_with_backtracking",
        "sampling_version": (
            2
            if uses_sqlite_backend(getattr(context, "network", None))
            else 1
        ),
        "graph_backend": (
            GRAPH_BACKEND_SQLITE
            if uses_sqlite_backend(getattr(context, "network", None))
            else GRAPH_BACKEND_NETWORKX
        ),
        "traversal_mode": getattr(
            context,
            "traversal_mode",
            TRAVERSAL_MODE_MIXED,
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if context.start_date is not None:
        metadata["start_date"] = context.start_date.isoformat()
    if context.end_date is not None:
        metadata["end_date"] = context.end_date.isoformat()

    return {
        "metadata": metadata,
        PARAMETERS_FIELD: merge_parameters(
            None,
            **{
                NUM_HOPS_PARAMETER_FIELD: max(context.target_num_nodes - 1, 0),
                NEEDLE_TYPE_PARAMETER_FIELD: None,
                DISTRACTING_CONTENT_TYPE_PARAMETER_FIELD: None,
                NEEDLE_ORDER_PARAMETER_FIELD: None,
                TARGET_CONTEXT_LENGTH_PARAMETER_FIELD: None,
            },
        ),
        "paths": paths,
    }


def read_output_payload(output_path: Path) -> dict[str, Any]:
    try:
        payload = orjson.loads(output_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"Could not read completed paths from {output_path}") from exc
    except orjson.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse completed paths from {output_path}") from exc

    if not isinstance(payload, dict):
        raise SystemExit(f"Expected a JSON object in {output_path}")
    return payload


def legacy_checkpoint_output_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / DEFAULT_PATHS_JSON_NAME


def output_path_sources(output_path: Path, checkpoint_dir: Path) -> list[Path]:
    canonical_output_path = output_path.resolve()
    existing_paths: dict[str, Path] = {}
    for path in (output_path, legacy_checkpoint_output_path(checkpoint_dir)):
        if not path.is_file():
            continue
        existing_paths.setdefault(str(path.resolve()), path)
    return sorted(
        existing_paths.values(),
        key=lambda path: (
            checkpoint_prune_key(path)[0],
            path.resolve() == canonical_output_path,
            str(path),
        ),
        reverse=True,
    )


def deserialize_output_edge(
    raw_edge: dict[str, Any] | None,
    from_node_id: str,
    to_node_id: str,
) -> tuple[str, str, dict[str, Any]]:
    if not isinstance(raw_edge, dict):
        return from_node_id, to_node_id, {"edge_type": "unknown"}

    attrs: dict[str, Any] = {"edge_type": str(raw_edge.get("edge_type") or "unknown")}
    if "position" in raw_edge:
        attrs["position"] = raw_edge["position"]
    return (
        str(raw_edge.get("from") or from_node_id),
        str(raw_edge.get("to") or to_node_id),
        attrs,
    )


def deserialize_trace_from_output_path(
    path_entry: dict[str, Any],
) -> tuple[list[tuple[str, tuple[str, str, dict[str, Any]] | None]], str | None]:
    raw_steps = path_entry.get("path")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("expected a non-empty path list")

    trace: list[tuple[str, tuple[str, str, dict[str, Any]] | None]] = []
    previous_node_id: str | None = None
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            raise ValueError("expected each path step to be an object")
        raw_node = raw_step.get("node")
        if not isinstance(raw_node, dict):
            raise ValueError("expected each path step to include a node object")
        node_id = str(raw_node.get("node_id") or "").strip()
        if not node_id:
            raise ValueError("expected each path step to include node.node_id")

        incoming_edge = None
        if index > 0:
            assert previous_node_id is not None
            incoming_edge = deserialize_output_edge(
                raw_step.get("edge_taken"),
                previous_node_id,
                node_id,
            )
        trace.append((node_id, incoming_edge))
        previous_node_id = node_id

    path_id = str(path_entry.get("path_id") or "").strip() or None
    return trace, path_id


def path_record_from_output_entry(
    path_entry: dict[str, Any],
    list_index: int,
    target_num_nodes: int,
) -> PathRecord:
    trace, loaded_path_id = deserialize_trace_from_output_path(path_entry)
    raw_path_index = path_entry.get("path_index", list_index)
    try:
        path_index = int(raw_path_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected path_index to be an integer") from exc
    return PathRecord(
        path_id=loaded_path_id or generate_path_id(),
        path_index=path_index,
        trace=trace,
        status=(
            PATH_STATUS_COMPLETE
            if len(trace) >= target_num_nodes
            else PATH_STATUS_INCOMPLETE
        ),
        source_path_id=(str(path_entry.get("source_path_id") or "").strip() or None),
        extended_from_endpoint=(
            str(path_entry.get("extended_from_endpoint") or "").strip() or None
        ),
        original_node_count=(
            int(path_entry["original_node_count"])
            if path_entry.get("original_node_count") is not None
            else None
        ),
        seed_title=str(path_entry.get("seed_title") or "").strip() or None,
    )


def load_path_records_from_payload(
    payload: dict[str, Any],
    target_num_nodes: int,
    *,
    source_label: str,
) -> list[PathRecord]:
    raw_paths = payload.get("paths")
    if not isinstance(raw_paths, list):
        raise SystemExit(f"Expected a 'paths' list in {source_label}")

    records: list[PathRecord] = []
    seen_path_ids: set[str] = set()
    for list_index, raw_path in enumerate(raw_paths):
        if not isinstance(raw_path, dict):
            raise SystemExit(
                f"Expected 'paths[{list_index}]' to be an object in {source_label}"
            )
        try:
            record = path_record_from_output_entry(
                raw_path,
                list_index,
                target_num_nodes,
            )
        except ValueError as exc:
            raise SystemExit(
                f"Could not parse paths[{list_index}] in {source_label}: {exc}"
            ) from exc
        if record.path_id in seen_path_ids:
            raise SystemExit(f"Duplicate path_id {record.path_id!r} in {source_label}")
        seen_path_ids.add(record.path_id)
        records.append(record)
    return records


def load_path_records(
    output_path: Path,
    target_num_nodes: int,
) -> tuple[dict[str, Any], list[PathRecord]]:
    payload = read_output_payload(output_path)
    return payload, load_path_records_from_payload(
        payload,
        target_num_nodes,
        source_label=str(output_path),
    )


def complete_record_count(records: list[PathRecord], target_num_nodes: int) -> int:
    return sum(len(record.trace) >= target_num_nodes for record in records)


def next_record_path_index(records: list[PathRecord]) -> int:
    if not records:
        return 0
    return max(record.path_index for record in records) + 1


def find_path_record(records: list[PathRecord], path_id: str) -> PathRecord | None:
    return next((record for record in records if record.path_id == path_id), None)


def trace_node_ids(
    trace: list[tuple[str, tuple[str, str, dict[str, Any]] | None]],
) -> list[str]:
    return [node_id for node_id, _edge in trace]


def append_source_path_record(
    destination_records: list[PathRecord],
    source_record: PathRecord,
    target_num_nodes: int,
) -> PathRecord:
    record = PathRecord(
        path_id=source_record.path_id,
        path_index=next_record_path_index(destination_records),
        trace=source_record.trace,
        status=(
            PATH_STATUS_COMPLETE
            if len(source_record.trace) >= target_num_nodes
            else PATH_STATUS_INCOMPLETE
        ),
        source_path_id=source_record.path_id,
        original_node_count=len(source_record.trace),
    )
    destination_records.append(record)
    return record


def refresh_record_statuses(
    records: list[PathRecord],
    target_num_nodes: int,
) -> None:
    for record in records:
        if len(record.trace) >= target_num_nodes:
            record.status = PATH_STATUS_COMPLETE
        elif record.status != PATH_STATUS_ACTIVE:
            record.status = PATH_STATUS_INCOMPLETE


def sync_context_completed_records(
    context: BuildContext,
    records: list[PathRecord],
) -> None:
    complete_records = [
        record for record in records if len(record.trace) >= context.target_num_nodes
    ]
    context.completed_traces = [record.trace for record in complete_records]
    context.completed_path_ids = [record.path_id for record in complete_records]


def build_persistent_result_payload(
    context: BuildContext,
    records: list[PathRecord],
    previous_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    refresh_record_statuses(records, context.target_num_nodes)
    db = DBReader(
        context.working_db_path,
        require_uuid_ids=not uses_sqlite_backend(context.network),
    )
    try:
        paths: list[dict[str, Any]] = []
        for record in records:
            if not record.trace:
                continue
            entry = build_path_entry(
                db,
                record.trace,
                record.path_id,
                record.path_index,
            )
            entry["status"] = record.status
            if record.source_path_id:
                entry["source_path_id"] = record.source_path_id
            if record.extended_from_endpoint:
                entry["extended_from_endpoint"] = record.extended_from_endpoint
            if record.original_node_count is not None:
                entry["original_node_count"] = record.original_node_count
            if record.seed_title:
                entry["seed_title"] = record.seed_title
            paths.append(entry)
    finally:
        db.close()

    previous_metadata = (previous_payload or {}).get("metadata")
    metadata = dict(previous_metadata) if isinstance(previous_metadata, dict) else {}
    complete_count = complete_record_count(records, context.target_num_nodes)
    incomplete_count = len(records) - complete_count
    metadata.update(
        {
            "graph_db_path": str(context.working_db_path),
            "db_path": str(context.working_db_path),
            "working_db_path": str(context.working_db_path),
            "seed_titles": context.seed_titles,
            "paths_per_seed": context.paths_per_seed,
            "target_num_paths": context.target_num_paths,
            "path_length": context.target_num_nodes,
            "requested_n": context.target_num_paths,
            "n": len(records),
            "complete_paths": complete_count,
            "stored_paths": len(records),
            "incomplete_paths": incomplete_count,
            "sampling": "semantic_scholar_endpoint_extension_with_backtracking",
            "sampling_version": 3,
            "path_store_version": PERSISTENT_PATH_STORE_VERSION,
            "graph_backend": (
                GRAPH_BACKEND_SQLITE
                if uses_sqlite_backend(context.network)
                else GRAPH_BACKEND_NETWORKX
            ),
            "traversal_mode": context.traversal_mode,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    if context.start_date is not None:
        metadata["start_date"] = context.start_date.isoformat()
    else:
        metadata.pop("start_date", None)
    if context.end_date is not None:
        metadata["end_date"] = context.end_date.isoformat()
    else:
        metadata.pop("end_date", None)

    return {
        "metadata": metadata,
        PARAMETERS_FIELD: merge_parameters(
            (previous_payload or {}).get(PARAMETERS_FIELD),
            **{
                NUM_HOPS_PARAMETER_FIELD: max(context.target_num_nodes - 1, 0),
                NEEDLE_TYPE_PARAMETER_FIELD: None,
                DISTRACTING_CONTENT_TYPE_PARAMETER_FIELD: None,
                NEEDLE_ORDER_PARAMETER_FIELD: None,
                TARGET_CONTEXT_LENGTH_PARAMETER_FIELD: None,
            },
        ),
        "paths": paths,
    }


def persistent_checkpoint_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "persistent_paths.state.json"


def build_persistent_state_payload(
    context: BuildContext,
    active_job: ActiveExtensionJob | None,
    traversal_rng: random.Random,
) -> dict[str, Any]:
    return {
        "version": PERSISTENT_STATE_VERSION,
        "run_mode": RUN_MODE_PERSISTENT,
        "output_path": str(context.output_path),
        "working_db_path": str(context.working_db_path),
        "graph_db_path": str(context.working_db_path),
        "graph_backend": (
            GRAPH_BACKEND_SQLITE
            if uses_sqlite_backend(context.network)
            else GRAPH_BACKEND_NETWORKX
        ),
        "target_num_nodes": context.target_num_nodes,
        "traversal_rng_state": jsonable_random_state(traversal_rng.getstate()),
        "active_job": (
            serialize_active_extension_job(active_job) if active_job else None
        ),
    }


def load_persistent_checkpoint(
    state_path: Path,
    output_path: Path,
) -> tuple[ActiveExtensionJob | None, Any | None]:
    if not state_path.is_file():
        return None, None
    try:
        state = orjson.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, orjson.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read persistent checkpoint {state_path}") from exc
    if not isinstance(state, dict) or state.get("run_mode") != RUN_MODE_PERSISTENT:
        raise SystemExit(f"Not a persistent-path checkpoint: {state_path}")
    if int(state.get("version", 0)) != PERSISTENT_STATE_VERSION:
        raise SystemExit(f"Unsupported checkpoint version in {state_path}")
    saved_output = str(state.get("output_path") or "").strip()
    if saved_output and Path(saved_output).resolve() != output_path.resolve():
        raise SystemExit(
            f"Checkpoint output {saved_output} does not match {output_path}"
        )
    raw_active_job = state.get("active_job")
    active_job = None
    if raw_active_job is not None:
        if not isinstance(raw_active_job, dict):
            raise SystemExit(f"Invalid active job in {state_path}")
        try:
            active_job = deserialize_active_extension_job(raw_active_job)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid active job in {state_path}: {exc}") from exc
    return active_job, state.get("traversal_rng_state")


def save_persistent_progress(
    context: BuildContext,
    records: list[PathRecord],
    previous_payload: dict[str, Any] | None,
    active_job: ActiveExtensionJob | None,
    traversal_rng: random.Random,
    state_path: Path,
) -> None:
    if active_job is not None:
        record = find_path_record(records, active_job.path_id)
        if record is None:
            raise RuntimeError(
                f"Active path {active_job.path_id} is missing from destination output"
            )
        record.trace = active_extension_canonical_trace(active_job)
        record.status = (
            PATH_STATUS_COMPLETE
            if len(record.trace) >= context.target_num_nodes
            else PATH_STATUS_ACTIVE
        )
    refresh_record_statuses(records, context.target_num_nodes)
    sync_context_completed_records(context, records)
    context.network.db.flush()
    write_json(
        context.output_path,
        build_persistent_result_payload(context, records, previous_payload),
    )
    write_json(
        state_path,
        build_persistent_state_payload(context, active_job, traversal_rng),
    )


def persistent_output_payload(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    metadata = payload.get("metadata")
    return bool(
        isinstance(metadata, dict)
        and metadata.get("path_store_version") == PERSISTENT_PATH_STORE_VERSION
    )


def trace_contains_trace(
    longer_trace: list[tuple[str, tuple[str, str, dict[str, Any]] | None]],
    shorter_trace: list[tuple[str, tuple[str, str, dict[str, Any]] | None]],
) -> bool:
    longer_nodes = trace_node_ids(longer_trace)
    shorter_nodes = trace_node_ids(shorter_trace)
    if len(shorter_nodes) > len(longer_nodes):
        return False
    width = len(shorter_nodes)
    return any(
        longer_nodes[index : index + width] == shorter_nodes
        for index in range(len(longer_nodes) - width + 1)
    )


def reconcile_source_records(
    destination_records: list[PathRecord],
    source_records: list[PathRecord],
    target_num_nodes: int,
    *,
    active_path_id: str | None = None,
) -> list[PathRecord]:
    destination_by_id = {record.path_id: record for record in destination_records}
    for source_record in source_records:
        destination_record = destination_by_id.get(source_record.path_id)
        if destination_record is None or destination_record.path_id == active_path_id:
            continue
        if len(source_record.trace) <= len(destination_record.trace):
            continue
        if not trace_contains_trace(source_record.trace, destination_record.trace):
            logger.warning(
                "Keeping destination path because source path_id={} has a "
                "different path",
                source_record.path_id,
            )
            continue
        destination_record.trace = source_record.trace
        destination_record.status = (
            PATH_STATUS_COMPLETE
            if len(source_record.trace) >= target_num_nodes
            else PATH_STATUS_INCOMPLETE
        )
        if destination_record.original_node_count is None:
            destination_record.original_node_count = len(source_record.trace)
        destination_record.source_path_id = source_record.path_id

    new_records: list[PathRecord] = []
    destination_trace_keys = {trace_key(record.trace) for record in destination_records}
    queued_trace_keys: set[str] = set()
    for source_record in source_records:
        if source_record.path_id in destination_by_id:
            continue
        source_key = trace_key(source_record.trace)
        if source_key in destination_trace_keys or source_key in queued_trace_keys:
            continue
        new_records.append(source_record)
        queued_trace_keys.add(source_key)
    return new_records


def run_active_extension_job(
    context: BuildContext,
    records: list[PathRecord],
    previous_payload: dict[str, Any] | None,
    job: ActiveExtensionJob,
    traversal_rng: random.Random,
    state_path: Path,
) -> str:
    record = find_path_record(records, job.path_id)
    if record is None:
        raise RuntimeError(f"Active path {job.path_id} is missing")

    context.rng = traversal_rng
    if record.original_node_count is None:
        record.original_node_count = len(job.base_trace)
    record.status = PATH_STATUS_ACTIVE
    save_persistent_progress(
        context,
        records,
        previous_payload,
        job,
        traversal_rng,
        state_path,
    )

    progress_bars_disabled = disable_progress_bars(context.no_tqdm_bar)
    hop_bar = tqdm(
        total=context.target_num_nodes,
        initial=min(len(job.frames), context.target_num_nodes),
        desc="Current path nodes",
        unit="node",
        leave=False,
        disable=progress_bars_disabled,
    )

    def save_hop_checkpoint(_frames: list[TravelFrame], moves: int) -> None:
        save_persistent_progress(
            context,
            records,
            previous_payload,
            job,
            traversal_rng,
            state_path,
        )
        logger.info(
            "Saved extension checkpoint | path_id={} | endpoint={} | added_nodes={}",
            job.path_id,
            active_extension_endpoint(job),
            moves,
        )

    try:
        while True:
            reset_hop_bar(hop_bar, len(job.frames), context.target_num_nodes)
            frames, status = construct_path(
                context,
                job.frames,
                hop_bar,
                minimum_frame_count=job.locked_frame_count,
                traversal_mode=job.traversal_mode,
                checkpoint_callback=save_hop_checkpoint,
            )
            job.frames = frames
            if status == PATH_COMPLETE:
                record.trace = active_extension_canonical_trace(job)
                record.status = PATH_STATUS_COMPLETE
                record.extended_from_endpoint = active_extension_endpoint(job)
                sync_context_completed_records(context, records)
                maybe_save_completed_path_graph_snapshot(context)
                save_persistent_progress(
                    context,
                    records,
                    previous_payload,
                    None,
                    traversal_rng,
                    state_path,
                )
                return PATH_COMPLETE
            if status == PATH_DEFERRED:
                save_persistent_progress(
                    context,
                    records,
                    previous_payload,
                    job,
                    traversal_rng,
                    state_path,
                )
                return PATH_DEFERRED
            if advance_active_extension_endpoint(job, context.traversal_mode):
                logger.info(
                    "First endpoint exhausted; trying other endpoint | path_id={} | endpoint={}",
                    job.path_id,
                    active_extension_endpoint(job),
                )
                save_persistent_progress(
                    context,
                    records,
                    previous_payload,
                    job,
                    traversal_rng,
                    state_path,
                )
                continue

            record.trace = job.base_trace
            record.status = PATH_STATUS_INCOMPLETE
            save_persistent_progress(
                context,
                records,
                previous_payload,
                None,
                traversal_rng,
                state_path,
            )
            return PATH_EXHAUSTED
    except KeyboardInterrupt:
        save_persistent_progress(
            context,
            records,
            previous_payload,
            job,
            traversal_rng,
            state_path,
        )
        raise
    finally:
        hop_bar.close()


def load_paths_from_output(
    output_path: Path,
) -> tuple[
    list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]],
    list[str],
]:
    payload = read_output_payload(output_path)
    raw_paths = payload.get("paths")
    if not isinstance(raw_paths, list):
        raise SystemExit(f"Expected a 'paths' list in {output_path}")

    traces: list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]] = []
    path_ids: list[str] = []
    for index, raw_path in enumerate(raw_paths):
        if not isinstance(raw_path, dict):
            raise SystemExit(f"Expected 'paths[{index}]' to be an object in {output_path}")
        try:
            trace, path_id = deserialize_trace_from_output_path(raw_path)
        except ValueError as exc:
            raise SystemExit(f"Could not parse paths[{index}] in {output_path}: {exc}") from exc
        traces.append(trace)
        path_ids.append(path_id or generate_path_id())
    return traces, path_ids


def load_completed_paths_from_output(
    output_path: Path,
    target_num_nodes: int | None = None,
    target_num_paths: int | None = None,
) -> tuple[
    list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]],
    list[str],
]:
    all_traces, all_path_ids = load_paths_from_output(output_path)
    traces: list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]] = []
    path_ids: list[str] = []
    for index, trace in enumerate(all_traces):
        if target_num_nodes is not None and len(trace) < target_num_nodes:
            continue
        traces.append(trace)
        path_ids.append(all_path_ids[index] if index < len(all_path_ids) else generate_path_id())
        if target_num_paths is not None and len(traces) >= target_num_paths:
            break
    return traces, path_ids


def split_output_paths(
    traces: list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]],
    path_ids: list[str],
    target_num_nodes: int,
) -> tuple[
    list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]],
    list[str],
    list[list[TravelFrame]],
    int,
    int,
]:
    complete_traces: list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]] = []
    complete_path_ids: list[str] = []
    pending_frames: list[list[TravelFrame]] = []
    complete_keys: set[str] = set()
    pending_keys: set[str] = set()
    complete_count = 0
    incomplete_count = 0

    for index, trace in enumerate(traces):
        if not trace:
            continue
        key = trace_key(trace)
        path_id = path_ids[index] if index < len(path_ids) else generate_path_id()
        if len(trace) >= target_num_nodes:
            complete_count += 1
            if key in complete_keys:
                continue
            complete_keys.add(key)
            complete_traces.append(trace)
            complete_path_ids.append(path_id)
            continue
        incomplete_count += 1
        if key in pending_keys:
            continue
        pending_keys.add(key)
        pending_frames.append(frames_from_trace(trace))

    return (
        complete_traces,
        complete_path_ids,
        pending_frames,
        complete_count,
        incomplete_count,
    )


def load_paths_from_outputs(
    output_paths: list[Path],
) -> tuple[
    list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]],
    list[str],
]:
    merged_traces: list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]] = []
    merged_path_ids: list[str] = []
    for output_path in output_paths:
        traces, path_ids = load_paths_from_output(output_path)
        merged_traces, merged_path_ids = merge_completed_paths(
            merged_traces,
            merged_path_ids,
            traces,
            path_ids,
        )
    return merged_traces, merged_path_ids


def merge_completed_paths(
    preferred_traces: list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]],
    preferred_path_ids: list[str],
    fallback_traces: list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]],
    fallback_path_ids: list[str],
    target_num_paths: int | None = None,
) -> tuple[
    list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]],
    list[str],
]:
    merged_traces: list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]] = []
    merged_path_ids: list[str] = []
    seen_keys: set[str] = set()

    def append_traces(
        traces: list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]],
        path_ids: list[str],
    ) -> None:
        for index, trace in enumerate(traces):
            key = trace_key(trace)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged_traces.append(trace)
            path_id = str(path_ids[index]).strip() if index < len(path_ids) else ""
            merged_path_ids.append(path_id or generate_path_id())
            if target_num_paths is not None and len(merged_traces) >= target_num_paths:
                return

    append_traces(preferred_traces, preferred_path_ids)
    if target_num_paths is None or len(merged_traces) < target_num_paths:
        append_traces(fallback_traces, fallback_path_ids)
    return merged_traces, merged_path_ids


def merge_pending_path_frames(
    preferred_frames: list[list[TravelFrame]],
    fallback_frames: list[list[TravelFrame]],
    resume_frames: list[TravelFrame] | None = None,
    target_num_nodes: int | None = None,
) -> list[list[TravelFrame]]:
    merged_frames: list[list[TravelFrame]] = []
    seen_keys: set[str] = set()

    if resume_frames:
        seen_keys.add(trace_key(trace_from_frames(resume_frames)))

    def append_frames(frames_list: list[list[TravelFrame]]) -> None:
        for frames in frames_list:
            if not frames:
                continue
            if target_num_nodes is not None and len(frames) >= target_num_nodes:
                continue
            key = trace_key(trace_from_frames(frames))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged_frames.append(frames)

    append_frames(preferred_frames)
    append_frames(fallback_frames)
    return merged_frames


def build_state_payload(
    context: BuildContext,
    frames: list[TravelFrame],
    graph_json: Path | None = None,
) -> dict[str, Any]:
    snapshot_path = graph_json
    if snapshot_path is None:
        snapshot_path = getattr(context, "last_graph_snapshot_path", None)
    return {
        "version": STATE_VERSION,
        "run_stem": context.run_stem,
        "output_path": str(context.output_path),
        "graph_db_path": str(snapshot_path) if snapshot_path is not None else None,
        "db_path": str(snapshot_path) if snapshot_path is not None else None,
        "working_db_path": str(context.working_db_path),
        "graph_backend": (
            GRAPH_BACKEND_SQLITE
            if uses_sqlite_backend(getattr(context, "network", None))
            else GRAPH_BACKEND_NETWORKX
        ),
        "last_graph_snapshot_complete_count": getattr(
            context,
            "last_graph_snapshot_complete_count",
            0,
        ),
        "graph_delta": build_graph_delta_payload(context),
        "seed_titles": context.seed_titles,
        "paths_per_seed": context.paths_per_seed,
        "target_num_paths": context.target_num_paths,
        "target_num_nodes": context.target_num_nodes,
        "traversal_mode": getattr(
            context,
            "traversal_mode",
            TRAVERSAL_MODE_MIXED,
        ),
        "start_date": context.start_date.isoformat() if context.start_date else None,
        "end_date": context.end_date.isoformat() if context.end_date else None,
        "current_seed_index": context.current_seed_index,
        "current_repeat_index": context.current_repeat_index,
        "expanded_nodes": (
            []
            if uses_sqlite_backend(getattr(context, "network", None))
            else sorted(context.expanded_nodes)
        ),
        "rng_state": jsonable_random_state(context.rng.getstate()),
        # [deepest path length, seed count] pairs; JSON keys cannot be ints.
        "failed_seed_max_nodes": [
            [nodes, count]
            for nodes, count in sorted(getattr(context, "failed_seed_max_nodes", {}).items())
        ],
        "frames": serialize_frames(frames),
        "in_progress_trace": serialize_trace(trace_from_frames(frames)),
        "completed_traces": [serialize_trace(trace) for trace in context.completed_traces],
        "completed_path_ids": ensure_completed_path_ids(context),
    }


def ensure_paper_venues_for_traces(
    context: BuildContext,
    traces: list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]],
) -> None:
    if uses_bulk_client(context):
        ensure_paper_venues_in_bulk(context, traces)
        return
    for trace in traces:
        for node_id, _edge in trace:
            if nid_type(node_id) != NODE_PAPER or node_id in context.venue_checked_nodes:
                continue
            row = context.network.db.get_paper(nid_db_id(node_id))
            if row is None or row["venue"]:
                context.venue_checked_nodes.add(node_id)
                continue
            semantic_scholar_id = clean_semantic_scholar_id(
                row["semantic_scholar_id"],
                node_id=node_id,
                entity_label="Paper",
            )
            if not semantic_scholar_id:
                context.venue_checked_nodes.add(node_id)
                continue
            try:
                paper = context.client.get_paper_by_id(semantic_scholar_id)
            except Exception as exc:
                logger.debug("Could not refresh venue for {}: {}", node_id, exc)
                continue
            venue = str((paper or {}).get("venue") or "")
            if venue:
                context.network.db.update_paper(nid_db_id(node_id), venue=venue)
            context.venue_checked_nodes.add(node_id)


def ensure_paper_venues_in_bulk(
    context: BuildContext,
    traces: list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]],
) -> None:
    """Bulk version of ``ensure_paper_venues_for_traces``: one lookup for all papers.

    A paper the index cannot find is marked checked too -- the index does not
    change during a run, so asking again would give the same answer.
    """
    semantic_scholar_id_by_node: dict[str, str] = {}
    for trace in traces:
        for node_id, _edge in trace:
            if (
                nid_type(node_id) != NODE_PAPER
                or node_id in context.venue_checked_nodes
                or node_id in semantic_scholar_id_by_node
            ):
                continue
            row = context.network.db.get_paper(nid_db_id(node_id))
            if row is None or row["venue"]:
                context.venue_checked_nodes.add(node_id)
                continue
            semantic_scholar_id = clean_semantic_scholar_id(
                row["semantic_scholar_id"],
                node_id=node_id,
                entity_label="Paper",
            )
            if not semantic_scholar_id:
                context.venue_checked_nodes.add(node_id)
                continue
            semantic_scholar_id_by_node[node_id] = semantic_scholar_id
    if not semantic_scholar_id_by_node:
        return

    try:
        papers = context.client.get_papers_batch(
            list(semantic_scholar_id_by_node.values()),
            fields=["venue"],
        )
    except Exception as exc:
        logger.debug(
            "Could not refresh venues for {} paper(s): {}",
            len(semantic_scholar_id_by_node),
            exc,
        )
        return
    for node_id, paper in zip(semantic_scholar_id_by_node, papers):
        venue = str((paper or {}).get("venue") or "")
        if venue:
            context.network.db.update_paper(nid_db_id(node_id), venue=venue)
        context.venue_checked_nodes.add(node_id)


def save_checkpoint(
    context: BuildContext,
    frames: list[TravelFrame],
    *,
    label: str,
    include_current_trace: bool = True,
) -> Path:
    context.checkpoint_count += 1
    traces = list(context.completed_traces)
    if (
        include_current_trace
        and frames
        and (not traces or not trace_equals_frames(traces[-1], frames))
    ):
        traces.append(trace_from_frames(frames))
    ensure_paper_venues_for_traces(context, traces)
    if label == "path_done":
        maybe_save_completed_path_graph_snapshot(context)

    graph_json = getattr(context, "last_graph_snapshot_path", None)
    state_path = context.checkpoint_dir / f"{context.run_stem}.state.json"
    write_json(state_path, build_state_payload(context, frames, graph_json))
    if label in {"path_done", "path_exhausted", "path_timed_out"}:
        write_json(
            context.output_path,
            build_result_payload(
                graph_json,
                context.completed_traces,
                context,
                db_path=context.working_db_path,
            ),
        )
        remove_legacy_checkpoint_output(context.checkpoint_dir, context.output_path)
    logger.info(
        "Saved lightweight checkpoint | {} | label={} | nodes={} | output={}",
        state_path.name,
        label,
        len(frames),
        context.output_path.name
        if label in {"path_done", "path_exhausted", "path_timed_out"}
        else "checkpoint-only",
    )
    prune_old_semantic_checkpoints(context)
    return state_path


def save_interrupted_run_state(
    context: BuildContext,
    frames: list[TravelFrame],
) -> Path | None:
    try:
        context.network.db.flush()
    except Exception as exc:
        logger.warning("Could not flush working database during shutdown: {}", exc)

    state_path = context.checkpoint_dir / f"{context.run_stem}.state.json"
    try:
        write_json(
            state_path,
            build_state_payload(context, frames),
        )
        try:
            write_json(
                context.output_path,
                build_result_payload(
                    context.working_db_path,
                    context.completed_traces,
                    context,
                    db_path=context.working_db_path,
                ),
            )
            remove_legacy_checkpoint_output(context.checkpoint_dir, context.output_path)
        except Exception as exc:
            logger.warning(
                "Could not refresh sampled paths output during shutdown: {}",
                exc,
            )
        logger.warning(
            "Saved shutdown checkpoint | {} | nodes={} | completed={}",
            state_path.name,
            len(frames),
            complete_path_count(context),
        )
        context.shutdown_checkpoint_path = state_path
        return state_path
    except Exception as exc:
        logger.error("Could not save shutdown checkpoint: {}", exc)
        return None


def set_bar_position(bar: tqdm | None, value: int) -> None:
    if bar is None:
        return
    value = max(0, min(value, bar.total or value))
    bar.n = value
    bar.refresh()


def reset_hop_bar(bar: tqdm | None, value: int, total: int) -> None:
    if bar is None:
        return
    bar.reset(total=total)
    set_bar_position(bar, value)


def construct_path(
    context: BuildContext,
    frames: list[TravelFrame],
    hop_bar: tqdm | None = None,
    *,
    minimum_frame_count: int = 1,
    traversal_mode: str | None = None,
    checkpoint_callback: Callable[[list[TravelFrame], int], None] | None = None,
    deadline: float | None = None,
) -> tuple[list[TravelFrame], str]:
    """Extend ``frames`` to the target length by random search with backtracking.

    When ``deadline`` (a ``time.monotonic()`` value) passes before the path is
    complete, returns ``PATH_TIMED_OUT``. The time is checked between steps.
    The deepest path length reached is stored in ``context.last_path_max_nodes``.
    """
    if not frames:
        raise ValueError("construct_path requires at least one frame")
    if minimum_frame_count < 1 or minimum_frame_count > len(frames):
        raise ValueError("minimum_frame_count must be within the initial frames")

    visited = {frame.node_id for frame in frames}
    context.last_path_max_nodes = len(frames)
    temporarily_skipped_edges: set[tuple[str, str]] = set()
    effective_traversal_mode = traversal_mode or getattr(
        context,
        "traversal_mode",
        TRAVERSAL_MODE_MIXED,
    )
    set_bar_position(hop_bar, len(frames))

    while frames:
        if len(frames) >= context.target_num_nodes:
            set_bar_position(hop_bar, len(frames))
            return frames, PATH_COMPLETE
        if deadline is not None and time.monotonic() >= deadline:
            return frames, PATH_TIMED_OUT

        current = frames[-1]
        try:
            candidates = [
                candidate
                for candidate in collect_candidates(context, current.node_id)
                if candidate_allowed_from_current(
                    current.node_id,
                    candidate.node_id,
                    effective_traversal_mode,
                    edge_type=str(candidate.edge[2].get("edge_type", "")),
                )
                and candidate.node_id not in visited
                and candidate.node_id not in current.tried_nodes
                and (current.node_id, candidate.node_id) not in temporarily_skipped_edges
            ]
        except SemanticScholarRateLimitError:
            logger.warning(
                "Deferring path at {} after Semantic Scholar rate limiting; waiting {:.0f}s before trying another branch",
                current.node_id,
                RATE_LIMIT_DEFER_WAIT_SECONDS,
            )
            time.sleep(RATE_LIMIT_DEFER_WAIT_SECONDS)
            if len(frames) == minimum_frame_count:
                return frames, PATH_DEFERRED
            abandoned = frames.pop()
            visited.remove(abandoned.node_id)
            parent = frames[-1]
            parent.tried_nodes.discard(abandoned.node_id)
            temporarily_skipped_edges.add((parent.node_id, abandoned.node_id))
            set_bar_position(hop_bar, len(frames))
            continue

        if not candidates:
            if len(frames) == minimum_frame_count:
                break
            abandoned = frames.pop()
            visited.remove(abandoned.node_id)
            logger.debug("Backtracking from {} to {}", abandoned.node_id, frames[-1].node_id)
            set_bar_position(hop_bar, len(frames))
            continue

        candidate = context.rng.choice(candidates)
        current.tried_nodes.add(candidate.node_id)
        visited.add(candidate.node_id)
        frames.append(
            TravelFrame(
                node_id=candidate.node_id,
                incoming_edge=candidate.edge,
            )
        )
        context.last_path_max_nodes = max(context.last_path_max_nodes, len(frames))
        set_bar_position(hop_bar, len(frames))

        moves = len(frames) - minimum_frame_count
        if context.checkpoint_every and moves % context.checkpoint_every == 0:
            if checkpoint_callback is None:
                save_checkpoint(context, frames, label=f"hop_{moves:03d}")
            else:
                checkpoint_callback(frames, moves)

    return frames, PATH_DEFERRED if temporarily_skipped_edges else PATH_EXHAUSTED


def upcoming_seed_batch(
    context: BuildContext,
    title: str,
    max_size: int,
    belongs: Callable[[str], bool],
) -> list[str]:
    """Return ``title`` followed by the next uncached seed titles after it for
    which ``belongs`` is true, up to ``max_size`` titles in total."""
    start = (
        context.seed_titles.index(title) + 1
        if title in context.seed_titles
        else len(context.seed_titles)
    )
    batch = [title]
    for upcoming_title in context.seed_titles[start:]:
        if len(batch) >= max_size:
            break
        if (
            upcoming_title not in context.seed_title_matches
            and upcoming_title not in batch
            and belongs(upcoming_title)
        ):
            batch.append(upcoming_title)
    return batch


def fetch_seed_paper_by_id(context: BuildContext, title: str) -> dict[str, Any] | None:
    """Return the paper of a seed whose CSV row gave a Semantic Scholar id, or
    None when the client has no paper with that id.

    A cache miss fetches this seed together with the next uncached id seeds in
    one ``get_papers_batch()`` call (at most ``bulk_title_search_batch_size``,
    and at most ONLINE_PAPER_BATCH_LIMIT online). Every result is cached for
    the run. The papers have the default fields, like title-search matches.
    """
    seed_ids = context.seed_semantic_scholar_ids
    if title not in context.seed_title_matches:
        max_size = context.bulk_title_search_batch_size
        if not uses_bulk_client(context):
            max_size = min(max_size, ONLINE_PAPER_BATCH_LIMIT)
        batch = upcoming_seed_batch(context, title, max_size, lambda upcoming: upcoming in seed_ids)
        papers = context.client.get_papers_batch([seed_ids[batch_title] for batch_title in batch])
        for batch_title, paper in zip(batch, papers):
            context.seed_title_matches[batch_title] = paper
    return context.seed_title_matches[title]


def search_seed_title(context: BuildContext, title: str) -> dict[str, Any] | None:
    """Return the Semantic Scholar match for one seed title.

    A seed with a Semantic Scholar id from a CSV --input-file is fetched by
    that id and never searched by title (see fetch_seed_paper_by_id()).

    With the bulk backend, a cache miss searches this title together with the
    next uncached seed titles without an id (up to
    ``bulk_title_search_batch_size``) in one ``search_by_titles_batch()`` call,
    and every match is cached for the run, so repeats and later paths from the
    same seed never search again. This only searches: a seed paper is written
    to the database when its path starts.
    """
    seed_ids = getattr(context, "seed_semantic_scholar_ids", {})
    if title in seed_ids:
        return fetch_seed_paper_by_id(context, title)
    if not uses_bulk_client(context):
        return context.client.search_by_title(title)
    if title not in context.seed_title_matches:
        batch = upcoming_seed_batch(
            context,
            title,
            context.bulk_title_search_batch_size,
            lambda upcoming: upcoming not in seed_ids,
        )
        for batch_title, paper in zip(batch, context.client.search_by_titles_batch(batch)):
            context.seed_title_matches[batch_title] = drop_weak_title_match(
                paper,
                context.bulk_title_min_match_score,
            )
    return context.seed_title_matches[title]


def resolve_seed_node(context: BuildContext, title: str) -> str | None:
    paper = search_seed_title(context, title)
    if not paper or not paper.get("paperId"):
        seed_id = getattr(context, "seed_semantic_scholar_ids", {}).get(title)
        if seed_id:
            logger.warning(
                "Skipping seed; Semantic Scholar has no paper with id {} (title: {})",
                seed_id,
                title,
            )
        else:
            logger.warning("Skipping seed; Semantic Scholar found no paper for title: {}", title)
        return None
    node_id = upsert_semantic_paper(
        context.network,
        paper,
        depth=0,
        explored=False,
    )
    if node_id is None:
        logger.warning("Skipping seed; could not insert matched paper for title: {}", title)
        return None
    return node_id


def next_seed_position(seed_index: int, repeat_index: int, paths_per_seed: int) -> tuple[int, int]:
    repeat_index += 1
    if repeat_index >= paths_per_seed:
        return seed_index + 1, 0
    return seed_index, repeat_index


def append_completed_trace(
    context: BuildContext,
    frames: list[TravelFrame],
) -> None:
    trace = trace_from_frames(frames)
    path_ids = ensure_completed_path_ids(context)
    if context.completed_traces and trace_equals_frames(
        context.completed_traces[-1],
        frames,
    ):
        context.completed_traces[-1] = trace
    else:
        context.completed_traces.append(trace)
        path_ids.append(generate_path_id())


def set_seed_progress(seed_bar: tqdm, seed_index: int) -> None:
    target = max(0, min(seed_index, seed_bar.total or seed_index))
    seed_bar.n = target
    seed_bar.refresh()


def disable_progress_bars(no_tqdm_bar: int = 0) -> bool:
    return bool(no_tqdm_bar) or not sys.stderr.isatty()


def construct_remaining_paths(
    context: BuildContext,
    resume_frames: list[TravelFrame] | None = None,
    pending_checkpoint_frames: list[list[TravelFrame]] | None = None,
) -> None:
    seed_index = context.current_seed_index
    repeat_index = context.current_repeat_index
    path_started = False
    active_frames: list[TravelFrame] = []
    pending_frames_list = list(pending_checkpoint_frames or [])
    if (
        resume_frames
        and len(resume_frames) < context.target_num_nodes
        and seed_index >= len(context.seed_titles)
    ):
        pending_frames_list.append(resume_frames)
        resume_frames = None
    pending_total = len(pending_frames_list)
    progress_bars_disabled = disable_progress_bars(
        getattr(context, "no_tqdm_bar", 0)
    )

    seed_bar = tqdm(
        total=pending_total if pending_total else len(context.seed_titles),
        initial=0 if pending_total else min(seed_index, len(context.seed_titles)),
        desc="Checkpoint paths" if pending_total else "Seed papers",
        unit="path" if pending_total else "seed",
        position=0,
        disable=progress_bars_disabled,
    )
    hop_bar = tqdm(
        total=context.target_num_nodes,
        desc="Current path nodes",
        unit="node",
        position=1,
        leave=False,
        disable=progress_bars_disabled,
    )

    try:
        if target_paths_reached(context):
            logger.info(
                "Target complete paths already reached | completed={}/{}",
                complete_path_count(context),
                target_path_count(context),
            )
            return

        for pending_index, pending_frames in enumerate(pending_frames_list, start=1):
            if target_paths_reached(context):
                logger.info(
                    "Target complete paths reached | completed={}/{}",
                    complete_path_count(context),
                    target_path_count(context),
                )
                break
            if not pending_frames:
                continue
            if path_started and not progress_bars_disabled:
                tqdm.write("")
            path_started = True
            active_frames = pending_frames
            reset_hop_bar(hop_bar, len(pending_frames), context.target_num_nodes)
            logger.info(
                "Continuing checkpoint path | {}/{} | nodes={}/{}",
                pending_index,
                pending_total,
                len(pending_frames),
                context.target_num_nodes,
            )
            frames, status = construct_path(
                context,
                pending_frames,
                hop_bar,
                deadline=seed_path_deadline(context),
            )
            if status == PATH_COMPLETE:
                append_completed_trace(context, frames)
                save_checkpoint(
                    context,
                    [],
                    label="path_done",
                    include_current_trace=False,
                )
                logger.info(
                    "Completed checkpoint path {} | nodes={}/{}",
                    completed_path_log_label(context),
                    len(frames),
                    context.target_num_nodes,
                )
                if target_paths_reached(context):
                    set_bar_position(seed_bar, pending_index)
                    logger.info(
                        "Target complete paths reached | completed={}/{}",
                        complete_path_count(context),
                        target_path_count(context),
                    )
                    break
            elif status == PATH_DEFERRED:
                save_checkpoint(
                    context,
                    frames,
                    label="path_deferred",
                    include_current_trace=False,
                )
                logger.warning(
                    "Checkpoint path deferred because Semantic Scholar stayed rate-limited | nodes={}/{}",
                    len(frames),
                    context.target_num_nodes,
                )
            elif status == PATH_TIMED_OUT:
                failure_summary = record_failed_seed(context)
                save_checkpoint(
                    context,
                    [],
                    label="path_timed_out",
                    include_current_trace=False,
                )
                logger.warning(
                    "Skipping checkpoint path after {:.1f} minutes without a complete path | nodes={}/{} | {}",
                    context.bulk_seed_timeout_seconds / 60,
                    len(frames),
                    context.target_num_nodes,
                    failure_summary,
                )
            else:
                failure_summary = record_failed_seed(context)
                save_checkpoint(
                    context,
                    [],
                    label="path_exhausted",
                    include_current_trace=False,
                )
                logger.warning(
                    "Checkpoint path cannot reach target after full backtracking | nodes={}/{} | {}",
                    len(frames),
                    context.target_num_nodes,
                    failure_summary,
                )
            set_bar_position(seed_bar, pending_index)
            active_frames = []

        if pending_total and seed_index < len(context.seed_titles):
            seed_bar.reset(total=len(context.seed_titles))
            seed_bar.set_description("Seed papers")
            seed_bar.unit = "seed"
            set_seed_progress(seed_bar, seed_index)

        if (
            resume_frames
            and seed_index >= len(context.seed_titles)
            and len(resume_frames) >= context.target_num_nodes
        ):
            logger.info(
                "Checkpoint already covers all input papers | paths={}",
                len(context.completed_traces),
            )
            set_seed_progress(seed_bar, len(context.seed_titles))
            return

        if resume_frames:
            if target_paths_reached(context):
                logger.info(
                    "Target complete paths reached | completed={}/{}",
                    complete_path_count(context),
                    target_path_count(context),
                )
                return
            if len(resume_frames) >= context.target_num_nodes:
                reset_hop_bar(hop_bar, len(resume_frames), context.target_num_nodes)
                logger.info(
                    "Checkpoint path already has enough nodes | seed {}/{} | nodes={}/{}",
                    seed_index + 1,
                    len(context.seed_titles),
                    len(resume_frames),
                    context.target_num_nodes,
                )
                seed_index, repeat_index = next_seed_position(
                    seed_index,
                    repeat_index,
                    context.paths_per_seed,
                )
                set_seed_progress(seed_bar, seed_index)
            else:
                if context.completed_traces and trace_equals_frames(
                    context.completed_traces[-1], resume_frames
                ):
                    context.completed_traces.pop()

                active_frames = resume_frames
                reset_hop_bar(hop_bar, len(resume_frames), context.target_num_nodes)
                logger.info(
                    "Resuming incomplete path | seed {}/{} | nodes={}/{}",
                    seed_index + 1,
                    len(context.seed_titles),
                    len(resume_frames),
                    context.target_num_nodes,
                )
                frames, status = construct_path(
                    context,
                    resume_frames,
                    hop_bar,
                    deadline=seed_path_deadline(context),
                )
                attempted_seed_index = seed_index
                attempted_repeat_index = repeat_index
                seed_index, repeat_index = next_seed_position(
                    seed_index,
                    repeat_index,
                    context.paths_per_seed,
                )
                if status == PATH_COMPLETE:
                    context.current_seed_index = seed_index
                    context.current_repeat_index = repeat_index
                    append_completed_trace(context, frames)
                    save_checkpoint(context, frames, label="path_done")
                    logger.info(
                        "Completed path {} | nodes={}/{}",
                        completed_path_log_label(context),
                        len(frames),
                        context.target_num_nodes,
                    )
                    if target_paths_reached(context):
                        logger.info(
                            "Target complete paths reached | completed={}/{}",
                            complete_path_count(context),
                            target_path_count(context),
                        )
                        return
                elif status == PATH_DEFERRED:
                    context.current_seed_index = seed_index
                    context.current_repeat_index = repeat_index
                    save_checkpoint(
                        context,
                        frames,
                        label="path_deferred",
                        include_current_trace=False,
                    )
                    logger.warning(
                        "Deferred path because Semantic Scholar stayed rate-limited | seed {}/{} | repeat {}/{} | nodes={}/{}",
                        attempted_seed_index + 1,
                        len(context.seed_titles),
                        attempted_repeat_index + 1,
                        context.paths_per_seed,
                        len(frames),
                        context.target_num_nodes,
                    )
                elif status == PATH_TIMED_OUT:
                    # Skip the whole seed, including its remaining repeats.
                    seed_index, repeat_index = attempted_seed_index + 1, 0
                    context.current_seed_index = seed_index
                    context.current_repeat_index = repeat_index
                    failure_summary = record_failed_seed(context)
                    save_checkpoint(
                        context,
                        [],
                        label="path_timed_out",
                        include_current_trace=False,
                    )
                    logger.warning(
                        "Skipping seed after {:.1f} minutes without a complete path | seed {}/{} | repeat {}/{} | nodes={}/{} | {}",
                        context.bulk_seed_timeout_seconds / 60,
                        attempted_seed_index + 1,
                        len(context.seed_titles),
                        attempted_repeat_index + 1,
                        context.paths_per_seed,
                        len(frames),
                        context.target_num_nodes,
                        failure_summary,
                    )
                else:
                    context.current_seed_index = seed_index
                    context.current_repeat_index = repeat_index
                    failure_summary = record_failed_seed(context)
                    save_checkpoint(
                        context,
                        [],
                        label="path_exhausted",
                        include_current_trace=False,
                    )
                    logger.warning(
                        "No complete path found after full backtracking | seed {}/{} | repeat {}/{} | target nodes={} | {}",
                        attempted_seed_index + 1,
                        len(context.seed_titles),
                        attempted_repeat_index + 1,
                        context.paths_per_seed,
                        context.target_num_nodes,
                        failure_summary,
                    )
                set_seed_progress(seed_bar, seed_index)
                path_started = True
                active_frames = []

        for current_seed_index in range(seed_index, len(context.seed_titles)):
            if target_paths_reached(context):
                break
            start_repeat = repeat_index if current_seed_index == seed_index else 0
            for current_repeat_index in range(start_repeat, context.paths_per_seed):
                if target_paths_reached(context):
                    break
                context.current_seed_index = current_seed_index
                context.current_repeat_index = current_repeat_index
                seed_title = context.seed_titles[current_seed_index]
                start_node = resolve_seed_node(context, seed_title)
                if start_node is None:
                    break
                frames = [TravelFrame(node_id=start_node)]
                if path_started and not progress_bars_disabled:
                    tqdm.write("")
                path_started = True
                active_frames = frames
                reset_hop_bar(hop_bar, len(frames), context.target_num_nodes)
                logger.info(
                    "Starting path | seed {}/{} | repeat {}/{} | {}",
                    current_seed_index + 1,
                    len(context.seed_titles),
                    current_repeat_index + 1,
                    context.paths_per_seed,
                    seed_title,
                )
                frames, status = construct_path(
                    context,
                    frames,
                    hop_bar,
                    deadline=seed_path_deadline(context),
                )
                if status == PATH_COMPLETE:
                    next_seed_index, next_repeat_index = next_seed_position(
                        current_seed_index,
                        current_repeat_index,
                        context.paths_per_seed,
                    )
                    context.current_seed_index = next_seed_index
                    context.current_repeat_index = next_repeat_index
                    append_completed_trace(context, frames)
                    save_checkpoint(context, frames, label="path_done")
                    logger.info(
                        "Completed path {} | nodes={}/{}",
                        completed_path_log_label(context),
                        len(frames),
                        context.target_num_nodes,
                    )
                    if target_paths_reached(context):
                        logger.info(
                            "Target complete paths reached | completed={}/{}",
                            complete_path_count(context),
                            target_path_count(context),
                        )
                        break
                elif status == PATH_DEFERRED:
                    next_seed_index, next_repeat_index = next_seed_position(
                        current_seed_index,
                        current_repeat_index,
                        context.paths_per_seed,
                    )
                    context.current_seed_index = next_seed_index
                    context.current_repeat_index = next_repeat_index
                    save_checkpoint(
                        context,
                        frames,
                        label="path_deferred",
                        include_current_trace=False,
                    )
                    logger.warning(
                        "Deferred path because Semantic Scholar stayed rate-limited | seed {}/{} | repeat {}/{} | nodes={}/{}",
                        current_seed_index + 1,
                        len(context.seed_titles),
                        current_repeat_index + 1,
                        context.paths_per_seed,
                        len(frames),
                        context.target_num_nodes,
                    )
                elif status == PATH_TIMED_OUT:
                    # Skip the whole seed, including its remaining repeats.
                    context.current_seed_index = current_seed_index + 1
                    context.current_repeat_index = 0
                    failure_summary = record_failed_seed(context)
                    save_checkpoint(
                        context,
                        [],
                        label="path_timed_out",
                        include_current_trace=False,
                    )
                    logger.warning(
                        "Skipping seed after {:.1f} minutes without a complete path | seed {}/{} | repeat {}/{} | nodes={}/{} | {} | {}",
                        context.bulk_seed_timeout_seconds / 60,
                        current_seed_index + 1,
                        len(context.seed_titles),
                        current_repeat_index + 1,
                        context.paths_per_seed,
                        len(frames),
                        context.target_num_nodes,
                        failure_summary,
                        seed_title,
                    )
                    active_frames = []
                    break
                else:
                    next_seed_index, next_repeat_index = next_seed_position(
                        current_seed_index,
                        current_repeat_index,
                        context.paths_per_seed,
                    )
                    context.current_seed_index = next_seed_index
                    context.current_repeat_index = next_repeat_index
                    failure_summary = record_failed_seed(context)
                    save_checkpoint(
                        context,
                        [],
                        label="path_exhausted",
                        include_current_trace=False,
                    )
                    logger.warning(
                        "No complete path found after full backtracking | seed {}/{} | repeat {}/{} | target nodes={} | {}",
                        current_seed_index + 1,
                        len(context.seed_titles),
                        current_repeat_index + 1,
                        context.paths_per_seed,
                        context.target_num_nodes,
                        failure_summary,
                    )
                active_frames = []
            repeat_index = 0
            set_seed_progress(seed_bar, current_seed_index + 1)
            if target_paths_reached(context):
                break
        if not target_paths_reached(context):
            context.current_seed_index = len(context.seed_titles)
            context.current_repeat_index = 0
        active_frames = []
    except KeyboardInterrupt:
        save_interrupted_run_state(context, active_frames)
        raise
    finally:
        hop_bar.close()
        seed_bar.close()


def copy_database(source_db: Path, target_db: Path) -> None:
    copy_sqlite_database(source_db, target_db)


def payload_database_path(payload: dict[str, Any] | None) -> Path | None:
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    keys = ("working_db_path", "db_path", "graph_db_path")
    if metadata.get("node_ids_mapped_to_graph_db"):
        keys = ("db_path", "graph_db_path", "working_db_path")
    for key in keys:
        raw_path = str(metadata.get(key) or "").strip()
        if not raw_path:
            continue
        candidate = Path(raw_path)
        if candidate.is_file() and candidate.suffix.lower() == ".db":
            return candidate
        sibling_db = candidate.with_suffix(".db")
        if sibling_db.is_file():
            return sibling_db
    return None


def remap_trace_node_ids(
    trace: list[tuple[str, tuple[str, str, dict[str, Any]] | None]],
    node_id_updates: dict[str, str],
) -> list[tuple[str, tuple[str, str, dict[str, Any]] | None]]:
    remapped: list[tuple[str, tuple[str, str, dict[str, Any]] | None]] = []
    for node_id, incoming_edge in trace:
        remapped_node_id = node_id_updates.get(node_id, node_id)
        if incoming_edge is None:
            remapped.append((remapped_node_id, None))
            continue
        source, target, attrs = incoming_edge
        remapped.append(
            (
                remapped_node_id,
                (
                    node_id_updates.get(source, source),
                    node_id_updates.get(target, target),
                    dict(attrs),
                ),
            )
        )
    return remapped


def align_path_records_database(
    payload: dict[str, Any] | None,
    records: list[PathRecord],
    working_db: Path,
) -> None:
    source_db = payload_database_path(payload)
    if source_db is None:
        return
    if not working_db.exists():
        if source_db.resolve() != working_db.resolve():
            copy_database(source_db, working_db)
        return
    if source_db.resolve() == working_db.resolve():
        return

    from merge_graph_databases import SourceBundle, merge_source

    canonical = DatabaseManager(working_db, validate_uuid_values=False)
    try:
        node_id_updates, _stats = merge_source(
            canonical,
            SourceBundle(db_path=source_db),
            dry_run=False,
        )
    finally:
        canonical.close()
    for record in records:
        record.trace = remap_trace_node_ids(record.trace, node_id_updates)
    logger.info(
        "Merged source graph database | source={} | destination={} | remapped={}",
        source_db,
        working_db,
        len(node_id_updates),
    )


def prepare_working_database_from_graph(
    graph_argument: str | None,
    working_db: Path,
) -> None:
    if not graph_argument or working_db.exists():
        return
    from src.graph.resolution import (
        resolve_graph_argument,
        resolve_latest_graph_db_paths,
    )

    graph_path = resolve_graph_argument([graph_argument])
    _graph_json, source_db, _resolution = resolve_latest_graph_db_paths(graph_path)
    if source_db.resolve() != working_db.resolve():
        copy_database(source_db, working_db)


def open_persistent_network(
    working_db: Path,
    run_stem: str,
    graph_backend: str,
) -> CitationNetwork | SQLiteCitationNetwork:
    if graph_backend == GRAPH_BACKEND_SQLITE:
        return open_sqlite_network(working_db, run_stem)
    if working_db.exists():
        from src.graph.loaders import load_graph

        return load_graph(working_db)
    network = CitationNetwork(DatabaseManager(working_db))
    network.max_depth = None
    return network


def validate_path_nodes(
    network: CitationNetwork | SQLiteCitationNetwork,
    records: list[PathRecord],
    *,
    source_label: str,
) -> None:
    missing: list[str] = []
    for record in records:
        for node_id, _edge in record.trace:
            node_type = nid_type(node_id)
            if node_type == NODE_PAPER:
                row = network.db.get_paper(nid_db_id(node_id))
            elif node_type == NODE_AUTHOR:
                row = network.db.get_author(nid_db_id(node_id))
            else:
                row = None
            if row is None:
                missing.append(node_id)
                if len(missing) >= 5:
                    break
        if len(missing) >= 5:
            break
    if missing:
        raise SystemExit(
            f"Path nodes from {source_label} are missing from the working database: "
            + ", ".join(missing)
        )


def load_network_for_mutation(
    graph_json: Path,
    working_db: Path,
) -> CitationNetwork:
    from src.graph.loaders import load_graph
    from src.graph.resolution import resolve_latest_graph_db_paths

    graph_json, source_db, db_resolution = resolve_latest_graph_db_paths(graph_json)
    if source_db.resolve() != working_db.resolve():
        copy_database(source_db, working_db)
    logger.disable("src.graph.loaders")
    try:
        network = load_graph(graph_json, db_path=working_db)
    finally:
        logger.enable("src.graph.loaders")
    logger.info(
        "Loaded checkpoint graph | graph={} | db={} | source_db={} [{}]",
        graph_json,
        working_db,
        source_db,
        db_resolution,
    )
    return network


def load_network_from_resume_state(
    state_path: Path,
    state: dict[str, Any],
    graph_dir: Path,
    working_db: Path,
) -> tuple[CitationNetwork, Path | None, set[str], set[tuple[str, str]]]:
    from src.graph.loaders import load_graph

    graph_delta = state.get("graph_delta") or {}
    saved_graph_text = str(
        state.get("graph_db_path")
        or state.get("db_path")
        or graph_delta.get("base_graph_db_path")
        or ""
    ).strip()
    state_working_db_text = str(state.get("working_db_path") or "").strip()

    db_candidates: list[Path] = []
    if state_working_db_text:
        db_candidates.append(Path(state_working_db_text))
    db_candidates.append(working_db)

    seen_candidates: set[str] = set()
    for candidate in db_candidates:
        candidate_key = str(candidate)
        if candidate_key in seen_candidates or not candidate.exists():
            continue
        seen_candidates.add(candidate_key)
        if candidate.resolve() != working_db.resolve():
            copy_database(candidate, working_db)
        network = load_graph(working_db)
        base_nodes, base_edges = capture_graph_base(network)
        logger.info(
            "Resumed checkpoint from working database | source_db={} | db={}",
            candidate,
            working_db,
        )
        return network, working_db, base_nodes, base_edges

    if not saved_graph_text:
        raise SystemExit(
            "Checkpoint has no saved graph snapshot and no working database is available: "
            f"{working_db}"
        )

    state["graph_db_path"] = saved_graph_text
    graph_json = resolve_resume_graph_json(state_path, state, graph_dir)
    network = load_network_for_mutation(graph_json, working_db)
    base_nodes, base_edges = capture_graph_base(network)
    if graph_delta:
        logger.warning(
            "Working database is missing; resumed from the last graph snapshot only. "
            "Checkpoint graph deltas were ignored because the DB is now the source of truth "
            "| delta nodes={} | delta edges={}",
            len(graph_delta.get("nodes") or []),
            len(graph_delta.get("edges") or []),
        )
    return network, graph_json, base_nodes, base_edges


def load_network_from_output_payload(
    output_payload: dict[str, Any],
    graph_dir: Path,
    working_db: Path,
) -> tuple[CitationNetwork, Path | None, set[str], set[tuple[str, str]]]:
    from src.graph.loaders import load_graph
    from src.graph.resolution import resolve_latest_graph_db_paths

    metadata = output_payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    metadata_db: Path | None = None
    metadata_db_keys = ("working_db_path", "db_path", "graph_db_path")
    if metadata.get("node_ids_mapped_to_graph_db"):
        metadata_db_keys = ("db_path", "graph_db_path", "working_db_path")
    for key in metadata_db_keys:
        raw_path = str(metadata.get(key) or "").strip()
        if not raw_path:
            continue
        candidate = Path(raw_path)
        if candidate.exists():
            metadata_db = candidate
            break

    db_candidates = [candidate for candidate in (metadata_db, working_db) if candidate is not None]
    seen_candidates: set[str] = set()
    for candidate in db_candidates:
        candidate_key = str(candidate)
        if candidate_key in seen_candidates or not candidate.exists():
            continue
        seen_candidates.add(candidate_key)
        if candidate.resolve() != working_db.resolve():
            copy_database(candidate, working_db)
        network = load_graph(working_db)
        base_nodes, base_edges = capture_graph_base(network)
        logger.info(
            "Loaded DB-backed output context | source_db={} | db={}",
            candidate,
            working_db,
        )
        return network, working_db, base_nodes, base_edges

    graph_json_text = str(metadata.get("graph_db_path") or metadata.get("db_path") or "").strip()
    if graph_json_text:
        graph_json = Path(graph_json_text)
        if graph_json.exists():
            network = load_network_for_mutation(graph_json, working_db)
            base_nodes, base_edges = capture_graph_base(network)
            logger.info(
                "Loaded output graph context | graph={} | db={}",
                graph_json,
                working_db,
            )
            return network, graph_json, base_nodes, base_edges
        logger.warning(
            "Output graph snapshot is missing; falling back to --graph-dir / DB metadata | graph={}",
            graph_json,
        )

    try:
        graph_json, _resolved_db, _db_resolution = resolve_latest_graph_db_paths(graph_dir)
    except FileNotFoundError:
        graph_json = None
    if graph_json is not None:
        network = load_network_for_mutation(graph_json, working_db)
        base_nodes, base_edges = capture_graph_base(network)
        logger.info(
            "Loaded latest graph context for output resume | graph={} | db={}",
            graph_json,
            working_db,
        )
        return network, graph_json, base_nodes, base_edges

    network = CitationNetwork(DatabaseManager(working_db))
    network.max_depth = None
    logger.warning(
        "No graph or DB context found for output resume; continuing with an empty network | output_db={} | graph_dir={}",
        metadata.get("working_db_path") or metadata.get("db_path"),
        graph_dir,
    )
    return network, None, set(), set()


def resolve_working_db_path(db_path: str | None, graph_dir: Path) -> Path:
    if db_path:
        return Path(db_path)
    return graph_dir / DEFAULT_SHARED_DB_NAME


def resolve_output_path(output: str | None, output_dir: Path) -> Path:
    if output:
        return Path(output)
    return output_dir / DEFAULT_PATHS_JSON_NAME


def count_completed_paths_in_output(
    output_path: Path,
    target_num_nodes: int | None = None,
    target_num_paths: int | None = None,
) -> int:
    traces, _path_ids = load_completed_paths_from_output(
        output_path,
        target_num_nodes,
        target_num_paths,
    )
    return len(traces)


def remove_legacy_checkpoint_output(checkpoint_dir: Path, output_path: Path) -> None:
    legacy_output_path = legacy_checkpoint_output_path(checkpoint_dir)
    if legacy_output_path.is_file():
        if not output_path.is_file():
            return
        if legacy_output_path.resolve() == output_path.resolve():
            return
        legacy_output_path.unlink()
        logger.info(
            "Removed legacy checkpoint paths file | {}",
            legacy_output_path,
        )


def resolve_output_locations(output_dir: str, output: str | None) -> tuple[Path, Path]:
    resolved_output_dir = Path(output_dir)
    if resolved_output_dir.suffix == ".json" or resolved_output_dir.is_file():
        return resolved_output_dir.parent, resolve_output_path(output, resolved_output_dir.parent)
    return resolved_output_dir, resolve_output_path(output, resolved_output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=RawDefaultsHelpFormatter,
    )
    parser.add_argument(
        "seed_titles",
        nargs="*",
        help="Seed paper title(s). Quote titles that contain spaces.",
    )
    parser.add_argument(
        "--input-file",
        dest="input_file",
        help="Text file with one paper title per line, or a .csv file with the columns "
        "semantic_scholar_id,title (from endpoints/export_s2_bulk_newest_papers.py) whose "
        "seeds are fetched by id; rows with an empty id are found by title.",
    )
    parser.add_argument(
        "--seed-file",
        dest="input_file",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--extend-from-output",
        default=None,
        help=(
            "Use paths from an earlier sampled_paths.json as starting paths. "
            "Cannot be combined with seed titles or --input-file."
        ),
    )
    parser.add_argument(
        "--num-hops",
        type=positive_int,
        required=True,
        help="Target number of nodes in each path, including the start node.",
    )
    parser.add_argument(
        "--paths-per-seed",
        type=positive_int,
        default=1,
        help="Number of independently sampled paths to construct for each seed.",
    )
    parser.add_argument(
        "--target-num-paths",
        type=positive_int,
        default=None,
        help="Stop after this many complete paths are discovered.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_start_date,
        default=None,
        help="Inclusive minimum paper date for travel candidates.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_end_date,
        default=None,
        help="Inclusive maximum paper date for travel candidates.",
    )
    parser.add_argument(
        "--traversal-mode",
        choices=TRAVERSAL_MODES,
        default=argparse.SUPPRESS,
        help=(
            "Allowed node-type transitions: mixed permits paper-to-paper and "
            "paper-to-author travel; papers-only excludes authors; papers-forward "
            "follows papers that cite the current paper; papers-backward follows "
            "references cited by the current paper; alternating enforces "
            "paper-author-paper travel. Defaults to mixed for new runs and to the "
            "saved mode when resuming."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Output root for traversal states and sampled_paths.json. "
            "The checkpoints/ directory stores only traversal .state.json files. "
            "Extension runs update short paths in place and append new paths. "
            "If checkpoints already exist and --load-graph / --resume-checkpoint "
            "are omitted, the newest checkpoint is resumed automatically."
        ),
    )
    parser.add_argument(
        "--graph-dir",
        default=str(DEFAULT_GRAPH_DIR),
        help="Directory for Semantic Scholar graph database snapshots.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Paths JSON path, rewritten once after each finished path. "
            "Defaults to <output-dir>/sampled_paths.json."
        ),
    )
    parser.add_argument(
        "--load-graph",
        default=None,
        help="Existing graph DB, legacy graph JSON, graphs directory, or output root to continue from.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help=(
            "Traversal .state.json checkpoint to resume. Extension checkpoints "
            "contain only the active path job."
        ),
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Working SQLite DB path. Defaults to <graph-dir>/semantic_scholar_paths.db.",
    )
    parser.add_argument(
        "--graph-backend",
        choices=GRAPH_BACKENDS,
        default=GRAPH_BACKEND_NETWORKX,
        help=(
            "Graph access used by this sampler. sqlite queries the shared DB for "
            "every neighborhood and disables per-worker graph snapshots."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=nonnegative_int,
        default=5,
        help=(
            "Save a lightweight traversal checkpoint after every N successful "
            "moves. 0 disables mid-path checkpoints."
        ),
    )
    parser.add_argument(
        "--max-checkpoints",
        type=nonnegative_int,
        default=2,
        help="Maximum checkpoint sets to keep. 0 keeps all.",
    )
    parser.add_argument(
        "--graph-snapshot-every",
        type=nonnegative_int,
        default=GRAPH_SNAPSHOT_COMPLETED_PATH_INTERVAL,
        help=(
            "Save a full graph snapshot after every N completed paths. 0 disables "
            "periodic snapshots (a final snapshot is still written at the end). "
            "Larger values speed up long runs; resume relies on the working DB and "
            "per-path state checkpoints between snapshots."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducible input order and travel choices.",
    )
    parser.add_argument(
        "--semantic-scholar-delay",
        type=float,
        default=2.0,
        help="Minimum seconds between online requests; ignored by the bulk backend.",
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
        "--bulk-title-search-batch-size",
        type=positive_int,
        default=BULK_TITLE_SEARCH_BATCH_SIZE,
        help="Upcoming seed titles resolved per bulk Spark title-search query (the online "
        "backend searches one title at a time). Also the upcoming CSV seed ids fetched "
        "together by id, with either backend (at most 500 online).",
    )
    parser.add_argument(
        "--bulk-title-min-match-score",
        type=title_match_score,
        default=BULK_TITLE_MIN_MATCH_SCORE,
        help="Reject bulk title matches (seeds and title-only papers) below this 0-100 "
        "score. Exact normalized title matches score 100. Ignored by the online backend.",
    )
    parser.add_argument(
        "--bulk-seed-timeout-minutes",
        type=nonnegative_float,
        default=BULK_SEED_TIMEOUT_MINUTES,
        help="Skip a seed when its path is not complete after this many minutes of path "
        "search; the seed is not retried on resume. 0 disables the limit. Ignored by "
        "the online backend.",
    )
    parser.add_argument(
        "--no-tqdm-bar",
        type=int,
        choices=(0, 1),
        default=0,
        help="Set to 1 to disable tqdm progress bars; informational logs remain visible.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def create_semantic_scholar_client(
    args: argparse.Namespace,
) -> SemanticScholarClient | SemanticScholarBulkClient:
    """Create the configured online or indexed bulk client."""
    backend = getattr(
        args,
        "semantic_scholar_backend",
        SEMANTIC_SCHOLAR_BACKEND_ONLINE,
    )
    if backend == SEMANTIC_SCHOLAR_BACKEND_BULK:
        data_root = Path(getattr(args, "bulk_data_root", DEFAULT_BULK_DATA_ROOT))
        index_root = Path(getattr(args, "bulk_index_root", DEFAULT_BULK_INDEX_ROOT))
        metadata_path = index_root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Semantic Scholar bulk index not found: {metadata_path}"
            )
        return SemanticScholarBulkClient(
            data_root=data_root,
            index_root=index_root,
            use_index=True,
        )
    if backend != SEMANTIC_SCHOLAR_BACKEND_ONLINE:
        raise ValueError(f"Unsupported Semantic Scholar backend: {backend}")
    return SemanticScholarClient(delay_seconds=args.semantic_scholar_delay)


def close_semantic_scholar_client(
    client: SemanticScholarClient | SemanticScholarBulkClient,
    backend: str,
) -> None:
    """Close the selected client and stop its Spark session when needed."""
    if backend == SEMANTIC_SCHOLAR_BACKEND_BULK:
        client.close(stop_spark=True)
        return
    client.close()


def raise_keyboard_interrupt(signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt(f"Received signal {signum}")


def install_shutdown_signal_handlers() -> None:
    signal.signal(signal.SIGINT, raise_keyboard_interrupt)
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)


def find_latest_state_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = [
        path
        for path in checkpoint_dir.glob("*.state.json")
        if path.is_file() and not path.name.endswith(".tmp")
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def find_existing_checkpoint_count(checkpoint_dir: Path, run_stem: str) -> int:
    pattern = re.compile(rf"^{re.escape(run_stem)}_(\d{{3}})_.+\.state\.json$")
    highest = 0
    for path in checkpoint_dir.glob(f"{run_stem}_*.state.json"):
        match = pattern.match(path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest


def infer_run_stem_from_state(state_path: Path, state: dict[str, Any]) -> str:
    run_stem = str(state.get("run_stem") or "").strip()
    if run_stem:
        return run_stem
    graph_json = str(state.get("graph_db_path") or state.get("db_path") or "")
    stem = Path(graph_json).stem if graph_json else state_path.name.removesuffix(".state.json")
    return re.sub(r"_\d{3}_.+$", "", stem)


def load_resume_state(state_path: Path) -> dict[str, Any]:
    state = orjson.loads(state_path.read_text(encoding="utf-8"))
    if int(state.get("version", 0)) != STATE_VERSION:
        raise SystemExit(f"Unsupported checkpoint version in {state_path}")
    return state


def resolve_resume_graph_json(
    state_path: Path,
    state: dict[str, Any],
    graph_dir: Path,
) -> Path:
    from src.graph.resolution import resolve_latest_graph_db_paths

    saved_graph_text = str(state.get("graph_db_path") or state.get("db_path") or "").strip()
    saved_graph_json = Path(saved_graph_text) if saved_graph_text else None
    try:
        latest_graph_json, latest_db, db_resolution = resolve_latest_graph_db_paths(graph_dir)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Checkpoint graph snapshot not found and no Semantic Scholar graph snapshot "
            f"exists in {graph_dir}: {saved_graph_json}"
        ) from exc

    if saved_graph_json is None or saved_graph_json.resolve() != latest_graph_json.resolve():
        logger.warning(
            "Ignoring checkpoint graph path; using latest graph snapshot/database in "
            "--graph-dir | checkpoint={} | checkpoint_graph={} | graph={} | "
            "db={} [{}]",
            state_path,
            saved_graph_json,
            latest_graph_json,
            latest_db,
            db_resolution,
        )
    state["graph_db_path"] = str(latest_db)
    state["db_path"] = str(latest_db)
    write_json(state_path, state)
    return latest_db


def open_sqlite_network(
    working_db: Path,
    run_stem: str,
) -> SQLiteCitationNetwork:
    owner_id = f"{run_stem}:{os.getpid()}:{uuid.uuid4()}"
    network = SQLiteCitationNetwork(
        DatabaseManager(working_db, validate_uuid_values=False),
        owner_id=owner_id,
    )
    logger.info("Using live SQLite graph backend | db={}", working_db)
    return network


def prepare_sqlite_resume_database(
    working_db: Path,
    state: dict[str, Any],
) -> None:
    if working_db.exists():
        return
    saved_path_text = str(
        state.get("working_db_path")
        or state.get("db_path")
        or state.get("graph_db_path")
        or ""
    ).strip()
    if not saved_path_text:
        return
    saved_path = Path(saved_path_text)
    if saved_path.exists() and saved_path.suffix.lower() == ".db":
        copy_database(saved_path, working_db)


def should_use_persistent_workflow(
    args: argparse.Namespace,
    output_path: Path,
    checkpoint_dir: Path,
) -> bool:
    if getattr(args, "extend_from_output", None):
        return True
    if output_path.is_file() and persistent_output_payload(
        read_output_payload(output_path)
    ):
        return True
    requested_checkpoint = str(getattr(args, "resume_checkpoint", None) or "").strip()
    state_path = (
        Path(requested_checkpoint)
        if requested_checkpoint
        else persistent_checkpoint_path(checkpoint_dir)
    )
    if not state_path.is_file():
        return False
    try:
        state = orjson.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, orjson.JSONDecodeError):
        return False
    return isinstance(state, dict) and state.get("run_mode") == RUN_MODE_PERSISTENT


def persistent_target_num_paths(
    requested_target: int | None,
    destination_records: list[PathRecord],
    new_source_records: list[PathRecord],
    seed_titles: list[str],
    paths_per_seed: int,
) -> int:
    if requested_target is not None:
        return requested_target
    return max(
        complete_record_count(destination_records, 1),
        len(destination_records)
        + len(new_source_records)
        + len(seed_titles) * paths_per_seed,
    )


def run_persistent_workflow(
    args: argparse.Namespace,
    graph_dir: Path,
    checkpoint_dir: Path,
    output_path: Path,
    working_db: Path,
) -> None:
    extension_path_text = str(getattr(args, "extend_from_output", None) or "").strip()
    extension_path = Path(extension_path_text) if extension_path_text else None
    if extension_path is not None and extension_path.resolve() == output_path.resolve():
        raise SystemExit("--extend-from-output must differ from the destination output")

    input_order_rng = random.Random(args.random_seed)
    traversal_rng = random.Random(args.random_seed)
    seed_titles = read_seed_titles(args, input_order_rng)

    destination_payload: dict[str, Any] = {}
    destination_records: list[PathRecord] = []
    if output_path.is_file():
        destination_payload, destination_records = load_path_records(
            output_path,
            args.num_hops,
        )

    source_payload: dict[str, Any] | None = None
    source_records: list[PathRecord] = []
    if extension_path is not None:
        if not extension_path.is_file():
            raise SystemExit(f"Extension paths file not found: {extension_path}")
        source_payload, source_records = load_path_records(
            extension_path,
            args.num_hops,
        )

    saved_mode = None
    destination_metadata = destination_payload.get("metadata")
    saved_metadata: dict[str, Any] = {}
    if isinstance(destination_metadata, dict) and destination_records:
        saved_metadata = destination_metadata
        saved_mode = str(
            destination_metadata.get("traversal_mode") or TRAVERSAL_MODE_MIXED
        )
    elif isinstance((source_payload or {}).get("metadata"), dict):
        source_metadata = (source_payload or {})["metadata"]
        saved_metadata = source_metadata
        saved_mode = str(source_metadata.get("traversal_mode") or TRAVERSAL_MODE_MIXED)
    traversal_mode = resolve_traversal_mode(
        getattr(args, "traversal_mode", None),
        saved_mode,
        saved_source="destination output",
    )
    start_date = args.start_date
    end_date = args.end_date
    if start_date is None and saved_metadata.get("start_date"):
        start_date = date.fromisoformat(str(saved_metadata["start_date"]))
    if end_date is None and saved_metadata.get("end_date"):
        end_date = date.fromisoformat(str(saved_metadata["end_date"]))
    if start_date and end_date and start_date > end_date:
        raise SystemExit("--start-date must be on or before --end-date")

    prepare_working_database_from_graph(getattr(args, "load_graph", None), working_db)
    align_path_records_database(
        destination_payload,
        destination_records,
        working_db,
    )
    align_path_records_database(source_payload, source_records, working_db)

    requested_state_path = str(getattr(args, "resume_checkpoint", None) or "").strip()
    state_path = (
        Path(requested_state_path)
        if requested_state_path
        else persistent_checkpoint_path(checkpoint_dir)
    )
    active_job, saved_traversal_rng_state = load_persistent_checkpoint(
        state_path,
        output_path,
    )
    if active_job is not None and saved_traversal_rng_state is not None:
        traversal_rng.setstate(random_state_from_json(saved_traversal_rng_state))

    active_path_id = active_job.path_id if active_job else None
    input_order_rng.shuffle(source_records)
    new_source_records = reconcile_source_records(
        destination_records,
        source_records,
        args.num_hops,
        active_path_id=active_path_id,
    )
    used_seed_keys = {
        seed_title_key(record.seed_title)
        for record in destination_records
        if record.seed_title
    }
    seed_titles = filter_unused_seed_titles(seed_titles, used_seed_keys)
    target_num_paths = persistent_target_num_paths(
        args.target_num_paths,
        destination_records,
        new_source_records,
        seed_titles,
        args.paths_per_seed,
    )

    run_stem = build_graph_stem("semantic_scholar_paths")
    network = open_persistent_network(
        working_db,
        run_stem,
        getattr(args, "graph_backend", GRAPH_BACKEND_NETWORKX),
    )
    semantic_scholar_backend = getattr(
        args,
        "semantic_scholar_backend",
        SEMANTIC_SCHOLAR_BACKEND_ONLINE,
    )
    client = create_semantic_scholar_client(args)
    context = BuildContext(
        network=network,
        client=client,
        graph_dir=graph_dir,
        checkpoint_dir=checkpoint_dir,
        output_path=output_path,
        working_db_path=working_db,
        run_stem=run_stem,
        seed_titles=seed_titles,
        seed_semantic_scholar_ids=read_seed_semantic_scholar_ids(args),
        paths_per_seed=args.paths_per_seed,
        target_num_paths=target_num_paths,
        target_num_nodes=args.num_hops,
        start_date=start_date,
        end_date=end_date,
        traversal_mode=traversal_mode,
        no_tqdm_bar=getattr(args, "no_tqdm_bar", 0),
        bulk_title_search_batch_size=getattr(
            args, "bulk_title_search_batch_size", BULK_TITLE_SEARCH_BATCH_SIZE
        ),
        bulk_title_min_match_score=getattr(
            args, "bulk_title_min_match_score", BULK_TITLE_MIN_MATCH_SCORE
        ),
        checkpoint_every=args.checkpoint_every,
        max_checkpoints=args.max_checkpoints,
        graph_snapshot_every=args.graph_snapshot_every,
        expanded_nodes=set(),
        rng=traversal_rng,
        completed_traces=[],
        completed_path_ids=[],
        last_graph_snapshot_path=working_db if working_db.exists() else None,
    )
    sync_context_completed_records(context, destination_records)

    try:
        validate_path_nodes(
            network,
            destination_records,
            source_label=str(output_path),
        )
        if extension_path is not None:
            validate_path_nodes(
                network,
                source_records,
                source_label=str(extension_path),
            )

        if not destination_records and not new_source_records and not seed_titles:
            raise SystemExit(
                "Provide --extend-from-output, a seed title, or --input-file."
            )

        if active_job is not None:
            active_record = find_path_record(destination_records, active_job.path_id)
            if active_record is None:
                raise SystemExit(
                    "The active checkpoint path is missing from the destination output: "
                    f"{active_job.path_id}"
                )
            active_job.path_index = active_record.path_index
            if len(active_record.trace) >= context.target_num_nodes:
                active_job = None
                save_persistent_progress(
                    context,
                    destination_records,
                    destination_payload,
                    None,
                    traversal_rng,
                    state_path,
                )
            else:
                status = run_active_extension_job(
                    context,
                    destination_records,
                    destination_payload,
                    active_job,
                    traversal_rng,
                    state_path,
                )
                if status == PATH_DEFERRED:
                    logger.warning(
                        "Stopped with an active rate-limited extension | path_id={}",
                        active_job.path_id,
                    )
                    return
                active_job = None

            traversal_rng = random.Random(args.random_seed)
            context.rng = traversal_rng

        incomplete_records = [
            record
            for record in destination_records
            if len(record.trace) < context.target_num_nodes
        ]
        input_order_rng.shuffle(incomplete_records)
        for record in incomplete_records:
            job = create_active_extension_job(
                record,
                traversal_rng,
                context.traversal_mode,
            )
            status = run_active_extension_job(
                context,
                destination_records,
                destination_payload,
                job,
                traversal_rng,
                state_path,
            )
            if status == PATH_DEFERRED:
                return

        for source_record in new_source_records:
            if (
                complete_record_count(destination_records, context.target_num_nodes)
                >= target_num_paths
            ):
                break
            record = append_source_path_record(
                destination_records,
                source_record,
                context.target_num_nodes,
            )
            if len(record.trace) >= context.target_num_nodes:
                save_persistent_progress(
                    context,
                    destination_records,
                    destination_payload,
                    None,
                    traversal_rng,
                    state_path,
                )
                continue
            job = create_active_extension_job(
                record,
                traversal_rng,
                context.traversal_mode,
            )
            status = run_active_extension_job(
                context,
                destination_records,
                destination_payload,
                job,
                traversal_rng,
                state_path,
            )
            if status == PATH_DEFERRED:
                return

        for seed_title in seed_titles:
            for _repeat_index in range(context.paths_per_seed):
                if (
                    complete_record_count(
                        destination_records,
                        context.target_num_nodes,
                    )
                    >= target_num_paths
                ):
                    break
                start_node = resolve_seed_node(context, seed_title)
                if start_node is None:
                    break
                record = PathRecord(
                    path_id=generate_path_id(),
                    path_index=next_record_path_index(destination_records),
                    trace=[(start_node, None)],
                    status=PATH_STATUS_INCOMPLETE,
                    original_node_count=1,
                    seed_title=seed_title,
                )
                destination_records.append(record)
                job = create_active_extension_job(
                    record,
                    traversal_rng,
                    context.traversal_mode,
                )
                status = run_active_extension_job(
                    context,
                    destination_records,
                    destination_payload,
                    job,
                    traversal_rng,
                    state_path,
                )
                if status == PATH_DEFERRED:
                    return

        if not uses_sqlite_backend(context.network):
            save_graph_snapshot(context, write_gexf=False)
        save_persistent_progress(
            context,
            destination_records,
            destination_payload,
            None,
            traversal_rng,
            state_path,
        )
        logger.info(
            "Done | complete paths={}/{} | stored paths={}",
            complete_record_count(destination_records, context.target_num_nodes),
            target_num_paths,
            len(destination_records),
        )
    except KeyboardInterrupt as exc:
        logger.warning("Interrupted persistent path construction: {}", exc)
        raise SystemExit(130) from exc
    finally:
        close_semantic_scholar_client(client, semantic_scholar_backend)
        network.db.close()


def main() -> None:
    args = build_parser().parse_args()
    validate_input_mode(args)
    sqlite_backend = (
        getattr(args, "graph_backend", GRAPH_BACKEND_NETWORKX)
        == GRAPH_BACKEND_SQLITE
    )
    requested_traversal_mode = getattr(args, "traversal_mode", None)
    traversal_mode = resolve_traversal_mode(requested_traversal_mode)
    bootstrap_runtime(args.log_level)
    install_shutdown_signal_handlers()
    load_dotenv_if_available()

    if args.start_date and args.end_date and args.start_date > args.end_date:
        raise SystemExit("--start-date must be on or before --end-date")

    output_dir = Path(args.output_dir)
    graph_dir = Path(args.graph_dir)
    checkpoint_dir = output_dir / "checkpoints"
    graph_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path = resolve_output_path(args.output, output_dir)
    working_db = resolve_working_db_path(args.db_path, graph_dir)
    working_db.parent.mkdir(parents=True, exist_ok=True)

    if should_use_persistent_workflow(args, output_path, checkpoint_dir):
        run_persistent_workflow(
            args,
            graph_dir,
            checkpoint_dir,
            output_path,
            working_db,
        )
        return

    rng = random.Random(args.random_seed)
    run_stem = build_graph_stem("semantic_scholar_paths")
    input_seed_titles = read_seed_titles(args, rng)
    seed_titles = input_seed_titles
    paths_per_seed = args.paths_per_seed
    target_num_paths = args.target_num_paths
    completed_traces: list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]] = []
    completed_path_ids: list[str] = []
    expanded_nodes: set[str] = set()
    resume_frames: list[TravelFrame] | None = None
    pending_checkpoint_frames: list[list[TravelFrame]] = []
    current_seed_index = 0
    current_repeat_index = 0
    failed_seed_max_nodes: dict[int, int] = {}
    completed_output_count: int | None = None
    incomplete_output_count = 0
    last_graph_snapshot_path: Path | None = None
    graph_snapshot_node_ids: set[str] = set()
    graph_snapshot_edge_keys: set[tuple[str, str]] = set()
    last_graph_snapshot_complete_count = 0
    output_payload: dict[str, Any] | None = None
    output_seed_titles: list[str] = []
    output_paths_per_seed: int | None = None
    output_target_num_paths: int | None = None
    output_start_date: date | None = None
    output_end_date: date | None = None
    output_traversal_mode: str | None = None
    output_completed_traces: list[list[tuple[str, tuple[str, str, dict[str, Any]] | None]]] = []
    output_completed_path_ids: list[str] = []
    output_pending_frames: list[list[TravelFrame]] = []
    output_paths = output_path_sources(output_path, checkpoint_dir)

    if output_paths:
        output_payload = read_output_payload(output_paths[0])
        output_traversal_mode = TRAVERSAL_MODE_MIXED
        output_metadata = output_payload.get("metadata")
        if isinstance(output_metadata, dict):
            output_seed_titles = [
                str(title)
                for title in output_metadata.get("seed_titles") or []
                if str(title).strip()
            ]
            raw_paths_per_seed = output_metadata.get("paths_per_seed")
            if raw_paths_per_seed is not None:
                output_paths_per_seed = int(raw_paths_per_seed)
            raw_target_num_paths = output_metadata.get("target_num_paths")
            if raw_target_num_paths is not None:
                output_target_num_paths = int(raw_target_num_paths)
            raw_start_date = str(output_metadata.get("start_date") or "").strip()
            if raw_start_date:
                output_start_date = date.fromisoformat(raw_start_date)
            raw_end_date = str(output_metadata.get("end_date") or "").strip()
            if raw_end_date:
                output_end_date = date.fromisoformat(raw_end_date)
            output_traversal_mode = str(
                output_metadata.get("traversal_mode") or TRAVERSAL_MODE_MIXED
            )
        traversal_mode = resolve_traversal_mode(
            requested_traversal_mode,
            output_traversal_mode,
            saved_source="existing output",
        )
        output_traces, output_path_ids = load_paths_from_outputs(output_paths)
        (
            output_completed_traces,
            output_completed_path_ids,
            output_pending_frames,
            completed_output_count,
            incomplete_output_count,
        ) = split_output_paths(
            output_traces,
            output_path_ids,
            args.num_hops,
        )
        logger.info(
            "Loaded sampled paths from output | completed={} | incomplete={} | outputs={}",
            completed_output_count,
            incomplete_output_count,
            ", ".join(str(path) for path in output_paths),
        )

    if not args.resume_checkpoint and not args.load_graph:
        latest_state = find_latest_state_checkpoint(checkpoint_dir)
        if latest_state is not None:
            args.resume_checkpoint = str(latest_state)
            logger.info(
                "Auto-resuming from latest checkpoint in --output-dir: {}",
                latest_state,
            )

    if not args.resume_checkpoint and output_payload is not None:
        if not input_seed_titles and output_seed_titles:
            seed_titles = output_seed_titles
        if output_paths_per_seed is not None:
            paths_per_seed = output_paths_per_seed
        if target_num_paths is None and output_target_num_paths is not None:
            target_num_paths = output_target_num_paths
        if args.start_date is None and output_start_date is not None:
            args.start_date = output_start_date
        if args.end_date is None and output_end_date is not None:
            args.end_date = output_end_date

    if args.resume_checkpoint:
        state_path = Path(args.resume_checkpoint)
        state = load_resume_state(state_path)
        state_traversal_mode = str(state.get("traversal_mode") or TRAVERSAL_MODE_MIXED)
        if output_traversal_mode is not None and output_traversal_mode != state_traversal_mode:
            raise SystemExit(
                "Checkpoint traversal mode "
                f"{state_traversal_mode!r} does not match existing output mode "
                f"{output_traversal_mode!r}"
            )
        traversal_mode = resolve_traversal_mode(
            requested_traversal_mode,
            state_traversal_mode,
            saved_source="checkpoint",
        )
        run_stem = infer_run_stem_from_state(state_path, state)
        state_seed_titles = list(state.get("seed_titles") or [])
        if not input_seed_titles:
            seed_titles = state_seed_titles
        paths_per_seed = int(state.get("paths_per_seed", paths_per_seed))
        if target_num_paths is None and state.get("target_num_paths") is not None:
            target_num_paths = int(state["target_num_paths"])
        if args.start_date is None and state.get("start_date"):
            args.start_date = date.fromisoformat(state["start_date"])
        if args.end_date is None and state.get("end_date"):
            args.end_date = date.fromisoformat(state["end_date"])
        completed_traces = [deserialize_trace(trace) for trace in state.get("completed_traces", [])]
        completed_path_ids = [
            str(path_id) for path_id in state.get("completed_path_ids", []) if path_id
        ]
        expanded_nodes = set(state.get("expanded_nodes") or [])
        resume_frames = deserialize_frames(state.get("frames") or [])
        (
            completed_traces,
            pending_checkpoint_frames,
            resume_frames,
            short_trace_count,
        ) = normalize_loaded_checkpoint_paths(
            completed_traces,
            resume_frames,
            args.num_hops,
        )
        completed_path_ids = completed_path_ids[: len(completed_traces)]
        resumed_complete_count = len(
            complete_traces_for_target(
                completed_traces,
                args.num_hops,
            )
        )
        if output_paths:
            completed_traces, completed_path_ids = merge_completed_paths(
                output_completed_traces,
                output_completed_path_ids,
                completed_traces,
                completed_path_ids,
            )
            pending_checkpoint_frames = merge_pending_path_frames(
                pending_checkpoint_frames,
                output_pending_frames,
                resume_frames,
                args.num_hops,
            )
        if short_trace_count or pending_checkpoint_frames:
            logger.info(
                "Checkpoint path length check | complete={} | continue={} | target nodes={}",
                len(completed_traces),
                len(pending_checkpoint_frames) + (1 if resume_frames else 0),
                args.num_hops,
            )
        rng.setstate(random_state_from_json(state["rng_state"]))
        current_seed_index = int(state.get("current_seed_index", 0))
        current_repeat_index = int(state.get("current_repeat_index", 0))
        failed_seed_max_nodes = failed_seed_max_nodes_from_state(state)
        if sqlite_backend:
            prepare_sqlite_resume_database(working_db, state)
            network = open_sqlite_network(working_db, run_stem)
            last_graph_snapshot_path = working_db
            graph_snapshot_node_ids, graph_snapshot_edge_keys = set(), set()
        else:
            (
                network,
                last_graph_snapshot_path,
                graph_snapshot_node_ids,
                graph_snapshot_edge_keys,
            ) = load_network_from_resume_state(state_path, state, graph_dir, working_db)
        if completed_output_count is not None and completed_output_count != resumed_complete_count:
            logger.warning(
                "Checkpoint/output completed path mismatch | checkpoint={} | output={} | output_path={}",
                resumed_complete_count,
                completed_output_count,
                output_path,
            )
        elif completed_output_count is not None:
            logger.info(
                "Resume path progress | completed={} | incomplete={}",
                completed_output_count,
                incomplete_output_count,
            )
        last_graph_snapshot_complete_count = int(
            state.get("last_graph_snapshot_complete_count") or 0
        )
        if completed_traces:
            last_graph_snapshot_complete_count = min(
                last_graph_snapshot_complete_count,
                len(completed_traces),
            )
        logger.info(
            "Resumed checkpoint progress | completed paths={}/{}",
            len(completed_traces),
            target_num_paths or len(seed_titles) * paths_per_seed,
        )
        if input_seed_titles:
            used_keys = collect_used_seed_title_keys(
                network,
                state_seed_titles=state_seed_titles,
                current_seed_index=current_seed_index,
                current_repeat_index=current_repeat_index,
                completed_traces=completed_traces,
                pending_frames=pending_checkpoint_frames,
                resume_frames=resume_frames,
            )
            seed_titles = filter_unused_seed_titles(input_seed_titles, used_keys)
            current_seed_index = 0
            current_repeat_index = 0
            if resume_frames:
                pending_checkpoint_frames = merge_pending_path_frames(
                    pending_checkpoint_frames,
                    [resume_frames],
                    target_num_nodes=args.num_hops,
                )
                resume_frames = None
            logger.info(
                "Seed titles shuffled and filtered | input={} used={} remaining={}",
                len(input_seed_titles),
                len(used_keys),
                len(seed_titles),
            )
    elif args.load_graph:
        from src.graph.resolution import resolve_graph_argument

        graph_json = resolve_graph_argument([args.load_graph])
        if sqlite_backend:
            if not working_db.exists():
                from src.graph.resolution import resolve_latest_graph_db_paths

                _graph_json, source_db, _resolution = resolve_latest_graph_db_paths(
                    graph_json
                )
                if source_db.resolve() != working_db.resolve():
                    copy_database(source_db, working_db)
            network = open_sqlite_network(working_db, run_stem)
            last_graph_snapshot_path = working_db
            graph_snapshot_node_ids, graph_snapshot_edge_keys = set(), set()
        else:
            network = load_network_for_mutation(graph_json, working_db)
            last_graph_snapshot_path = graph_json
            graph_snapshot_node_ids, graph_snapshot_edge_keys = capture_graph_base(network)
        if output_paths:
            completed_traces, completed_path_ids = merge_completed_paths(
                output_completed_traces,
                output_completed_path_ids,
                completed_traces,
                completed_path_ids,
            )
            pending_checkpoint_frames = merge_pending_path_frames(
                pending_checkpoint_frames,
                output_pending_frames,
                None,
                args.num_hops,
            )
            if seed_titles:
                seed_input_count = len(seed_titles)
                used_keys = collect_used_seed_title_keys(
                    network,
                    state_seed_titles=[],
                    current_seed_index=0,
                    current_repeat_index=0,
                    completed_traces=completed_traces,
                    pending_frames=pending_checkpoint_frames,
                    resume_frames=None,
                )
                seed_titles = filter_unused_seed_titles(seed_titles, used_keys)
                logger.info(
                    "Seed titles filtered from output resume | input={} used={} remaining={}",
                    seed_input_count,
                    len(used_keys),
                    len(seed_titles),
                )
            logger.info(
                "Resume path progress | completed={} | incomplete={}",
                completed_output_count or 0,
                incomplete_output_count,
            )
    else:
        if output_payload is not None:
            if sqlite_backend:
                network = open_sqlite_network(working_db, run_stem)
                last_graph_snapshot_path = working_db
                graph_snapshot_node_ids, graph_snapshot_edge_keys = set(), set()
            else:
                (
                    network,
                    last_graph_snapshot_path,
                    graph_snapshot_node_ids,
                    graph_snapshot_edge_keys,
                ) = load_network_from_output_payload(output_payload, graph_dir, working_db)
            completed_traces, completed_path_ids = merge_completed_paths(
                output_completed_traces,
                output_completed_path_ids,
                completed_traces,
                completed_path_ids,
            )
            pending_checkpoint_frames = merge_pending_path_frames(
                pending_checkpoint_frames,
                output_pending_frames,
                None,
                args.num_hops,
            )
            if seed_titles:
                seed_input_count = len(seed_titles)
                used_keys = collect_used_seed_title_keys(
                    network,
                    state_seed_titles=[],
                    current_seed_index=0,
                    current_repeat_index=0,
                    completed_traces=completed_traces,
                    pending_frames=pending_checkpoint_frames,
                    resume_frames=None,
                )
                seed_titles = filter_unused_seed_titles(seed_titles, used_keys)
                logger.info(
                    "Seed titles filtered from output resume | input={} used={} remaining={}",
                    seed_input_count,
                    len(used_keys),
                    len(seed_titles),
                )
            logger.info(
                "Resume path progress | completed={} | incomplete={}",
                completed_output_count or 0,
                incomplete_output_count,
            )
        else:
            if sqlite_backend:
                network = open_sqlite_network(working_db, run_stem)
                last_graph_snapshot_path = working_db
            else:
                network = CitationNetwork(DatabaseManager(working_db))
                network.max_depth = None

    if (
        not seed_titles
        and resume_frames is None
        and not pending_checkpoint_frames
        and not completed_traces
    ):
        network.db.close()
        raise SystemExit("Provide at least one seed title or --input-file.")
    semantic_scholar_backend = getattr(
        args,
        "semantic_scholar_backend",
        SEMANTIC_SCHOLAR_BACKEND_ONLINE,
    )
    client = create_semantic_scholar_client(args)
    context = BuildContext(
        network=network,
        client=client,
        graph_dir=graph_dir,
        checkpoint_dir=checkpoint_dir,
        output_path=output_path,
        working_db_path=working_db,
        run_stem=run_stem,
        seed_titles=seed_titles,
        seed_semantic_scholar_ids=read_seed_semantic_scholar_ids(args),
        paths_per_seed=paths_per_seed,
        target_num_paths=target_num_paths,
        target_num_nodes=args.num_hops,
        start_date=args.start_date,
        end_date=args.end_date,
        traversal_mode=traversal_mode,
        no_tqdm_bar=getattr(args, "no_tqdm_bar", 0),
        bulk_title_search_batch_size=getattr(
            args, "bulk_title_search_batch_size", BULK_TITLE_SEARCH_BATCH_SIZE
        ),
        bulk_title_min_match_score=getattr(
            args, "bulk_title_min_match_score", BULK_TITLE_MIN_MATCH_SCORE
        ),
        bulk_seed_timeout_seconds=60
        * getattr(args, "bulk_seed_timeout_minutes", BULK_SEED_TIMEOUT_MINUTES),
        failed_seed_max_nodes=failed_seed_max_nodes,
        checkpoint_every=args.checkpoint_every,
        max_checkpoints=args.max_checkpoints,
        graph_snapshot_every=args.graph_snapshot_every,
        expanded_nodes=expanded_nodes,
        rng=rng,
        completed_traces=completed_traces,
        completed_path_ids=completed_path_ids,
        current_seed_index=current_seed_index,
        current_repeat_index=current_repeat_index,
        checkpoint_count=find_existing_checkpoint_count(checkpoint_dir, run_stem),
        last_graph_snapshot_path=last_graph_snapshot_path,
        graph_snapshot_node_ids=graph_snapshot_node_ids,
        graph_snapshot_edge_keys=graph_snapshot_edge_keys,
        last_graph_snapshot_complete_count=last_graph_snapshot_complete_count,
    )

    try:
        target_was_reached_before_construction = target_paths_reached(context)
        construct_remaining_paths(context, resume_frames, pending_checkpoint_frames)

        (
            final_graph_json,
            skipped_final_graph_snapshot,
        ) = resolve_final_graph_snapshot(
            context,
            target_was_reached_before_construction=(target_was_reached_before_construction),
        )
        result = build_result_payload(
            final_graph_json,
            context.completed_traces,
            context,
            db_path=context.working_db_path if skipped_final_graph_snapshot else None,
        )
        write_json(context.output_path, result)
        remove_legacy_checkpoint_output(checkpoint_dir, context.output_path)
        final_state = checkpoint_dir / f"{run_stem}.final.state.json"
        final_frames = (
            frames_from_trace(context.completed_traces[-1]) if context.completed_traces else []
        )
        write_json(
            final_state,
            build_state_payload(context, final_frames),
        )

        complete_count = complete_path_count(context)
        logger.info("=" * 48)
        logger.info(
            "Done | completed paths={}/{} | {}",
            complete_count,
            target_path_count(context),
            context.network.stats(),
        )
        logger.info("Graph  -> {}", final_graph_json)
        logger.info("DB     -> {}", working_db)
        logger.info("Paths  -> {}", context.output_path)
        logger.info("State  -> {}", final_state)
        logger.info("=" * 48)
    except KeyboardInterrupt as exc:
        checkpoint_path = context.shutdown_checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = save_interrupted_run_state(context, [])
        logger.warning("Interrupted traversal: {}", exc)
        if checkpoint_path is not None:
            logger.warning("Resume from checkpoint -> {}", checkpoint_path)
        raise SystemExit(130) from exc
    finally:
        close_semantic_scholar_client(context.client, semantic_scholar_backend)
        context.network.db.close()


if __name__ == "__main__":
    main()
