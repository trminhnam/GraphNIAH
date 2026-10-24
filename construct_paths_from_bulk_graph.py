#!/usr/bin/env python3
"""Sample papers-only citation paths from the local Semantic Scholar bulk index.

No seed papers are needed. The script loads the whole citation graph among
the papers published in [--start-date, --end-date], removes the papers that
cannot lie on a path of --num-hops papers, and samples paths in memory (see
src/graph/long_path_sampler.py). A path may follow a citation in either
direction, like the ``papers-only`` mode of
construct_paths_from_semantic_scholar.py.

A paper is eligible when its publication date is a complete date inside the
window, its title is not empty and has no MathML markup, and (with
--category cs) it is a Computer Science paper. The rule is applied together
with the date filter, before any citation is read, so no path contains an
ineligible paper (see src/graph/bulk_window_graph.py).

Spark work per run:
  1. Only when the window graph is not cached: one scan of the papers table
     and one scan of the citation table's two id columns. The edge list is
     cached in --graph-cache-dir and reused by later runs with the same
     window, category, and bulk index.
  2. One bucketed read of the papers table for the papers on the new paths.

Path papers are written to --db-path as unexplored papers, with their authors
and the citations between consecutive path papers. Expand them afterwards
with endpoints/expand_path_neighborhood.py, as for the old sampler. The
output has the format of construct_paths_from_semantic_scholar.py; each path
also stores its Semantic Scholar ``corpus_ids``.

Rerunning with the same settings keeps the saved paths and samples only the
missing ones, so --target-num-paths can grow between runs.

Arguments
---------
--num-hops N
    Required number of papers in each path, including the start paper (the
    same meaning as in construct_paths_from_semantic_scholar.py).
--target-num-paths N
    Required number of distinct paths. A path and its reverse count as one.
--start-date DATE
    Required inclusive earliest publication date. Accepts YYYY, YYYY-MM, or
    YYYY-MM-DD.
--end-date DATE
    Optional inclusive latest publication date.
--category {all,cs}
    ``cs`` keeps only papers whose fields of study include Computer Science.
    Default: all.
--output-dir PATH
    Output folder. Default: output/semantic_scholar_paths.
--output PATH
    Paths JSON. Default: <output-dir>/sampled_paths.json.
--db-path PATH
    SQLite graph database that receives the path papers.
    Default: resources/semantic_scholar_graph/semantic_scholar_paths.db.
--bulk-data-root PATH / --bulk-index-root PATH
    Raw Semantic Scholar bulk datasets and their built index.
--graph-cache-dir PATH
    Folder for cached window edge lists. Default: window_graphs/ next to
    --bulk-index-root.
--random-seed INT
    Seed for reproducible start papers and steps.
--max-steps-per-attempt N
    Steps one search may take before it gives up on its start paper for now.
    Default: 100000.
--max-failed-attempts N
    Stop sampling after this many attempts in a row without a new path (a
    failed search or a repeated path). Default: 1000.
--max-paths-per-paper K
    Use each paper in at most K paths of the output file, saved paths
    included, so the paths overlap less. A smaller K allows fewer paths: once
    the most-cited papers are used up, fewer long paths remain. Default: no
    limit.
--broadcast-max-mib MIB
    Memory budget for broadcasting the eligible paper ids to the citation scan.
    Above it, Spark uses a shuffle join. Default: 4096.
--no-tqdm-bar {0,1}
    Set to 1 to disable progress bars. Default: 0.
--log-level {DEBUG,INFO,WARNING,ERROR}
    Console log level. Default: INFO.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import orjson
from loguru import logger
from tqdm import tqdm

from construct_paths_from_semantic_scholar import (
    DEFAULT_PATHS_JSON_NAME,
    RawDefaultsHelpFormatter,
    build_path_entry,
    disable_progress_bars,
    generate_path_id,
    install_shutdown_signal_handlers,
    open_sqlite_network,
    parse_end_date,
    parse_start_date,
    positive_int,
    upsert_semantic_paper,
)
from src.clients.semantic_scholar_bulk_client import (
    DEFAULT_DATA_ROOT as DEFAULT_BULK_DATA_ROOT,
    DEFAULT_INDEX_ROOT as DEFAULT_BULK_INDEX_ROOT,
    DEFAULT_SPARK_DRIVER_MEMORY,
    DEFAULT_SPARK_INPUT_MAX_PARTITION_BYTES,
    DEFAULT_SPARK_MASTER,
    DEFAULT_SPARK_SHUFFLE_PARTITIONS,
    INDEX_METADATA_FILE_NAME,
    SemanticScholarBulkClient,
    import_pyspark_sql,
)
from src.graph.bulk_window_graph import (
    WINDOW_CATEGORIES,
    extract_window_edges,
    load_window_graph_cache,
    save_window_graph_cache,
    window_graph_cache_key,
    window_graph_cache_path,
)
from src.graph.db_network import SQLiteCitationNetwork
from src.graph.long_path_sampler import (
    CitationGraph,
    LongPathSampler,
    PrunedGraph,
    build_citation_graph,
    prune_for_path_length,
)
from src.graph.readers import DBReader
from src.graph.sampled_paths import write_paths_payload_atomic
from src.questions.mathml import title_has_mathml
from src.setup import bootstrap_runtime
from src.setup.constants import EDGE_CITED_BY, EDGE_CITING
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
DEFAULT_DB_PATH = Path("resources") / "semantic_scholar_graph" / "semantic_scholar_paths.db"
SAMPLING_NAME = "bulk_window_bounded_dfs"
SAMPLING_VERSION = 1
TRAVERSAL_MODE = "papers-only"
GRAPH_BACKEND = "bulk_window"
# Saved paths are only extended when these metadata values match the run.
RESUME_SETTING_KEYS = ("sampling", "path_length", "start_date", "end_date", "category")
DEFAULT_MAX_STEPS_PER_ATTEMPT = 100_000
DEFAULT_MAX_FAILED_ATTEMPTS = 1_000
DEFAULT_BROADCAST_MAX_MIB = 4096
# Lets the eligible-id broadcast use the --broadcast-max-mib budget.
SPARK_MAX_RESULT_SIZE = "8g"

EdgeTrace = list[tuple[str, tuple[str, str, dict[str, Any]] | None]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=RawDefaultsHelpFormatter,
    )
    parser.add_argument("--num-hops", type=positive_int, required=True)
    parser.add_argument("--target-num-paths", type=positive_int, required=True)
    parser.add_argument("--start-date", type=parse_start_date, required=True)
    parser.add_argument("--end-date", type=parse_end_date, default=None)
    parser.add_argument("--category", choices=WINDOW_CATEGORIES, default="all")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output", default=None)
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--bulk-data-root", default=str(DEFAULT_BULK_DATA_ROOT))
    parser.add_argument("--bulk-index-root", default=str(DEFAULT_BULK_INDEX_ROOT))
    parser.add_argument("--graph-cache-dir", default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument(
        "--max-steps-per-attempt",
        type=positive_int,
        default=DEFAULT_MAX_STEPS_PER_ATTEMPT,
    )
    parser.add_argument(
        "--max-failed-attempts",
        type=positive_int,
        default=DEFAULT_MAX_FAILED_ATTEMPTS,
    )
    parser.add_argument("--max-paths-per-paper", type=positive_int, default=None)
    parser.add_argument(
        "--broadcast-max-mib",
        type=positive_int,
        default=DEFAULT_BROADCAST_MAX_MIB,
    )
    parser.add_argument("--no-tqdm-bar", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def run_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "sampling": SAMPLING_NAME,
        "path_length": args.num_hops,
        "start_date": args.start_date.isoformat(),
        "end_date": args.end_date.isoformat() if args.end_date else None,
        "category": args.category,
    }


def load_saved_paths(
    output_path: Path,
    settings: dict[str, Any],
    db_path: Path,
) -> list[dict[str, Any]]:
    """Return the paths of an earlier run with the same settings."""
    if not output_path.exists():
        return []
    payload = orjson.loads(output_path.read_bytes())
    metadata = payload.get("metadata") or {}
    differences = [
        f"{key}: saved {metadata.get(key)!r}, requested {settings[key]!r}"
        for key in RESUME_SETTING_KEYS
        if metadata.get(key) != settings[key]
    ]
    saved_db = str(metadata.get("db_path") or "")
    if saved_db and Path(saved_db).resolve() != db_path.resolve():
        differences.append(f"db_path: saved {saved_db!r}, requested {str(db_path)!r}")
    if differences:
        raise SystemExit(
            f"{output_path} was sampled with different settings ("
            + "; ".join(differences)
            + "). Use another --output-dir or --output."
        )
    paths = list(payload.get("paths") or [])
    for index, entry in enumerate(paths):
        corpus_ids = entry.get("corpus_ids")
        if not isinstance(corpus_ids, list) or len(corpus_ids) != settings["path_length"]:
            raise SystemExit(f"{output_path}: paths[{index}] has no valid corpus_ids")
    return paths


def read_index_created_at(index_root: Path) -> str:
    metadata_path = index_root / INDEX_METADATA_FILE_NAME
    if not metadata_path.is_file():
        raise SystemExit(f"Semantic Scholar bulk index not found: {metadata_path}")
    return str(orjson.loads(metadata_path.read_bytes()).get("createdAt") or "")


class BulkSession:
    """Start Spark and the bulk client only when a run needs them."""

    def __init__(self, data_root: Path, index_root: Path) -> None:
        self.data_root = data_root
        self.index_root = index_root
        self.spark: Any = None
        self.bulk_client: SemanticScholarBulkClient | None = None

    def client(self) -> SemanticScholarBulkClient:
        if self.bulk_client is None:
            SparkSession, _, _ = import_pyspark_sql()
            self.spark = (
                SparkSession.builder.appName("bulk_window_paths")
                .master(DEFAULT_SPARK_MASTER)
                .config("spark.driver.memory", DEFAULT_SPARK_DRIVER_MEMORY)
                .config("spark.driver.maxResultSize", SPARK_MAX_RESULT_SIZE)
                .config(
                    "spark.sql.shuffle.partitions",
                    str(DEFAULT_SPARK_SHUFFLE_PARTITIONS),
                )
                .config(
                    "spark.sql.files.maxPartitionBytes",
                    DEFAULT_SPARK_INPUT_MAX_PARTITION_BYTES,
                )
                .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                .getOrCreate()
            )
            self.bulk_client = SemanticScholarBulkClient(
                data_root=self.data_root,
                index_root=self.index_root,
                spark_session=self.spark,
                use_index=True,
            )
        return self.bulk_client

    def close(self) -> None:
        if self.bulk_client is not None:
            self.bulk_client.close()
        if self.spark is not None:
            self.spark.stop()


def load_window_edges(
    args: argparse.Namespace,
    bulk: BulkSession,
    index_created_at: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the window's citation edges, from the cache when it is current."""
    cache_dir = (
        Path(args.graph_cache_dir)
        if args.graph_cache_dir
        else bulk.index_root.parent / "window_graphs"
    )
    cache_path = window_graph_cache_path(
        cache_dir, args.category, args.start_date, args.end_date
    )
    key = window_graph_cache_key(
        index_created_at, args.category, args.start_date, args.end_date
    )
    cached = load_window_graph_cache(cache_path, key)
    if cached is not None:
        logger.info("Loaded cached window graph | {}", cache_path)
        return cached

    client = bulk.client()
    for table in ("papers", "citations_by_citing"):
        if not (bulk.index_root / table / "_SUCCESS").is_file():
            raise SystemExit(
                f"Missing or incomplete bulk index table: {bulk.index_root / table}"
            )
    logger.info(
        "Extracting the citation graph among papers published on or after {}{} "
        "(category={}) with Spark",
        args.start_date,
        f" and on or before {args.end_date}" if args.end_date else "",
        args.category,
    )
    citing, cited, num_eligible = extract_window_edges(
        client.read_index_table("papers"),
        client.read_index_table("citations_by_citing"),
        args.start_date,
        args.end_date,
        args.category,
        broadcast_max_mib=args.broadcast_max_mib,
    )
    metadata = {
        **key,
        "num_eligible_papers": num_eligible,
        "num_citation_rows": len(citing),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_window_graph_cache(cache_path, citing, cited, metadata)
    logger.info("Saved window graph cache | {}", cache_path)
    return citing, cited, metadata


def sample_paths(
    sampler: LongPathSampler,
    count: int,
    *,
    max_failed_attempts: int,
    disable_bar: bool,
) -> list[list[int]]:
    paths: list[list[int]] = []
    with tqdm(
        total=count,
        desc="Sampling paths",
        unit="path",
        miniters=1,
        disable=disable_bar,
    ) as bar:
        for path in sampler.sample(count, max_failed_attempts=max_failed_attempts):
            paths.append(path)
            bar.update(1)
            bar.set_postfix(
                attempts=sampler.stats.attempts,
                repeated=sampler.stats.duplicates,
                refresh=False,
            )
    return paths


def store_path_papers(
    network: SQLiteCitationNetwork,
    corpus_ids: list[int],
    papers_by_corpus_id: dict[int, dict[str, Any]],
    *,
    disable_bar: bool,
) -> tuple[dict[int, str], set[int]]:
    """Upsert the papers in one transaction; return node ids and rejected papers.

    A paper is rejected when the bulk index returns no usable record for it,
    its title has MathML markup, or it shares a database row with another
    path paper (so a path could visit that row twice).
    """
    node_ids: dict[int, str] = {}
    rejected: set[int] = set()
    with network.db.batch(immediate=True):
        for corpus_id in tqdm(
            corpus_ids, desc="Storing path papers", unit="paper", disable=disable_bar
        ):
            paper = papers_by_corpus_id.get(corpus_id)
            if (
                paper is None
                or not paper.get("paperId")
                or title_has_mathml(paper.get("title"))
            ):
                rejected.add(corpus_id)
                continue
            try:
                node_id = upsert_semantic_paper(network, paper)
            except RuntimeError as exc:
                logger.warning("Skipping paper corpusId={}: {}", corpus_id, exc)
                rejected.add(corpus_id)
                continue
            if node_id is None:
                rejected.add(corpus_id)
                continue
            node_ids[corpus_id] = node_id
    shared_rows = {
        node_id for node_id, count in Counter(node_ids.values()).items() if count > 1
    }
    for corpus_id in [c for c, node_id in node_ids.items() if node_id in shared_rows]:
        rejected.add(corpus_id)
        del node_ids[corpus_id]
    return node_ids, rejected


def store_path_citations(
    network: SQLiteCitationNetwork,
    graph: CitationGraph,
    graph_paths: list[list[int]],
    node_ids: dict[int, str],
) -> list[EdgeTrace]:
    """Insert each path's citations and return its node/edge trace."""
    traces: list[EdgeTrace] = []
    corpus_ids = graph.corpus_ids
    with network.db.batch(immediate=True):
        for nodes in graph_paths:
            ids = [node_ids[int(corpus_ids[node])] for node in nodes]
            trace: EdgeTrace = [(ids[0], None)]
            for (source, target), (source_id, target_id) in zip(
                zip(nodes, nodes[1:]), zip(ids, ids[1:])
            ):
                forward = graph.cites(source, target)
                if forward:
                    network.add_citation(source_id, target_id)
                if graph.cites(target, source):
                    network.add_citation(target_id, source_id)
                edge_type = EDGE_CITING if forward else EDGE_CITED_BY
                trace.append((target_id, (source_id, target_id, {"edge_type": edge_type})))
            traces.append(trace)
    return traces


def build_output_payload(
    args: argparse.Namespace,
    entries: list[dict[str, Any]],
    db_path: Path,
    run_info: dict[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "graph_db_path": str(db_path),
        "db_path": str(db_path),
        "working_db_path": str(db_path),
        "target_num_paths": args.target_num_paths,
        "path_length": args.num_hops,
        "requested_n": args.target_num_paths,
        "n": len(entries),
        "complete_paths": len(entries),
        "sampling": SAMPLING_NAME,
        "sampling_version": SAMPLING_VERSION,
        "graph_backend": GRAPH_BACKEND,
        "traversal_mode": TRAVERSAL_MODE,
        "start_date": args.start_date.isoformat(),
        "category": args.category,
        "random_seed": args.random_seed,
        "max_paths_per_paper": args.max_paths_per_paper,
        **run_info,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if args.end_date is not None:
        metadata["end_date"] = args.end_date.isoformat()
    return {
        "metadata": metadata,
        PARAMETERS_FIELD: merge_parameters(
            None,
            **{
                NUM_HOPS_PARAMETER_FIELD: args.num_hops - 1,
                NEEDLE_TYPE_PARAMETER_FIELD: None,
                DISTRACTING_CONTENT_TYPE_PARAMETER_FIELD: None,
                NEEDLE_ORDER_PARAMETER_FIELD: None,
                TARGET_CONTEXT_LENGTH_PARAMETER_FIELD: None,
            },
        ),
        "paths": entries,
    }


def pruned_node_lookup(graph: CitationGraph, pruned: PrunedGraph):
    """Map corpus ids to pruned-graph node indices (None when pruned away)."""
    pruned_corpus_ids = graph.corpus_ids[pruned.nodes]

    def lookup(corpus_ids: list[int]) -> list[int] | None:
        wanted = np.asarray(corpus_ids, dtype=np.int64)
        positions = np.searchsorted(pruned_corpus_ids, wanted)
        positions = np.minimum(positions, max(len(pruned_corpus_ids) - 1, 0))
        if len(pruned_corpus_ids) == 0 or not (
            pruned_corpus_ids[positions] == wanted
        ).all():
            return None
        return positions.tolist()

    return lookup


def sample_and_store(
    args: argparse.Namespace,
    bulk: BulkSession,
    graph: CitationGraph,
    pruned: PrunedGraph,
    sampler: LongPathSampler,
    db_path: Path,
    count: int,
    disable_bar: bool,
) -> tuple[list[EdgeTrace], list[list[int]]]:
    """Sample ``count`` paths whose papers are all stored in the database.

    Papers rejected while storing are blocked in the sampler, and their paths
    are replaced in another round. The other papers of a dropped path keep that
    path in their --max-paths-per-paper count, so the limit only gets stricter.
    """
    traces: list[EdgeTrace] = []
    corpus_paths: list[list[int]] = []
    network = open_sqlite_network(db_path, "bulk_window_paths")
    try:
        while len(traces) < count:
            needed = count - len(traces)
            new_paths = sample_paths(
                sampler,
                needed,
                max_failed_attempts=args.max_failed_attempts,
                disable_bar=disable_bar,
            )
            if not new_paths:
                break
            graph_paths = [pruned.nodes[path].tolist() for path in new_paths]
            wanted = sorted({int(graph.corpus_ids[node]) for path in graph_paths for node in path})
            logger.info("Fetching metadata for {:,} path papers", len(wanted))
            papers = bulk.client().collect_indexed_papers_by_corpus_ids(wanted)
            node_ids, rejected = store_path_papers(
                network, wanted, papers, disable_bar=disable_bar
            )
            kept = [
                path
                for path in graph_paths
                if all(int(graph.corpus_ids[node]) in node_ids for node in path)
            ]
            traces.extend(store_path_citations(network, graph, kept, node_ids))
            corpus_paths.extend(
                [int(graph.corpus_ids[node]) for node in path] for path in kept
            )
            if rejected:
                logger.warning(
                    "Rejected {:,} papers without usable metadata; replacing {:,} paths",
                    len(rejected),
                    len(graph_paths) - len(kept),
                )
                sampler.block_nodes(pruned_node_lookup(graph, pruned)(sorted(rejected)))
            elif len(new_paths) < needed:
                break
    finally:
        network.db.conn.close()
    return traces, corpus_paths


def build_new_entries(
    db_path: Path,
    traces: list[EdgeTrace],
    corpus_paths: list[list[int]],
    first_index: int,
    disable_bar: bool,
) -> list[dict[str, Any]]:
    # Path entries need only in-memory lookups, so skip the on-disk prepared cache.
    db = DBReader(db_path, prepared_cache_dir=None)
    try:
        entries = []
        for offset, (trace, corpus_ids) in enumerate(
            tqdm(
                list(zip(traces, corpus_paths)),
                desc="Building path entries",
                unit="path",
                disable=disable_bar,
            )
        ):
            entry = build_path_entry(db, trace, generate_path_id(), first_index + offset)
            entry["corpus_ids"] = corpus_ids
            entries.append(entry)
        return entries
    finally:
        db.close()


def main() -> None:
    args = build_parser().parse_args()
    if args.num_hops < 2:
        raise SystemExit("--num-hops must be at least 2")
    if args.end_date is not None and args.start_date > args.end_date:
        raise SystemExit("--start-date must be on or before --end-date")
    bootstrap_runtime(args.log_level, setup_litellm_runtime=False)
    install_shutdown_signal_handlers()
    disable_bar = disable_progress_bars(args.no_tqdm_bar)

    output_dir = Path(args.output_dir)
    output_path = Path(args.output) if args.output else output_dir / DEFAULT_PATHS_JSON_NAME
    db_path = Path(args.db_path)
    saved = load_saved_paths(output_path, run_settings(args), db_path)
    missing = args.target_num_paths - len(saved)
    if missing <= 0:
        logger.info(
            "Target already reached | saved={:,} target={:,} | {}",
            len(saved),
            args.target_num_paths,
            output_path,
        )
        return
    if saved:
        logger.info("Extending {:,} saved paths | {}", len(saved), output_path)

    index_root = Path(args.bulk_index_root)
    index_created_at = read_index_created_at(index_root)
    bulk = BulkSession(Path(args.bulk_data_root), index_root)
    try:
        citing, cited, window = load_window_edges(args, bulk, index_created_at)
        graph = build_citation_graph(citing, cited)
        logger.info(
            "Window graph | eligible papers={:,} | papers with a citation={:,} | "
            "citation edges={:,}",
            window["num_eligible_papers"],
            graph.num_nodes,
            graph.num_edges,
        )
        pruned = prune_for_path_length(graph, args.num_hops)
        for round_index, stats in enumerate(pruned.rounds, start=1):
            logger.info(
                "Pruning round {} | papers={:,} edges={:,} blocks={:,} "
                "largest block={:,} removed by bound={:,}",
                round_index,
                stats["nodes"],
                stats["edges"],
                stats["blocks"],
                stats["largest_block"],
                stats["removed_by_bound"],
            )
        sampler = LongPathSampler(
            pruned,
            args.num_hops,
            random.Random(args.random_seed),
            max_steps_per_attempt=args.max_steps_per_attempt,
            max_paths_per_node=args.max_paths_per_paper,
        )
        logger.info(
            "Papers that may lie on a {}-paper path: {:,} | possible start papers: {:,}",
            args.num_hops,
            pruned.num_nodes,
            len(sampler.starts),
        )
        lookup = pruned_node_lookup(graph, pruned)
        for entry in saved:
            nodes = lookup(entry["corpus_ids"])
            if nodes is not None:
                sampler.record_path(nodes)

        traces, corpus_paths = sample_and_store(
            args, bulk, graph, pruned, sampler, db_path, missing, disable_bar
        )
    finally:
        bulk.close()

    stats = sampler.stats
    logger.info(
        "Sampler | attempts={:,} paths={:,} repeated={:,} exhausted starts={:,} "
        "step-limit stops={:,} steps={:,}",
        stats.attempts,
        stats.paths,
        stats.duplicates,
        stats.exhausted_starts,
        stats.step_limit_hits,
        stats.steps,
    )
    if args.max_paths_per_paper is not None:
        # Paths dropped while storing still count here, so this may overstate use.
        logger.info(
            "Paper use | papers at the limit of {} paths: {:,} | most paths per paper: {:,}",
            args.max_paths_per_paper,
            sum(
                count >= args.max_paths_per_paper
                for count in sampler.paths_per_node.values()
            ),
            max(sampler.paths_per_node.values(), default=0),
        )
    if not saved and not traces:
        raise SystemExit(
            f"No path of {args.num_hops} papers was found among the eligible papers."
        )
    entries = saved + build_new_entries(
        db_path, traces, corpus_paths, len(saved), disable_bar
    )
    if len(entries) < args.target_num_paths:
        logger.warning(
            "Found {:,} of {:,} requested paths; the sampler stopped after "
            "{:,} attempts in a row without a new path or ran out of start papers.{}",
            len(entries),
            args.target_num_paths,
            args.max_failed_attempts,
            (
                f" --max-paths-per-paper {args.max_paths_per_paper} limits how "
                "many paths can exist; a larger value allows more."
                if args.max_paths_per_paper is not None
                else ""
            ),
        )
    run_info = {
        "bulk_index_created_at": index_created_at,
        "window_graph": {
            "num_eligible_papers": window["num_eligible_papers"],
            "num_papers_with_citations": graph.num_nodes,
            "num_citation_edges": graph.num_edges,
            "num_papers_after_pruning": pruned.num_nodes,
            "pruning_rounds": pruned.rounds,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_paths_payload_atomic(
        output_path, build_output_payload(args, entries, db_path, run_info)
    )
    logger.info("Saved {:,} paths | {}", len(entries), output_path)


if __name__ == "__main__":
    main()
