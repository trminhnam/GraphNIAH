#!/usr/bin/env python3
"""
sample_paths.py — Random Walk Path Sampler for Citation Networks
================================================================

Samples ``n`` random-walk paths up to a requested node count from a citation
graph previously built by ``create_graph.py``.

Each path targets *path_length* nodes visited by following edges
(authorship or citation). In ``random`` mode the sampler picks an edge type
first, then a concrete neighbour, and backtracks when a branch dead-ends
before the requested length so earlier nodes can try other branches. If no
eligible extension exists, the returned path can be shorter than *path_length*.
When ``seed`` is omitted in ``random`` mode, start nodes that still cannot
reach ``path_length`` after backtracking are removed from the start-node pool
instead of being returned, and sampling stops early with a warning if that
pool is exhausted.
The output format mirrors the ``walk_path_*.json`` files produced by the
walking scripts, with every node's full DB attributes included at each step.

Database resolution order
-------------------------
The SQLite database is located automatically from the graph JSON; an explicit
path is only needed when the auto-detection would pick the wrong file.

  1. ``--db-path`` / ``db_path`` argument provided → use it directly.
  2. ``<json_stem>.db`` exists alongside the JSON → use it (exact match).
  3. ``_meta.db_path`` in the JSON is set and the path exists on disk → use it.
  4. A newer checkpoint ``.db`` sharing the same base stem exists in the same
     folder as the JSON → use it with a printed warning.
  5. Nothing found → fall back to ``<json_stem>.db`` and raise
     ``FileNotFoundError`` if that file is absent.

Search priority for the ``seed`` starting node
-----------------------------------------------
  1. Exact arXiv-ID match  (``seed`` is treated as an arXiv ID)
  2. Exact normalised title match  (title index)
  3. Exact normalised author match (author index)
  4. Aggressive-normalised title match  (strips articles, hyphens)
  5. Fuzzy title match  (shared string matcher, threshold 0.80)
  6. Fuzzy author match (shared string matcher, threshold 0.80)
  7. Phrase-level token search — titles then authors
  8. Random fallback  (no match found)

``same_start`` controls whether every path in a batch begins at the same
resolved start node (True) or each path independently picks a random start
(False, default when seed is None).

When ``seed`` is omitted and the database contains paper rows with ``depth=0``,
random starts are sampled from those root papers instead of from every node in
the graph. If no such papers are present in the graph, sampling falls back to
the old all-node behavior. In ``random`` mode, any start node that cannot
produce a full-length path is removed from that pool. When ``cutoff_date`` is
provided, every sampled paper node must be published on or after that date.

Usage (CLI)
-----------

python sample_paths.py \
    --graph ./output/graphs/logitscope \
    --seed "LogitScope: A Framework for Analyzing LLM Uncertainty Through Information Metrics" \
    --same-start \
    --path-length 3 \
    --n 1000000 \
    --sampling-style all \
    --random-seed 1 \
    --output ./output/graphs/logitscope/walks/logitscope_paths_all.json

  # DB resolved automatically from the JSON stem
  python sample_paths.py \\
      --graph GPT-4_Technical_Report_20260320_100223.json \\
      --n 5 --path-length 8 \\
      --seed "2303.08774v6"

  # Override DB explicitly
  python sample_paths.py \\
      --graph graph.json --db-path /data/my.db \\
      --n 10 --path-length 6 --output sampled_paths.json

  # Same start, reproducible
  python sample_paths.py \\
      --graph graph.json \\
      --n 4 --path-length 5 \\
      --seed "Attention Is All You Need" --same-start --random-seed 42
"""

from __future__ import annotations

import argparse
import orjson
import random
import re
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from src.graph.readers import CitationGraphLoader, DBReader, build_node_payload
from src.graph.resolution import resolve_graph_argument, resolve_graph_db_paths
from src.setup.pipeline_parameters import (
    DISTRACTING_CONTENT_TYPE_PARAMETER_FIELD,
    NEEDLE_ORDER_PARAMETER_FIELD,
    NEEDLE_TYPE_PARAMETER_FIELD,
    PARAMETERS_FIELD,
    TARGET_CONTEXT_LENGTH_PARAMETER_FIELD,
    merge_parameters,
    NUM_HOPS_PARAMETER_FIELD,
)
from src.setup.constants import (
    EDGE_AUTHORSHIP,
    EDGE_CITED_BY,
    EDGE_CITING,
    normalize_text,
    normalize_text_aggressive,
    nid_db_id,
    nid_type,
)
from src.utils.string_matching import search_strings


# ─────────────────────────────────────────────────────────────────────────────
# Seed resolution
# ─────────────────────────────────────────────────────────────────────────────


def find_fuzzy_best(
    query_norm: str,
    index: dict[str, int],
    threshold: float = 0.80,
) -> int | None:
    """Return the DB id of the best fuzzy match in *index*, or None."""
    candidates = list(index.keys())
    match_index = search_strings(query_norm, candidates, threshold)
    if match_index < 0:
        return None
    return index[candidates[match_index]]


def token_search(
    seed: str,
    index: dict[str, int],
) -> int | None:
    """Return the first DB id whose index key contains any token from *seed*."""
    tokens = [t for t in re.split(r"[\s_\-]+", seed.lower()) if len(t) > 2]
    for key, db_id in index.items():
        if any(tok in key for tok in tokens):
            return db_id
    return None


def try_resolve(
    db_id: int | None,
    node_type: str,
    graph: CitationGraphLoader,
) -> str | None:
    """Return ``"<node_type>:<db_id>"`` if *db_id* is set and the node exists."""
    if db_id is None:
        return None
    nid = f"{node_type}:{db_id}"
    return nid if nid in graph.nodes else None


def resolve_seed(
    seed: str,
    graph: CitationGraphLoader,
) -> tuple[str | None, str]:
    """Resolve *seed* to a node ID using the priority chain.

    Returns (node_id_or_None, resolution_method_string).
    """
    seed = seed.strip()
    norm = normalize_text(seed)
    agg = normalize_text_aggressive(seed)

    # (db_id_or_None, node_type, label)  — evaluated lazily via short-circuit
    strategies: list[tuple[int | None, str, str]] = [
        (graph.arxiv_index.get(seed), "paper", "exact_arxiv_id"),
        (graph.paper_index.get(norm), "paper", "exact_title"),
        (graph.author_index.get(norm), "author", "exact_author"),
        (graph.paper_agg_index.get(agg), "paper", "aggressive_title"),
        (find_fuzzy_best(norm, graph.paper_index), "paper", "fuzzy_title"),
        (find_fuzzy_best(norm, graph.author_index), "author", "fuzzy_author"),
        (token_search(norm, graph.paper_index), "paper", "token_search_title"),
        (token_search(norm, graph.author_index), "author", "token_search_author"),
    ]

    for db_id, node_type, label in strategies:
        nid = try_resolve(db_id, node_type, graph)
        if nid is not None:
            return nid, label

    return None, "random_fallback"


# ─────────────────────────────────────────────────────────────────────────────
# Path sampling
# ─────────────────────────────────────────────────────────────────────────────

# Order of edge types for CLI weight parsing: citing, cited_by, authorship
EDGE_WEIGHT_ORDER = (EDGE_CITING, EDGE_CITED_BY, EDGE_AUTHORSHIP)


@dataclass
class PathFrame:
    node_id: str
    incoming_edge: tuple[str, str, dict[str, Any]] | None = None
    tried_targets_by_type: dict[str, set[str]] = field(default_factory=dict)


def parse_cutoff_date_input(raw: str) -> date:
    """Parse ``YYYY`` / ``YYYY-MM`` / ``YYYY-MM-DD`` into a concrete date."""
    text = raw.strip()
    if re.fullmatch(r"\d{4}", text):
        text = f"{text}-01-01"
    elif re.fullmatch(r"\d{4}-\d{2}", text):
        text = f"{text}-01"
    elif not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ValueError("cut-off date must use YYYY, YYYY-MM, or YYYY-MM-DD format")

    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"invalid cut-off date: {raw!r}") from exc


def resolve_cutoff_date(
    cutoff_date: str | date | None,
) -> tuple[date | None, str | None]:
    """Return the parsed cutoff date and its normalized ISO string."""
    if cutoff_date is None:
        return None, None
    if isinstance(cutoff_date, date):
        return cutoff_date, cutoff_date.isoformat()
    parsed = parse_cutoff_date_input(cutoff_date)
    return parsed, parsed.isoformat()


def parse_published_date_value(raw: str) -> date | None:
    """Parse a paper ``published`` field into a date."""
    text = raw.strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    if len(text) >= 10:
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def paper_node_allowed(
    node_id: str,
    db: DBReader,
    cutoff_date: date | None,
    published_date_cache: dict[int, date | None],
) -> bool:
    """Return True when a paper node is explored and satisfies the cut-off."""
    if nid_type(node_id) != "paper":
        return True

    paper_db_id = nid_db_id(node_id)
    if not db.get_paper_explored(paper_db_id):
        return False
    if cutoff_date is None:
        return True

    if paper_db_id not in published_date_cache:
        published_date_cache[paper_db_id] = parse_published_date_value(
            db.get_paper_published(paper_db_id)
        )
    published_date = published_date_cache[paper_db_id]
    return published_date is not None and published_date >= cutoff_date


def filter_eligible_start_nodes(
    node_ids: list[str],
    db: DBReader,
    cutoff_date: date | None,
    published_date_cache: dict[int, date | None],
) -> list[str]:
    """Return the start nodes that satisfy the current sampling rules."""
    return [
        node_id
        for node_id in node_ids
        if paper_node_allowed(node_id, db, cutoff_date, published_date_cache)
    ]


def candidate_allowed(
    next_nid: str,
    edge_type: str,
    visited: set[str],
    db: DBReader,
    cutoff_date: date | None,
    published_date_cache: dict[int, date | None],
) -> bool:
    """Return True when a candidate node is eligible for traversal."""
    del edge_type
    if next_nid in visited:
        return False
    return paper_node_allowed(next_nid, db, cutoff_date, published_date_cache)


def choose_edge_type(
    available_types: list[str],
    rng: random.Random,
    edge_weights: dict[str, float] | None = None,
) -> str:
    """Choose one eligible edge type."""
    if len(available_types) == 1:
        return available_types[0]

    if edge_weights is None:
        return rng.choice(available_types)

    weights = [edge_weights.get(edge_type, 1.0) for edge_type in available_types]
    if sum(weights) <= 0:
        return rng.choice(available_types)
    return rng.choices(available_types, weights=weights, k=1)[0]


def collect_available_edge_types(
    current_nid: str,
    graph: CitationGraphLoader,
    db: DBReader,
    visited: set[str],
    cutoff_date: date | None,
    published_date_cache: dict[int, date | None],
    tried_targets_by_type: dict[str, set[str]],
) -> list[str]:
    """Return edge types that still have at least one eligible next node."""
    available_types: list[str] = []
    seen_types: set[str] = set()

    for next_nid, edge_attrs in graph.adj.get(current_nid, []):
        edge_type = edge_attrs.get("edge_type", "")
        if edge_type not in EDGE_WEIGHT_ORDER or edge_type in seen_types:
            continue
        if next_nid in tried_targets_by_type.get(edge_type, set()):
            continue
        if not candidate_allowed(
            next_nid,
            edge_type,
            visited,
            db,
            cutoff_date,
            published_date_cache,
        ):
            continue
        available_types.append(edge_type)
        seen_types.add(edge_type)

    return available_types


def collect_eligible_candidates_for_type(
    current_nid: str,
    edge_type: str,
    graph: CitationGraphLoader,
    db: DBReader,
    visited: set[str],
    cutoff_date: date | None,
    published_date_cache: dict[int, date | None],
    tried_targets_by_type: dict[str, set[str]],
) -> list[tuple[str, dict[str, Any]]]:
    """Return all eligible next-node candidates for one edge type."""
    candidates: list[tuple[str, dict[str, Any]]] = []
    tried_targets = tried_targets_by_type.get(edge_type, set())

    for next_nid, edge_attrs in graph.adj.get(current_nid, []):
        if edge_attrs.get("edge_type", "") != edge_type:
            continue
        if next_nid in tried_targets:
            continue
        if not candidate_allowed(
            next_nid,
            edge_type,
            visited,
            db,
            cutoff_date,
            published_date_cache,
        ):
            continue
        candidates.append((next_nid, edge_attrs))

    return candidates


def collect_eligible_edges(
    current_nid: str,
    graph: CitationGraphLoader,
    db: DBReader,
    visited: set[str],
    cutoff_date: date | None,
    published_date_cache: dict[int, date | None],
) -> list[tuple[str, dict[str, Any]]]:
    """Return every eligible outgoing edge from the current node."""
    eligible: list[tuple[str, dict[str, Any]]] = []
    for next_nid, edge_attrs in graph.adj.get(current_nid, []):
        edge_type = edge_attrs.get("edge_type", "")
        if edge_type not in EDGE_WEIGHT_ORDER:
            continue
        if not candidate_allowed(
            next_nid,
            edge_type,
            visited,
            db,
            cutoff_date,
            published_date_cache,
        ):
            continue
        eligible.append((next_nid, edge_attrs))
    return eligible


def build_edge_record(
    from_nid: str,
    to_nid: str,
    edge_attrs: dict,
) -> dict[str, Any]:
    """Build the edge record dict shared by all walk output formats."""
    edge_type = edge_attrs.get("edge_type", "unknown")
    record: dict[str, Any] = {
        "from": from_nid,
        "to": to_nid,
        "edge_type": edge_type,
        "traversal": f"{nid_type(from_nid)} -[{edge_type}]-> {nid_type(to_nid)}",
    }
    if "position" in edge_attrs:
        record["position"] = edge_attrs["position"]
    return record


def build_step(
    step_idx: int,
    nid: str,
    graph: CitationGraphLoader | None,
    db: DBReader,
    edge: tuple[str, str, dict] | None = None,
) -> dict[str, Any]:
    """Build a single step dict.

    *edge* is ``(from_nid, to_nid, edge_attrs)`` for walk steps, or
    ``None`` for the start step.
    """
    return {
        "step": step_idx,
        "node": build_node_payload(nid, graph, db),
        "edge_taken": build_edge_record(*edge) if edge else None,
        "reason": "start" if edge is None else "walk",
    }


def snapshot_path_frames(
    frames: deque[PathFrame],
) -> list[tuple[str, tuple[str, str, dict[str, Any]] | None]]:
    """Capture the current path without copying traversal bookkeeping."""
    return [(frame.node_id, frame.incoming_edge) for frame in frames]


def build_steps_from_trace(
    trace: list[tuple[str, tuple[str, str, dict[str, Any]] | None]],
    graph: CitationGraphLoader | None,
    db: DBReader,
) -> list[dict[str, Any]]:
    """Build the JSON-friendly step list from a node/edge trace."""
    return [
        build_step(step_idx, nid, graph, db, edge)
        for step_idx, (nid, edge) in enumerate(trace)
    ]


def enumerate_all_paths(
    start_nid: str,
    path_length: int,
    graph: CitationGraphLoader,
    db: DBReader,
    cutoff_date: date | None = None,
    published_date_cache: dict[int, date | None] | None = None,
) -> list[list[dict[str, Any]]]:
    """Return all simple paths of up to *path_length* nodes from *start_nid*.

    Uses iterative DFS with backtracking.  Each node appears at most once per
    path (no revisiting).  Paths that reach a dead-end before *path_length*
    nodes are included as shorter paths.
    """
    all_paths: list[list[dict[str, Any]]] = []
    if published_date_cache is None:
        published_date_cache = {}
    if not paper_node_allowed(start_nid, db, cutoff_date, published_date_cache):
        return all_paths

    # Stack: (current_nid, visited, steps, edge_from_parent_or_None)
    stack: list[tuple[str, set[str], list[dict[str, Any]], tuple | None]] = [
        (start_nid, {start_nid}, [], None)
    ]

    while stack:
        current, visited, steps, edge = stack.pop()
        step = build_step(len(steps), current, graph, db, edge)
        new_steps = steps + [step]

        if len(new_steps) >= path_length:
            all_paths.append(new_steps)
            continue

        eligible_edges = collect_eligible_edges(
            current,
            graph,
            db,
            visited,
            cutoff_date,
            published_date_cache,
        )
        if not eligible_edges:
            all_paths.append(new_steps)
            continue

        for next_nid, next_edge in eligible_edges:
            stack.append(
                (
                    next_nid,
                    visited | {next_nid},
                    new_steps,
                    (current, next_nid, next_edge),
                )
            )

    return all_paths


def sample_single_path(
    start_nid: str,
    path_length: int,
    graph: CitationGraphLoader,
    db: DBReader,
    rng: random.Random,
    edge_weights: dict[str, float] | None = None,
    cutoff_date: date | None = None,
    published_date_cache: dict[int, date | None] | None = None,
) -> list[dict[str, Any]]:
    """Sample one path, backtracking when a chosen branch dead-ends."""
    if published_date_cache is None:
        published_date_cache = {}
    if not paper_node_allowed(start_nid, db, cutoff_date, published_date_cache):
        return []

    frames: deque[PathFrame] = deque([PathFrame(node_id=start_nid)])
    visited: set[str] = {start_nid}
    best_trace = snapshot_path_frames(frames)

    while frames:
        if len(frames) > len(best_trace):
            best_trace = snapshot_path_frames(frames)
        if len(frames) >= path_length:
            return build_steps_from_trace(snapshot_path_frames(frames), graph, db)

        current = frames[-1]
        available_types = collect_available_edge_types(
            current.node_id,
            graph,
            db,
            visited,
            cutoff_date,
            published_date_cache,
            current.tried_targets_by_type,
        )
        if not available_types:
            if len(frames) == 1:
                break
            abandoned = frames.pop()
            visited.remove(abandoned.node_id)
            continue

        chosen_type = choose_edge_type(available_types, rng, edge_weights)
        candidates = collect_eligible_candidates_for_type(
            current.node_id,
            chosen_type,
            graph,
            db,
            visited,
            cutoff_date,
            published_date_cache,
            current.tried_targets_by_type,
        )
        if not candidates:
            continue

        next_nid, next_edge = rng.choice(candidates)
        current.tried_targets_by_type.setdefault(chosen_type, set()).add(next_nid)
        visited.add(next_nid)
        frames.append(
            PathFrame(
                node_id=next_nid,
                incoming_edge=(current.node_id, next_nid, next_edge),
            )
        )

    return build_steps_from_trace(best_trace, graph, db)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def sample_paths(
    graph_json: str | Path | list[str | Path] | tuple[str | Path, ...],
    n: int,
    path_length: int,
    db_path: str | Path | None = None,
    seed: str | None = None,
    same_start: bool = False,
    random_seed: int | None = None,
    edge_weights: dict[str, float] | None = None,
    sampling_style: str = "random",
    cutoff_date: str | date | None = None,
) -> dict[str, Any]:
    """
    Sample *n* random-walk paths with a target length of *path_length* nodes.

    Parameters
    ----------
    graph_json   : Path to the graph JSON, a graphs directory, or an output
                   root produced by create_graph.py.
    n            : Requested number of paths to sample.
    path_length  : Target number of nodes in each path (including the start node).
                   Dead ends can produce shorter paths.
    db_path      : Optional explicit path to the accompanying ``*.db`` SQLite
                   file.  When omitted the database is located automatically
                   via :func:`resolve_db_path` (see module docstring for the
                   full resolution order).
    seed         : Optional query string used to determine the start node.
                   Accepts: arXiv ID, paper title, author name, free-text phrase.
                   Resolved via the priority chain described in the module docstring.
                   ``None`` → random paper whose DB row has ``depth=0`` when
                   available in the graph; otherwise random graph node. In
                   ``random`` mode, start nodes that cannot reach
                   ``path_length`` are removed from the pool and the function
                   may return fewer than *n* paths if the pool is exhausted.
    same_start   : If True, all *n* paths share the same resolved start node.
                   Ignored when ``seed`` is None.
    random_seed  : Optional integer seed for reproducibility.
    sampling_style : ``"random"`` (default) samples *n* random walks.
                     ``"all"`` enumerates every simple path up to *path_length*
                     nodes from the start node, including shorter dead-end paths
                     (``n`` is ignored).
    cutoff_date  : Optional minimum published date for paper nodes. When
                   provided, every sampled paper node must be published on or
                   after this date. Accepts ``YYYY``, ``YYYY-MM``, or
                   ``YYYY-MM-DD``; partial inputs are normalized to the first
                   day of the specified period.

    Returns
    -------
    A dictionary with:
      ``paths``         – list of path objects (each is a list of step dicts)
      ``metadata``      – sampling parameters and resolution info
      ``sampled_at``    – ISO-8601 timestamp
    """
    if sampling_style not in ("random", "all"):
        raise ValueError(
            f"sampling_style must be 'random' or 'all', got {sampling_style!r}"
        )
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if path_length < 1:
        raise ValueError(f"path_length must be >= 1, got {path_length}")

    graph_json, resolved_db, db_resolution = resolve_graph_db_paths(
        graph_json,
        db_path,
    )
    if db_resolution == "newer_checkpoint":
        warnings.warn(
            f"No DB found for '{graph_json.stem}.db'; using newer checkpoint '{resolved_db.name}'. Pass --db-path to suppress this warning.",
            UserWarning,
            stacklevel=2,
        )

    rng = random.Random(random_seed)
    graph = CitationGraphLoader(graph_json)
    db = DBReader(resolved_db)
    cutoff_date_value, cutoff_date_text = resolve_cutoff_date(cutoff_date)
    published_date_cache: dict[int, date | None] = {}
    all_nodes = filter_eligible_start_nodes(
        list(graph.nodes),
        db,
        cutoff_date_value,
        published_date_cache,
    )
    db_root_start_nodes = [
        node_id
        for node_id in db.get_paper_node_ids_by_depth(0)
        if node_id in graph.nodes
    ]
    root_start_nodes = filter_eligible_start_nodes(
        db_root_start_nodes,
        db,
        cutoff_date_value,
        published_date_cache,
    )
    if db_root_start_nodes:
        random_start_nodes = root_start_nodes
        start_pool_name = "root_papers"
    else:
        random_start_nodes = all_nodes
        start_pool_name = "all_nodes"

    # ── Resolve start node ────────────────────────────────────────────────────
    resolved_start: str | None = None
    resolution_method: str = "none"

    if seed is not None:
        resolved_start, resolution_method = resolve_seed(seed, graph)
        if resolved_start is None:
            if not all_nodes:
                raise ValueError("No eligible start nodes are available for sampling.")
            resolved_start = rng.choice(all_nodes)
            resolution_method = "random_fallback"
        elif not paper_node_allowed(
            resolved_start,
            db,
            cutoff_date_value,
            published_date_cache,
        ):
            if cutoff_date_text is not None:
                raise ValueError(
                    "Resolved seed node "
                    f"{resolved_start!r} does not satisfy the cut-off date "
                    f"{cutoff_date_text}."
                )
            raise ValueError(
                f"Resolved seed node {resolved_start!r} is not eligible for sampling."
            )
    elif root_start_nodes:
        resolution_method = "root_papers"

    # ── Collect (start_nid, steps) pairs ──────────────────────────────────────
    raw_paths: list[tuple[str, list[dict[str, Any]]]] = []

    if sampling_style == "all":
        if resolved_start is not None:
            start = resolved_start
        elif random_start_nodes:
            start = rng.choice(random_start_nodes)
        else:
            start = None

        if start is not None:
            for steps in enumerate_all_paths(
                start,
                path_length,
                graph,
                db,
                cutoff_date_value,
                published_date_cache,
            ):
                raw_paths.append((start, steps))
    else:
        if seed is not None:
            path_sub_seeds = [rng.randint(0, 2**31 - 1) for _ in range(n)]
            for path_idx in range(n):
                path_rng = random.Random(path_sub_seeds[path_idx])
                start = resolved_start
                steps = sample_single_path(
                    start,
                    path_length,
                    graph,
                    db,
                    path_rng,
                    edge_weights,
                    cutoff_date_value,
                    published_date_cache,
                )
                raw_paths.append((start, steps))
        else:
            available_start_nodes = list(random_start_nodes)
            while len(raw_paths) < n:
                if not available_start_nodes:
                    warnings.warn(
                        (
                            "Sampling stopped early after "
                            f"{len(raw_paths)} of {n} requested path(s): no "
                            "remaining start nodes in the "
                            f"{start_pool_name} pool can reach path length "
                            f"{path_length}."
                        ),
                        UserWarning,
                        stacklevel=2,
                    )
                    break

                path_rng = random.Random(rng.randint(0, 2**31 - 1))
                start = path_rng.choice(available_start_nodes)
                steps = sample_single_path(
                    start,
                    path_length,
                    graph,
                    db,
                    path_rng,
                    edge_weights,
                    cutoff_date_value,
                    published_date_cache,
                )
                if len(steps) < path_length:
                    available_start_nodes.remove(start)
                    continue
                raw_paths.append((start, steps))

    db.close()

    paths_output = [
        {"path_index": i, "start_node": s, "moves": len(p) - 1, "path": p}
        for i, (s, p) in enumerate(raw_paths)
    ]

    metadata = {
        "graph_json": str(graph_json),
        "db_path": str(resolved_db),
        "db_resolution": db_resolution,
        "requested_n": n,
        "n": len(paths_output),
        "sampling_style": sampling_style,
        "path_length": path_length,
        "start_pool": ("seed" if seed is not None else start_pool_name),
    }
    if seed is not None:
        metadata["seed"] = seed
    if same_start:
        metadata["same_start"] = same_start
    if random_seed is not None:
        metadata["random_seed"] = random_seed
    if edge_weights is not None:
        metadata["edge_weights"] = edge_weights
    if cutoff_date_text is not None:
        metadata["cutoff_date"] = cutoff_date_text
    if resolved_start is not None:
        metadata["resolved_start_node"] = resolved_start

    return {
        "metadata": metadata,
        PARAMETERS_FIELD: merge_parameters(
            None,
            **{
                NUM_HOPS_PARAMETER_FIELD: max(path_length - 1, 0),
                NEEDLE_TYPE_PARAMETER_FIELD: None,
                DISTRACTING_CONTENT_TYPE_PARAMETER_FIELD: None,
                NEEDLE_ORDER_PARAMETER_FIELD: None,
                TARGET_CONTEXT_LENGTH_PARAMETER_FIELD: None,
            },
        ),
        "paths": paths_output,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sample_paths",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--graph",
        required=True,
        nargs="+",
        metavar="JSON",
        help=(
            "Path to the graph JSON, a graphs directory, or an output root. "
            "When multiple files are given (e.g. via a shell glob *.json), "
            "the newest file by modification time is selected automatically."
        ),
    )
    p.add_argument(
        "--db-path",
        default=None,
        metavar="DB",
        help=(
            "Path to the SQLite database file. Optional — when omitted the DB is located automatically (see module docstring for resolution order)."
        ),
    )
    p.add_argument(
        "--n",
        type=int,
        default=5,
        metavar="N",
        help="Requested number of paths to sample (default: 5).",
    )
    p.add_argument(
        "--path-length",
        type=int,
        default=8,
        metavar="L",
        help=(
            "Target number of nodes per path, including the start node. "
            "Dead ends can produce shorter paths (default: 8)."
        ),
    )
    p.add_argument(
        "--seed",
        default=None,
        metavar="QUERY",
        help=(
            "Start-node query: arXiv ID, paper title, author name, or free-text phrase. "
            "Omit to sample starts from DB papers with depth=0 when available in the graph; otherwise all graph nodes. "
            "In random mode, starts that dead-end before path-length are removed from that pool."
        ),
    )
    p.add_argument(
        "--same-start",
        action="store_true",
        help="All n paths start from the same resolved start node.",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=None,
        metavar="INT",
        help="Integer seed for the random number generator (reproducibility).",
    )
    p.add_argument(
        "--edge-sampling-weights",
        default=None,
        metavar="W1,W2,W3",
        help=(
            "Comma-separated sampling weights for edge types in the order: citing, cited_by, authorship. Example: '2.0,1.0,0.5' makes citing edges twice as likely as cited_by and four times as likely as authorship. Default: uniform (all types equally likely)."
        ),
    )
    p.add_argument(
        "--sampling-style",
        choices=["random", "all"],
        default="random",
        help=(
            "Sampling strategy: 'random' samples n random walks (default), 'all' enumerates every simple path of the given length from the start node (n is ignored)."
        ),
    )
    p.add_argument(
        "--cut-off-date",
        "--cutoff-date",
        "--cutoff_date",
        dest="cutoff_date",
        type=parse_cutoff_date_input,
        default=None,
        metavar="DATE",
        help=(
            "Require sampled paper nodes to be published on or after this date. Accepted formats: YYYY, YYYY-MM, YYYY-MM-DD. Partial inputs are normalized to the first day of that period."
        ),
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Write the JSON result to this file (default: print to stdout).",
    )
    p.add_argument(
        "--indent",
        type=int,
        default=2,
        metavar="N",
        help="JSON indentation width (default: 2).",
    )
    return p


def parse_edge_weights(raw: str) -> dict[str, float]:
    """Parse a ``w1,w2,w3`` string into ``{edge_type: weight}``."""
    parts = [x.strip() for x in raw.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected 3 comma-separated weights (citing,cited_by,authorship), got {len(parts)}"
        )
    return dict(zip(EDGE_WEIGHT_ORDER, (float(p) for p in parts)))


def main() -> None:
    args = build_parser().parse_args()
    graph_json = resolve_graph_argument(args.graph)
    if len(args.graph) > 1:
        print(
            "\n[sample_paths]"
            f"\n  Graph inputs   : {len(args.graph)} matches"
            f"\n  Using newest   : {graph_json.name}"
        )

    edge_weights = (
        parse_edge_weights(args.edge_sampling_weights)
        if args.edge_sampling_weights
        else None
    )

    result = sample_paths(
        graph_json=graph_json,
        db_path=args.db_path,
        n=args.n,
        path_length=args.path_length,
        seed=args.seed,
        same_start=args.same_start,
        random_seed=args.random_seed,
        edge_weights=edge_weights,
        sampling_style=args.sampling_style,
        cutoff_date=args.cutoff_date,
    )

    output_json = orjson.dumps(result, option=orjson.OPT_INDENT_2).decode("utf-8")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_json, encoding="utf-8")
        meta = result["metadata"]
        style = meta.get("sampling_style", "random")
        target_nodes = int(meta["path_length"])
        target_hops = max(target_nodes - 1, 0)
        action = "Enumerated" if style == "all" else "Sampled"
        print(
            "\n[sample_paths]"
            f"\n  Result         : {action} {meta['n']} path(s)"
            f"\n  Sampling style : {style}"
            f"\n  Target hops    : {target_hops}"
            f"\n  Target nodes   : {target_nodes}"
        )
        print(f"  Seed query     : {meta.get('seed')!r}")
        print(f"  Start node     : {meta.get('resolved_start_node')}")
        print(f"  Start pool     : {meta['start_pool']}")
        print(f"  Same start     : {meta.get('same_start', False)}")
        print(f"  Random seed    : {meta.get('random_seed')}")
        print(f"  Edge weights   : {meta.get('edge_weights') or 'uniform'}")
        print(f"  Cut-off date   : {meta.get('cutoff_date') or 'none'}")
        print(f"  Saved to       : {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
