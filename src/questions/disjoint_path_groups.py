"""Usable input paths and their disjoint groups, for create_dataset.v2.py.

Question depth k uses input paths of exactly 3k - 2 papers, used whole: a
k-paper branch into the source, the k-paper gold path, and a k-paper branch
after the target (the two endpoints are shared). A path is usable when its
starting trees build, its papers pass the explored and cutoff checks, and its
citation edges are in the database. Two usable paths conflict when they share
a paper or a normalized title, or a citation edge joins them: the check growth
itself starts with.

The starting trees of the v8 (create_dataset.v1.py) and v10
(create_dataset.v2.py) pipelines hold the same papers, titles, and citation
neighbors, so one cache next to each paths file serves both.

For each group size, the cache keeps every disjoint group when there are at
most ``group_limit`` of them. Otherwise it keeps only the conflicting pairs: a
group is drawn at random and redrawn while any pair conflicts. Both ways draw
uniformly from all disjoint groups, and the drawn paths are then shuffled.
"""

from __future__ import annotations

import itertools
import math
import os
import random
from array import array
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import orjson

from src.graph.resolution import resolve_latest_graph_db_paths
from src.questions import construction_run
from src.questions import reachability_construction as base
from src.questions import reachability_rendered_pipeline_v10 as growth
from src.questions.explored import configure_explored_context
from src.questions.reachability_long_path_profiles import whole_path_entry

SCHEMA_VERSION = 1
CACHE_NAME = "disjoint_path_groups.json"
CONFLICT_RULE = "shared-paper-or-title-or-citation-edge:v1"
GROUP_SIZES = (2, 4, 8, 16)
GROUP_LIMIT = 10_000_000
# Fixed so serial and parallel runs split the work the same way.
ENUMERATION_CHUNKS = 64
ESTIMATE_SAMPLES = 100_000
DRAW_TRIES = 100_000


def path_depth(length: int) -> int | None:
    """Question depth k of a path with 3k - 2 papers, or None."""
    return (length + 2) // 3 if length >= 4 and (length + 2) % 3 == 0 else None


def cache_path_for(paths_file: Path) -> Path:
    return Path(paths_file).with_name(CACHE_NAME)


def load_depth_paths(path: Path) -> tuple[int, list[dict[str, Any]]]:
    """Read one papers-only file whose paths all have 3k - 2 papers.

    Identical paths are merged and every path gets a unique ID, so the cache
    and the dataset name paths the same way.
    """
    payload = construction_run.load_walk_data(path)
    if payload.get("metadata", {}).get("traversal_mode") != "papers-only":
        raise ValueError(f"{path}: expected papers-only sampled paths.")
    entries = base.combine_reachability_walk_data([(path, payload)])["paths"]
    if not entries:
        raise ValueError(f"{path}: no sampled paths.")
    lengths = {len(entry.get("path") or entry.get("steps") or []) for entry in entries}
    if len(lengths) != 1:
        raise ValueError(f"{path}: paths have different lengths {sorted(lengths)}.")
    length = lengths.pop()
    depth = path_depth(length)
    if depth is None:
        raise ValueError(f"{path}: {length}-paper paths are not 3k - 2 for any k.")
    for index, entry in enumerate(entries):
        steps = entry.get("path") or entry.get("steps") or []
        if any(
            not str(step.get("node", {}).get("node_id", "")).startswith("paper:")
            for step in steps
        ):
            raise ValueError(f"{path}: path {index} contains non-paper nodes.")
    return depth, entries


@dataclass
class GraphChecker:
    """Read-only graph database access for path checks and growth."""

    context: Any
    facts: Any
    cutoff: Any
    graph_path: Path
    db_path: Path

    def close(self) -> None:
        self.context.db.close()


def open_graph_checker(graph_db: Path, cutoff_date: str, only_explored: int) -> GraphChecker:
    graph_path, db_path, _resolution = resolve_latest_graph_db_paths(graph_db, None)
    context = base.open_lazy_caching_checker(
        graph_path,
        db_path,
        include_authorship_edges=False,
        cache_max_entries=512,
        sqlite_cache_size_kib=262144,
        sqlite_mmap_size_bytes=536870912,
    )
    try:
        configure_explored_context(context, only_explored)
        facts = growth.paper_facts_reader(
            context,
            include_author_identity=False,
            cache_max_entries=512,
            identity_cache_max_entries=8192,
        )
        cutoff = growth.PaperDateCutoff(
            context.db, growth.parse_cutoff_date(cutoff_date), {}
        )
    except BaseException:
        context.db.close()
        raise
    return GraphChecker(context, facts, cutoff, graph_path, db_path)


@dataclass(frozen=True)
class PathFacts:
    """What decides whether two paths conflict (over all their orientations)."""

    papers: frozenset[str]
    titles: frozenset[str]
    neighbors: frozenset[str]


def prepare_path(
    entry: dict[str, Any],
    depth: int,
    *,
    context: Any,
    facts: Any,
    cutoff: Any,
    pipeline: Any = growth,
) -> tuple[list[Any], dict[str, Any]] | str:
    """Starting trees (one per orientation) of a whole path, or why it is unusable.

    ``pipeline`` is the growth module whose trees are built (v10 by default).
    """
    try:
        sliced = whole_path_entry(entry, depth)
    except ValueError:
        return "invalid_layout"
    reason = pipeline.entry_cutoff_rejection_reason(sliced, cutoff)
    if reason:
        return reason
    prepared = pipeline.PreparedPathStore(
        entries_by_path_id={sliced["path_id"]: sliced},
        context=context,
        facts_reader=facts,
        traversal_mode="papers-only",
        max_author_answer_index=10,
        cache={},
    ).get(sliced["path_id"])
    if not prepared:
        return "invalid_initial_tree"
    # Check protected edges against canonical adjacency, even for cached inputs.
    for variant in prepared:
        for owner, cited in variant.state.protected_reference_pairs:
            if cited not in facts.get(owner).outgoing_paper_ids:
                return "stale_path_edge"
    return prepared, sliced


def path_facts(prepared: Sequence[Any]) -> PathFacts:
    trees = [variant.state for variant in prepared]
    return PathFacts(
        papers=frozenset().union(*(tree.visible_paper_ids for tree in trees)),
        titles=frozenset().union(*(tree.used_title_keys for tree in trees)),
        neighbors=frozenset().union(*(tree.citation_neighbor_ids for tree in trees)),
    )


def check_paths(
    entries: Sequence[dict[str, Any]], depth: int, *, context: Any, facts: Any, cutoff: Any
) -> list[tuple[str, PathFacts | str]]:
    """(path ID, facts or the reason it is unusable) for each entry, in order."""
    results = []
    for entry in entries:
        prepared = prepare_path(entry, depth, context=context, facts=facts, cutoff=cutoff)
        results.append(
            (
                entry["path_id"],
                prepared if isinstance(prepared, str) else path_facts(prepared[0]),
            )
        )
    return results


def paths_conflict(first: PathFacts, second: PathFacts) -> bool:
    return not (
        first.papers.isdisjoint(second.papers)
        and first.titles.isdisjoint(second.titles)
        and first.neighbors.isdisjoint(second.papers)
        and second.neighbors.isdisjoint(first.papers)
    )


def conflicting_pairs(facts: Sequence[PathFacts]) -> list[tuple[int, int]]:
    """Sorted (i, j), i < j, of conflicting paths, found through inverted indexes."""
    by_paper: dict[str, list[int]] = defaultdict(list)
    by_title: dict[str, list[int]] = defaultdict(list)
    by_neighbor: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(facts):
        for paper in item.papers:
            by_paper[paper].append(index)
        for title in item.titles:
            by_title[title].append(index)
        for neighbor in item.neighbors:
            by_neighbor[neighbor].append(index)
    pairs: set[tuple[int, int]] = set()
    for index, item in enumerate(facts):
        others: set[int] = set()
        for paper in item.papers:
            others.update(by_paper[paper])
            others.update(by_neighbor.get(paper, ()))
        for title in item.titles:
            others.update(by_title[title])
        for neighbor in item.neighbors:
            others.update(by_paper.get(neighbor, ()))
        pairs.update((index, other) for other in others if other > index)
    return sorted(pairs)


def compatible_masks(count: int, conflicts: Sequence[tuple[int, int]]) -> list[int]:
    """Bit j of mask i is set when i < j and paths i and j do not conflict."""
    everyone = (1 << count) - 1
    masks = [everyone & ~((1 << (index + 1)) - 1) for index in range(count)]
    for first, second in conflicts:
        masks[first] &= ~(1 << second)
    return masks


# Set in each worker (and in this process when there is one worker).
ENUMERATION_STATE: dict[str, Any] = {}


def start_enumeration(masks: list[int], size: int, cap: int) -> None:
    ENUMERATION_STATE.update(masks=masks, size=size, cap=cap)


def enumerate_chunk(first_indexes: Sequence[int]) -> tuple[array, bool]:
    """Disjoint groups whose smallest path is in ``first_indexes``.

    Returns the groups flattened, and True if more than ``cap`` were found
    (then the enumeration stopped early).
    """
    masks, size, cap = (ENUMERATION_STATE[key] for key in ("masks", "size", "cap"))
    found = array("I")
    count = 0

    def extend(group: tuple[int, ...], candidates: int) -> bool:
        nonlocal count
        if len(group) == size:
            found.extend(group)
            count += 1
            return count > cap
        while candidates:
            lowest = candidates & -candidates
            candidates ^= lowest
            index = lowest.bit_length() - 1
            if extend((*group, index), candidates & masks[index]):
                return True
        return False

    for first in first_indexes:
        if extend((first,), masks[first]):
            return found, True
    return found, False


def map_in_workers(
    function: Callable[[Any], Any],
    items: Sequence[Any],
    workers: int,
    initializer: Callable[..., None],
    initargs: tuple[Any, ...],
) -> Iterator[Any]:
    """Results in item order; spawned workers never inherit SQLite handles."""
    if workers <= 1:
        initializer(*initargs)
        yield from map(function, items)
        return
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=initializer,
        initargs=initargs,
    )
    try:
        yield from executor.map(function, items)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def estimate_disjoint_groups(
    count: int, conflicts: set[tuple[int, int]], size: int, seed: int
) -> float:
    """Number of disjoint groups, from the share of random groups that are disjoint."""
    if count < size:
        return 0.0
    rng = random.Random(seed)
    disjoint = 0
    for _sample in range(ESTIMATE_SAMPLES):
        group = sorted(rng.sample(range(count), size))
        disjoint += not any(
            pair in conflicts for pair in itertools.combinations(group, 2)
        )
    return math.comb(count, size) * disjoint / ESTIMATE_SAMPLES


def build_groups(
    count: int,
    conflicts: Sequence[tuple[int, int]],
    *,
    sizes: Sequence[int] = GROUP_SIZES,
    limit: int = GROUP_LIMIT,
    workers: int = 1,
    on_chunk: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """For each size: every disjoint group (at most ``limit``), or draw-and-retry."""
    masks = compatible_masks(count, conflicts)
    conflict_set = set(conflicts)
    chunks = [
        list(range(start, count, ENUMERATION_CHUNKS))
        for start in range(min(ENUMERATION_CHUNKS, count))
    ]
    groups: dict[str, Any] = {}
    for size in sizes:
        found = array("I")
        over_limit = False
        if count >= size:
            results = map_in_workers(
                enumerate_chunk, chunks, workers, start_enumeration, (masks, size, limit)
            )
            try:
                for chunk_groups, overflow in results:
                    if on_chunk is not None:
                        on_chunk()
                    found.extend(chunk_groups)
                    if overflow or len(found) > limit * size:
                        over_limit = True
                        break
            finally:
                # Stops the remaining workers as soon as the limit is passed.
                results.close()
        if over_limit:
            groups[str(size)] = {
                "method": "draw_and_retry",
                "count": None,
                "estimated_count": estimate_disjoint_groups(
                    count, conflict_set, size, seed=size
                ),
            }
            continue
        # Sorted, so serial and parallel runs store the same list.
        tuples = sorted(zip(*(iter(found),) * size))
        groups[str(size)] = {
            "method": "stored",
            "count": len(tuples),
            "groups": [index for group in tuples for index in group],
        }
    return groups


def groups_payload(
    checks: Sequence[tuple[str, PathFacts | str]],
    *,
    sizes: Sequence[int] = GROUP_SIZES,
    limit: int = GROUP_LIMIT,
    workers: int = 1,
    on_chunk: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Usable paths, rejection counts, conflicting pairs, and groups of each size."""
    usable = [(path_id, item) for path_id, item in checks if isinstance(item, PathFacts)]
    rejected: dict[str, int] = defaultdict(int)
    for _path_id, item in checks:
        if isinstance(item, str):
            rejected[item] += 1
    conflicts = conflicting_pairs([item for _path_id, item in usable])
    return {
        "usable_path_ids": [path_id for path_id, _item in usable],
        "rejected": dict(sorted(rejected.items())),
        "conflicting_pairs": [index for pair in conflicts for index in pair],
        "groups": build_groups(
            len(usable),
            conflicts,
            sizes=sizes,
            limit=limit,
            workers=workers,
            on_chunk=on_chunk,
        ),
    }


def write_cache(
    paths_file: Path,
    *,
    depth: int,
    path_count: int,
    checks: Sequence[tuple[str, PathFacts | str]],
    database: str,
    cutoff_date: str,
    only_explored: int,
    paths_name: str,
    sizes: Sequence[int] = GROUP_SIZES,
    limit: int = GROUP_LIMIT,
    workers: int = 1,
    on_chunk: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Build and atomically write the cache next to ``paths_file``; return it."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "conflict_rule": CONFLICT_RULE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paths_file": paths_name,
        "database": database,
        "cutoff_date": cutoff_date,
        "only_explored": only_explored,
        "question_depth": depth,
        "path_length": 3 * depth - 2,
        "path_count": path_count,
        "group_limit": limit,
        **groups_payload(
            checks, sizes=sizes, limit=limit, workers=workers, on_chunk=on_chunk
        ),
    }
    output = cache_path_for(paths_file)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_bytes(orjson.dumps(payload))
    os.replace(temporary, output)
    return payload


class PathGroups:
    """The usable paths of one depth, and uniform draws of disjoint groups."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.usable: list[str] = payload["usable_path_ids"]
        flat = payload["conflicting_pairs"]
        self.conflicts = set(zip(flat[0::2], flat[1::2]))
        self.groups: dict[int, array | None] = {}
        for size, info in payload["groups"].items():
            self.groups[int(size)] = (
                array("I", info["groups"]) if info["method"] == "stored" else None
            )

    def draw(self, size: int, rng: random.Random) -> list[str] | None:
        """A uniformly drawn disjoint group of path IDs, shuffled; None if none."""
        if size == 1:
            group: Sequence[int] = [rng.randrange(len(self.usable))] if self.usable else []
        elif self.groups.get(size) is not None:
            stored = self.groups[size]
            if not stored:
                return None
            start = rng.randrange(len(stored) // size) * size
            group = stored[start : start + size]
        elif size in self.groups:
            if len(self.usable) < size:
                return None
            for _try in range(DRAW_TRIES):
                group = sorted(rng.sample(range(len(self.usable)), size))
                if not any(
                    pair in self.conflicts for pair in itertools.combinations(group, 2)
                ):
                    break
            else:
                return None
        else:
            raise ValueError(f"The cache has no groups of {size} paths.")
        if not group:
            return None
        path_ids = [self.usable[index] for index in group]
        rng.shuffle(path_ids)
        return path_ids


def load_cache(
    paths_file: Path,
    *,
    database: str,
    cutoff_date: str,
    only_explored: int,
    paths_name: str,
) -> PathGroups:
    """Read the cache next to ``paths_file``; refuse one built from other inputs."""
    path = cache_path_for(paths_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Build it with endpoints/precompute_disjoint_path_groups.py."
        )
    payload = orjson.loads(path.read_bytes())
    expected = {
        "schema_version": SCHEMA_VERSION,
        "conflict_rule": CONFLICT_RULE,
        "paths_file": paths_name,
        "database": database,
        "cutoff_date": cutoff_date,
        "only_explored": only_explored,
    }
    stale = {
        key: payload.get(key)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if stale:
        raise ValueError(
            f"{path} was built from other inputs ({stale}); rebuild it with "
            "endpoints/precompute_disjoint_path_groups.py."
        )
    return PathGroups(payload)
