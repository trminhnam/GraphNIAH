"""Persistent maximal-tree records and strict crossed-tree compatibility."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import orjson
from pathlib import Path
import sqlite3
from typing import Any
import zlib

from src.prompts.reachability_yesno_qa import (
    TRAVERSAL_MODE_ALTERNATING,
    TRAVERSAL_MODE_MIXED,
    is_paper_only_traversal_mode,
)
from src.setup.constants import (
    EDGE_AUTHORSHIP,
    EDGE_CITED_BY,
    EDGE_CITING,
    NODE_AUTHOR,
    NODE_PAPER,
    nid_type,
)
from src.utils.resume import load_json_output, write_json_output


SEARCH_TREE_CACHE_VERSION = 9
SEARCH_TREE_CACHE_STORAGE_VERSION = 1
NO_CONSTRUCTION = "crossed-disjoint-trees"
CACHE_COMPATIBILITY_FIELDS = (
    "version",
    "traversal_mode",
    "needle_type",
    "max_author_answer_index",
    "identity_policy_version",
    "tree_construction_strategy",
    "question_hops",
)


def wrong_branch_step_allowed(
    source_id: str,
    target_id: str,
    edge_type: str,
    traversal_mode: str,
) -> bool:
    """Return whether an edge belongs to a wrong branch's traversal family."""
    source_type = nid_type(source_id)
    target_type = nid_type(target_id)
    if is_paper_only_traversal_mode(traversal_mode):
        return (
            source_type == target_type == NODE_PAPER
            and edge_type in {EDGE_CITING, EDGE_CITED_BY}
        )
    if traversal_mode == TRAVERSAL_MODE_ALTERNATING:
        return (
            edge_type == EDGE_AUTHORSHIP
            and {source_type, target_type} == {NODE_PAPER, NODE_AUTHOR}
        )
    if traversal_mode == TRAVERSAL_MODE_MIXED:
        return (
            source_type == target_type == NODE_PAPER
            and edge_type in {EDGE_CITING, EDGE_CITED_BY}
        ) or (
            edge_type == EDGE_AUTHORSHIP
            and {source_type, target_type} == {NODE_PAPER, NODE_AUTHOR}
        )
    raise ValueError(f"Unsupported traversal mode: {traversal_mode!r}")


@dataclass
class CachedSearchTree:
    """One maximal sanitized tree stored before bucket-specific pruning."""

    cache_key: str
    path_id: str
    orientation_index: int
    source_node_id: str
    target_node_id: str
    paper_ids: list[str]
    authors_by_paper: dict[str, list[str]]
    references_by_paper: dict[str, list[str]]
    protected_reference_pairs: list[tuple[str, str]]
    witness: tuple[str, ...]
    closure_paper_ids: frozenset[str]
    closure_author_ids: frozenset[str]
    closure_title_keys: frozenset[str]
    closure_author_name_keys: frozenset[str]
    citation_neighbor_paper_ids: frozenset[str]
    # The gold path and supplied branch-segment structure are recorded separately
    # so bucket pruning can protect the backbone and remove optional branches
    # from their deepest leaves upward. Defaults keep pre-v2 payloads readable.
    backbone_paper_ids: list[str] = field(default_factory=list)
    branch_parent_by_paper: dict[str, str] = field(default_factory=dict)
    branch_depth_by_paper: dict[str, int] = field(default_factory=dict)
    optional_reference_pairs: list[tuple[str, str]] = field(
        default_factory=list
    )
    # Per-paper identity records let a pruned variant rebuild its exact
    # closure without querying the database again.
    title_key_by_paper: dict[str, str] = field(default_factory=dict)
    author_ids_by_paper_identity: dict[str, list[str]] = field(
        default_factory=dict
    )
    author_name_keys_by_paper: dict[str, list[str]] = field(
        default_factory=dict
    )
    citation_neighbors_by_paper: dict[str, list[str]] = field(
        default_factory=dict
    )
    # Role-specific endpoint branches prevent the gold path from being the
    # uniquely deepest continuation. Positive trees protect one exact-depth
    # branch at each endpoint; negative trees protect only the endpoint that
    # is queried after trees are crossed.
    tree_role: str = "positive"
    gold_depth: int = 0
    wrong_branch_paths: list[list[str]] = field(default_factory=list)


@dataclass
class TreeBucketVariant:
    """One cached tree pruned for a requested YES-work or component bucket."""

    tree: CachedSearchTree
    references_by_paper: dict[str, list[str]]
    yes_complexity: float
    component_size: int
    removed_reference_count: int
    # ``None`` means this object came from the legacy constructor and should
    # fall back to the maximal tree field. An explicit empty collection is a
    # valid final value after compaction and therefore must remain distinct.
    paper_ids: list[str] | None = None
    authors_by_paper: dict[str, list[str]] | None = None
    closure_paper_ids: frozenset[str] | None = None
    closure_author_ids: frozenset[str] | None = None
    closure_title_keys: frozenset[str] | None = None
    closure_author_name_keys: frozenset[str] | None = None
    citation_neighbor_paper_ids: frozenset[str] | None = None
    removed_reference_pairs: list[tuple[str, str]] = field(
        default_factory=list
    )
    variant_signature: str = ""


@dataclass
class IncrementalSearchTreeCache:
    """SQLite-backed cache that reads and writes one path entry at a time."""

    path: Path
    configuration: dict[str, Any]
    connection: sqlite3.Connection


SearchTreeCache = dict[str, Any] | IncrementalSearchTreeCache


def path_tree_cache_key(entry: dict[str, Any]) -> str:
    """Return a content-derived key that is stable across input-file ordering."""
    path_value = entry.get("path") or entry.get("steps") or []
    long_path_profile = entry.get("reachability_long_path_profile")
    encoded = orjson.dumps({
            "path": path_value,
            "reachability_long_path_profile": long_path_profile,
        }, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(encoded).hexdigest()


def storage_fingerprint(path: Path) -> dict[str, Any]:
    """Return a cheap storage fingerprint for cache audit metadata."""
    resolved = path.resolve()
    try:
        stat = resolved.stat()
    except FileNotFoundError:
        return {"path": str(resolved), "missing": True}
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def search_tree_cache_configuration(
    *,
    graph_path: Path,
    db_path: Path,
    traversal_mode: str,
    needle_type: str,
    max_author_answer_index: int | None,
    identity_policy_version: int,
    tree_construction_strategy: str,
    question_hops: int,
    only_explored: int = 0,
) -> dict[str, Any]:
    """Return tree-construction settings plus storage audit metadata."""
    return {
        **({"only_explored": 1} if only_explored else {}),
        "version": SEARCH_TREE_CACHE_VERSION,
        "graph": storage_fingerprint(graph_path),
        "db": storage_fingerprint(db_path),
        "traversal_mode": traversal_mode,
        "needle_type": needle_type,
        "max_author_answer_index": max_author_answer_index,
        "identity_policy_version": identity_policy_version,
        "tree_construction_strategy": tree_construction_strategy,
        "question_hops": question_hops,
    }


def search_tree_cache_configurations_match(
    saved_configuration: object,
    configuration: dict[str, Any],
) -> bool:
    """Return whether saved tree-construction settings are reusable."""
    return (
        isinstance(saved_configuration, dict)
        and saved_configuration.get("only_explored", 0)
        == configuration.get("only_explored", 0)
        and all(
            field in saved_configuration
            and field in configuration
            and saved_configuration[field] == configuration[field]
            for field in CACHE_COMPATIBILITY_FIELDS
        )
    )


def search_tree_cache_configuration_error(path: Path) -> ValueError:
    """Return the standard incompatible-cache error for *path*."""
    return ValueError(
        f"Search-tree cache configuration does not match: {path}. "
        "Use a fresh cache path for the current cache version, traversal "
        "mode, needle type, and maximum author-answer index."
    )


def load_incremental_search_tree_cache(
    path: Path,
    configuration: dict[str, Any],
) -> IncrementalSearchTreeCache:
    """Open or initialize an entry-addressed SQLite search-tree cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS cache_metadata (
            storage_version INTEGER NOT NULL,
            configuration TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS tree_entries (
            entry_key TEXT PRIMARY KEY,
            tree_count INTEGER NOT NULL,
            payload BLOB NOT NULL
        )
        """
    )
    row = connection.execute(
        "SELECT storage_version, configuration FROM cache_metadata LIMIT 1"
    ).fetchone()
    if row is None:
        connection.execute(
            "INSERT INTO cache_metadata(storage_version, configuration) "
            "VALUES (?, ?)",
            (
                SEARCH_TREE_CACHE_STORAGE_VERSION,
                orjson.dumps(configuration, option=orjson.OPT_SORT_KEYS).decode("utf-8"),
            ),
        )
        connection.commit()
        return IncrementalSearchTreeCache(path, configuration, connection)

    storage_version, saved_configuration_text = row
    try:
        saved_configuration = orjson.loads(saved_configuration_text)
    except orjson.JSONDecodeError as error:
        connection.close()
        raise ValueError(
            f"Search-tree cache configuration is invalid: {path}"
        ) from error
    if (
        int(storage_version) != SEARCH_TREE_CACHE_STORAGE_VERSION
        or not search_tree_cache_configurations_match(
            saved_configuration,
            configuration,
        )
    ):
        connection.close()
        raise search_tree_cache_configuration_error(path)

    storage_changed = any(
        saved_configuration.get(field) != configuration.get(field)
        for field in ("graph", "db")
    )
    if storage_changed:
        connection.execute("DELETE FROM tree_entries WHERE tree_count = 0")
    connection.execute("DELETE FROM cache_metadata")
    connection.execute(
        "INSERT INTO cache_metadata(storage_version, configuration) VALUES (?, ?)",
        (
            SEARCH_TREE_CACHE_STORAGE_VERSION,
            orjson.dumps(configuration, option=orjson.OPT_SORT_KEYS).decode("utf-8"),
        ),
    )
    connection.commit()
    return IncrementalSearchTreeCache(path, configuration, connection)


def incremental_search_tree_cache_get(
    cache: IncrementalSearchTreeCache,
    entry_key: str,
) -> list[Any] | None:
    """Load one cached path entry without reading unrelated entries."""
    row = cache.connection.execute(
        "SELECT payload FROM tree_entries WHERE entry_key = ?",
        (entry_key,),
    ).fetchone()
    if row is None:
        return None
    try:
        payload = orjson.loads(zlib.decompress(row[0]).decode("utf-8"))
    except (orjson.JSONDecodeError, UnicodeDecodeError, zlib.error) as error:
        raise ValueError(
            f"Search-tree cache entry is invalid: {cache.path} ({entry_key})"
        ) from error
    if not isinstance(payload, list):
        raise ValueError(
            f"Search-tree cache entry must be a list: {cache.path} ({entry_key})"
        )
    return payload


def incremental_search_tree_cache_put(
    cache: IncrementalSearchTreeCache,
    entry_key: str,
    payload: list[dict[str, Any]],
) -> None:
    """Atomically persist one completed path entry."""
    encoded = orjson.dumps(payload)
    compressed = zlib.compress(encoded, level=1)
    cache.connection.execute(
        """
        INSERT INTO tree_entries(entry_key, tree_count, payload)
        VALUES (?, ?, ?)
        ON CONFLICT(entry_key) DO UPDATE SET
            tree_count = excluded.tree_count,
            payload = excluded.payload
        """,
        (entry_key, len(payload), compressed),
    )
    cache.connection.commit()


def incremental_search_tree_cache_remove(
    cache: IncrementalSearchTreeCache,
    entry_key: str,
) -> None:
    """Remove one invalidated path entry."""
    cache.connection.execute(
        "DELETE FROM tree_entries WHERE entry_key = ?",
        (entry_key,),
    )
    cache.connection.commit()


def incremental_search_tree_cache_entry_count(
    cache: IncrementalSearchTreeCache,
) -> int:
    """Return the number of independently persisted path entries."""
    row = cache.connection.execute("SELECT COUNT(*) FROM tree_entries").fetchone()
    return int(row[0]) if row is not None else 0


def close_search_tree_cache(cache: SearchTreeCache) -> None:
    """Close an incremental cache; legacy in-memory caches need no action."""
    if isinstance(cache, IncrementalSearchTreeCache):
        cache.connection.close()


def load_search_tree_cache(
    path: Path | None,
    configuration: dict[str, Any],
) -> SearchTreeCache:
    """Load a cache compatible with the current tree-construction settings.

    Graph and database fingerprints are retained as audit metadata but do not
    invalidate the cache because those stores are updated continuously.
    """
    if path is not None and path.suffix.lower() in {".sqlite", ".sqlite3"}:
        return load_incremental_search_tree_cache(path, configuration)
    if path is None or not path.exists():
        return {"configuration": configuration, "entries": {}}
    payload = load_json_output(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Search-tree cache must be a JSON object: {path}")
    saved_configuration = payload.get("configuration")
    if not search_tree_cache_configurations_match(
        saved_configuration,
        configuration,
    ):
        raise search_tree_cache_configuration_error(path)
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"Search-tree cache entries must be an object: {path}")
    storage_changed = any(
        saved_configuration.get(field) != configuration.get(field)
        for field in ("graph", "db")
    )
    if storage_changed:
        for cache_key in [
            key for key, value in entries.items() if value == []
        ]:
            entries.pop(cache_key)
    payload["configuration"] = configuration
    return payload


def write_search_tree_cache(
    path: Path | None,
    cache: SearchTreeCache,
    indent: int,
) -> None:
    """Atomically persist maximal trees accumulated during this run."""
    if isinstance(cache, IncrementalSearchTreeCache):
        cache.connection.commit()
        return
    if path is not None:
        write_json_output(path, cache, indent)


def cached_tree_payload(tree: CachedSearchTree) -> dict[str, Any]:
    """Return a JSON-compatible cached-tree record."""
    return {
        "cache_key": tree.cache_key,
        "path_id": tree.path_id,
        "orientation_index": tree.orientation_index,
        "source_node_id": tree.source_node_id,
        "target_node_id": tree.target_node_id,
        "paper_ids": tree.paper_ids,
        "authors_by_paper": tree.authors_by_paper,
        "references_by_paper": tree.references_by_paper,
        "protected_reference_pairs": [
            list(pair) for pair in tree.protected_reference_pairs
        ],
        "witness": list(tree.witness),
        "closure_paper_ids": sorted(tree.closure_paper_ids),
        "closure_author_ids": sorted(tree.closure_author_ids),
        "closure_title_keys": sorted(tree.closure_title_keys),
        "closure_author_name_keys": sorted(tree.closure_author_name_keys),
        "citation_neighbor_paper_ids": sorted(tree.citation_neighbor_paper_ids),
        "backbone_paper_ids": tree.backbone_paper_ids,
        "branch_parent_by_paper": tree.branch_parent_by_paper,
        "branch_depth_by_paper": tree.branch_depth_by_paper,
        "optional_reference_pairs": [
            list(pair) for pair in tree.optional_reference_pairs
        ],
        "title_key_by_paper": tree.title_key_by_paper,
        "author_ids_by_paper_identity": (
            tree.author_ids_by_paper_identity
        ),
        "author_name_keys_by_paper": tree.author_name_keys_by_paper,
        "citation_neighbors_by_paper": tree.citation_neighbors_by_paper,
        "tree_role": tree.tree_role,
        "gold_depth": tree.gold_depth,
        "wrong_branch_paths": tree.wrong_branch_paths,
    }


def cached_tree_from_payload(payload: dict[str, Any]) -> CachedSearchTree:
    """Restore one cached maximal tree from JSON data."""
    return CachedSearchTree(
        cache_key=str(payload["cache_key"]),
        path_id=str(payload["path_id"]),
        orientation_index=int(payload["orientation_index"]),
        source_node_id=str(payload["source_node_id"]),
        target_node_id=str(payload["target_node_id"]),
        paper_ids=[str(value) for value in payload.get("paper_ids", [])],
        authors_by_paper={
            str(owner): [str(value) for value in values]
            for owner, values in (payload.get("authors_by_paper") or {}).items()
        },
        references_by_paper={
            str(owner): [str(value) for value in values]
            for owner, values in (payload.get("references_by_paper") or {}).items()
        },
        protected_reference_pairs=[
            (str(pair[0]), str(pair[1]))
            for pair in payload.get("protected_reference_pairs", [])
        ],
        witness=tuple(str(value) for value in payload.get("witness", [])),
        closure_paper_ids=frozenset(
            str(value) for value in payload.get("closure_paper_ids", [])
        ),
        closure_author_ids=frozenset(
            str(value) for value in payload.get("closure_author_ids", [])
        ),
        closure_title_keys=frozenset(
            str(value) for value in payload.get("closure_title_keys", [])
        ),
        closure_author_name_keys=frozenset(
            str(value) for value in payload.get("closure_author_name_keys", [])
        ),
        citation_neighbor_paper_ids=frozenset(
            str(value)
            for value in payload.get("citation_neighbor_paper_ids", [])
        ),
        backbone_paper_ids=[
            str(value) for value in payload.get("backbone_paper_ids", [])
        ],
        branch_parent_by_paper={
            str(paper_id): str(parent_id)
            for paper_id, parent_id in (
                payload.get("branch_parent_by_paper") or {}
            ).items()
        },
        branch_depth_by_paper={
            str(paper_id): int(depth)
            for paper_id, depth in (
                payload.get("branch_depth_by_paper") or {}
            ).items()
        },
        optional_reference_pairs=[
            (str(pair[0]), str(pair[1]))
            for pair in payload.get("optional_reference_pairs", [])
        ],
        title_key_by_paper={
            str(paper_id): str(title_key)
            for paper_id, title_key in (
                payload.get("title_key_by_paper") or {}
            ).items()
        },
        author_ids_by_paper_identity={
            str(paper_id): [str(author_id) for author_id in author_ids]
            for paper_id, author_ids in (
                payload.get("author_ids_by_paper_identity") or {}
            ).items()
        },
        author_name_keys_by_paper={
            str(paper_id): [str(name_key) for name_key in name_keys]
            for paper_id, name_keys in (
                payload.get("author_name_keys_by_paper") or {}
            ).items()
        },
        citation_neighbors_by_paper={
            str(paper_id): [str(neighbor_id) for neighbor_id in neighbors]
            for paper_id, neighbors in (
                payload.get("citation_neighbors_by_paper") or {}
            ).items()
        },
        tree_role=str(payload.get("tree_role") or "positive"),
        gold_depth=int(payload.get("gold_depth") or 0),
        wrong_branch_paths=[
            [str(node_id) for node_id in path]
            for path in payload.get("wrong_branch_paths", [])
        ],
    )


def tree_disjointness_reason(
    first: CachedSearchTree,
    second: CachedSearchTree,
) -> str | None:
    """Return the first strict pair incompatibility, or ``None`` when disjoint."""
    if first.path_id == second.path_id:
        return "same_input_path"
    if first.closure_paper_ids.intersection(second.closure_paper_ids):
        return "shared_paper_ids"
    if first.closure_author_ids.intersection(second.closure_author_ids):
        return "shared_author_ids"
    if first.closure_title_keys.intersection(second.closure_title_keys):
        return "shared_normalized_paper_titles"
    if first.closure_author_name_keys.intersection(
        second.closure_author_name_keys
    ):
        return "shared_normalized_author_names"
    if first.citation_neighbor_paper_ids.intersection(second.closure_paper_ids):
        return "citation_edge_crosses_trees"
    if second.citation_neighbor_paper_ids.intersection(first.closure_paper_ids):
        return "citation_edge_crosses_trees"
    return None


def tree_variant_disjointness_reason(
    first: TreeBucketVariant,
    second: TreeBucketVariant,
) -> str | None:
    """Return an incompatibility using the two final pruned variants.

    Legacy variants that do not yet carry a final closure fall back to their
    maximal cached-tree closure. New variants may explicitly carry an empty
    closure field, which is different from leaving that field as ``None``.
    """
    if first.tree.path_id == second.tree.path_id:
        return "same_input_path"

    first_paper_ids = first.closure_paper_ids
    if first_paper_ids is None:
        first_paper_ids = first.tree.closure_paper_ids
    second_paper_ids = second.closure_paper_ids
    if second_paper_ids is None:
        second_paper_ids = second.tree.closure_paper_ids

    first_author_ids = first.closure_author_ids
    if first_author_ids is None:
        first_author_ids = first.tree.closure_author_ids
    second_author_ids = second.closure_author_ids
    if second_author_ids is None:
        second_author_ids = second.tree.closure_author_ids

    first_title_keys = first.closure_title_keys
    if first_title_keys is None:
        first_title_keys = first.tree.closure_title_keys
    second_title_keys = second.closure_title_keys
    if second_title_keys is None:
        second_title_keys = second.tree.closure_title_keys

    first_author_name_keys = first.closure_author_name_keys
    if first_author_name_keys is None:
        first_author_name_keys = first.tree.closure_author_name_keys
    second_author_name_keys = second.closure_author_name_keys
    if second_author_name_keys is None:
        second_author_name_keys = second.tree.closure_author_name_keys

    first_citation_neighbors = first.citation_neighbor_paper_ids
    if first_citation_neighbors is None:
        first_citation_neighbors = first.tree.citation_neighbor_paper_ids
    second_citation_neighbors = second.citation_neighbor_paper_ids
    if second_citation_neighbors is None:
        second_citation_neighbors = second.tree.citation_neighbor_paper_ids

    if first_paper_ids.intersection(second_paper_ids):
        return "shared_paper_ids"
    if first_author_ids.intersection(second_author_ids):
        return "shared_author_ids"
    if first_title_keys.intersection(second_title_keys):
        return "shared_normalized_paper_titles"
    if first_author_name_keys.intersection(second_author_name_keys):
        return "shared_normalized_author_names"
    if first_citation_neighbors.intersection(second_paper_ids):
        return "citation_edge_crosses_trees"
    if second_citation_neighbors.intersection(first_paper_ids):
        return "citation_edge_crosses_trees"
    return None
