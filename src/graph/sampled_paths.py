"""Read sampled paths and resolve their graph database."""

# Invalid shapes in parsed JSON are invalid input data, not Python argument types.
# ruff: noqa: TRY004

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import orjson

from src.setup.constants import NODE_AUTHOR, NODE_PAPER, is_uuid_text

DB_METADATA_KEYS = ("db_path", "graph_db_path", "working_db_path")


def load_paths_payload(paths_json: Path) -> dict[str, Any]:
    if not paths_json.is_file():
        raise ValueError(f"Sampled paths file not found: {paths_json}")
    try:
        payload = orjson.loads(paths_json.read_text(encoding="utf-8"))
    except (OSError, orjson.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read sampled paths JSON {paths_json}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Sampled paths JSON must contain an object")
    return payload


def database_path_candidates(raw_path: str, paths_json: Path) -> list[Path]:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return [candidate]
    return [candidate, paths_json.parent / candidate]


def resolve_database_path(
    payload: dict[str, Any],
    paths_json: Path,
    explicit_db_path: Path | None = None,
    *,
    allow_missing: bool = False,
) -> Path:
    """Return the existing graph DB, or the explicit path when it may be created."""
    if explicit_db_path is not None:
        candidates = [explicit_db_path.expanduser()]
    else:
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(
                "Sampled paths metadata is missing; pass the graph DB with --db-path"
            )
        candidates = []
        for key in DB_METADATA_KEYS:
            raw_path = str(metadata.get(key) or "").strip()
            if raw_path:
                candidates.extend(database_path_candidates(raw_path, paths_json))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate

    if explicit_db_path is not None:
        if allow_missing:
            return candidates[0]
        raise ValueError(f"Graph database not found: {explicit_db_path}")
    if allow_missing:
        raise ValueError(
            "No graph database referenced by the sampled paths metadata exists; "
            "pass --db-path to choose where the new database is created"
        )
    raise ValueError(
        "No graph database referenced by the sampled paths metadata exists; "
        "pass the current location with --db-path"
    )


def iter_path_nodes(
    payload: dict[str, Any],
) -> Iterator[tuple[int, int, dict[str, Any]]]:
    paths = payload.get("paths")
    if not isinstance(paths, list):
        raise ValueError("Sampled paths JSON must contain a paths list")

    for path_index, path_entry in enumerate(paths):
        if not isinstance(path_entry, dict) or not isinstance(
            path_entry.get("path"), list
        ):
            raise ValueError(f"paths[{path_index}] must contain a path list")
        for step_index, step in enumerate(path_entry["path"]):
            if not isinstance(step, dict) or not isinstance(step.get("node"), dict):
                raise ValueError(
                    f"paths[{path_index}].path[{step_index}] must contain a node object"
                )
            yield path_index, step_index, step["node"]


def collect_path_papers(
    paths_files: list[Path],
    explicit_db_path: Path | None = None,
    *,
    allow_missing_db: bool = False,
) -> tuple[Path, dict[str, str], int]:
    """Collect UUID -> Semantic Scholar ID in first-seen order across files.

    All files must address the same database. Titles in saved paths may be
    stale; only conflicting, non-empty identifiers are rejected here.
    ``allow_missing_db`` accepts an explicit ``--db-path`` that does not exist yet.
    """
    db_path = None
    papers: dict[str, str] = {}
    path_count = 0
    for paths_file in dict.fromkeys(paths_files):
        payload = load_paths_payload(paths_file)
        resolved = resolve_database_path(
            payload, paths_file, explicit_db_path, allow_missing=allow_missing_db
        ).resolve()
        if db_path is not None and resolved != db_path:
            raise ValueError(
                f"Path files reference different databases: {db_path} and {resolved}. "
                "Use --db-path only when all path UUIDs belong to that database."
            )
        db_path = resolved
        for path_index, step_index, node in iter_path_nodes(payload):
            location = f"{paths_file}: paths[{path_index}].path[{step_index}]"
            node_id = str(node.get("node_id") or "").strip()
            node_type, separator, db_id = node_id.partition(":")
            declared_type = str(node.get("node_type") or "").strip()
            if not separator or node_type not in {NODE_PAPER, NODE_AUTHOR}:
                raise ValueError(f"{location}: invalid graph node ID {node_id!r}")
            if declared_type and declared_type != node_type:
                raise ValueError(f"{location}: node_type does not match {node_id!r}")
            if node_type == NODE_AUTHOR:
                continue
            if not is_uuid_text(db_id):
                raise ValueError(
                    f"{location}: paper node ID is not UUID-backed: {node_id!r}"
                )
            semantic_id = str(node.get("semantic_scholar_id") or "").strip()
            previous = papers.get(db_id, "")
            if previous and semantic_id and previous != semantic_id:
                raise ValueError(
                    f"{location}: conflicting Semantic Scholar IDs for {node_id}"
                )
            papers[db_id] = previous or semantic_id
        path_count += len(payload["paths"])
    if db_path is None:
        raise ValueError("At least one sampled paths file is required")
    return db_path, papers, path_count


def collect_path_nodes(paths_files: list[Path]) -> dict[str, dict[str, Any]]:
    """Return each paper/author node ID with its first saved node record."""
    nodes: dict[str, dict[str, Any]] = {}
    for paths_file in dict.fromkeys(paths_files):
        for _, _, node in iter_path_nodes(load_paths_payload(paths_file)):
            node_id = str(node.get("node_id") or "").strip()
            node_type, separator, db_id = node_id.partition(":")
            if (
                separator
                and node_type in {NODE_PAPER, NODE_AUTHOR}
                and is_uuid_text(db_id)
            ):
                nodes.setdefault(node_id, node)
    return nodes


def refresh_path_count_metadata(payload: dict[str, Any]) -> None:
    """Update existing path-count metadata; requested sampling targets stay unchanged."""
    kept = payload["paths"]
    metadata = payload.get("metadata", {})
    for key in ("n", "stored_paths"):
        if key in metadata:
            metadata[key] = len(kept)
    target_length = metadata.get("path_length")
    complete = sum(
        len(entry["path"]) >= target_length
        if target_length is not None
        else entry.get("status", "complete") == "complete"
        for entry in kept
    )
    for key, count in (
        ("complete_paths", complete),
        ("incomplete_paths", len(kept) - complete),
    ):
        if key in metadata:
            metadata[key] = count


def write_paths_payload_atomic(paths_json: Path, payload: dict[str, Any]) -> None:
    """Replace a sampled-paths file atomically, keeping its file mode."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=paths_json.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(orjson.dumps(payload))
        if paths_json.exists():
            os.chmod(temporary, paths_json.stat().st_mode)
        temporary.replace(paths_json)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def apply_node_id_updates_to_walk_data(
    walk_data: dict[str, Any],
    node_id_updates: dict[str, str],
) -> bool:
    if not node_id_updates:
        return False

    changed = False
    for path_entry in walk_data.get("paths", []):
        start_node_id = str(path_entry.get("start_node") or "")
        if start_node_id in node_id_updates:
            path_entry["start_node"] = node_id_updates[start_node_id]
            changed = True

        for step in path_entry.get("path", []):
            node = step.get("node") or {}
            node_id = str(node.get("node_id") or "")
            if node_id in node_id_updates:
                node["node_id"] = node_id_updates[node_id]
                changed = True

            edge = step.get("edge_taken")
            if not isinstance(edge, dict):
                continue
            edge_from = str(edge.get("from") or "")
            edge_to = str(edge.get("to") or "")
            if edge_from in node_id_updates:
                edge["from"] = node_id_updates[edge_from]
                changed = True
            if edge_to in node_id_updates:
                edge["to"] = node_id_updates[edge_to]
                changed = True
    return changed
