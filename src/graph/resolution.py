from __future__ import annotations

import orjson
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from loguru import logger

from .checkpoints import checkpoint_timestamp, extract_base_stem, find_latest_db


def read_meta_db_path(graph_source: Path) -> str:
    """Return the DB path recorded by a saved graph input, if any."""
    if graph_source.suffix.lower() == ".db":
        try:
            with sqlite3.connect(str(graph_source)) as conn:
                row = conn.execute(
                    "SELECT value FROM graph_metadata WHERE key = 'db_path'"
                ).fetchone()
        except sqlite3.Error:
            return str(graph_source)
        if row is None or not row[0]:
            return str(graph_source)
        raw_value = str(row[0])
        try:
            return str(orjson.loads(raw_value))
        except orjson.JSONDecodeError:
            return raw_value

    try:
        with graph_source.open(encoding="utf-8") as handle:
            payload = orjson.loads(handle.read())
    except Exception:
        return ""
    return str(payload.get("_meta", {}).get("db_path", "") or "")


def is_graph_output_root(path: Path) -> bool:
    """Return whether *path* looks like a previous graph run output root."""
    return (
        path.is_dir()
        and (path / "graphs").is_dir()
        and (path / "pdfs").is_dir()
        and (path / "texts").is_dir()
    )


def collect_graph_json_candidates(path: Path) -> list[Path]:
    """Expand a graph input into concrete DB or legacy JSON graph files."""
    if not path.exists():
        raise FileNotFoundError(f"Graph path not found: {path}")

    if path.is_file():
        if path.suffix.lower() not in {".db", ".json"}:
            raise FileNotFoundError(
                "Graph path must be a .db/.json file or a directory containing graph snapshots: "
                f"{path}"
            )
        return [path]

    search_dir = path / "graphs" if is_graph_output_root(path) else path
    json_candidates = sorted(
        candidate for candidate in search_dir.glob("*.json") if candidate.is_file()
    )
    db_candidates = sorted(
        candidate for candidate in search_dir.glob("*.db") if candidate.is_file()
    )
    candidates = json_candidates + db_candidates
    if candidates:
        return candidates

    raise FileNotFoundError(f"No graph JSON files found in: {search_dir}")


def newest_file(directory: Path, pattern: str, label: str) -> Path:
    """Return the newest file in *directory* matching *pattern*."""
    if not directory.exists():
        raise FileNotFoundError(f"{label} directory not found: {directory}")
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} source is not a directory: {directory}")

    candidates = [path for path in directory.glob(pattern) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No {label} files matching {pattern!r} found in: {directory}"
        )

    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def infer_output_root_from_graph_path(path: Path) -> Path | None:
    """Infer the previous run root from an output root, graphs dir, or graph file."""
    if is_graph_output_root(path):
        return path

    if path.is_dir() and path.name == "graphs" and is_graph_output_root(path.parent):
        return path.parent

    if path.is_file() and path.suffix.lower() in {".json", ".db"}:
        graph_dir = path.parent
        if graph_dir.name == "graphs" and is_graph_output_root(graph_dir.parent):
            return graph_dir.parent

    return None


def resolve_load_graph_path(load_graph: str | Path) -> tuple[Path, Path | None]:
    """Resolve ``--load-graph`` to the concrete graph snapshot plus its output root."""
    raw_path = Path(load_graph)
    graph_path = resolve_graph_argument([str(raw_path)])
    output_root = infer_output_root_from_graph_path(raw_path)
    if output_root is None:
        output_root = infer_output_root_from_graph_path(graph_path)
    return graph_path, output_root


def resolve_output_root(
    output_dir: str | None,
    load_graph: str | Path | None,
) -> tuple[Path, Path | None]:
    """Resolve the output root and optional graph snapshot from the CLI arguments."""
    resolved_graph_path: Path | None = None
    inferred_output_root: Path | None = None
    if load_graph is not None:
        resolved_graph_path, inferred_output_root = resolve_load_graph_path(load_graph)

    if output_dir is not None:
        return Path(output_dir), resolved_graph_path

    if resolved_graph_path is None:
        raise ValueError("--output-dir is required when --load-graph is not provided.")

    if inferred_output_root is None:
        raise ValueError(
            "Could not infer --output-dir from --load-graph. "
            "Pass --output-dir explicitly, or use a previous run root / graphs dir / graph file "
            "under a root containing 'graphs', 'pdfs', and 'texts'."
        )

    return inferred_output_root, resolved_graph_path


def resolve_graph_db_paths(
    graph: str | Path | Sequence[str | Path],
    db_path_hint: str | Path | None = None,
) -> tuple[Path, Path, str]:
    """Resolve a graph input plus optional DB override into concrete paths."""
    graph_inputs = (
        [str(graph)]
        if isinstance(graph, (str, Path))
        else [str(path) for path in graph]
    )
    graph_path = resolve_graph_argument(graph_inputs)
    db_hint = Path(db_path_hint) if db_path_hint is not None else None
    if graph_path.suffix.lower() == ".db":
        if db_hint is not None:
            if db_hint.exists():
                return graph_path, db_hint, "explicit"
            raise FileNotFoundError(
                f"Explicitly provided database path does not exist: {db_hint}"
            )
        return graph_path, graph_path, "graph_db"

    resolved_db, db_resolution = resolve_db_path(graph_path, db_hint)
    return graph_path, resolved_db, db_resolution


def resolve_latest_graph_db_paths(
    graph: str | Path | Sequence[str | Path],
    db_path_hint: str | Path | None = None,
) -> tuple[Path, Path, str]:
    """Resolve the newest graph DB, or a legacy JSON plus newest DB in its directory."""
    graph_inputs = (
        [str(graph)]
        if isinstance(graph, (str, Path))
        else [str(path) for path in graph]
    )
    graph_path = resolve_graph_argument(graph_inputs)

    if db_path_hint is not None:
        db_path = Path(db_path_hint)
        if db_path.exists():
            return graph_path, db_path, "explicit"
        raise FileNotFoundError(
            f"Explicitly provided database path does not exist: {db_path}"
        )

    if graph_path.suffix.lower() == ".db":
        return graph_path, graph_path, "graph_db"

    return (
        graph_path,
        newest_file(graph_path.parent, "*.db", "database"),
        "newest_in_directory",
    )


def resolve_db_path(
    graph_path: Path,
    db_path_hint: Path | None = None,
    *,
    allow_any_checkpoint: bool = False,
    meta_db_path: str | None = None,
) -> tuple[Path, str]:
    """Locate the SQLite database that accompanies *graph_path*."""
    if graph_path.suffix.lower() == ".db":
        if db_path_hint is not None:
            if db_path_hint.exists():
                return db_path_hint, "explicit"
            raise FileNotFoundError(
                f"Explicitly provided database path does not exist: {db_path_hint}"
            )
        return graph_path, "graph_db"

    graph_dir = graph_path.parent
    json_stem = graph_path.stem

    if db_path_hint is not None:
        if db_path_hint.exists():
            return db_path_hint, "explicit"
        raise FileNotFoundError(
            f"Explicitly provided database path does not exist: {db_path_hint}"
        )

    sibling = graph_dir / f"{json_stem}.db"
    if sibling.exists():
        return sibling, "sibling_exact"

    meta_db_path = read_meta_db_path(graph_path) if meta_db_path is None else meta_db_path
    if meta_db_path:
        meta_path = Path(meta_db_path)
        if meta_path.exists():
            return meta_path, "meta_db_path"
        relative_meta_path = graph_dir / meta_path
        if relative_meta_path.exists():
            return relative_meta_path, "meta_db_path"

    base_stem = extract_base_stem(json_stem)
    json_timestamp = checkpoint_timestamp(json_stem)
    latest_db = find_latest_db(
        graph_dir,
        base_stem,
        newer_than=None if allow_any_checkpoint else json_timestamp,
    )
    if latest_db is not None:
        method = "latest_checkpoint" if allow_any_checkpoint else "newer_checkpoint"
        return latest_db, method

    if allow_any_checkpoint:
        return sibling, "fallback"

    raise FileNotFoundError(
        f"Could not locate a database for graph '{graph_path.name}'.\n"
        f"  Tried (1) sibling '{sibling.name}', "
        f"(2) metadata db_path='{meta_db_path or '<unset>'}', "
        f"(3) matching checkpoints in '{graph_dir}'.\n"
        "  Pass --db-path explicitly to provide the database path."
    )


def resolve_graph_argument(paths: list[str]) -> Path:
    """Pick the newest graph DB, or a legacy JSON if no DB snapshots are present."""
    candidates: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        for candidate in collect_graph_json_candidates(Path(raw_path)):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError("No graph DB/JSON files were provided.")

    json_candidates = [
        candidate for candidate in candidates if candidate.suffix.lower() == ".json"
    ]
    if json_candidates:
        candidates = json_candidates
    else:
        db_candidates = [
            candidate for candidate in candidates if candidate.suffix.lower() == ".db"
        ]
        if db_candidates:
            candidates = db_candidates

    if len(candidates) == 1:
        return cast(Path, candidates[0])

    ranked_candidates = sorted(
        candidates,
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )

    for candidate in ranked_candidates:
        if candidate.suffix.lower() == ".db":
            return candidate
        try:
            with candidate.open(encoding="utf-8") as handle:
                orjson.loads(handle.read())
        except (OSError, orjson.JSONDecodeError) as exc:
            logger.warning(
                "Skipping invalid legacy graph JSON checkpoint {} while resolving --load-graph: {}",
                candidate,
                exc,
            )
            continue
        return candidate

    return cast(Path, ranked_candidates[0])
