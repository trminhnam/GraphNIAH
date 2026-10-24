"""Notebook duplicate audit and in-place database/path updates with backups."""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
import time
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import orjson
from loguru import logger

from merge_graph_databases import (
    apply_duplicate_paper_groups,
    apply_node_id_updates_to_walk_data,
    find_duplicate_paper_groups,
    pick_group_winner,
    read_walk_data,
    table_exists,
    walk_node_ids,
    write_json_atomic,
)
from src.graph.readers import DBReader, build_node_payload
from src.graph.sampled_paths import DB_METADATA_KEYS
from src.utils.process_pool import process_results


def open_database(path: Path, *, readonly: bool) -> sqlite3.Connection:
    conn = sqlite3.connect(
        path.resolve().as_uri() + ("?mode=ro" if readonly else "?mode=rw"), uri=True
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def make_plan(
    conn: sqlite3.Connection, max_group_size: int, num_workers: int
) -> tuple[list, dict]:
    groups, oversized = find_duplicate_paper_groups(
        conn, max_group_size=max_group_size, num_workers=num_workers
    )
    merges = []
    updates = {}
    for group in groups:
        if not group.is_clique:
            continue
        winner = pick_group_winner(conn, group.paper_ids)
        losers = [paper_id for paper_id in group.paper_ids if paper_id != winner]
        merges.append({"title": group.title, "winner_id": winner, "loser_ids": losers})
        updates.update({f"paper:{loser}": f"paper:{winner}" for loser in losers})
    return groups, {
        "merges": merges,
        "node_id_updates": updates,
        "ambiguous_groups": [asdict(group) for group in groups if not group.is_clique],
        "oversized_titles": oversized,
    }


def remap_checkpoint(value: Any, updates: dict[str, str]) -> tuple[Any, int]:
    """Replace exact graph IDs, including frame lists and graph-delta endpoints."""
    if isinstance(value, str):
        return updates.get(value, value), int(value in updates)
    count = 0
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index], changed = remap_checkpoint(item, updates)
            count += changed
    elif isinstance(value, dict):
        for key, item in value.items():
            value[key], changed = remap_checkpoint(item, updates)
            count += changed
    return value, count


def path_issues(
    payload: dict, conn: sqlite3.Connection, updates: dict[str, str]
) -> list[dict]:
    """Check saved walks; in dry runs also predict newly collapsed nodes/edges."""
    members: dict[str, list[str]] = {}
    for old, new in updates.items():
        members.setdefault(new, [new]).append(old)

    @lru_cache(maxsize=100_000)
    def node_exists(node_id: str) -> bool:
        kind, _, db_id = node_id.partition(":")
        table = {"paper": "papers", "author": "authors"}.get(kind)
        return bool(
            table
            and conn.execute(f"SELECT 1 FROM {table} WHERE id = ?", (db_id,)).fetchone()
        )

    @lru_cache(maxsize=100_000)
    def edge_exists(source: str, target: str, kind: str) -> bool:
        source_type, _, source_id = source.partition(":")
        target_type, _, target_id = target.partition(":")
        if kind in {"citing", "cited_by"} and source_type == target_type == "paper":
            if kind == "cited_by":
                source_id, target_id = target_id, source_id
            return (
                conn.execute(
                    "SELECT 1 FROM citations WHERE citing_paper_id=? AND cited_paper_id=?",
                    (source_id, target_id),
                ).fetchone()
                is not None
            )
        if kind == "authorship" and {source_type, target_type} == {"paper", "author"}:
            if source_type == "author":
                source_id, target_id = target_id, source_id
            return (
                conn.execute(
                    "SELECT 1 FROM authorship WHERE paper_id=? AND author_id=?",
                    (source_id, target_id),
                ).fetchone()
                is not None
            )
        return False

    def projected_edge_exists(source: str, target: str, kind: str) -> bool:
        source, target = updates.get(source, source), updates.get(target, target)
        if source == target and kind in {"citing", "cited_by"}:
            return False
        return any(
            edge_exists(original_source, original_target, kind)
            for original_source in members.get(source, [source])
            for original_target in members.get(target, [target])
        )

    issues = []
    seen = {}
    for index, entry in enumerate(payload["paths"]):
        reasons = set()
        nodes = [step["node"].get("node_id", "") for step in entry["path"]]
        mapped = tuple(updates.get(node, node) for node in nodes)
        if not nodes:
            reasons.add("empty_path")
        if "moves" in entry and entry["moves"] != len(nodes) - 1:
            reasons.add("move_count_mismatch")
        if len(set(mapped)) < len(mapped):
            reasons.add("repeated_node")
        if len(set(mapped)) < len(set(nodes)):
            reasons.add("merge_creates_repeated_node")
        if mapped in seen:
            reasons.add("duplicate_path")
        else:
            seen[mapped] = index
        if any(not node_exists(node) for node in walk_node_ids({"paths": [entry]})):
            reasons.add("missing_node")
        if nodes and entry.get("start_node") != nodes[0]:
            reasons.add("start_node_mismatch")
        for step_index, step in enumerate(entry["path"]):
            edge = step.get("edge_taken")
            if step_index == 0:
                if edge is not None:
                    reasons.add("unexpected_start_edge")
                continue
            if not edge or (edge.get("from"), edge.get("to")) != (
                nodes[step_index - 1],
                nodes[step_index],
            ):
                reasons.add("edge_endpoint_mismatch")
            if edge:
                source, target = edge.get("from", ""), edge.get("to", "")
                if not projected_edge_exists(source, target, edge.get("edge_type", "")):
                    reasons.add("missing_edge")
                if source != target and updates.get(source, source) == updates.get(
                    target, target
                ):
                    reasons.add("merge_removes_self_citation")
        if reasons:
            issues.append(
                {
                    "path_index": index,
                    "path_id": entry.get("path_id"),
                    "reasons": sorted(reasons),
                }
            )
    return issues


def rewrite_paths(
    payload: dict, updates: dict[str, str], reader: DBReader, db_path: Path
) -> None:
    apply_node_id_updates_to_walk_data(payload, updates)
    winners = set(updates.values())
    cache = {}
    for entry in payload["paths"]:
        for step in entry["path"]:
            node_id = step["node"].get("node_id", "")
            if node_id in winners:
                if node_id not in cache:
                    cache[node_id] = build_node_payload(node_id, None, reader)
                step["node"].update(cache[node_id])
            edge = step.get("edge_taken")
            if (
                edge
                and edge.get("edge_type") == "authorship"
                and {edge.get("from"), edge.get("to")} & winners
                and edge.get("from")
                and edge.get("to")
            ):
                paper, author = edge["from"], edge["to"]
                if paper.startswith("author:"):
                    paper, author = author, paper
                row = reader.conn.execute(
                    "SELECT position FROM authorship WHERE paper_id=? AND author_id=?",
                    (paper.split(":", 1)[1], author.split(":", 1)[1]),
                ).fetchone()
                if row is not None:
                    edge["position"] = row[0]
    metadata = payload.setdefault("metadata", {})
    for key in DB_METADATA_KEYS:
        metadata[key] = str(db_path)


def process_path_file(
    task: tuple[Path, Path, Path, Path, dict[str, str], bool],
) -> dict:
    """Own one file and independent read-only database handles in this process."""
    path, paths_root, run_dir, db_path, updates, apply = task
    reader = None
    conn = open_database(db_path, readonly=True)
    try:
        logger.info("Scanning {}", path)
        relative = path.relative_to(paths_root)
        payload = orjson.loads(path.read_bytes())
        record = {"file": str(relative)}
        if isinstance(payload, dict) and "paths" in payload:
            if apply:
                reader = DBReader(
                    db_path,
                    prepared_cache_dir=None,
                    cache_max_entries=10_000,
                    sqlite_cache_size_kib=65_536,
                    sqlite_mmap_size_bytes=0,
                )
            payload = read_walk_data(path)
            record["kind"] = "sampled_paths"
            record["path_count"] = len(payload["paths"])
            record["remapped_node_ids"] = len(walk_node_ids(payload) & updates.keys())
            if apply:
                rewrite_paths(payload, updates, reader, db_path)
            record["issues"] = path_issues(payload, conn, {} if apply else updates)
            removed_indices = {
                issue["path_index"]
                for issue in record["issues"]
                if set(issue["reasons"]) - {"duplicate_path"}
            }
            record["paths_to_remove"] = len(removed_indices)
            if apply:
                removed = [
                    entry
                    for index, entry in enumerate(payload["paths"])
                    if index in removed_indices
                ]
                payload["paths"] = [
                    entry
                    for index, entry in enumerate(payload["paths"])
                    if index not in removed_indices
                ]
                for index, entry in enumerate(payload["paths"]):
                    entry["path_index"] = index
                for key in ("n", "complete_paths"):
                    if key in payload["metadata"]:
                        payload["metadata"][key] = len(payload["paths"])
                if removed:
                    archive = run_dir / "removed_paths" / relative
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    write_json_atomic(
                        archive, {"paths": removed, "issues": record["issues"]}
                    )
                write_json_atomic(path, payload)
                if walk_node_ids(payload) & updates.keys():
                    raise RuntimeError(f"Removed paper IDs remain in {path}")
        elif isinstance(payload, dict) and "completed_traces" in payload:
            record["kind"] = "checkpoint"
            payload, count = remap_checkpoint(payload, updates)
            record["remapped_references"] = count
            # A remap cannot repair DFS state or stale graph deltas. Keep an
            # inspectable archive, explicitly unusable as a sampler checkpoint.
            record["action"] = "archive_only_restart_sampling"
            if apply:
                payload["version"] = "deduplicated_archive_not_resumable"
                for key in DB_METADATA_KEYS:
                    payload[key] = str(db_path)
                destination = (
                    run_dir
                    / "checkpoint_archive"
                    / relative.with_suffix(".json.disabled")
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                write_json_atomic(destination, payload)
                path.unlink()
        else:
            raise ValueError(f"Unrecognized JSON file under paths root: {path}")
        return record
    finally:
        if reader is not None:
            reader.close()
        conn.close()


def run_deduplication(
    db_path: Path,
    paths_root: Path,
    *,
    apply: bool = False,
    max_group_size: int = 50,
    num_workers: int = 4,
) -> dict:
    """Dry-run by default; back up both inputs before updating them in place."""
    db_path, paths_root = db_path.resolve(), paths_root.resolve()
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if max_group_size < 2:
        raise ValueError("max_group_size must be at least 2")
    if not db_path.is_file() or not paths_root.is_dir():
        raise ValueError("Database and paths directory must exist")
    if paths_root in db_path.parents:
        raise ValueError("Database must be outside paths-root for consistent backups")
    files = sorted(paths_root.rglob("*.json"))
    if not any(path.name == "sampled_paths.json" for path in files):
        raise ValueError("No sampled_paths.json files found")
    run_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{paths_root.name}.paper_dedup_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_",
            dir=paths_root.parent,
        )
    )
    backup_db = run_dir / "backup" / "database" / db_path.name
    backup_paths = run_dir / "backup" / "paths"
    report = {
        "status": "running",
        "source_db": str(db_path),
        "run_dir": str(run_dir),
        "backup_db": str(backup_db) if apply else None,
        "backup_paths": str(backup_paths) if apply else None,
        "paths_root": str(paths_root),
        "apply": apply,
        "num_workers": num_workers,
        "timings_seconds": {},
        "files": [],
    }
    report_path = run_dir / "report.json"
    write_json_atomic(report_path, report)
    conn = open_database(db_path, readonly=True)
    changes_started = False
    try:
        if apply:
            started = time.monotonic()
            logger.info("Backing up database and paths to {}", run_dir / "backup")
            backup_db.parent.mkdir(parents=True)
            with closing(sqlite3.connect(backup_db)) as destination:
                conn.backup(destination)
            shutil.copytree(paths_root, backup_paths)
            report["timings_seconds"]["backup"] = time.monotonic() - started
            report["backup_complete"] = True
            write_json_atomic(report_path, report)
            conn.close()
            conn = open_database(db_path, readonly=False)
        started = time.monotonic()
        groups, plan = make_plan(conn, max_group_size, num_workers)
        report["timings_seconds"]["audit"] = time.monotonic() - started
        updates = plan["node_id_updates"]
        write_json_atomic(run_dir / "id_mapping.json", plan)
        report.update(
            groups_to_merge=len(plan["merges"]),
            papers_to_remove=len(updates),
            ambiguous_groups=len(plan["ambiguous_groups"]),
            oversized_titles=len(plan["oversized_titles"]),
        )
        if apply:
            started = time.monotonic()
            changes_started = True
            conn.execute("BEGIN IMMEDIATE")
            # Preserve remaps written by path expansion, if this DB has them.
            if table_exists(conn, "path_paper_merges"):
                for old, new in updates.items():
                    conn.execute(
                        "UPDATE path_paper_merges SET winner_id=? WHERE winner_id=?",
                        (new.split(":", 1)[1], old.split(":", 1)[1]),
                    )
            report["merge_stats"] = asdict(
                apply_duplicate_paper_groups(conn, groups, dry_run=False)
            )
            report["timings_seconds"]["merge"] = time.monotonic() - started
        started = time.monotonic()
        tasks = ((path, paths_root, run_dir, db_path, updates, apply) for path in files)
        with process_results(process_path_file, tasks, num_workers) as results:
            for record in results:
                report["files"].append(record)
                write_json_atomic(report_path, report)
        report["timings_seconds"]["paths"] = time.monotonic() - started
        report["status"] = "complete" if apply else "dry_run"
        report["paths_with_issues"] = sum(
            len(record.get("issues", [])) for record in report["files"]
        )
        report["paths_to_remove"] = sum(
            record.get("paths_to_remove", 0) for record in report["files"]
        )
        write_json_atomic(report_path, report)
        return report
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        if changes_started:
            conn.rollback()
            try:
                with closing(open_database(backup_db, readonly=True)) as backup:
                    backup.backup(conn)
                shutil.copytree(backup_paths, paths_root, dirs_exist_ok=True)
                report["restored_from_backup"] = True
            except (OSError, sqlite3.Error) as restore_error:
                report["restored_from_backup"] = False
                report["restore_error"] = str(restore_error)
        write_json_atomic(report_path, report)
        raise
    finally:
        conn.close()
