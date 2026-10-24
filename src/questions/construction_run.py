"""Shared run plumbing for path-based question constructors.

Loads sampled-path files, resolves the graph database, validates path edges
against the graph, and tracks used paths, run state, and per-bucket outputs.
"""

from __future__ import annotations

import argparse
import hashlib
from os import PathLike
from pathlib import Path
import sys
from typing import Any, Sequence

import orjson

from src.graph.resolution import resolve_latest_graph_db_paths
from src.questions.ambiguity import AmbiguityChecker, missing_ambiguity_node_ids
from src.utils.resume import write_json_output


DEFAULT_GRAPH_DIR = Path("resources") / "semantic_scholar_graph"
DEFAULT_GRAPH_DB = DEFAULT_GRAPH_DIR / "semantic_scholar_paths.db"
MAX_PATH_EDGE_VALIDATION_ERRORS = 10
DEFAULT_GRAPH_SOURCE = Path("resources") / "semantic_scholar_graph"
DEFAULT_TARGET_QUESTIONS = None
USED_PATH_IDS_FIELD = "path_ids"


def load_walk_data(paths_file: Path) -> dict[str, Any]:
    if not paths_file.exists():
        raise FileNotFoundError(f"Paths file not found: {paths_file}")
    with open(paths_file, encoding="utf-8") as handle:
        return orjson.loads(handle.read())


def resolve_question_graph_paths(
    walk_data: dict[str, Any],
    *,
    graph_dir: Path,
    db_path_hint: str | None,
) -> tuple[Path, Path, str]:
    del walk_data
    del graph_dir
    del db_path_hint
    return resolve_latest_graph_db_paths(DEFAULT_GRAPH_DB, None)


def values_match(left: Any, right: Any) -> bool:
    """Return True when two edge-attribute values match after int normalization."""
    if left == right:
        return True
    try:
        return int(left) == int(right)
    except (TypeError, ValueError):
        return False


def graph_edge_candidates(
    checker: AmbiguityChecker,
    source_node_id: str,
    edge_type: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Return graph edges matching the checker direction and edge type."""
    return [
        (target_node_id, attrs)
        for target_node_id, attrs in checker.graph.adj.get(source_node_id, [])
        if attrs.get("edge_type") == edge_type
    ]


def format_validation_error(
    *,
    graph_path: Path,
    path_index: Any,
    edge_index: int,
    source_node_id: str,
    target_node_id: str,
    edge_type: str,
    reasons: list[str],
) -> str:
    """Return one path-edge validation error message."""
    return (
        f"graph={graph_path} path_index={path_index} edge_index={edge_index} "
        f"{source_node_id} --{edge_type}--> {target_node_id}: "
        + "; ".join(reasons)
    )


def validate_path_edge(
    checker: AmbiguityChecker,
    graph_path: Path,
    path_index: Any,
    edge_index: int,
    previous_step: dict[str, Any],
    current_step: dict[str, Any],
) -> str | None:
    """Return a validation error for one sampled path edge, if invalid."""
    source_node_id = str(previous_step.get("node", {}).get("node_id") or "")
    target_node_id = str(current_step.get("node", {}).get("node_id") or "")
    edge = current_step.get("edge_taken") or {}
    edge_type = str(edge.get("edge_type") or "")
    reasons: list[str] = []

    if not source_node_id:
        reasons.append("source node_id is missing from sampled path")

    if not target_node_id:
        reasons.append("target node_id is missing from sampled path")

    if not edge_type:
        reasons.append("edge_type is missing from sampled path")

    edge_from = str(edge.get("from") or "")
    edge_to = str(edge.get("to") or "")
    if edge_from and edge_from != source_node_id:
        reasons.append(f"edge.from={edge_from} does not match path source")
    if edge_to and edge_to != target_node_id:
        reasons.append(f"edge.to={edge_to} does not match path target")

    if reasons:
        return format_validation_error(
            graph_path=graph_path,
            path_index=path_index,
            edge_index=edge_index,
            source_node_id=source_node_id or "?",
            target_node_id=target_node_id or "?",
            edge_type=edge_type or "?",
            reasons=reasons,
        )

    if missing_ambiguity_node_ids(checker, [source_node_id, target_node_id]):
        return None

    if source_node_id and target_node_id and edge_type:
        candidates = graph_edge_candidates(checker, source_node_id, edge_type)
        target_candidates = [
            attrs
            for candidate_node_id, attrs in candidates
            if candidate_node_id == target_node_id
        ]
        if not target_candidates:
            if candidates:
                sample_targets = ", ".join(
                    candidate_node_id for candidate_node_id, _attrs in candidates[:5]
                )
                if len(candidates) > 5:
                    sample_targets += ", ..."
                reasons.append(
                    f"target is not a {edge_type!r} neighbor of source "
                    f"(candidate count={len(candidates)}: {sample_targets})"
                )
            else:
                reasons.append(f"source has no outgoing {edge_type!r} edges in graph")
        elif edge_type == "authorship" and edge.get("position") is not None:
            sampled_position = edge.get("position")
            graph_positions = [attrs.get("position") for attrs in target_candidates]
            if not any(
                values_match(position, sampled_position)
                for position in graph_positions
            ):
                reasons.append(
                    "authorship position mismatch "
                    f"(sampled={sampled_position}, graph={graph_positions})"
                )

    if not reasons:
        return None

    return format_validation_error(
        graph_path=graph_path,
        path_index=path_index,
        edge_index=edge_index,
        source_node_id=source_node_id or "?",
        target_node_id=target_node_id or "?",
        edge_type=edge_type or "?",
        reasons=reasons,
    )


def collect_path_edge_validation_errors(
    walk_data: dict[str, Any],
    checker: AmbiguityChecker,
    graph_path: Path,
    max_errors: int = MAX_PATH_EDGE_VALIDATION_ERRORS,
) -> tuple[int, list[str]]:
    """Return sampled-path edge validation failures for the checker graph."""
    total_errors = 0
    errors: list[str] = []
    for fallback_path_index, path_entry in enumerate(walk_data.get("paths", [])):
        path = path_entry.get("path", [])
        path_index = path_entry.get("path_index", fallback_path_index)
        for edge_index in range(1, len(path)):
            error = validate_path_edge(
                checker,
                graph_path,
                path_index,
                edge_index,
                path[edge_index - 1],
                path[edge_index],
            )
            if error is None:
                continue
            total_errors += 1
            if len(errors) < max_errors:
                errors.append(error)
    return total_errors, errors


def print_path_edge_validation_errors(
    total_errors: int,
    errors: list[str],
) -> None:
    """Print sampled-path validation failures before question construction."""
    print(
        "\n[construct_questions] ERROR: sampled paths do not match the "
        "ambiguity-check graph."
        f"\n  Invalid edges  : {total_errors}",
        file=sys.stderr,
    )
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    if total_errors > len(errors):
        print(
            f"  ... {total_errors - len(errors)} more invalid edge(s)",
            file=sys.stderr,
        )


def stable_path_seed(random_seed: int | None, path_key: Any) -> int:
    """Return a deterministic per-path RNG seed."""
    digest = hashlib.sha256(
        f"{random_seed!r}:{path_key!r}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def normalize_paths_files(
    paths_file: str | PathLike[str] | Sequence[str | PathLike[str]],
) -> tuple[Path, ...]:
    """Return one or more sampled-path files as a tuple."""
    if isinstance(paths_file, (str, PathLike)):
        return (Path(paths_file),)
    paths_files = tuple(Path(path) for path in paths_file)
    if not paths_files:
        raise ValueError("At least one paths file is required.")
    return paths_files


def source_paths_value(paths_files: Sequence[Path]) -> str | list[str]:
    """Return the JSON source_paths value for checkpoint metadata."""
    values = [str(path) for path in paths_files]
    return values[0] if len(values) == 1 else values


def load_walk_data_files(
    paths_files: Sequence[Path],
) -> list[tuple[Path, dict[str, Any]]]:
    """Load sampled-path payloads from all requested files."""
    return [(paths_file, load_walk_data(paths_file)) for paths_file in paths_files]


def load_used_path_ids(path: Path | None) -> set[str]:
    """Load a global accepted-path registry."""
    if path is None or not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        data = orjson.loads(handle.read())

    values: set[str] = set()
    if isinstance(data, list):
        values.update(str(value) for value in data if value is not None)
        return values
    if not isinstance(data, dict):
        raise ValueError(f"Used paths file must be a JSON object or list: {path}")

    for field_name in (USED_PATH_IDS_FIELD, "used_path_ids", "processed_path_ids"):
        field_value = data.get(field_name)
        if isinstance(field_value, list):
            values.update(str(value) for value in field_value if value is not None)
    entries = data.get("entries")
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and entry.get("path_id") is not None:
                values.add(str(entry["path_id"]))
    return values


def write_used_path_ids(
    path: Path | None,
    path_ids: set[str],
    indent: int,
) -> None:
    """Write the global accepted-path registry."""
    if path is None:
        return
    write_json_output(
        path,
        {
            "total_used_paths": len(path_ids),
            USED_PATH_IDS_FIELD: sorted(path_ids),
        },
        indent,
    )


def parse_expected_search_steps_buckets(raw: str) -> list[tuple[str, float, float]]:
    """Parse ``label:min:max[,label:min:max...]`` bucket specifications."""
    bucket_specs: list[tuple[str, float, float]] = []
    seen_labels: set[str] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        pieces = part.split(":")
        if len(pieces) != 3:
            raise argparse.ArgumentTypeError(
                f"bucket must use label:min:max format: {part!r}"
            )
        label = pieces[0].strip()
        if not label:
            raise argparse.ArgumentTypeError(f"bucket label must be non-empty: {part!r}")
        if label in seen_labels:
            raise argparse.ArgumentTypeError(f"duplicate bucket label: {label!r}")
        try:
            bucket_min = float(pieces[1])
            bucket_max = float(pieces[2])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"bucket bounds must be numbers: {part!r}"
            ) from exc
        if bucket_min < 0 or bucket_min >= bucket_max:
            raise argparse.ArgumentTypeError(
                f"bucket bounds must satisfy 0 <= min < max: {part!r}"
            )
        seen_labels.add(label)
        bucket_specs.append((label, bucket_min, bucket_max))
    if not bucket_specs:
        raise argparse.ArgumentTypeError("at least one bucket is required")
    return sorted(bucket_specs, key=lambda spec: (spec[1], spec[2], spec[0]))


BUCKET_OUTPUT_PLACEHOLDER = "{bucket}"


def resolve_bucket_output_path(template: Path, bucket_label: str) -> Path:
    """Return one bucket's output path from a ``{bucket}`` path template."""
    return Path(str(template).replace(BUCKET_OUTPUT_PLACEHOLDER, bucket_label))


def target_questions_increased(
    saved_target: int | None,
    requested_target: int | None,
) -> bool:
    """Return whether a requested quota is larger than a saved quota."""
    if saved_target is None:
        return False
    return requested_target is None or requested_target > saved_target


def load_run_state(
    path: Path | None,
) -> tuple[set[str], set[str], dict[str, int], set[str] | None]:
    """Load shared processed/skipped path state for a multi-bucket run.

    The final element is the bucket-label set the state was written with, or
    ``None`` for legacy state files that predate bucket-label tracking.
    """
    if path is None or not path.exists():
        return set(), set(), {}, None
    with path.open(encoding="utf-8") as handle:
        data = orjson.loads(handle.read())
    if not isinstance(data, dict):
        raise ValueError(f"Run state file must be a JSON object: {path}")
    processed = {
        str(value)
        for value in data.get("processed_path_ids", [])
        if value is not None
    }
    skipped = {
        str(value)
        for value in data.get("skipped_path_ids", [])
        if value is not None
    }
    raw_counts = data.get("skip_reason_counts") or {}
    skip_reason_counts = (
        {str(reason): int(count) for reason, count in raw_counts.items()}
        if isinstance(raw_counts, dict)
        else {}
    )
    raw_labels = data.get("bucket_labels")
    bucket_labels = (
        {str(label) for label in raw_labels}
        if isinstance(raw_labels, list)
        else None
    )
    return processed, skipped, skip_reason_counts, bucket_labels
