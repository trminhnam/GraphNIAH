"""Cache sampled paths after they are merged and sliced for one question depth.

Both stages of a data run (questions and needles, then distractors and
prompts) load the same sampled-path JSON files, merge them, and cut every path
into the three-segment initial structure. The result depends only on those
files, the question depth, and the random seed, so it is stored once in a much
smaller JSON file and reused by later runs and worker processes.

``SLICED_PATH_CACHE_DIR`` moves the cache directory (tests use this).
"""

from __future__ import annotations

import functools
import gc
import hashlib
import inspect
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import orjson

from src.questions import construction_run, shared
from src.questions import reachability_long_path_profiles as long_path_profiles
from src.questions.reachability_construction import combine_reachability_walk_data
from src.questions.reachability_crossed_trees import (
    path_tree_cache_key,
    storage_fingerprint,
)

SLICED_PATH_CACHE_DIR_ENV = "SLICED_PATH_CACHE_DIR"
DEFAULT_SLICED_PATH_CACHE_DIR = (
    Path(__file__).resolve().parents[2] / "resources/cache/sliced-paths"
)


class SlicedWalkData(NamedTuple):
    """Merged walk data whose ``paths`` are sliced, plus slicing diagnostics."""

    walk_data: dict[str, Any]
    rejection_counts: dict[str, int]
    had_original_paths: bool


def sliced_path_cache_dir() -> Path:
    """Return the cache directory, which ``SLICED_PATH_CACHE_DIR`` can move."""
    configured = os.environ.get(SLICED_PATH_CACHE_DIR_ENV, "").strip()
    return Path(configured) if configured else DEFAULT_SLICED_PATH_CACHE_DIR


@functools.cache
def slicing_code_fingerprint() -> str:
    """Hash the code that loads, merges, and slices paths.

    Any edit to this code changes the cache key, so a cache file is never
    reused with different slicing rules.
    """
    sources = [
        inspect.getsource(sys.modules[__name__]),
        inspect.getsource(long_path_profiles),
        inspect.getsource(shared),
        inspect.getsource(combine_reachability_walk_data),
        inspect.getsource(path_tree_cache_key),
        inspect.getsource(construction_run.load_walk_data),
        inspect.getsource(construction_run.source_paths_value),
    ]
    return hashlib.sha256("\n".join(sources).encode("utf-8")).hexdigest()


def sliced_path_cache_key(
    paths_files: Sequence[Path],
    *,
    question_hops: int,
    random_seed: int | None,
) -> dict[str, Any]:
    """Return everything the sliced result depends on.

    The path strings are part of the key because merging stores them in every
    entry; the size and modification time detect rewritten input files.
    """
    return {
        "code": slicing_code_fingerprint(),
        "paths_files": [str(path) for path in paths_files],
        "storage": [storage_fingerprint(Path(path)) for path in paths_files],
        "question_hops": question_hops,
        "random_seed": random_seed,
    }


def sliced_path_cache_file(key: dict[str, Any], cache_dir: Path) -> Path:
    """Return the cache file named by a hash of ``key``."""
    digest = hashlib.sha256(orjson.dumps(key, option=orjson.OPT_SORT_KEYS)).hexdigest()
    return Path(cache_dir) / f"sliced-paths-{digest[:24]}.json"


def read_sliced_path_cache(
    cache_file: Path,
    key: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the cached payload, or None when it is absent, unreadable, or stale."""
    try:
        payload = orjson.loads(cache_file.read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("key") != key:
        return None
    return payload


def write_sliced_path_cache(cache_file: Path, payload: dict[str, Any]) -> None:
    """Publish the cache atomically through a uniquely named temporary file.

    Concurrent writers store the same content, and the last ``os.replace`` wins.
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=cache_file.parent, prefix=f"{cache_file.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(orjson.dumps(payload))
        os.replace(temporary, cache_file)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def saved_traversal_modes(
    path_payloads: Sequence[tuple[Path, dict[str, Any]]],
) -> list[list[str]]:
    """Return the traversal mode saved in each sampled-path file."""
    return [
        [
            str(source_path),
            str((payload.get("metadata") or {}).get("traversal_mode") or "").strip(),
        ]
        for source_path, payload in path_payloads
    ]


def validate_traversal_modes(
    saved_modes: Sequence[Sequence[str]],
    traversal_mode: str,
) -> None:
    """Reject sampled paths that were collected with another traversal mode."""
    for source_path, saved_mode in saved_modes:
        if saved_mode and saved_mode != traversal_mode:
            raise ValueError(
                f"Traversal mode mismatch in {source_path}: "
                f"{saved_mode!r} != {traversal_mode!r}."
            )


def ignore_message(message: str) -> None:
    """Discard a progress message."""
    del message


def load_sliced_walk_data(
    paths_files: Sequence[Path],
    *,
    question_hops: int,
    random_seed: int | None,
    traversal_mode: str,
    log: Callable[[str], None] = ignore_message,
) -> SlicedWalkData:
    """Load, merge, and slice sampled paths, reusing a valid cached result."""
    key = sliced_path_cache_key(
        paths_files,
        question_hops=question_hops,
        random_seed=random_seed,
    )
    cache_file = None
    if not any(storage.get("missing") for storage in key["storage"]):
        cache_file = sliced_path_cache_file(key, sliced_path_cache_dir())
        payload = read_sliced_path_cache(cache_file, key)
        if payload is not None:
            validate_traversal_modes(payload["traversal_modes"], traversal_mode)
            log(
                f"Loaded {len(payload['walk_data']['paths']):,} sliced paths "
                f"from cache {cache_file}."
            )
            return SlicedWalkData(
                payload["walk_data"],
                payload["rejection_counts"],
                payload["had_original_paths"],
            )

    log(f"Loading sampled paths from {len(paths_files)} file(s)...")
    path_payloads = construction_run.load_walk_data_files(paths_files)
    traversal_modes = saved_traversal_modes(path_payloads)
    validate_traversal_modes(traversal_modes, traversal_mode)
    log("Merging and deduplicating sampled paths...")
    walk_data = combine_reachability_walk_data(path_payloads)
    del path_payloads
    original_paths = list(walk_data.get("paths") or [])
    had_original_paths = bool(original_paths)
    log(f"Slicing {len(original_paths):,} sampled paths...")
    paths, rejection_counts = long_path_profiles.slice_long_path_entries(
        original_paths,
        question_hops=question_hops,
        random_seed=random_seed,
    )
    del original_paths
    walk_data["paths"] = paths

    # Skip saving when an input file changed while it was being read.
    unchanged = key == sliced_path_cache_key(
        paths_files,
        question_hops=question_hops,
        random_seed=random_seed,
    )
    if cache_file is not None and unchanged:
        try:
            write_sliced_path_cache(
                cache_file,
                {
                    "key": key,
                    "traversal_modes": traversal_modes,
                    "walk_data": walk_data,
                    "rejection_counts": rejection_counts,
                    "had_original_paths": had_original_paths,
                },
            )
            log(f"Saved sliced-path cache {cache_file}.")
        except OSError as error:
            log(f"Could not save sliced-path cache {cache_file}: {error}")
    return SlicedWalkData(walk_data, rejection_counts, had_original_paths)


def load_question_walk_data(
    paths_files: Sequence[Path],
    *,
    question_hops: int,
    random_seed: int | None,
    traversal_mode: str,
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Return sliced walk data for question construction.

    Raise when the input has paths but none of them is long enough for the
    three segments of the initial structure.
    """
    sliced = load_sliced_walk_data(
        paths_files,
        question_hops=question_hops,
        random_seed=random_seed,
        traversal_mode=traversal_mode,
        log=log,
    )
    walk_data = sliced.walk_data
    paths = walk_data["paths"]
    if not paths and sliced.had_original_paths:
        details = "; ".join(
            f"{reason} ({count})"
            for reason, count in sorted(sliced.rejection_counts.items())
        )
        raise ValueError(
            "No input path can supply the requested three segments"
            + (f": {details}" if details else ".")
        )
    walk_metadata = walk_data.setdefault("metadata", {})
    walk_metadata["question_hops"] = question_hops
    walk_metadata["long_path_profile_policy"] = (
        long_path_profiles.LONG_PATH_PROFILE_POLICY
    )
    gc.collect()
    log(f"Prepared {len(paths):,} paths.")
    return walk_data
