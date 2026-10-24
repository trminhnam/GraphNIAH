from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from loguru import logger


GRAPH_STEM = "graph"
DATETIME_STEM_RE = re.compile(r"^(.+)_(\d{8}_\d{6})$")
TIMESTAMP_SUFFIX_RE = re.compile(r"_\d{8}_\d{6}$")
CHECKPOINT_ARTIFACT_SUFFIXES = (
    ".json",
    ".gexf",
    ".db",
    ".db-shm",
    ".db-wal",
    ".png",
)


def extract_base_stem(stem: str) -> str:
    """Return the base portion of a run stem, stripping datetime suffixes."""
    while True:
        match = DATETIME_STEM_RE.match(stem)
        if not match:
            return stem
        stem = match.group(1)


def checkpoint_timestamp(stem: str) -> str:
    """Return the ``YYYYMMDD_HHMMSS`` suffix from *stem*, or ``""`` when absent."""
    match = TIMESTAMP_SUFFIX_RE.search(stem)
    return match.group(0).lstrip("_") if match else ""


def build_graph_stem(
    base_stem: str = GRAPH_STEM,
    when: datetime | None = None,
) -> str:
    """Return ``<base>_YYYYMMDD_HHMMSS`` using the stripped checkpoint base."""
    timestamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    stem_root = extract_base_stem(base_stem) or GRAPH_STEM
    return f"{stem_root}_{timestamp}"


def find_latest_db(
    folder: Path,
    base_stem: str,
    *,
    newer_than: str | None = None,
) -> Path | None:
    """Return the newest checkpoint DB matching *base_stem*."""
    candidates: list[tuple[str, Path]] = []
    for db_file in folder.glob("*.db"):
        match = DATETIME_STEM_RE.match(db_file.stem)
        if not match or match.group(1) != base_stem:
            continue
        timestamp = match.group(2)
        if newer_than is not None and timestamp <= newer_than:
            continue
        candidates.append((timestamp, db_file))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def collect_checkpoint_stems(
    graph_dir: Path,
    base_stem: str,
) -> list[tuple[str, str]]:
    """Return unique checkpoint stems for *base_stem* sorted newest first."""
    checkpoint_stems: set[tuple[str, str]] = set()
    for artifact in graph_dir.iterdir():
        if not artifact.is_file() or artifact.suffix not in CHECKPOINT_ARTIFACT_SUFFIXES:
            continue
        match = DATETIME_STEM_RE.match(artifact.stem)
        if match and match.group(1) == base_stem:
            checkpoint_stems.add((match.group(2), artifact.stem))
    return sorted(checkpoint_stems, key=lambda item: item[0], reverse=True)


def prune_checkpoints(graph_dir: Path, base_stem: str, max_keep: int) -> None:
    """Delete old checkpoint sets, keeping only the *max_keep* newest ones."""
    if max_keep <= 0:
        return

    pruned_files: list[Path] = []
    checkpoint_stems = collect_checkpoint_stems(graph_dir, base_stem)
    for _, old_stem in checkpoint_stems[max_keep:]:
        for extension in CHECKPOINT_ARTIFACT_SUFFIXES:
            old_file = graph_dir / f"{old_stem}{extension}"
            if old_file.exists():
                old_file.unlink()
                pruned_files.append(old_file)

    if pruned_files:
        logger.info(
            "Pruned old checkpoint files → {}",
            ", ".join(str(path) for path in pruned_files),
        )
