from __future__ import annotations

from pathlib import Path
from typing import Any


REACHABILITY_SAFETY_FIELD = "reachability_safety"
REACHABILITY_SAFETY_POLICY_VERSION = 1
REACHABILITY_SAFETY_SCOPE_FIELD = "scope"
REACHABILITY_SAFETY_SCOPE_DATABASE_SNAPSHOT = "database-snapshot"
REACHABILITY_SAFETY_SCOPE_RENDERED_TREES = "rendered-trees"
RESERVED_CONTROL_TOKEN_MARKERS = (
    "<|endofprompt|>",
    "<|endoftext|>",
)


def build_database_snapshot(db_path: str | Path) -> dict[str, Any]:
    """Return a cheap identity snapshot for one SQLite graph database."""
    path = Path(db_path)
    stat = path.stat()
    wal_path = Path(f"{path}-wal")
    wal_size = wal_path.stat().st_size if wal_path.exists() else 0
    wal_mtime_ns = (
        wal_path.stat().st_mtime_ns if wal_path.exists() and wal_size else 0
    )
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "wal_size": wal_size,
        "wal_mtime_ns": wal_mtime_ns,
    }


def build_reachability_safety_certificate(
    db_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Return metadata certifying reachability-safe paper distractors.

    ``paper_index:`` distractors store only a paper ID, so the prompt stage
    reads their reference lists from the live graph. A database that grew
    between the two stages can hand such a distractor a new edge into the
    answer graph, which is why this certificate pins the database identity.
    """
    return {
        "validated": True,
        "policy_version": REACHABILITY_SAFETY_POLICY_VERSION,
        REACHABILITY_SAFETY_SCOPE_FIELD: (
            REACHABILITY_SAFETY_SCOPE_DATABASE_SNAPSHOT
        ),
        "database_snapshot": dict(db_snapshot),
    }


def build_rendered_tree_safety_certificate() -> dict[str, Any]:
    """Return metadata certifying distractors rendered from stored trees.

    Rendered-edge negative trees serialize every paper block they display and
    every reference inside it, and the prompt stage renders exactly that stored
    payload instead of reading live adjacency. Adding citations to the database
    therefore cannot change what one of these prompts shows, so the certificate
    carries no database snapshot and stays valid across a resumed run.
    """
    return {
        "validated": True,
        "policy_version": REACHABILITY_SAFETY_POLICY_VERSION,
        REACHABILITY_SAFETY_SCOPE_FIELD: (
            REACHABILITY_SAFETY_SCOPE_RENDERED_TREES
        ),
    }


def reachability_safety_certificate_matches(
    certificate: Any,
    db_path: str | Path,
) -> bool:
    """Return whether one safety certificate still covers its distractors.

    A certificate written before this scope field existed has no ``scope`` key
    and is read as a database snapshot, so old files keep their old meaning.
    """
    if not isinstance(certificate, dict):
        return False
    if certificate.get("validated") is not True:
        return False
    if certificate.get("policy_version") != REACHABILITY_SAFETY_POLICY_VERSION:
        return False
    scope = certificate.get(
        REACHABILITY_SAFETY_SCOPE_FIELD,
        REACHABILITY_SAFETY_SCOPE_DATABASE_SNAPSHOT,
    )
    if scope == REACHABILITY_SAFETY_SCOPE_RENDERED_TREES:
        return True
    if scope != REACHABILITY_SAFETY_SCOPE_DATABASE_SNAPSHOT:
        return False
    saved_snapshot = certificate.get("database_snapshot")
    if not isinstance(saved_snapshot, dict):
        return False
    try:
        return saved_snapshot == build_database_snapshot(db_path)
    except OSError:
        return False


def text_contains_reserved_control_token(value: Any) -> bool:
    """Return whether text contains a model control-token marker."""
    text = str(value or "")
    return any(marker in text for marker in RESERVED_CONTROL_TOKEN_MARKERS)
