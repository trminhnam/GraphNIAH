"""Seed missing path nodes into a graph database and drop paths the bulk data disproves.

Used by ``--seed-missing-papers`` (seeding) and ``--drop-disproved-paths``
(dropping) in ``endpoints/expand_papers_in_paths.py`` and
``endpoints/expand_path_neighborhood.py``. Seeded rows keep the UUIDs saved in
the path files, so the files stay valid. After expansion, a path is dropped
only when the bulk data proves it wrong: one of its papers was not found, or
both ends of a saved edge are verified but the edge does not exist. Papers
that failed for other reasons leave their paths in place as unverified, so a
rerun can finish them.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import orjson
from loguru import logger

from sample_paths import parse_published_date_value
from src.graph.db_network import SQLiteCitationNetwork
from src.graph.sampled_paths import (
    collect_path_nodes,
    load_paths_payload,
    refresh_path_count_metadata,
    write_paths_payload_atomic,
)
from src.graph.storage import DatabaseManager
from src.setup.constants import (
    EDGE_AUTHORSHIP,
    EDGE_CITED_BY,
    EDGE_CITING,
    NODE_AUTHOR,
    NODE_PAPER,
)

# Expansion statuses that prove a paper is absent from the bulk data.
MISSING_IN_BULK_STATUSES = frozenset(
    {"not_found", "no_semantic_scholar_id", "cached_not_found"}
)
PAPER_SEED_FIELDS = (
    "title",
    "arxiv_id",
    "semantic_scholar_id",
    "abstract",
    "published",
    "venue",
    "pdf_url",
    "text_path",
)
NODE_VERIFIED = "verified"
NODE_MISSING = "missing"
NODE_UNVERIFIED = "unverified"


@dataclass
class SeedCounts:
    papers: int = 0
    authors: int = 0
    remapped: int = 0
    conflicts: list[str] = field(default_factory=list)


@dataclass
class PathFileReport:
    kept: int = 0
    dropped: int = 0
    unverified: int = 0
    reasons: Counter[str] = field(default_factory=Counter)


def node_text(node: dict[str, Any], key: str) -> str:
    return str(node.get(key) or "").strip()


def seed_missing_path_nodes(
    network: SQLiteCitationNetwork, nodes: dict[str, dict[str, Any]]
) -> SeedCounts:
    """Insert path papers/authors absent from the database under their saved UUIDs.

    Seeded papers are unexplored roots (``depth = 0``). A missing paper whose
    Semantic Scholar or arXiv ID already belongs to another row is remapped to
    that row in ``path_paper_merges`` (created by ``PathDuplicateMerger.prepare``),
    so the merger rewrites the path files. Identity conflicts stay unseeded.
    """
    counts = SeedCounts()
    with network.expansion_batch():
        for node_id, node in nodes.items():
            node_type, _, db_id = node_id.partition(":")
            if node_type == NODE_PAPER:
                seed_paper(network.db, db_id, node, counts)
            elif node_type == NODE_AUTHOR:
                seed_author(network.db, db_id, node, counts)
    return counts


def seed_path_files(
    network: SQLiteCitationNetwork, paths_files: list[Path]
) -> SeedCounts:
    """Seed every node of the supplied path files and log the outcome.

    Callers reload their merger's remaps afterward.
    """
    counts = seed_missing_path_nodes(network, collect_path_nodes(paths_files))
    logger.info(
        "Seeded {} paper(s) and {} author(s) under their path UUIDs | "
        "remapped {} path paper(s) to existing rows",
        counts.papers,
        counts.authors,
        counts.remapped,
    )
    for conflict in counts.conflicts:
        logger.warning(conflict)
    return counts


def seed_paper(
    db: DatabaseManager, paper_id: str, node: dict[str, Any], counts: SeedCounts
) -> None:
    if db.get_paper(paper_id) is not None:
        return
    semantic_id = node_text(node, "semantic_scholar_id")
    arxiv_id = node_text(node, "arxiv_id")
    owner = db.get_paper_by_semantic_scholar_id(semantic_id)
    if owner is None:
        owner = db.get_paper_by_arxiv_id(arxiv_id)
        owner_semantic_id = str(owner["semantic_scholar_id"] or "") if owner else ""
        if owner_semantic_id and semantic_id and owner_semantic_id != semantic_id:
            counts.conflicts.append(
                f"paper:{paper_id} shares arXiv ID {arxiv_id} with paper:{owner['id']} "
                "but has a different Semantic Scholar ID; not seeded"
            )
            return
    if owner is not None:
        db.conn.execute(
            "INSERT INTO path_paper_merges (loser_id, winner_id) VALUES (?, ?)",
            (paper_id, str(owner["id"])),
        )
        counts.remapped += 1
        return
    db.insert_paper(
        paper_id=paper_id,
        depth=0,
        **{key: node_text(node, key) for key in PAPER_SEED_FIELDS},
    )
    counts.papers += 1


def seed_author(
    db: DatabaseManager, author_id: str, node: dict[str, Any], counts: SeedCounts
) -> None:
    if db.get_author(author_id) is not None:
        return
    semantic_id = node_text(node, "semantic_scholar_id")
    orcid = node_text(node, "orcid")
    owner = db.get_author_by_semantic_scholar_id(semantic_id)
    if owner is None and orcid:
        owner = db.conn.execute(
            "SELECT * FROM authors WHERE orcid = ? LIMIT 1", (orcid,)
        ).fetchone()
    if owner is not None:
        counts.conflicts.append(
            f"author:{author_id} matches existing author:{owner['id']}; not seeded"
        )
        return
    db.insert_author(
        author_id=author_id,
        name=node_text(node, "name"),
        semantic_scholar_id=semantic_id,
        orcid=orcid,
    )
    counts.authors += 1


def path_edge_exists(
    conn: sqlite3.Connection, source: str, target: str, edge_type: str
) -> bool:
    """Check one saved ``edge_taken`` against the citation/authorship tables."""
    source_type, _, source_id = source.partition(":")
    target_type, _, target_id = target.partition(":")
    if (
        edge_type in {EDGE_CITING, EDGE_CITED_BY}
        and source_type == target_type == NODE_PAPER
    ):
        if edge_type == EDGE_CITED_BY:
            source_id, target_id = target_id, source_id
        sql = "SELECT 1 FROM citations WHERE citing_paper_id = ? AND cited_paper_id = ?"
    elif edge_type == EDGE_AUTHORSHIP and {source_type, target_type} == {
        NODE_PAPER,
        NODE_AUTHOR,
    }:
        if source_type == NODE_AUTHOR:
            source_id, target_id = target_id, source_id
        sql = "SELECT 1 FROM authorship WHERE paper_id = ? AND author_id = ?"
    else:
        return False
    return conn.execute(sql, (source_id, target_id)).fetchone() is not None


@dataclass
class PathChecker:
    """Classify path nodes/edges after expansion; node states are cached.

    Without ``cutoff``, a paper is verified only after a full expansion. With
    ``cutoff`` (neighborhood expansion), papers expanded under the neighbor
    date filter are verified too. A filtered expansion stores an edge only to a
    neighbor dated on/after the cutoff, so a missing citation edge proves
    nothing when neither end has a full edge list or such a date; that path
    is kept as unverified instead of dropped.
    """

    conn: sqlite3.Connection
    remaps: dict[str, str]
    missing_in_bulk: set[str]
    cutoff: date | None = None
    nodes: dict[str, tuple[str, bool]] = field(default_factory=dict)

    def canonical(self, node_id: str) -> str:
        node_type, _, db_id = node_id.partition(":")
        if node_type == NODE_PAPER:
            return f"{NODE_PAPER}:{self.remaps.get(db_id, db_id)}"
        return node_id

    def node(self, node_id: str) -> tuple[str, bool]:
        """Return (state, whether a missing edge at this node is conclusive)."""
        if node_id not in self.nodes:
            self.nodes[node_id] = self.lookup_node(node_id)
        return self.nodes[node_id]

    def lookup_node(self, node_id: str) -> tuple[str, bool]:
        node_type, _, db_id = node_id.partition(":")
        if node_type != NODE_PAPER:
            row = self.conn.execute(
                "SELECT 1 FROM authors WHERE id = ?", (db_id,)
            ).fetchone()
            # Authorship comes from the paper's own record, never date-filtered.
            return (NODE_VERIFIED if row is not None else NODE_UNVERIFIED), True
        if db_id in self.missing_in_bulk:
            return NODE_MISSING, False
        row = self.conn.execute(
            "SELECT explored, explored_after_cutoff, published FROM papers WHERE id = ?",
            (db_id,),
        ).fetchone()
        if row is None or not row[0]:
            return NODE_UNVERIFIED, False
        full_edges = not row[1]
        if self.cutoff is None:
            return (NODE_VERIFIED if full_edges else NODE_UNVERIFIED), True
        published = parse_published_date_value(row[2])
        return NODE_VERIFIED, full_edges or (
            published is not None and published >= self.cutoff
        )

    def check(self, entry: dict[str, Any]) -> tuple[list[dict[str, str]], bool]:
        """Return reasons to drop the path and whether any part is unverified."""
        reasons: list[dict[str, str]] = []
        unverified = False
        for step in entry["path"]:
            node_id = self.canonical(node_text(step["node"], "node_id"))
            state, _ = self.node(node_id)
            if state == NODE_MISSING:
                reasons.append({"reason": "missing_node", "node": node_id})
            unverified = unverified or state == NODE_UNVERIFIED
            edge = step.get("edge_taken")
            if not isinstance(edge, dict):
                continue
            source = self.canonical(node_text(edge, "from"))
            target = self.canonical(node_text(edge, "to"))
            source_state, source_conclusive = self.node(source)
            target_state, target_conclusive = self.node(target)
            if source_state != NODE_VERIFIED or target_state != NODE_VERIFIED:
                continue
            edge_type = node_text(edge, "edge_type")
            if path_edge_exists(self.conn, source, target, edge_type):
                continue
            if not (source_conclusive or target_conclusive):
                unverified = True
                continue
            reasons.append(
                {
                    "reason": "missing_edge",
                    "from": source,
                    "to": target,
                    "edge_type": edge_type,
                }
            )
        return reasons, unverified


def dropped_paths_file(paths_json: Path) -> Path:
    """JSONL beside the paths file; ``.jsonl`` keeps it out of ``*.json`` scans."""
    return paths_json.with_name(f"{paths_json.stem}.dropped.jsonl")


def drop_unsupported_paths(paths_json: Path, checker: PathChecker) -> PathFileReport:
    """Remove disproved paths, saving each one with its reasons before rewriting."""
    payload = load_paths_payload(paths_json)
    report = PathFileReport()
    kept, dropped = [], []
    for entry in payload["paths"]:
        reasons, unverified = checker.check(entry)
        if reasons:
            dropped.append(
                {"path_id": entry.get("path_id"), "reasons": reasons, "entry": entry}
            )
            report.reasons.update(reason["reason"] for reason in reasons)
            continue
        kept.append(entry)
        report.unverified += unverified
    report.kept, report.dropped = len(kept), len(dropped)
    if dropped:
        append_dropped_paths(dropped_paths_file(paths_json), dropped)
        payload["paths"] = kept
        refresh_path_count_metadata(payload)
        write_paths_payload_atomic(paths_json, payload)
    return report


def drop_unsupported_paths_in_files(
    paths_files: list[Path], checker: PathChecker
) -> None:
    """Apply ``drop_unsupported_paths`` to each file once and log the counts."""
    for paths_file in dict.fromkeys(paths_files):
        report = drop_unsupported_paths(paths_file, checker)
        logger.info(
            "{} | kept={} | dropped={} (missing_node={}, missing_edge={}) | "
            "unverified={}",
            paths_file,
            report.kept,
            report.dropped,
            report.reasons["missing_node"],
            report.reasons["missing_edge"],
            report.unverified,
        )
        if report.dropped:
            logger.info("Saved dropped paths to {}", dropped_paths_file(paths_file))


def append_dropped_paths(target: Path, records: list[dict[str, Any]]) -> None:
    """Append records, skipping path IDs saved by an interrupted earlier run."""
    saved = set()
    if target.is_file():
        with target.open("rb") as handle:
            saved = {
                orjson.loads(line).get("path_id") for line in handle if line.strip()
            }
    with target.open("ab") as handle:
        for record in records:
            if record["path_id"] is None or record["path_id"] not in saved:
                handle.write(orjson.dumps(record) + b"\n")
