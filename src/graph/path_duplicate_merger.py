"""Audit-confirmed merges with durable UUID remaps for sampled paths."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from merge_graph_databases import (
    DuplicatePaperGroup,
    apply_duplicate_paper_groups,
    apply_node_id_updates_to_walk_data,
    find_duplicate_paper_groups,
    inspect_duplicate_paper_group,
    write_json_atomic,
)
from src.graph.path_expansion_writer import PathExpansionWriter
from src.graph.sampled_paths import iter_path_nodes, load_paths_payload


@dataclass
class PathDuplicateMerger:
    writer: PathExpansionWriter
    paths_files: list[Path]
    remaps: dict[str, str] = field(default_factory=dict)
    audit_groups: dict[str, tuple[str, ...]] | None = None

    def prepare(self) -> None:
        """Recover remaps even if a previous process died before writing JSON."""
        with self.writer.network.expansion_batch():
            self.writer.network.db.conn.execute(
                "CREATE TABLE IF NOT EXISTS path_paper_merges ("
                "loser_id TEXT PRIMARY KEY, winner_id TEXT NOT NULL)"
            )
        self.rewrite_paths()

    def load_remaps(self) -> None:
        self.remaps = dict(
            self.writer.network.db.conn.execute(
                "SELECT loser_id, winner_id FROM path_paper_merges"
            ).fetchall()
        )
        for loser in self.remaps:
            winner = self.remaps[loser]
            while winner in self.remaps:
                winner = self.remaps[winner]
            self.remaps[loser] = winner

    def apply(self, group: DuplicatePaperGroup, winner: str) -> bool:
        """Reuse the notebook's clique and oversized-title safeguards once/run."""
        network = self.writer.network
        if self.audit_groups is None:
            logger.info("Auditing duplicate groups before the first path-paper merge")
            groups, _ = find_duplicate_paper_groups(network.db.conn)
            self.audit_groups = {
                paper_id: candidate.paper_ids
                for candidate in groups
                if candidate.is_clique
                for paper_id in candidate.paper_ids
            }
        members = self.audit_groups.get(winner, ())
        if not set(group.paper_ids).issubset(members):
            logger.warning(
                "Skipping merge into {}: not in one unambiguous audit group", winner
            )
            return False

        with network.expansion_batch():
            # Recheck the entire surviving group after prior ingests/merges.
            placeholders = ",".join("?" for _ in members)
            current_ids = tuple(
                row[0]
                for row in network.db.conn.execute(
                    f"SELECT id FROM papers WHERE id IN ({placeholders})", members
                )
            )
            current = inspect_duplicate_paper_group(network.db.conn, current_ids)
            if current is None or not current.is_clique:
                return False
            for paper_id in group.paper_ids:
                lease = network.db.get_expansion_lease("paper", paper_id)
                if (
                    lease is not None
                    and lease["state"] == "expanding"
                    and lease["owner_id"] != network.owner_id
                    and lease["lease_until"] > time.time()
                ):
                    return False
            for loser in group.paper_ids:
                if loser != winner:
                    network.db.conn.execute(
                        "INSERT INTO path_paper_merges (loser_id, winner_id) VALUES (?, ?)",
                        (loser, winner),
                    )
            apply_duplicate_paper_groups(
                network.db.conn, [group], dry_run=False, commit=False
            )
        # Same-connection writes do not change PRAGMA data_version.
        self.writer.committed_neighbors.clear()
        self.load_remaps()
        return True

    def canonical_papers(self, papers: dict[str, str]) -> dict[str, str]:
        """Select surviving UUIDs, including owners absent from the original paths."""
        result = {}
        for paper_id, semantic_id in papers.items():
            winner = self.remaps.get(paper_id, paper_id)
            if winner != paper_id:
                row = self.writer.network.db.get_paper(winner)
                semantic_id = str(row["semantic_scholar_id"] or "")
            result[winner] = semantic_id
        return result

    def rewrite_paths(self) -> None:
        """Atomically replace each supplied file; durable remaps allow retry."""
        self.load_remaps()
        if not self.remaps:
            return
        updates = {f"paper:{old}": f"paper:{new}" for old, new in self.remaps.items()}
        semantic_ids = {
            f"paper:{winner}": str(
                self.writer.network.db.get_paper(winner)["semantic_scholar_id"] or ""
            )
            for winner in set(self.remaps.values())
        }
        for path in dict.fromkeys(self.paths_files):
            payload = load_paths_payload(path)
            collapsed_paths = 0
            for entry in payload["paths"]:
                node_ids = {step["node"]["node_id"] for step in entry["path"]}
                if len({updates.get(node_id, node_id) for node_id in node_ids}) < len(
                    node_ids
                ):
                    collapsed_paths += 1
            for _, _, node in iter_path_nodes(payload):
                replacement = updates.get(node.get("node_id"))
                if replacement:
                    node["semantic_scholar_id"] = semantic_ids[replacement]
            if apply_node_id_updates_to_walk_data(payload, updates):
                write_json_atomic(path, payload)
                logger.info("Updated merged paper UUIDs in {}", path)
                if collapsed_paths:
                    logger.warning(
                        "{}: merging creates repeated nodes in {} path(s); "
                        "resample these paths if distinct nodes are required",
                        path,
                        collapsed_paths,
                    )
