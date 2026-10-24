"""Atomic path expansion with reuse of committed neighbor metadata."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import orjson

from construct_paths_from_semantic_scholar import (
    refresh_current_paper,
    upsert_semantic_paper,
)
from src.graph.db_network import SQLiteCitationNetwork
from src.setup.constants import nid_db_id


@dataclass
class PathExpansionWriter:
    network: SQLiteCitationNetwork
    cache_limit: int = 50_000
    committed_neighbors: OrderedDict[bytes, str] = field(default_factory=OrderedDict)
    data_version: int | None = None
    elapsed_seconds: float = 0.0
    cache_hits: int = 0
    after_cutoff: bool = False
    on_success: Callable[[str], None] | None = None

    def ingest(
        self,
        node_id: str,
        paper: dict[str, Any],
        citations: list[dict[str, Any]],
        references: list[dict[str, Any]],
    ) -> None:
        """Publish cached UUIDs only after the whole paper transaction commits."""
        started = time.perf_counter()
        pending: dict[bytes, str] = {}
        try:
            with self.network.expansion_batch():
                # An external writer may have merged/deleted rows since our last
                # commit. BEGIN IMMEDIATE prevents changes during this check/write.
                version = self.network.db.conn.execute(
                    "PRAGMA data_version"
                ).fetchone()[0]
                if version != self.data_version:
                    self.committed_neighbors.clear()
                    self.data_version = version
                refresh_current_paper(
                    self.network,
                    node_id,
                    paper,
                    references,
                    after_cutoff=self.after_cutoff,
                )
                source_id = nid_db_id(node_id)
                edges = []
                for records, wrapper in (
                    (citations, "citingPaper"),
                    (references, "citedPaper"),
                ):
                    for item in records:
                        neighbor = item.get(wrapper) or {}
                        if not neighbor:
                            continue
                        key = hashlib.blake2b(
                            orjson.dumps(neighbor, option=orjson.OPT_SORT_KEYS),
                            digest_size=32,
                        ).digest()
                        neighbor_id = pending.get(key) or self.committed_neighbors.get(
                            key
                        )
                        if neighbor_id is None:
                            neighbor_node = upsert_semantic_paper(
                                self.network, neighbor
                            )
                            if neighbor_node is None:
                                continue
                            neighbor_id = nid_db_id(neighbor_node)
                            pending[key] = neighbor_id
                        else:
                            self.cache_hits += 1
                            if key in self.committed_neighbors:
                                self.committed_neighbors.move_to_end(key)
                        edges.append(
                            (neighbor_id, source_id)
                            if wrapper == "citingPaper"
                            else (source_id, neighbor_id)
                        )
                self.write_edges(edges)
                self.network.mark_explored(node_id, after_cutoff=self.after_cutoff)
                self.network.complete_expansion(node_id)
                if self.on_success is not None:
                    self.on_success(source_id)
            self.committed_neighbors.update(pending)
            while len(self.committed_neighbors) > self.cache_limit:
                self.committed_neighbors.popitem(last=False)
        except BaseException:
            self.committed_neighbors.clear()
            raise
        finally:
            self.elapsed_seconds += time.perf_counter() - started

    def write_edges(self, edges: list[tuple[str, str]]) -> None:
        """Use UUIDs returned by the graph writer; retain foreign-key checks."""
        self.network.db.conn.executemany(
            "INSERT INTO citations (citing_paper_id, cited_paper_id) VALUES (?, ?) "
            "ON CONFLICT(citing_paper_id, cited_paper_id) DO NOTHING",
            edges,
        )
