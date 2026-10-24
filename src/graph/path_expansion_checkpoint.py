"""Durable frontiers and per-paper results for sampled-path expansion."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import orjson
from loguru import logger

from src.graph.db_network import SQLiteCitationNetwork

SUCCESS_STATUSES = {"expanded", "already_explored", "merged"}


@dataclass
class PathExpansionCheckpoint:
    network: SQLiteCitationNetwork
    run_id: str = ""

    def prepare(
        self,
        settings: dict,
        seeds: dict[str, str],
        remaps: dict[str, str],
        restart: bool,
    ) -> None:
        payload = orjson.dumps(settings, option=orjson.OPT_SORT_KEYS)
        self.run_id = hashlib.sha256(payload).hexdigest()
        connection = self.network.db.conn
        with self.network.expansion_batch():
            connection.execute(
                "CREATE TABLE IF NOT EXISTS path_expansion_runs ("
                "run_id TEXT PRIMARY KEY, settings BLOB NOT NULL, seeds BLOB NOT NULL, "
                "level INTEGER NOT NULL, finished INTEGER NOT NULL)"
            )
            # No paper foreign keys: durable merge remaps resolve deleted UUIDs.
            connection.execute(
                "CREATE TABLE IF NOT EXISTS path_expansion_frontiers ("
                "run_id TEXT NOT NULL, level INTEGER NOT NULL, paper_id TEXT NOT NULL, "
                "PRIMARY KEY (run_id, level, paper_id))"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS path_expansion_results ("
                "run_id TEXT NOT NULL, paper_id TEXT NOT NULL, status TEXT NOT NULL, "
                "PRIMARY KEY (run_id, paper_id))"
            )
            row = connection.execute(
                "SELECT seeds FROM path_expansion_runs WHERE run_id = ?", (self.run_id,)
            ).fetchone()
            if row is not None:
                saved_seeds = {
                    remaps.get(pid, pid) for pid in orjson.loads(row["seeds"])
                }
                if saved_seeds != set(seeds):
                    logger.info(
                        "Sampled seed papers changed; rebuilding the traversal checkpoint"
                    )
                restart = restart or saved_seeds != set(seeds)
            if restart:
                for table in (
                    "path_expansion_results",
                    "path_expansion_frontiers",
                    "path_expansion_runs",
                ):
                    connection.execute(
                        f"DELETE FROM {table} WHERE run_id = ?", (self.run_id,)
                    )
                row = None
            if row is None:
                connection.execute(
                    "INSERT INTO path_expansion_runs VALUES (?, ?, ?, 0, 0)",
                    (self.run_id, payload, orjson.dumps(sorted(seeds))),
                )
                self.save_level(0, seeds, finished=False)

    def load(self) -> tuple[int, dict[str, str], set[str], set[str], bool, bool]:
        connection = self.network.db.conn
        row = connection.execute(
            "SELECT level, finished FROM path_expansion_runs WHERE run_id = ?",
            (self.run_id,),
        ).fetchone()
        level = row["level"]
        frontier = dict.fromkeys(
            (
                item[0]
                for item in connection.execute(
                    "SELECT paper_id FROM path_expansion_frontiers WHERE run_id = ? AND level = ?",
                    (self.run_id, level),
                )
            ),
            "",
        )
        visited = {
            item[0]
            for item in connection.execute(
                "SELECT paper_id FROM path_expansion_frontiers WHERE run_id = ? AND level < ?",
                (self.run_id, level),
            )
        }
        results = connection.execute(
            "SELECT paper_id, status FROM path_expansion_results WHERE run_id = ?",
            (self.run_id,),
        )
        # A merged loser does not prove that its surviving winner was expanded.
        attempted = set()
        incomplete = False
        for paper_id, status in results:
            if status != "merged":
                attempted.add(paper_id)
            if status not in SUCCESS_STATUSES:
                incomplete = True
        return level, frontier, visited, attempted, incomplete, bool(row["finished"])

    def record_success(self, paper_id: str) -> None:
        """Called inside the graph writer's transaction; never commit separately."""
        self.network.db.conn.execute(
            "INSERT OR REPLACE INTO path_expansion_results VALUES (?, ?, 'expanded')",
            (self.run_id, paper_id),
        )

    def record_results(self, results: list[tuple[str, str]]) -> None:
        with self.network.expansion_batch():
            self.network.db.conn.executemany(
                "INSERT OR REPLACE INTO path_expansion_results VALUES (?, ?, ?)",
                ((self.run_id, paper_id, status) for paper_id, status in results),
            )

    def papers_with_status(self, statuses: frozenset[str]) -> set[str]:
        placeholders = ",".join("?" for _ in statuses)
        return {
            row[0]
            for row in self.network.db.conn.execute(
                "SELECT paper_id FROM path_expansion_results "
                f"WHERE run_id = ? AND status IN ({placeholders})",
                (self.run_id, *statuses),
            )
        }

    def replace_seeds(self, seeds: dict[str, str], remaps: dict[str, str]) -> set[str]:
        """Keep only these seeds after paths were dropped; return removed seeds.

        Reruns then match the rewritten path files instead of resetting. Removed
        seeds lose their results but keep level-0 frontier rows, so later levels
        still treat them as visited.
        """
        connection = self.network.db.conn
        with self.network.expansion_batch():
            row = connection.execute(
                "SELECT seeds FROM path_expansion_runs WHERE run_id = ?", (self.run_id,)
            ).fetchone()
            saved = {remaps.get(pid, pid) for pid in orjson.loads(row["seeds"])}
            removed = saved - set(seeds)
            connection.execute(
                "UPDATE path_expansion_runs SET seeds = ? WHERE run_id = ?",
                (orjson.dumps(sorted(seeds)), self.run_id),
            )
            connection.executemany(
                "DELETE FROM path_expansion_results WHERE run_id = ? AND paper_id = ?",
                ((self.run_id, paper_id) for paper_id in removed),
            )
        return removed

    def has_unresolved_results(self) -> bool:
        placeholders = ",".join("?" for _ in SUCCESS_STATUSES)
        return (
            self.network.db.conn.execute(
                "SELECT 1 FROM path_expansion_results "
                f"WHERE run_id = ? AND status NOT IN ({placeholders}) LIMIT 1",
                (self.run_id, *SUCCESS_STATUSES),
            ).fetchone()
            is not None
        )

    def save_level(
        self, level: int, frontier: dict[str, str], *, finished: bool
    ) -> None:
        """Publish the next frontier and advance the cursor atomically."""
        with self.network.expansion_batch():
            self.network.db.conn.executemany(
                "INSERT INTO path_expansion_frontiers VALUES (?, ?, ?)",
                ((self.run_id, level, paper_id) for paper_id in frontier),
            )
            self.network.db.conn.execute(
                "UPDATE path_expansion_runs SET level = ?, finished = ? WHERE run_id = ?",
                (level, int(finished), self.run_id),
            )
