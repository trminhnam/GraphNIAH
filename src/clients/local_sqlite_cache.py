"""SQLite-backed cache for HTTP client responses."""

from __future__ import annotations

from copy import deepcopy
import orjson
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterable

from src.clients.local_json_cache import LocalJsonCache


class LocalSqliteCache:
    """Persist request/response entries in a local SQLite database."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self.lock = threading.Lock()
        self.connection: sqlite3.Connection | None = None

    @classmethod
    def make_key(
        cls,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        body: Any = None,
    ) -> str:
        return LocalJsonCache.make_key(method, endpoint, params=params, body=body)

    @classmethod
    def make_request(
        cls,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        body: Any = None,
    ) -> dict[str, Any]:
        return LocalJsonCache.make_request(method, endpoint, params=params, body=body)

    @classmethod
    def normalize(cls, value: Any) -> Any:
        return LocalJsonCache.normalize(value)

    def get(self, key: str) -> tuple[bool, Any]:
        if self.path is None:
            return False, None

        with self.lock:
            connection = self.connect()
            row = connection.execute(
                "SELECT response FROM http_cache WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return False, None
            return True, deepcopy(orjson.loads(row["response"]))

    def store(
        self,
        key: str,
        method: str,
        endpoint: str,
        response: Any,
        *,
        params: dict[str, Any] | None = None,
        body: Any = None,
    ) -> None:
        if self.path is None:
            return

        request = self.make_request(method, endpoint, params=params, body=body)
        with self.lock:
            self.store_rows(
                [
                    (
                        key,
                        request["method"],
                        request["endpoint"],
                        json_text(request["params"]),
                        json_text(request["body"]),
                        json_text(response),
                    )
                ]
            )

    def store_entries(self, entries: Iterable[tuple[str, dict[str, Any], Any]]) -> int:
        rows = []
        for key, request, response in entries:
            rows.append(
                (
                    key,
                    str(request.get("method", "")).upper(),
                    str(request.get("endpoint", "")),
                    json_text(request.get("params")),
                    json_text(request.get("body")),
                    json_text(response),
                )
            )

        if self.path is None or not rows:
            return 0

        with self.lock:
            self.store_rows(rows)
        return len(rows)

    def connect(self) -> sqlite3.Connection:
        if self.connection is not None:
            return self.connection
        if self.path is None:
            raise RuntimeError("Cannot open disabled cache")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS http_cache (
                key TEXT PRIMARY KEY,
                method TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                params TEXT,
                body TEXT,
                response TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_http_cache_endpoint ON http_cache(endpoint)"
        )
        connection.commit()
        self.connection = connection
        return connection

    def store_rows(self, rows: list[tuple[str, str, str, str, str, str]]) -> None:
        connection = self.connect()
        try:
            connection.executemany(
                """
                INSERT INTO http_cache (
                    key,
                    method,
                    endpoint,
                    params,
                    body,
                    response,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    method = excluded.method,
                    endpoint = excluded.endpoint,
                    params = excluded.params,
                    body = excluded.body,
                    response = excluded.response,
                    updated_at = CURRENT_TIMESTAMP
                """,
                rows,
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None


def json_text(value: Any) -> str:
    return orjson.dumps(LocalJsonCache.normalize(value), option=orjson.OPT_SORT_KEYS).decode("utf-8")
