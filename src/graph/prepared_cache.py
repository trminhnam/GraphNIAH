"""Small, disposable graph caches that fall back during database updates."""

from __future__ import annotations

import hashlib
import sqlite3
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

import orjson

DEFAULT_PREPARED_CACHE_DIR = Path(__file__).resolve().parents[2] / "resources/cache"
CACHE_FORMAT = 1


def database_revision(path: Path) -> str:
    """Detect replacements and ordinary SQLite writes, including uncheckpointed WAL."""
    stamps = []
    for source in (path, Path(str(path) + "-wal")):
        try:
            stat = source.stat()
            # Read-only connections may create/remove an empty WAL. It holds
            # no database changes and must not invalidate every new process.
            if source != path and stat.st_size == 0:
                stamps.append(None)
                continue
            stamps.append(
                (
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                )
            )
        except FileNotFoundError:
            stamps.append(None)
    return hashlib.sha256(orjson.dumps([CACHE_FORMAT, stamps])).hexdigest()


def compact_key(value: str) -> bytes:
    """Store canonical UUIDs in 16 bytes; preserve all other keys exactly."""
    try:
        parsed = UUID(value)
        if str(parsed) == value:
            return b"u" + parsed.bytes
    except ValueError:
        pass
    return b"s" + value.encode("utf-8")


def encode_value(value: Any) -> bytes:
    raw = orjson.dumps(value)
    compressed = zlib.compress(raw, level=1) if len(raw) >= 256 else raw
    return b"z" + compressed if len(compressed) < len(raw) else b"j" + raw


def decode_value(value: bytes) -> Any:
    return orjson.loads(zlib.decompress(value[1:]) if value[:1] == b"z" else value[1:])


class PreparedGraphCache:
    """One compact SQLite cache per source DB, invalidated as one generation.

    A source change disables this instance and drops pending writes. Readers
    can continue with their process-local caches without publishing stale data
    to the shared disk cache.
    """

    def __init__(self, source: Path, directory: Path) -> None:
        self.source = source.resolve()
        self.revision = database_revision(self.source)
        directory.mkdir(parents=True, exist_ok=True)
        source_key = hashlib.sha256(str(self.source).encode()).hexdigest()[:24]
        self.path = directory / f"prepared-graph-{source_key}.sqlite3"
        self.conn = sqlite3.connect(self.path, timeout=60)
        self.conn.execute("PRAGMA auto_vacuum=FULL")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.pending: dict[tuple[str, bytes], bytes] = {}
        self.pending_bytes = 0
        self.enabled = True
        with self.conn:
            self.conn.execute("CREATE TABLE IF NOT EXISTS metadata (revision TEXT)")
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS entries (kind TEXT, key BLOB, value BLOB, "
                "PRIMARY KEY (kind, key)) WITHOUT ROWID"
            )
            self.conn.execute("BEGIN IMMEDIATE")
            previous = self.conn.execute("SELECT revision FROM metadata").fetchone()
            if previous != (self.revision,):
                self.conn.execute("DELETE FROM entries")
                self.conn.execute("DELETE FROM metadata")
                self.conn.execute("INSERT INTO metadata VALUES (?)", (self.revision,))

    def disable(self) -> None:
        """Discard pending data and bypass this cache for the rest of its life."""
        self.enabled = False
        self.pending.clear()
        self.pending_bytes = 0

    def is_current(self) -> bool:
        """Allow database updates without reusing or publishing disk entries."""
        if self.enabled and database_revision(self.source) != self.revision:
            self.disable()
        return self.enabled

    def get_many(self, kind: str, keys: Sequence[str]) -> dict[str, Any]:
        if not self.is_current():
            return {}
        result = {}
        missing = []
        for key in keys:
            packed = compact_key(key)
            value = self.pending.get((kind, packed))
            if value is None:
                missing.append((key, packed))
            else:
                result[key] = decode_value(value)
        for start in range(0, len(missing), 400):
            chunk = missing[start : start + 400]
            placeholders = ",".join("?" for _ in chunk)
            rows = dict(
                self.conn.execute(
                    f"SELECT key, value FROM entries WHERE kind=? AND key IN ({placeholders})",
                    [kind, *(packed for _, packed in chunk)],
                )
            )
            for key, packed in chunk:
                if packed in rows:
                    result[key] = decode_value(rows[packed])
        return result if self.is_current() else {}

    def put_many(self, kind: str, values: dict[str, Any]) -> None:
        if not self.is_current():
            return
        for key, value in values.items():
            packed = compact_key(key)
            encoded = encode_value(value)
            old = self.pending.get((kind, packed), b"")
            self.pending_bytes += len(encoded) - len(old)
            self.pending[kind, packed] = encoded
        if len(self.pending) >= 512 or self.pending_bytes >= 1024 * 1024:
            self.flush()

    def flush(self) -> None:
        if not self.is_current() or not self.pending:
            return
        with self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            revision = self.conn.execute("SELECT revision FROM metadata").fetchone()
            if revision != (self.revision,):
                self.disable()
                return
            self.conn.executemany(
                "INSERT OR REPLACE INTO entries VALUES (?, ?, ?)",
                [(kind, key, value) for (kind, key), value in self.pending.items()],
            )
        self.pending.clear()
        self.pending_bytes = 0

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self.conn.close()
