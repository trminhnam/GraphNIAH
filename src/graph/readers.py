from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Sequence
from functools import wraps
from pathlib import Path
from typing import Any

import orjson

from src.setup.constants import (
    EDGE_AUTHORSHIP,
    EDGE_CITED_BY,
    EDGE_CITING,
    NODE_AUTHOR,
    NODE_PAPER,
    author_nid,
    local_node_id_uses_uuid_db_id,
    nid_db_id,
    nid_type,
    normalize_text,
    normalize_text_aggressive,
    paper_nid,
)

from .json_utils import iter_normalized_graph_edges
from .prepared_cache import (
    DEFAULT_PREPARED_CACHE_DIR,
    PreparedGraphCache,
    database_revision,
)
from .storage import configure_sqlite_read_connection, ensure_uuid_graph_store

CACHE_MISS = object()


def persistent_preparation(
    kind: str, memory_name: str, *, batch: bool = False
) -> Callable:
    """Hydrate the existing LRU in batches and persist only cache misses."""

    def decorate(method: Callable) -> Callable:
        @wraps(method)
        def read(reader: Any, keys: Any, *args: Any, **kwargs: Any) -> Any:
            db = reader.db if isinstance(reader, SQLiteAdjacencyView) else reader
            disk = getattr(db, "prepared_cache", None)
            if disk is None or not disk.is_current():
                return method(reader, keys, *args, **kwargs)
            memory = getattr(reader, memory_name)
            requested = list(
                dict.fromkeys(
                    str(key) if kind == "adjacency" else db.resolve_paper_id(key)
                    for key in (keys if batch else [keys])
                )
            )
            if batch and kind in {"title", "published"}:
                requested = [key for key in requested if key]
            if not batch and kind == "adjacency":
                try:
                    node_type = nid_type(requested[0])
                except (IndexError, ValueError):
                    return method(reader, keys, *args, **kwargs)
                if node_type not in {NODE_PAPER, NODE_AUTHOR} or (
                    node_type == NODE_AUTHOR and not reader.include_authorship_edges
                ):
                    return method(reader, keys, *args, **kwargs)
            missing = [key for key in requested if key not in memory]
            namespace = kind
            if kind == "adjacency":
                namespace += f":authors={int(reader.include_authorship_edges)}"
            stored = disk.get_many(namespace, missing) if missing else {}
            if kind == "adjacency":
                stored = {
                    key: [(node, attrs) for node, attrs in value]
                    for key, value in stored.items()
                }
            # Return disk hits alongside source misses without filling a bounded
            # LRU with more entries than it can hold before the underlying call.
            unresolved = [key for key in missing if key not in stored]
            if batch:
                needed = [key for key in requested if key not in stored]
                result = {**method(reader, needed, *args, **kwargs), **stored}
                fresh = {key: result[key] for key in unresolved if key in result}
                result = {key: result[key] for key in requested if key in result}
            elif stored:
                result = stored[requested[0]]
                fresh = {}
            else:
                result = method(reader, keys, *args, **kwargs)
                fresh = {requested[0]: result} if unresolved else {}
            loaded = {**stored, **fresh}
            for key in missing:
                if key not in loaded:
                    continue
                value = loaded[key]
                lru_cache_put(memory, key, value, reader.cache_max_entries)
            if fresh:
                disk.put_many(namespace, fresh)
            return result

        return read

    return decorate


def lru_cache_get(cache: OrderedDict[str, Any], key: str) -> Any:
    """Return and refresh one cached value, or ``CACHE_MISS``."""
    if key not in cache:
        return CACHE_MISS
    cache.move_to_end(key)
    return cache[key]


def lru_cache_put(
    cache: OrderedDict[str, Any],
    key: str,
    value: Any,
    max_entries: int | None,
) -> None:
    """Insert one value and evict the least recently used entry if needed."""
    cache[key] = value
    cache.move_to_end(key)
    if max_entries is not None and len(cache) > max_entries:
        cache.popitem(last=False)


def ensure_uuid_local_node_id(node_id: object, graph_source: Path) -> None:
    """Reject local graph node IDs that do not use UUID-backed DB IDs."""
    if not isinstance(node_id, str):
        raise RuntimeError(
            f"{graph_source} contains a non-string graph node ID: {node_id!r}"
        )
    if ":" not in node_id:
        return
    node_type = nid_type(node_id)
    if node_type not in {NODE_PAPER, NODE_AUTHOR}:
        return
    if local_node_id_uses_uuid_db_id(node_id):
        return
    raise RuntimeError(
        f"{graph_source} contains a non-UUID local graph node ID: {node_id!r}. "
        "Local incremental database IDs are no longer supported."
    )


class CitationGraphLoader:
    """Load graph topology from the DB-backed store or a legacy JSON graph."""

    def __init__(self, graph_source: Path, *, require_uuid_ids: bool = False) -> None:
        self.graph_source = graph_source
        self.require_uuid_ids = require_uuid_ids
        self.adj: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        self.nodes: set[str] = set()
        self.paper_index: dict[str, str] = {}
        self.paper_agg_index: dict[str, str] = {}
        self.arxiv_index: dict[str, str] = {}
        self.author_index: dict[str, str] = {}
        self.root_papers: list[str] = []
        self.meta: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        if self.graph_source.suffix.lower() == ".json":
            self.load_legacy_json()
            return
        self.load_db_graph()

    def load_db_graph(self) -> None:
        db = DBReader(self.graph_source, require_uuid_ids=self.require_uuid_ids)
        try:
            self.meta = db.all_graph_metadata()
            self.meta["db_path"] = str(self.graph_source)
            self.meta.setdefault("graph_format", "sqlite")

            for row in db.all_papers():
                db_id = str(row["id"])
                node_id = paper_nid(db_id)
                self.nodes.add(node_id)
                title = str(row["title"] or "")
                if title:
                    self.paper_index[normalize_text(title)] = db_id
                    self.paper_agg_index[normalize_text_aggressive(title)] = db_id
                arxiv_id = str(row["arxiv_id"] or "")
                if arxiv_id:
                    self.arxiv_index[arxiv_id] = db_id
                try:
                    if int(row["depth"]) == 0:
                        self.root_papers.append(node_id)
                except (TypeError, ValueError):
                    continue

            for row in db.all_authors():
                db_id = str(row["id"])
                node_id = author_nid(db_id)
                self.nodes.add(node_id)
                name = str(row["name"] or "")
                if name:
                    self.author_index[normalize_text(name)] = db_id

            for row in db.all_citations():
                source = paper_nid(str(row["citing_paper_id"]))
                target = paper_nid(str(row["cited_paper_id"]))
                self.nodes.add(source)
                self.nodes.add(target)
                self.adj[source].append((target, {"edge_type": EDGE_CITING}))
                self.adj[target].append((source, {"edge_type": EDGE_CITED_BY}))

            for row in db.all_authorships():
                paper_node = paper_nid(str(row["paper_id"]))
                author_node = author_nid(str(row["author_id"]))
                attrs: dict[str, Any] = {"edge_type": EDGE_AUTHORSHIP}
                if row["position"] is not None:
                    attrs["position"] = row["position"]
                self.nodes.add(paper_node)
                self.nodes.add(author_node)
                self.adj[paper_node].append((author_node, dict(attrs)))
                self.adj[author_node].append((paper_node, dict(attrs)))
        finally:
            db.close()

    def load_legacy_json(self) -> None:
        with self.graph_source.open(encoding="utf-8") as handle:
            data = orjson.loads(handle.read())

        self.meta = data.get("_meta", {})
        self.paper_index = {
            key: str(value) for key, value in data.get("paper_index", {}).items()
        }
        self.paper_agg_index = {
            key: str(value) for key, value in data.get("paper_agg_index", {}).items()
        }
        self.arxiv_index = {
            key: str(value) for key, value in data.get("arxiv_index", {}).items()
        }
        self.author_index = {
            key: str(value) for key, value in data.get("author_index", {}).items()
        }
        raw_root_papers = data.get("root_papers") or []

        graph = data["graph"]
        for node_entry in graph["nodes"]:
            node_id = node_entry["id"]
            if self.require_uuid_ids:
                ensure_uuid_local_node_id(node_id, self.graph_source)
            self.nodes.add(node_id)

        for source, target, edge_attrs in iter_normalized_graph_edges(data):
            if self.require_uuid_ids:
                ensure_uuid_local_node_id(source, self.graph_source)
                ensure_uuid_local_node_id(target, self.graph_source)
            self.adj[source].append((target, edge_attrs))
            self.nodes.add(source)
            self.nodes.add(target)

            edge_type = edge_attrs.get("edge_type", "")
            if edge_type == EDGE_CITING:
                reverse_attrs = dict(edge_attrs)
                reverse_attrs["edge_type"] = EDGE_CITED_BY
                self.adj[target].append((source, reverse_attrs))
            elif (
                edge_type == EDGE_AUTHORSHIP
                and nid_type(source) == NODE_PAPER
                and nid_type(target) == NODE_AUTHOR
            ):
                self.adj[target].append((source, dict(edge_attrs)))

        self.root_papers = [
            node_id
            for node_id in raw_root_papers
            if isinstance(node_id, str)
            and node_id in self.nodes
            and nid_type(node_id) == NODE_PAPER
        ]


PAPER_TITLE_COVERING_INDEX = "idx_papers_id_title"


def indexed_by_clause(conn: sqlite3.Connection, index_name: str) -> str:
    """Return an ``INDEXED BY`` clause when ``index_name`` exists.

    A database with no ``sqlite_stat1`` statistics makes SQLite prefer the
    primary-key index, so every matching paper costs a second random read to
    fetch its row from the table. Naming the covering index removes that read
    without needing an ANALYZE pass, which would also re-plan unrelated
    queries. Databases created before the index exists keep the old plan.
    """
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
        (index_name,),
    ).fetchone()
    return f" INDEXED BY {index_name}" if row is not None else ""


GRAPH_DB_IMMUTABLE_ENV = "GRAPH_DB_IMMUTABLE"
DISABLE_PREPARED_GRAPH_CACHE_ENV = "DISABLE_PREPARED_GRAPH_CACHE"
DISABLED_ENV_VALUES = frozenset({"", "0", "false", "no", "off"})


def env_flag_enabled(name: str) -> bool:
    """Return whether the environment variable ``name`` is set to a true value."""
    return os.environ.get(name, "").strip().lower() not in DISABLED_ENV_VALUES


def connect_immutable_database(db_path: Path) -> sqlite3.Connection:
    """Open ``db_path`` read-only, without SQLite locking or change detection.

    On a shared network filesystem such as Lustre, every SQLite transaction
    otherwise pays network round trips for its file locks, and WAL mode is not
    supported across machines. Immutable mode is correct only while no process
    writes the database. SQLite also ignores the WAL file in this mode, so a
    non-empty WAL is refused: it may hold committed changes that are not yet
    copied into the database file.

    The path is resolved once, so the WAL check and the connection use the
    same file even through a symbolic link. ``as_uri`` percent-encodes
    characters such as spaces, ``?`` and ``#``, which SQLite would otherwise
    read as URI syntax.
    """
    source = Path(db_path).resolve()
    wal_path = Path(f"{source}-wal")
    if wal_path.exists() and wal_path.stat().st_size > 0:
        raise RuntimeError(
            f"{GRAPH_DB_IMMUTABLE_ENV}=1 cannot open {db_path}: its WAL file "
            f"{wal_path} is not empty and may hold committed changes that "
            "immutable mode would ignore. Stop every process that writes this "
            "database, checkpoint it through SQLite (PRAGMA "
            "wal_checkpoint(TRUNCATE)), and retry. Do not delete the WAL file "
            "by hand."
        )
    return sqlite3.connect(source.as_uri() + "?mode=ro&immutable=1", uri=True)


def uuid_validation_marker_path(db_path: Path, directory: Path) -> Path:
    """Return the marker file that records a passed UUID scan of ``db_path``.

    The name hashes the resolved path with :func:`database_revision`, so any
    write, replacement, or non-empty WAL gives a new name and a fresh scan.
    """
    source = Path(db_path).resolve()
    key = hashlib.sha256(f"{source}\n{database_revision(source)}".encode()).hexdigest()
    return Path(directory) / f"uuid-validated-{key[:24]}.json"


def write_uuid_validation_marker(marker: Path, db_path: Path) -> None:
    """Publish ``marker`` atomically through a uniquely named temporary file.

    Threads share a process ID, and processes on different nodes can have the
    same one, so ``mkstemp`` picks the name. Concurrent writers store the same
    content, and the last ``os.replace`` wins.
    """
    marker.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=marker.parent, prefix=f"{marker.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                orjson.dumps(
                    {"database": str(Path(db_path).resolve()), "check": "uuid-v1"}
                )
            )
        os.replace(temporary, marker)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


class DBReader:
    """Read-only access to the SQLite papers/authors/authorship/citations tables.

    ``GRAPH_DB_IMMUTABLE=1`` opens the database with
    :func:`connect_immutable_database`, and ``DISABLE_PREPARED_GRAPH_CACHE=1``
    skips the shared on-disk prepared cache; a passed UUID scan is then kept in
    a marker file in that cache directory instead. Both are off unless set,
    because other jobs read databases that they are still writing.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        require_uuid_ids: bool = False,
        cache_max_entries: int | None = None,
        sqlite_cache_size_kib: int = 1048576,
        sqlite_mmap_size_bytes: int = 2147418112,
        prepared_cache_dir: Path | None = DEFAULT_PREPARED_CACHE_DIR,
    ) -> None:
        if cache_max_entries is not None and cache_max_entries <= 0:
            raise ValueError("cache_max_entries must be positive when supplied.")
        self.db_path = db_path
        self.conn = (
            connect_immutable_database(db_path)
            if env_flag_enabled(GRAPH_DB_IMMUTABLE_ENV)
            else sqlite3.connect(str(db_path))
        )
        self.conn.row_factory = sqlite3.Row
        configure_sqlite_read_connection(
            self.conn,
            cache_size_kib=sqlite_cache_size_kib,
            mmap_size_bytes=sqlite_mmap_size_bytes,
        )
        uuid_marker_dir = None
        if env_flag_enabled(DISABLE_PREPARED_GRAPH_CACHE_ENV):
            # Still remember a passed UUID scan, in a plain marker file.
            uuid_marker_dir, prepared_cache_dir = prepared_cache_dir, None
        self.prepared_cache = (
            PreparedGraphCache(Path(db_path), Path(prepared_cache_dir))
            if prepared_cache_dir is not None
            else None
        )
        if require_uuid_ids:
            try:
                marker = (
                    uuid_validation_marker_path(db_path, uuid_marker_dir)
                    if uuid_marker_dir is not None
                    else None
                )
                validated = (
                    self.prepared_cache.get_many("validation", ["uuid-v1"])
                    if self.prepared_cache is not None
                    else {"uuid-v1": marker is not None and marker.exists()}
                )
                if not validated.get("uuid-v1"):
                    ensure_uuid_graph_store(self.conn, db_path)
                    if self.prepared_cache is not None:
                        self.prepared_cache.put_many("validation", {"uuid-v1": True})
                        self.prepared_cache.flush()
                    elif marker is not None:
                        write_uuid_validation_marker(marker, db_path)
            except BaseException:
                self.close()
                raise
        self.cache_max_entries = cache_max_entries
        self.paper_title_index_clause = indexed_by_clause(
            self.conn,
            PAPER_TITLE_COVERING_INDEX,
        )
        self.paper_published_cache: OrderedDict[str, str] = OrderedDict()
        self.paper_explored_cache: OrderedDict[str, bool] = OrderedDict()
        self.paper_desc_cache: OrderedDict[str, dict[str, Any] | None] = OrderedDict()
        self.paper_title_cache: OrderedDict[str, str] = OrderedDict()
        self.author_desc_cache: OrderedDict[str, dict[str, Any] | None] = OrderedDict()
        self.expanded_authors_cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    def resolve_paper_id(self, db_id: object) -> str:
        return str(db_id or "").strip()

    def resolve_author_id(self, db_id: object) -> str:
        return str(db_id or "").strip()

    def get_paper(self, db_id: str) -> dict[str, Any] | None:
        db_id = self.resolve_paper_id(db_id)
        cached = lru_cache_get(self.paper_desc_cache, db_id)
        if cached is not CACHE_MISS:
            return cached
        row = self.conn.execute(
            "SELECT * FROM papers WHERE id = ?",
            (db_id,),
        ).fetchone()
        result = dict(row) if row else None
        lru_cache_put(
            self.paper_desc_cache,
            db_id,
            result,
            self.cache_max_entries,
        )
        return result

    @persistent_preparation("title", "paper_title_cache")
    def get_paper_title(self, db_id: str) -> str:
        """Return a paper title without loading and caching the full row."""
        db_id = self.resolve_paper_id(db_id)
        cached = lru_cache_get(self.paper_title_cache, db_id)
        if cached is not CACHE_MISS:
            return str(cached)
        row = self.conn.execute(
            "SELECT title FROM papers WHERE id = ?",
            (db_id,),
        ).fetchone()
        title = str(row["title"] or "") if row else ""
        lru_cache_put(
            self.paper_title_cache,
            db_id,
            title,
            self.cache_max_entries,
        )
        return title

    @persistent_preparation("title", "paper_title_cache", batch=True)
    def get_paper_titles(self, db_ids: Sequence[str]) -> dict[str, str]:
        """Return paper titles for many IDs using compact batch queries."""
        resolved_ids: list[str] = []
        seen_ids: set[str] = set()
        for db_id in db_ids:
            resolved_id = self.resolve_paper_id(db_id)
            if resolved_id and resolved_id not in seen_ids:
                seen_ids.add(resolved_id)
                resolved_ids.append(resolved_id)

        results: dict[str, str] = {}
        uncached_ids: list[str] = []
        for db_id in resolved_ids:
            cached = lru_cache_get(self.paper_title_cache, db_id)
            if cached is CACHE_MISS:
                uncached_ids.append(db_id)
            else:
                results[db_id] = str(cached)

        for start in range(0, len(uncached_ids), 500):
            chunk = uncached_ids[start : start + 500]
            placeholders = ",".join("?" for _db_id in chunk)
            rows = self.conn.execute(
                f"SELECT id, title FROM papers{self.paper_title_index_clause} "
                f"WHERE id IN ({placeholders})",
                chunk,
            ).fetchall()
            titles_by_id = {
                str(row["id"]): str(row["title"] or "") for row in rows
            }
            for db_id in chunk:
                title = titles_by_id.get(db_id, "")
                results[db_id] = title
                lru_cache_put(
                    self.paper_title_cache,
                    db_id,
                    title,
                    self.cache_max_entries,
                )

        return {db_id: results[db_id] for db_id in resolved_ids}

    def get_papers(
        self,
        db_ids: Sequence[str],
    ) -> dict[str, dict[str, Any] | None]:
        """Return paper rows for many ids using cached batch queries."""
        resolved_ids: list[str] = []
        seen_ids: set[str] = set()
        for db_id in db_ids:
            resolved_id = self.resolve_paper_id(db_id)
            if resolved_id and resolved_id not in seen_ids:
                seen_ids.add(resolved_id)
                resolved_ids.append(resolved_id)
        results: dict[str, dict[str, Any] | None] = {}
        uncached_ids: list[str] = []
        for db_id in resolved_ids:
            cached = lru_cache_get(self.paper_desc_cache, db_id)
            if cached is CACHE_MISS:
                uncached_ids.append(db_id)
            else:
                results[db_id] = cached
        for start in range(0, len(uncached_ids), 500):
            chunk = uncached_ids[start : start + 500]
            placeholders = ",".join("?" for _db_id in chunk)
            rows = self.conn.execute(
                f"SELECT * FROM papers WHERE id IN ({placeholders})",
                chunk,
            ).fetchall()
            rows_by_id = {str(row["id"]): dict(row) for row in rows}
            for db_id in chunk:
                result = rows_by_id.get(db_id)
                results[db_id] = result
                lru_cache_put(
                    self.paper_desc_cache,
                    db_id,
                    result,
                    self.cache_max_entries,
                )

        return {db_id: results[db_id] for db_id in resolved_ids}

    def get_author(self, db_id: str) -> dict[str, Any] | None:
        db_id = self.resolve_author_id(db_id)
        cached = lru_cache_get(self.author_desc_cache, db_id)
        if cached is not CACHE_MISS:
            return cached
        row = self.conn.execute(
            "SELECT * FROM authors WHERE id = ?",
            (db_id,),
        ).fetchone()
        result = dict(row) if row else None
        lru_cache_put(
            self.author_desc_cache,
            db_id,
            result,
            self.cache_max_entries,
        )
        return result

    def all_papers(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM papers").fetchall()

    def all_authors(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM authors").fetchall()

    def all_citations(self) -> list[sqlite3.Row]:
        try:
            return self.conn.execute(
                "SELECT * FROM citations ORDER BY citing_paper_id, cited_paper_id"
            ).fetchall()
        except sqlite3.OperationalError:
            return []

    def all_authorships(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM authorship ORDER BY paper_id, position, author_id"
        ).fetchall()

    def all_graph_metadata(self) -> dict[str, Any]:
        try:
            rows = self.conn.execute(
                "SELECT key, value FROM graph_metadata ORDER BY key"
            ).fetchall()
        except sqlite3.OperationalError:
            return {}

        metadata: dict[str, Any] = {}
        for row in rows:
            key = str(row["key"])
            raw_value = str(row["value"] or "")
            try:
                metadata[key] = orjson.loads(raw_value)
            except orjson.JSONDecodeError:
                metadata[key] = raw_value
        return metadata

    @persistent_preparation("published", "paper_published_cache")
    def get_paper_published(self, db_id: str) -> str:
        db_id = self.resolve_paper_id(db_id)
        cached = lru_cache_get(self.paper_published_cache, db_id)
        if cached is not CACHE_MISS:
            return str(cached)
        row = self.conn.execute(
            "SELECT published FROM papers WHERE id = ?",
            (db_id,),
        ).fetchone()
        published = str(row["published"] or "") if row else ""
        lru_cache_put(
            self.paper_published_cache,
            db_id,
            published,
            self.cache_max_entries,
        )
        return published

    @persistent_preparation("published", "paper_published_cache", batch=True)
    def get_paper_published_many(self, db_ids: Sequence[str]) -> dict[str, str]:
        """Return publication values for many IDs using batch queries."""
        resolved_ids: list[str] = []
        seen_ids: set[str] = set()
        for db_id in db_ids:
            resolved_id = self.resolve_paper_id(db_id)
            if resolved_id and resolved_id not in seen_ids:
                seen_ids.add(resolved_id)
                resolved_ids.append(resolved_id)

        results: dict[str, str] = {}
        uncached_ids: list[str] = []
        for db_id in resolved_ids:
            cached = lru_cache_get(self.paper_published_cache, db_id)
            if cached is CACHE_MISS:
                uncached_ids.append(db_id)
            else:
                results[db_id] = str(cached)

        for start in range(0, len(uncached_ids), 500):
            chunk = uncached_ids[start : start + 500]
            placeholders = ",".join("?" for db_id in chunk)
            rows = self.conn.execute(
                f"SELECT id, published FROM papers WHERE id IN ({placeholders})",
                chunk,
            ).fetchall()
            published_by_id = {
                str(row["id"]): str(row["published"] or "") for row in rows
            }
            for db_id in chunk:
                published = published_by_id.get(db_id, "")
                results[db_id] = published
                lru_cache_put(
                    self.paper_published_cache,
                    db_id,
                    published,
                    self.cache_max_entries,
                )

        return {db_id: results[db_id] for db_id in resolved_ids}

    @persistent_preparation("explored", "paper_explored_cache")
    def get_paper_explored(self, db_id: str) -> bool:
        db_id = self.resolve_paper_id(db_id)
        cached = lru_cache_get(self.paper_explored_cache, db_id)
        if cached is not CACHE_MISS:
            return bool(cached)
        row = self.conn.execute(
            "SELECT explored FROM papers WHERE id = ?",
            (db_id,),
        ).fetchone()
        explored = row["explored"] == 1 if row else False
        lru_cache_put(
            self.paper_explored_cache,
            db_id,
            explored,
            self.cache_max_entries,
        )
        return explored

    @persistent_preparation("explored", "paper_explored_cache", batch=True)
    def get_paper_explored_many(self, db_ids: Sequence[str]) -> dict[str, bool]:
        """Read explored flags in batches, caching missing papers as false."""
        resolved_ids = list(dict.fromkeys(self.resolve_paper_id(value) for value in db_ids))
        results: dict[str, bool] = {}
        missing: list[str] = []
        for db_id in resolved_ids:
            cached = lru_cache_get(self.paper_explored_cache, db_id)
            if cached is CACHE_MISS:
                missing.append(db_id)
            else:
                results[db_id] = bool(cached)
        for start in range(0, len(missing), 500):
            chunk = missing[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"SELECT id, explored FROM papers WHERE id IN ({placeholders})",
                chunk,
            ).fetchall()
            flags = {str(row["id"]): row["explored"] == 1 for row in rows}
            for db_id in chunk:
                explored = flags.get(db_id, False)
                results[db_id] = explored
                lru_cache_put(
                    self.paper_explored_cache, db_id, explored, self.cache_max_entries
                )
        return results

    def get_paper_node_ids_by_depth(self, depth: int) -> list[str]:
        rows = self.conn.execute(
            "SELECT id FROM papers WHERE depth = ? ORDER BY title, id",
            (depth,),
        ).fetchall()
        return [paper_nid(str(row["id"])) for row in rows]

    def get_expanded_authors_for_paper(self, paper_db_id: str) -> list[dict[str, Any]]:
        paper_db_id = self.resolve_paper_id(paper_db_id)
        cached = lru_cache_get(self.expanded_authors_cache, paper_db_id)
        if cached is not CACHE_MISS:
            return [dict(row) for row in cached]
        rows = self.conn.execute(
            """
            SELECT
                a.id           AS author_id,
                a.name         AS name,
                a.orcid        AS orcid,
                a.semantic_scholar_id AS semantic_scholar_id,
                s.organization AS organization,
                s.email        AS email,
                s.position     AS position
            FROM authorship s
            JOIN authors a ON a.id = s.author_id
            WHERE s.paper_id = ?
            ORDER BY s.position, a.id
            """,
            (paper_db_id,),
        ).fetchall()
        result = [dict(row) for row in rows]
        lru_cache_put(
            self.expanded_authors_cache,
            paper_db_id,
            result,
            self.cache_max_entries,
        )
        return [dict(row) for row in result]

    def get_expanded_authors_for_papers(
        self,
        paper_db_ids: Sequence[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Return expanded authors for many papers using cached batch queries."""
        resolved_ids: list[str] = []
        seen_ids: set[str] = set()
        for paper_db_id in paper_db_ids:
            resolved_id = self.resolve_paper_id(paper_db_id)
            if resolved_id and resolved_id not in seen_ids:
                seen_ids.add(resolved_id)
                resolved_ids.append(resolved_id)
        results: dict[str, list[dict[str, Any]]] = {}
        uncached_ids: list[str] = []
        for paper_db_id in resolved_ids:
            cached = lru_cache_get(
                self.expanded_authors_cache,
                paper_db_id,
            )
            if cached is CACHE_MISS:
                uncached_ids.append(paper_db_id)
            else:
                results[paper_db_id] = cached
        for start in range(0, len(uncached_ids), 500):
            chunk = uncached_ids[start : start + 500]
            placeholders = ",".join("?" for _paper_db_id in chunk)
            rows = self.conn.execute(
                f"""
                SELECT
                    s.paper_id     AS paper_id,
                    a.id           AS author_id,
                    a.name         AS name,
                    a.orcid        AS orcid,
                    a.semantic_scholar_id AS semantic_scholar_id,
                    s.organization AS organization,
                    s.email        AS email,
                    s.position     AS position
                FROM authorship s
                JOIN authors a ON a.id = s.author_id
                WHERE s.paper_id IN ({placeholders})
                ORDER BY s.paper_id, s.position, a.id
                """,
                chunk,
            ).fetchall()
            authors_by_paper = {paper_db_id: [] for paper_db_id in chunk}
            for row in rows:
                author = dict(row)
                paper_db_id = str(author.pop("paper_id"))
                authors_by_paper[paper_db_id].append(author)
            for paper_db_id, authors in authors_by_paper.items():
                results[paper_db_id] = authors
                lru_cache_put(
                    self.expanded_authors_cache,
                    paper_db_id,
                    authors,
                    self.cache_max_entries,
                )

        return {
            paper_db_id: [
                dict(author)
                for author in results.get(paper_db_id, [])
            ]
            for paper_db_id in resolved_ids
        }

    def author_name_variants_for_author(self, author_id: str) -> list[dict[str, Any]]:
        author_id = self.resolve_author_id(author_id)
        try:
            rows = self.conn.execute(
                """SELECT * FROM author_name_variants
                   WHERE author_id = ?
                   ORDER BY variant_name""",
                (author_id,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [dict(row) for row in rows]

    def close(self) -> None:
        try:
            if self.prepared_cache is not None:
                self.prepared_cache.close()
        finally:
            self.conn.close()


class SQLiteNodeMembership:
    """Lazy node-membership checks over the graph SQLite tables."""

    def __init__(
        self,
        db: DBReader,
        *,
        cache_max_entries: int | None = None,
    ) -> None:
        self.db = db
        self.cache_max_entries = cache_max_entries
        self.cache: OrderedDict[str, bool] = OrderedDict()

    def __contains__(self, node_id: object) -> bool:
        if not isinstance(node_id, str):
            return False
        cached = lru_cache_get(self.cache, node_id)
        if cached is not CACHE_MISS:
            return bool(cached)
        try:
            node_type = nid_type(node_id)
            db_id = nid_db_id(node_id)
        except (IndexError, ValueError):
            return False
        table = {
            NODE_PAPER: "papers",
            NODE_AUTHOR: "authors",
        }.get(node_type)
        if table is None:
            return False
        exists = (
            self.db.conn.execute(
                f"SELECT 1 FROM {table} WHERE id = ? LIMIT 1",
                (db_id,),
            ).fetchone()
            is not None
        )
        lru_cache_put(
            self.cache,
            node_id,
            exists,
            self.cache_max_entries,
        )
        return exists


class SQLiteAdjacencyView:
    """Load and memoize adjacency only for nodes reached during construction."""

    def __init__(
        self,
        db: DBReader,
        *,
        include_authorship_edges: bool = True,
        cache_max_entries: int | None = None,
    ) -> None:
        self.db = db
        self.include_authorship_edges = include_authorship_edges
        self.cache_max_entries = cache_max_entries
        self.cache: OrderedDict[str, list[tuple[str, dict[str, Any]]]] = OrderedDict()

    @persistent_preparation("adjacency", "cache")
    def get(
        self,
        node_id: str,
        default: Any = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        cached = lru_cache_get(self.cache, node_id)
        if cached is not CACHE_MISS:
            return cached

        neighbours: list[tuple[str, dict[str, Any]]] = []
        try:
            node_type = nid_type(node_id)
            db_id = nid_db_id(node_id)
        except (IndexError, ValueError):
            return [] if default is None else default

        if node_type == NODE_PAPER:
            citing_rows = self.db.conn.execute(
                """
                SELECT cited_paper_id
                FROM citations
                WHERE citing_paper_id = ?
                ORDER BY cited_paper_id
                """,
                (db_id,),
            ).fetchall()
            neighbours.extend(
                (paper_nid(str(row["cited_paper_id"])), {"edge_type": EDGE_CITING})
                for row in citing_rows
            )

            cited_by_rows = self.db.conn.execute(
                """
                SELECT citing_paper_id
                FROM citations
                WHERE cited_paper_id = ?
                ORDER BY citing_paper_id
                """,
                (db_id,),
            ).fetchall()
            neighbours.extend(
                (paper_nid(str(row["citing_paper_id"])), {"edge_type": EDGE_CITED_BY})
                for row in cited_by_rows
            )

            if self.include_authorship_edges:
                authorship_rows = self.db.conn.execute(
                    """
                    SELECT author_id, position
                    FROM authorship
                    WHERE paper_id = ?
                    ORDER BY position, author_id
                    """,
                    (db_id,),
                ).fetchall()
                for row in authorship_rows:
                    attrs: dict[str, Any] = {"edge_type": EDGE_AUTHORSHIP}
                    if row["position"] is not None:
                        attrs["position"] = row["position"]
                    neighbours.append((author_nid(str(row["author_id"])), attrs))
        elif node_type == NODE_AUTHOR and self.include_authorship_edges:
            authorship_rows = self.db.conn.execute(
                """
                SELECT paper_id, position
                FROM authorship
                WHERE author_id = ?
                ORDER BY paper_id
                """,
                (db_id,),
            ).fetchall()
            for row in authorship_rows:
                attrs = {"edge_type": EDGE_AUTHORSHIP}
                if row["position"] is not None:
                    attrs["position"] = row["position"]
                neighbours.append((paper_nid(str(row["paper_id"])), attrs))
        else:
            return [] if default is None else default

        lru_cache_put(
            self.cache,
            node_id,
            neighbours,
            self.cache_max_entries,
        )
        return neighbours

    @persistent_preparation("adjacency", "cache", batch=True)
    def get_many(
        self,
        node_ids: Sequence[str],
    ) -> dict[str, list[tuple[str, dict[str, Any]]]]:
        """Load adjacency for many paper nodes using bounded batch queries."""
        normalized_ids = list(dict.fromkeys(str(value) for value in node_ids))
        results: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        uncached_paper_ids: list[str] = []
        for node_id in normalized_ids:
            cached = lru_cache_get(self.cache, node_id)
            if cached is not CACHE_MISS:
                results[node_id] = cached
                continue
            try:
                node_type = nid_type(node_id)
            except (IndexError, ValueError):
                results[node_id] = []
                continue
            if node_type != NODE_PAPER:
                results[node_id] = self.get(node_id, [])
                continue
            results[node_id] = []
            uncached_paper_ids.append(node_id)

        for start in range(0, len(uncached_paper_ids), 500):
            paper_chunk = uncached_paper_ids[start : start + 500]
            db_ids = [nid_db_id(paper_id) for paper_id in paper_chunk]
            placeholders = ",".join("?" for _db_id in db_ids)

            citing_rows = self.db.conn.execute(
                f"""
                SELECT citing_paper_id, cited_paper_id
                FROM citations
                WHERE citing_paper_id IN ({placeholders})
                ORDER BY citing_paper_id, cited_paper_id
                """,
                db_ids,
            ).fetchall()
            for row in citing_rows:
                source_id = paper_nid(str(row["citing_paper_id"]))
                results[source_id].append(
                    (
                        paper_nid(str(row["cited_paper_id"])),
                        {"edge_type": EDGE_CITING},
                    )
                )

            cited_by_rows = self.db.conn.execute(
                f"""
                SELECT citing_paper_id, cited_paper_id
                FROM citations
                WHERE cited_paper_id IN ({placeholders})
                ORDER BY cited_paper_id, citing_paper_id
                """,
                db_ids,
            ).fetchall()
            for row in cited_by_rows:
                target_id = paper_nid(str(row["cited_paper_id"]))
                results[target_id].append(
                    (
                        paper_nid(str(row["citing_paper_id"])),
                        {"edge_type": EDGE_CITED_BY},
                    )
                )

            if self.include_authorship_edges:
                authorship_rows = self.db.conn.execute(
                    f"""
                    SELECT paper_id, author_id, position
                    FROM authorship
                    WHERE paper_id IN ({placeholders})
                    ORDER BY paper_id, position, author_id
                    """,
                    db_ids,
                ).fetchall()
                for row in authorship_rows:
                    paper_id = paper_nid(str(row["paper_id"]))
                    attrs: dict[str, Any] = {"edge_type": EDGE_AUTHORSHIP}
                    if row["position"] is not None:
                        attrs["position"] = row["position"]
                    results[paper_id].append(
                        (author_nid(str(row["author_id"])), attrs)
                    )

            for paper_id in paper_chunk:
                lru_cache_put(
                    self.cache,
                    paper_id,
                    results[paper_id],
                    self.cache_max_entries,
                )

        return {node_id: results[node_id] for node_id in normalized_ids}


class SQLiteGraphView:
    """Minimal graph API whose nodes and edges are queried lazily from SQLite."""

    def __init__(
        self,
        db: DBReader,
        *,
        include_authorship_edges: bool = True,
        cache_max_entries: int | None = None,
    ) -> None:
        self.db = db
        self.adj = SQLiteAdjacencyView(
            db,
            include_authorship_edges=include_authorship_edges,
            cache_max_entries=cache_max_entries,
        )
        self.nodes = SQLiteNodeMembership(
            db,
            cache_max_entries=cache_max_entries,
        )
        self.meta = db.all_graph_metadata()
        self.meta["db_path"] = str(db.db_path)
        self.meta.setdefault("graph_format", "sqlite")


def build_node_payload(
    node_id: str,
    graph: CitationGraphLoader | None,
    db: DBReader,
) -> dict[str, Any]:
    """Build the full node dictionary with all DB attributes."""
    del graph
    kind = nid_type(node_id)
    db_id = nid_db_id(node_id)

    if kind == NODE_PAPER:
        row = db.get_paper(db_id)
        if row is None:
            return {
                "node_type": NODE_PAPER,
                "title": "",
                "arxiv_id": "",
                "semantic_scholar_id": "",
                "authors": [],
                "expanded_authors": [],
                "abstract": "",
                "published": "",
                "venue": "",
                "pdf_url": "",
                "explored": False,
                "text_path": "",
                "node_id": node_id,
            }

        expanded_authors = db.get_expanded_authors_for_paper(db_id)
        author_names = [author["name"] for author in expanded_authors]
        return {
            "node_type": NODE_PAPER,
            "title": row["title"],
            "arxiv_id": row["arxiv_id"],
            "semantic_scholar_id": row["semantic_scholar_id"],
            "authors": author_names,
            "expanded_authors": expanded_authors,
            "abstract": row["abstract"],
            "published": row["published"],
            "venue": row["venue"],
            "pdf_url": row["pdf_url"],
            "explored": bool(row["explored"]),
            "text_path": row["text_path"],
            "node_id": node_id,
        }

    row = db.get_author(db_id)
    if row is None:
        return {
            "node_type": NODE_AUTHOR,
            "name": "",
            "semantic_scholar_id": "",
            "orcid": "",
            "node_id": node_id,
        }
    return {
        "node_type": NODE_AUTHOR,
        "name": row["name"],
        "semantic_scholar_id": row["semantic_scholar_id"],
        "orcid": row["orcid"] or "",
        "node_id": node_id,
    }
