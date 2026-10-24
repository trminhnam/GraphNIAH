from __future__ import annotations

import orjson
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from src.setup.constants import is_uuid_text, preprocess_paper_title


DATABASE_SIDECAR_SUFFIXES = ("", "-shm", "-wal")


def database_file_set(db_path: Path) -> list[Path]:
    return [Path(f"{db_path}{suffix}") for suffix in DATABASE_SIDECAR_SUFFIXES]


def database_sidecar_file_set(db_path: Path) -> list[Path]:
    return [Path(f"{db_path}{suffix}") for suffix in DATABASE_SIDECAR_SUFFIXES if suffix]


def remove_database_file_set(db_path: Path) -> None:
    for path in database_file_set(db_path):
        if path.exists():
            path.unlink()


def remove_database_sidecar_file_set(db_path: Path) -> None:
    for path in database_sidecar_file_set(db_path):
        if path.exists():
            path.unlink()


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        file_descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def temporary_database_path(target_path: Path) -> Path:
    return target_path.with_name(f".{target_path.name}.{os.getpid()}.tmp")


def configure_sqlite_connection(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    connection.execute("PRAGMA busy_timeout=30000;")
    connection.execute("PRAGMA foreign_keys=ON;")


DEFAULT_SQLITE_READ_CACHE_KIB = 1048576
DEFAULT_SQLITE_READ_MMAP_BYTES = 2147418112


def configure_sqlite_read_connection(
    connection: sqlite3.Connection,
    *,
    cache_size_kib: int = DEFAULT_SQLITE_READ_CACHE_KIB,
    mmap_size_bytes: int = DEFAULT_SQLITE_READ_MMAP_BYTES,
) -> None:
    """Size the page cache and mmap window for read-heavy graph traversal.

    SQLite defaults to a 2 MB page cache and no memory mapping, which is far
    too small for the multi-gigabyte citation graph: nearly every lookup falls
    through to a random disk read.
    """
    if cache_size_kib <= 0:
        raise ValueError("cache_size_kib must be positive.")
    if mmap_size_bytes < 0:
        raise ValueError("mmap_size_bytes must be non-negative.")
    connection.execute(f"PRAGMA cache_size=-{cache_size_kib};")
    connection.execute(f"PRAGMA mmap_size={mmap_size_bytes};")
    connection.execute("PRAGMA temp_store=MEMORY;")


def backup_connection_to_path(
    source: sqlite3.Connection,
    target_path: Path,
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temporary_database_path(target_path)
    remove_database_file_set(temp_path)
    try:
        with sqlite3.connect(str(temp_path), timeout=30.0) as destination:
            destination.execute("PRAGMA journal_mode=DELETE;")
            destination.execute("PRAGMA synchronous=FULL;")
            destination.execute("PRAGMA busy_timeout=30000;")
            source.backup(destination)
            destination.commit()
            quick_check = destination.execute("PRAGMA quick_check;").fetchone()
            if quick_check is None or quick_check[0] != "ok":
                raise sqlite3.DatabaseError(f"Snapshot integrity check failed: {quick_check}")
        fsync_file(temp_path)
        remove_database_sidecar_file_set(target_path)
        temp_path.replace(target_path)
        remove_database_sidecar_file_set(target_path)
        fsync_directory(target_path.parent)
    except Exception:
        remove_database_file_set(temp_path)
        raise
    finally:
        remove_database_file_set(temp_path)


def copy_sqlite_database(source_path: Path, target_path: Path) -> None:
    source_path = Path(source_path)
    target_path = Path(target_path)
    with sqlite3.connect(str(source_path), timeout=30.0) as source:
        source.execute("PRAGMA busy_timeout=30000;")
        backup_connection_to_path(source, target_path)


def generate_uuid() -> str:
    return str(uuid.uuid4())


def json_dumps(value: Any) -> str:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS, default=str).decode("utf-8")


UUID_GRAPH_ID_COLUMNS = {
    "papers": ("id",),
    "authors": ("id",),
    "authorship": ("paper_id", "author_id"),
    "citations": ("citing_paper_id", "cited_paper_id"),
    "pending_reference_authors": ("paper_id", "source_paper_id"),
    "author_name_variants": ("author_id",),
}
UUID_PRIMARY_VALUE_COLUMNS = {
    "papers": ("id",),
    "authors": ("id",),
}


def require_uuid_id(value: object, field_name: str) -> str:
    """Return *value* as text when it is a UUID-backed local graph ID."""
    text = str(value or "").strip()
    if is_uuid_text(text):
        return text
    raise ValueError(
        f"{field_name} must be a UUID string. "
        "Local incremental database IDs are no longer supported; "
        "keep numeric values only in semantic_scholar_id."
    )


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def column_declared_type(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
) -> str | None:
    if not table_exists(connection, table_name):
        return None
    for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall():
        if row["name"] == column_name:
            return str(row["type"] or "").upper()
    return None


def ensure_uuid_graph_column_types(
    connection: sqlite3.Connection,
    db_path: Path,
) -> None:
    invalid_columns: list[str] = []
    for table_name, column_names in UUID_GRAPH_ID_COLUMNS.items():
        for column_name in column_names:
            declared_type = column_declared_type(connection, table_name, column_name)
            if declared_type is None:
                raise sqlite3.DatabaseError(
                    f"Expected {table_name}.{column_name} in graph schema"
                )
            if declared_type != "TEXT":
                invalid_columns.append(f"{table_name}.{column_name}")
    if invalid_columns:
        raise RuntimeError(
            f"{db_path} uses a legacy non-UUID schema for: "
            f"{', '.join(invalid_columns)}. "
            "Create a fresh UUID-backed database before using the new graph store."
        )


def ensure_uuid_primary_key_values(
    connection: sqlite3.Connection,
    db_path: Path,
) -> None:
    for table_name, column_names in UUID_PRIMARY_VALUE_COLUMNS.items():
        for column_name in column_names:
            for row in connection.execute(f"SELECT {column_name} FROM {table_name}"):
                value = str(row[0] or "").strip()
                if is_uuid_text(value):
                    continue
                raise RuntimeError(
                    f"{db_path} contains a non-UUID {table_name}.{column_name} "
                    f"value: {value!r}. Local incremental database IDs are no "
                    "longer supported; keep numeric values only in "
                    "semantic_scholar_id."
                )


def ensure_uuid_graph_store(
    connection: sqlite3.Connection,
    db_path: Path,
) -> None:
    ensure_uuid_graph_column_types(connection, db_path)
    ensure_uuid_primary_key_values(connection, db_path)


class DatabaseManager:
    """Thin wrapper around the UUID-backed citation-network SQLite schema."""

    UUID_ID_COLUMNS = UUID_GRAPH_ID_COLUMNS

    PAPERS_DDL = """
        CREATE TABLE IF NOT EXISTS papers (
            id          TEXT PRIMARY KEY,
            arxiv_id    TEXT NOT NULL DEFAULT '',
            semantic_scholar_id TEXT NOT NULL DEFAULT '',
            title       TEXT NOT NULL DEFAULT '',
            abstract    TEXT NOT NULL DEFAULT '',
            published   TEXT NOT NULL DEFAULT '',
            venue       TEXT NOT NULL DEFAULT '',
            pdf_url     TEXT NOT NULL DEFAULT '',
            explored    INTEGER NOT NULL DEFAULT 0,
            text_path   TEXT NOT NULL DEFAULT '',
            depth       INTEGER NOT NULL DEFAULT -1
        );
    """

    AUTHORS_DDL = """
        CREATE TABLE IF NOT EXISTS authors (
            id           TEXT PRIMARY KEY,
            name         TEXT NOT NULL DEFAULT '',
            semantic_scholar_id TEXT NOT NULL DEFAULT '',
            orcid        TEXT UNIQUE
        );
    """

    AUTHORSHIP_DDL = """
        CREATE TABLE IF NOT EXISTS authorship (
            paper_id     TEXT NOT NULL REFERENCES papers(id),
            author_id    TEXT NOT NULL REFERENCES authors(id),
            organization TEXT NOT NULL DEFAULT '',
            email        TEXT NOT NULL DEFAULT '',
            position     INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (paper_id, author_id)
        );
    """

    CITATIONS_DDL = """
        CREATE TABLE IF NOT EXISTS citations (
            citing_paper_id TEXT NOT NULL REFERENCES papers(id),
            cited_paper_id  TEXT NOT NULL REFERENCES papers(id),
            PRIMARY KEY (citing_paper_id, cited_paper_id)
        );
    """

    PENDING_REFERENCE_AUTHORS_DDL = """
        CREATE TABLE IF NOT EXISTS pending_reference_authors (
            paper_id        TEXT NOT NULL REFERENCES papers(id),
            source_paper_id TEXT NOT NULL REFERENCES papers(id),
            position        INTEGER NOT NULL,
            name            TEXT NOT NULL,
            PRIMARY KEY (paper_id, source_paper_id, position)
        );
    """

    AUTHOR_NAME_VARIANTS_DDL = """
        CREATE TABLE IF NOT EXISTS author_name_variants (
            author_id       TEXT NOT NULL REFERENCES authors(id),
            variant_name    TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            PRIMARY KEY (author_id, normalized_name)
        );
    """

    GRAPH_METADATA_DDL = """
        CREATE TABLE IF NOT EXISTS graph_metadata (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """

    EXPANSION_LEASES_DDL = """
        CREATE TABLE IF NOT EXISTS expansion_leases (
            entity_type  TEXT NOT NULL,
            entity_id    TEXT NOT NULL,
            state        TEXT NOT NULL,
            owner_id     TEXT NOT NULL DEFAULT '',
            lease_until  REAL NOT NULL DEFAULT 0,
            updated_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (entity_type, entity_id),
            CHECK (entity_type IN ('paper', 'author')),
            CHECK (state IN ('pending', 'expanding', 'complete'))
        );
    """

    # Papers whose title/ID couldn't be resolved to anything in a local bulk
    # Semantic Scholar index (endpoints/expand_unexplored_papers.py), so a
    # later run can skip re-attempting them. Scoped to `index_created_at` (the
    # index's metadata.json "createdAt") so a rebuilt index -- which may cover
    # more papers -- makes stale rows stop matching and those papers retry
    # automatically, with no manual cache-clearing needed.
    BULK_LOOKUP_MISSES_DDL = """
        CREATE TABLE IF NOT EXISTS bulk_lookup_misses (
            paper_id         TEXT PRIMARY KEY REFERENCES papers(id),
            index_created_at TEXT NOT NULL,
            checked_at       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """

    INDEXES_DDL = """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_arxiv_id_unique
            ON papers(arxiv_id) WHERE arxiv_id != '';
        CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_semantic_scholar_id_unique
            ON papers(semantic_scholar_id) WHERE semantic_scholar_id != '';
        CREATE UNIQUE INDEX IF NOT EXISTS idx_authors_semantic_scholar_id_unique
            ON authors(semantic_scholar_id) WHERE semantic_scholar_id != '';
        CREATE INDEX IF NOT EXISTS idx_authors_name ON authors(name);
        CREATE INDEX IF NOT EXISTS idx_papers_explored ON papers(explored);
        CREATE INDEX IF NOT EXISTS idx_papers_explored_published
            ON papers(explored, published);
        CREATE INDEX IF NOT EXISTS idx_papers_depth ON papers(depth);
        CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title);
        CREATE INDEX IF NOT EXISTS idx_papers_id_title ON papers(id, title);
        CREATE INDEX IF NOT EXISTS idx_papers_venue ON papers(venue);
        CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published);
        CREATE INDEX IF NOT EXISTS idx_authorship_author_id ON authorship(author_id);
        CREATE INDEX IF NOT EXISTS idx_authorship_position_author_id
            ON authorship(position, author_id);
        CREATE INDEX IF NOT EXISTS idx_citations_cited_covering
            ON citations(cited_paper_id, citing_paper_id);
        CREATE INDEX IF NOT EXISTS idx_pending_reference_authors_paper
            ON pending_reference_authors(paper_id);
        CREATE INDEX IF NOT EXISTS idx_pending_reference_authors_source
            ON pending_reference_authors(source_paper_id);
        CREATE INDEX IF NOT EXISTS idx_author_name_variants_normalized
            ON author_name_variants(normalized_name);
        CREATE INDEX IF NOT EXISTS idx_bulk_lookup_misses_index_created_at
            ON bulk_lookup_misses(index_created_at);
    """

    def __init__(self, db_path: Path, *, validate_uuid_values: bool = True) -> None:
        self.db_path = db_path
        self.validate_uuid_values = validate_uuid_values
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            str(db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row
        configure_sqlite_connection(self.conn)
        self.auto_commit = True
        self.create_tables()

    def create_tables(self) -> None:
        self.conn.executescript(
            self.PAPERS_DDL
            + self.AUTHORS_DDL
            + self.AUTHORSHIP_DDL
            + self.CITATIONS_DDL
            + self.PENDING_REFERENCE_AUTHORS_DDL
            + self.AUTHOR_NAME_VARIANTS_DDL
            + self.GRAPH_METADATA_DDL
            + self.EXPANSION_LEASES_DDL
            + self.BULK_LOOKUP_MISSES_DDL
        )
        self.ensure_schema_columns()
        if self.validate_uuid_values:
            self.ensure_uuid_graph_store()
        else:
            ensure_uuid_graph_column_types(self.conn, self.db_path)
        self.conn.executescript(self.INDEXES_DDL)
        self.conn.commit()

    def ensure_schema_columns(self) -> None:
        table_columns = {
            "papers": {
                row["name"]
                for row in self.conn.execute("PRAGMA table_info(papers)").fetchall()
            },
            "authors": {
                row["name"]
                for row in self.conn.execute("PRAGMA table_info(authors)").fetchall()
            },
        }
        if "semantic_scholar_id" not in table_columns["papers"]:
            self.conn.execute(
                "ALTER TABLE papers ADD COLUMN semantic_scholar_id TEXT NOT NULL DEFAULT ''"
            )
        if "venue" not in table_columns["papers"]:
            self.conn.execute(
                "ALTER TABLE papers ADD COLUMN venue TEXT NOT NULL DEFAULT ''"
            )
        if "semantic_scholar_id" not in table_columns["authors"]:
            self.conn.execute(
                "ALTER TABLE authors ADD COLUMN semantic_scholar_id TEXT NOT NULL DEFAULT ''"
            )
        if "reference_count" not in table_columns["papers"]:
            self.conn.execute("ALTER TABLE papers ADD COLUMN reference_count INTEGER")
        if "valid_references" not in table_columns["papers"]:
            self.conn.execute("ALTER TABLE papers ADD COLUMN valid_references INTEGER")
        if "explored_after_cutoff" not in table_columns["papers"]:
            # NULL means "not explored under a neighbor-date cutoff filter" --
            # covers both never-explored papers and normally (fully) explored
            # ones. 1 means explored=1 was reached with
            # --bulk-filter-neighbors-by-cutoff on, so citations/references
            # were deliberately restricted to that cutoff and are incomplete
            # for any consumer that isn't itself cutoff-restricted. Existing
            # explored==1 checks (src/questions/explored.py, readers.py) are
            # unaffected by this column; only acquire_expansion() treats it
            # as reopening a paper for re-expansion.
            self.conn.execute(
                "ALTER TABLE papers ADD COLUMN explored_after_cutoff INTEGER"
            )

    def table_exists(self, table_name: str) -> bool:
        return table_exists(self.conn, table_name)

    def column_declared_type(self, table_name: str, column_name: str) -> str | None:
        return column_declared_type(self.conn, table_name, column_name)

    def ensure_uuid_graph_store(self) -> None:
        ensure_uuid_graph_store(self.conn, self.db_path)

    def maybe_commit(self) -> None:
        if self.auto_commit:
            try:
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def flush(self) -> None:
        """Flush the current connection state to SQLite."""
        try:
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    @contextmanager
    def batch(self, *, immediate: bool = False) -> Generator[None, None, None]:
        """Defer all commits to a single transaction."""
        if not self.auto_commit:
            yield
            return
        self.auto_commit = False
        try:
            if immediate:
                self.conn.execute("BEGIN IMMEDIATE")
            yield
        except BaseException:
            self.conn.rollback()
            raise
        else:
            self.conn.commit()
        finally:
            self.auto_commit = True

    def get_expansion_lease(
        self,
        entity_type: str,
        entity_id: str,
    ) -> sqlite3.Row | None:
        return self.conn.execute(
            """SELECT entity_type, entity_id, state, owner_id, lease_until
               FROM expansion_leases
               WHERE entity_type = ? AND entity_id = ?""",
            (entity_type, entity_id),
        ).fetchone()

    def claim_expansion(
        self,
        entity_type: str,
        entity_id: str,
        owner_id: str,
        *,
        now: float,
        lease_until: float,
        require_full_exploration: bool = False,
    ) -> str:
        """Return ``claimed``, ``complete``, or ``busy`` for an expansion.

        A "complete" lease is normally trusted as-is. ``require_full_exploration``
        adds one check for papers: a lease completed while the paper's
        explored_after_cutoff=1 (only --bulk-filter-neighbors-by-cutoff
        neighbors were stored) is not treated as complete, so it falls
        through and gets reclaimed instead.
        """
        with self.batch(immediate=True):
            row = self.get_expansion_lease(entity_type, entity_id)
            if row is not None and str(row["state"]) == "complete":
                cutoff_tainted = False
                if require_full_exploration and entity_type == "paper":
                    paper_row = self.get_paper(entity_id)
                    cutoff_tainted = paper_row is not None and bool(
                        paper_row["explored_after_cutoff"]
                    )
                if not cutoff_tainted:
                    return "complete"
            if (
                row is not None
                and str(row["state"]) == "expanding"
                and str(row["owner_id"]) != owner_id
                and float(row["lease_until"] or 0) > now
            ):
                return "busy"
            self.conn.execute(
                """INSERT INTO expansion_leases
                       (entity_type, entity_id, state, owner_id, lease_until, updated_at)
                   VALUES (?, ?, 'expanding', ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(entity_type, entity_id) DO UPDATE SET
                       state = 'expanding',
                       owner_id = excluded.owner_id,
                       lease_until = excluded.lease_until,
                       updated_at = CURRENT_TIMESTAMP""",
                (entity_type, entity_id, owner_id, lease_until),
            )
        return "claimed"

    def complete_expansion(
        self,
        entity_type: str,
        entity_id: str,
        owner_id: str,
    ) -> None:
        cursor = self.conn.execute(
            """UPDATE expansion_leases
               SET state = 'complete', owner_id = '', lease_until = 0,
                   updated_at = CURRENT_TIMESTAMP
               WHERE entity_type = ? AND entity_id = ? AND owner_id = ?""",
            (entity_type, entity_id, owner_id),
        )
        if cursor.rowcount == 0:
            row = self.get_expansion_lease(entity_type, entity_id)
            if row is None or str(row["state"]) != "complete":
                raise RuntimeError(
                    f"Expansion lease for {entity_type}:{entity_id} is no longer owned by "
                    f"{owner_id}"
                )
        self.maybe_commit()

    def release_expansion(
        self,
        entity_type: str,
        entity_id: str,
        owner_id: str,
    ) -> None:
        self.conn.execute(
            """UPDATE expansion_leases
               SET state = 'pending', owner_id = '', lease_until = 0,
                   updated_at = CURRENT_TIMESTAMP
               WHERE entity_type = ? AND entity_id = ? AND owner_id = ?""",
            (entity_type, entity_id, owner_id),
        )
        self.maybe_commit()

    def resolve_paper_id(self, paper_id: object, field_name: str = "papers.id") -> str:
        """Return a UUID paper primary key."""
        return require_uuid_id(paper_id, field_name)

    def resolve_author_id(
        self,
        author_id: object,
        field_name: str = "authors.id",
    ) -> str:
        """Return a UUID author primary key."""
        return require_uuid_id(author_id, field_name)

    def insert_paper(
        self,
        *,
        paper_id: str | None = None,
        arxiv_id: str = "",
        semantic_scholar_id: str = "",
        title: str = "",
        abstract: str = "",
        published: str = "",
        venue: str = "",
        pdf_url: str = "",
        explored: bool = False,
        text_path: str = "",
        depth: int = -1,
    ) -> str:
        paper_id = require_uuid_id(paper_id or generate_uuid(), "papers.id")
        title = preprocess_paper_title(title)
        self.conn.execute(
            """INSERT INTO papers
               (id, arxiv_id, semantic_scholar_id, title, abstract, published, venue, pdf_url, explored, text_path, depth)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                paper_id,
                arxiv_id,
                semantic_scholar_id,
                title,
                abstract,
                published,
                venue,
                pdf_url,
                int(explored),
                text_path,
                depth,
            ),
        )
        self.maybe_commit()
        return paper_id

    def update_paper(self, paper_id: str, **kwargs: Any) -> None:
        if not kwargs:
            return
        paper_id = self.resolve_paper_id(paper_id, "papers.id")
        allowed = {
            "arxiv_id",
            "semantic_scholar_id",
            "title",
            "abstract",
            "published",
            "venue",
            "pdf_url",
            "explored",
            "explored_after_cutoff",
            "text_path",
            "depth",
            "reference_count",
            "valid_references",
        }
        bad = set(kwargs) - allowed
        if bad:
            raise ValueError(f"Unknown paper columns: {bad}")
        if "title" in kwargs:
            kwargs["title"] = preprocess_paper_title(str(kwargs["title"]))
        if "explored" in kwargs:
            kwargs["explored"] = int(kwargs["explored"])
        if "explored_after_cutoff" in kwargs and kwargs["explored_after_cutoff"] is not None:
            kwargs["explored_after_cutoff"] = int(kwargs["explored_after_cutoff"])
        if "valid_references" in kwargs and kwargs["valid_references"] is not None:
            kwargs["valid_references"] = int(kwargs["valid_references"])
        set_clause = ", ".join(f"{key} = ?" for key in kwargs)
        self.conn.execute(
            f"UPDATE papers SET {set_clause} WHERE id = ?",
            (*kwargs.values(), paper_id),
        )
        self.maybe_commit()

    def get_paper(self, paper_id: str) -> sqlite3.Row | None:
        paper_id = self.resolve_paper_id(paper_id, "papers.id")
        return self.conn.execute(
            "SELECT * FROM papers WHERE id = ?",
            (paper_id,),
        ).fetchone()

    def get_paper_by_semantic_scholar_id(
        self,
        semantic_scholar_id: str,
    ) -> sqlite3.Row | None:
        if not semantic_scholar_id:
            return None
        return self.conn.execute(
            "SELECT * FROM papers "
            "WHERE semantic_scholar_id = ? AND semantic_scholar_id != ''",
            (semantic_scholar_id,),
        ).fetchone()

    def get_paper_by_arxiv_id(self, arxiv_id: str) -> sqlite3.Row | None:
        if not arxiv_id:
            return None
        return self.conn.execute(
            "SELECT * FROM papers WHERE arxiv_id = ? AND arxiv_id != ''",
            (arxiv_id,),
        ).fetchone()

    def all_papers(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM papers").fetchall()

    def count_papers(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    def insert_author(
        self,
        *,
        author_id: str | None = None,
        name: str = "",
        semantic_scholar_id: str = "",
        orcid: str = "",
    ) -> str:
        author_id = require_uuid_id(author_id or generate_uuid(), "authors.id")
        self.conn.execute(
            "INSERT INTO authors (id, name, semantic_scholar_id, orcid) VALUES (?, ?, ?, ?)",
            (author_id, name, semantic_scholar_id, orcid or None),
        )
        self.maybe_commit()
        return author_id

    def update_author(self, author_id: str, **kwargs: Any) -> None:
        if not kwargs:
            return
        author_id = self.resolve_author_id(author_id, "authors.id")
        allowed = {"name", "semantic_scholar_id", "orcid"}
        bad = set(kwargs) - allowed
        if bad:
            raise ValueError(f"Unknown author columns: {bad}")
        set_clause = ", ".join(f"{key} = ?" for key in kwargs)
        self.conn.execute(
            f"UPDATE authors SET {set_clause} WHERE id = ?",
            (*kwargs.values(), author_id),
        )
        self.maybe_commit()

    def get_author(self, author_id: str) -> sqlite3.Row | None:
        author_id = self.resolve_author_id(author_id, "authors.id")
        return self.conn.execute(
            "SELECT * FROM authors WHERE id = ?",
            (author_id,),
        ).fetchone()

    def get_author_by_semantic_scholar_id(
        self,
        semantic_scholar_id: str,
    ) -> sqlite3.Row | None:
        if not semantic_scholar_id:
            return None
        return self.conn.execute(
            "SELECT * FROM authors "
            "WHERE semantic_scholar_id = ? AND semantic_scholar_id != ''",
            (semantic_scholar_id,),
        ).fetchone()

    def all_authors(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM authors").fetchall()

    def count_authors(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM authors").fetchone()[0]

    def insert_authorship(
        self,
        *,
        paper_id: str,
        author_id: str,
        organization: str = "",
        email: str = "",
        position: int = 0,
    ) -> None:
        paper_id = self.resolve_paper_id(paper_id, "authorship.paper_id")
        author_id = self.resolve_author_id(author_id, "authorship.author_id")
        self.conn.execute(
            """INSERT INTO authorship (paper_id, author_id, organization, email, position)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(paper_id, author_id) DO UPDATE SET
                   organization = CASE WHEN excluded.organization != ''
                                       THEN excluded.organization
                                       ELSE authorship.organization END,
                   email        = CASE WHEN excluded.email != ''
                                       THEN excluded.email
                                       ELSE authorship.email END,
                   position     = CASE WHEN excluded.position != 0
                                       THEN excluded.position
                                       ELSE authorship.position END""",
            (paper_id, author_id, organization, email, position),
        )
        self.maybe_commit()

    def update_authorship(self, paper_id: str, author_id: str, **kwargs: Any) -> None:
        if not kwargs:
            return
        paper_id = self.resolve_paper_id(paper_id, "authorship.paper_id")
        author_id = self.resolve_author_id(author_id, "authorship.author_id")
        allowed = {"organization", "email", "position"}
        bad = set(kwargs) - allowed
        if bad:
            raise ValueError(f"Unknown authorship columns: {bad}")
        set_clause = ", ".join(f"{key} = ?" for key in kwargs)
        self.conn.execute(
            f"UPDATE authorship SET {set_clause} WHERE paper_id = ? AND author_id = ?",
            (*kwargs.values(), paper_id, author_id),
        )
        self.maybe_commit()

    def get_authorship(self, paper_id: str, author_id: str) -> sqlite3.Row | None:
        paper_id = self.resolve_paper_id(paper_id, "authorship.paper_id")
        author_id = self.resolve_author_id(author_id, "authorship.author_id")
        return self.conn.execute(
            "SELECT * FROM authorship WHERE paper_id = ? AND author_id = ?",
            (paper_id, author_id),
        ).fetchone()

    def authorships_for_paper(self, paper_id: str) -> list[sqlite3.Row]:
        paper_id = self.resolve_paper_id(paper_id, "authorship.paper_id")
        return self.conn.execute(
            "SELECT * FROM authorship WHERE paper_id = ? ORDER BY position, author_id",
            (paper_id,),
        ).fetchall()

    def authorships_for_author(self, author_id: str) -> list[sqlite3.Row]:
        author_id = self.resolve_author_id(author_id, "authorship.author_id")
        return self.conn.execute(
            "SELECT * FROM authorship WHERE author_id = ? ORDER BY paper_id",
            (author_id,),
        ).fetchall()

    def all_authorships(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM authorship ORDER BY paper_id, position, author_id"
        ).fetchall()

    def insert_citation(
        self,
        *,
        citing_paper_id: str,
        cited_paper_id: str,
    ) -> None:
        citing_paper_id = self.resolve_paper_id(
            citing_paper_id,
            "citations.citing_paper_id",
        )
        cited_paper_id = self.resolve_paper_id(
            cited_paper_id,
            "citations.cited_paper_id",
        )
        self.conn.execute(
            """INSERT INTO citations (citing_paper_id, cited_paper_id)
               VALUES (?, ?)
               ON CONFLICT(citing_paper_id, cited_paper_id) DO NOTHING""",
            (citing_paper_id, cited_paper_id),
        )
        self.maybe_commit()

    def all_citations(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM citations ORDER BY citing_paper_id, cited_paper_id"
        ).fetchall()

    def insert_pending_reference_author(
        self,
        *,
        paper_id: str,
        source_paper_id: str,
        position: int,
        name: str,
    ) -> None:
        paper_id = self.resolve_paper_id(paper_id, "pending_reference_authors.paper_id")
        source_paper_id = self.resolve_paper_id(
            source_paper_id,
            "pending_reference_authors.source_paper_id",
        )
        self.conn.execute(
            """INSERT INTO pending_reference_authors
               (paper_id, source_paper_id, position, name)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(paper_id, source_paper_id, position) DO UPDATE SET
                   name = excluded.name""",
            (paper_id, source_paper_id, position, name),
        )
        self.maybe_commit()

    def pending_reference_authors_for_paper(
        self,
        paper_id: str,
    ) -> list[sqlite3.Row]:
        paper_id = self.resolve_paper_id(paper_id, "pending_reference_authors.paper_id")
        return self.conn.execute(
            """SELECT * FROM pending_reference_authors
               WHERE paper_id = ?
               ORDER BY source_paper_id, position""",
            (paper_id,),
        ).fetchall()

    def delete_pending_reference_authors_for_paper(self, paper_id: str) -> None:
        paper_id = self.resolve_paper_id(paper_id, "pending_reference_authors.paper_id")
        self.conn.execute(
            "DELETE FROM pending_reference_authors WHERE paper_id = ?",
            (paper_id,),
        )
        self.maybe_commit()

    def insert_author_name_variant(
        self,
        *,
        author_id: str,
        variant_name: str,
        normalized_name: str,
    ) -> None:
        author_id = self.resolve_author_id(author_id, "author_name_variants.author_id")
        self.conn.execute(
            """INSERT INTO author_name_variants
               (author_id, variant_name, normalized_name)
               VALUES (?, ?, ?)
               ON CONFLICT(author_id, normalized_name) DO NOTHING""",
            (author_id, variant_name, normalized_name),
        )
        self.maybe_commit()

    def author_name_variants_for_author(self, author_id: str) -> list[sqlite3.Row]:
        author_id = self.resolve_author_id(author_id, "author_name_variants.author_id")
        return self.conn.execute(
            """SELECT * FROM author_name_variants
               WHERE author_id = ?
               ORDER BY variant_name""",
            (author_id,),
        ).fetchall()

    def all_author_name_variants(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM author_name_variants ORDER BY rowid"
        ).fetchall()

    def explored_paper_ids(self) -> set[str]:
        return {
            str(row[0])
            for row in self.conn.execute(
                "SELECT id FROM papers WHERE explored = 1"
            ).fetchall()
        }

    def paper_ids_in(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        valid_ids: list[str] = []
        for raw_id in ids:
            try:
                valid_ids.append(self.resolve_paper_id(raw_id))
            except ValueError:
                continue

        existing: set[str] = set()
        for start in range(0, len(valid_ids), 900):
            chunk = valid_ids[start : start + 900]
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT id FROM papers WHERE id IN ({placeholders})",
                chunk,
            ).fetchall()
            existing.update(str(row[0]) for row in rows)
        return existing

    def author_ids_in(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        valid_ids: list[str] = []
        for raw_id in ids:
            try:
                valid_ids.append(self.resolve_author_id(raw_id))
            except ValueError:
                continue

        existing: set[str] = set()
        for start in range(0, len(valid_ids), 900):
            chunk = valid_ids[start : start + 900]
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT id FROM authors WHERE id IN ({placeholders})",
                chunk,
            ).fetchall()
            existing.update(str(row[0]) for row in rows)
        return existing

    def papers_titles_by_ids(self, ids: list[str]) -> dict[str, str]:
        if not ids:
            return {}
        result: dict[str, str] = {}
        for raw_id in ids:
            try:
                resolved_id = self.resolve_paper_id(raw_id)
            except ValueError:
                continue
            row = self.conn.execute(
                "SELECT title FROM papers WHERE id = ?",
                (resolved_id,),
            ).fetchone()
            if row is not None:
                result[str(raw_id)] = row["title"] or ""
        return result

    def authors_names_by_ids(self, ids: list[str]) -> dict[str, str]:
        if not ids:
            return {}
        result: dict[str, str] = {}
        for raw_id in ids:
            try:
                resolved_id = self.resolve_author_id(raw_id)
            except ValueError:
                continue
            row = self.conn.execute(
                "SELECT name FROM authors WHERE id = ?",
                (resolved_id,),
            ).fetchone()
            if row is not None:
                result[str(raw_id)] = row["name"] or ""
        return result

    def set_graph_metadata(self, key: str, value: Any) -> None:
        self.conn.execute(
            """INSERT INTO graph_metadata (key, value)
               VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (key, json_dumps(value)),
        )
        self.maybe_commit()

    def get_graph_metadata(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute(
            "SELECT value FROM graph_metadata WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return default
        raw_value = str(row["value"] or "")
        try:
            return orjson.loads(raw_value)
        except orjson.JSONDecodeError:
            return raw_value

    def all_graph_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for row in self.conn.execute(
            "SELECT key, value FROM graph_metadata ORDER BY key"
        ).fetchall():
            key = str(row["key"])
            raw_value = str(row["value"] or "")
            try:
                metadata[key] = orjson.loads(raw_value)
            except orjson.JSONDecodeError:
                metadata[key] = raw_value
        return metadata

    def snapshot_to(self, snapshot_path: Path) -> None:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn.commit()

        current_path = self.db_path.resolve()
        target_path = snapshot_path.resolve()
        if current_path == target_path:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            self.conn.commit()
            return

        backup_connection_to_path(self.conn, snapshot_path)

    def close(self) -> None:
        self.conn.close()

    def __del__(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self) -> "DatabaseManager":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
