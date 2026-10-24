from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

from loguru import logger

from src.setup.constants import (
    EDGE_AUTHORSHIP,
    EDGE_CITED_BY,
    EDGE_CITING,
    NODE_AUTHOR,
    NODE_PAPER,
    author_nid,
    nid_db_id,
    nid_type,
    paper_nid,
    preprocess_paper_title,
)

from .network import GraphLimits
from .storage import DatabaseManager


EXPANSION_LEASE_SECONDS = 1800.0
EXPANSION_POLL_SECONDS = 1.0
SHARED_DB_BUSY_TIMEOUT_MS = 300000


def paper_merge_updates(
    row: Any,
    *,
    arxiv_id: str,
    semantic_scholar_id: str,
    title: str,
    abstract: str,
    published: str,
    venue: str,
    pdf_url: str,
    explored: bool,
    text_path: str,
    depth: int,
) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    existing_title = preprocess_paper_title(str(row["title"] or ""))
    if title and (
        not existing_title
        or str(row["title"] or "") != existing_title
        or (arxiv_id and len(title) > len(existing_title))
    ):
        updates["title"] = title
    for field, value in (
        ("arxiv_id", arxiv_id),
        ("semantic_scholar_id", semantic_scholar_id),
        ("abstract", abstract),
        ("published", published),
        ("venue", venue),
        ("pdf_url", pdf_url),
        ("text_path", text_path),
    ):
        if value and not row[field]:
            updates[field] = value
    if explored and not row["explored"]:
        updates["explored"] = 1
    if depth >= 0:
        existing_depth = int(row["depth"] if row["depth"] is not None else -1)
        if existing_depth < 0 or depth < existing_depth:
            updates["depth"] = depth
    return updates


class SQLiteCitationNetwork:
    """Sampler-specific citation store whose reads always query SQLite."""

    is_sqlite_backend = True

    def __init__(
        self,
        db: DatabaseManager,
        *,
        owner_id: str,
        limits: GraphLimits | None = None,
    ) -> None:
        self.db = db
        self.db.conn.execute(f"PRAGMA busy_timeout={SHARED_DB_BUSY_TIMEOUT_MS}")
        self.owner_id = owner_id
        self.limits = limits or GraphLimits()
        self.max_depth: int | None = None

    def find_paper_row(
        self,
        *,
        semantic_scholar_id: str,
        arxiv_id: str,
        title: str,
    ) -> Any | None:
        matches: dict[str, Any] = {}
        if semantic_scholar_id:
            row = self.db.get_paper_by_semantic_scholar_id(semantic_scholar_id)
            if row is not None:
                matches[str(row["id"])] = row
        if arxiv_id:
            row = self.db.get_paper_by_arxiv_id(arxiv_id)
            if row is not None:
                matches[str(row["id"])] = row
        if len(matches) > 1:
            raise RuntimeError(
                "Paper identifiers resolve to different rows: "
                f"semantic_scholar_id={semantic_scholar_id!r}, arxiv_id={arxiv_id!r}"
            )
        if matches:
            return next(iter(matches.values()))
        if semantic_scholar_id or arxiv_id:
            return None
        if not title:
            return None
        return self.db.conn.execute(
            "SELECT * FROM papers WHERE title = ? ORDER BY id LIMIT 1",
            (title,),
        ).fetchone()

    def update_existing_paper(
        self,
        db_id: str,
        *,
        arxiv_id: str,
        semantic_scholar_id: str = "",
        title: str,
        abstract: str,
        published: str,
        venue: str = "",
        pdf_url: str,
        explored: bool,
        text_path: str,
        depth: int = -1,
        reference_count: int | None = None,
        valid_references: bool | None = None,
    ) -> str:
        title = preprocess_paper_title(title)
        row = self.db.get_paper(db_id)
        if row is None:
            raise RuntimeError(f"Paper row disappeared during update: {db_id}")
        updates = paper_merge_updates(
            row,
            arxiv_id=arxiv_id,
            semantic_scholar_id=semantic_scholar_id,
            title=title,
            abstract=abstract,
            published=published,
            venue=venue,
            pdf_url=pdf_url,
            explored=explored,
            text_path=text_path,
            depth=depth,
        )
        # Fresh Semantic Scholar validation values always overwrite old ones.
        if reference_count is not None:
            updates["reference_count"] = reference_count
        if valid_references is not None:
            updates["valid_references"] = valid_references
        if updates:
            self.db.update_paper(db_id, **updates)
        return paper_nid(db_id)

    def upsert_paper(
        self,
        *,
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
    ) -> str | None:
        title = preprocess_paper_title(title)
        if not title and not arxiv_id and not semantic_scholar_id:
            return None
        with self.db.batch(immediate=True):
            row = self.find_paper_row(
                semantic_scholar_id=semantic_scholar_id,
                arxiv_id=arxiv_id,
                title=title,
            )
            if row is None:
                db_id = self.db.insert_paper(
                    arxiv_id=arxiv_id,
                    semantic_scholar_id=semantic_scholar_id,
                    title=title,
                    abstract=abstract,
                    published=published,
                    venue=venue,
                    pdf_url=pdf_url,
                    explored=explored,
                    text_path=text_path,
                    depth=depth,
                )
                return paper_nid(db_id)
            return self.update_existing_paper(
                str(row["id"]),
                arxiv_id=arxiv_id,
                semantic_scholar_id=semantic_scholar_id,
                title=title,
                abstract=abstract,
                published=published,
                venue=venue,
                pdf_url=pdf_url,
                explored=explored,
                text_path=text_path,
                depth=depth,
            )

    def upsert_author(
        self,
        *,
        name: str = "",
        semantic_scholar_id: str = "",
        orcid: str = "",
    ) -> str | None:
        name = name.strip()
        if not name and not semantic_scholar_id:
            return None
        with self.db.batch(immediate=True):
            row = self.db.get_author_by_semantic_scholar_id(semantic_scholar_id)
            if row is None and orcid:
                row = self.db.conn.execute(
                    "SELECT * FROM authors WHERE orcid = ? LIMIT 1",
                    (orcid,),
                ).fetchone()
            if row is None and name and not semantic_scholar_id and not orcid:
                row = self.db.conn.execute(
                    "SELECT * FROM authors WHERE name = ? ORDER BY id LIMIT 1",
                    (name,),
                ).fetchone()
            if row is None:
                db_id = self.db.insert_author(
                    name=name,
                    semantic_scholar_id=semantic_scholar_id,
                    orcid=orcid,
                )
                return author_nid(db_id)

            updates: dict[str, Any] = {}
            if name and not row["name"]:
                updates["name"] = name
            if semantic_scholar_id and not row["semantic_scholar_id"]:
                updates["semantic_scholar_id"] = semantic_scholar_id
            if orcid and not row["orcid"]:
                updates["orcid"] = orcid
            if updates:
                self.db.update_author(str(row["id"]), **updates)
            return author_nid(str(row["id"]))

    def add_authorship(
        self,
        paper_node: str,
        author_node: str,
        *,
        position: int = 0,
        organization: str = "",
        email: str = "",
    ) -> None:
        self.db.insert_authorship(
            paper_id=nid_db_id(paper_node),
            author_id=nid_db_id(author_node),
            position=position,
            organization=organization,
            email=email,
        )

    def add_citation(self, citing_node: str, cited_node: str) -> None:
        self.db.insert_citation(
            citing_paper_id=nid_db_id(citing_node),
            cited_paper_id=nid_db_id(cited_node),
        )

    def mark_explored(self, paper_node: str, *, after_cutoff: bool = False) -> None:
        """Mark a paper fully explored.

        ``after_cutoff=True`` records that this exploration only stored
        citations/references on or after a neighbor-date cutoff (see
        --bulk-filter-neighbors-by-cutoff), so explored=1 here does not mean
        a complete edge set. A later full (non-cutoff) exploration clears the
        marker again, since explored_after_cutoff describes the most recent
        exploration, not the exploration history.
        """
        self.db.update_paper(
            nid_db_id(paper_node),
            explored=1,
            explored_after_cutoff=1 if after_cutoff else None,
        )

    def list_neighbors(
        self,
        node_id: str,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Read the current adjacency from SQLite without process-local caching."""
        node_type = nid_type(node_id)
        db_id = nid_db_id(node_id)
        neighbors: list[tuple[str, dict[str, Any]]] = []
        if node_type == NODE_PAPER:
            rows = self.db.conn.execute(
                """SELECT c.cited_paper_id AS neighbor_id, p.published AS published
                   FROM citations c
                   JOIN papers p ON p.id = c.cited_paper_id
                   WHERE c.citing_paper_id = ?
                   ORDER BY c.cited_paper_id""",
                (db_id,),
            ).fetchall()
            neighbors.extend(
                (
                    paper_nid(str(row["neighbor_id"])),
                    {
                        "edge_type": EDGE_CITING,
                        "published": str(row["published"] or ""),
                    },
                )
                for row in rows
            )
            rows = self.db.conn.execute(
                """SELECT c.citing_paper_id AS neighbor_id, p.published AS published
                   FROM citations c
                   JOIN papers p ON p.id = c.citing_paper_id
                   WHERE c.cited_paper_id = ?
                   ORDER BY c.citing_paper_id""",
                (db_id,),
            ).fetchall()
            neighbors.extend(
                (
                    paper_nid(str(row["neighbor_id"])),
                    {
                        "edge_type": EDGE_CITED_BY,
                        "published": str(row["published"] or ""),
                    },
                )
                for row in rows
            )
            rows = self.db.conn.execute(
                """SELECT author_id, position
                   FROM authorship
                   WHERE paper_id = ?
                   ORDER BY position, author_id""",
                (db_id,),
            ).fetchall()
            for row in rows:
                attrs: dict[str, Any] = {"edge_type": EDGE_AUTHORSHIP}
                if row["position"] is not None:
                    attrs["position"] = int(row["position"])
                neighbors.append((author_nid(str(row["author_id"])), attrs))
        elif node_type == NODE_AUTHOR:
            rows = self.db.conn.execute(
                """SELECT s.paper_id AS neighbor_id, s.position AS position,
                          p.published AS published
                   FROM authorship s
                   JOIN papers p ON p.id = s.paper_id
                   WHERE s.author_id = ?
                   ORDER BY s.paper_id""",
                (db_id,),
            ).fetchall()
            for row in rows:
                attrs = {
                    "edge_type": EDGE_AUTHORSHIP,
                    "published": str(row["published"] or ""),
                }
                if row["position"] is not None:
                    attrs["position"] = int(row["position"])
                neighbors.append((paper_nid(str(row["neighbor_id"])), attrs))
        return neighbors

    def acquire_expansion(
        self, node_id: str, *, require_full_exploration: bool = False
    ) -> str:
        """Wait for or claim expansion, returning ``claimed`` or ``complete``.

        ``require_full_exploration=True`` means the caller wants a complete
        (non-cutoff-restricted) edge set: a paper whose only completion was
        under --bulk-filter-neighbors-by-cutoff (explored=1,
        explored_after_cutoff=1) is then not treated as complete, so it gets
        reclaimed and re-expanded rather than skipped as already-done.
        """
        entity_type = nid_type(node_id)
        entity_id = nid_db_id(node_id)
        if entity_type == NODE_PAPER:
            row = self.db.get_paper(entity_id)
            lease = self.db.get_expansion_lease(entity_type, entity_id)
            cutoff_tainted = require_full_exploration and bool(
                row["explored_after_cutoff"] if row is not None else False
            )
            if row is not None and bool(row["explored"]) and lease is None and not cutoff_tainted:
                return "complete"

        logged_wait = False
        while True:
            now = time.time()
            status = self.db.claim_expansion(
                entity_type,
                entity_id,
                self.owner_id,
                now=now,
                lease_until=now + EXPANSION_LEASE_SECONDS,
                require_full_exploration=require_full_exploration,
            )
            if status != "busy":
                return status
            if not logged_wait:
                logger.info("Waiting for shared expansion of {}", node_id)
                logged_wait = True
            time.sleep(EXPANSION_POLL_SECONDS)

    def complete_expansion(self, node_id: str) -> None:
        self.db.complete_expansion(
            nid_type(node_id),
            nid_db_id(node_id),
            self.owner_id,
        )

    def release_expansion(self, node_id: str) -> None:
        self.db.release_expansion(
            nid_type(node_id),
            nid_db_id(node_id),
            self.owner_id,
        )

    @contextmanager
    def expansion_batch(self) -> Generator[None, None, None]:
        with self.db.batch(immediate=True):
            yield

    def stats(self) -> str:
        paper_count = self.db.count_papers()
        author_count = self.db.count_authors()
        citation_count = self.db.conn.execute(
            "SELECT COUNT(*) FROM citations"
        ).fetchone()[0]
        authorship_count = self.db.conn.execute(
            "SELECT COUNT(*) FROM authorship"
        ).fetchone()[0]
        explored_count = self.db.conn.execute(
            "SELECT COUNT(*) FROM papers WHERE explored = 1"
        ).fetchone()[0]
        return (
            f"SQLite graph: {paper_count + author_count} nodes "
            f"({paper_count} papers, {explored_count} explored, {author_count} authors), "
            f"{citation_count + authorship_count} edges"
        )
