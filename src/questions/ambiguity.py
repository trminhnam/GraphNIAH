from __future__ import annotations

from typing import Any

from src.graph.readers import CitationGraphLoader, DBReader
from src.setup.constants import (
    NODE_AUTHOR,
    NODE_PAPER,
    author_nid,
    nid_db_id,
    nid_type,
    paper_nid,
)

from .descriptors import paper_published_year_month_descriptor_value
from .text import unpack_author_pair, unpack_author_triplet


def descriptor_is_omitted(descriptor: dict[str, Any]) -> bool:
    """Return True when a node descriptor intentionally omits identity details."""
    return descriptor.get("kind") is None or descriptor.get("value") is None


def consecutive_omitted_descriptor_pairs(
    node_descriptors: list[dict[str, Any]],
) -> list[list[int]]:
    """Return adjacent node-index pairs where both descriptors are omitted."""
    pairs: list[list[int]] = []
    for index in range(1, len(node_descriptors)):
        if descriptor_is_omitted(
            node_descriptors[index - 1]
        ) and descriptor_is_omitted(node_descriptors[index]):
            pairs.append([index - 1, index])
    return pairs


class AmbiguityChecker:
    """Forward-walk ambiguity checker bound to a graph + SQLite DB."""

    def __init__(self, graph_loader: CitationGraphLoader, db_reader: DBReader) -> None:
        self.graph = graph_loader
        self.db = db_reader
        self.first_author: dict[str, str] = {}
        self.title: dict[str, str] = {}
        self.venue: dict[str, str] = {}
        self.published_year_month: dict[str, str] = {}
        self.author_name: dict[str, str] = {}
        self.paper_author_list: dict[str, list[str]] = {}
        self.anchor_matches: dict[tuple, list[str]] = {}

    def paper_node_for_db_id(self, db_id: Any) -> str:
        return paper_nid(str(db_id))

    def author_node_for_db_id(self, db_id: Any) -> str:
        return author_nid(str(db_id))

    def paper_first_author(self, paper_node_id: str) -> str:
        if paper_node_id in self.first_author:
            return self.first_author[paper_node_id]
        rows = self.db.get_expanded_authors_for_paper(nid_db_id(paper_node_id))
        value = rows[0]["name"] if rows else ""
        self.first_author[paper_node_id] = value
        return value

    def paper_authors(self, paper_node_id: str) -> list[str]:
        if paper_node_id in self.paper_author_list:
            return self.paper_author_list[paper_node_id]
        rows = self.db.get_expanded_authors_for_paper(nid_db_id(paper_node_id))
        names = [row["name"] for row in rows if row.get("name")]
        self.paper_author_list[paper_node_id] = names
        return names

    def paper_title(self, paper_node_id: str) -> str:
        if paper_node_id in self.title:
            return self.title[paper_node_id]
        row = self.db.get_paper(nid_db_id(paper_node_id))
        value = row["title"] if row else ""
        self.title[paper_node_id] = value
        return value

    def paper_venue(self, paper_node_id: str) -> str:
        if paper_node_id in self.venue:
            return self.venue[paper_node_id]
        row = self.db.get_paper(nid_db_id(paper_node_id))
        value = str(row["venue"] or "").strip() if row else ""
        self.venue[paper_node_id] = value
        return value

    def paper_published_year_month(self, paper_node_id: str) -> str:
        if paper_node_id in self.published_year_month:
            return self.published_year_month[paper_node_id]
        row = self.db.get_paper(nid_db_id(paper_node_id))
        value = paper_published_year_month_descriptor_value(row or {}) or ""
        self.published_year_month[paper_node_id] = value
        return value

    def author_name_of(self, author_node_id: str) -> str:
        if author_node_id in self.author_name:
            return self.author_name[author_node_id]
        row = self.db.get_author(nid_db_id(author_node_id))
        value = row["name"] if row else ""
        self.author_name[author_node_id] = value
        return value

    def forward_neighbour_edges(self, node_id: str, edge_type: str) -> list[tuple[str, dict]]:
        return [
            (target, attrs)
            for (target, attrs) in self.graph.adj.get(node_id, [])
            if attrs.get("edge_type") == edge_type
        ]

    def forward_neighbours(self, node_id: str, edge_type: str) -> list[str]:
        return [target for (target, _) in self.forward_neighbour_edges(node_id, edge_type)]

    def matches_descriptor(
        self,
        candidate_node_id: str,
        kind: str | None,
        value: Any | None,
        *,
        edge_attrs: dict | None = None,
    ) -> bool:
        if kind is None or value is None:
            return True
        node_type = nid_type(candidate_node_id)
        if kind == "title" and node_type == "paper":
            return self.paper_title(candidate_node_id) == value
        if kind == "venue" and node_type == "paper":
            return self.paper_venue(candidate_node_id) == value
        if kind == "published_year_month" and node_type == "paper":
            return self.paper_published_year_month(candidate_node_id) == value
        if kind == "first_author" and node_type == "paper":
            return self.paper_first_author(candidate_node_id) == value
        if kind == "title_first_author" and node_type == "paper":
            if not isinstance(value, dict):
                return False
            return self.paper_title(candidate_node_id) == value.get("title") and self.paper_first_author(candidate_node_id) == value.get("first_author")
        if kind == "some_author" and node_type == "paper":
            return value in self.paper_authors(candidate_node_id)
        if kind == "some_author_pair" and node_type == "paper":
            try:
                author_one, author_two = unpack_author_pair(value)
            except ValueError:
                return False
            paper_authors = self.paper_authors(candidate_node_id)
            return author_one in paper_authors and author_two in paper_authors
        if kind == "some_author_triplet" and node_type == "paper":
            try:
                author_one, author_two, author_three = unpack_author_triplet(value)
            except ValueError:
                return False
            paper_authors = self.paper_authors(candidate_node_id)
            return (
                author_one in paper_authors
                and author_two in paper_authors
                and author_three in paper_authors
            )
        if kind == "name" and node_type == "author":
            return self.author_name_of(candidate_node_id) == value
        if kind == "author_position" and node_type == "author":
            if edge_attrs is None:
                return False
            return edge_attrs.get("position") == value
        return False

    def descriptor_unique_for_edge(
        self,
        source_node_id: str,
        edge_type: str,
        target_node_id: str,
        kind: str | None,
        value: Any | None,
    ) -> bool:
        result = self.check_edge(source_node_id, edge_type, target_node_id, kind, value)
        return result["match_count"] == 1 and result["expected_in_matches"]

    def descriptor_unique_for_anchor(
        self,
        anchor_node_id: str,
        kind: str | None,
        value: Any | None,
    ) -> bool:
        result = self.check_anchor(anchor_node_id, kind, value)
        return result["match_count"] == 1 and result["expected_in_matches"]

    def check_edge(
        self,
        source_node_id: str,
        edge_type: str,
        target_node_id: str,
        descriptor_kind: str | None,
        descriptor_value: Any | None,
    ) -> dict[str, Any]:
        neighbour_edges = self.forward_neighbour_edges(source_node_id, edge_type)
        matches = [
            node_id
            for node_id, attrs in neighbour_edges
            if self.matches_descriptor(
                node_id,
                descriptor_kind,
                descriptor_value,
                edge_attrs=attrs,
            )
        ]
        descriptor_omitted = descriptor_kind is None or descriptor_value is None
        return {
            "from": source_node_id,
            "to": target_node_id,
            "edge_type": edge_type,
            "descriptor": {"kind": descriptor_kind, "value": descriptor_value},
            "descriptor_omitted": descriptor_omitted,
            "forward_neighbour_count": len(neighbour_edges),
            "match_count": len(matches),
            "matches": matches,
            "expected_in_matches": target_node_id in matches,
            "is_ambiguous": False if descriptor_omitted else len(matches) > 1,
        }

    @staticmethod
    def anchor_cache_key(kind: str | None, value: Any | None) -> tuple | None:
        if kind is None or value is None:
            return None
        if kind == "title_first_author":
            if not isinstance(value, dict):
                return None
            return (kind, value.get("title"), value.get("first_author"))
        if kind == "some_author_pair":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return None
            return (kind, str(value[0]), str(value[1]))
        if kind == "some_author_triplet":
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                return None
            return (kind, str(value[0]), str(value[1]), str(value[2]))
        if kind in (
            "title",
            "venue",
            "published_year_month",
            "first_author",
            "name",
            "some_author",
        ):
            return (kind, value)
        return None

    def compute_anchor_matches(self, kind: str, value: Any) -> list[str]:
        if kind == "title":
            rows = self.db.conn.execute(
                "SELECT id FROM papers WHERE title = ?",
                (value,),
            ).fetchall()
            return [self.paper_node_for_db_id(row[0]) for row in rows]

        if kind == "venue":
            rows = self.db.conn.execute(
                "SELECT id FROM papers WHERE venue = ?",
                (value,),
            ).fetchall()
            return [self.paper_node_for_db_id(row[0]) for row in rows]

        if kind == "published_year_month":
            # Range scan instead of LIKE so idx_papers_published applies; the
            # exact prefix condition is re-checked below.
            rows = self.db.conn.execute(
                "SELECT id, published FROM papers "
                "WHERE published >= ? AND published < ?",
                (str(value), str(value) + "\uffff"),
            ).fetchall()
            return [
                self.paper_node_for_db_id(row["id"])
                for row in rows
                if paper_published_year_month_descriptor_value(dict(row)) == value
            ]

        if kind == "first_author":
            rows = self.db.conn.execute(
                """
                SELECT DISTINCT s.paper_id
                FROM authorship s
                JOIN authors a ON a.id = s.author_id
                WHERE a.name = ?
                """,
                (value,),
            ).fetchall()
            return [
                self.paper_node_for_db_id(row[0])
                for row in rows
                if self.paper_first_author(self.paper_node_for_db_id(row[0])) == value
            ]

        if kind == "title_first_author" and isinstance(value, dict):
            title = value.get("title")
            first_author = value.get("first_author")
            rows = self.db.conn.execute(
                "SELECT id FROM papers WHERE title = ?",
                (title,),
            ).fetchall()
            return [
                self.paper_node_for_db_id(row[0])
                for row in rows
                if self.paper_first_author(self.paper_node_for_db_id(row[0]))
                == first_author
            ]

        if kind == "name":
            rows = self.db.conn.execute(
                "SELECT id FROM authors WHERE name = ?",
                (value,),
            ).fetchall()
            return [self.author_node_for_db_id(row[0]) for row in rows]

        if kind == "some_author":
            rows = self.db.conn.execute(
                """
                SELECT DISTINCT s.paper_id
                FROM authorship s
                JOIN authors a ON a.id = s.author_id
                WHERE a.name = ?
                """,
                (value,),
            ).fetchall()
            return [self.paper_node_for_db_id(row[0]) for row in rows]

        if kind == "some_author_pair":
            try:
                author_one, author_two = unpack_author_pair(value)
            except ValueError:
                return []
            rows = self.db.conn.execute(
                """
                SELECT s1.paper_id
                FROM authorship s1
                JOIN authors ath1 ON ath1.id = s1.author_id
                JOIN authorship s2 ON s2.paper_id = s1.paper_id
                JOIN authors ath2 ON ath2.id = s2.author_id
                WHERE ath1.name = ? AND ath2.name = ?
                """,
                (author_one, author_two),
            ).fetchall()
            return [self.paper_node_for_db_id(row[0]) for row in rows]

        if kind == "some_author_triplet":
            try:
                author_one, author_two, author_three = unpack_author_triplet(value)
            except ValueError:
                return []
            rows = self.db.conn.execute(
                """
                SELECT DISTINCT s.paper_id
                FROM authorship s
                JOIN authors a ON a.id = s.author_id
                WHERE a.name = ?
                """,
                (author_one,),
            ).fetchall()
            return [
                self.paper_node_for_db_id(row[0])
                for row in rows
                if (
                    author_two in self.paper_authors(self.paper_node_for_db_id(row[0]))
                    and author_three
                    in self.paper_authors(self.paper_node_for_db_id(row[0]))
                )
            ]

        return []

    def check_anchor(
        self,
        anchor_node_id: str,
        kind: str | None,
        value: Any | None,
    ) -> dict[str, Any]:
        cache_key = self.anchor_cache_key(kind, value)
        if cache_key is None:
            matches: list[str] = [anchor_node_id]
        else:
            cached = self.anchor_matches.get(cache_key)
            if cached is None:
                matches = self.compute_anchor_matches(kind, value)
                self.anchor_matches[cache_key] = matches
            else:
                matches = cached
        return {
            "node_id": anchor_node_id,
            "descriptor": {"kind": kind, "value": value},
            "match_count": len(matches),
            "matches": matches,
            "expected_in_matches": anchor_node_id in matches,
            "is_ambiguous": len(matches) > 1,
        }


def check_ambiguity(
    path_steps: list[dict],
    footprint: dict,
    graph_loader: CitationGraphLoader | None = None,
    db_reader: DBReader | None = None,
    *,
    checker: AmbiguityChecker | None = None,
) -> dict[str, Any]:
    """Run the forward-walk ambiguity check for one question footprint."""
    if checker is None:
        if graph_loader is None or db_reader is None:
            raise ValueError(
                "check_ambiguity requires either `checker` or both "
                "`graph_loader` and `db_reader`."
            )
        checker = AmbiguityChecker(graph_loader, db_reader)

    node_descriptors = footprint["node_descriptors"]
    edge_types = footprint["edge_types"]
    node_ids = [step["node"]["node_id"] for step in path_steps]

    mismatches: list[int] = []
    for index, (step, edge_type) in enumerate(zip(path_steps[1:], edge_types), start=1):
        step_edge = step.get("edge_taken") or {}
        if step_edge.get("edge_type") != edge_type:
            mismatches.append(index)

    anchor_descriptor = node_descriptors[0]
    anchor_descriptor_omitted = descriptor_is_omitted(anchor_descriptor)
    anchor_result = checker.check_anchor(
        node_ids[0],
        anchor_descriptor["kind"],
        anchor_descriptor["value"],
    )

    edge_results: list[dict[str, Any]] = []
    ambiguous_indices: list[int] = []
    omitted_descriptor_pairs = consecutive_omitted_descriptor_pairs(node_descriptors)
    for index, edge_type in enumerate(edge_types, start=1):
        node_descriptor = node_descriptors[index]
        result = checker.check_edge(
            source_node_id=node_ids[index - 1],
            edge_type=edge_type,
            target_node_id=node_ids[index],
            descriptor_kind=node_descriptor["kind"],
            descriptor_value=node_descriptor["value"],
        )
        result["edge_index"] = index
        edge_results.append(result)
        if result["is_ambiguous"]:
            ambiguous_indices.append(index)

    payload: dict[str, Any] = {
        "is_ambiguous": (
            anchor_result["is_ambiguous"]
            or anchor_descriptor_omitted
            or bool(ambiguous_indices)
            or bool(omitted_descriptor_pairs)
        ),
        "anchor": anchor_result,
        "edges": edge_results,
        "ambiguous_edge_indices": ambiguous_indices,
    }
    if anchor_descriptor_omitted:
        payload["anchor_descriptor_omitted"] = True
    if omitted_descriptor_pairs:
        payload["consecutive_omitted_descriptor_pairs"] = omitted_descriptor_pairs
    if mismatches:
        payload["footprint_edge_mismatch_indices"] = mismatches
    return payload


def is_ambiguous(
    path_steps: list[dict],
    footprint: dict,
    graph_loader: CitationGraphLoader | None = None,
    db_reader: DBReader | None = None,
    *,
    checker: AmbiguityChecker | None = None,
) -> bool:
    """Boolean convenience wrapper around :func:`check_ambiguity`."""
    return check_ambiguity(
        path_steps,
        footprint,
        graph_loader,
        db_reader,
        checker=checker,
    )["is_ambiguous"]


def annotate_question_with_ambiguity(
    question: dict[str, Any],
    steps: list[dict],
    checker: AmbiguityChecker,
) -> None:
    """Attach ambiguity metadata to a generated question in place."""
    ambiguity = check_ambiguity(steps, question["footprint"], checker=checker)
    question["is_ambiguous"] = ambiguity["is_ambiguous"]
    question["ambiguity"] = ambiguity


class CachingAmbiguityChecker(AmbiguityChecker):
    """Ambiguity checker that memoizes per-(node, edge-type) adjacency.

    Only safe over an immutable graph: cached neighbour lists never expire.
    The search loops re-expand the same nodes across prefixes, directions,
    specs, and paths, so this turns repeated adjacency scans into dict hits.
    """

    def __init__(self, graph_loader: Any, db_reader: Any) -> None:
        super().__init__(graph_loader, db_reader)
        self._neighbour_edges_cache: dict[
            tuple[str, str],
            list[tuple[str, dict]],
        ] = {}

    def forward_neighbour_edges(
        self,
        node_id: str,
        edge_type: str,
    ) -> list[tuple[str, dict]]:
        cache_key = (node_id, edge_type)
        cached = self._neighbour_edges_cache.get(cache_key)
        if cached is None:
            cached = super().forward_neighbour_edges(node_id, edge_type)
            self._neighbour_edges_cache[cache_key] = cached
        return cached


def node_has_ambiguity_data(
    checker: Any,
    node_id: str,
) -> bool:
    if not node_id or ":" not in node_id:
        return False
    if node_id not in checker.graph.nodes:
        return False

    try:
        node_type = nid_type(node_id)
        db_id = nid_db_id(node_id)
    except (IndexError, ValueError):
        return False

    if node_type == NODE_PAPER:
        return checker.db.get_paper(db_id) is not None
    if node_type == NODE_AUTHOR:
        return checker.db.get_author(db_id) is not None
    return False


def missing_ambiguity_node_ids(
    checker: Any,
    node_ids: list[str],
) -> list[str]:
    missing_node_ids: list[str] = []
    seen: set[str] = set()
    for node_id in node_ids:
        missing_label = node_id or "?"
        if missing_label in seen:
            continue
        seen.add(missing_label)
        if not node_has_ambiguity_data(checker, node_id):
            missing_node_ids.append(missing_label)
    return missing_node_ids


def skipped_anchor_ambiguity_result(
    node_id: str,
    kind: str | None,
    value: Any | None,
    missing_node_ids: list[str],
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "descriptor": {"kind": kind, "value": value},
        "match_count": 1,
        "matches": [node_id],
        "expected_in_matches": True,
        "is_ambiguous": False,
        "ambiguity_skipped": True,
        "skipped_node_ids": missing_node_ids,
        "skip_reason": "node information missing from graph/database",
    }


def skipped_edge_ambiguity_result(
    source_node_id: str,
    edge_type: str,
    target_node_id: str,
    kind: str | None,
    value: Any | None,
    missing_node_ids: list[str],
) -> dict[str, Any]:
    return {
        "from": source_node_id,
        "to": target_node_id,
        "edge_type": edge_type,
        "descriptor": {"kind": kind, "value": value},
        "forward_neighbour_count": 0,
        "match_count": 1,
        "matches": [target_node_id],
        "expected_in_matches": True,
        "is_ambiguous": False,
        "ambiguity_skipped": True,
        "skipped_node_ids": missing_node_ids,
        "skip_reason": "node information missing from graph/database",
    }


class MissingNodeTolerantAmbiguityChecker:
    """Skip graph/DB-backed ambiguity checks for missing path nodes."""

    def __init__(self, checker: AmbiguityChecker) -> None:
        self.checker = checker
        self.graph = checker.graph
        self.db = checker.db

    def descriptor_unique_for_anchor(
        self,
        anchor_node_id: str,
        kind: str | None,
        value: Any | None,
    ) -> bool:
        if not node_has_ambiguity_data(self, anchor_node_id):
            return True
        return self.checker.descriptor_unique_for_anchor(
            anchor_node_id,
            kind,
            value,
        )

    def descriptor_unique_for_edge(
        self,
        source_node_id: str,
        edge_type: str,
        target_node_id: str,
        kind: str | None,
        value: Any | None,
    ) -> bool:
        if missing_ambiguity_node_ids(self, [source_node_id, target_node_id]):
            return True
        return self.checker.descriptor_unique_for_edge(
            source_node_id,
            edge_type,
            target_node_id,
            kind,
            value,
        )

    def forward_neighbour_edges(
        self,
        node_id: str,
        edge_type: str,
    ) -> list[tuple[str, dict[str, Any]]]:
        return self.checker.forward_neighbour_edges(node_id, edge_type)

    def matches_descriptor(
        self,
        candidate_node_id: str,
        kind: str | None,
        value: Any | None,
        *,
        edge_attrs: dict[str, Any] | None = None,
    ) -> bool:
        return self.checker.matches_descriptor(
            candidate_node_id,
            kind,
            value,
            edge_attrs=edge_attrs,
        )

    def check_anchor(
        self,
        anchor_node_id: str,
        kind: str | None,
        value: Any | None,
    ) -> dict[str, Any]:
        missing_node_ids = missing_ambiguity_node_ids(self, [anchor_node_id])
        if missing_node_ids:
            return skipped_anchor_ambiguity_result(
                anchor_node_id,
                kind,
                value,
                missing_node_ids,
            )
        return self.checker.check_anchor(anchor_node_id, kind, value)

    def check_edge(
        self,
        source_node_id: str,
        edge_type: str,
        target_node_id: str,
        descriptor_kind: str | None,
        descriptor_value: Any | None,
    ) -> dict[str, Any]:
        missing_node_ids = missing_ambiguity_node_ids(
            self,
            [source_node_id, target_node_id],
        )
        if missing_node_ids:
            return skipped_edge_ambiguity_result(
                source_node_id,
                edge_type,
                target_node_id,
                descriptor_kind,
                descriptor_value,
                missing_node_ids,
            )
        return self.checker.check_edge(
            source_node_id,
            edge_type,
            target_node_id,
            descriptor_kind,
            descriptor_value,
        )
