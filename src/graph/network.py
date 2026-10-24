from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass
from typing import Any

import networkx as nx
from loguru import logger

from src.setup.constants import (
    EDGE_AUTHORSHIP,
    EDGE_CITING,
    FUZZY_MATCHING_THRESHOLD,
    author_nid,
    nid_db_id,
    normalize_text,
    normalize_text_aggressive,
    paper_nid,
    preprocess_paper_title,
)
from src.utils.string_matching import compact_alnum_string, search_strings

from .storage import DatabaseManager


@dataclass
class GraphLimits:
    """Hard caps that keep the graph from growing unboundedly.

    ``max_papers`` caps explored papers, not total paper rows. Unexplored
    frontier/reference papers may exceed it so runs can resume from the saved
    frontier with a higher cap.
    """

    max_papers: int = 0
    max_authors: int = 0
    max_nodes: int = 0
    max_refs: int = 0

    def paper_ok(self, current_explored: int) -> bool:
        """Return whether another paper may be fetched and marked explored."""
        return self.max_papers == 0 or current_explored < self.max_papers

    def author_ok(self, current: int) -> bool:
        return self.max_authors == 0 or current < self.max_authors

    def nodes_ok(self, current: int) -> bool:
        return self.max_nodes == 0 or current < self.max_nodes

    def summary(self) -> str:
        parts: list[str] = []
        if self.max_papers:
            parts.append(f"explored-papers≤{self.max_papers}")
        if self.max_authors:
            parts.append(f"authors≤{self.max_authors}")
        if self.max_nodes:
            parts.append(f"nodes≤{self.max_nodes}")
        if self.max_refs:
            parts.append(f"refs/paper≤{self.max_refs}")
        return ", ".join(parts) if parts else "unlimited"

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "GraphLimits":
        valid_keys = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in valid_keys})


class MutationJournal:
    """Record small in-memory mutations so a failed DB batch can be rolled back."""

    def __init__(self, network: "CitationNetwork") -> None:
        self.network = network
        self.original_counters: dict[str, int] = {}
        self.original_index_values: dict[str, dict[str, tuple[bool, str | None]]] = {
            "_paper_index": {},
            "_paper_agg_index": {},
            "_arxiv_index": {},
            "_author_index": {},
        }
        self.original_explored_membership: dict[str, bool] = {}
        self.added_nodes: list[str] = []
        self.added_node_ids: set[str] = set()
        self.closed = False

    def record_counter(self, attr_name: str) -> None:
        if attr_name not in self.original_counters:
            self.original_counters[attr_name] = int(getattr(self.network, attr_name))

    def record_index_value(self, attr_name: str, key: str) -> None:
        originals = self.original_index_values[attr_name]
        if key in originals:
            return
        mapping = getattr(self.network, attr_name)
        if key in mapping:
            originals[key] = (True, str(mapping[key]))
            return
        originals[key] = (False, None)

    def record_explored_membership(self, db_id: str) -> None:
        if db_id not in self.original_explored_membership:
            self.original_explored_membership[db_id] = db_id in self.network._explored_set

    def record_node_addition(self, node_id: str) -> None:
        if node_id in self.added_node_ids or node_id in self.network.G:
            return
        self.added_nodes.append(node_id)
        self.added_node_ids.add(node_id)

    def commit(self) -> None:
        self.close()

    def rollback(self) -> None:
        for node_id in reversed(self.added_nodes):
            if node_id in self.network.G:
                self.network.G.remove_node(node_id)

        for db_id, was_member in self.original_explored_membership.items():
            if was_member:
                self.network._explored_set.add(db_id)
            else:
                self.network._explored_set.discard(db_id)

        for attr_name, originals in self.original_index_values.items():
            mapping = getattr(self.network, attr_name)
            for key, (existed, value) in originals.items():
                if existed and value is not None:
                    mapping[key] = value
                else:
                    mapping.pop(key, None)

        for attr_name, value in self.original_counters.items():
            setattr(self.network, attr_name, value)

        self.close()

    def close(self) -> None:
        if self.closed:
            return
        if self.network.active_mutation_journal is self:
            self.network.active_mutation_journal = None
        self.closed = True


class CitationNetwork:
    """Directed citation-network graph backed by a SQLite database."""

    def __init__(self, db: DatabaseManager, limits: GraphLimits | None = None) -> None:
        self.G: nx.DiGraph = nx.DiGraph()
        self.db = db
        self.limits = limits or GraphLimits()
        self.max_depth: int | None = 0

        self._paper_index: dict[str, str] = {}
        self._paper_agg_index: dict[str, str] = {}
        self._arxiv_index: dict[str, str] = {}
        self._author_index: dict[str, str] = {}
        self._explored_set: set[str] = set()

        self._n_papers = 0
        self._n_authors = 0
        self.litellm_usage: dict[str, Any] = {}
        self.active_mutation_journal: MutationJournal | None = None

    def start_mutation_journal(self) -> MutationJournal:
        if self.active_mutation_journal is not None:
            raise RuntimeError("Nested mutation journals are not supported")
        journal = MutationJournal(self)
        self.active_mutation_journal = journal
        return journal

    def record_counter_before_change(self, attr_name: str) -> None:
        if self.active_mutation_journal is not None:
            self.active_mutation_journal.record_counter(attr_name)

    def record_index_before_change(self, attr_name: str, key: str) -> None:
        if self.active_mutation_journal is not None:
            self.active_mutation_journal.record_index_value(attr_name, key)

    def record_explored_membership_before_change(self, db_id: str) -> None:
        if self.active_mutation_journal is not None:
            self.active_mutation_journal.record_explored_membership(db_id)

    def record_node_addition_before_change(self, node_id: str) -> None:
        if self.active_mutation_journal is not None:
            self.active_mutation_journal.record_node_addition(node_id)

    def index_paper(self, db_id: str, *, title: str = "", arxiv_id: str = "") -> None:
        title = preprocess_paper_title(title)
        if title:
            paper_key = normalize_text(title)
            paper_agg_key = normalize_text_aggressive(title)
            self.record_index_before_change("_paper_index", paper_key)
            self.record_index_before_change("_paper_agg_index", paper_agg_key)
            self._paper_index[paper_key] = db_id
            self._paper_agg_index[paper_agg_key] = db_id
        if arxiv_id:
            self.record_index_before_change("_arxiv_index", arxiv_id)
            self._arxiv_index[arxiv_id] = db_id

    def find_paper(self, title: str) -> str | None:
        title = preprocess_paper_title(title)
        return self._paper_index.get(normalize_text(title))

    def find_paper_aggressive(self, title: str) -> str | None:
        title = preprocess_paper_title(title)
        return self._paper_agg_index.get(normalize_text_aggressive(title))

    def find_paper_fuzzy(
        self,
        title: str,
        threshold: float = FUZZY_MATCHING_THRESHOLD,
    ) -> str | None:
        title = preprocess_paper_title(title)
        stored_titles = list(self._paper_index.keys())
        match_index = search_strings(normalize_text(title), stored_titles, threshold)
        if match_index < 0:
            return None
        return self._paper_index[stored_titles[match_index]]

    def resolve_paper(self, title: str, arxiv_id: str = "") -> str | None:
        title = preprocess_paper_title(title)
        if arxiv_id:
            db_id = self._arxiv_index.get(arxiv_id)
            if db_id is not None:
                return db_id
        db_id = self.find_paper(title)
        if db_id is not None:
            return db_id
        db_id = self.find_paper_aggressive(title)
        if db_id is not None:
            return db_id
        if len(title) > 20:
            return self.find_paper_fuzzy(title)
        return None

    def find_author(self, name: str) -> str | None:
        return self._author_index.get(normalize_text(name))

    def is_explored(self, db_id: str) -> bool:
        return db_id in self._explored_set

    def explored_count(self) -> int:
        return len(self._explored_set)

    def insert_new_paper(
        self,
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
    ) -> str:
        title = preprocess_paper_title(title)
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
        self.record_counter_before_change("_n_papers")
        self._n_papers += 1
        self.index_paper(db_id, title=title, arxiv_id=arxiv_id)

        node_id = paper_nid(db_id)
        self.record_node_addition_before_change(node_id)
        self.G.add_node(node_id)
        if explored:
            self.record_explored_membership_before_change(db_id)
            self._explored_set.add(db_id)

        logger.debug("Inserted paper db_id={} title={!r}", db_id, title[:60])
        return node_id

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
        updates: dict[str, Any] = {}

        if title:
            existing_title = row["title"] or ""
            cleaned_existing_title = preprocess_paper_title(existing_title)
            # Replace the stored title only when the new one is strictly better:
            #   1. existing is empty — accept anything,
            #   2. existing is not in canonical (preprocessed) form — replace with the cleaner copy,
            #   3. canonical metadata arrived (arxiv_id present) and the new title is longer
            #      than the cleaned existing one — prefer the fuller arXiv version.
            if existing_title != title and (
                not existing_title
                or existing_title != cleaned_existing_title
                or (arxiv_id and len(title) > len(cleaned_existing_title))
            ):
                if existing_title:
                    existing_paper_key = normalize_text(existing_title)
                    existing_paper_agg_key = normalize_text_aggressive(existing_title)
                    self.record_index_before_change("_paper_index", existing_paper_key)
                    self.record_index_before_change(
                        "_paper_agg_index",
                        existing_paper_agg_key,
                    )
                    self._paper_index.pop(existing_paper_key, None)
                    self._paper_agg_index.pop(existing_paper_agg_key, None)
                updates["title"] = title
                self.index_paper(db_id, title=title)

        if arxiv_id and not row["arxiv_id"]:
            updates["arxiv_id"] = arxiv_id
            self.index_paper(db_id, arxiv_id=arxiv_id)
        if semantic_scholar_id and not row["semantic_scholar_id"]:
            updates["semantic_scholar_id"] = semantic_scholar_id
        if abstract and not row["abstract"]:
            updates["abstract"] = abstract
        if published and not row["published"]:
            updates["published"] = published
        if venue and not row["venue"]:
            updates["venue"] = venue
        if pdf_url and not row["pdf_url"]:
            updates["pdf_url"] = pdf_url
        if text_path and not row["text_path"]:
            updates["text_path"] = text_path
        if explored and not row["explored"]:
            updates["explored"] = 1
            self.record_explored_membership_before_change(db_id)
            self._explored_set.add(db_id)
        if depth >= 0:
            existing_depth = row["depth"] if row["depth"] is not None else -1
            if int(existing_depth) < 0 or depth < int(existing_depth):
                updates["depth"] = depth
        # Fresh Semantic Scholar validation values always overwrite old ones.
        if reference_count is not None:
            updates["reference_count"] = reference_count
        if valid_references is not None:
            updates["valid_references"] = valid_references

        if updates:
            self.db.update_paper(db_id, **updates)

        node_id = paper_nid(db_id)
        if node_id not in self.G:
            self.record_node_addition_before_change(node_id)
            self.G.add_node(node_id)
        return node_id

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
            logger.warning("Skipping paper with no title, arxiv_id, or Semantic Scholar ID")
            return None

        existing_db_id = self.resolve_paper(title, arxiv_id=arxiv_id)
        if existing_db_id is None:
            if not self.limits.nodes_ok(self.G.number_of_nodes()):
                logger.debug(
                    "Node cap ({}) reached — skipping paper {!r}",
                    self.limits.max_nodes,
                    title[:60],
                )
                return None
            return self.insert_new_paper(
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

        return self.update_existing_paper(
            existing_db_id,
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
        return self.insert_or_update_author(
            name=name,
            semantic_scholar_id=semantic_scholar_id,
            orcid=orcid,
        )

    def insert_new_author(
        self,
        *,
        name: str = "",
        semantic_scholar_id: str = "",
        orcid: str = "",
    ) -> str | None:
        if not self.limits.author_ok(self._n_authors):
            logger.debug(
                "Author cap ({}) reached — skipping {}",
                self.limits.max_authors,
                name,
            )
            return None
        if not self.limits.nodes_ok(self.G.number_of_nodes()):
            logger.debug(
                "Node cap ({}) reached — skipping author {}",
                self.limits.max_nodes,
                name,
            )
            return None

        db_id = self.db.insert_author(
            name=name,
            semantic_scholar_id=semantic_scholar_id,
            orcid=orcid,
        )
        self.record_counter_before_change("_n_authors")
        self._n_authors += 1
        author_key = normalize_text(name)
        self.record_index_before_change("_author_index", author_key)
        self._author_index[author_key] = db_id

        node_id = author_nid(db_id)
        self.record_node_addition_before_change(node_id)
        self.G.add_node(node_id)
        logger.debug("Inserted author db_id={} name={} orcid={}", db_id, name, orcid)
        return node_id

    def insert_or_update_author(
        self,
        *,
        name: str = "",
        semantic_scholar_id: str = "",
        orcid: str = "",
    ) -> str | None:
        existing_db_id = self.find_author(name)
        if existing_db_id is None:
            return self.insert_new_author(
                name=name,
                semantic_scholar_id=semantic_scholar_id,
                orcid=orcid,
            )

        db_id = existing_db_id
        if semantic_scholar_id or orcid:
            row = self.db.get_author(db_id)
            updates: dict[str, Any] = {}
            if row and semantic_scholar_id and not row["semantic_scholar_id"]:
                updates["semantic_scholar_id"] = semantic_scholar_id
            if row and orcid and not row["orcid"]:
                updates["orcid"] = orcid
            if updates:
                self.db.update_author(db_id, **updates)
                logger.debug("Updated author db_id={} fields={}", db_id, sorted(updates))
        node_id = author_nid(db_id)
        if node_id not in self.G:
            self.record_node_addition_before_change(node_id)
            self.G.add_node(node_id)
        return node_id

    def add_authorship(
        self,
        paper_node: str,
        author_node: str,
        *,
        position: int = 0,
        organization: str = "",
        email: str = "",
    ) -> None:
        if paper_node not in self.G or author_node not in self.G:
            return
        if not self.G.has_edge(paper_node, author_node):
            self.G.add_edge(
                paper_node,
                author_node,
                edge_type=EDGE_AUTHORSHIP,
                position=position,
            )
        self.db.insert_authorship(
            paper_id=nid_db_id(paper_node),
            author_id=nid_db_id(author_node),
            organization=organization,
            email=email,
            position=position,
        )

    def add_pending_reference_author(
        self,
        paper_node: str,
        source_paper_node: str,
        *,
        position: int,
        name: str,
    ) -> None:
        if paper_node not in self.G or source_paper_node not in self.G:
            return
        name = name.strip()
        if not name:
            return
        self.db.insert_pending_reference_author(
            paper_id=nid_db_id(paper_node),
            source_paper_id=nid_db_id(source_paper_node),
            position=position,
            name=name,
        )

    def add_author_name_variant(self, author_node: str, variant_name: str) -> None:
        if author_node not in self.G:
            return
        variant_name = variant_name.strip()
        normalized_name = compact_alnum_string(variant_name)
        if not variant_name or not normalized_name:
            return
        self.db.insert_author_name_variant(
            author_id=nid_db_id(author_node),
            variant_name=variant_name,
            normalized_name=normalized_name,
        )

    def add_citation(self, citing_node: str, cited_node: str) -> None:
        if citing_node not in self.G or cited_node not in self.G:
            return
        if not self.G.has_edge(citing_node, cited_node):
            self.G.add_edge(citing_node, cited_node, edge_type=EDGE_CITING)
        self.db.insert_citation(
            citing_paper_id=nid_db_id(citing_node),
            cited_paper_id=nid_db_id(cited_node),
        )

    def mark_explored(self, paper_node: str) -> None:
        db_id = nid_db_id(paper_node)
        self.db.update_paper(db_id, explored=1)
        self.record_explored_membership_before_change(db_id)
        self._explored_set.add(db_id)

    def mark_unexplored(self, paper_node: str) -> None:
        db_id = nid_db_id(paper_node)
        self.db.update_paper(db_id, explored=0)
        self.record_explored_membership_before_change(db_id)
        self._explored_set.discard(db_id)

    def at_capacity(self, type_count: int, type_cap: int) -> bool:
        if self.limits.max_nodes and self.G.number_of_nodes() >= self.limits.max_nodes:
            return True
        return bool(type_cap and type_count >= type_cap)

    def at_paper_capacity(self) -> bool:
        return bool(
            self.limits.max_papers
            and self.explored_count() >= self.limits.max_papers
        )

    def at_author_capacity(self) -> bool:
        return self.at_capacity(self._n_authors, self.limits.max_authors)

    def stats(self) -> str:
        cap_a = f"/{self.limits.max_authors}" if self.limits.max_authors else ""
        explored_cap = (
            f"/{self.limits.max_papers}" if self.limits.max_papers else ""
        )
        return (
            f"Graph: {self.G.number_of_nodes()} nodes "
            f"({self._n_papers} papers, {self.explored_count()}{explored_cap} explored, "
            f"{self._n_authors}{cap_a} authors), "
            f"{self.G.number_of_edges()} edges"
        )
