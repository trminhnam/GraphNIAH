"""Visible-graph helpers for scholarly reachability yes/no questions."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

import networkx as nx


QUESTION_TYPE = "reachability-yesno"
ANSWER_TYPE = "REACHABILITY_YESNO"
ANSWER_YES = "YES"
ANSWER_NO = "NO"
ANSWER_UNKNOWN = "UNKNOWN"
COMPLEXITY_EXPECTED = "symmetric_expected_bfs"
COMPLEXITY_EXPECTED_DFS = "symmetric_expected_dfs"
COMPLEXITY_EXHAUSTIVE = "symmetric_exhaustive_component"
TERMINAL_REFERENCE_PREFIX = "terminal-reference:"

CitationPair = tuple[str, str]
AuthorshipPair = tuple[str, str]


@dataclass(frozen=True)
class VisibleReachabilityGraph:
    """One structured graph exactly matching the displayed paper needles."""

    graph: nx.Graph
    expanded_paper_ids: frozenset[str]
    authors_by_paper: dict[str, tuple[str, ...]]
    references_by_paper: dict[str, tuple[str, ...]]
    terminal_reference_nodes: dict[str, tuple[CitationPair, ...]]


@dataclass(frozen=True)
class ReferenceSelection:
    """Reference allowlist that yields one visible source-target path."""

    references_by_paper: dict[str, tuple[str, ...]]
    excluded_references: tuple[dict[str, str], ...]
    witness_node_ids: tuple[str, ...]


@dataclass(frozen=True)
class IdentityReferenceSelection:
    """Reference allowlist whose visible paper/author identities are unique."""

    references_by_paper: dict[str, tuple[str, ...]]
    excluded_references: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class VisibleIdentityRegistry:
    """Model-visible identity keys for one rendered reachability graph."""

    paper_titles_by_visible_node: dict[str, str]
    author_names_by_node: dict[str, str]
    paper_collisions: dict[str, tuple[str, ...]]
    author_collisions: dict[str, tuple[str, ...]]
    missing_paper_node_ids: tuple[str, ...]
    missing_author_node_ids: tuple[str, ...]
    reference_owner_collisions: dict[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    paper_author_id_collisions: dict[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    paper_author_name_collisions: dict[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    disallowed_expanded_reference_pairs: tuple[CitationPair, ...] = ()
    hidden_citation_pairs: tuple[CitationPair, ...] = ()

    @property
    def is_unique(self) -> bool:
        """Return whether every visible node has one collision-free identity."""
        return not (
            self.paper_collisions
            or self.author_collisions
            or self.reference_owner_collisions
            or self.paper_author_id_collisions
            or self.paper_author_name_collisions
            or self.disallowed_expanded_reference_pairs
            or self.hidden_citation_pairs
            or self.missing_paper_node_ids
            or self.missing_author_node_ids
        )


def deduplicate(values: Iterable[str]) -> tuple[str, ...]:
    """Return non-empty strings once, preserving their input order."""
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return tuple(output)


def normalize_mapping(
    values_by_key: Mapping[str, Iterable[str]],
) -> dict[str, tuple[str, ...]]:
    """Return one deterministic mapping of identifiers to unique identifiers."""
    return {
        str(key): deduplicate(values)
        for key, values in sorted(values_by_key.items(), key=lambda item: str(item[0]))
    }


def terminal_reference_node_id(
    citing_node_id: str,
    cited_node_id: str,
) -> str:
    """Return the legacy occurrence-node ID used before canonical paper nodes."""
    return f"{TERMINAL_REFERENCE_PREFIX}{citing_node_id}->{cited_node_id}"


def normalize_visible_identity(value: Any) -> str:
    """Return a Unicode-aware key approximating model-visible identity."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = "".join(character if character.isalnum() else " " for character in text)
    return re.sub(r"\s+", " ", text).strip()


def build_visible_graph(
    *,
    expanded_paper_ids: Iterable[str],
    authors_by_paper: Mapping[str, Iterable[str]],
    references_by_paper: Mapping[str, Iterable[str]],
) -> VisibleReachabilityGraph:
    """Build the finite graph exposed by structured paper needles.

    Every occurrence of the same paper identifier uses the same graph node,
    including papers that appear only in reference lists. This matches the
    prompt's paper-identity semantics: one paper is one vertex.
    """
    expanded = frozenset(deduplicate(expanded_paper_ids))
    normalized_authors = normalize_mapping(authors_by_paper)
    normalized_references = normalize_mapping(references_by_paper)
    graph = nx.Graph()
    terminal_pairs_by_node: dict[str, list[CitationPair]] = {}

    for paper_id in sorted(expanded):
        graph.add_node(paper_id, node_type="paper", expanded=True)

    for paper_id in sorted(expanded):
        for author_id in normalized_authors.get(paper_id, ()):
            graph.add_node(author_id, node_type="author", expanded=True)
            graph.add_edge(paper_id, author_id, relation_types={"authorship"})

        for cited_id in normalized_references.get(paper_id, ()):
            visible_cited_id = cited_id
            if cited_id in expanded:
                graph.add_node(cited_id, node_type="paper", expanded=True)
            else:
                terminal_pairs_by_node.setdefault(cited_id, []).append(
                    (paper_id, cited_id)
                )
                graph.add_node(
                    visible_cited_id,
                    node_type="paper",
                    expanded=False,
                    canonical_node_id=cited_id,
                )
            if graph.has_edge(paper_id, visible_cited_id):
                graph[paper_id][visible_cited_id].setdefault(
                    "relation_types", set()
                ).add("citation")
                graph[paper_id][visible_cited_id].setdefault(
                    "citation_pairs", set()
                ).add((paper_id, cited_id))
            else:
                graph.add_edge(
                    paper_id,
                    visible_cited_id,
                    relation_types={"citation"},
                    citation_pairs={(paper_id, cited_id)},
                )

    return VisibleReachabilityGraph(
        graph=graph,
        expanded_paper_ids=expanded,
        authors_by_paper=normalized_authors,
        references_by_paper=normalized_references,
        terminal_reference_nodes={
            node_id: tuple(sorted(pairs))
            for node_id, pairs in sorted(terminal_pairs_by_node.items())
        },
    )


def build_visible_identity_registry(
    visible_graph: VisibleReachabilityGraph,
    *,
    paper_titles_by_paper: Mapping[str, str],
    author_names_by_author: Mapping[str, str],
    citation_owners_by_paper: Mapping[str, Iterable[str]] | None = None,
    paper_author_ids_by_paper: Mapping[str, Iterable[str]] | None = None,
    allowed_reference_pairs: Iterable[CitationPair] = (),
    allowed_terminal_reference_owners: Mapping[
        str, Iterable[str]
    ] | None = None,
    citation_neighbors_by_paper: Mapping[str, Iterable[str]] | None = None,
) -> VisibleIdentityRegistry:
    """Build the identity graph the model can infer from rendered strings.

    The optional arguments enable stricter checks for rendered citation
    items. ``citation_owners_by_paper`` maps a cited paper ID to the paper
    blocks whose reference lists contain it. When supplied, the visible graph's
    own citation owners are merged into that mapping. A terminal paper may then
    have only one owner unless its exact owner set appears in
    ``allowed_terminal_reference_owners``. References to an expanded paper must
    be explicitly
    listed in ``allowed_reference_pairs`` because that paper is rendered both
    as a reference item and as a full block.

    ``paper_author_ids_by_paper`` supplies author identities for terminal
    citation items. When supplied, those identities are combined with authors
    already present on expanded blocks, and an author ID or normalized author
    name may belong to only one distinct rendered paper identity.

    ``citation_neighbors_by_paper`` optionally records canonical citation
    neighbours so hidden edges between two retained papers can be rejected.
    Omitting the optional arguments preserves the legacy validation behavior.
    """
    paper_titles_by_visible_node: dict[str, str] = {}
    for paper_id in sorted(visible_graph.expanded_paper_ids):
        paper_titles_by_visible_node[paper_id] = str(
            paper_titles_by_paper.get(paper_id) or ""
        ).strip()
    for terminal_id, citation_pairs in sorted(
        visible_graph.terminal_reference_nodes.items()
    ):
        paper_titles_by_visible_node[terminal_id] = str(
            paper_titles_by_paper.get(citation_pairs[0][1]) or ""
        ).strip()

    author_names_by_node: dict[str, str] = {
        str(node_id): str(author_names_by_author.get(str(node_id)) or "").strip()
        for node_id, attributes in visible_graph.graph.nodes(data=True)
        if attributes.get("node_type") == "author"
    }

    rendered_author_ids_by_paper: dict[str, tuple[str, ...]] = {}
    if paper_author_ids_by_paper is not None:
        visible_paper_ids = set(paper_titles_by_visible_node)
        for paper_id in sorted(visible_paper_ids):
            rendered_author_ids_by_paper[paper_id] = deduplicate(
                list(visible_graph.authors_by_paper.get(paper_id, ()))
                + list(paper_author_ids_by_paper.get(paper_id, ()))
            )
        for author_ids in rendered_author_ids_by_paper.values():
            for author_id in author_ids:
                author_names_by_node[author_id] = str(
                    author_names_by_author.get(author_id) or ""
                ).strip()

    paper_nodes_by_key: dict[str, list[str]] = {}
    missing_papers: list[str] = []
    for node_id, title in paper_titles_by_visible_node.items():
        key = normalize_visible_identity(title)
        if not key:
            missing_papers.append(node_id)
            continue
        paper_nodes_by_key.setdefault(key, []).append(node_id)

    author_nodes_by_key: dict[str, list[str]] = {}
    missing_authors: list[str] = []
    for node_id, name in author_names_by_node.items():
        key = normalize_visible_identity(name)
        if not key:
            missing_authors.append(node_id)
            continue
        author_nodes_by_key.setdefault(key, []).append(node_id)

    reference_owner_collisions: dict[str, tuple[str, ...]] = {}
    disallowed_expanded_reference_pairs: list[CitationPair] = []
    if citation_owners_by_paper is not None:
        allowed_terminal_owners = {
            str(cited_id): tuple(sorted(deduplicate(owners)))
            for cited_id, owners in (
                allowed_terminal_reference_owners or {}
            ).items()
        }
        owners_by_cited: dict[str, list[str]] = {
            str(cited_id): list(deduplicate(owners))
            for cited_id, owners in citation_owners_by_paper.items()
        }
        for citing_id, cited_ids in visible_graph.references_by_paper.items():
            for cited_id in cited_ids:
                owners_by_cited.setdefault(cited_id, []).append(citing_id)
        normalized_owners = {
            cited_id: deduplicate(owners)
            for cited_id, owners in owners_by_cited.items()
        }
        allowed_pairs = {
            (str(citing_id), str(cited_id))
            for citing_id, cited_id in allowed_reference_pairs
        }
        for cited_id, owners in sorted(normalized_owners.items()):
            if cited_id not in paper_titles_by_visible_node:
                continue
            if cited_id not in visible_graph.expanded_paper_ids:
                normalized_owner_set = tuple(sorted(owners))
                if (
                    len(normalized_owner_set) > 1
                    and allowed_terminal_owners.get(cited_id)
                    != normalized_owner_set
                ):
                    reference_owner_collisions[cited_id] = normalized_owner_set
                continue
            disallowed_expanded_reference_pairs.extend(
                (owner_id, cited_id)
                for owner_id in owners
                if (owner_id, cited_id) not in allowed_pairs
            )

    papers_by_author_id: dict[str, list[str]] = {}
    papers_by_author_name: dict[str, list[str]] = {}
    for paper_id, author_ids in rendered_author_ids_by_paper.items():
        for author_id in author_ids:
            papers_by_author_id.setdefault(author_id, []).append(paper_id)
            author_key = normalize_visible_identity(
                author_names_by_author.get(author_id)
            )
            if author_key:
                papers_by_author_name.setdefault(author_key, []).append(paper_id)

    hidden_citation_pairs: set[CitationPair] = set()
    if citation_neighbors_by_paper is not None:
        visible_paper_ids = set(paper_titles_by_visible_node)
        rendered_edges = {
            frozenset((str(owner), str(cited_id)))
            for owner, cited_ids in visible_graph.references_by_paper.items()
            for cited_id in cited_ids
        }
        for paper_id, neighbor_ids in citation_neighbors_by_paper.items():
            for neighbor_id in deduplicate(neighbor_ids):
                if (
                    paper_id == neighbor_id
                    or paper_id not in visible_paper_ids
                    or neighbor_id not in visible_paper_ids
                ):
                    continue
                if frozenset((paper_id, neighbor_id)) not in rendered_edges:
                    hidden_citation_pairs.add(
                        tuple(sorted((str(paper_id), str(neighbor_id))))
                    )

    return VisibleIdentityRegistry(
        paper_titles_by_visible_node=paper_titles_by_visible_node,
        author_names_by_node=author_names_by_node,
        paper_collisions={
            key: tuple(sorted(node_ids))
            for key, node_ids in paper_nodes_by_key.items()
            if len(set(node_ids)) > 1
        },
        author_collisions={
            key: tuple(sorted(node_ids))
            for key, node_ids in author_nodes_by_key.items()
            if len(set(node_ids)) > 1
        },
        missing_paper_node_ids=tuple(sorted(missing_papers)),
        missing_author_node_ids=tuple(sorted(missing_authors)),
        reference_owner_collisions=reference_owner_collisions,
        paper_author_id_collisions={
            author_id: tuple(sorted(set(paper_ids)))
            for author_id, paper_ids in papers_by_author_id.items()
            if len(set(paper_ids)) > 1
        },
        paper_author_name_collisions={
            author_key: tuple(sorted(set(paper_ids)))
            for author_key, paper_ids in papers_by_author_name.items()
            if len(set(paper_ids)) > 1
        },
        disallowed_expanded_reference_pairs=tuple(
            sorted(set(disallowed_expanded_reference_pairs))
        ),
        hidden_citation_pairs=tuple(sorted(hidden_citation_pairs)),
    )


def sanitize_references_for_identity_uniqueness(
    *,
    expanded_paper_ids: Iterable[str],
    authors_by_paper: Mapping[str, Iterable[str]],
    references_by_paper: Mapping[str, Iterable[str]],
    protected_reference_pairs: Iterable[CitationPair],
    paper_titles_by_paper: Mapping[str, str],
    author_names_by_author: Mapping[str, str],
) -> IdentityReferenceSelection | None:
    """Prune optional references until every model-visible identity is unique."""
    allowed = {
        str(owner): list(deduplicate(references))
        for owner, references in references_by_paper.items()
    }
    protected = {
        (str(citing), str(cited))
        for citing, cited in protected_reference_pairs
    }
    excluded: list[dict[str, str]] = []

    while True:
        visible = build_visible_graph(
            expanded_paper_ids=expanded_paper_ids,
            authors_by_paper=authors_by_paper,
            references_by_paper=allowed,
        )
        registry = build_visible_identity_registry(
            visible,
            paper_titles_by_paper=paper_titles_by_paper,
            author_names_by_author=author_names_by_author,
        )
        if registry.author_collisions or registry.missing_author_node_ids:
            return None

        removable_pairs: set[CitationPair] = set()
        for node_id in registry.missing_paper_node_ids:
            pairs = visible.terminal_reference_nodes.get(node_id)
            if not pairs or any(pair in protected for pair in pairs):
                return None
            removable_pairs.update(pairs)

        for node_ids in registry.paper_collisions.values():
            nonremovable = [
                node_id
                for node_id in node_ids
                if node_id in visible.expanded_paper_ids
                or any(
                    pair in protected
                    for pair in visible.terminal_reference_nodes.get(node_id, ())
                )
            ]
            if len(nonremovable) > 1:
                return None
            kept_node_id = nonremovable[0] if nonremovable else min(node_ids)
            for node_id in node_ids:
                if node_id == kept_node_id:
                    continue
                pairs = visible.terminal_reference_nodes.get(node_id)
                if not pairs or any(pair in protected for pair in pairs):
                    return None
                removable_pairs.update(pairs)

        if not removable_pairs:
            return (
                IdentityReferenceSelection(
                    references_by_paper=normalize_mapping(allowed),
                    excluded_references=tuple(excluded),
                )
                if registry.is_unique
                else None
            )

        for citing_id, cited_id in sorted(removable_pairs):
            if cited_id not in allowed.get(citing_id, []):
                continue
            allowed[citing_id].remove(cited_id)
            excluded.append(
                {
                    "citing_node_id": citing_id,
                    "cited_node_id": cited_id,
                    "reason": "model_visible_identity_collision",
                }
            )


def find_path(
    visible_graph: VisibleReachabilityGraph,
    source_node_id: str,
    target_node_id: str,
) -> tuple[str, ...]:
    """Return one deterministic shortest path, or an empty tuple."""
    graph = visible_graph.graph
    if source_node_id not in graph or target_node_id not in graph:
        return ()
    try:
        paths = nx.all_shortest_paths(graph, source_node_id, target_node_id)
        return min(tuple(path) for path in paths)
    except nx.NetworkXNoPath:
        return ()


def bridge_edge_keys(graph: nx.Graph) -> set[frozenset[str]]:
    """Return undirected bridge keys for one graph."""
    return {frozenset((str(left), str(right))) for left, right in nx.bridges(graph)}


def path_conflict_count(
    visible_graph: VisibleReachabilityGraph,
    source_node_id: str,
    target_node_id: str,
) -> int | None:
    """Return non-bridge edges on one path, or ``None`` when unreachable."""
    path = find_path(visible_graph, source_node_id, target_node_id)
    if not path:
        return None
    bridges = bridge_edge_keys(visible_graph.graph)
    return sum(
        frozenset((path[index - 1], path[index])) not in bridges
        for index in range(1, len(path))
    )


def path_count_capped(
    visible_graph: VisibleReachabilityGraph,
    source_node_id: str,
    target_node_id: str,
) -> int:
    """Return 0, 1, or 2, where 2 means two or more simple paths."""
    conflict_count = path_conflict_count(
        visible_graph,
        source_node_id,
        target_node_id,
    )
    if conflict_count is None:
        return 0
    return 1 if conflict_count == 0 else 2


def sanitize_references_for_unique_path(
    *,
    expanded_paper_ids: Iterable[str],
    authors_by_paper: Mapping[str, Iterable[str]],
    references_by_paper: Mapping[str, Iterable[str]],
    protected_reference_pairs: Iterable[CitationPair],
    source_node_id: str,
    target_node_id: str,
    witness_node_ids: Sequence[str] | None = None,
) -> ReferenceSelection | None:
    """Remove only reference items found on concrete alternative paths."""
    allowed = {
        str(owner): list(deduplicate(references))
        for owner, references in references_by_paper.items()
    }
    protected = {
        (str(citing), str(cited))
        for citing, cited in protected_reference_pairs
    }
    excluded: list[dict[str, str]] = []

    initial_visible = build_visible_graph(
        expanded_paper_ids=expanded_paper_ids,
        authors_by_paper=authors_by_paper,
        references_by_paper=allowed,
    )
    if witness_node_ids is None:
        protected_references = {
            owner: [
                cited_id
                for cited_id in references
                if (owner, cited_id) in protected
            ]
            for owner, references in allowed.items()
        }
        protected_visible = build_visible_graph(
            expanded_paper_ids=expanded_paper_ids,
            authors_by_paper=authors_by_paper,
            references_by_paper=protected_references,
        )
        witness = find_path(
            protected_visible,
            source_node_id,
            target_node_id,
        )
    else:
        witness = tuple(str(node_id) for node_id in witness_node_ids)
    if (
        not witness
        or witness[0] != source_node_id
        or witness[-1] != target_node_id
        or any(
            not initial_visible.graph.has_edge(witness[index - 1], witness[index])
            for index in range(1, len(witness))
        )
    ):
        return None

    witness_edge_keys = {
        frozenset((witness[index - 1], witness[index]))
        for index in range(1, len(witness))
    }

    while True:
        visible = build_visible_graph(
            expanded_paper_ids=expanded_paper_ids,
            authors_by_paper=authors_by_paper,
            references_by_paper=allowed,
        )
        graph = visible.graph
        if any(
            not graph.has_edge(witness[index - 1], witness[index])
            for index in range(1, len(witness))
        ):
            return None
        bridges = bridge_edge_keys(graph)
        conflicting_witness_edges = [
            (witness[index - 1], witness[index])
            for index in range(1, len(witness))
            if frozenset((witness[index - 1], witness[index])) not in bridges
        ]
        if not conflicting_witness_edges:
            return ReferenceSelection(
                references_by_paper=normalize_mapping(allowed),
                excluded_references=tuple(excluded),
                witness_node_ids=witness,
            )

        conflict_left, conflict_right = conflicting_witness_edges[0]
        graph_without_witness_edge = graph.copy()
        graph_without_witness_edge.remove_edge(conflict_left, conflict_right)
        try:
            alternative_path = nx.shortest_path(
                graph_without_witness_edge,
                conflict_left,
                conflict_right,
            )
        except nx.NetworkXNoPath:
            return None

        removable_edges: list[tuple[tuple[CitationPair, ...], str, str]] = []
        for index in range(1, len(alternative_path)):
            left = str(alternative_path[index - 1])
            right = str(alternative_path[index])
            if frozenset((left, right)) in witness_edge_keys:
                continue
            citation_pairs = tuple(
                sorted(graph[left][right].get("citation_pairs") or ())
            )
            if citation_pairs and all(
                pair not in protected for pair in citation_pairs
            ):
                removable_edges.append((citation_pairs, left, right))

        if not removable_edges:
            return None
        citation_pairs, _left, _right = min(
            removable_edges,
            key=lambda item: (item[0], item[1], item[2]),
        )
        for citing_id, cited_id in citation_pairs:
            allowed[citing_id].remove(cited_id)
            excluded.append(
                {
                    "citing_node_id": citing_id,
                    "cited_node_id": cited_id,
                    "reason": "alternative_reachability_path",
                }
            )


def select_bridge_safe_references(
    *,
    expanded_paper_ids: Iterable[str],
    authors_by_paper: Mapping[str, Iterable[str]],
    references_by_paper: Mapping[str, Iterable[str]],
    protected_reference_pairs: Iterable[CitationPair],
    source_node_id: str,
    target_node_id: str,
    witness_node_ids: Sequence[str],
) -> ReferenceSelection | None:
    """Add optional references only while the witness remains the unique path."""
    normalized_references = normalize_mapping(references_by_paper)
    protected = {
        (str(citing), str(cited))
        for citing, cited in protected_reference_pairs
    }
    witness = tuple(str(node_id) for node_id in witness_node_ids)
    allowed: dict[str, list[str]] = {
        owner: [
            cited_id
            for cited_id in references
            if (owner, cited_id) in protected
        ]
        for owner, references in normalized_references.items()
    }
    excluded: list[dict[str, str]] = []

    protected_visible = build_visible_graph(
        expanded_paper_ids=expanded_paper_ids,
        authors_by_paper=authors_by_paper,
        references_by_paper=allowed,
    )
    if (
        not witness
        or witness[0] != source_node_id
        or witness[-1] != target_node_id
        or path_count_capped(
            protected_visible,
            source_node_id,
            target_node_id,
        )
        != 1
    ):
        return None

    for owner, references in normalized_references.items():
        for cited_id in references:
            pair = (owner, cited_id)
            if pair in protected:
                continue
            allowed.setdefault(owner, []).append(cited_id)
            visible = build_visible_graph(
                expanded_paper_ids=expanded_paper_ids,
                authors_by_paper=authors_by_paper,
                references_by_paper=allowed,
            )
            if path_count_capped(visible, source_node_id, target_node_id) == 1:
                continue
            allowed[owner].remove(cited_id)
            excluded.append(
                {
                    "citing_node_id": owner,
                    "cited_node_id": cited_id,
                    "reason": "alternative_reachability_path",
                }
            )

    return ReferenceSelection(
        references_by_paper=normalize_mapping(allowed),
        excluded_references=tuple(excluded),
        witness_node_ids=witness,
    )


def yes_expected_search_steps(
    visible_graph: VisibleReachabilityGraph,
    source_node_id: str,
    target_node_id: str,
) -> float | None:
    """Return the cheaper endpoint's expected BFS visits for a Yes graph."""
    if path_count_capped(visible_graph, source_node_id, target_node_id) != 1:
        return None
    source_cost = rooted_expected_bfs_steps(
        visible_graph.graph,
        source_node_id,
        target_node_id,
    )
    target_cost = rooted_expected_bfs_steps(
        visible_graph.graph,
        target_node_id,
        source_node_id,
    )
    if source_cost is None or target_cost is None:
        return None
    return min(source_cost, target_cost)


def no_exhaustive_search_steps(
    visible_graph: VisibleReachabilityGraph,
    source_node_id: str,
    target_node_id: str,
) -> float | None:
    """Return the cheaper endpoint component size for a valid No graph."""
    graph = visible_graph.graph
    if source_node_id not in graph or target_node_id not in graph:
        return None
    if nx.has_path(graph, source_node_id, target_node_id):
        return None
    source_size = len(nx.node_connected_component(graph, source_node_id))
    target_size = len(nx.node_connected_component(graph, target_node_id))
    return float(min(source_size, target_size))


def rooted_expected_bfs_steps(
    graph: nx.Graph,
    source_node_id: str,
    target_node_id: str,
) -> float | None:
    """Return expected BFS visits until target with random within-layer order."""
    if source_node_id not in graph or target_node_id not in graph:
        return None
    distances = nx.single_source_shortest_path_length(graph, source_node_id)
    target_distance = distances.get(target_node_id)
    if target_distance is None:
        return None
    earlier_count = sum(
        distance < target_distance for distance in distances.values()
    )
    target_layer_size = sum(
        distance == target_distance for distance in distances.values()
    )
    return float(earlier_count) + (float(target_layer_size) + 1.0) / 2.0


def rooted_expected_dfs_steps(
    graph: nx.Graph,
    source_node_id: str,
    target_node_id: str,
) -> float | None:
    """Return manuscript-style expected DFS visits to one tree target."""
    if source_node_id not in graph or target_node_id not in graph:
        return None
    if not nx.has_path(graph, source_node_id, target_node_id):
        return None

    component_nodes = nx.node_connected_component(graph, source_node_id)
    component = graph.subgraph(component_nodes).copy()
    component.remove_edges_from(nx.selfloop_edges(component))
    if not nx.is_tree(component):
        return None

    children_by_node: dict[str, list[str]] = {
        str(node_id): [] for node_id in component.nodes
    }
    parent_by_node: dict[str, str | None] = {source_node_id: None}
    traversal_order: list[str] = []
    stack = [source_node_id]
    while stack:
        node_id = stack.pop()
        traversal_order.append(node_id)
        for neighbor_id in sorted(component.neighbors(node_id), reverse=True):
            if neighbor_id == parent_by_node[node_id]:
                continue
            parent_by_node[neighbor_id] = node_id
            children_by_node[node_id].append(neighbor_id)
            stack.append(neighbor_id)

    subtree_sizes: dict[str, int] = {}
    for node_id in reversed(traversal_order):
        subtree_sizes[node_id] = 1 + sum(
            subtree_sizes[child_id]
            for child_id in children_by_node[node_id]
        )

    correct_path = nx.shortest_path(
        component,
        source_node_id,
        target_node_id,
    )
    expected_steps = 1.0
    for path_index in range(len(correct_path) - 2, -1, -1):
        node_id = correct_path[path_index]
        correct_child = correct_path[path_index + 1]
        wrong_subtree_size = sum(
            subtree_sizes[child_id]
            for child_id in children_by_node[node_id]
            if child_id != correct_child
        )
        expected_steps = (
            1.0 + 0.5 * float(wrong_subtree_size) + expected_steps
        )
    return expected_steps


def yes_expected_dfs_steps(
    visible_graph: VisibleReachabilityGraph,
    source_node_id: str,
    target_node_id: str,
) -> float | None:
    """Return the cheaper endpoint's manuscript-style expected DFS visits."""
    source_cost = rooted_expected_dfs_steps(
        visible_graph.graph,
        source_node_id,
        target_node_id,
    )
    target_cost = rooted_expected_dfs_steps(
        visible_graph.graph,
        target_node_id,
        source_node_id,
    )
    if source_cost is None or target_cost is None:
        return None
    return min(source_cost, target_cost)


def remove_reference_pair(
    references_by_paper: Mapping[str, Iterable[str]],
    citation_pair: CitationPair,
) -> dict[str, tuple[str, ...]] | None:
    """Return references without one displayed citation, if present."""
    citing_id, cited_id = citation_pair
    references = {
        str(owner): list(deduplicate(values))
        for owner, values in references_by_paper.items()
    }
    if cited_id not in references.get(citing_id, []):
        return None
    references[citing_id].remove(cited_id)
    return normalize_mapping(references)


def remove_authorship_pair(
    authors_by_paper: Mapping[str, Iterable[str]],
    authorship_pair: AuthorshipPair,
) -> dict[str, tuple[str, ...]] | None:
    """Return displayed authors without one paper-author relation."""
    paper_id, author_id = authorship_pair
    authors = {
        str(owner): list(deduplicate(values))
        for owner, values in authors_by_paper.items()
    }
    if author_id not in authors.get(paper_id, []):
        return None
    authors[paper_id].remove(author_id)
    return normalize_mapping(authors)


def build_question_text(
    *,
    source_label: str,
    target_label: str,
    include_authorship: bool = True,
) -> str:
    """Return the standardized reachability question text."""
    if not include_authorship:
        return (
            "Using only the relationships described in the context, can you start "
            f"from the {source_label} and arrive at the {target_label} by following "
            "one or more `cites` or `cited_by` relationships?"
        )
    return (
        "Using only the relationships described in the context, is the "
        f"{target_label} reachable from the {source_label} through one or more "
        "`cites`, `cited_by`, `author_of`, or `authored_by` relationships?"
    )


def serialize_visible_graph(
    visible_graph: VisibleReachabilityGraph,
    *,
    paper_titles_by_paper: Mapping[str, str] | None = None,
    author_names_by_author: Mapping[str, str] | None = None,
    paper_author_ids_by_paper: Mapping[str, Iterable[str]] | None = None,
    allowed_reference_pairs: Iterable[CitationPair] = (),
    allowed_terminal_reference_owners: Mapping[
        str, Iterable[str]
    ] | None = None,
    citation_neighbors_by_paper: Mapping[str, Iterable[str]] | None = None,
    allow_hidden_citations: bool = False,
) -> dict[str, Any]:
    """Return JSON-compatible graph data for auditing and distractor filtering."""
    edges = []
    for left, right, attributes in sorted(
        visible_graph.graph.edges(data=True),
        key=lambda item: (str(item[0]), str(item[1])),
    ):
        edges.append(
            {
                "source_node_id": str(left),
                "target_node_id": str(right),
                "relation_types": sorted(attributes.get("relation_types") or ()),
            }
        )
    payload = {
        "expanded_paper_ids": sorted(visible_graph.expanded_paper_ids),
        "authors_by_paper": {
            key: list(values)
            for key, values in visible_graph.authors_by_paper.items()
        },
        "references_by_paper": {
            key: list(values)
            for key, values in visible_graph.references_by_paper.items()
        },
        "terminal_reference_nodes": {
            key: [list(pair) for pair in value]
            for key, value in sorted(
                visible_graph.terminal_reference_nodes.items()
            )
        },
        "edges": edges,
    }
    if paper_titles_by_paper is not None:
        visible_paper_ids = set(visible_graph.expanded_paper_ids)
        visible_paper_ids.update(visible_graph.terminal_reference_nodes)
        payload["paper_titles_by_paper"] = {
            paper_id: str(paper_titles_by_paper.get(paper_id) or "")
            for paper_id in sorted(visible_paper_ids)
        }
    if author_names_by_author is not None:
        visible_author_ids = {
            str(node_id)
            for node_id, attributes in visible_graph.graph.nodes(data=True)
            if attributes.get("node_type") == "author"
        }
        if paper_author_ids_by_paper is not None:
            visible_author_ids.update(
                str(author_id)
                for author_ids in paper_author_ids_by_paper.values()
                for author_id in author_ids
            )
        payload["author_names_by_author"] = {
            author_id: str(author_names_by_author.get(author_id) or "")
            for author_id in sorted(visible_author_ids)
        }
    if paper_author_ids_by_paper is not None:
        payload["paper_author_ids_by_paper"] = {
            str(paper_id): list(deduplicate(author_ids))
            for paper_id, author_ids in sorted(
                paper_author_ids_by_paper.items(),
                key=lambda item: str(item[0]),
            )
        }
        payload["allowed_reference_pairs"] = [
            list(pair)
            for pair in sorted(
                {
                    (str(owner), str(cited_id))
                    for owner, cited_id in allowed_reference_pairs
                }
            )
        ]
    if allowed_terminal_reference_owners:
        payload["allowed_terminal_reference_owners"] = {
            str(cited_id): sorted(deduplicate(owner_ids))
            for cited_id, owner_ids in sorted(
                allowed_terminal_reference_owners.items(),
                key=lambda item: str(item[0]),
            )
        }
    if citation_neighbors_by_paper is not None:
        payload["citation_neighbors_by_paper"] = {
            str(paper_id): list(deduplicate(neighbor_ids))
            for paper_id, neighbor_ids in sorted(
                citation_neighbors_by_paper.items(),
                key=lambda item: str(item[0]),
            )
        }
    if allow_hidden_citations:
        payload["allow_hidden_citations"] = True
    return payload


def visible_graph_from_metadata(metadata: Mapping[str, Any]) -> VisibleReachabilityGraph:
    """Rebuild a visible graph stored in reachability metadata."""
    expanded = metadata.get("expanded_paper_ids") or []
    authors = metadata.get("authors_by_paper") or {}
    references = metadata.get("references_by_paper") or {}
    if not isinstance(authors, Mapping) or not isinstance(references, Mapping):
        raise ValueError("Invalid reachability visible-graph metadata.")
    return build_visible_graph(
        expanded_paper_ids=expanded,
        authors_by_paper=authors,
        references_by_paper=references,
    )


def visible_identity_registry_from_metadata(
    metadata: Mapping[str, Any],
) -> VisibleIdentityRegistry:
    """Rebuild and validate the model-visible identities stored in metadata."""
    visible = visible_graph_from_metadata(metadata)
    paper_titles = metadata.get("paper_titles_by_paper") or {}
    author_names = metadata.get("author_names_by_author") or {}
    paper_author_ids = metadata.get("paper_author_ids_by_paper")
    allowed_terminal_owners = metadata.get(
        "allowed_terminal_reference_owners"
    )
    citation_neighbors = metadata.get("citation_neighbors_by_paper")
    if not isinstance(paper_titles, Mapping) or not isinstance(author_names, Mapping):
        raise ValueError("Invalid reachability visible-identity metadata.")
    if paper_author_ids is not None and not isinstance(paper_author_ids, Mapping):
        raise ValueError("Invalid per-paper author identity metadata.")
    if allowed_terminal_owners is not None and not isinstance(
        allowed_terminal_owners, Mapping
    ):
        raise ValueError("Invalid terminal-reference owner metadata.")
    if citation_neighbors is not None and not isinstance(
        citation_neighbors, Mapping
    ):
        raise ValueError("Invalid citation-neighbor identity metadata.")
    return build_visible_identity_registry(
        visible,
        paper_titles_by_paper=paper_titles,
        author_names_by_author=author_names,
        citation_owners_by_paper=(
            {}
            if paper_author_ids is not None
            or allowed_terminal_owners is not None
            else None
        ),
        paper_author_ids_by_paper=paper_author_ids,
        allowed_reference_pairs=metadata.get("allowed_reference_pairs") or (),
        allowed_terminal_reference_owners=allowed_terminal_owners,
        citation_neighbors_by_paper=(
            None
            if bool(metadata.get("allow_hidden_citations"))
            else citation_neighbors
        ),
    )


def graph_invariant_holds(
    visible_graph: VisibleReachabilityGraph,
    *,
    source_node_id: str,
    target_node_id: str,
    answer: str,
) -> bool:
    """Return whether one graph still supports its categorical answer."""
    path_count = path_count_capped(
        visible_graph,
        source_node_id,
        target_node_id,
    )
    if str(answer).upper() == ANSWER_YES:
        return path_count == 1
    if str(answer).upper() == ANSWER_NO:
        return path_count == 0
    return False


def add_expanded_paper(
    visible_graph: VisibleReachabilityGraph,
    *,
    paper_node_id: str,
    author_node_ids: Sequence[str],
    cited_node_ids: Sequence[str],
) -> VisibleReachabilityGraph:
    """Return the visible graph after adding one full distractor paper block."""
    expanded = set(visible_graph.expanded_paper_ids)
    expanded.add(paper_node_id)
    authors = dict(visible_graph.authors_by_paper)
    references = dict(visible_graph.references_by_paper)
    authors[paper_node_id] = deduplicate(author_node_ids)
    references[paper_node_id] = deduplicate(cited_node_ids)
    return build_visible_graph(
        expanded_paper_ids=expanded,
        authors_by_paper=authors,
        references_by_paper=references,
    )
