"""Helpers for single-source scholarly-graph reachability counting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import orjson
from typing import Any, Mapping, Sequence

import networkx as nx

from src.questions.reachability_yesno import (
    VisibleReachabilityGraph,
    normalize_visible_identity,
)
from src.setup.constants import NODE_AUTHOR, NODE_PAPER, author_nid, nid_db_id, nid_type


QUESTION_TYPE = "reachability-counting"
ANSWER_TYPE = "REACHABILITY_COUNTING"
COMPLEXITY_MEASURE = "bidirectional_source_component_size"
PAPER_IDENTITY_RULE = "normalized-title-and-order-insensitive-author-set"
PAPER_IDENTITY_PREFIX = "paper-identity:"


@dataclass(frozen=True)
class CountingReachabilityGraph:
    """Identity-contracted graph used to calculate one counting answer."""

    graph: nx.Graph
    source_node_id: str
    original_to_counting_node: dict[str, str]
    paper_identity_groups: dict[str, tuple[str, ...]]
    paper_identity_values: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ReachabilityCount:
    """Source-component size and source-excluding entity subtotals."""

    complexity: int
    answer: int
    reachable_papers: int
    reachable_authors: int
    component_node_ids: tuple[str, ...]


def normalize_author_list(author_names: Sequence[str]) -> tuple[str, ...]:
    """Return a deduplicated, order-insensitive visible author identity."""
    return tuple(
        sorted(
            {
                key
                for name in author_names
                if (key := normalize_visible_identity(name))
            }
        )
    )


def paper_identity_value(
    title: str,
    author_names: Sequence[str],
) -> tuple[str, tuple[str, ...]] | None:
    """Return one normalized title/author identity, or ``None`` if untitled."""
    title_key = normalize_visible_identity(title)
    if not title_key:
        return None
    return title_key, normalize_author_list(author_names)


def paper_identity_node_id(
    value: tuple[str, tuple[str, ...]],
) -> str:
    """Return a stable graph node ID for one visible paper identity."""
    encoded = orjson.dumps([value[0], list(value[1])])
    return f"{PAPER_IDENTITY_PREFIX}{hashlib.sha256(encoded).hexdigest()}"


def paper_nodes_have_safe_missing_authors(
    values_by_paper: Mapping[str, tuple[str, tuple[str, ...]]],
) -> bool:
    """Reject equal-title multi-paper groups when any author list is missing."""
    papers_by_title: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    for paper_id, (title_key, author_keys) in values_by_paper.items():
        papers_by_title.setdefault(title_key, []).append((paper_id, author_keys))
    return all(
        len(rows) == 1 or all(author_keys for _paper_id, author_keys in rows)
        for rows in papers_by_title.values()
    )


def build_counting_graph(
    visible_graph: VisibleReachabilityGraph,
    *,
    source_node_id: str,
    paper_titles_by_paper: Mapping[str, str],
    paper_authors_by_paper: Mapping[str, Sequence[str]],
    author_names_by_author: Mapping[str, str],
) -> CountingReachabilityGraph | None:
    """Contract equivalent papers and return the model-visible counting graph.

    Papers are equivalent when their normalized titles and deduplicated,
    order-insensitive author-name sets match. Distinct visible author nodes must
    retain distinct normalized names; ambiguous author homonyms are rejected.
    """
    raw_graph = visible_graph.graph
    if source_node_id not in raw_graph:
        return None

    paper_values: dict[str, tuple[str, tuple[str, ...]]] = {}
    for node_id, attributes in raw_graph.nodes(data=True):
        if attributes.get("node_type") != NODE_PAPER:
            continue
        identity = paper_identity_value(
            str(paper_titles_by_paper.get(str(node_id)) or ""),
            paper_authors_by_paper.get(str(node_id), ()),
        )
        if identity is None:
            return None
        paper_values[str(node_id)] = identity
    if not paper_nodes_have_safe_missing_authors(paper_values):
        return None

    author_keys: dict[str, str] = {}
    author_nodes_by_key: dict[str, list[str]] = {}
    for node_id, attributes in raw_graph.nodes(data=True):
        if attributes.get("node_type") != NODE_AUTHOR:
            continue
        key = normalize_visible_identity(author_names_by_author.get(str(node_id)))
        if not key:
            return None
        author_keys[str(node_id)] = key
        author_nodes_by_key.setdefault(key, []).append(str(node_id))
    if any(len(set(node_ids)) > 1 for node_ids in author_nodes_by_key.values()):
        return None

    graph = nx.Graph()
    original_to_counting_node: dict[str, str] = {}
    paper_groups: dict[str, list[str]] = {}
    paper_identity_values: dict[str, dict[str, Any]] = {}
    for node_id, attributes in raw_graph.nodes(data=True):
        raw_node_id = str(node_id)
        node_type = str(attributes.get("node_type") or nid_type(raw_node_id))
        if node_type == NODE_PAPER:
            identity_value = paper_values[raw_node_id]
            counting_node_id = paper_identity_node_id(identity_value)
            paper_groups.setdefault(counting_node_id, []).append(raw_node_id)
            paper_identity_values.setdefault(
                counting_node_id,
                {
                    "normalized_title": identity_value[0],
                    "normalized_authors": list(identity_value[1]),
                },
            )
        else:
            counting_node_id = raw_node_id
        original_to_counting_node[raw_node_id] = counting_node_id
        graph.add_node(counting_node_id, node_type=node_type)

    for left, right, attributes in raw_graph.edges(data=True):
        counting_left = original_to_counting_node[str(left)]
        counting_right = original_to_counting_node[str(right)]
        if counting_left == counting_right:
            continue
        relation_types = set(attributes.get("relation_types") or ())
        if graph.has_edge(counting_left, counting_right):
            graph[counting_left][counting_right].setdefault(
                "relation_types", set()
            ).update(relation_types)
        else:
            graph.add_edge(
                counting_left,
                counting_right,
                relation_types=relation_types,
            )

    counting_source = original_to_counting_node[source_node_id]
    return CountingReachabilityGraph(
        graph=graph,
        source_node_id=counting_source,
        original_to_counting_node=original_to_counting_node,
        paper_identity_groups={
            node_id: tuple(sorted(group))
            for node_id, group in sorted(paper_groups.items())
        },
        paper_identity_values=paper_identity_values,
    )


def calculate_reachability_count(
    counting_graph: CountingReachabilityGraph,
) -> ReachabilityCount | None:
    """Return the source component size and source-excluding answer."""
    graph = counting_graph.graph
    source_node_id = counting_graph.source_node_id
    if source_node_id not in graph:
        return None
    component = set(nx.node_connected_component(graph, source_node_id))
    reachable = component.difference({source_node_id})
    paper_count = sum(
        graph.nodes[node_id].get("node_type") == NODE_PAPER
        for node_id in reachable
    )
    author_count = sum(
        graph.nodes[node_id].get("node_type") == NODE_AUTHOR
        for node_id in reachable
    )
    return ReachabilityCount(
        complexity=len(component),
        answer=len(reachable),
        reachable_papers=paper_count,
        reachable_authors=author_count,
        component_node_ids=tuple(sorted(component)),
    )


def graph_invariant_holds(
    counting_graph: CountingReachabilityGraph,
    *,
    expected_answer: int,
) -> bool:
    """Return whether the graph still supports the stored numeric answer."""
    result = calculate_reachability_count(counting_graph)
    return result is not None and result.answer == int(expected_answer)


def build_question_text(
    *,
    source_label: str,
    source_type: str,
    include_authorship: bool,
) -> str:
    """Return one explicit source-excluding reachability-counting question."""
    source_kind = "author" if source_type == NODE_AUTHOR else "paper"
    if include_authorship:
        return (
            "Using only the relationships described in the context, how many distinct "
            f"papers and authors in total are reachable from the {source_label} "
            "through one or more `cites`, `cited_by`, `author_of`, or `authored_by` "
            "relationships? Do not "
            f"count the starting {source_kind}."
        )
    return (
        "Using only the relationships described in the context, how many "
        f"distinct papers are reachable from {source_label} by following one or "
        "more `cites` or `cited_by` relationships? Count each reachable paper once, "
        "whether it has its own paper block or appears only in another paper’s "
        "reference list. Do not count the starting paper."
    )


def serialize_counting_graph(
    counting_graph: CountingReachabilityGraph,
) -> dict[str, Any]:
    """Return JSON-compatible identity contraction and graph audit data."""
    return {
        "source_counting_node_id": counting_graph.source_node_id,
        "original_to_counting_node": dict(
            sorted(counting_graph.original_to_counting_node.items())
        ),
        "paper_identity_groups": {
            node_id: list(group)
            for node_id, group in counting_graph.paper_identity_groups.items()
        },
        "paper_identity_values": dict(
            sorted(counting_graph.paper_identity_values.items())
        ),
        "nodes": [
            {
                "node_id": str(node_id),
                "node_type": str(attributes.get("node_type") or ""),
            }
            for node_id, attributes in sorted(
                counting_graph.graph.nodes(data=True),
                key=lambda item: str(item[0]),
            )
        ],
        "edges": [
            {
                "source_node_id": str(left),
                "target_node_id": str(right),
                "relation_types": sorted(attributes.get("relation_types") or ()),
            }
            for left, right, attributes in sorted(
                counting_graph.graph.edges(data=True),
                key=lambda item: (str(item[0]), str(item[1])),
            )
        ],
    }


def collect_paper_authors(
    paper_node_id: str,
    db: Any,
) -> tuple[list[str], list[str]]:
    """Return full author node IDs and names for paper identity/rendering."""
    author_node_ids: list[str] = []
    author_names: list[str] = []
    for author in db.get_expanded_authors_for_paper(nid_db_id(paper_node_id)):
        raw_author_id = str(author.get("author_id") or "").strip()
        if not raw_author_id:
            continue
        node_id = (
            raw_author_id
            if raw_author_id.startswith(f"{NODE_AUTHOR}:")
            else author_nid(raw_author_id)
        )
        name = str(author.get("name") or "").strip()
        if node_id in author_node_ids:
            index = author_node_ids.index(node_id)
            if name and not author_names[index]:
                author_names[index] = name
            continue
        author_node_ids.append(node_id)
        author_names.append(name)
    return author_node_ids, author_names


def graph_papers_have_rendered_mentions(
    visible: Any,
    paper_block_ids: Sequence[str],
    references_by_paper: dict[str, list[str]],
) -> bool:
    """Return whether every graph paper has a block or reference-list mention."""
    rendered_paper_ids = set(paper_block_ids)
    rendered_paper_ids.update(
        cited_id
        for cited_ids in references_by_paper.values()
        for cited_id in cited_ids
    )
    graph_paper_ids = {
        str(node_id)
        for node_id, attributes in visible.graph.nodes(data=True)
        if attributes.get("node_type") == NODE_PAPER
    }
    return graph_paper_ids == rendered_paper_ids
