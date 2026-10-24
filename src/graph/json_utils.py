from __future__ import annotations

import orjson
from typing import Any, Iterator

import networkx as nx

from src.setup.constants import (
    EDGE_AUTHORSHIP,
    EDGE_CITED_BY,
    EDGE_CITING,
    NODE_AUTHOR,
    NODE_PAPER,
    nid_type,
)


LEGACY_EDGE_CITATION = "citation"


def detect_edges_key(raw: dict) -> str:
    """Return the edge-list key used by a node-link payload."""
    if "edges" in raw:
        return "edges"
    if "links" in raw:
        return "links"
    version = tuple(int(part) for part in nx.__version__.split(".")[:2])
    return "edges" if version >= (3, 0) else "links"


def normalize_saved_edge(
    source: str,
    target: str,
    attrs: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Return the canonical forward-only representation of one saved edge."""
    normalized_attrs = dict(attrs)
    edge_type = normalized_attrs.get("edge_type", "")

    if edge_type in (EDGE_CITING, LEGACY_EDGE_CITATION):
        normalized_attrs["edge_type"] = EDGE_CITING
        return source, target, normalized_attrs

    if edge_type == EDGE_CITED_BY:
        normalized_attrs["edge_type"] = EDGE_CITING
        return target, source, normalized_attrs

    if edge_type == EDGE_AUTHORSHIP:
        source_type = nid_type(source)
        target_type = nid_type(target)
        if source_type == NODE_AUTHOR and target_type == NODE_PAPER:
            return target, source, normalized_attrs
        return source, target, normalized_attrs

    return source, target, normalized_attrs


def canonical_edge_key(
    source: str,
    target: str,
    attrs: dict[str, Any],
) -> tuple[str, str, str]:
    """Return a stable hashable key for one normalized edge."""
    return source, target, orjson.dumps(attrs, option=orjson.OPT_SORT_KEYS, default=str).decode("utf-8")


def iter_normalized_graph_edges(payload: dict) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield canonical forward-only edges from a saved graph payload."""
    graph_data = payload.get("graph", payload)
    edges_key = payload.get("edges_key") or detect_edges_key(graph_data)
    seen: set[tuple[str, str, str]] = set()

    for edge in graph_data.get(edges_key, []):
        source = edge["source"]
        target = edge["target"]
        attrs = {
            key: value
            for key, value in edge.items()
            if key not in ("source", "target", "key")
        }
        normalized_source, normalized_target, normalized_attrs = normalize_saved_edge(
            source,
            target,
            attrs,
        )
        edge_key = canonical_edge_key(
            normalized_source,
            normalized_target,
            normalized_attrs,
        )
        if edge_key in seen:
            continue
        seen.add(edge_key)
        yield normalized_source, normalized_target, normalized_attrs


def build_nx_digraph_from_payload(payload: dict) -> nx.DiGraph:
    """Reconstruct a forward-only ``DiGraph`` from a saved node-link payload."""
    graph_data = payload.get("graph", payload)

    graph = nx.DiGraph()
    for node in graph_data.get("nodes", []):
        node_id = node["id"]
        attrs = {key: value for key, value in node.items() if key != "id"}
        graph.add_node(node_id, **attrs)

    for source, target, attrs in iter_normalized_graph_edges(payload):
        graph.add_edge(source, target, **attrs)

    return graph


def truncate_text(text: str, max_length: int) -> str:
    """Return *text* truncated to *max_length* characters."""
    return text if len(text) <= max_length else text[: max_length - 1] + "…"
