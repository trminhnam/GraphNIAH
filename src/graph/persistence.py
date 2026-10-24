from __future__ import annotations

import orjson
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.setup.constants import (
    EDGE_AUTHORSHIP,
    EDGE_CITING,
    NODE_AUTHOR,
    NODE_PAPER,
    nid_db_id,
    nid_type,
)
from src.setup.litellm import get_litellm_usage, normalize_litellm_usage

from .resolution import resolve_db_path


def collect_root_papers(network: Any) -> list[str]:
    """Return graph paper node IDs whose DB rows are marked depth 0."""
    root_papers: list[str] = []
    for node_id in network.G.nodes():
        if nid_type(node_id) != NODE_PAPER:
            continue
        row = network.db.get_paper(nid_db_id(node_id))
        if row is None:
            continue
        try:
            depth = int(row["depth"])
        except (TypeError, ValueError):
            continue
        if depth == 0:
            root_papers.append(node_id)
    return root_papers


def update_graph_metadata(network: Any, graph_db_path: Path) -> None:
    """Write graph-level metadata into the SQLite store before snapshotting."""
    root_papers = collect_root_papers(network)
    litellm_usage = normalize_litellm_usage(get_litellm_usage())
    network.litellm_usage = litellm_usage

    metadata = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(graph_db_path),
        "n_papers": network._n_papers,
        "n_authors": network._n_authors,
        "limits": network.limits.to_dict(),
        "max_depth": network.max_depth,
        "root_papers": root_papers,
        "root_paper_count": len(root_papers),
        "litellm_usage": litellm_usage,
        "graph_format": "sqlite",
    }
    with network.db.batch():
        for key, value in metadata.items():
            network.db.set_graph_metadata(key, value)


def build_graph_json_payload(
    network: Any,
    *,
    graph_db_path: Path,
) -> dict[str, Any]:
    from networkx.readwrite import json_graph

    root_papers = collect_root_papers(network)
    metadata = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "db_path": graph_db_path.name,
        "n_papers": network._n_papers,
        "n_authors": network._n_authors,
        "limits": network.limits.to_dict(),
        "max_depth": network.max_depth,
        "root_papers": root_papers,
        "root_paper_count": len(root_papers),
        "litellm_usage": normalize_litellm_usage(get_litellm_usage()),
        "graph_format": "sqlite",
    }
    return {
        "_meta": metadata,
        "root_papers": root_papers,
        "edges_key": "edges",
        "graph": json_graph.node_link_data(network.G, edges="edges"),
    }


def write_graph_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2, default=str).decode("utf-8"))
            handle.write("\n")
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def save_graph(
    network: Any,
    output_dir: Path,
    stem: str = "citation_network",
    json_path: Path | None = None,
    write_gexf: bool = True,
    write_json: bool = True,
) -> Path:
    """Persist graph topology into JSON plus a matching SQLite snapshot.

    When ``write_json`` is False, only the SQLite ``.db`` snapshot is written and
    the (expensive) full-graph JSON serialization is skipped. The returned path
    still points at the intended ``.json`` location so callers can derive the
    sibling ``.db`` path; the ``.db`` is always written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_json_out = (
        json_path.with_suffix(".json")
        if json_path is not None and json_path.suffix.lower() == ".db"
        else (json_path if json_path is not None else output_dir / f"{stem}.json")
    )
    graph_db_out = graph_json_out.with_suffix(".db")
    gexf_out = graph_json_out.with_suffix(".gexf")

    update_graph_metadata(network, graph_db_out)
    network.db.snapshot_to(graph_db_out)
    if write_json:
        write_graph_json(
            graph_json_out,
            build_graph_json_payload(network, graph_db_path=graph_db_out),
        )
    json_display = str(graph_json_out) if write_json else "skipped"

    if write_gexf:
        import networkx as nx

        gexf_graph = network.G.copy()
        for _, _, data in gexf_graph.edges(data=True):
            for key, value in list(data.items()):
                if not isinstance(value, (str, int, float, bool, type(None))):
                    data[key] = orjson.dumps(value, default=str).decode("utf-8")
        nx.write_gexf(gexf_graph, str(gexf_out))
        logger.info(
            "Graph checkpoint saved → db: {} | json: {} ({} nodes, {} edges) | gexf: {}",
            graph_db_out,
            json_display,
            network.G.number_of_nodes(),
            network.G.number_of_edges(),
            gexf_out,
        )
    else:
        logger.info(
            "Graph checkpoint saved → db: {} | json: {} ({} nodes, {} edges) | gexf: skipped",
            graph_db_out,
            json_display,
            network.G.number_of_nodes(),
            network.G.number_of_edges(),
        )

    return graph_json_out


def validate_loaded_graph(network: Any) -> list[str]:
    """Cross-check graph topology against the DB after a load."""
    issues: list[str] = []

    paper_node_ids = [node_id for node_id in network.G.nodes() if nid_type(node_id) == NODE_PAPER]
    author_node_ids = [node_id for node_id in network.G.nodes() if nid_type(node_id) == NODE_AUTHOR]
    graph_paper_count = len(paper_node_ids)
    graph_author_count = len(author_node_ids)

    paper_db_ids = [nid_db_id(node_id) for node_id in paper_node_ids]
    author_db_ids = [nid_db_id(node_id) for node_id in author_node_ids]
    existing_paper_ids = network.db.paper_ids_in(paper_db_ids)
    existing_author_ids = network.db.author_ids_in(author_db_ids)
    ghost_papers = [
        node_id
        for node_id, db_id in zip(paper_node_ids, paper_db_ids)
        if db_id not in existing_paper_ids
    ]
    ghost_authors = [
        node_id
        for node_id, db_id in zip(author_node_ids, author_db_ids)
        if db_id not in existing_author_ids
    ]

    for label, ghosts in (("paper", ghost_papers), ("author", ghost_authors)):
        if ghosts:
            sample = ghosts[:5]
            suffix = "..." if len(ghosts) > 5 else ""
            issues.append(
                f"{len(ghosts)} {label} node(s) in memory have no DB row "
                f"(sample: {sample}{suffix})."
            )

    db_papers = network.db.count_papers()
    db_authors = network.db.count_authors()
    if graph_paper_count > db_papers:
        issues.append(
            f"Graph has {graph_paper_count} paper nodes but DB has only {db_papers} paper rows."
        )
    if graph_author_count > db_authors:
        issues.append(
            f"Graph has {graph_author_count} author nodes but DB has only {db_authors} author rows."
        )

    graph_citation_edges = sum(
        1
        for _source, _target, attrs in network.G.edges(data=True)
        if attrs.get("edge_type") == EDGE_CITING
    )
    graph_authorship_edges = sum(
        1
        for _source, _target, attrs in network.G.edges(data=True)
        if attrs.get("edge_type") == EDGE_AUTHORSHIP
    )
    db_citation_edges = len(network.db.all_citations())
    db_authorship_edges = len(network.db.all_authorships())
    if graph_citation_edges != db_citation_edges:
        issues.append(
            f"Graph has {graph_citation_edges} citation edges but DB has {db_citation_edges} citation rows."
        )
    if graph_authorship_edges != db_authorship_edges:
        issues.append(
            f"Graph has {graph_authorship_edges} authorship edges but DB has {db_authorship_edges} authorship rows."
        )

    return issues


def resolve_saved_graph_db_path(
    graph_path: Path,
    meta: dict[str, Any],
    db_path_override: Path | None,
) -> tuple[Path, str]:
    """Resolve the SQLite DB path that should accompany a saved graph input."""
    if graph_path.suffix.lower() == ".db":
        if db_path_override is not None:
            return db_path_override, "explicit"
        return graph_path, "graph_db"
    return resolve_db_path(
        graph_path,
        db_path_override,
        allow_any_checkpoint=True,
        meta_db_path=str(meta.get("db_path", "") or ""),
    )
