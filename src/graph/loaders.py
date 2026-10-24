from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.setup.constants import (
    EDGE_AUTHORSHIP,
    EDGE_CITING,
    NODE_PAPER,
    author_nid,
    nid_db_id,
    nid_type,
    normalize_text,
    paper_nid,
)
from src.setup.litellm import normalize_litellm_usage

from .json_utils import build_nx_digraph_from_payload
from .network import CitationNetwork, GraphLimits
from .persistence import validate_loaded_graph
from .resolution import resolve_graph_db_paths
from .storage import DatabaseManager


def load_graph(
    graph_source: Path,
    limits: GraphLimits | None = None,
    db_path: Path | None = None,
    *,
    validate_integrity: bool = True,
) -> CitationNetwork:
    """Reconstruct a ``CitationNetwork`` from the DB-backed graph store."""
    resolved_graph_path, resolved_db_path, db_resolution = resolve_graph_db_paths(
        graph_source,
        db_path,
    )
    if not resolved_db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {resolved_db_path}. Pass --db-path explicitly if it has moved."
        )

    db = DatabaseManager(resolved_db_path)
    meta = db.all_graph_metadata()
    meta["db_path"] = str(resolved_db_path)
    meta.setdefault("graph_format", "sqlite")
    if limits is None:
        limits = GraphLimits.from_dict(meta.get("limits", {}))

    network = CitationNetwork(db=db, limits=limits)
    network.litellm_usage = normalize_litellm_usage(meta.get("litellm_usage"))
    max_depth_value = meta.get("max_depth", 0)
    network.max_depth = None if max_depth_value is None else int(max_depth_value)

    for row in db.all_papers():
        db_id = str(row["id"])
        node_id = paper_nid(db_id)
        network.G.add_node(node_id)
        network.index_paper(
            db_id,
            title=str(row["title"] or ""),
            arxiv_id=str(row["arxiv_id"] or ""),
        )
        if row["explored"]:
            network._explored_set.add(db_id)

    for row in db.all_authors():
        db_id = str(row["id"])
        node_id = author_nid(db_id)
        network.G.add_node(node_id)
        name = str(row["name"] or "")
        if name:
            network._author_index[normalize_text(name)] = db_id

    for row in db.all_citations():
        network.G.add_edge(
            paper_nid(str(row["citing_paper_id"])),
            paper_nid(str(row["cited_paper_id"])),
            edge_type=EDGE_CITING,
        )

    for row in db.all_authorships():
        edge_attrs = {"edge_type": EDGE_AUTHORSHIP}
        if row["position"] is not None:
            edge_attrs["position"] = row["position"]
        network.G.add_edge(
            paper_nid(str(row["paper_id"])),
            author_nid(str(row["author_id"])),
            **edge_attrs,
        )

    if resolved_graph_path.suffix.lower() == ".json":
        import orjson

        try:
            with resolved_graph_path.open(encoding="utf-8") as handle:
                payload = orjson.loads(handle.read())
            json_graph = build_nx_digraph_from_payload(payload)
        except (OSError, orjson.JSONDecodeError, KeyError):
            json_graph = None
        if json_graph is not None:
            for node_id, attrs in json_graph.nodes(data=True):
                network.G.add_node(node_id, **attrs)
                if nid_type(str(node_id)) == NODE_PAPER:
                    row = network.db.get_paper(nid_db_id(str(node_id)))
                    if row is not None:
                        network.index_paper(
                            nid_db_id(str(node_id)),
                            title=str(row["title"] or ""),
                            arxiv_id=str(row["arxiv_id"] or ""),
                        )
            for source, target, attrs in json_graph.edges(data=True):
                network.G.add_edge(source, target, **attrs)

    network._n_papers = db.count_papers()
    network._n_authors = db.count_authors()

    logger.info(
        "Loaded graph from {} | DB: {} [{}] | {} nodes ({} papers, {} authors), {} edges | limits: {} | max_depth recorded: {}",
        resolved_graph_path,
        resolved_db_path,
        db_resolution,
        network.G.number_of_nodes(),
        network._n_papers,
        network._n_authors,
        network.G.number_of_edges(),
        network.limits.summary(),
        "unbounded"
        if network.max_depth is None
        else network.max_depth if network.max_depth > 0 else "unknown",
    )

    if validate_integrity:
        issues = validate_loaded_graph(network)
        if issues:
            logger.warning(
                "Load integrity check found {} issue(s); the graph/database may be inconsistent:",
                len(issues),
            )
            for issue in issues:
                logger.warning("  • {}", issue)
        else:
            logger.debug("Load integrity check passed; graph and DB are consistent.")
    else:
        logger.debug("Load integrity check skipped.")

    return network
