from __future__ import annotations

from importlib import import_module

LAZY_EXPORTS = {
    "ARXIV_ID_RE": (".expansion", "ARXIV_ID_RE"),
    "AuthorStub": (".expansion", "AuthorStub"),
    "CitationGraphLoader": (".readers", "CitationGraphLoader"),
    "CitationNetwork": (".network", "CitationNetwork"),
    "DBReader": (".readers", "DBReader"),
    "DatabaseManager": (".storage", "DatabaseManager"),
    "GraphLimits": (".network", "GraphLimits"),
    "PaperExpander": (".expansion", "PaperExpander"),
    "PaperIOBundle": (".expansion", "PaperIOBundle"),
    "apply_fetched_paper": (".expansion", "apply_fetched_paper"),
    "apply_reference_batch": (".reference_merge", "apply_reference_batch"),
    "build_fallback_author_infos": (".expansion", "build_fallback_author_infos"),
    "build_node_payload": (".readers", "build_node_payload"),
    "build_nx_digraph_from_payload": (".json_utils", "build_nx_digraph_from_payload"),
    "checkpoint_timestamp": (".checkpoints", "checkpoint_timestamp"),
    "detect_edges_key": (".json_utils", "detect_edges_key"),
    "enqueue_citation_neighbors": (".expansion", "enqueue_citation_neighbors"),
    "extract_base_stem": (".checkpoints", "extract_base_stem"),
    "fetch_paper_data": (".expansion", "fetch_paper_data"),
    "find_latest_db": (".checkpoints", "find_latest_db"),
    "get_paper_depth": (".expansion", "get_paper_depth"),
    "is_et_al": (".expansion", "is_et_al"),
    "load_graph": (".loaders", "load_graph"),
    "newest_file": (".resolution", "newest_file"),
    "prune_checkpoints": (".checkpoints", "prune_checkpoints"),
    "read_meta_db_path": (".resolution", "read_meta_db_path"),
    "resolve_db_path": (".resolution", "resolve_db_path"),
    "resolve_graph_db_paths": (".resolution", "resolve_graph_db_paths"),
    "resolve_full_author_list": (".expansion", "resolve_full_author_list"),
    "resolve_graph_argument": (".resolution", "resolve_graph_argument"),
    "resolve_latest_graph_db_paths": (".resolution", "resolve_latest_graph_db_paths"),
    "resolve_saved_graph_db_path": (".persistence", "resolve_saved_graph_db_path"),
    "save_graph": (".persistence", "save_graph"),
    "truncate_text": (".json_utils", "truncate_text"),
    "validate_loaded_graph": (".persistence", "validate_loaded_graph"),
}

__all__ = list(LAZY_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
