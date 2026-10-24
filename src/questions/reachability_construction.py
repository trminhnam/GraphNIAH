"""Shared construction helpers for scholarly-graph reachability questions.

Used by the YES/NO, multiple-choice, and counting constructors: checker and
graph opening, needle rendering, walk-data merging, label allocation, needle
records, bucket specifications, and per-bucket output validation and writing.
"""

from __future__ import annotations

import glob
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from create_needles import (
    build_apa_block,
    build_apa_reference_bullet,
    build_exact_jump_needle,
    build_general_reference_bullet,
    build_metadata_needle,
    format_authors_text,
    get_cached_paper,
    get_paper_author_names,
    get_paper_title_text,
)
from src.graph.readers import (
    CitationGraphLoader,
    DBReader,
    SQLiteGraphView,
)
from src.prompts.reachability_yesno_qa import (
    is_paper_only_traversal_mode,
)
from src.questions import construction_run
from src.questions.ambiguity import (
    CachingAmbiguityChecker,
    MissingNodeTolerantAmbiguityChecker,
)
from src.questions.explored import (
    allowed_explored_paper_ids,
    configure_only_explored,
    validate_explored_records,
)
from src.questions.reachability_crossed_trees import (
    NO_CONSTRUCTION,
    CachedSearchTree,
    path_tree_cache_key,
    wrong_branch_step_allowed,
)
from src.questions.reachability_long_path_profiles import (
    LONG_PATH_PROFILE_POLICY,
)
from src.questions.reachability_yesno import (
    ANSWER_NO,
    ANSWER_TYPE,
    ANSWER_YES,
    QUESTION_TYPE,
    build_question_text,
    build_visible_identity_registry,
    normalize_visible_identity,
    rooted_expected_dfs_steps,
    serialize_visible_graph,
)
from src.questions.rerender import (
    rerender_needle_records,
    rerender_questions_enabled,
)
from src.questions.shared import (
    NEEDLE_REFERENCE_STATISTICS_FIELD,
    calculate_needle_reference_statistics,
)
from src.questions.text import quoted_title
from src.setup.constants import (
    EDGE_AUTHORSHIP,
    EDGE_CITED_BY,
    EDGE_CITING,
    NODE_AUTHOR,
    NODE_PAPER,
    author_nid,
    nid_db_id,
    nid_type,
)
from src.utils.resume import load_json_output, write_json_output

# Values of the original YES/NO constructor. Log lines use its name, and the
# resume error message falls back to its format version.
CLI_NAME = "construct_reachability_yesno_questions_needles"
QUESTION_FORMAT_VERSION = 22
PAIR_ALLOCATION_STRATEGY = (
    "streaming-ascending-lower-bound-indexed-exact-components"
)
DEFAULT_NEEDLE_TYPE = "apa-describe"
SUPPORTED_NEEDLE_TYPES = ("apa-describe", "exact-describe", "general-describe")
DEFAULT_MAX_AUTHOR_ANSWER_INDEX = 10
IDENTITY_POLICY_VERSION = 3
DEFAULT_ANSWER_LABEL = "mixed"
DEFAULT_YES_RATIO = 0.5
TREE_ROLE_POSITIVE = "positive"
TREE_ROLE_SOURCE_NEGATIVE = "source-negative"
TREE_ROLE_TARGET_NEGATIVE = "target-negative"


@dataclass
class ReachabilityCheckerContext:
    """Read-only graph/db views used for one construction run."""

    checker: Any
    graph: Any
    db: Any
    paper_author_name_keys: dict[str, frozenset[str]]


@dataclass(frozen=True, slots=True)
class TreeBucketSelection:
    """Compact result of testing one tree against one complexity bucket."""

    yes_complexity: float
    component_size: int
    removed_reference_count: int
    removed_reference_pairs: tuple[tuple[str, str], ...] = ()


def open_lazy_caching_checker(
    graph_path: Path,
    db_path: Path,
    *,
    include_authorship_edges: bool = True,
    cache_max_entries: int | None = None,
    sqlite_cache_size_kib: int = 1048576,
    sqlite_mmap_size_bytes: int = 2147418112,
) -> ReachabilityCheckerContext:
    """Open memoized read-only graph/db views plus an ambiguity checker.

    Deliberately skips ``warm_ambiguity_checker_caches``. That helper bulk-loads
    every paper title/venue/published and author name into dicts so descriptor
    matching never hits SQLite, but reachability names an explicit source and
    target and has no descriptors. The only consumer of ``checker`` here is
    ``collect_path_edge_validation_errors``, which reads ``graph.nodes``,
    ``graph.adj``, and ``db.get_paper``/``get_author`` -- never those caches.
    Warming them is a full scan of both tables for data nothing reads.
    """
    shared_sqlite_graph = (
        graph_path.suffix.lower() == ".db"
        and graph_path.resolve() == db_path.resolve()
    )
    # Merged path files already enforce UUID node IDs. Avoid rescanning every
    # paper and author solely to open the same canonical DB as a lazy graph.
    db_reader = DBReader(
        db_path,
        require_uuid_ids=not shared_sqlite_graph,
        cache_max_entries=cache_max_entries,
        sqlite_cache_size_kib=sqlite_cache_size_kib,
        sqlite_mmap_size_bytes=sqlite_mmap_size_bytes,
    )
    graph_loader = (
        SQLiteGraphView(
            db_reader,
            include_authorship_edges=include_authorship_edges,
            cache_max_entries=cache_max_entries,
        )
        if shared_sqlite_graph
        else CitationGraphLoader(graph_path, require_uuid_ids=True)
    )
    checker = CachingAmbiguityChecker(graph_loader, db_reader)
    return ReachabilityCheckerContext(
        checker=MissingNodeTolerantAmbiguityChecker(checker),
        graph=graph_loader,
        db=db_reader,
        paper_author_name_keys={},
    )


def target_author_index_allowed(
    nodes: Sequence[dict[str, Any]],
    edges: Sequence[dict[str, Any] | None],
    max_author_answer_index: int | None,
) -> bool:
    """Return whether an author target sits within the allowed author position."""
    if max_author_answer_index is None:
        return True
    if not nodes or not edges:
        return True
    final_edge = edges[-1]
    if not isinstance(final_edge, dict):
        return True
    if str(final_edge.get("edge_type") or "") != EDGE_AUTHORSHIP:
        return True
    position = final_edge.get("position")
    if position is None:
        return True
    try:
        return int(position) <= int(max_author_answer_index)
    except (TypeError, ValueError):
        return True


def invert_reachability_edge(edge: dict[str, Any]) -> dict[str, Any]:
    """Return one traversal edge oriented in the reverse walk direction."""
    reversed_edge = dict(edge)
    reversed_edge["from"] = edge.get("to")
    reversed_edge["to"] = edge.get("from")
    edge_type = str(edge.get("edge_type") or "")
    if edge_type == EDGE_CITING:
        reversed_edge["edge_type"] = EDGE_CITED_BY
    elif edge_type == EDGE_CITED_BY:
        reversed_edge["edge_type"] = EDGE_CITING
    return reversed_edge


def reachability_path_orientations(
    nodes: Sequence[dict[str, Any]],
    edges: Sequence[dict[str, Any] | None],
) -> tuple[
    tuple[list[dict[str, Any]], list[dict[str, Any] | None]],
    ...,
]:
    """Return the forward orientation and its reversed twin.

    Undirected reachability means either endpoint may be the source, so a path
    yields two questions with the citation direction flipped on the reverse.
    """
    forward = (list(nodes), list(edges))
    node_count = len(nodes)
    reversed_nodes = list(reversed(nodes))
    reversed_edges: list[dict[str, Any] | None] = [None]
    for position in range(1, node_count):
        original = edges[node_count - position]
        reversed_edges.append(
            invert_reachability_edge(original)
            if isinstance(original, dict)
            else None
        )
    return forward, (reversed_nodes, reversed_edges)


def traversal_graph_neighbors(
    node_id: str,
    *,
    traversal_mode: str,
    graph: Any,
    neighbor_cache: dict[tuple[str, str], list[tuple[str, str]]] | None = None,
) -> list[tuple[str, str]]:
    """Return stable unique neighbors allowed by the traversal family."""
    cache_key = (node_id, traversal_mode)
    if neighbor_cache is not None and cache_key in neighbor_cache:
        return neighbor_cache[cache_key]
    neighbors = sorted(
        {
            (
                str(neighbor_id),
                str(attributes.get("edge_type") or ""),
            )
            for neighbor_id, attributes in graph.adj.get(node_id, [])
            if wrong_branch_step_allowed(
                node_id,
                str(neighbor_id),
                str(attributes.get("edge_type") or ""),
                traversal_mode,
            )
        }
    )
    if neighbor_cache is not None:
        neighbor_cache[cache_key] = neighbors
    return neighbors


def provided_wrong_branch_candidate(
    profile: dict[str, Any],
    *,
    endpoint: str,
    oriented_nodes: Sequence[dict[str, Any]],
    orientation_index: int,
    traversal_mode: str,
    graph: Any,
    neighbor_cache: dict[tuple[str, str], list[tuple[str, str]]] | None = None,
) -> list[tuple[list[str], list[str]]]:
    """Validate and return one endpoint branch sliced from the input path.

    Extra citations back into the visible graph are sanitized after the
    protected gold and wrong-branch paths are assembled. Reject here only
    when the supplied main branch itself is malformed or absent from the
    canonical graph.
    """
    if profile.get("policy") != LONG_PATH_PROFILE_POLICY:
        return []
    branch_name = endpoint
    if orientation_index == 1:
        branch_name = "target" if endpoint == "source" else "source"
    raw_branch = profile.get(f"{branch_name}_wrong_branch")
    if not isinstance(raw_branch, dict):
        return []
    node_ids = [str(value) for value in raw_branch.get("node_ids") or []]
    edge_types = [str(value) for value in raw_branch.get("edge_types") or []]
    expected_root = str(
        oriented_nodes[0 if endpoint == "source" else -1]["node_id"]
    )
    gold_node_ids = {
        str(node.get("node_id") or "") for node in oriented_nodes
    }
    if (
        len(node_ids) != len(oriented_nodes)
        or len(edge_types) != len(node_ids) - 1
        or not node_ids
        or node_ids[0] != expected_root
        or len(set(node_ids)) != len(node_ids)
        or set(node_ids[1:]).intersection(gold_node_ids)
    ):
        return []

    for parent_id, child_id, edge_type in zip(
        node_ids, node_ids[1:], edge_types
    ):
        neighbors = traversal_graph_neighbors(
            parent_id,
            traversal_mode=traversal_mode,
            graph=graph,
            neighbor_cache=neighbor_cache,
        )
        if (child_id, edge_type) not in neighbors:
            return []
    return [(node_ids, edge_types)]


def filtered_author_names(
    paper_node_id: str,
    allowed_author_node_ids: Sequence[str],
    db: Any,
    authors_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> list[str]:
    """Return DB author names restricted to one paper's displayed allowlist."""
    allowed = set(allowed_author_node_ids)
    if not allowed:
        return []
    names: list[str] = []
    if authors_cache is not None:
        if paper_node_id not in authors_cache:
            authors_cache[paper_node_id] = db.get_expanded_authors_for_paper(
                nid_db_id(paper_node_id)
            )
        author_rows = authors_cache[paper_node_id]
    else:
        author_rows = db.get_expanded_authors_for_paper(nid_db_id(paper_node_id))
    for author in author_rows:
        author_node_id = author_nid(str(author.get("author_id") or ""))
        if author_node_id in allowed:
            names.append(str(author.get("name") or ""))
    return names


def ordered_reference_node_ids(
    paper_node_id: str,
    cited_node_ids: Sequence[str],
    *,
    random_seed: int | None,
    path_key: Any,
) -> list[str]:
    """Return one deterministic per-paper ordering of allowed references."""
    reference_node_ids = list(cited_node_ids)
    reference_rng = random.Random(
        construction_run.stable_path_seed(
            random_seed,
            ("reference-order", path_key, paper_node_id),
        )
    )
    reference_rng.shuffle(reference_node_ids)
    return reference_node_ids


def render_apa_needle(
    paper_node_id: str,
    allowed_author_node_ids: Sequence[str],
    cited_node_ids: Sequence[str],
    db: Any,
    paper_cache: dict[str, dict[str, Any] | None],
    authors_cache: dict[str, list[dict[str, Any]]],
    *,
    random_seed: int | None,
    path_key: Any,
    display_all_database_authors: bool = False,
) -> str:
    """Return one APA needle with graph-safe references and author metadata."""
    paper = get_cached_paper(paper_node_id, db, paper_cache) or {}
    author_names = (
        get_paper_author_names(paper_node_id, db, authors_cache)
        if display_all_database_authors
        else filtered_author_names(
            paper_node_id,
            allowed_author_node_ids,
            db,
            authors_cache,
        )
    )
    authors_text = format_authors_text(author_names)
    reference_node_ids = ordered_reference_node_ids(
        paper_node_id,
        cited_node_ids,
        random_seed=random_seed,
        path_key=path_key,
    )
    reference_bullets = [
        build_apa_reference_bullet(cited_node_id, db, paper_cache, authors_cache)
        for cited_node_id in reference_node_ids
    ]
    return build_apa_block(
        get_paper_title_text(paper_node_id, db, paper_cache),
        authors_text,
        str(paper.get("published") or ""),
        str(paper.get("venue") or ""),
        str(paper.get("arxiv_id") or ""),
        reference_bullets,
    )


def render_general_needle(
    paper_node_id: str,
    cited_node_ids: Sequence[str],
    db: Any,
    paper_cache: dict[str, dict[str, Any] | None],
    authors_cache: dict[str, list[dict[str, Any]]],
    *,
    random_seed: int | None,
    path_key: Any,
) -> str:
    """Return one general-description paper block from allowed references."""
    metadata = build_metadata_needle(
        paper_node_id,
        db,
        paper_cache,
        authors_cache,
        include_venue=True,
    )
    reference_node_ids = ordered_reference_node_ids(
        paper_node_id,
        cited_node_ids,
        random_seed=random_seed,
        path_key=path_key,
    )
    bullets = [
        build_general_reference_bullet(
            cited_node_id,
            db,
            paper_cache,
            authors_cache,
            include_venue=True,
        )
        for cited_node_id in reference_node_ids
    ]
    title = quoted_title(get_paper_title_text(paper_node_id, db, paper_cache))
    reference_list = f"Paper {title} references the following papers:"
    if bullets:
        reference_list = f"{reference_list}\n" + "\n".join(bullets)
    return f"{metadata}\n{reference_list}"


def render_needles(
    *,
    paper_node_ids: Sequence[str],
    authors_by_paper: dict[str, Sequence[str]],
    references_by_paper: dict[str, Sequence[str]],
    db: Any,
    needle_type: str = DEFAULT_NEEDLE_TYPE,
    random_seed: int | None = None,
    path_key: Any = None,
    display_all_database_authors: bool = False,
) -> list[str]:
    """Render paper needles without coupling metadata to graph authors."""
    if needle_type in {"apa-describe", "general-describe"}:
        invalid_node_ids = [
            node_id
            for node_id in paper_node_ids
            if nid_type(str(node_id)) != NODE_PAPER
        ]
        if invalid_node_ids:
            raise ValueError(
                "APA/general reachability needles must contain paper nodes only: "
                + ", ".join(str(node_id) for node_id in invalid_node_ids)
            )

    paper_cache: dict[str, dict[str, Any] | None] = {}
    authors_cache: dict[str, list[dict[str, Any]]] = {}
    rendered_paper_ids = list(
        dict.fromkeys(
            [
                *(str(node_id) for node_id in paper_node_ids),
                *(
                    str(cited_id)
                    for cited_ids in references_by_paper.values()
                    for cited_id in cited_ids
                ),
            ]
        )
    )
    rendered_db_ids = [nid_db_id(node_id) for node_id in rendered_paper_ids]
    paper_batch_loader = getattr(db, "get_papers", None)
    if callable(paper_batch_loader):
        papers_by_db_id = paper_batch_loader(rendered_db_ids)
        paper_cache.update(
            {
                node_id: papers_by_db_id.get(db_id)
                for node_id, db_id in zip(rendered_paper_ids, rendered_db_ids)
            }
        )
    author_batch_loader = getattr(db, "get_expanded_authors_for_papers", None)
    if callable(author_batch_loader):
        authors_by_db_id = author_batch_loader(rendered_db_ids)
        authors_cache.update(
            {
                node_id: list(authors_by_db_id.get(db_id, []))
                for node_id, db_id in zip(rendered_paper_ids, rendered_db_ids)
            }
        )
    needles: list[str] = []
    for paper_node_id in paper_node_ids:
        allowed_authors = tuple(authors_by_paper.get(paper_node_id, ()))
        cited_node_ids = tuple(references_by_paper.get(paper_node_id, ()))
        if needle_type == "apa-describe":
            needles.append(
                render_apa_needle(
                    paper_node_id,
                    allowed_authors,
                    cited_node_ids,
                    db,
                    paper_cache,
                    authors_cache,
                    random_seed=random_seed,
                    path_key=path_key,
                    display_all_database_authors=display_all_database_authors,
                )
            )
            continue
        if needle_type == "general-describe":
            needles.append(
                render_general_needle(
                    paper_node_id,
                    cited_node_ids,
                    db,
                    paper_cache,
                    authors_cache,
                    random_seed=random_seed,
                    path_key=path_key,
                )
            )
            continue
        metadata_needle = build_metadata_needle(
            paper_node_id,
            db,
            paper_cache,
            authors_cache,
            include_venue=True,
        )
        needles.append(metadata_needle)
        for cited_node_id in cited_node_ids:
            jump_needle = build_exact_jump_needle(
                paper_node_id,
                cited_node_id,
                EDGE_CITING,
                db,
                paper_cache,
                authors_cache,
            )
            if jump_needle:
                needles.append(jump_needle)
    return needles


def paper_node_ids_on_path(nodes: Sequence[dict[str, Any]]) -> list[str]:
    """Return the paper node ids on one prepared path in order."""
    return [
        str(node["node_id"])
        for node in nodes
        if nid_type(str(node["node_id"])) == NODE_PAPER
    ]


def node_label(node_id: str, db: Any, paper_cache: dict[str, dict[str, Any] | None]) -> str:
    """Return a human-facing label for one source/target node."""
    if nid_type(node_id) == NODE_AUTHOR:
        author = db.get_author(nid_db_id(node_id)) if hasattr(db, "get_author") else None
        name = str((author or {}).get("name") or node_id)
        return f"Author {name}"
    title = get_paper_title_text(node_id, db, paper_cache)
    return f'Paper "{title}"'


def paper_visible_identity(
    paper_id: str,
    context: ReachabilityCheckerContext,
    paper_cache: dict[str, dict[str, Any] | None],
    *,
    include_author_identity: bool = True,
) -> tuple[str, frozenset[str], frozenset[str]] | None:
    """Return the title, author-id, and author-name keys visible for a paper."""
    if paper_id not in allowed_explored_paper_ids(context.db, (paper_id,)):
        return None
    title_loader = getattr(context.db, "get_paper_title", None)
    if callable(title_loader):
        title = title_loader(nid_db_id(paper_id))
    else:
        title = get_paper_title_text(paper_id, context.db, paper_cache)
    title_key = normalize_visible_identity(title)
    if not title_key:
        return None
    if not include_author_identity:
        return title_key, frozenset(), frozenset()
    author_ids: set[str] = set()
    author_name_keys: set[str] = set()
    for author in context.db.get_expanded_authors_for_paper(nid_db_id(paper_id)):
        author_id = str(author.get("author_id") or "").strip()
        author_name_key = normalize_visible_identity(author.get("name"))
        if not author_id or not author_name_key:
            return None
        author_ids.add(
            author_id if author_id.startswith("author:") else author_nid(author_id)
        )
        author_name_keys.add(author_name_key)
    return title_key, frozenset(author_ids), frozenset(author_name_keys)


def paper_titles_for_visible(
    paper_ids: Sequence[str],
    references_by_paper: dict[str, Sequence[str]],
    db: Any,
    paper_cache: dict[str, dict[str, Any] | None],
) -> dict[str, str]:
    """Return titles for every paper node that can appear in the visible graph."""
    titles: dict[str, str] = {}
    visible_paper_ids = set(paper_ids)
    for cited in references_by_paper.values():
        visible_paper_ids.update(cited)
    for paper_id in visible_paper_ids:
        titles[paper_id] = get_paper_title_text(paper_id, db, paper_cache)
    return titles


def author_names_for_visible(
    authors_by_paper: dict[str, Sequence[str]],
    db: Any,
) -> dict[str, str]:
    """Return names for every displayed author node."""
    names: dict[str, str] = {}
    for paper_id, author_node_ids in authors_by_paper.items():
        db_authors = {
            author_nid(str(author.get("author_id") or "")): str(author.get("name") or "")
            for author in db.get_expanded_authors_for_paper(nid_db_id(paper_id))
        }
        for author_node_id in author_node_ids:
            names.setdefault(author_node_id, db_authors.get(author_node_id, ""))
    return names


def combine_reachability_walk_data(
    path_payloads: Sequence[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    """Merge path files, deduplicating content rather than file-local path IDs."""
    if not path_payloads:
        raise ValueError("At least one paths file is required.")
    candidates: list[tuple[Path, int, dict[str, Any], str, str]] = []
    seen_content_keys: set[str] = set()
    for source_path, payload in path_payloads:
        for source_index, raw_entry in enumerate(payload.get("paths") or []):
            if not isinstance(raw_entry, dict):
                continue
            entry = dict(raw_entry)
            content_key = path_tree_cache_key(entry)
            if content_key in seen_content_keys:
                continue
            seen_content_keys.add(content_key)
            original_path_id = str(
                entry.get("path_id") or entry.get("path_index") or source_index
            )
            candidates.append(
                (source_path, source_index, entry, content_key, original_path_id)
            )

    id_counts: dict[str, int] = {}
    for _source, _index, _entry, _key, original_path_id in candidates:
        id_counts[original_path_id] = id_counts.get(original_path_id, 0) + 1

    first_payload = path_payloads[0][1]
    combined = dict(first_payload)
    first_metadata = first_payload.get("metadata")
    combined["metadata"] = {
        **(first_metadata if isinstance(first_metadata, dict) else {}),
        "source_paths": construction_run.source_paths_value(
            [source_path for source_path, _payload in path_payloads]
        ),
    }
    combined_paths: list[dict[str, Any]] = []
    for source_path, source_index, entry, content_key, original_path_id in candidates:
        entry["original_path_id"] = original_path_id
        entry["source_path"] = str(source_path)
        entry["source_path_index"] = entry.get("path_index", source_index)
        entry["path_index"] = len(combined_paths)
        entry["path_id"] = (
            original_path_id
            if id_counts[original_path_id] == 1
            else f"{original_path_id}:{content_key[:12]}"
        )
        combined_paths.append(entry)
    combined["paths"] = combined_paths
    return combined


def wrong_branches_are_valid(
    *,
    visible: Any,
    tree_role: str,
    source_node_id: str,
    target_node_id: str,
    witness: Sequence[str],
    gold_depth: int,
    wrong_branch_paths: Sequence[Sequence[str]],
) -> bool:
    """Return whether a final tree retains its required exact-depth branches."""
    expected_roots = {
        TREE_ROLE_POSITIVE: [source_node_id, target_node_id],
        TREE_ROLE_SOURCE_NEGATIVE: [source_node_id],
        TREE_ROLE_TARGET_NEGATIVE: [target_node_id],
    }.get(tree_role)
    if expected_roots is None or len(wrong_branch_paths) != len(expected_roots):
        return False
    gold_nodes = set(str(node_id) for node_id in witness)
    for expected_root, path in zip(expected_roots, wrong_branch_paths):
        normalized_path = [str(node_id) for node_id in path]
        if (
            len(normalized_path) != gold_depth + 1
            or normalized_path[0] != expected_root
            or len(set(normalized_path)) != len(normalized_path)
            or set(normalized_path[1:]).intersection(gold_nodes)
        ):
            return False
        if any(
            not visible.graph.has_edge(left, right)
            for left, right in zip(normalized_path, normalized_path[1:])
        ):
            return False
    if (
        tree_role == TREE_ROLE_POSITIVE
        and set(wrong_branch_paths[0][1:]).intersection(
            wrong_branch_paths[1][1:]
        )
    ):
        return False
    return True


def allowed_answers(answer_label: str) -> tuple[str, ...]:
    """Return answer labels enabled by one construction setting."""
    return {
        "yes": (ANSWER_YES,),
        "no": (ANSWER_NO,),
        "mixed": (ANSWER_YES, ANSWER_NO),
    }[answer_label]


def count_yes_no(records: dict[str, dict[str, Any]]) -> tuple[int, int]:
    """Return ``(yes_count, no_count)`` among accepted needle records."""
    yes_count = 0
    no_count = 0
    for record in records.values():
        answers = record.get("answer") or []
        if answers and str(answers[0]).upper() == ANSWER_YES:
            yes_count += 1
        else:
            no_count += 1
    return yes_count, no_count


def bucket_label_count_lines(
    results: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> list[str]:
    """Return one final YES/NO count line per bucket in result order."""
    lines: list[str] = []
    for label, (_questions_payload, needles_payload) in results.items():
        records = {
            str(index): record
            for index, record in enumerate(needles_payload.get("questions", []))
            if isinstance(record, dict)
        }
        yes_count, no_count = count_yes_no(records)
        lines.append(f"bucket={label} YES={yes_count} NO={no_count}")
    return lines


def target_label_counts(
    target_questions: int | None,
    *,
    answer_label: str,
    yes_ratio: float,
) -> tuple[int, int] | None:
    """Return exact ``(YES, NO)`` quotas, or ``None`` for an unbounded run."""
    if answer_label not in {"yes", "no", "mixed"}:
        raise ValueError(f"Unsupported answer label: {answer_label!r}.")
    if not 0.0 <= yes_ratio <= 1.0:
        raise ValueError("yes_ratio must be between 0 and 1, inclusive.")
    if target_questions is None:
        return None
    if target_questions < 0:
        raise ValueError("target_questions must be non-negative.")
    if answer_label == "yes":
        return target_questions, 0
    if answer_label == "no":
        return 0, target_questions
    target_yes = int(float(target_questions) * yes_ratio + 0.5)
    return target_yes, target_questions - target_yes


def open_answers_for_counts(
    counts: tuple[int, int],
    *,
    answer_label: str,
    target_counts: tuple[int, int] | None,
) -> set[str]:
    """Return labels that may still be accepted for one bucket."""
    allowed = {
        "yes": {ANSWER_YES},
        "no": {ANSWER_NO},
        "mixed": {ANSWER_YES, ANSWER_NO},
    }[answer_label]
    if target_counts is None:
        return allowed
    yes_count, no_count = counts
    target_yes, target_no = target_counts
    open_answers: set[str] = set()
    if ANSWER_YES in allowed and yes_count < target_yes:
        open_answers.add(ANSWER_YES)
    if ANSWER_NO in allowed and no_count < target_no:
        open_answers.add(ANSWER_NO)
    return open_answers


def needle_paper_ids_for_trees(
    paper_ids: Sequence[str],
    primary_tree: CachedSearchTree,
    partner_tree: CachedSearchTree | None,
    visible: Any = None,
) -> list[str]:
    """Return every paper that must receive its own rendered needle block.

    Restricted to the graph's visible paper blocks when one is supplied, so
    branch papers disconnected by pruning are not rendered back in.
    """
    ordered_ids = [
        *primary_tree.paper_ids,
        *(partner_tree.paper_ids if partner_tree is not None else ()),
        *paper_ids,
    ]
    needle_ids = list(dict.fromkeys(str(paper_id) for paper_id in ordered_ids))
    if visible is None:
        return needle_ids
    return [
        paper_id
        for paper_id in needle_ids
        if paper_id in visible.expanded_paper_ids
    ]


def final_tree_identity_inputs(
    *,
    paper_ids: Sequence[str],
    references_by_paper: dict[str, Sequence[str]],
    trees: Sequence[CachedSearchTree],
    db: Any,
    include_author_identity: bool = True,
) -> tuple[
    dict[str, list[str]],
    dict[str, str],
    list[tuple[str, str]],
    dict[str, list[str]],
]:
    """Return strict identity data for one final rendered tree combination."""
    visible_paper_ids = set(str(value) for value in paper_ids)
    for cited_ids in references_by_paper.values():
        visible_paper_ids.update(str(value) for value in cited_ids)

    author_ids_by_paper: dict[str, list[str]] = {}
    citation_neighbors_by_paper: dict[str, list[str]] = {}
    for paper_id in sorted(visible_paper_ids):
        author_ids = (
            {
                author_id
                for tree in trees
                for author_id in tree.author_ids_by_paper_identity.get(
                    paper_id, []
                )
            }
            if include_author_identity
            else set()
        )
        author_loader = getattr(db, "get_expanded_authors_for_paper", None)
        if include_author_identity and not author_ids and callable(author_loader):
            author_ids = {
                author_nid(str(author.get("author_id") or ""))
                for author in author_loader(nid_db_id(paper_id))
                if str(author.get("author_id") or "").strip()
            }
        author_ids_by_paper[paper_id] = sorted(author_ids)
        citation_neighbors_by_paper[paper_id] = sorted(
            {
                neighbor_id
                for tree in trees
                for neighbor_id in tree.citation_neighbors_by_paper.get(
                    paper_id, []
                )
            }
        )

    intended_reference_pairs = {
        pair
        for tree in trees
        for pair in [
            *tree.protected_reference_pairs,
            *(
                (parent_id, expanded_id)
                for expanded_id, parent_id in (
                    tree.branch_parent_by_paper.items()
                )
            ),
        ]
    }
    expanded_paper_ids = set(str(value) for value in paper_ids)
    allowed_reference_pairs = sorted(
        (str(owner), str(cited_id))
        for owner, cited_ids in references_by_paper.items()
        for cited_id in cited_ids
        if cited_id in expanded_paper_ids
        and (str(owner), str(cited_id)) in intended_reference_pairs
    )
    author_names = (
        author_names_for_visible(author_ids_by_paper, db)
        if include_author_identity
        else {}
    )
    return (
        author_ids_by_paper,
        author_names,
        allowed_reference_pairs,
        citation_neighbors_by_paper,
    )


def build_needle_record(
    *,
    qid: str,
    answer: str,
    complexity: float,
    source_node_id: str,
    target_node_id: str,
    visible: Any,
    paper_ids: list[str],
    paper_block_ids: Sequence[str] | None = None,
    db: Any,
    needle_type: str,
    traversal_mode: str,
    random_seed: int | None,
    path_key: Any,
    paper_cache: dict[str, dict[str, Any] | None],
    context_group_id: str,
    primary_tree: CachedSearchTree,
    partner_tree: CachedSearchTree | None,
    component_sizes: tuple[int, ...],
    removed_reference_counts: tuple[int, ...],
    variant_signatures: tuple[str, ...] = (),
    removed_reference_pairs: tuple[
        tuple[tuple[str, str], ...], ...
    ] = (),
    pair_family_id: str = "",
    allocation_bucket: str = "",
    allow_hidden_citations: bool = False,
    allowed_terminal_reference_owners: Mapping[
        str, Sequence[str]
    ] | None = None,
) -> dict[str, Any]:
    """Assemble one needle record with reachability audit metadata."""
    paper_only_traversal = is_paper_only_traversal_mode(traversal_mode)
    default_needle_paper_ids = needle_paper_ids_for_trees(
        paper_ids,
        primary_tree,
        partner_tree,
        visible,
    )
    if paper_block_ids is None:
        needle_paper_ids = default_needle_paper_ids
    else:
        needle_paper_ids = list(paper_block_ids)
        if (
            len(set(needle_paper_ids)) != len(needle_paper_ids)
            or not set(needle_paper_ids).issubset(default_needle_paper_ids)
        ):
            raise ValueError("Explicit paper block IDs are invalid.")
    authors_by_paper = {
        paper_id: list(authors)
        for paper_id, authors in visible.authors_by_paper.items()
    }
    references_by_paper = {
        paper_id: list(references)
        for paper_id, references in visible.references_by_paper.items()
    }
    paper_titles = paper_titles_for_visible(
        needle_paper_ids,
        references_by_paper,
        db,
        paper_cache,
    )
    trees = [primary_tree]
    if partner_tree is not None:
        trees.append(partner_tree)
    if primary_tree.gold_depth > 0:
        expected_primary_role = (
            TREE_ROLE_POSITIVE
            if answer == ANSWER_YES
            else TREE_ROLE_SOURCE_NEGATIVE
        )
        if (
            primary_tree.tree_role != expected_primary_role
            or not wrong_branches_are_valid(
                visible=visible,
                tree_role=primary_tree.tree_role,
                source_node_id=primary_tree.source_node_id,
                target_node_id=primary_tree.target_node_id,
                witness=primary_tree.witness,
                gold_depth=primary_tree.gold_depth,
                wrong_branch_paths=primary_tree.wrong_branch_paths,
            )
        ):
            raise ValueError(
                "The primary tree is missing its required exact-depth wrong "
                "branch."
            )
    if answer == ANSWER_NO and partner_tree is not None and partner_tree.gold_depth > 0:
        if (
            partner_tree.tree_role != TREE_ROLE_TARGET_NEGATIVE
            or not wrong_branches_are_valid(
                visible=visible,
                tree_role=partner_tree.tree_role,
                source_node_id=partner_tree.source_node_id,
                target_node_id=partner_tree.target_node_id,
                witness=partner_tree.witness,
                gold_depth=partner_tree.gold_depth,
                wrong_branch_paths=partner_tree.wrong_branch_paths,
            )
        ):
            raise ValueError(
                "The target tree is missing its required exact-depth wrong "
                "branch."
            )
    (
        identity_author_ids_by_paper,
        author_names,
        allowed_reference_pairs,
        citation_neighbors_by_paper,
    ) = final_tree_identity_inputs(
        paper_ids=paper_ids,
        references_by_paper=references_by_paper,
        trees=trees,
        db=db,
        include_author_identity=not paper_only_traversal,
    )
    identity_citation_neighbors = (
        None if allow_hidden_citations else citation_neighbors_by_paper
    )
    identity_registry = build_visible_identity_registry(
        visible,
        paper_titles_by_paper=paper_titles,
        author_names_by_author=author_names,
        citation_owners_by_paper={},
        paper_author_ids_by_paper=identity_author_ids_by_paper,
        allowed_reference_pairs=allowed_reference_pairs,
        allowed_terminal_reference_owners=(
            allowed_terminal_reference_owners
        ),
        citation_neighbors_by_paper=identity_citation_neighbors,
    )
    if not identity_registry.is_unique:
        conflict_fields = [
            field
            for field in (
                "paper_collisions",
                "author_collisions",
                "missing_paper_node_ids",
                "missing_author_node_ids",
                "reference_owner_collisions",
                "paper_author_id_collisions",
                "paper_author_name_collisions",
                "disallowed_expanded_reference_pairs",
                "hidden_citation_pairs",
            )
            if getattr(identity_registry, field)
        ]
        raise ValueError(
            "Reachability graph has ambiguous visible identities: "
            + ", ".join(conflict_fields)
            + "."
        )
    needles = render_needles(
        paper_node_ids=needle_paper_ids,
        authors_by_paper=authors_by_paper,
        references_by_paper=references_by_paper,
        db=db,
        needle_type=needle_type,
        random_seed=random_seed,
        path_key=path_key,
        display_all_database_authors=paper_only_traversal,
    )
    needle_reference_statistics = calculate_needle_reference_statistics(
        needle_paper_ids,
        references_by_paper,
    )
    question_text = build_question_text(
        source_label=node_label(source_node_id, db, paper_cache),
        target_label=node_label(target_node_id, db, paper_cache),
        include_authorship=not paper_only_traversal,
    )
    if answer == ANSWER_YES:
        source_work = rooted_expected_dfs_steps(
            visible.graph,
            source_node_id,
            target_node_id,
        )
        target_work = rooted_expected_dfs_steps(
            visible.graph,
            target_node_id,
            source_node_id,
        )
        complexity_measure = "symmetric_expected_dfs"
    else:
        source_work = float(
            len(nx.node_connected_component(visible.graph, source_node_id))
        )
        target_work = float(
            len(nx.node_connected_component(visible.graph, target_node_id))
        )
        complexity_measure = "symmetric_exhaustive_component"
        if (
            len(component_sizes) != 2
            or component_sizes[0] != component_sizes[1]
            or source_work != target_work
            or source_work != float(complexity)
            or tuple(float(value) for value in component_sizes)
            != (source_work, target_work)
        ):
            raise ValueError(
                "A NO question must contain two exactly equal-sized "
                "source and target components."
            )
    tree_pair: dict[str, Any] = {
        "strategy": (
            "single-tree-pruning-cascade"
            if partner_tree is None
            else NO_CONSTRUCTION
        ),
        "allocation_strategy": PAIR_ALLOCATION_STRATEGY,
        "pair_family_id": pair_family_id,
        "path_pair_id": pair_family_id,
        "allocation_bucket": allocation_bucket,
        "primary_path_id": primary_tree.path_id,
        "primary_tree_cache_key": primary_tree.cache_key,
        "primary_tree_role": primary_tree.tree_role,
        "gold_depth": primary_tree.gold_depth,
        "primary_wrong_branch_paths": primary_tree.wrong_branch_paths,
        "component_sizes": list(component_sizes),
        "removed_optional_reference_counts": list(removed_reference_counts),
        "variant_signatures": list(variant_signatures),
        "removed_optional_reference_pairs": [
            [list(pair) for pair in pairs]
            for pairs in removed_reference_pairs
        ],
    }
    path_ids = [primary_tree.path_id]
    if partner_tree is not None:
        path_ids.append(partner_tree.path_id)
        tree_pair.update(
            {
                "partner_path_id": partner_tree.path_id,
                "partner_tree_cache_key": partner_tree.cache_key,
                "partner_tree_role": partner_tree.tree_role,
                "partner_gold_depth": partner_tree.gold_depth,
                "partner_wrong_branch_paths": (
                    partner_tree.wrong_branch_paths
                ),
                "disjointness_validated": True,
            }
        )
    reachability = {
        "label": answer,
        "source_node_id": source_node_id,
        "target_node_id": target_node_id,
        "complexity": {
            "measure": complexity_measure,
            "value": complexity,
            "source_endpoint_work": source_work,
            "target_endpoint_work": target_work,
        },
        "tree_pair": tree_pair,
        "author_identity_checks": not paper_only_traversal,
        "visible_graph": serialize_visible_graph(
            visible,
            paper_titles_by_paper=paper_titles,
            author_names_by_author=author_names,
            paper_author_ids_by_paper=identity_author_ids_by_paper,
            allowed_reference_pairs=allowed_reference_pairs,
            allowed_terminal_reference_owners=(
                allowed_terminal_reference_owners
            ),
            citation_neighbors_by_paper=citation_neighbors_by_paper,
            allow_hidden_citations=allow_hidden_citations,
        ),
    }
    if answer == ANSWER_NO:
        reachability["negative_construction"] = tree_pair
    return {
        "qid": qid,
        "context_group_id": context_group_id,
        "path_ids": path_ids,
        "needle_paper_ids": needle_paper_ids,
        "question_type": QUESTION_TYPE,
        "question": question_text,
        "needles": needles,
        NEEDLE_REFERENCE_STATISTICS_FIELD: needle_reference_statistics,
        "answer": [answer],
        "answer_type": ANSWER_TYPE,
        "reachability": reachability,
    }


def hardest_bucket_specs(
    bucket_specs: Sequence[tuple[str, float, float]],
) -> list[tuple[str, float, float]]:
    """Return configured buckets from greatest to least complexity."""
    return sorted(
        bucket_specs,
        key=lambda spec: (spec[1], spec[2], spec[0]),
        reverse=True,
    )


def ascending_bucket_specs(
    bucket_specs: Sequence[tuple[str, float, float]],
    *,
    source: str = "Requested",
) -> list[tuple[str, float, float]]:
    """Validate unique labels and return buckets in ascending order."""
    normalized: list[tuple[str, float, float]] = []
    labels: set[str] = set()
    for raw_spec in bucket_specs:
        if not isinstance(raw_spec, (list, tuple)) or len(raw_spec) != 3:
            raise ValueError(
                f"{source} bucket must contain label, minimum, and maximum."
            )
        label = str(raw_spec[0])
        if not label:
            raise ValueError(f"{source} bucket label must be non-empty.")
        if label in labels:
            raise ValueError(
                f"{source} buckets contain duplicate label {label!r}."
            )
        try:
            minimum = float(raw_spec[1])
            maximum = float(raw_spec[2])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{source} bucket {label!r} has non-numeric bounds."
            ) from error
        if minimum < 0 or not minimum < maximum:
            raise ValueError(
                f"{source} bucket {label!r} must satisfy 0 <= min < max."
            )
        labels.add(label)
        normalized.append((label, minimum, maximum))
    return sorted(normalized, key=lambda spec: (spec[1], spec[2], spec[0]))


def overlapping_bucket_pairs(
    bucket_specs: Sequence[tuple[str, float, float]],
) -> list[tuple[str, str]]:
    """Return pairs of bucket labels whose numeric intervals overlap."""
    pairs: list[tuple[str, str]] = []
    for index, (label, minimum, maximum) in enumerate(bucket_specs):
        for other_label, other_minimum, other_maximum in bucket_specs[index + 1 :]:
            if minimum < other_maximum and other_minimum < maximum:
                pairs.append((label, other_label))
    return pairs


def normalize_bucket_specs(
    bucket_specs: Any,
    *,
    source: str,
) -> dict[str, tuple[float, float]]:
    """Return validated bucket bounds keyed by label."""
    if not isinstance(bucket_specs, (list, tuple)):
        raise TypeError(f"{source} buckets must be a list.")
    return {
        label: (minimum, maximum)
        for label, minimum, maximum in ascending_bucket_specs(
            bucket_specs,
            source=source,
        )
    }


def validate_bucket_extension(
    saved_bucket_specs: Any,
    requested_bucket_specs: Any,
) -> set[str]:
    """Match buckets by exact label and bounds across independent active sets."""
    saved = normalize_bucket_specs(saved_bucket_specs, source="Saved")
    requested = normalize_bucket_specs(requested_bucket_specs, source="Requested")
    changed = sorted(
        label
        for label, bounds in saved.items()
        if label in requested and requested[label] != bounds
    )
    if changed:
        raise ValueError(
            "saved buckets have changed bounds: " + ", ".join(changed)
        )
    return set(requested).difference(saved)


def validate_explored_output_files(
    needles_output: Path,
    only_explored: int,
    *,
    multi_bucket: bool,
) -> None:
    """Recheck saved paper IDs even when all requested buckets are complete.

    This runs before the sampled paths are loaded; the graph location does
    not depend on them.
    """
    if not only_explored:
        return
    targets = (
        existing_bucket_output_paths(needles_output)
        if multi_bucket else [needles_output]
    )
    records = [
        record for target in targets
        for record in (load_json_output(target) or {}).get("questions", [])
    ]
    if not records:
        return
    _graph_path, db_path, _resolution = construction_run.resolve_question_graph_paths(
        {}, graph_dir=construction_run.DEFAULT_GRAPH_SOURCE, db_path_hint=None
    )
    db = DBReader(db_path)
    try:
        configure_only_explored(db, only_explored)
        validate_explored_records(records, db)
    finally:
        db.close()


def validate_resume_parameters(
    target: Path,
    saved_parameters: dict[str, Any],
    expected_parameters: dict[str, Any],
    *,
    mutable_parameter_keys: set[str] | None = None,
    allow_bucket_extension: bool = False,
) -> None:
    """Reject incompatible construction settings in one resume payload."""
    mutable_keys = mutable_parameter_keys or set()
    mismatched: list[str] = []
    bucket_error: ValueError | None = None
    for key, expected_value in expected_parameters.items():
        if key in mutable_keys:
            continue
        saved_value = saved_parameters.get(key)
        if key in {"none_option", "only_explored"} and key not in saved_parameters:
            saved_value = 0
        if key == "buckets" and allow_bucket_extension:
            try:
                validate_bucket_extension(saved_value, expected_value)
            except ValueError as error:
                mismatched.append(key)
                bucket_error = error
            continue
        if saved_value != expected_value:
            mismatched.append(key)
    if not mismatched:
        return
    names = ", ".join(sorted(mismatched))
    expected_format_version = expected_parameters.get(
        "format_version", QUESTION_FORMAT_VERSION
    )
    bucket_details = f": {bucket_error}" if bucket_error is not None else ""
    raise ValueError(
        f"Cannot resume {target} with different construction settings "
        f"({names}){bucket_details}. Use fresh output/state paths for format "
        f"version {expected_format_version}."
    )


def existing_bucket_output_paths(needles_output: Path) -> list[Path]:
    """Return existing files matching one multi-bucket output template."""
    placeholder = construction_run.BUCKET_OUTPUT_PLACEHOLDER
    template = str(needles_output)
    if placeholder not in template:
        return [needles_output] if needles_output.is_file() else []
    pattern = "*".join(glob.escape(part) for part in template.split(placeholder))
    return sorted(
        path
        for value in glob.glob(pattern)
        if (path := Path(value)).is_file()
    )


def validate_existing_bucket_outputs(
    needles_output: Path,
    *,
    multi_bucket: bool,
    expected_parameters: dict[str, Any],
    mutable_parameter_keys: set[str] | None = None,
) -> None:
    """Validate existing outputs before an independent bucket resume."""
    if not multi_bucket:
        return
    for target in existing_bucket_output_paths(needles_output):
        payload = load_json_output(target)
        if not payload:
            continue
        saved_parameters = payload.get("parameters")
        if not isinstance(saved_parameters, dict):
            raise ValueError(
                f"Cannot resume {target}: its construction parameters are missing."
            )
        validate_resume_parameters(
            target,
            saved_parameters,
            expected_parameters,
            mutable_parameter_keys=mutable_parameter_keys,
            allow_bucket_extension=True,
        )


def bucket_parameters(
    parameters: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    """Return output parameters with one immutable bucket identity."""
    buckets = normalize_bucket_specs(parameters.get("buckets"), source="Requested")
    if label not in buckets:
        raise ValueError(f"Missing bucket parameters for {label!r}.")
    minimum, maximum = buckets[label]
    result = dict(parameters)
    result["bucket"] = {
        "label": label,
        "minimum": minimum,
        "maximum": maximum,
    }
    return result


def validate_existing_bucket_identity(
    target: Path,
    label: str,
    saved_parameters: dict[str, Any],
    expected_parameters: dict[str, Any],
) -> None:
    """Require an existing output to describe the requested bucket exactly."""
    expected = normalize_bucket_specs(
        expected_parameters.get("buckets"),
        source="Requested",
    )
    if label not in expected:
        raise ValueError(f"Requested bucket {label!r} is missing from parameters.")
    expected_bounds = expected[label]
    raw_local = saved_parameters.get("bucket")
    if raw_local is not None:
        if not isinstance(raw_local, dict):
            raise ValueError(f"Cannot resume {target}: bucket metadata is invalid.")
        local_label = str(raw_local.get("label") or "")
        try:
            local_bounds = (
                float(raw_local["minimum"]),
                float(raw_local["maximum"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Cannot resume {target}: bucket metadata is invalid."
            ) from error
        if local_label != label or local_bounds != expected_bounds:
            raise ValueError(
                f"Cannot resume {target}: saved bucket identity does not match "
                f"requested bucket {label!r}."
            )
        return
    saved = normalize_bucket_specs(
        saved_parameters.get("buckets"),
        source="Saved",
    )
    if saved.get(label) != expected_bounds:
        raise ValueError(
            f"Cannot resume {target}: legacy bucket {label!r} is missing or "
            "has different bounds."
        )


def write_bucket_outputs(
    *,
    label: str,
    multi_bucket: bool,
    questions_output: Path,
    needles_output: Path,
    needle_records: dict[str, dict[str, Any]],
    parameters: dict[str, Any],
    skipped_path_ids: set[str],
    indent: int,
) -> None:
    """Write one bucket's questions.json and needles.json payloads."""
    questions_target = (
        construction_run.resolve_bucket_output_path(questions_output, label)
        if multi_bucket
        else questions_output
    )
    needles_target = (
        construction_run.resolve_bucket_output_path(needles_output, label)
        if multi_bucket
        else needles_output
    )
    records = list(needle_records.values())
    output_parameters = bucket_parameters(parameters, label)
    question_payload = {
        "parameters": output_parameters,
        "total_questions": len(records),
        "skipped_path_ids": sorted(skipped_path_ids),
        "questions": [
            {key: value for key, value in record.items() if key != "needles"}
            for record in records
        ],
    }
    needle_payload = {
        "parameters": output_parameters,
        "total_questions": len(records),
        "questions": records,
    }
    write_json_output(questions_target, question_payload, indent)
    write_json_output(needles_target, needle_payload, indent)


def existing_needle_records(
    label: str,
    multi_bucket: bool,
    needles_output: Path,
    *,
    expected_parameters: dict[str, Any],
    mutable_parameter_keys: set[str] | None = None,
    allow_bucket_extension: bool = False,
) -> dict[str, dict[str, Any]]:
    """Return already-accepted needle records for resume, keyed by qid."""
    target = (
        construction_run.resolve_bucket_output_path(needles_output, label)
        if multi_bucket
        else needles_output
    )
    payload = load_json_output(target)
    if not payload:
        return {}
    saved_parameters = payload.get("parameters")
    if not isinstance(saved_parameters, dict):
        raise ValueError(
            f"Cannot resume {target}: its construction parameters are missing."
        )
    validate_resume_parameters(
        target,
        saved_parameters,
        expected_parameters,
        mutable_parameter_keys=mutable_parameter_keys,
        allow_bucket_extension=allow_bucket_extension,
    )
    validate_existing_bucket_identity(
        target,
        label,
        saved_parameters,
        expected_parameters,
    )
    if not payload.get("questions"):
        return {}
    records = {
        str(record.get("qid")): record
        for record in payload.get("questions", [])
        if record.get("qid") is not None
    }
    if not rerender_questions_enabled():
        return records
    records, stats = rerender_needle_records(records)
    print(f"[{CLI_NAME}] re-rendered {target}: {stats.summary()}")
    return records


def saved_target_questions(
    bucket_labels: Sequence[str],
    *,
    multi_bucket: bool,
    needles_output: Path,
) -> tuple[bool, int | None]:
    """Return the consistent saved run quota across existing bucket outputs."""
    saved_values: list[int | None] = []
    for label in bucket_labels:
        target = (
            construction_run.resolve_bucket_output_path(needles_output, label)
            if multi_bucket
            else needles_output
        )
        payload = load_json_output(target)
        if not payload:
            continue
        parameters = payload.get("parameters")
        if not isinstance(parameters, dict) or "target_questions" not in parameters:
            continue
        raw_value = parameters.get("target_questions")
        saved_values.append(None if raw_value is None else int(raw_value))
    if not saved_values:
        return False, None
    first_value = saved_values[0]
    if any(value != first_value for value in saved_values[1:]):
        raise ValueError("Existing bucket outputs have inconsistent target quotas.")
    return True, first_value
