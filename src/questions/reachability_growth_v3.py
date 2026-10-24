"""Growth-state and negative-distractor-tree helpers from the v3 pipeline.

Prepared growth paths, paper facts, static question growth states, and the
serialized negative distractor trees used by distractor and prompt
construction. Author collections here represent graph identity only.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx
import orjson

from src.prompts.reachability_yesno_qa import is_paper_only_traversal_mode
from src.questions import reachability_construction as yesno_base
from src.questions.explored import entry_is_explored
from src.questions.reachability_crossed_trees import path_tree_cache_key
from src.questions.reachability_level_growth import (
    TARGET_COMPONENT,
    GrowthTarget,
    GrowthTreeState,
    PaperFacts,
    PaperFactsReader,
    QuestionGrowthGroup,
    ReferenceCalibration,
    initialize_growth_tree,
    stable_growth_seed,
)
from src.questions.reachability_long_path_profiles import (
    LONG_PATH_PROFILE_FIELD,
)
from src.questions.reachability_yesno import (
    visible_graph_from_metadata,
)
from src.questions.shared import (
    DEPTH_ZERO_REFERENCE_STATISTICS_FIELD,
    needle_reference_distribution,
    prepare_question_path,
)
from src.setup.constants import (
    EDGE_CITED_BY,
    EDGE_CITING,
    NODE_PAPER,
    author_nid,
    nid_db_id,
    nid_type,
)

V3_GROWTH_FIELD = "level_growth"
V3_TREE_CONSTRUCTION_STRATEGY = (
    "three-segment-thin-tree-target-driven-level-growth-v3"
)
NEGATIVE_DISTRACTOR_TREE_VERSION = 1


@dataclass(frozen=True, slots=True)
class PreparedGrowthPath:
    """One input path orientation converted to a validated thin tree."""

    entry: dict[str, Any]
    state: GrowthTreeState

    @property
    def path_id(self) -> str:
        """Return the stable sampled-path identifier."""
        return self.state.path_id


@dataclass(slots=True)
class PreparedPathStore:
    """Build thin orientations lazily for paths attempted by an allocator."""

    entries_by_path_id: dict[str, dict[str, Any]]
    context: yesno_base.ReachabilityCheckerContext
    facts_reader: PaperFactsReader
    traversal_mode: str
    max_author_answer_index: int | None
    cache: dict[str, list[PreparedGrowthPath]]

    def get(self, path_id: str) -> list[PreparedGrowthPath]:
        """Return cached validated thin orientations for one path."""
        if path_id not in self.cache:
            self.cache[path_id] = build_prepared_growth_paths(
                self.entries_by_path_id[path_id],
                context=self.context,
                facts_reader=self.facts_reader,
                traversal_mode=self.traversal_mode,
                max_author_answer_index=self.max_author_answer_index,
            )
        return self.cache[path_id]


def stable_order(
    values: Sequence[str],
    *,
    random_seed: int | None,
    key: Any,
) -> list[str]:
    """Return a deterministic uniform shuffle for one allocation decision."""
    result = list(values)
    random.Random(stable_growth_seed(random_seed, key)).shuffle(result)
    return result


def orientation_order(
    prepared: Sequence[PreparedGrowthPath],
    *,
    random_seed: int | None,
    key: Any,
) -> list[PreparedGrowthPath]:
    """Return candidate thin-tree orientations in a stable random order."""
    result = list(prepared)
    random.Random(stable_growth_seed(random_seed, key)).shuffle(result)
    return result


def paper_facts_reader(
    context: yesno_base.ReachabilityCheckerContext,
    *,
    include_author_identity: bool,
) -> PaperFactsReader:
    """Return one lazy, memoized reader for growth and disjointness facts."""
    paper_cache: dict[str, dict[str, Any] | None] = {}

    def load_facts(paper_id: str) -> PaperFacts | None:
        if nid_type(paper_id) != NODE_PAPER:
            return None
        identity = yesno_base.paper_visible_identity(
            paper_id,
            context,
            paper_cache,
            include_author_identity=include_author_identity,
        )
        if identity is None:
            return None
        title_key, author_ids, author_name_keys = identity
        outgoing: set[str] = set()
        incoming: set[str] = set()
        for neighbor_id, attributes in context.graph.adj.get(paper_id, []):
            normalized_neighbor = str(neighbor_id)
            if nid_type(normalized_neighbor) != NODE_PAPER:
                continue
            edge_type = str(attributes.get("edge_type") or "")
            if edge_type == EDGE_CITING:
                outgoing.add(normalized_neighbor)
            elif edge_type == EDGE_CITED_BY:
                incoming.add(normalized_neighbor)
        return PaperFacts(
            paper_id=paper_id,
            title_key=title_key,
            author_ids=author_ids,
            author_name_keys=author_name_keys,
            outgoing_paper_ids=tuple(sorted(outgoing)),
            incoming_paper_ids=tuple(sorted(incoming)),
            citation_neighbor_ids=frozenset(outgoing | incoming),
        )

    return PaperFactsReader(load_facts)


def paper_author_node_ids(
    paper_ids: Sequence[str],
    context: yesno_base.ReachabilityCheckerContext,
) -> dict[str, list[str]]:
    """Load all rendered author IDs for the supplied paper blocks in a batch."""
    normalized_papers = list(dict.fromkeys(str(value) for value in paper_ids))
    db_ids = [nid_db_id(paper_id) for paper_id in normalized_papers]
    batch_loader = getattr(context.db, "get_expanded_authors_for_papers", None)
    if callable(batch_loader):
        rows_by_db_id = batch_loader(db_ids)
    else:
        rows_by_db_id = {
            db_id: context.db.get_expanded_authors_for_paper(db_id)
            for db_id in db_ids
        }
    result: dict[str, list[str]] = {}
    for paper_id, db_id in zip(normalized_papers, db_ids):
        rows = rows_by_db_id.get(db_id, [])
        result[paper_id] = list(
            dict.fromkeys(
                author_nid(str(row.get("author_id") or ""))
                for row in rows
                if str(row.get("author_id") or "").strip()
            )
        )
    return result


def add_required_citation(
    references_by_paper: dict[str, list[str]],
    protected_pairs: list[tuple[str, str]],
    source_id: str,
    target_id: str,
    edge_type: str,
) -> None:
    """Add one citation edge traversed in either stored direction."""
    if edge_type not in {EDGE_CITING, EDGE_CITED_BY}:
        return
    owner_id, cited_id = (
        (source_id, target_id)
        if edge_type == EDGE_CITING
        else (target_id, source_id)
    )
    references_by_paper.setdefault(owner_id, [])
    if cited_id not in references_by_paper[owner_id]:
        references_by_paper[owner_id].append(cited_id)
    pair = (owner_id, cited_id)
    if pair not in protected_pairs:
        protected_pairs.append(pair)


def thin_tree_inputs(
    *,
    oriented_nodes: Sequence[dict[str, Any]],
    oriented_edges: Sequence[dict[str, Any] | None],
    source_branch: tuple[Sequence[str], Sequence[str]],
    target_branch: tuple[Sequence[str], Sequence[str]],
    context: yesno_base.ReachabilityCheckerContext,
    include_authors: bool,
) -> tuple[
    list[str],
    dict[str, list[str]],
    dict[str, list[str]],
    list[tuple[str, str]],
    tuple[str, ...],
]:
    """Build only the three protected path segments, with no extra citations."""
    witness = tuple(str(node["node_id"]) for node in oriented_nodes)
    source_path = [str(value) for value in source_branch[0]]
    target_path = [str(value) for value in target_branch[0]]
    full_chain = [
        *reversed(source_path[1:]),
        *witness,
        *target_path[1:],
    ]
    paper_ids = list(
        dict.fromkeys(
            node_id for node_id in full_chain if nid_type(node_id) == NODE_PAPER
        )
    )
    references_by_paper = {paper_id: [] for paper_id in paper_ids}
    protected_pairs: list[tuple[str, str]] = []

    for position in range(1, len(oriented_nodes)):
        edge = oriented_edges[position]
        if not isinstance(edge, dict):
            continue
        add_required_citation(
            references_by_paper,
            protected_pairs,
            str(edge.get("from") or witness[position - 1]),
            str(edge.get("to") or witness[position]),
            str(edge.get("edge_type") or ""),
        )
    for node_ids, edge_types in (source_branch, target_branch):
        for source_id, target_id, edge_type in zip(
            node_ids,
            node_ids[1:],
            edge_types,
        ):
            add_required_citation(
                references_by_paper,
                protected_pairs,
                str(source_id),
                str(target_id),
                str(edge_type),
            )

    authors_by_paper = (
        paper_author_node_ids(paper_ids, context)
        if include_authors
        else {paper_id: [] for paper_id in paper_ids}
    )
    return (
        paper_ids,
        authors_by_paper,
        references_by_paper,
        protected_pairs,
        witness,
    )


def build_prepared_growth_paths(
    entry: dict[str, Any],
    *,
    context: yesno_base.ReachabilityCheckerContext,
    facts_reader: PaperFactsReader,
    traversal_mode: str,
    max_author_answer_index: int | None,
) -> list[PreparedGrowthPath]:
    """Return every valid orientation of one three-segment input path."""
    if not entry_is_explored(entry, context.db):
        return []
    prepared = prepare_question_path(
        entry.get("path") or entry.get("steps") or []
    )
    if prepared is None:
        return []
    nodes, edges = prepared
    paper_only = is_paper_only_traversal_mode(traversal_mode)
    if paper_only and any(
        nid_type(str(node.get("node_id") or "")) != NODE_PAPER
        for node in nodes
    ):
        return []
    profile = entry.get(LONG_PATH_PROFILE_FIELD)
    if not isinstance(profile, dict):
        return []
    path_id = str(entry.get("path_id") or entry.get("path_index") or "")
    neighbor_cache: dict[tuple[str, str], list[tuple[str, str]]] = {}
    results: list[PreparedGrowthPath] = []
    for orientation_index, (oriented_nodes, oriented_edges) in enumerate(
        yesno_base.reachability_path_orientations(nodes, edges)
    ):
        if not yesno_base.target_author_index_allowed(
            oriented_nodes,
            oriented_edges,
            max_author_answer_index,
        ):
            continue
        source_candidates = yesno_base.provided_wrong_branch_candidate(
            profile,
            endpoint="source",
            oriented_nodes=oriented_nodes,
            orientation_index=orientation_index,
            traversal_mode=traversal_mode,
            graph=context.graph,
            neighbor_cache=neighbor_cache,
        )
        target_candidates = yesno_base.provided_wrong_branch_candidate(
            profile,
            endpoint="target",
            oriented_nodes=oriented_nodes,
            orientation_index=orientation_index,
            traversal_mode=traversal_mode,
            graph=context.graph,
            neighbor_cache=neighbor_cache,
        )
        if not source_candidates or not target_candidates:
            continue
        source_branch = source_candidates[0]
        target_branch = target_candidates[0]
        (
            paper_ids,
            authors_by_paper,
            references_by_paper,
            protected_pairs,
            witness,
        ) = thin_tree_inputs(
            oriented_nodes=oriented_nodes,
            oriented_edges=oriented_edges,
            source_branch=source_branch,
            target_branch=target_branch,
            context=context,
            include_authors=not paper_only,
        )
        entry_cache_key = path_tree_cache_key(entry)
        initialized = initialize_growth_tree(
            cache_key=f"{entry_cache_key}:{orientation_index}:v3",
            path_id=path_id,
            orientation_index=orientation_index,
            source_node_id=str(oriented_nodes[0]["node_id"]),
            target_node_id=str(oriented_nodes[-1]["node_id"]),
            witness=witness,
            source_wrong_path=source_branch[0],
            target_wrong_path=target_branch[0],
            gold_paper_ids=paper_ids,
            expanded_paper_ids=paper_ids,
            authors_by_paper=authors_by_paper,
            references_by_paper=references_by_paper,
            protected_reference_pairs=protected_pairs,
            facts_reader=facts_reader,
            include_author_identity=not paper_only,
        )
        if initialized is not None:
            results.append(PreparedGrowthPath(entry=entry, state=initialized))
    return results


def negative_component_target(question: Mapping[str, Any]) -> int | None:
    """Return the integer component target for a negative distractor tree."""
    reachability = question.get("reachability")
    if not isinstance(reachability, Mapping):
        return None
    growth = reachability.get(V3_GROWTH_FIELD)
    if not isinstance(growth, Mapping):
        return None
    raw_target = growth.get("target_value")
    if (
        not isinstance(raw_target, (int, float))
        or isinstance(raw_target, bool)
        or not math.isfinite(float(raw_target))
        or float(raw_target) < 0
    ):
        return None
    return math.ceil(float(raw_target))


def question_reference_calibration(
    question: Mapping[str, Any],
) -> ReferenceCalibration | None:
    """Return the final main-tree calibration saved on one v3 question."""
    distribution = needle_reference_distribution(question)
    if distribution is None:
        return None
    statistics = question.get(DEPTH_ZERO_REFERENCE_STATISTICS_FIELD)
    citation_counts: tuple[int, ...] = ()
    if isinstance(statistics, Mapping):
        raw_counts = statistics.get("citation_counts")
        if isinstance(raw_counts, list) and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in raw_counts
        ):
            citation_counts = tuple(raw_counts)
    mean, std = distribution
    return ReferenceCalibration((), citation_counts, mean, std)


def static_question_growth_states(
    question: Mapping[str, Any],
    *,
    facts_reader: PaperFactsReader,
    include_author_identity: bool,
) -> tuple[GrowthTreeState, ...] | None:
    """Rebuild the question's final components as already-reached states."""
    reachability = question.get("reachability")
    if not isinstance(reachability, Mapping):
        return None
    metadata = reachability.get("visible_graph")
    if not isinstance(metadata, dict):
        return None
    try:
        visible = visible_graph_from_metadata(metadata)
    except ValueError:
        return None
    if not visible.graph.nodes:
        return None

    stored_expanded = [
        str(value) for value in metadata.get("expanded_paper_ids") or []
    ]
    qid = str(question.get("qid") or "question")
    components = sorted(
        (set(component) for component in nx.connected_components(visible.graph)),
        key=lambda component: min(str(value) for value in component),
    )
    states: list[GrowthTreeState] = []
    for component_index, component in enumerate(components):
        expanded = [
            paper_id for paper_id in stored_expanded if paper_id in component
        ]
        if not expanded:
            return None
        authors_by_paper = {
            paper_id: list(visible.authors_by_paper.get(paper_id, ()))
            for paper_id in expanded
        }
        references_by_paper = {
            paper_id: [
                cited_id
                for cited_id in visible.references_by_paper.get(paper_id, ())
                if cited_id in component
            ]
            for paper_id in expanded
        }
        endpoints = sorted(str(value) for value in component)
        source_node_id = endpoints[0]
        target_node_id = endpoints[-1]
        try:
            witness = tuple(
                str(value)
                for value in nx.shortest_path(
                    visible.graph,
                    source_node_id,
                    target_node_id,
                )
            )
        except nx.NetworkXNoPath:
            return None
        protected_pairs = [
            (paper_id, cited_id)
            for paper_id, cited_ids in references_by_paper.items()
            for cited_id in cited_ids
        ]
        state = initialize_growth_tree(
            cache_key=f"question:{qid}:component:{component_index}",
            path_id=f"question:{qid}:component:{component_index}",
            orientation_index=0,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            witness=witness,
            source_wrong_path=(source_node_id,),
            target_wrong_path=(target_node_id,),
            gold_paper_ids=expanded,
            expanded_paper_ids=expanded,
            authors_by_paper=authors_by_paper,
            references_by_paper=references_by_paper,
            protected_reference_pairs=protected_pairs,
            facts_reader=facts_reader,
            include_author_identity=include_author_identity,
        )
        if state is None:
            return None
        states.append(state)
    group = QuestionGrowthGroup(
        trees=states,
        targets=[
            GrowthTarget(TARGET_COMPONENT, state.component_size)
            for state in states
        ],
        facts_reader=facts_reader,
        include_author_identity=include_author_identity,
    )
    if group.rebuild_registries() is not None:
        return None
    return tuple(states)


def serialize_negative_distractor_tree(
    tree: GrowthTreeState,
    *,
    target_component_size: int,
) -> dict[str, Any]:
    """Serialize one exact negative component for later atomic rendering."""
    expanded = [
        paper_id
        for paper_id in tree.expanded_paper_order
        if paper_id in tree.expanded_paper_ids
    ]
    payload: dict[str, Any] = {
        "version": NEGATIVE_DISTRACTOR_TREE_VERSION,
        "cache_key": tree.cache_key,
        "path_id": tree.path_id,
        "orientation_index": tree.orientation_index,
        "source_node_id": tree.source_node_id,
        "target_node_id": tree.target_node_id,
        "witness": list(tree.witness),
        "source_wrong_path": list(tree.source_wrong_path),
        "target_wrong_path": list(tree.target_wrong_path),
        "gold_paper_ids": list(tree.gold_paper_ids),
        "expanded_paper_ids": expanded,
        "authors_by_paper": {
            paper_id: sorted(tree.authors_by_paper.get(paper_id, set()))
            for paper_id in expanded
        },
        "references_by_paper": {
            paper_id: sorted(tree.references_by_paper.get(paper_id, set()))
            for paper_id in expanded
        },
        "protected_reference_pairs": [
            list(pair) for pair in sorted(tree.protected_reference_pairs)
        ],
        "paper_depth": dict(sorted(tree.paper_depth.items())),
        "sampled_reference_caps": dict(sorted(tree.sampled_caps_by_paper.items())),
        "component_size": tree.component_size,
        "target_component_size": target_component_size,
    }
    payload["tree_id"] = negative_distractor_tree_signature(payload)
    return payload


def negative_distractor_tree_signature(payload: Mapping[str, Any]) -> str:
    """Return the content signature of one serialized negative tree."""
    signed_payload = {
        str(key): value for key, value in payload.items() if key != "tree_id"
    }
    return hashlib.sha256(
        orjson.dumps(signed_payload, option=orjson.OPT_SORT_KEYS)
    ).hexdigest()


def deserialize_negative_distractor_tree(
    payload: Mapping[str, Any],
    *,
    facts_reader: PaperFactsReader,
    include_author_identity: bool,
) -> GrowthTreeState | None:
    """Rebuild and validate one serialized negative distractor component."""
    if (
        payload.get("version") != NEGATIVE_DISTRACTOR_TREE_VERSION
        or payload.get("tree_id") != negative_distractor_tree_signature(payload)
    ):
        return None
    expanded = payload.get("expanded_paper_ids")
    authors = payload.get("authors_by_paper")
    references = payload.get("references_by_paper")
    protected = payload.get("protected_reference_pairs")
    if (
        not isinstance(expanded, list)
        or not isinstance(authors, Mapping)
        or not isinstance(references, Mapping)
        or not isinstance(protected, list)
    ):
        return None
    try:
        state = initialize_growth_tree(
            cache_key=str(payload.get("cache_key") or payload.get("tree_id") or ""),
            path_id=str(payload.get("path_id") or ""),
            orientation_index=int(payload.get("orientation_index") or 0),
            source_node_id=str(payload.get("source_node_id") or ""),
            target_node_id=str(payload.get("target_node_id") or ""),
            witness=[str(value) for value in payload.get("witness") or []],
            source_wrong_path=[
                str(value) for value in payload.get("source_wrong_path") or []
            ],
            target_wrong_path=[
                str(value) for value in payload.get("target_wrong_path") or []
            ],
            gold_paper_ids=[
                str(value) for value in payload.get("gold_paper_ids") or []
            ],
            expanded_paper_ids=[str(value) for value in expanded],
            authors_by_paper={
                str(paper_id): [str(value) for value in author_ids]
                for paper_id, author_ids in authors.items()
                if isinstance(author_ids, list)
            },
            references_by_paper={
                str(paper_id): [str(value) for value in cited_ids]
                for paper_id, cited_ids in references.items()
                if isinstance(cited_ids, list)
            },
            protected_reference_pairs=[
                (str(pair[0]), str(pair[1]))
                for pair in protected
                if isinstance(pair, list) and len(pair) == 2
            ],
            facts_reader=facts_reader,
            include_author_identity=include_author_identity,
        )
    except (TypeError, ValueError):
        return None
    if state is None:
        return None
    try:
        expected_size = int(payload.get("component_size"))
        target_size = int(payload.get("target_component_size"))
    except (TypeError, ValueError):
        return None
    if state.component_size != expected_size or expected_size != target_size:
        return None
    return state
