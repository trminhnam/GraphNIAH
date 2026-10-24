"""Shared construction helpers for the v4 reachability pipelines.

This module connects the target-driven level-growth engine to the existing
reachability renderers and final invariant checks.  The two v4 entry points
keep their allocation logic separate, but share thin-tree construction,
disjointness caching, fallback pruning, and final tree materialization here.
Author collections in this module represent graph identity only. Paper-only
rendering loads bibliographic author names separately from the database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import orjson
import math
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

import networkx as nx

from src.questions import reachability_construction as yesno_base
from src.questions.explored import allowed_explored_paper_ids, entry_is_explored
from src.prompts.reachability_yesno_qa import is_paper_only_traversal_mode
from src.questions.reachability_crossed_trees import (
    CachedSearchTree,
    TreeBucketVariant,
    path_tree_cache_key,
)
from src.questions.reachability_level_growth import (
    apply_pruning_prefix,
    build_pruning_ladder,
    reference_calibration,
    stable_growth_seed,
)
from src.questions.reachability_rendered_growth import (
    GrowthGroupResult,
    GrowthTarget,
    GrowthTreeState,
    PaperFacts,
    PaperFactsReader,
    QuestionGrowthGroup,
    ReferenceCalibration,
    TARGET_COMPONENT,
    TARGET_YES,
    grow_question_group,
    initialize_growth_tree,
    validate_rendered_growth_tree,
)
from src.questions.reachability_long_path_profiles import (
    LONG_PATH_PROFILE_FIELD,
)
from src.questions.reachability_yesno import (
    normalize_visible_identity,
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
from src.utils.resume import (
    format_resume_mismatches,
    load_json_output,
    write_json_output,
)


V4_GROWTH_FIELD = "level_growth"
V4_PAIR_CACHE_VERSION = 1
V4_TREE_CONSTRUCTION_STRATEGY = "rendered-edge-level-growth-v1"
V4_RENDERED_GRAPH_POLICY_VERSION = 1
V4_PAPER_IDENTITY_POLICY = "normalized-title-single-node-v1"
V4_CANDIDATE_COMMIT_POLICY = "render-before-next-depth-v1"
V4_NEGATIVE_DISTRACTOR_TREE_VERSION = 1
TARGET_LIMIT_REJECTION_REASONS = frozenset(
    {
        "target_overshoot",
        "center_promotion_target_overshoot",
    }
)
FINAL_IDENTITY_CONFLICT_FIELDS = (
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


def record_final_invariant_failure(
    skip_reason_counts: dict[str, int],
    error: ValueError,
) -> None:
    """Record one failed final audit and each named identity conflict field."""
    total_key = f"final_invariant:{type(error).__name__}"
    skip_reason_counts[total_key] = skip_reason_counts.get(total_key, 0) + 1
    message = str(error)
    for field in FINAL_IDENTITY_CONFLICT_FIELDS:
        if field not in message:
            continue
        field_key = f"final_invariant_field:{field}"
        skip_reason_counts[field_key] = (
            skip_reason_counts.get(field_key, 0) + 1
        )


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
    cache_max_entries: int | None = None

    def get(self, path_id: str) -> list[PreparedGrowthPath]:
        """Return cached validated thin orientations for one path."""
        if path_id in self.cache:
            prepared = self.cache.pop(path_id)
            self.cache[path_id] = prepared
            return prepared
        prepared = build_prepared_growth_paths(
            self.entries_by_path_id[path_id],
            context=self.context,
            facts_reader=self.facts_reader,
            traversal_mode=self.traversal_mode,
            max_author_answer_index=self.max_author_answer_index,
        )
        self.cache[path_id] = prepared
        if (
            self.cache_max_entries is not None
            and len(self.cache) > self.cache_max_entries
        ):
            self.cache.pop(next(iter(self.cache)))
        return prepared


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


@dataclass(frozen=True, slots=True)
class GrowthPlan:
    """One exact final group selected for an output bucket."""

    trees: tuple[GrowthTreeState, ...]
    target_kinds: tuple[str, ...]
    target_value: float
    bucket_label: str
    calibration: ReferenceCalibration
    used_fallback: bool
    reused_higher_bucket_tree: bool
    removed_paper_counts: tuple[int, ...]
    removed_reference_pairs: tuple[tuple[tuple[str, str], ...], ...]
    maximum_depth_processed: int
    rejection_counts: dict[str, int]


@dataclass(slots=True)
class ReusableGrowthGroup:
    """A higher-bucket final group that may be pruned for smaller buckets."""

    trees: tuple[GrowthTreeState, ...]
    target_kinds: tuple[str, ...]
    source_bucket_label: str


@dataclass(slots=True)
class PairDisjointnessCache:
    """Cache permanent thin-tree overlap results across questions and runs."""

    path: Path | None
    configuration: dict[str, Any]
    results: dict[str, str] = field(default_factory=dict)
    dirty: bool = False
    max_entries: int | None = None

    def key(self, first_cache_key: str, second_cache_key: str) -> str:
        """Return one canonical key for an unordered thin-tree pair."""
        return "\u001f".join(sorted((first_cache_key, second_cache_key)))

    def get(self, first_cache_key: str, second_cache_key: str) -> str | None:
        """Return ``compatible``, a rejection reason, or ``None`` if unknown."""
        key = self.key(first_cache_key, second_cache_key)
        if key not in self.results:
            return None
        result = self.results.pop(key)
        self.results[key] = result
        return result

    def put(
        self,
        first_cache_key: str,
        second_cache_key: str,
        result: str,
    ) -> None:
        """Save one permanent thin-tree comparison result."""
        key = self.key(first_cache_key, second_cache_key)
        if self.results.get(key) == result:
            return
        self.results.pop(key, None)
        self.results[key] = result
        if self.max_entries is not None and len(self.results) > self.max_entries:
            self.results.pop(next(iter(self.results)))
        self.dirty = True

    def write(self, indent: int) -> None:
        """Checkpoint the cache when a path was configured."""
        if self.path is None or not self.dirty:
            return
        write_json_output(
            self.path,
            {
                "version": V4_PAIR_CACHE_VERSION,
                "configuration": self.configuration,
                "pair_results": self.results,
            },
            indent,
        )
        self.dirty = False


def load_pair_disjointness_cache(
    path: Path | None,
    configuration: dict[str, Any],
    *,
    max_entries: int | None = None,
    ignored_configuration_fields: Sequence[str] = (),
) -> PairDisjointnessCache:
    """Load a v4 pair cache or create an empty in-memory cache."""
    cache = PairDisjointnessCache(
        path=path,
        configuration=configuration,
        max_entries=max_entries,
    )
    if path is None or not path.exists():
        return cache
    payload = load_json_output(path)
    if not payload:
        return cache
    ignored_fields = set(ignored_configuration_fields)
    saved_configuration = payload.get("configuration")
    if isinstance(saved_configuration, Mapping):
        saved_configuration = {
            key: value
            for key, value in saved_configuration.items()
            if key not in ignored_fields
        }
    current_configuration = {
        key: value for key, value in configuration.items() if key not in ignored_fields
    }
    if (
        payload.get("version") != V4_PAIR_CACHE_VERSION
        or saved_configuration != current_configuration
    ):
        mismatches = format_resume_mismatches(
            {
                "cache_version": payload.get("version"),
                "configuration": saved_configuration,
            },
            {
                "cache_version": V4_PAIR_CACHE_VERSION,
                "configuration": current_configuration,
            },
        )
        raise ValueError(
            f"V4 disjointness cache configuration does not match: {path}.\n"
            f"Mismatches:\n{mismatches}\n"
            "Use a fresh --search-tree-cache path."
        )
    raw_results = payload.get("pair_results")
    if isinstance(raw_results, dict):
        cache.results = {
            str(key): str(value) for key, value in raw_results.items()
        }
        if max_entries is not None:
            while len(cache.results) > max_entries:
                cache.results.pop(next(iter(cache.results)))
    return cache


def cache_configuration(
    *,
    graph_path: Path,
    db_path: Path,
    traversal_mode: str,
    question_hops: int,
    max_author_answer_index: int | None,
    only_explored: int = 0,
) -> dict[str, Any]:
    """Return the settings that make permanent pair results reusable."""

    def fingerprint(path: Path) -> dict[str, Any]:
        resolved = path.resolve()
        try:
            stat = resolved.stat()
        except FileNotFoundError:
            return {"path": str(resolved), "missing": True}
        return {
            "path": str(resolved),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    return {
        **({"only_explored": 1} if only_explored else {}),
        "version": V4_PAIR_CACHE_VERSION,
        "constructor_version": "v4",
        "graph": fingerprint(graph_path),
        "db": fingerprint(db_path),
        "traversal_mode": traversal_mode,
        "question_hops": question_hops,
        "max_author_answer_index": max_author_answer_index,
        "tree_construction_strategy": V4_TREE_CONSTRUCTION_STRATEGY,
        "rendered_graph_policy_version": V4_RENDERED_GRAPH_POLICY_VERSION,
        "paper_identity_policy": V4_PAPER_IDENTITY_POLICY,
        "candidate_commit_policy": V4_CANDIDATE_COMMIT_POLICY,
    }


def paper_facts_reader(
    context: yesno_base.ReachabilityCheckerContext,
    *,
    include_author_identity: bool,
    cache_max_entries: int | None = None,
    identity_cache_max_entries: int | None = None,
) -> PaperFactsReader:
    """Load identity before adjacency and memoize both independently."""
    paper_cache: dict[str, dict[str, Any] | None] = {}

    def load_identity(paper_id: str) -> PaperFacts | None:
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
        return PaperFacts(
            paper_id=paper_id,
            title_key=title_key,
            author_ids=author_ids,
            author_name_keys=author_name_keys,
            outgoing_paper_ids=(),
            incoming_paper_ids=(),
            citation_neighbor_ids=frozenset(),
        )

    def load_identities(
        paper_ids: Sequence[str],
    ) -> Mapping[str, PaperFacts | None]:
        normalized_ids = list(dict.fromkeys(str(value) for value in paper_ids))
        valid_ids = [
            paper_id
            for paper_id in normalized_ids
            if nid_type(paper_id) == NODE_PAPER
        ]
        allowed_ids = allowed_explored_paper_ids(context.db, valid_ids)
        valid_ids = [paper_id for paper_id in valid_ids if paper_id in allowed_ids]
        title_loader = getattr(context.db, "get_paper_titles", None)
        if not callable(title_loader):
            return {paper_id: load_identity(paper_id) for paper_id in normalized_ids}

        db_ids = [nid_db_id(paper_id) for paper_id in valid_ids]
        titles_by_db_id = title_loader(db_ids)
        authors_by_db_id: Mapping[str, Sequence[Mapping[str, Any]]]
        if include_author_identity:
            author_loader = getattr(
                context.db,
                "get_expanded_authors_for_papers",
                None,
            )
            if callable(author_loader):
                authors_by_db_id = author_loader(db_ids)
            else:
                authors_by_db_id = {
                    db_id: context.db.get_expanded_authors_for_paper(db_id)
                    for db_id in db_ids
                }
        else:
            authors_by_db_id = {}

        results: dict[str, PaperFacts | None] = {
            paper_id: None for paper_id in normalized_ids
        }
        for paper_id, db_id in zip(valid_ids, db_ids):
            title = titles_by_db_id.get(db_id, "")
            title_key = normalize_visible_identity(title)
            if not title_key:
                continue
            author_ids: set[str] = set()
            author_name_keys: set[str] = set()
            identity_is_valid = True
            for author in authors_by_db_id.get(db_id, ()):
                author_id = str(author.get("author_id") or "").strip()
                author_name_key = normalize_visible_identity(
                    author.get("name")
                )
                if not author_id or not author_name_key:
                    identity_is_valid = False
                    break
                author_ids.add(
                    author_id
                    if author_id.startswith("author:")
                    else author_nid(author_id)
                )
                author_name_keys.add(author_name_key)
            if not identity_is_valid:
                continue
            results[paper_id] = PaperFacts(
                paper_id=paper_id,
                title_key=title_key,
                author_ids=frozenset(author_ids),
                author_name_keys=frozenset(author_name_keys),
                outgoing_paper_ids=(),
                incoming_paper_ids=(),
                citation_neighbor_ids=frozenset(),
            )
        return results

    def facts_with_adjacency(
        paper_id: str,
        identity: PaperFacts | None,
        adjacency: Sequence[tuple[str, Mapping[str, Any]]],
    ) -> PaperFacts | None:
        if identity is None:
            return None
        outgoing: set[str] = set()
        incoming: set[str] = set()
        for neighbor_id, attributes in adjacency:
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
            title_key=identity.title_key,
            author_ids=identity.author_ids,
            author_name_keys=identity.author_name_keys,
            outgoing_paper_ids=tuple(sorted(outgoing)),
            incoming_paper_ids=tuple(sorted(incoming)),
            citation_neighbor_ids=frozenset(outgoing | incoming),
        )

    def load_facts(paper_id: str) -> PaperFacts | None:
        return facts_with_adjacency(
            paper_id,
            reader.get_identity(paper_id),
            context.graph.adj.get(paper_id, []),
        )

    def load_facts_batch(
        paper_ids: Sequence[str],
    ) -> Mapping[str, PaperFacts | None]:
        normalized_ids = list(dict.fromkeys(str(value) for value in paper_ids))
        adjacency_loader = getattr(context.graph.adj, "get_many", None)
        if not callable(adjacency_loader):
            return {paper_id: load_facts(paper_id) for paper_id in normalized_ids}
        reader.prime_identities(normalized_ids)
        adjacency_by_id = adjacency_loader(normalized_ids)
        return {
            paper_id: facts_with_adjacency(
                paper_id,
                reader.get_identity(paper_id),
                adjacency_by_id.get(paper_id, ()),
            )
            for paper_id in normalized_ids
        }

    reader = PaperFactsReader(
        load_facts,
        identity_loader=load_identity,
        cache_max_entries=cache_max_entries,
        identity_cache_max_entries=identity_cache_max_entries,
        identity_batch_loader=load_identities,
        facts_batch_loader=load_facts_batch,
    )
    return reader


def negative_component_target(question: Mapping[str, Any]) -> int | None:
    """Return the integer component target for a v4 negative distractor."""
    reachability = question.get("reachability")
    if not isinstance(reachability, Mapping):
        return None
    growth = reachability.get(V4_GROWTH_FIELD)
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
    """Return the saved depth-zero reference profile for one v4 question."""
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
    """Rebuild v4 main components using only their rendered edges."""
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
            cache_key=f"question:{qid}:component:{component_index}:v4",
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
            cache_key=f"{entry_cache_key}:{orientation_index}:v4",
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


def legal_target_values(
    bucket_min: float,
    bucket_max: float,
    *,
    target_kind: str,
) -> list[float]:
    """Return every legal half-integer or integer value in ``[min, max)``."""
    if not math.isfinite(bucket_min) or not math.isfinite(bucket_max):
        raise ValueError("V4 complexity buckets must have finite bounds.")
    if bucket_min < 0 or bucket_max <= bucket_min:
        raise ValueError("A complexity bucket must satisfy 0 <= min < max.")
    scale = 2 if target_kind == TARGET_YES else 1
    first = math.ceil(bucket_min * scale)
    last = math.ceil(bucket_max * scale) - 1
    return [value / scale for value in range(first, last + 1)]


def select_uniform_target(
    values: Sequence[float],
    *,
    random_seed: int | None,
    key: Any,
) -> float:
    """Choose one exact target uniformly with a stable independent seed."""
    if not values:
        raise ValueError("Cannot sample from an empty target set.")
    rng = random.Random(stable_growth_seed(random_seed, ("target", key)))
    return float(rng.choice(list(values)))


def record_accepted_question(
    accepted_since_checkpoint: int,
    *,
    checkpoint_every: int,
    progress_update: Callable[[int], Any],
    write_outputs: Callable[[], None],
    write_pair_cache: Callable[[], None],
) -> int:
    """Record one acceptance and checkpoint after each complete interval."""
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive.")
    progress_update(1)
    accepted_since_checkpoint += 1
    if accepted_since_checkpoint < checkpoint_every:
        return accepted_since_checkpoint
    write_outputs()
    write_pair_cache()
    return 0


def pairwise_thin_disjointness_reason(
    trees: Sequence[GrowthTreeState],
    *,
    facts_reader: PaperFactsReader,
    include_author_identity: bool,
    cache: PairDisjointnessCache | None,
) -> str | None:
    """Check and cache permanent overlap among thin input trees."""
    for first_index, first in enumerate(trees):
        for second in trees[first_index + 1 :]:
            if first.path_id == second.path_id:
                return "same_input_path"
            cached = (
                cache.get(first.cache_key, second.cache_key)
                if cache is not None
                else None
            )
            if cached is not None:
                if cached != "compatible":
                    return cached
                continue
            # Thin trees already carry validated identity registries. Comparing
            # them is equivalent to rebuilding a two-tree group, without copies.
            reason = None
            if not first.visible_paper_ids.isdisjoint(second.visible_paper_ids):
                reason = "shared_paper_ids"
            elif not first.used_title_keys.isdisjoint(second.used_title_keys):
                reason = "shared_normalized_paper_titles"
            elif include_author_identity and not first.author_owner_by_id.keys().isdisjoint(
                second.author_owner_by_id
            ):
                reason = "shared_author_ids"
            elif include_author_identity and not first.author_owner_by_name.keys().isdisjoint(
                second.author_owner_by_name
            ):
                reason = "shared_normalized_author_names"
            elif (
                not first.citation_neighbor_ids.isdisjoint(second.visible_paper_ids)
                or not second.citation_neighbor_ids.isdisjoint(first.visible_paper_ids)
            ):
                reason = "citation_edge_crosses_trees"
            if cache is not None:
                cache.put(
                    first.cache_key,
                    second.cache_key,
                    reason or "compatible",
                )
            if reason is not None:
                return reason
    return None


def intersect_ladder_values(ladders: Sequence[Any]) -> set[float]:
    """Return exact complexity values shared by all pruning ladders."""
    common = set(ladders[0].values)
    for ladder in ladders[1:]:
        common.intersection_update(ladder.values)
    return common


def exhausted_growth_can_be_reused(result: GrowthGroupResult) -> bool:
    """Return whether a failed direct run is also a valid saturated run."""
    return not result.all_targets_reached and not any(
        result.rejection_counts.get(reason, 0)
        for reason in TARGET_LIMIT_REJECTION_REASONS
    )


def final_plan_from_ladders(
    trees: Sequence[GrowthTreeState],
    target_kinds: Sequence[str],
    *,
    bucket_specs: Sequence[tuple[str, float, float]],
    facts_reader: PaperFactsReader,
    random_seed: int | None,
    key: Any,
    used_fallback: bool,
    reused_higher_bucket_tree: bool,
    maximum_depth_processed: int,
    rejection_counts: Mapping[str, int],
) -> GrowthPlan | None:
    """Choose the hardest feasible bucket and apply a shared exact target."""
    ladders = [
        build_pruning_ladder(
            tree,
            target_kind=target_kind,
            facts_reader=facts_reader,
            random_seed=random_seed,
            key=(key, tree.cache_key),
        )
        for tree, target_kind in zip(trees, target_kinds)
    ]
    common_values = intersect_ladder_values(ladders)
    for label, bucket_min, bucket_max in yesno_base.hardest_bucket_specs(
        bucket_specs
    ):
        feasible_values = sorted(
            value
            for value in common_values
            if bucket_min <= value < bucket_max
        )
        if not feasible_values:
            continue
        target_value = select_uniform_target(
            feasible_values,
            random_seed=random_seed,
            key=(key, "fallback", label),
        )
        final_trees = tuple(
            apply_pruning_prefix(
                tree,
                ladder,
                ladder.prefix_by_value[target_value],
                facts_reader=facts_reader,
            )
            for tree, ladder in zip(trees, ladders)
        )
        if not all(
            validate_rendered_growth_tree(
                tree,
                facts_reader=facts_reader,
                include_author_identity=bool(
                    tree.author_owner_by_id or tree.author_owner_by_name
                ),
            )
            for tree in final_trees
        ):
            return None
        return GrowthPlan(
            trees=final_trees,
            target_kinds=tuple(target_kinds),
            target_value=target_value,
            bucket_label=label,
            calibration=reference_calibration(final_trees),
            used_fallback=used_fallback,
            reused_higher_bucket_tree=reused_higher_bucket_tree,
            removed_paper_counts=tuple(
                ladder.prefix_by_value[target_value] for ladder in ladders
            ),
            removed_reference_pairs=tuple(
                tuple(
                    tree.discovery_pair_by_paper[paper_id]
                    for paper_id in ladder.removal_order[
                        : ladder.prefix_by_value[target_value]
                    ]
                )
                for tree, ladder in zip(trees, ladders)
            ),
            maximum_depth_processed=maximum_depth_processed,
            rejection_counts=dict(rejection_counts),
        )
    return None


def plan_new_growth_group(
    initial_trees: Sequence[GrowthTreeState],
    target_kinds: Sequence[str],
    *,
    sampled_target: float,
    current_bucket: tuple[str, float, float],
    fallback_bucket_specs: Sequence[tuple[str, float, float]],
    facts_reader: PaperFactsReader,
    include_author_identity: bool,
    random_seed: int | None,
    key: Any,
    progress_callback: Callable[[], None] | None = None,
) -> GrowthPlan | None:
    """Try the sampled target, then saturate and prune on failure."""
    targets = [
        GrowthTarget(target_kind, sampled_target)
        for target_kind in target_kinds
    ]
    direct = grow_question_group(
        initial_trees,
        targets,
        facts_reader=facts_reader,
        include_author_identity=include_author_identity,
        random_seed=random_seed,
        group_key=(key, "growth"),
        progress_callback=progress_callback,
    )
    if direct is not None and direct.all_targets_reached:
        if not all(
            validate_rendered_growth_tree(
                tree,
                facts_reader=facts_reader,
                include_author_identity=include_author_identity,
            )
            for tree in direct.trees
        ):
            return None
        return GrowthPlan(
            trees=direct.trees,
            target_kinds=tuple(target_kinds),
            target_value=sampled_target,
            bucket_label=current_bucket[0],
            calibration=direct.calibration,
            used_fallback=False,
            reused_higher_bucket_tree=False,
            removed_paper_counts=tuple(0 for _tree in direct.trees),
            removed_reference_pairs=tuple(() for _tree in direct.trees),
            maximum_depth_processed=direct.maximum_depth_processed,
            rejection_counts=direct.rejection_counts,
        )

    saturation_values = legal_target_values(
        current_bucket[1],
        current_bucket[2],
        target_kind=target_kinds[0],
    )
    if not saturation_values:
        return None
    saturation_target = max(saturation_values)
    saturated: GrowthGroupResult | None
    if direct is not None and (
        math.isclose(sampled_target, saturation_target)
        or exhausted_growth_can_be_reused(direct)
    ):
        saturated = direct
    else:
        saturated = grow_question_group(
            initial_trees,
            [
                GrowthTarget(target_kind, saturation_target)
                for target_kind in target_kinds
            ],
            facts_reader=facts_reader,
            include_author_identity=include_author_identity,
            random_seed=random_seed,
            group_key=(key, "growth"),
            progress_callback=progress_callback,
        )
    if saturated is None:
        return None
    return final_plan_from_ladders(
        saturated.trees,
        target_kinds,
        bucket_specs=fallback_bucket_specs,
        facts_reader=facts_reader,
        random_seed=random_seed,
        key=key,
        used_fallback=True,
        reused_higher_bucket_tree=False,
        maximum_depth_processed=saturated.maximum_depth_processed,
        rejection_counts=saturated.rejection_counts,
    )


def plan_reused_growth_group(
    reusable: ReusableGrowthGroup,
    *,
    bucket_spec: tuple[str, float, float],
    facts_reader: PaperFactsReader,
    random_seed: int | None,
    key: Any,
) -> GrowthPlan | None:
    """Prune one accepted higher-bucket group into an exact smaller bucket."""
    return final_plan_from_ladders(
        reusable.trees,
        reusable.target_kinds,
        bucket_specs=[bucket_spec],
        facts_reader=facts_reader,
        random_seed=random_seed,
        key=key,
        used_fallback=False,
        reused_higher_bucket_tree=True,
        maximum_depth_processed=max(
            (max(tree.paper_depth.values(), default=0) for tree in reusable.trees),
            default=0,
        ),
        rejection_counts={},
    )


def role_wrong_paths(tree: GrowthTreeState, tree_role: str) -> list[list[str]]:
    """Return the legacy role-specific audit paths for one full gold chain."""
    if tree_role == yesno_base.TREE_ROLE_POSITIVE:
        return [list(tree.source_wrong_path), list(tree.target_wrong_path)]
    if tree_role == yesno_base.TREE_ROLE_SOURCE_NEGATIVE:
        return [list(tree.source_wrong_path)]
    if tree_role == yesno_base.TREE_ROLE_TARGET_NEGATIVE:
        return [list(tree.target_wrong_path)]
    raise ValueError(f"Unsupported v4 tree role: {tree_role!r}.")


def growth_state_variant(
    tree: GrowthTreeState,
    *,
    tree_role: str,
    facts_reader: PaperFactsReader,
    include_author_identity: bool,
    removed_paper_count: int = 0,
    removed_reference_pairs: Sequence[tuple[str, str]] = (),
) -> TreeBucketVariant:
    """Convert one final v4 state into the existing audited variant format."""
    if not validate_rendered_growth_tree(
        tree,
        facts_reader=facts_reader,
        include_author_identity=include_author_identity,
    ):
        raise ValueError("Final v4 rendered growth state is invalid.")
    paper_ids = [
        paper_id
        for paper_id in tree.expanded_paper_order
        if paper_id in tree.expanded_paper_ids
    ]
    authors_by_paper = {
        paper_id: sorted(tree.authors_by_paper.get(paper_id, set()))
        for paper_id in paper_ids
    }
    references_by_paper = {
        paper_id: sorted(tree.references_by_paper.get(paper_id, set()))
        for paper_id in paper_ids
    }
    title_key_by_paper: dict[str, str] = {}
    author_ids_by_paper: dict[str, list[str]] = {}
    author_name_keys_by_paper: dict[str, list[str]] = {}
    citation_neighbors_by_paper: dict[str, list[str]] = {}
    for paper_id in sorted(tree.visible_paper_ids):
        facts = facts_reader.get_identity(paper_id)
        if facts is None:
            raise ValueError(f"Missing facts for final paper {paper_id!r}.")
        title_key_by_paper[paper_id] = facts.title_key
        author_ids_by_paper[paper_id] = (
            sorted(facts.author_ids) if include_author_identity else []
        )
        author_name_keys_by_paper[paper_id] = (
            sorted(facts.author_name_keys) if include_author_identity else []
        )
        citation_neighbors_by_paper[paper_id] = []
    closure_author_ids = frozenset(
        author_id
        for author_ids in author_ids_by_paper.values()
        for author_id in author_ids
    )
    closure_author_name_keys = frozenset(
        name_key
        for name_keys in author_name_keys_by_paper.values()
        for name_key in name_keys
    )
    internal_reference_pairs = {
        (owner_id, cited_id)
        for owner_id, cited_ids in references_by_paper.items()
        for cited_id in cited_ids
        if cited_id in tree.expanded_paper_ids
    }
    protected_pairs = sorted(
        set(tree.protected_reference_pairs) | internal_reference_pairs
    )
    wrong_paths = role_wrong_paths(tree, tree_role)
    cache_key = f"{tree.cache_key}:{tree_role}"
    cached_tree = CachedSearchTree(
        cache_key=cache_key,
        path_id=tree.path_id,
        orientation_index=tree.orientation_index,
        source_node_id=tree.source_node_id,
        target_node_id=tree.target_node_id,
        paper_ids=paper_ids,
        authors_by_paper=authors_by_paper,
        references_by_paper=references_by_paper,
        protected_reference_pairs=protected_pairs,
        witness=tree.witness,
        closure_paper_ids=frozenset(tree.visible_paper_ids),
        closure_author_ids=closure_author_ids,
        closure_title_keys=frozenset(title_key_by_paper.values()),
        closure_author_name_keys=closure_author_name_keys,
        citation_neighbor_paper_ids=frozenset(),
        backbone_paper_ids=[
            node_id
            for node_id in tree.witness
            if nid_type(node_id) == NODE_PAPER
        ],
        branch_parent_by_paper=dict(tree.parent_by_paper),
        branch_depth_by_paper=dict(tree.paper_depth),
        optional_reference_pairs=sorted(tree.optional_reference_pairs),
        title_key_by_paper=title_key_by_paper,
        author_ids_by_paper_identity=author_ids_by_paper,
        author_name_keys_by_paper=author_name_keys_by_paper,
        citation_neighbors_by_paper=citation_neighbors_by_paper,
        tree_role=tree_role,
        gold_depth=len(tree.witness) - 1,
        wrong_branch_paths=wrong_paths,
    )
    signature_payload = {
        "cache_key": cache_key,
        "paper_ids": paper_ids,
        "authors_by_paper": authors_by_paper,
        "references_by_paper": references_by_paper,
    }
    signature = hashlib.sha256(
        orjson.dumps(signature_payload, option=orjson.OPT_SORT_KEYS)
    ).hexdigest()
    return TreeBucketVariant(
        tree=cached_tree,
        references_by_paper=references_by_paper,
        yes_complexity=tree.yes_complexity(),
        component_size=tree.component_size,
        removed_reference_count=removed_paper_count,
        paper_ids=paper_ids,
        authors_by_paper=authors_by_paper,
        closure_paper_ids=frozenset(tree.visible_paper_ids),
        closure_author_ids=closure_author_ids,
        closure_title_keys=frozenset(title_key_by_paper.values()),
        closure_author_name_keys=closure_author_name_keys,
        citation_neighbor_paper_ids=frozenset(),
        removed_reference_pairs=[
            (str(owner_id), str(cited_id))
            for owner_id, cited_id in removed_reference_pairs
        ],
        variant_signature=signature,
    )


def combine_growth_variants(
    variants: Sequence[TreeBucketVariant],
) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    """Combine already-disjoint final tree mappings."""
    paper_ids: list[str] = []
    authors_by_paper: dict[str, list[str]] = {}
    references_by_paper: dict[str, list[str]] = {}
    for variant in variants:
        paper_ids.extend(variant.paper_ids or variant.tree.paper_ids)
        authors_by_paper.update(
            variant.authors_by_paper or variant.tree.authors_by_paper
        )
        references_by_paper.update(variant.references_by_paper)
    return paper_ids, authors_by_paper, references_by_paper


def negative_distractor_tree_signature(payload: Mapping[str, Any]) -> str:
    """Return the content signature of one serialized v4 distractor tree."""
    signed_payload = {
        str(key): value for key, value in payload.items() if key != "tree_id"
    }
    return hashlib.sha256(
        orjson.dumps(signed_payload, option=orjson.OPT_SORT_KEYS)
    ).hexdigest()


def serialize_negative_distractor_tree(
    tree: GrowthTreeState,
    *,
    target_component_size: int,
) -> dict[str, Any]:
    """Serialize one complete rendered-edge negative component."""
    expanded = [
        paper_id
        for paper_id in tree.expanded_paper_order
        if paper_id in tree.expanded_paper_ids
    ]
    payload: dict[str, Any] = {
        "version": V4_NEGATIVE_DISTRACTOR_TREE_VERSION,
        "constructor_version": "v4",
        "edge_scope": "rendered_only",
        "tree_construction_strategy": V4_TREE_CONSTRUCTION_STRATEGY,
        "paper_identity_policy": V4_PAPER_IDENTITY_POLICY,
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
        "sampled_reference_caps": dict(
            sorted(tree.sampled_caps_by_paper.items())
        ),
        "component_size": tree.component_size,
        "target_component_size": target_component_size,
    }
    payload["tree_id"] = negative_distractor_tree_signature(payload)
    return payload


def deserialize_negative_distractor_tree(
    payload: Mapping[str, Any],
    *,
    facts_reader: PaperFactsReader,
    include_author_identity: bool,
) -> GrowthTreeState | None:
    """Rebuild one signed v4 distractor using rendered-edge semantics."""
    if (
        payload.get("version") != V4_NEGATIVE_DISTRACTOR_TREE_VERSION
        or payload.get("constructor_version") != "v4"
        or payload.get("edge_scope") != "rendered_only"
        or payload.get("tree_construction_strategy")
        != V4_TREE_CONSTRUCTION_STRATEGY
        or payload.get("paper_identity_policy") != V4_PAPER_IDENTITY_POLICY
        or payload.get("tree_id") != negative_distractor_tree_signature(payload)
    ):
        return None
    expanded = payload.get("expanded_paper_ids")
    authors = payload.get("authors_by_paper")
    references = payload.get("references_by_paper")
    protected = payload.get("protected_reference_pairs")
    if (
        not isinstance(expanded, list)
        or not expanded
        or not all(isinstance(value, str) for value in expanded)
        or len(set(expanded)) != len(expanded)
        or not isinstance(authors, Mapping)
        or not isinstance(references, Mapping)
        or set(authors) != set(expanded)
        or set(references) != set(expanded)
        or not all(isinstance(value, list) for value in authors.values())
        or not all(isinstance(value, list) for value in references.values())
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
            expanded_paper_ids=expanded,
            authors_by_paper={
                str(paper_id): [str(value) for value in author_ids]
                for paper_id, author_ids in authors.items()
            },
            references_by_paper={
                str(paper_id): [str(value) for value in cited_ids]
                for paper_id, cited_ids in references.items()
            },
            protected_reference_pairs=[
                (str(pair[0]), str(pair[1]))
                for pair in protected
                if isinstance(pair, list) and len(pair) == 2
            ],
            facts_reader=facts_reader,
            include_author_identity=include_author_identity,
        )
        expected_size = int(payload.get("component_size"))
        target_size = int(payload.get("target_component_size"))
    except (TypeError, ValueError):
        return None
    if state is None:
        return None
    if state.component_size != expected_size or expected_size != target_size:
        return None
    return state


def add_growth_metadata(record: dict[str, Any], plan: GrowthPlan) -> None:
    """Attach compact v4 growth and calibration audit metadata in place."""
    record[DEPTH_ZERO_REFERENCE_STATISTICS_FIELD] = (
        plan.calibration.as_dict()
    )
    reachability = record.setdefault("reachability", {})
    visible_graph = reachability.get("visible_graph")
    if isinstance(visible_graph, dict):
        visible_graph["allow_hidden_citations"] = True
    reachability[V4_GROWTH_FIELD] = {
        "constructor_version": "v4",
        "strategy": V4_TREE_CONSTRUCTION_STRATEGY,
        "edge_scope": "rendered_only",
        "rendered_graph_policy_version": V4_RENDERED_GRAPH_POLICY_VERSION,
        "paper_identity_policy": V4_PAPER_IDENTITY_POLICY,
        "candidate_commit_policy": V4_CANDIDATE_COMMIT_POLICY,
        "target_value": plan.target_value,
        "target_kinds": list(plan.target_kinds),
        "allocation_bucket": plan.bucket_label,
        "used_maximal_fallback": plan.used_fallback,
        "reused_higher_bucket_tree": plan.reused_higher_bucket_tree,
        "maximum_depth_processed": plan.maximum_depth_processed,
        "pruned_paper_counts": list(plan.removed_paper_counts),
        "rejection_counts": dict(plan.rejection_counts),
        "depth_zero_statistics_scope": "all_question_components",
        "growth_history_stored": False,
    }


def plan_signature(plan: GrowthPlan) -> str:
    """Return a stable identity for one exact final state group."""
    payload = {
        "bucket": plan.bucket_label,
        "target": plan.target_value,
        "trees": [
            {
                "cache_key": tree.cache_key,
                "papers": sorted(tree.expanded_paper_ids),
                "references": {
                    owner: sorted(cited_ids)
                    for owner, cited_ids in sorted(
                        tree.references_by_paper.items()
                    )
                },
            }
            for tree in plan.trees
        ],
    }
    return hashlib.sha256(
        orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    ).hexdigest()
