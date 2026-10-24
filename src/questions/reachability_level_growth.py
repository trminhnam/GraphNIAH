"""Target-driven, depth-level growth for v3 reachability questions.

The engine in this module is deliberately independent from file orchestration
and rendering.  It grows one or more already validated thin trees, one depth
level at a time, and keeps only the current tree and current frontier state.
No chronological growth log is retained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import orjson
import math
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

import networkx as nx

from src.questions.reachability_yesno import (
    build_visible_graph,
    rooted_expected_dfs_steps,
)


DIRECTION_CITES = "cites"
DIRECTION_CITED_BY = "cited_by"
TARGET_YES = "yes"
TARGET_COMPONENT = "component"
TARGET_MATCHING = "matching"
SUPPORTED_TARGET_KINDS = {
    TARGET_YES,
    TARGET_COMPONENT,
    TARGET_MATCHING,
}
FACTS_CACHE_MISS = object()


def facts_cache_get(cache: dict[str, Any], key: str) -> Any:
    """Return and refresh one cached fact, or ``FACTS_CACHE_MISS``."""
    if key not in cache:
        return FACTS_CACHE_MISS
    value = cache.pop(key)
    cache[key] = value
    return value


def facts_cache_put(
    cache: dict[str, Any],
    key: str,
    value: Any,
    max_entries: int | None,
) -> None:
    """Insert one fact and evict the least recently used entry if needed."""
    cache.pop(key, None)
    cache[key] = value
    if max_entries is not None and len(cache) > max_entries:
        cache.pop(next(iter(cache)))


def stable_growth_seed(random_seed: int | None, key: Any) -> int:
    """Return a stable seed for one independent v3 random decision."""
    encoded = orjson.dumps([random_seed, key], option=orjson.OPT_SORT_KEYS, default=str)
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class PaperFacts:
    """Immutable visible identity and citation data for one paper."""

    paper_id: str
    title_key: str
    author_ids: frozenset[str]
    author_name_keys: frozenset[str]
    outgoing_paper_ids: tuple[str, ...]
    incoming_paper_ids: tuple[str, ...]
    citation_neighbor_ids: frozenset[str]


@dataclass(slots=True)
class PaperFactsReader:
    """Memoize paper facts loaded lazily from the graph and database."""

    loader: Callable[[str], PaperFacts | None]
    cache: dict[str, PaperFacts | None] = field(default_factory=dict)
    identity_loader: Callable[[str], PaperFacts | None] | None = None
    identity_cache: dict[str, PaperFacts | None] = field(default_factory=dict)
    cache_max_entries: int | None = None
    identity_cache_max_entries: int | None = None
    identity_batch_loader: (
        Callable[[Sequence[str]], Mapping[str, PaperFacts | None]] | None
    ) = None
    facts_batch_loader: (
        Callable[[Sequence[str]], Mapping[str, PaperFacts | None]] | None
    ) = None

    def get(self, paper_id: str) -> PaperFacts | None:
        """Return cached facts for ``paper_id``."""
        normalized_id = str(paper_id)
        cached = facts_cache_get(self.cache, normalized_id)
        if cached is FACTS_CACHE_MISS:
            cached = self.loader(normalized_id)
            facts_cache_put(
                self.cache,
                normalized_id,
                cached,
                self.cache_max_entries,
            )
        if (
            self.identity_loader is None
            and facts_cache_get(self.identity_cache, normalized_id)
            is FACTS_CACHE_MISS
        ):
            facts_cache_put(
                self.identity_cache,
                normalized_id,
                cached,
                self.identity_cache_max_entries,
            )
        return cached

    def get_identity(self, paper_id: str) -> PaperFacts | None:
        """Return identity facts without requiring citation adjacency."""
        if self.identity_loader is None:
            return self.get(paper_id)
        normalized_id = str(paper_id)
        cached = facts_cache_get(self.identity_cache, normalized_id)
        if cached is FACTS_CACHE_MISS:
            cached = self.identity_loader(normalized_id)
            facts_cache_put(
                self.identity_cache,
                normalized_id,
                cached,
                self.identity_cache_max_entries,
            )
        return cached

    def prime_identities(self, paper_ids: Sequence[str]) -> None:
        """Load uncached identities together when batch loading is available."""
        if self.identity_batch_loader is None:
            return
        missing_ids: list[str] = []
        seen_ids: set[str] = set()
        for paper_id in paper_ids:
            normalized_id = str(paper_id)
            if (
                normalized_id not in seen_ids
                and normalized_id not in self.identity_cache
            ):
                seen_ids.add(normalized_id)
                missing_ids.append(normalized_id)
        if not missing_ids:
            return
        loaded = self.identity_batch_loader(missing_ids)
        for paper_id in missing_ids:
            facts_cache_put(
                self.identity_cache,
                paper_id,
                loaded.get(paper_id),
                self.identity_cache_max_entries,
            )

    def prime_facts(self, paper_ids: Sequence[str]) -> None:
        """Load uncached expansion facts together when batching is available."""
        if self.facts_batch_loader is None:
            return
        missing_ids: list[str] = []
        seen_ids: set[str] = set()
        for paper_id in paper_ids:
            normalized_id = str(paper_id)
            if normalized_id not in seen_ids and normalized_id not in self.cache:
                seen_ids.add(normalized_id)
                missing_ids.append(normalized_id)
        if not missing_ids:
            return
        loaded = self.facts_batch_loader(missing_ids)
        for paper_id in missing_ids:
            facts_cache_put(
                self.cache,
                paper_id,
                loaded.get(paper_id),
                self.cache_max_entries,
            )


@dataclass(frozen=True, slots=True)
class ReferenceCalibration:
    """Frozen reference-count profile measured from depth-zero blocks."""

    paper_node_ids: tuple[str, ...]
    citation_counts: tuple[int, ...]
    mean: float
    std: float

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible calibration record."""
        return {
            "version": 1,
            "scope": "all_question_components",
            "paper_count": len(self.paper_node_ids),
            "citation_counts": list(self.citation_counts),
            "mean": self.mean,
            "std": self.std,
            "std_type": "population",
            "source_depth": 0,
        }


@dataclass(frozen=True, slots=True)
class GrowthTarget:
    """One exact target for a tree in a coordinated growth group."""

    kind: str
    value: float

    def validate(self) -> None:
        """Raise when this target does not use a supported exact scale."""
        if self.kind not in SUPPORTED_TARGET_KINDS:
            raise ValueError(f"Unsupported growth target kind: {self.kind!r}.")
        if not math.isfinite(float(self.value)) or float(self.value) < 0:
            raise ValueError("Growth target values must be finite and non-negative.")
        if self.kind == TARGET_YES:
            doubled = float(self.value) * 2.0
            if not math.isclose(doubled, round(doubled)):
                raise ValueError("YES targets must use integer or half-integer values.")
        elif not float(self.value).is_integer():
            raise ValueError("Component and matching targets must be integers.")


@dataclass(slots=True)
class PaperExpansionState:
    """Round-robin state kept only while one paper's level is active."""

    tree_index: int
    paper_id: str
    depth: int
    outgoing_candidates: list[str]
    incoming_candidates: list[str]
    reference_cap: int | None
    attempts: int = 0
    accepted_outgoing: int = 0
    accepted_incoming: int = 0
    retired_reason: str = ""

    def current_reference_count(self, tree: "GrowthTreeState") -> int:
        """Return the references currently owned by this paper block."""
        return len(tree.references_by_paper.get(self.paper_id, set()))

    def has_outgoing_capacity(self, tree: "GrowthTreeState") -> bool:
        """Return whether the paper may receive another scheduling turn."""
        if not self.outgoing_candidates:
            return False
        if self.reference_cap is None:
            return True
        return self.current_reference_count(tree) < self.reference_cap

    def retire_if_needed(self, tree: "GrowthTreeState") -> bool:
        """Set the retirement reason and return whether the paper is retired."""
        if self.reference_cap is not None and (
            self.current_reference_count(tree) >= self.reference_cap
        ):
            self.retired_reason = "reference_cap_reached"
            return True
        if not self.outgoing_candidates:
            self.retired_reason = "outgoing_candidates_exhausted"
            return True
        return False


@dataclass(slots=True)
class GrowthTreeState:
    """Compact current state for one growing visible search tree."""

    cache_key: str
    path_id: str
    orientation_index: int
    source_node_id: str
    target_node_id: str
    witness: tuple[str, ...]
    source_wrong_path: tuple[str, ...]
    target_wrong_path: tuple[str, ...]
    gold_paper_ids: tuple[str, ...]
    initial_expanded_paper_ids: set[str]
    expanded_paper_order: list[str]
    expanded_paper_seen_ids: set[str]
    expanded_paper_ids: set[str]
    visible_paper_ids: set[str]
    graph_author_ids: set[str]
    authors_by_paper: dict[str, set[str]]
    references_by_paper: dict[str, set[str]]
    protected_reference_pairs: set[tuple[str, str]]
    optional_reference_pairs: set[tuple[str, str]]
    paper_depth: dict[str, int]
    parent_by_paper: dict[str, str]
    child_count_by_paper: dict[str, int]
    discovery_pair_by_paper: dict[str, tuple[str, str]]
    anchor_index_by_node: dict[str, int]
    source_cost_units: int
    target_cost_units: int
    component_size: int
    title_owner_by_key: dict[str, str]
    used_title_keys: set[str]
    author_owner_by_id: dict[str, str]
    author_owner_by_name: dict[str, str]
    citation_neighbor_ids: set[str]
    citation_neighbor_count_by_id: dict[str, int]
    sampled_caps_by_paper: dict[str, int] = field(default_factory=dict)
    achieved_counts_by_paper: dict[str, int] = field(default_factory=dict)
    retirement_reason_by_paper: dict[str, str] = field(default_factory=dict)
    processed_center_ids: set[str] = field(default_factory=set)
    # Rendered (owner, cited) pairs that close a cycle, with the growth center
    # that drew them (create_dataset v3/v4 only; empty for a tree).
    cycle_center_by_pair: dict[tuple[str, str], str] = field(default_factory=dict)

    def clone(self) -> "GrowthTreeState":
        """Return an independent mutable copy without a growth-history log."""
        return GrowthTreeState(
            cache_key=self.cache_key,
            path_id=self.path_id,
            orientation_index=self.orientation_index,
            source_node_id=self.source_node_id,
            target_node_id=self.target_node_id,
            witness=self.witness,
            source_wrong_path=self.source_wrong_path,
            target_wrong_path=self.target_wrong_path,
            gold_paper_ids=self.gold_paper_ids,
            initial_expanded_paper_ids=set(self.initial_expanded_paper_ids),
            expanded_paper_order=list(self.expanded_paper_order),
            expanded_paper_seen_ids=set(self.expanded_paper_seen_ids),
            expanded_paper_ids=set(self.expanded_paper_ids),
            visible_paper_ids=set(self.visible_paper_ids),
            graph_author_ids=set(self.graph_author_ids),
            authors_by_paper={
                paper_id: set(author_ids)
                for paper_id, author_ids in self.authors_by_paper.items()
            },
            references_by_paper={
                paper_id: set(reference_ids)
                for paper_id, reference_ids in self.references_by_paper.items()
            },
            protected_reference_pairs=set(self.protected_reference_pairs),
            optional_reference_pairs=set(self.optional_reference_pairs),
            paper_depth=dict(self.paper_depth),
            parent_by_paper=dict(self.parent_by_paper),
            child_count_by_paper=dict(self.child_count_by_paper),
            discovery_pair_by_paper=dict(self.discovery_pair_by_paper),
            anchor_index_by_node=dict(self.anchor_index_by_node),
            source_cost_units=self.source_cost_units,
            target_cost_units=self.target_cost_units,
            component_size=self.component_size,
            title_owner_by_key=dict(self.title_owner_by_key),
            used_title_keys=set(self.used_title_keys),
            author_owner_by_id=dict(self.author_owner_by_id),
            author_owner_by_name=dict(self.author_owner_by_name),
            citation_neighbor_ids=set(self.citation_neighbor_ids),
            citation_neighbor_count_by_id=dict(
                self.citation_neighbor_count_by_id
            ),
            sampled_caps_by_paper=dict(self.sampled_caps_by_paper),
            achieved_counts_by_paper=dict(self.achieved_counts_by_paper),
            retirement_reason_by_paper=dict(self.retirement_reason_by_paper),
            processed_center_ids=set(self.processed_center_ids),
            cycle_center_by_pair=dict(self.cycle_center_by_pair),
        )

    def yes_complexity(self) -> float:
        """Return the current exact symmetric expected DFS complexity."""
        return float(min(self.source_cost_units, self.target_cost_units)) / 2.0

    def target_value(self, kind: str) -> float:
        """Return this tree's current value on one target scale."""
        if kind == TARGET_YES:
            return self.yes_complexity()
        if kind == TARGET_COMPONENT:
            return float(self.component_size)
        if kind == TARGET_MATCHING:
            return float(math.ceil(self.yes_complexity()))
        raise ValueError(f"Unsupported growth target kind: {kind!r}.")

    def target_reached(self, target: GrowthTarget) -> bool:
        """Return whether the current tree exactly reaches ``target``."""
        return math.isclose(self.target_value(target.kind), float(target.value))

    def target_exceeded(self, target: GrowthTarget) -> bool:
        """Return whether the current tree is already above ``target``."""
        return self.target_value(target.kind) > float(target.value)


@dataclass(slots=True)
class GrowthBlockers:
    """Compact identities and citation boundaries of fixed disjoint trees."""

    paper_ids: set[str] = field(default_factory=set)
    title_keys: set[str] = field(default_factory=set)
    author_ids: set[str] = field(default_factory=set)
    author_name_keys: set[str] = field(default_factory=set)
    citation_neighbor_ids: set[str] = field(default_factory=set)

    def conflict_reason(
        self,
        tree: GrowthTreeState,
        *,
        include_author_identity: bool,
    ) -> str | None:
        """Return why ``tree`` is not disjoint from the fixed trees."""
        if self.paper_ids.intersection(tree.visible_paper_ids):
            return "shared_paper_ids"
        if self.title_keys.intersection(tree.title_owner_by_key):
            return "shared_normalized_paper_titles"
        if include_author_identity:
            if self.author_ids.intersection(tree.author_owner_by_id):
                return "shared_author_ids"
            if self.author_name_keys.intersection(tree.author_owner_by_name):
                return "shared_normalized_author_names"
        if self.paper_ids.intersection(
            tree.citation_neighbor_ids
        ) or self.citation_neighbor_ids.intersection(tree.visible_paper_ids):
            return "citation_edge_crosses_trees"
        return None

    def add_tree(
        self,
        tree: GrowthTreeState,
        *,
        include_author_identity: bool,
    ) -> str | None:
        """Add one disjoint tree to the fixed registry."""
        reason = self.conflict_reason(
            tree,
            include_author_identity=include_author_identity,
        )
        if reason is not None:
            return reason
        self.paper_ids.update(tree.visible_paper_ids)
        self.title_keys.update(tree.title_owner_by_key)
        if include_author_identity:
            self.author_ids.update(tree.author_owner_by_id)
            self.author_name_keys.update(tree.author_owner_by_name)
        self.citation_neighbor_ids.update(tree.citation_neighbor_ids)
        return None


@dataclass(frozen=True, slots=True)
class GrowthAddition:
    """One fully validated atomic mutation ready to commit."""

    tree_index: int
    center_paper_id: str
    candidate_paper_id: str
    direction: str
    reference_pair: tuple[str, str]
    candidate_facts: PaperFacts
    promote_center: bool
    center_author_ids: frozenset[str]
    added_graph_node_count: int
    anchor_index: int
    depth: int
    resulting_source_cost_units: int
    resulting_target_cost_units: int
    resulting_component_size: int


@dataclass(slots=True)
class QuestionGrowthGroup:
    """Coordinate exact targets and pairwise disjointness for one question."""

    trees: list[GrowthTreeState]
    targets: list[GrowthTarget]
    facts_reader: PaperFactsReader
    include_author_identity: bool
    blockers: GrowthBlockers | None = None
    paper_owner: dict[str, int] = field(default_factory=dict)
    title_owner: dict[str, tuple[int, str]] = field(default_factory=dict)
    author_id_owner: dict[str, tuple[int, str]] = field(default_factory=dict)
    author_name_owner: dict[str, tuple[int, str]] = field(default_factory=dict)
    used_title_keys: set[str] = field(default_factory=set)

    def rebuild_registries(self) -> str | None:
        """Build global ownership maps and return the first conflict reason."""
        self.paper_owner.clear()
        self.title_owner.clear()
        self.author_id_owner.clear()
        self.author_name_owner.clear()
        self.used_title_keys.clear()
        for tree_index, tree in enumerate(self.trees):
            for paper_id in tree.visible_paper_ids:
                previous_tree = self.paper_owner.get(paper_id)
                if previous_tree is not None and previous_tree != tree_index:
                    return "shared_paper_ids"
                self.paper_owner[paper_id] = tree_index
            tree_title_keys = tree.used_title_keys
            if not self.used_title_keys.isdisjoint(tree_title_keys):
                return "shared_normalized_paper_titles"
            self.used_title_keys.update(tree_title_keys)
            for title_key, paper_id in tree.title_owner_by_key.items():
                self.title_owner[title_key] = (tree_index, paper_id)
            if self.include_author_identity:
                for author_id, paper_id in tree.author_owner_by_id.items():
                    previous = self.author_id_owner.get(author_id)
                    if previous is not None and previous[0] != tree_index:
                        return "shared_author_ids"
                    self.author_id_owner[author_id] = (tree_index, paper_id)
                for name_key, paper_id in tree.author_owner_by_name.items():
                    previous = self.author_name_owner.get(name_key)
                    if previous is not None and previous[0] != tree_index:
                        return "shared_normalized_author_names"
                    self.author_name_owner[name_key] = (tree_index, paper_id)

        for first_index, first in enumerate(self.trees):
            for second in self.trees[first_index + 1 :]:
                if first.citation_neighbor_ids.intersection(
                    second.visible_paper_ids
                ) or second.citation_neighbor_ids.intersection(
                    first.visible_paper_ids
                ):
                    return "citation_edge_crosses_trees"
        if self.blockers is not None:
            for tree in self.trees:
                reason = self.blockers.conflict_reason(
                    tree,
                    include_author_identity=self.include_author_identity,
                )
                if reason is not None:
                    return reason
        return None

    def all_targets_reached(self) -> bool:
        """Return whether every coordinated tree reached its exact target."""
        return all(
            tree.target_reached(target)
            for tree, target in zip(self.trees, self.targets)
        )


@dataclass(frozen=True, slots=True)
class GrowthGroupResult:
    """Final target-driven or saturated states for one group."""

    trees: tuple[GrowthTreeState, ...]
    calibration: ReferenceCalibration
    all_targets_reached: bool
    maximum_depth_processed: int
    rejection_counts: dict[str, int]


def reference_calibration(
    trees: Sequence[GrowthTreeState],
) -> ReferenceCalibration:
    """Calculate one joint population profile from depth-zero paper blocks."""
    paper_ids: list[str] = []
    counts: list[int] = []
    for tree in trees:
        for paper_id in tree.gold_paper_ids:
            paper_ids.append(paper_id)
            counts.append(len(tree.references_by_paper.get(paper_id, set())))
    if not counts:
        return ReferenceCalibration((), (), 0.0, 0.0)
    mean = math.fsum(counts) / len(counts)
    variance = math.fsum((count - mean) ** 2 for count in counts) / len(counts)
    return ReferenceCalibration(
        tuple(paper_ids),
        tuple(counts),
        mean,
        math.sqrt(variance),
    )


def sample_reference_cap(
    *,
    calibration: ReferenceCalibration,
    current_count: int,
    database_outgoing_count: int,
    random_seed: int | None,
    key: Any,
) -> int:
    """Sample one total reference-list cap with mandatory-count clamping."""
    rng = random.Random(stable_growth_seed(random_seed, ("reference-cap", key)))
    sampled = math.floor(rng.gauss(calibration.mean, calibration.std) + 0.5)
    bounded_sample = min(max(sampled, 0), max(database_outgoing_count, 0))
    return max(current_count, bounded_sample)


def initialize_growth_tree(
    *,
    cache_key: str,
    path_id: str,
    orientation_index: int,
    source_node_id: str,
    target_node_id: str,
    witness: Sequence[str],
    source_wrong_path: Sequence[str],
    target_wrong_path: Sequence[str],
    gold_paper_ids: Sequence[str],
    expanded_paper_ids: Sequence[str],
    authors_by_paper: Mapping[str, Iterable[str]],
    references_by_paper: Mapping[str, Iterable[str]],
    protected_reference_pairs: Iterable[tuple[str, str]],
    facts_reader: PaperFactsReader,
    include_author_identity: bool,
    rendered_edges_only: bool = False,
) -> GrowthTreeState | None:
    """Build and validate one thin tree with incremental complexity counters."""
    expanded_order = list(dict.fromkeys(str(value) for value in expanded_paper_ids))
    expanded = set(expanded_order)
    authors = {
        str(paper_id): {str(author_id) for author_id in author_ids}
        for paper_id, author_ids in authors_by_paper.items()
        if str(paper_id) in expanded
    }
    references = {
        str(paper_id): {str(cited_id) for cited_id in cited_ids}
        for paper_id, cited_ids in references_by_paper.items()
        if str(paper_id) in expanded
    }
    for paper_id in expanded:
        authors.setdefault(paper_id, set())
        references.setdefault(paper_id, set())
    visible = build_visible_graph(
        expanded_paper_ids=expanded_order,
        authors_by_paper=authors,
        references_by_paper=references,
    )
    graph = visible.graph
    normalized_witness = tuple(str(value) for value in witness)
    if (
        not normalized_witness
        or normalized_witness[0] != source_node_id
        or normalized_witness[-1] != target_node_id
        or source_node_id not in graph
        or target_node_id not in graph
        or not nx.has_path(graph, source_node_id, target_node_id)
    ):
        return None
    component_nodes = nx.node_connected_component(graph, source_node_id)
    if len(component_nodes) != graph.number_of_nodes():
        return None
    component = graph.subgraph(component_nodes).copy()
    component.remove_edges_from(nx.selfloop_edges(component))
    if not nx.is_tree(component):
        return None
    try:
        actual_witness = tuple(
            nx.shortest_path(component, source_node_id, target_node_id)
        )
    except nx.NetworkXNoPath:
        return None
    if actual_witness != normalized_witness:
        return None

    source_cost = rooted_expected_dfs_steps(
        component,
        source_node_id,
        target_node_id,
    )
    target_cost = rooted_expected_dfs_steps(
        component,
        target_node_id,
        source_node_id,
    )
    if source_cost is None or target_cost is None:
        return None

    path_index = {
        node_id: index for index, node_id in enumerate(normalized_witness)
    }
    anchor_by_node: dict[str, int] = dict(path_index)
    path_nodes = set(normalized_witness)
    for node_id, index in path_index.items():
        stack = [
            str(neighbor_id)
            for neighbor_id in component.neighbors(node_id)
            if str(neighbor_id) not in path_nodes
        ]
        while stack:
            branch_node_id = stack.pop()
            if branch_node_id in anchor_by_node:
                continue
            anchor_by_node[branch_node_id] = index
            stack.extend(
                str(neighbor_id)
                for neighbor_id in component.neighbors(branch_node_id)
                if str(neighbor_id) not in anchor_by_node
                and str(neighbor_id) not in path_nodes
            )
    if set(anchor_by_node) != set(component.nodes):
        return None

    visible_papers = set(expanded)
    for cited_ids in references.values():
        visible_papers.update(cited_ids)
    title_owner: dict[str, str] = {}
    author_owner_by_id: dict[str, str] = {}
    author_owner_by_name: dict[str, str] = {}
    citation_neighbors: set[str] = set()
    citation_neighbor_counts: dict[str, int] = {}
    rendered_neighbors: dict[str, set[str]] = {
        paper_id: set() for paper_id in visible_papers
    }
    for owner_id, cited_ids in references.items():
        for cited_id in cited_ids:
            rendered_neighbors.setdefault(owner_id, set()).add(cited_id)
            rendered_neighbors.setdefault(cited_id, set()).add(owner_id)
    for paper_id in sorted(visible_papers):
        facts = (
            facts_reader.get_identity(paper_id)
            if rendered_edges_only
            else facts_reader.get(paper_id)
        )
        if facts is None or not facts.title_key:
            return None
        if not rendered_edges_only:
            if any(
                cited_id not in facts.outgoing_paper_ids
                for cited_id in references.get(paper_id, set())
            ):
                return None
            hidden_neighbors = (
                set(facts.citation_neighbor_ids).intersection(visible_papers)
                - rendered_neighbors.get(paper_id, set())
                - {paper_id}
            )
            if hidden_neighbors:
                return None
        previous_paper = title_owner.get(facts.title_key)
        if previous_paper is not None and previous_paper != paper_id:
            return None
        title_owner[facts.title_key] = paper_id
        if not rendered_edges_only:
            for neighbor_id in facts.citation_neighbor_ids:
                citation_neighbors.add(neighbor_id)
                citation_neighbor_counts[neighbor_id] = (
                    citation_neighbor_counts.get(neighbor_id, 0) + 1
                )
        if not include_author_identity:
            continue
        for author_id in facts.author_ids:
            previous_paper = author_owner_by_id.get(author_id)
            if previous_paper is not None and previous_paper != paper_id:
                return None
            author_owner_by_id[author_id] = paper_id
        for name_key in facts.author_name_keys:
            previous_paper = author_owner_by_name.get(name_key)
            if previous_paper is not None and previous_paper != paper_id:
                return None
            author_owner_by_name[name_key] = paper_id

    gold_papers = tuple(dict.fromkeys(str(value) for value in gold_paper_ids))
    if any(paper_id not in expanded for paper_id in gold_papers):
        return None
    return GrowthTreeState(
        cache_key=cache_key,
        path_id=path_id,
        orientation_index=orientation_index,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        witness=normalized_witness,
        source_wrong_path=tuple(str(value) for value in source_wrong_path),
        target_wrong_path=tuple(str(value) for value in target_wrong_path),
        gold_paper_ids=gold_papers,
        initial_expanded_paper_ids=set(expanded),
        expanded_paper_order=expanded_order,
        expanded_paper_seen_ids=set(expanded),
        expanded_paper_ids=expanded,
        visible_paper_ids=visible_papers,
        graph_author_ids={
            author_id for author_ids in authors.values() for author_id in author_ids
        },
        authors_by_paper=authors,
        references_by_paper=references,
        protected_reference_pairs={
            (str(owner), str(cited_id))
            for owner, cited_id in protected_reference_pairs
        },
        optional_reference_pairs=set(),
        paper_depth={paper_id: 0 for paper_id in visible_papers},
        parent_by_paper={},
        child_count_by_paper={paper_id: 0 for paper_id in visible_papers},
        discovery_pair_by_paper={},
        anchor_index_by_node=anchor_by_node,
        source_cost_units=int(round(float(source_cost) * 2.0)),
        target_cost_units=int(round(float(target_cost) * 2.0)),
        component_size=component.number_of_nodes(),
        title_owner_by_key=title_owner,
        used_title_keys=set(title_owner),
        author_owner_by_id=author_owner_by_id,
        author_owner_by_name=author_owner_by_name,
        citation_neighbor_ids=citation_neighbors,
        citation_neighbor_count_by_id=citation_neighbor_counts,
    )


def candidate_conflict_reason(
    group: QuestionGrowthGroup,
    tree_index: int,
    center_paper_id: str,
    candidate_facts: PaperFacts,
) -> str | None:
    """Return the first local or cross-tree identity conflict for a candidate."""
    candidate_id = candidate_facts.paper_id
    blockers = group.blockers
    if blockers is not None:
        if candidate_id in blockers.paper_ids:
            return "paper_already_visible"
        if candidate_facts.title_key in blockers.title_keys:
            return "shared_normalized_paper_title"
        if group.include_author_identity:
            if blockers.author_ids.intersection(candidate_facts.author_ids):
                return "shared_author_id"
            if blockers.author_name_keys.intersection(
                candidate_facts.author_name_keys
            ):
                return "shared_normalized_author_name"
        if (
            candidate_id in blockers.citation_neighbor_ids
            or blockers.paper_ids.intersection(
                candidate_facts.citation_neighbor_ids
            )
        ):
            return "citation_edge_crosses_trees"
    paper_owner = group.paper_owner.get(candidate_id)
    if paper_owner is not None:
        return "paper_already_visible"
    title_owner = group.title_owner.get(candidate_facts.title_key)
    if title_owner is not None:
        return "shared_normalized_paper_title"
    if group.include_author_identity:
        if any(
            author_id in group.author_id_owner
            for author_id in candidate_facts.author_ids
        ):
            return "shared_author_id"
        if any(
            name_key in group.author_name_owner
            for name_key in candidate_facts.author_name_keys
        ):
            return "shared_normalized_author_name"
    for neighbor_id in candidate_facts.citation_neighbor_ids:
        owner_index = group.paper_owner.get(neighbor_id)
        if owner_index is None:
            continue
        if owner_index != tree_index:
            return "citation_edge_crosses_trees"
        if neighbor_id != center_paper_id:
            return "hidden_citation_would_create_loop"
    return None


def plan_growth_addition(
    group: QuestionGrowthGroup,
    expansion: PaperExpansionState,
    *,
    direction: str,
    candidate_paper_id: str,
) -> tuple[GrowthAddition | None, str | None]:
    """Validate one candidate without mutating the tree."""
    tree = group.trees[expansion.tree_index]
    candidate_facts = group.facts_reader.get(candidate_paper_id)
    if candidate_facts is None or not candidate_facts.title_key:
        return None, "missing_candidate_identity"
    reason = candidate_conflict_reason(
        group,
        expansion.tree_index,
        expansion.paper_id,
        candidate_facts,
    )
    if reason is not None:
        return None, reason
    center_id = expansion.paper_id
    if direction == DIRECTION_CITES:
        reference_pair = (center_id, candidate_paper_id)
        promote_center = center_id not in tree.expanded_paper_ids
    elif direction == DIRECTION_CITED_BY:
        reference_pair = (candidate_paper_id, center_id)
        promote_center = False
    else:
        return None, "unsupported_direction"
    if reference_pair[1] in tree.references_by_paper.get(reference_pair[0], set()):
        return None, "reference_already_visible"

    center_author_ids: frozenset[str] = frozenset()
    added_author_ids: set[str] = set()
    if group.include_author_identity and promote_center:
        center_facts = group.facts_reader.get(center_id)
        if center_facts is None:
            return None, "missing_center_identity"
        center_author_ids = center_facts.author_ids
        added_author_ids.update(
            author_id
            for author_id in center_author_ids
            if author_id not in tree.graph_author_ids
        )
    if group.include_author_identity and direction == DIRECTION_CITED_BY:
        added_author_ids.update(
            author_id
            for author_id in candidate_facts.author_ids
            if author_id not in tree.graph_author_ids
        )

    added_node_count = 1 + len(added_author_ids)
    anchor_index = tree.anchor_index_by_node[center_id]
    path_last_index = len(tree.witness) - 1
    source_units = tree.source_cost_units
    target_units = tree.target_cost_units
    if anchor_index < path_last_index:
        source_units += added_node_count
    if anchor_index > 0:
        target_units += added_node_count
    return (
        GrowthAddition(
            tree_index=expansion.tree_index,
            center_paper_id=center_id,
            candidate_paper_id=candidate_paper_id,
            direction=direction,
            reference_pair=reference_pair,
            candidate_facts=candidate_facts,
            promote_center=promote_center,
            center_author_ids=center_author_ids,
            added_graph_node_count=added_node_count,
            anchor_index=anchor_index,
            depth=expansion.depth + 1,
            resulting_source_cost_units=source_units,
            resulting_target_cost_units=target_units,
            resulting_component_size=tree.component_size + added_node_count,
        ),
        None,
    )


def addition_target_value(
    addition: GrowthAddition,
    target_kind: str,
) -> float:
    """Return the target metric after applying one planned addition."""
    if target_kind == TARGET_COMPONENT:
        return float(addition.resulting_component_size)
    yes_value = float(
        min(
            addition.resulting_source_cost_units,
            addition.resulting_target_cost_units,
        )
    ) / 2.0
    if target_kind == TARGET_YES:
        return yes_value
    if target_kind == TARGET_MATCHING:
        return float(math.ceil(yes_value))
    raise ValueError(f"Unsupported growth target kind: {target_kind!r}.")


def apply_growth_addition(
    group: QuestionGrowthGroup,
    addition: GrowthAddition,
) -> None:
    """Commit one previously validated atomic addition."""
    tree = group.trees[addition.tree_index]
    candidate_id = addition.candidate_paper_id
    facts = addition.candidate_facts
    if addition.promote_center:
        center_id = addition.center_paper_id
        tree.expanded_paper_ids.add(center_id)
        if center_id not in tree.expanded_paper_seen_ids:
            tree.expanded_paper_order.append(center_id)
            tree.expanded_paper_seen_ids.add(center_id)
        tree.authors_by_paper[center_id] = set(addition.center_author_ids)
        tree.references_by_paper.setdefault(center_id, set())
        tree.graph_author_ids.update(addition.center_author_ids)
        for author_id in addition.center_author_ids:
            tree.anchor_index_by_node[author_id] = addition.anchor_index

    owner_id, cited_id = addition.reference_pair
    tree.references_by_paper.setdefault(owner_id, set()).add(cited_id)
    if addition.direction == DIRECTION_CITED_BY:
        tree.expanded_paper_ids.add(candidate_id)
        tree.expanded_paper_order.append(candidate_id)
        tree.expanded_paper_seen_ids.add(candidate_id)
        tree.authors_by_paper[candidate_id] = (
            set(facts.author_ids) if group.include_author_identity else set()
        )
        tree.references_by_paper.setdefault(candidate_id, set())
        tree.graph_author_ids.update(tree.authors_by_paper[candidate_id])
        for author_id in tree.authors_by_paper[candidate_id]:
            tree.anchor_index_by_node[author_id] = addition.anchor_index

    tree.visible_paper_ids.add(candidate_id)
    tree.paper_depth[candidate_id] = addition.depth
    tree.parent_by_paper[candidate_id] = addition.center_paper_id
    tree.child_count_by_paper[candidate_id] = 0
    tree.child_count_by_paper[addition.center_paper_id] = (
        tree.child_count_by_paper.get(addition.center_paper_id, 0) + 1
    )
    tree.discovery_pair_by_paper[candidate_id] = addition.reference_pair
    tree.anchor_index_by_node[candidate_id] = addition.anchor_index
    tree.optional_reference_pairs.add(addition.reference_pair)
    tree.source_cost_units = addition.resulting_source_cost_units
    tree.target_cost_units = addition.resulting_target_cost_units
    tree.component_size = addition.resulting_component_size
    tree.title_owner_by_key[facts.title_key] = candidate_id
    tree.used_title_keys.add(facts.title_key)
    if group.include_author_identity:
        for author_id in facts.author_ids:
            tree.author_owner_by_id[author_id] = candidate_id
        for name_key in facts.author_name_keys:
            tree.author_owner_by_name[name_key] = candidate_id
    for neighbor_id in facts.citation_neighbor_ids:
        tree.citation_neighbor_ids.add(neighbor_id)
        tree.citation_neighbor_count_by_id[neighbor_id] = (
            tree.citation_neighbor_count_by_id.get(neighbor_id, 0) + 1
        )

    group.paper_owner[candidate_id] = addition.tree_index
    group.title_owner[facts.title_key] = (addition.tree_index, candidate_id)
    group.used_title_keys.add(facts.title_key)
    if group.include_author_identity:
        for author_id in facts.author_ids:
            group.author_id_owner[author_id] = (addition.tree_index, candidate_id)
        for name_key in facts.author_name_keys:
            group.author_name_owner[name_key] = (
                addition.tree_index,
                candidate_id,
            )


def expansion_state_for_paper(
    *,
    tree_index: int,
    tree: GrowthTreeState,
    paper_id: str,
    depth: int,
    facts_reader: PaperFactsReader,
    calibration: ReferenceCalibration | None,
    random_seed: int | None,
    group_key: Any,
    candidate_paper_ids: tuple[Sequence[str], Sequence[str]] | None = None,
) -> PaperExpansionState | None:
    """Create one deterministically shuffled active-paper state."""
    facts: PaperFacts | None = None
    if candidate_paper_ids is None or calibration is not None:
        facts = facts_reader.get(paper_id)
        if facts is None:
            return None
    if candidate_paper_ids is None:
        assert facts is not None
        outgoing = list(facts.outgoing_paper_ids)
        incoming = list(facts.incoming_paper_ids)
    else:
        outgoing = list(candidate_paper_ids[0])
        incoming = list(candidate_paper_ids[1])
    random.Random(
        stable_growth_seed(
            random_seed,
            ("outgoing-order", group_key, tree.cache_key, depth, paper_id),
        )
    ).shuffle(outgoing)
    random.Random(
        stable_growth_seed(
            random_seed,
            ("incoming-order", group_key, tree.cache_key, depth, paper_id),
        )
    ).shuffle(incoming)
    cap: int | None = None
    if calibration is not None:
        assert facts is not None
        current_count = len(tree.references_by_paper.get(paper_id, set()))
        cap = sample_reference_cap(
            calibration=calibration,
            current_count=current_count,
            database_outgoing_count=len(facts.outgoing_paper_ids),
            random_seed=random_seed,
            key=(group_key, tree.cache_key, depth, paper_id),
        )
        tree.sampled_caps_by_paper[paper_id] = cap
    return PaperExpansionState(
        tree_index=tree_index,
        paper_id=paper_id,
        depth=depth,
        outgoing_candidates=outgoing,
        incoming_candidates=incoming,
        reference_cap=cap,
    )


def grow_question_group(
    initial_trees: Sequence[GrowthTreeState],
    targets: Sequence[GrowthTarget],
    *,
    facts_reader: PaperFactsReader,
    include_author_identity: bool,
    random_seed: int | None,
    group_key: Any,
    initial_calibration: ReferenceCalibration | None = None,
    blockers: GrowthBlockers | None = None,
) -> GrowthGroupResult | None:
    """Grow coordinated trees shallow-first until exact targets or exhaustion.

    When ``initial_calibration`` is supplied, its reference caps apply from
    depth zero. Otherwise, depth zero remains uncapped and supplies the
    sample-specific calibration used by later levels.
    """
    if len(initial_trees) != len(targets) or not initial_trees:
        raise ValueError("Every growth tree requires exactly one target.")
    for target in targets:
        target.validate()
    trees = [tree.clone() for tree in initial_trees]
    group = QuestionGrowthGroup(
        trees=trees,
        targets=list(targets),
        facts_reader=facts_reader,
        include_author_identity=include_author_identity,
        blockers=blockers,
    )
    conflict = group.rebuild_registries()
    if conflict is not None:
        return None
    if any(tree.target_exceeded(target) for tree, target in zip(trees, targets)):
        return None

    frontiers: list[set[str]] = [set(tree.gold_paper_ids) for tree in trees]
    calibration = initial_calibration
    depth = 0
    rejection_counts: dict[str, int] = {}
    while True:
        active: list[PaperExpansionState] = []
        next_frontiers: list[set[str]] = [set() for tree in trees]
        for tree_index, tree in enumerate(trees):
            if tree.target_reached(targets[tree_index]):
                continue
            for paper_id in sorted(frontiers[tree_index]):
                expansion = expansion_state_for_paper(
                    tree_index=tree_index,
                    tree=tree,
                    paper_id=paper_id,
                    depth=depth,
                    facts_reader=facts_reader,
                    calibration=calibration,
                    random_seed=random_seed,
                    group_key=group_key,
                )
                if expansion is not None:
                    active.append(expansion)

        sweep_index = 0
        while active:
            sweep = list(active)
            random.Random(
                stable_growth_seed(
                    random_seed,
                    ("sweep", group_key, depth, sweep_index),
                )
            ).shuffle(sweep)
            next_active: list[PaperExpansionState] = []
            for expansion in sweep:
                tree = trees[expansion.tree_index]
                target = targets[expansion.tree_index]
                if tree.target_reached(target):
                    continue
                if expansion.reference_cap is not None and (
                    expansion.current_reference_count(tree)
                    >= expansion.reference_cap
                ):
                    expansion.retired_reason = "reference_cap_reached"
                    tree.retirement_reason_by_paper[expansion.paper_id] = (
                        expansion.retired_reason
                    )
                    continue

                outgoing_addition: GrowthAddition | None = None
                while expansion.outgoing_candidates:
                    outgoing_candidate_id = expansion.outgoing_candidates[-1]
                    outgoing_addition, outgoing_reason = plan_growth_addition(
                        group,
                        expansion,
                        direction=DIRECTION_CITES,
                        candidate_paper_id=outgoing_candidate_id,
                    )
                    if outgoing_addition is not None and addition_target_value(
                        outgoing_addition,
                        target.kind,
                    ) > float(target.value):
                        outgoing_reason = "target_overshoot"
                        outgoing_addition = None
                    if outgoing_addition is not None:
                        break
                    expansion.outgoing_candidates.pop()
                    reason_key = outgoing_reason or "candidate_rejected"
                    rejection_counts[reason_key] = (
                        rejection_counts.get(reason_key, 0) + 1
                    )
                if outgoing_addition is None:
                    expansion.retired_reason = "outgoing_candidates_exhausted"
                    tree.retirement_reason_by_paper[expansion.paper_id] = (
                        expansion.retired_reason
                    )
                    continue

                available_directions = [DIRECTION_CITES]
                if expansion.incoming_candidates:
                    available_directions.append(DIRECTION_CITED_BY)
                direction_rng = random.Random(
                    stable_growth_seed(
                        random_seed,
                        (
                            "direction",
                            group_key,
                            depth,
                            sweep_index,
                            expansion.tree_index,
                            expansion.paper_id,
                        ),
                    )
                )
                direction = direction_rng.choice(available_directions)
                expansion.attempts += 1
                if direction == DIRECTION_CITES:
                    expansion.outgoing_candidates.pop()
                    addition = outgoing_addition
                    reason = None
                else:
                    candidate_id = expansion.incoming_candidates.pop()
                    addition, reason = plan_growth_addition(
                        group,
                        expansion,
                        direction=direction,
                        candidate_paper_id=candidate_id,
                    )
                    if addition is not None:
                        proposed_value = addition_target_value(
                            addition,
                            target.kind,
                        )
                        if proposed_value > float(target.value):
                            reason = "target_overshoot"
                            addition = None
                if addition is None:
                    reason_key = reason or "candidate_rejected"
                    rejection_counts[reason_key] = (
                        rejection_counts.get(reason_key, 0) + 1
                    )
                else:
                    apply_growth_addition(group, addition)
                    next_frontiers[expansion.tree_index].add(
                        addition.candidate_paper_id
                    )
                    if direction == DIRECTION_CITES:
                        expansion.accepted_outgoing += 1
                    else:
                        expansion.accepted_incoming += 1
                if (
                    not tree.target_reached(target)
                    and not expansion.retire_if_needed(tree)
                ):
                    next_active.append(expansion)
                elif expansion.retired_reason:
                    tree.retirement_reason_by_paper[expansion.paper_id] = (
                        expansion.retired_reason
                    )
                tree.achieved_counts_by_paper[expansion.paper_id] = len(
                    tree.references_by_paper.get(expansion.paper_id, set())
                )
            active = next_active
            sweep_index += 1

        if depth == 0 and calibration is None:
            calibration = reference_calibration(trees)
        if group.all_targets_reached():
            assert calibration is not None
            return GrowthGroupResult(
                trees=tuple(trees),
                calibration=calibration,
                all_targets_reached=True,
                maximum_depth_processed=depth,
                rejection_counts=rejection_counts,
            )
        if not any(next_frontiers):
            assert calibration is not None
            return GrowthGroupResult(
                trees=tuple(trees),
                calibration=calibration,
                all_targets_reached=False,
                maximum_depth_processed=depth,
                rejection_counts=rejection_counts,
            )
        frontiers = next_frontiers
        depth += 1


def validate_incremental_complexity(tree: GrowthTreeState) -> bool:
    """Return whether incremental counters match the full graph calculation."""
    visible = build_visible_graph(
        expanded_paper_ids=(
            paper_id
            for paper_id in tree.expanded_paper_order
            if paper_id in tree.expanded_paper_ids
        ),
        authors_by_paper=tree.authors_by_paper,
        references_by_paper=tree.references_by_paper,
    )
    source_cost = rooted_expected_dfs_steps(
        visible.graph,
        tree.source_node_id,
        tree.target_node_id,
    )
    target_cost = rooted_expected_dfs_steps(
        visible.graph,
        tree.target_node_id,
        tree.source_node_id,
    )
    expected_child_counts = {
        paper_id: 0 for paper_id in tree.visible_paper_ids
    }
    for paper_id, parent_id in tree.parent_by_paper.items():
        if paper_id in tree.visible_paper_ids:
            expected_child_counts[parent_id] += 1
    return (
        source_cost is not None
        and target_cost is not None
        and int(round(float(source_cost) * 2.0)) == tree.source_cost_units
        and int(round(float(target_cost) * 2.0)) == tree.target_cost_units
        and visible.graph.number_of_nodes() == tree.component_size
        and nx.is_tree(visible.graph)
        and tree.citation_neighbor_ids
        == set(tree.citation_neighbor_count_by_id)
        and all(
            count > 0
            for count in tree.citation_neighbor_count_by_id.values()
        )
        and tree.child_count_by_paper == expected_child_counts
        and set(tree.expanded_paper_order) == tree.expanded_paper_seen_ids
        and tree.expanded_paper_ids.issubset(tree.expanded_paper_seen_ids)
    )


@dataclass(frozen=True, slots=True)
class PruningLadder:
    """Compact randomized deepest-first removal order and attainable values."""

    target_kind: str
    removal_order: tuple[str, ...]
    prefix_by_value: dict[float, int]

    @property
    def values(self) -> frozenset[float]:
        """Return every exact value reached by this pruning order."""
        return frozenset(self.prefix_by_value)


def deepest_leaf_paper_ids(tree: GrowthTreeState) -> list[str]:
    """Return removable leaf papers at the deepest current growth level."""
    candidates = [
        paper_id
        for paper_id in tree.visible_paper_ids
        if paper_id not in tree.gold_paper_ids
        and paper_id in tree.parent_by_paper
        and tree.child_count_by_paper.get(paper_id, 0) == 0
    ]
    if not candidates:
        return []
    maximum_depth = max(tree.paper_depth[paper_id] for paper_id in candidates)
    return sorted(
        paper_id
        for paper_id in candidates
        if tree.paper_depth[paper_id] == maximum_depth
    )


def decrement_cost_for_node(tree: GrowthTreeState, node_id: str) -> None:
    """Remove one graph node's contribution from incremental DFS counters."""
    anchor_index = tree.anchor_index_by_node.pop(node_id)
    if anchor_index < len(tree.witness) - 1:
        tree.source_cost_units -= 1
    if anchor_index > 0:
        tree.target_cost_units -= 1
    tree.component_size -= 1


def prune_leaf_paper(
    tree: GrowthTreeState,
    paper_id: str,
    *,
    facts_reader: PaperFactsReader,
) -> None:
    """Remove one deepest leaf and demote its terminal parent when needed."""
    if (
        paper_id in tree.gold_paper_ids
        or paper_id not in tree.parent_by_paper
        or tree.child_count_by_paper.get(paper_id, 0) != 0
    ):
        raise ValueError(f"Paper {paper_id!r} is not a removable leaf.")
    reference_pair = tree.discovery_pair_by_paper[paper_id]
    parent_id = tree.parent_by_paper[paper_id]
    owner_id, cited_id = reference_pair
    tree.references_by_paper[owner_id].remove(cited_id)
    tree.optional_reference_pairs.discard(reference_pair)

    if paper_id in tree.expanded_paper_ids:
        if tree.references_by_paper.get(paper_id):
            raise ValueError("A removable expanded leaf still owns references.")
        for author_id in sorted(tree.authors_by_paper.get(paper_id, set())):
            tree.graph_author_ids.remove(author_id)
            decrement_cost_for_node(tree, author_id)
        tree.authors_by_paper.pop(paper_id, None)
        tree.references_by_paper.pop(paper_id, None)
        tree.expanded_paper_ids.remove(paper_id)

    decrement_cost_for_node(tree, paper_id)
    tree.visible_paper_ids.remove(paper_id)
    facts = facts_reader.get_identity(paper_id)
    if facts is None:
        raise ValueError(f"Missing identity facts for pruned paper {paper_id!r}.")
    tree.title_owner_by_key.pop(facts.title_key, None)
    tree.used_title_keys.discard(facts.title_key)
    for author_id in facts.author_ids:
        tree.author_owner_by_id.pop(author_id, None)
    for name_key in facts.author_name_keys:
        tree.author_owner_by_name.pop(name_key, None)
    tree.paper_depth.pop(paper_id, None)
    tree.parent_by_paper.pop(paper_id, None)
    tree.child_count_by_paper.pop(paper_id, None)
    tree.child_count_by_paper[parent_id] -= 1
    tree.discovery_pair_by_paper.pop(paper_id, None)
    tree.sampled_caps_by_paper.pop(paper_id, None)
    tree.achieved_counts_by_paper.pop(paper_id, None)
    tree.retirement_reason_by_paper.pop(paper_id, None)
    tree.processed_center_ids.discard(paper_id)

    owner_is_promoted_terminal = (
        owner_id not in tree.initial_expanded_paper_ids
        and owner_id in tree.expanded_paper_ids
        and owner_id not in tree.processed_center_ids
        and not tree.references_by_paper.get(owner_id)
        and tree.discovery_pair_by_paper.get(owner_id, (None, None))[1]
        == owner_id
    )
    if owner_is_promoted_terminal:
        for author_id in sorted(tree.authors_by_paper.get(owner_id, set())):
            tree.graph_author_ids.remove(author_id)
            decrement_cost_for_node(tree, author_id)
        tree.authors_by_paper.pop(owner_id, None)
        tree.references_by_paper.pop(owner_id, None)
        tree.expanded_paper_ids.remove(owner_id)

    for neighbor_id in facts.citation_neighbor_ids:
        remaining = tree.citation_neighbor_count_by_id[neighbor_id] - 1
        if remaining <= 0:
            tree.citation_neighbor_count_by_id.pop(neighbor_id, None)
            tree.citation_neighbor_ids.discard(neighbor_id)
        else:
            tree.citation_neighbor_count_by_id[neighbor_id] = remaining


def build_pruning_ladder(
    tree: GrowthTreeState,
    *,
    target_kind: str,
    facts_reader: PaperFactsReader,
    random_seed: int | None,
    key: Any,
) -> PruningLadder:
    """Return exact values from uniform deepest-level leaf pruning."""
    if target_kind not in SUPPORTED_TARGET_KINDS:
        raise ValueError(f"Unsupported growth target kind: {target_kind!r}.")
    probe = tree.clone()
    removal_order: list[str] = []
    prefix_by_value = {probe.target_value(target_kind): 0}
    leaves_by_depth: dict[int, set[str]] = {}
    for paper_id in probe.visible_paper_ids:
        if (
            paper_id not in probe.gold_paper_ids
            and paper_id in probe.parent_by_paper
            and probe.child_count_by_paper.get(paper_id, 0) == 0
        ):
            leaves_by_depth.setdefault(probe.paper_depth[paper_id], set()).add(
                paper_id
            )
    depth = max(leaves_by_depth, default=0)
    while depth > 0:
        leaves = sorted(leaves_by_depth.pop(depth, set()))
        if not leaves:
            depth -= 1
            continue
        random.Random(
            stable_growth_seed(
                random_seed,
                ("prune-level", key, target_kind, depth, tuple(leaves)),
            )
        ).shuffle(leaves)
        while leaves:
            paper_id = leaves.pop()
            parent_id = probe.parent_by_paper[paper_id]
            prune_leaf_paper(probe, paper_id, facts_reader=facts_reader)
            removal_order.append(paper_id)
            prefix_by_value.setdefault(
                probe.target_value(target_kind),
                len(removal_order),
            )
            if (
                parent_id not in probe.gold_paper_ids
                and parent_id in probe.parent_by_paper
                and probe.child_count_by_paper.get(parent_id, 0) == 0
            ):
                leaves_by_depth.setdefault(
                    probe.paper_depth[parent_id],
                    set(),
                ).add(parent_id)
        depth -= 1
    return PruningLadder(
        target_kind=target_kind,
        removal_order=tuple(removal_order),
        prefix_by_value=prefix_by_value,
    )


def apply_pruning_prefix(
    tree: GrowthTreeState,
    ladder: PruningLadder,
    prefix_length: int,
    *,
    facts_reader: PaperFactsReader,
) -> GrowthTreeState:
    """Apply one compact ladder prefix to a fresh clone of ``tree``."""
    if prefix_length < 0 or prefix_length > len(ladder.removal_order):
        raise ValueError("Pruning prefix is outside the ladder.")
    result = tree.clone()
    for paper_id in ladder.removal_order[:prefix_length]:
        prune_leaf_paper(result, paper_id, facts_reader=facts_reader)
    return result
