"""Zero-child-alpha preferential growth for v8 reachability questions.

A frontier paper with no accepted optional children has weight ``alpha``.  A
paper with one or more accepted children has weight equal to that child count.
After choosing a paper, v8 flips a fair coin between its available ``cites``
and ``cited_by`` directions, then uniformly samples one candidate in that
direction.
"""

from __future__ import annotations

import math
import random
from bisect import bisect_left, bisect_right, insort
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Any

from src.questions.reachability_level_growth import (
    DIRECTION_CITED_BY,
    DIRECTION_CITES,
    TARGET_COMPONENT,
    TARGET_MATCHING,
    TARGET_YES,
    GrowthAddition,
    GrowthBlockers,
    GrowthTarget,
    GrowthTreeState,
    PaperFacts,
    PaperFactsReader,
    QuestionGrowthGroup,
    ReferenceCalibration,
    addition_target_value,
    apply_growth_addition,
    stable_growth_seed,
)
from src.questions.reachability_rendered_growth import (
    candidate_conflict_reason,
    initialize_growth_tree,
    plan_center_promotion,
    validate_rendered_growth_tree,
)
from src.questions.reachability_rendered_growth_v5 import (
    final_reference_calibration,
)

GROWTH_REACHED = "target_reached"
GROWTH_TARGET_BLOCKED = "target_blocked"
GROWTH_SATURATED = "saturated"
SUPPORTED_GROWTH_STATUSES = {
    GROWTH_REACHED,
    GROWTH_TARGET_BLOCKED,
    GROWTH_SATURATED,
}
UNLIMITED_CHILDREN_PER_NODE = -1

PaperCandidateFilter = Callable[[Sequence[str]], frozenset[str]]


@dataclass(frozen=True, slots=True)
class CandidateEdge:
    """One citation-direction entry in a combined candidate pool."""

    paper_id: str
    direction: str


@dataclass(slots=True)
class FrontierNodeState:
    """Direction-separated candidate state across bucket targets."""

    remaining_by_direction: dict[str, list[CandidateEdge]] = field(
        default_factory=lambda: {
            DIRECTION_CITES: [],
            DIRECTION_CITED_BY: [],
        }
    )
    deferred_by_direction: dict[str, list[CandidateEdge]] = field(
        default_factory=lambda: {
            DIRECTION_CITES: [],
            DIRECTION_CITED_BY: [],
        }
    )
    loaded: bool = False
    permanently_exhausted: bool = False

    def has_remaining(self) -> bool:
        """Return whether at least one direction has a candidate."""
        return any(self.remaining_by_direction.values())

    def has_deferred(self) -> bool:
        """Return whether at least one candidate waits for a larger target."""
        return any(self.deferred_by_direction.values())

    def clear_candidates(self) -> None:
        """Release all remaining and deferred candidates."""
        for candidates in self.remaining_by_direction.values():
            candidates.clear()
        for candidates in self.deferred_by_direction.values():
            candidates.clear()


@dataclass(slots=True)
class PreferentialGrowthSession:
    """Mutable child-weighted frontier state for one growth run."""

    group: QuestionGrowthGroup
    num_children_per_node: int
    alpha: float
    random_seed: int | None
    group_key: Any
    candidate_cache: dict[str, tuple[CandidateEdge, ...]]
    paper_candidate_is_allowed: Callable[[str], bool] | None
    paper_candidate_filter: PaperCandidateFilter | None
    frontier_states: dict[tuple[int, str], FrontierNodeState]
    expandable_nodes: set[tuple[int, str]]
    sorted_expandable_nodes: list[tuple[int, str]]
    rejection_counts: dict[str, int]
    random_generator: random.Random
    turn_count: int = 0
    maximum_depth: int = 0
    eligible_nodes: list[tuple[int, str]] = field(default_factory=list)
    eligible_weights: list[float] = field(default_factory=list)
    eligibility_signature: tuple[Any, ...] = ()

    def refresh_frontier_node(self, node_key: tuple[int, str]) -> None:
        """Update only the frontier entry affected by a candidate attempt."""
        index = bisect_left(self.eligible_nodes, node_key)
        present = (
            index < len(self.eligible_nodes) and self.eligible_nodes[index] == node_key
        )
        if node_key in self.expandable_nodes and frontier_node_is_eligible(
            self, node_key
        ):
            tree_index, paper_id = node_key
            weight = frontier_attachment_weight(
                self.group.trees[tree_index].child_count_by_paper.get(paper_id, 0),
                self.alpha,
            )
            if present:
                self.eligible_weights[index] = weight
            else:
                self.eligible_nodes.insert(index, node_key)
                self.eligible_weights.insert(index, weight)
        elif present:
            self.eligible_nodes.pop(index)
            self.eligible_weights.pop(index)

    def add_expandable_node(self, node_key: tuple[int, str]) -> None:
        """Add one frontier node while preserving deterministic order."""
        if node_key in self.expandable_nodes:
            return
        self.expandable_nodes.add(node_key)
        insort(self.sorted_expandable_nodes, node_key)
        self.refresh_frontier_node(node_key)

    def discard_expandable_node(self, node_key: tuple[int, str]) -> None:
        """Remove one frontier node from both frontier indexes."""
        if node_key not in self.expandable_nodes:
            return
        self.expandable_nodes.remove(node_key)
        index = bisect_left(self.sorted_expandable_nodes, node_key)
        self.sorted_expandable_nodes.pop(index)
        self.refresh_frontier_node(node_key)


@dataclass(frozen=True, slots=True)
class GrowthAdvanceResult:
    """Snapshot and stop reason after advancing one sampled target."""

    trees: tuple[GrowthTreeState, ...]
    calibration: ReferenceCalibration
    status: str
    target_value: float
    target_kinds: tuple[str, ...]
    maximum_depth_processed: int
    rejection_counts: dict[str, int]

    @property
    def all_targets_reached(self) -> bool:
        """Return whether the sampled target was reached exactly."""
        return self.status == GROWTH_REACHED


def validate_num_children_per_node(value: int) -> None:
    """Require ``-1`` for unlimited children or one positive child cap."""
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value == 0
        or value < UNLIMITED_CHILDREN_PER_NODE
    ):
        raise ValueError("num_children_per_node must be -1 or positive.")


def validate_alpha(value: float) -> None:
    """Require a finite positive baseline weight."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("alpha must be a finite positive number.")


def create_preferential_growth_session(
    initial_trees: Sequence[GrowthTreeState],
    target_kinds: Sequence[str],
    *,
    facts_reader: PaperFactsReader,
    include_author_identity: bool,
    num_children_per_node: int,
    alpha: float,
    random_seed: int | None,
    group_key: Any,
    candidate_cache: dict[str, tuple[CandidateEdge, ...]] | None = None,
    paper_candidate_is_allowed: Callable[[str], bool] | None = None,
    paper_candidate_filter: PaperCandidateFilter | None = None,
    blockers: GrowthBlockers | None = None,
) -> PreferentialGrowthSession | None:
    """Create one persistent v8 session from thin growth trees."""
    validate_num_children_per_node(num_children_per_node)
    validate_alpha(alpha)
    if not initial_trees or len(initial_trees) != len(target_kinds):
        raise ValueError("Every growth tree requires exactly one target kind.")
    targets = [GrowthTarget(kind, 0) for kind in target_kinds]
    for target in targets:
        target.validate()
    trees = [tree.clone() for tree in initial_trees]
    group = QuestionGrowthGroup(
        trees=trees,
        targets=targets,
        facts_reader=facts_reader,
        include_author_identity=include_author_identity,
        blockers=blockers,
    )
    if group.rebuild_registries() is not None:
        return None
    expandable_nodes = {
        (tree_index, paper_id)
        for tree_index, tree in enumerate(trees)
        for paper_id in tree.gold_paper_ids
    }
    return PreferentialGrowthSession(
        group=group,
        num_children_per_node=num_children_per_node,
        alpha=float(alpha),
        random_seed=random_seed,
        group_key=group_key,
        candidate_cache=(candidate_cache if candidate_cache is not None else {}),
        paper_candidate_is_allowed=paper_candidate_is_allowed,
        paper_candidate_filter=paper_candidate_filter,
        frontier_states={},
        expandable_nodes=expandable_nodes,
        sorted_expandable_nodes=sorted(expandable_nodes),
        rejection_counts={},
        random_generator=random.Random(
            stable_growth_seed(
                random_seed,
                ("v8-preferential-session", group_key, float(alpha)),
            )
        ),
    )


def record_rejection(session: PreferentialGrowthSession, reason: str) -> None:
    """Increment one cumulative rejection counter."""
    session.rejection_counts[reason] = session.rejection_counts.get(reason, 0) + 1


def combined_candidate_edges(
    session: PreferentialGrowthSession,
    paper_id: str,
) -> tuple[CandidateEdge, ...]:
    """Return cached outgoing and incoming entries for one frontier paper."""
    cached = session.candidate_cache.get(paper_id)
    if cached is not None:
        return cached
    facts = session.group.facts_reader.get(paper_id)
    if facts is None:
        edges: tuple[CandidateEdge, ...] = ()
    else:
        entries = {
            CandidateEdge(str(candidate_id), DIRECTION_CITES)
            for candidate_id in facts.outgoing_paper_ids
        }
        entries.update(
            CandidateEdge(str(candidate_id), DIRECTION_CITED_BY)
            for candidate_id in facts.incoming_paper_ids
        )
        edges = tuple(sorted(entries, key=lambda item: (item.paper_id, item.direction)))
    session.candidate_cache[paper_id] = edges
    return edges


def frontier_state(
    session: PreferentialGrowthSession,
    node_key: tuple[int, str],
) -> FrontierNodeState:
    """Load and partition one paper's candidates at most once."""
    state = session.frontier_states.setdefault(node_key, FrontierNodeState())
    if state.loaded:
        return state
    state.loaded = True
    candidates = combined_candidate_edges(session, node_key[1])
    if candidates:
        if session.paper_candidate_filter is not None:
            allowed_paper_ids = session.paper_candidate_filter(
                tuple(edge.paper_id for edge in candidates)
            )
            candidates = tuple(
                edge for edge in candidates if edge.paper_id in allowed_paper_ids
            )
        elif session.paper_candidate_is_allowed is not None:
            candidates = tuple(
                edge
                for edge in candidates
                if session.paper_candidate_is_allowed(edge.paper_id)
            )
    for edge in candidates:
        state.remaining_by_direction[edge.direction].append(edge)
    if not state.has_remaining():
        state.permanently_exhausted = True
    return state


def center_can_improve_target(
    tree: GrowthTreeState,
    target: GrowthTarget,
    paper_id: str,
) -> bool:
    """Return whether a child at this anchor can raise the target metric."""
    if target.kind == TARGET_COMPONENT:
        return True
    anchor_index = tree.anchor_index_by_node.get(paper_id)
    if anchor_index is None:
        return False
    last_anchor_index = len(tree.witness) - 1
    if tree.source_cost_units == tree.target_cost_units:
        return anchor_index < last_anchor_index or anchor_index > 0
    if tree.source_cost_units < tree.target_cost_units:
        return anchor_index < last_anchor_index
    return anchor_index > 0


def plan_preferential_growth_addition(
    group: QuestionGrowthGroup,
    tree_index: int,
    center_id: str,
    edge: CandidateEdge,
) -> tuple[GrowthAddition | None, str | None]:
    """Validate one candidate and atomically include a needed promotion."""
    tree = group.trees[tree_index]
    candidate_id = edge.paper_id
    if candidate_id == center_id:
        return None, "self_loop"
    if candidate_id in group.paper_owner:
        return None, "paper_already_visible"
    if group.blockers is not None and candidate_id in group.blockers.paper_ids:
        return None, "shared_paper_ids"

    candidate_facts = group.facts_reader.get_identity(candidate_id)
    if candidate_facts is None or not candidate_facts.title_key:
        return None, "missing_candidate_identity"
    reason = candidate_conflict_reason(group, candidate_facts)
    if reason is not None:
        return None, reason

    if edge.direction == DIRECTION_CITES:
        reference_pair = (center_id, candidate_id)
    elif edge.direction == DIRECTION_CITED_BY:
        reference_pair = (candidate_id, center_id)
    else:
        return None, "unsupported_direction"
    if reference_pair[1] in tree.references_by_paper.get(reference_pair[0], set()):
        return None, "reference_already_visible"

    promotion, promotion_reason = plan_center_promotion(
        group,
        tree_index,
        center_id,
    )
    if promotion_reason is not None:
        return None, promotion_reason
    center_author_ids = frozenset() if promotion is None else promotion.author_ids
    promoted_author_count = 0
    if promotion is not None:
        promoted_author_count = len(set(promotion.author_ids) - tree.graph_author_ids)
    candidate_author_ids = (
        set(candidate_facts.author_ids) - tree.graph_author_ids
        if group.include_author_identity and edge.direction == DIRECTION_CITED_BY
        else set()
    )
    added_node_count = 1 + promoted_author_count + len(candidate_author_ids)
    anchor_index = tree.anchor_index_by_node[center_id]
    source_units = tree.source_cost_units
    target_units = tree.target_cost_units
    if anchor_index < len(tree.witness) - 1:
        source_units += added_node_count
    if anchor_index > 0:
        target_units += added_node_count
    return (
        GrowthAddition(
            tree_index=tree_index,
            center_paper_id=center_id,
            candidate_paper_id=candidate_id,
            direction=edge.direction,
            reference_pair=reference_pair,
            candidate_facts=candidate_facts,
            promote_center=promotion is not None,
            center_author_ids=center_author_ids,
            added_graph_node_count=added_node_count,
            anchor_index=anchor_index,
            depth=tree.paper_depth[center_id] + 1,
            resulting_source_cost_units=source_units,
            resulting_target_cost_units=target_units,
            resulting_component_size=tree.component_size + added_node_count,
        ),
        None,
    )


def reset_deferred_candidates(session: PreferentialGrowthSession) -> None:
    """Make target-blocked candidates eligible for a larger bucket."""
    for state in session.frontier_states.values():
        if state.has_deferred() and not state.permanently_exhausted:
            for direction in (DIRECTION_CITES, DIRECTION_CITED_BY):
                state.remaining_by_direction[direction].extend(
                    state.deferred_by_direction[direction]
                )
                state.deferred_by_direction[direction].clear()


def frontier_node_is_eligible(
    session: PreferentialGrowthSession,
    node_key: tuple[int, str],
) -> bool:
    """Check one paper's current eligibility, including deferred candidates."""
    tree_index, paper_id = node_key
    tree = session.group.trees[tree_index]
    target = session.group.targets[tree_index]
    state = session.frontier_states.get(node_key)
    return not (
        tree.target_reached(target)
        or tree.target_exceeded(target)
        or (
            session.num_children_per_node != UNLIMITED_CHILDREN_PER_NODE
            and tree.child_count_by_paper.get(paper_id, 0)
            >= session.num_children_per_node
        )
        or (
            state is not None
            and (state.permanently_exhausted or not state.has_remaining())
        )
        or not center_can_improve_target(tree, target, paper_id)
    )


def eligible_frontier_nodes(
    session: PreferentialGrowthSession,
) -> list[tuple[int, str]]:
    """Reuse the frontier until a target or the cheaper endpoint changes."""
    signature = tuple(
        (
            target.kind,
            target.value,
            tree.target_reached(target) or tree.target_exceeded(target),
            0
            if target.kind == TARGET_COMPONENT
            else (
                (tree.source_cost_units > tree.target_cost_units)
                - (tree.source_cost_units < tree.target_cost_units)
            ),
        )
        for tree, target in zip(session.group.trees, session.group.targets)
    )
    if signature != session.eligibility_signature:
        session.eligible_nodes.clear()
        session.eligible_weights.clear()
        for node_key in session.sorted_expandable_nodes:
            session.refresh_frontier_node(node_key)
        session.eligibility_signature = signature
    return session.eligible_nodes


def choose_weighted_frontier_node(
    session: PreferentialGrowthSession,
    eligible: Sequence[tuple[int, str]],
) -> tuple[int, str]:
    """Sample by alpha for a leaf, otherwise by accepted-child count."""
    weights = (
        session.eligible_weights
        if eligible is session.eligible_nodes
        else [
            frontier_attachment_weight(
                session.group.trees[tree_index].child_count_by_paper.get(paper_id, 0),
                session.alpha,
            )
            for tree_index, paper_id in eligible
        ]
    )
    total_weight = math.fsum(weights)
    threshold = session.random_generator.random() * total_weight
    # accumulate uses the same left-to-right additions as the original loop.
    index = bisect_right(list(accumulate(weights)), threshold)
    return eligible[min(index, len(eligible) - 1)]


def frontier_attachment_weight(accepted_children: int, alpha: float) -> float:
    """Return professor-style weight: alpha at zero, child count otherwise."""
    return float(alpha if accepted_children == 0 else accepted_children)


def choose_available_direction(
    session: PreferentialGrowthSession,
    state: FrontierNodeState,
) -> str:
    """Flip a fair coin when both citation directions are available."""
    available_directions = [
        direction
        for direction in (DIRECTION_CITES, DIRECTION_CITED_BY)
        if state.remaining_by_direction[direction]
    ]
    if len(available_directions) == 1:
        return available_directions[0]
    return session.random_generator.choice(available_directions)


def choose_candidate(
    session: PreferentialGrowthSession,
    node_key: tuple[int, str],
) -> GrowthAddition | None:
    """Coin-flip directions until a valid non-overshooting edge is found."""
    tree_index, center_id = node_key
    target = session.group.targets[tree_index]
    state = frontier_state(session, node_key)
    while state.has_remaining():
        direction = choose_available_direction(session, state)
        candidates = state.remaining_by_direction[direction]
        edge_index = session.random_generator.randrange(len(candidates))
        edge = candidates.pop(edge_index)
        addition, reason = plan_preferential_growth_addition(
            session.group,
            tree_index,
            center_id,
            edge,
        )
        if addition is not None and (
            addition_target_value(addition, target.kind) > float(target.value)
        ):
            state.deferred_by_direction[direction].append(edge)
            record_rejection(session, "target_overshoot")
            continue
        if addition is not None:
            return addition
        record_rejection(session, reason or "candidate_rejected")
    if not state.has_deferred():
        state.permanently_exhausted = True
        session.discard_expandable_node(node_key)
        tree = session.group.trees[tree_index]
        tree.achieved_counts_by_paper[center_id] = tree.child_count_by_paper.get(
            center_id, 0
        )
        tree.retirement_reason_by_paper[center_id] = "citation_candidates_exhausted"
    return None


def deferred_candidate_exists(session: PreferentialGrowthSession) -> bool:
    """Return whether a still-valid candidate is blocked only by this target."""
    for node_key, state in list(session.frontier_states.items()):
        if node_key not in session.expandable_nodes or not state.has_deferred():
            continue
        tree_index, center_id = node_key
        tree = session.group.trees[tree_index]
        target = session.group.targets[tree_index]
        if tree.target_reached(target) or not center_can_improve_target(
            tree, target, center_id
        ):
            continue
        any_still_deferred = False
        for direction in (DIRECTION_CITES, DIRECTION_CITED_BY):
            still_deferred: list[CandidateEdge] = []
            for edge in state.deferred_by_direction[direction]:
                addition, reason = plan_preferential_growth_addition(
                    session.group,
                    tree_index,
                    center_id,
                    edge,
                )
                if addition is None:
                    record_rejection(session, reason or "candidate_rejected")
                    continue
                if addition_target_value(addition, target.kind) > float(target.value):
                    still_deferred.append(edge)
            state.deferred_by_direction[direction] = still_deferred
            any_still_deferred = any_still_deferred or bool(still_deferred)
        if any_still_deferred:
            return True
        if not state.has_remaining():
            state.permanently_exhausted = True
            session.discard_expandable_node(node_key)
            tree.achieved_counts_by_paper[center_id] = tree.child_count_by_paper.get(
                center_id, 0
            )
            tree.retirement_reason_by_paper[center_id] = "citation_candidates_exhausted"
    return False


def growth_snapshot(
    session: PreferentialGrowthSession,
    status: str,
    target_value: float,
) -> GrowthAdvanceResult:
    """Clone the current trees into an immutable advance result."""
    if status not in SUPPORTED_GROWTH_STATUSES:
        raise ValueError(f"Unsupported v8 growth status: {status!r}.")
    trees = tuple(tree.clone() for tree in session.group.trees)
    return GrowthAdvanceResult(
        trees=trees,
        calibration=final_reference_calibration(trees),
        status=status,
        target_value=target_value,
        target_kinds=tuple(target.kind for target in session.group.targets),
        maximum_depth_processed=session.maximum_depth,
        rejection_counts=dict(session.rejection_counts),
    )


def advance_preferential_growth_session(
    session: PreferentialGrowthSession,
    target_value: float,
    *,
    progress_callback: Callable[[], None] | None = None,
) -> GrowthAdvanceResult:
    """Grow the persistent session until exact target, block, or saturation."""
    targets = [
        GrowthTarget(target.kind, target_value) for target in session.group.targets
    ]
    for target in targets:
        target.validate()
    session.group.targets = targets
    reset_deferred_candidates(session)
    session.eligibility_signature = ()

    if any(
        tree.target_exceeded(target)
        for tree, target in zip(session.group.trees, targets)
    ):
        record_rejection(session, "current_value_exceeds_target")
        return growth_snapshot(session, GROWTH_TARGET_BLOCKED, target_value)

    while True:
        if progress_callback is not None:
            progress_callback()
        if session.group.all_targets_reached():
            return growth_snapshot(session, GROWTH_REACHED, target_value)
        eligible = eligible_frontier_nodes(session)
        if not eligible:
            status = (
                GROWTH_TARGET_BLOCKED
                if deferred_candidate_exists(session)
                else GROWTH_SATURATED
            )
            record_rejection(session, status)
            return growth_snapshot(session, status, target_value)

        node_key = choose_weighted_frontier_node(session, eligible)
        addition = choose_candidate(session, node_key)
        session.turn_count += 1
        if addition is None:
            session.refresh_frontier_node(node_key)
            continue
        apply_growth_addition(session.group, addition)
        tree = session.group.trees[addition.tree_index]
        center_count = tree.child_count_by_paper[addition.center_paper_id]
        session.refresh_frontier_node(node_key)
        tree.achieved_counts_by_paper[addition.center_paper_id] = center_count
        tree.processed_center_ids.add(addition.center_paper_id)
        child_key = (addition.tree_index, addition.candidate_paper_id)
        session.add_expandable_node(child_key)
        session.maximum_depth = max(session.maximum_depth, addition.depth)
        if (
            session.num_children_per_node != UNLIMITED_CHILDREN_PER_NODE
            and center_count >= session.num_children_per_node
        ):
            tree.retirement_reason_by_paper[addition.center_paper_id] = (
                "num_children_budget_reached"
            )
            session.discard_expandable_node(node_key)
            state = session.frontier_states[node_key]
            state.clear_candidates()
            state.permanently_exhausted = True
        else:
            state = session.frontier_states[node_key]
            if not state.has_remaining() and not state.has_deferred():
                state.permanently_exhausted = True
                tree.retirement_reason_by_paper[addition.center_paper_id] = (
                    "citation_candidates_exhausted"
                )
                session.discard_expandable_node(node_key)


def equal_current_target_value(
    trees: Sequence[GrowthTreeState],
    target_kinds: Sequence[str],
) -> float | None:
    """Return the common current value, or ``None`` when trees differ."""
    values = [
        tree.target_value(target_kind) for tree, target_kind in zip(trees, target_kinds)
    ]
    if not values or not all(math.isclose(values[0], value) for value in values):
        return None
    return values[0]


__all__ = [
    "DIRECTION_CITED_BY",
    "DIRECTION_CITES",
    "GROWTH_REACHED",
    "GROWTH_SATURATED",
    "GROWTH_TARGET_BLOCKED",
    "TARGET_COMPONENT",
    "TARGET_MATCHING",
    "TARGET_YES",
    "CandidateEdge",
    "GrowthAdvanceResult",
    "GrowthBlockers",
    "GrowthTarget",
    "GrowthTreeState",
    "PaperCandidateFilter",
    "PaperFacts",
    "PaperFactsReader",
    "PreferentialGrowthSession",
    "ReferenceCalibration",
    "advance_preferential_growth_session",
    "center_can_improve_target",
    "combined_candidate_edges",
    "create_preferential_growth_session",
    "equal_current_target_value",
    "final_reference_calibration",
    "frontier_attachment_weight",
    "initialize_growth_tree",
    "plan_preferential_growth_addition",
    "validate_alpha",
    "validate_rendered_growth_tree",
]
