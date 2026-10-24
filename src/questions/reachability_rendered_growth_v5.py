"""Balanced rendered-edge growth for v5 reachability questions.

V5 keeps the rendered-edge disjointness rules from v4. It replaces anchor
saturation and reference-list caps with one shared child budget for every
expanded center paper. The budget counts accepted ``cites`` and ``cited_by``
children together. After each complete depth, v5 stops component trees with
empty frontiers and stops YES or matching trees whose frontier cannot advance
the smaller counter. Tied counters continue when the frontier can advance
either counter.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from src.questions.reachability_level_growth import (
    DIRECTION_CITED_BY,
    DIRECTION_CITES,
    TARGET_COMPONENT,
    TARGET_MATCHING,
    TARGET_YES,
    GrowthAddition,
    GrowthBlockers,
    GrowthGroupResult,
    GrowthTarget,
    GrowthTreeState,
    PaperExpansionState,
    PaperFacts,
    PaperFactsReader,
    QuestionGrowthGroup,
    ReferenceCalibration,
    addition_target_value,
    apply_growth_addition,
    expansion_state_for_paper,
    stable_growth_seed,
)
from src.questions.reachability_rendered_growth import (
    apply_center_promotion,
    candidate_conflict_reason,
    initialize_growth_tree,
    plan_center_promotion,
    prime_expansion_candidate_identities,
    primed_expansion_paper_ids,
    target_value_after_promotion,
    validate_rendered_growth_tree,
)

MINIMUM_IDENTITY_PREFETCH_BATCH_SIZE = 4

PaperCandidateFilter = Callable[[Sequence[str]], frozenset[str]]


@dataclass(frozen=True, slots=True)
class CandidatePool:
    """Immutable citation candidates cached for one center paper."""

    outgoing_paper_ids: tuple[str, ...]
    incoming_paper_ids: tuple[str, ...]
    cutoff_rejected_count: int


CandidatePoolCache = dict[str, CandidatePool]


@dataclass(slots=True)
class LevelGrowthSession:
    """Persistent v5 trees that can continue into a larger bucket."""

    trees: tuple[GrowthTreeState, ...]
    target_kinds: tuple[str, ...]
    facts_reader: PaperFactsReader
    include_author_identity: bool
    num_children_per_node: int
    random_seed: int | None
    group_key: Any
    blockers: GrowthBlockers | None
    paper_candidate_is_allowed: Callable[[str], bool] | None
    paper_candidate_filter: PaperCandidateFilter | None
    candidate_pool_cache: CandidatePoolCache | None
    rejection_counts: dict[str, int]
    maximum_depth_processed: int = 0
    advance_count: int = 0


def final_reference_calibration(
    trees: Sequence[GrowthTreeState],
) -> ReferenceCalibration:
    """Measure references only for final rendered paper blocks."""
    paper_ids: list[str] = []
    counts: list[int] = []
    for tree in trees:
        for paper_id in tree.expanded_paper_order:
            if paper_id not in tree.expanded_paper_ids:
                continue
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


def accepted_child_count(expansion: PaperExpansionState) -> int:
    """Return accepted children across both citation directions."""
    return expansion.accepted_outgoing + expansion.accepted_incoming


def candidate_identity_batch_size(num_children_per_node: int) -> int:
    """Return how many candidate identities one direction scan prefetches.

    A center stops at ``num_children_per_node`` accepted children, so a scan
    rarely reads further than that before it accepts or exhausts the queue.
    Prefetching more only issues database reads for candidates nothing looks
    at, and the waste grows with the tree size. The floor keeps a small child
    budget from making the batches too small to amortize.
    """
    return max(MINIMUM_IDENTITY_PREFETCH_BATCH_SIZE, num_children_per_node)


def candidate_pool_for_paper(
    paper_id: str,
    *,
    facts_reader: PaperFactsReader,
    paper_candidate_filter: PaperCandidateFilter | None,
    candidate_pool_cache: CandidatePoolCache | None,
) -> CandidatePool | None:
    """Return cached, cutoff-filtered candidates for one center paper."""
    normalized_id = str(paper_id)
    if candidate_pool_cache is not None:
        cached = candidate_pool_cache.get(normalized_id)
        if cached is not None:
            return cached

    facts = facts_reader.get(normalized_id)
    if facts is None:
        return None
    outgoing = facts.outgoing_paper_ids
    incoming = facts.incoming_paper_ids
    cutoff_rejected_count = 0
    if paper_candidate_filter is not None:
        allowed_ids = paper_candidate_filter((*outgoing, *incoming))
        filtered_outgoing = tuple(
            candidate_id for candidate_id in outgoing if candidate_id in allowed_ids
        )
        filtered_incoming = tuple(
            candidate_id for candidate_id in incoming if candidate_id in allowed_ids
        )
        cutoff_rejected_count = (
            len(outgoing)
            + len(incoming)
            - len(filtered_outgoing)
            - len(filtered_incoming)
        )
    else:
        filtered_outgoing = outgoing
        filtered_incoming = incoming
    pool = CandidatePool(
        outgoing_paper_ids=filtered_outgoing,
        incoming_paper_ids=filtered_incoming,
        cutoff_rejected_count=cutoff_rejected_count,
    )
    if candidate_pool_cache is not None:
        candidate_pool_cache[normalized_id] = pool
    return pool


def frontier_growth_stop_reason(
    tree: GrowthTreeState,
    target: GrowthTarget,
    next_frontier: set[str],
) -> str | None:
    """Return why one unfinished tree cannot make useful future progress."""
    if not next_frontier:
        return "frontier_exhausted"
    if target.kind == TARGET_COMPONENT:
        return None

    last_anchor_index = len(tree.witness) - 1
    if tree.source_cost_units == tree.target_cost_units:
        if any(
            tree.anchor_index_by_node[paper_id] < last_anchor_index
            or tree.anchor_index_by_node[paper_id] > 0
            for paper_id in next_frontier
        ):
            return None
        return "frontier_cannot_increase_either"

    if tree.source_cost_units < tree.target_cost_units:
        if any(
            tree.anchor_index_by_node[paper_id] < last_anchor_index
            for paper_id in next_frontier
        ):
            return None
        return "frontier_cannot_increase_source"

    if any(tree.anchor_index_by_node[paper_id] > 0 for paper_id in next_frontier):
        return None
    return "frontier_cannot_increase_target"


def plan_growth_addition(
    group: QuestionGrowthGroup,
    expansion: PaperExpansionState,
    *,
    direction: str,
    candidate_paper_id: str,
) -> tuple[GrowthAddition | None, str | None]:
    """Validate one v5 child without anchor-side saturation checks."""
    tree = group.trees[expansion.tree_index]
    center_id = expansion.paper_id
    if candidate_paper_id == center_id:
        return None, "self_loop"
    if candidate_paper_id in group.paper_owner:
        return None, "paper_already_visible"
    if group.blockers is not None and (candidate_paper_id in group.blockers.paper_ids):
        return None, "shared_paper_ids"

    candidate_facts = group.facts_reader.get_identity(candidate_paper_id)
    if candidate_facts is None or not candidate_facts.title_key:
        return None, "missing_candidate_identity"
    reason = candidate_conflict_reason(group, candidate_facts)
    if reason is not None:
        return None, reason
    if center_id not in tree.expanded_paper_ids:
        return None, "center_block_not_rendered"

    if direction == DIRECTION_CITES:
        reference_pair = (center_id, candidate_paper_id)
    elif direction == DIRECTION_CITED_BY:
        reference_pair = (candidate_paper_id, center_id)
    else:
        return None, "unsupported_direction"
    if reference_pair[0] == reference_pair[1]:
        return None, "self_loop"
    if reference_pair[1] in tree.references_by_paper.get(reference_pair[0], set()):
        return None, "reference_already_visible"

    added_author_ids = (
        {
            author_id
            for author_id in candidate_facts.author_ids
            if author_id not in tree.graph_author_ids
        }
        if group.include_author_identity and direction == DIRECTION_CITED_BY
        else set()
    )
    added_node_count = 1 + len(added_author_ids)
    anchor_index = tree.anchor_index_by_node[center_id]
    source_units = tree.source_cost_units
    target_units = tree.target_cost_units
    if anchor_index < len(tree.witness) - 1:
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
            promote_center=False,
            center_author_ids=frozenset(),
            added_graph_node_count=added_node_count,
            anchor_index=anchor_index,
            depth=expansion.depth + 1,
            resulting_source_cost_units=source_units,
            resulting_target_cost_units=target_units,
            resulting_component_size=tree.component_size + added_node_count,
        ),
        None,
    )


def find_direction_addition(
    group: QuestionGrowthGroup,
    expansion: PaperExpansionState,
    target: GrowthTarget,
    *,
    direction: str,
    rejection_counts: dict[str, int],
    identity_batch_size: int,
    paper_candidate_is_allowed: Callable[[str], bool] | None = None,
    target_blocked_centers: set[tuple[int, str]] | None = None,
) -> GrowthAddition | None:
    """Scan one shuffled direction queue until one child is accepted."""
    candidates = (
        expansion.outgoing_candidates
        if direction == DIRECTION_CITES
        else expansion.incoming_candidates
    )
    while candidates:
        prime_expansion_candidate_identities(
            group,
            expansion,
            direction=direction,
            batch_size=identity_batch_size,
        )
        candidate_id = candidates.pop()
        if paper_candidate_is_allowed is not None and not paper_candidate_is_allowed(
            candidate_id
        ):
            reason = "paper_candidate_not_allowed"
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        addition, reason = plan_growth_addition(
            group,
            expansion,
            direction=direction,
            candidate_paper_id=candidate_id,
        )
        if addition is not None and addition_target_value(
            addition, target.kind
        ) > float(target.value):
            addition = None
            reason = "target_overshoot"
            if target_blocked_centers is not None:
                target_blocked_centers.add((expansion.tree_index, expansion.paper_id))
        if addition is not None:
            return addition
        reason_key = reason or "candidate_rejected"
        rejection_counts[reason_key] = rejection_counts.get(reason_key, 0) + 1
    return None


def grow_question_group(
    initial_trees: Sequence[GrowthTreeState],
    targets: Sequence[GrowthTarget],
    *,
    facts_reader: PaperFactsReader,
    include_author_identity: bool,
    num_children_per_node: int,
    random_seed: int | None,
    group_key: Any,
    blockers: GrowthBlockers | None = None,
    progress_callback: Callable[[], None] | None = None,
    paper_candidate_is_allowed: Callable[[str], bool] | None = None,
    paper_candidate_filter: PaperCandidateFilter | None = None,
    candidate_pool_cache: CandidatePoolCache | None = None,
    resume_existing_trees: bool = False,
) -> GrowthGroupResult | None:
    """Grow trees shallow-first with an equal child budget per center."""
    if len(initial_trees) != len(targets) or not initial_trees:
        raise ValueError("Every growth tree requires exactly one target.")
    if (
        not isinstance(num_children_per_node, int)
        or isinstance(num_children_per_node, bool)
        or num_children_per_node <= 0
    ):
        raise ValueError("num_children_per_node must be positive.")
    for target in targets:
        target.validate()

    identity_batch_size = candidate_identity_batch_size(num_children_per_node)
    trees = [tree.clone() for tree in initial_trees]
    group = QuestionGrowthGroup(
        trees=trees,
        targets=list(targets),
        facts_reader=facts_reader,
        include_author_identity=include_author_identity,
        blockers=blockers,
    )
    if group.rebuild_registries() is not None:
        return None
    if any(tree.target_exceeded(target) for tree, target in zip(trees, targets)):
        return None

    if resume_existing_trees:
        frontiers = [
            resumable_frontier_paper_ids(tree, num_children_per_node) for tree in trees
        ]
    else:
        frontiers = [set(tree.gold_paper_ids) for tree in trees]
    stuck_tree_indexes: set[int] = set()
    rejection_counts: dict[str, int] = {}
    target_blocked_centers: set[tuple[int, str]] = set()
    depth = min(
        (
            tree.paper_depth.get(paper_id, 0)
            for tree, frontier in zip(trees, frontiers)
            for paper_id in frontier
        ),
        default=0,
    )

    def report_growth_progress() -> None:
        if progress_callback is not None:
            progress_callback()

    while True:
        report_growth_progress()
        active: list[PaperExpansionState] = []
        next_frontiers: list[set[str]] = [set() for tree in trees]
        for tree_index, tree in enumerate(trees):
            target = targets[tree_index]
            if tree.target_reached(target) or tree_index in stuck_tree_indexes:
                continue
            frontier_paper_ids = sorted(frontiers[tree_index])
            for paper_id in primed_expansion_paper_ids(
                facts_reader,
                frontier_paper_ids,
                cached_candidate_paper_ids=candidate_pool_cache,
            ):
                promotion, promotion_reason = plan_center_promotion(
                    group,
                    tree_index,
                    paper_id,
                )
                if promotion_reason is not None:
                    rejection_counts[promotion_reason] = (
                        rejection_counts.get(promotion_reason, 0) + 1
                    )
                    tree.retirement_reason_by_paper[paper_id] = promotion_reason
                    continue
                if promotion is not None:
                    proposed_value = target_value_after_promotion(
                        promotion,
                        target.kind,
                    )
                    if proposed_value > float(target.value):
                        reason = "center_promotion_target_overshoot"
                        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                        tree.retirement_reason_by_paper[paper_id] = reason
                        continue
                    apply_center_promotion(group, promotion)
                    if tree.target_reached(target):
                        continue
                candidate_pool = candidate_pool_for_paper(
                    paper_id,
                    facts_reader=facts_reader,
                    paper_candidate_filter=paper_candidate_filter,
                    candidate_pool_cache=candidate_pool_cache,
                )
                if candidate_pool is None:
                    reason = "missing_center_expansion"
                    tree.retirement_reason_by_paper[paper_id] = reason
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    continue
                if candidate_pool.cutoff_rejected_count:
                    reason = "paper_candidate_not_allowed_prefiltered"
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + (
                        candidate_pool.cutoff_rejected_count
                    )
                expansion = expansion_state_for_paper(
                    tree_index=tree_index,
                    tree=tree,
                    paper_id=paper_id,
                    depth=depth,
                    facts_reader=facts_reader,
                    calibration=None,
                    random_seed=random_seed,
                    group_key=group_key,
                    candidate_paper_ids=(
                        candidate_pool.outgoing_paper_ids,
                        candidate_pool.incoming_paper_ids,
                    ),
                )
                assert expansion is not None
                if resume_existing_trees:
                    expansion.accepted_outgoing = tree.child_count_by_paper.get(
                        paper_id,
                        0,
                    )
                tree.processed_center_ids.add(paper_id)
                active.append(expansion)

        sweep_index = 0
        while active:
            sweep = list(active)
            random.Random(
                stable_growth_seed(
                    random_seed,
                    ("v5-sweep", group_key, depth, sweep_index),
                )
            ).shuffle(sweep)
            next_active: list[PaperExpansionState] = []
            for expansion in sweep:
                tree = trees[expansion.tree_index]
                target = targets[expansion.tree_index]
                if tree.target_reached(target):
                    continue
                if accepted_child_count(expansion) >= num_children_per_node:
                    reason = "num_children_budget_reached"
                    expansion.retired_reason = reason
                    tree.retirement_reason_by_paper[expansion.paper_id] = reason
                    continue

                preferred_direction = random.Random(
                    stable_growth_seed(
                        random_seed,
                        (
                            "v5-direction",
                            group_key,
                            depth,
                            sweep_index,
                            expansion.tree_index,
                            expansion.paper_id,
                            expansion.attempts,
                        ),
                    )
                ).choice((DIRECTION_CITES, DIRECTION_CITED_BY))
                expansion.attempts += 1
                alternate_direction = (
                    DIRECTION_CITED_BY
                    if preferred_direction == DIRECTION_CITES
                    else DIRECTION_CITES
                )
                addition = find_direction_addition(
                    group,
                    expansion,
                    target,
                    direction=preferred_direction,
                    rejection_counts=rejection_counts,
                    identity_batch_size=identity_batch_size,
                    paper_candidate_is_allowed=paper_candidate_is_allowed,
                    target_blocked_centers=target_blocked_centers,
                )
                if addition is None:
                    addition = find_direction_addition(
                        group,
                        expansion,
                        target,
                        direction=alternate_direction,
                        rejection_counts=rejection_counts,
                        identity_batch_size=identity_batch_size,
                        paper_candidate_is_allowed=paper_candidate_is_allowed,
                        target_blocked_centers=target_blocked_centers,
                    )

                if addition is not None:
                    apply_growth_addition(group, addition)
                    next_frontiers[expansion.tree_index].add(
                        addition.candidate_paper_id
                    )
                    if addition.direction == DIRECTION_CITES:
                        expansion.accepted_outgoing += 1
                    else:
                        expansion.accepted_incoming += 1

                achieved = accepted_child_count(expansion)
                tree.achieved_counts_by_paper[expansion.paper_id] = achieved
                if tree.target_reached(target):
                    continue
                if achieved >= num_children_per_node:
                    reason = "num_children_budget_reached"
                    expansion.retired_reason = reason
                    tree.retirement_reason_by_paper[expansion.paper_id] = reason
                    continue
                if not (expansion.outgoing_candidates or expansion.incoming_candidates):
                    center_key = (expansion.tree_index, expansion.paper_id)
                    reason = (
                        "target_overshoot"
                        if center_key in target_blocked_centers
                        else "citation_candidates_exhausted"
                    )
                    expansion.retired_reason = reason
                    tree.retirement_reason_by_paper[expansion.paper_id] = reason
                    continue
                next_active.append(expansion)

            active = next_active
            sweep_index += 1
            report_growth_progress()

        if group.all_targets_reached():
            return GrowthGroupResult(
                trees=tuple(trees),
                calibration=final_reference_calibration(trees),
                all_targets_reached=True,
                maximum_depth_processed=depth,
                rejection_counts=rejection_counts,
            )

        for tree_index, tree in enumerate(trees):
            target = targets[tree_index]
            if tree.target_reached(target) or tree_index in stuck_tree_indexes:
                continue
            stop_reason = frontier_growth_stop_reason(
                tree,
                target,
                next_frontiers[tree_index],
            )
            if stop_reason is None:
                continue
            stuck_tree_indexes.add(tree_index)
            rejection_counts[stop_reason] = rejection_counts.get(stop_reason, 0) + 1

        if all(
            tree.target_reached(targets[tree_index]) or tree_index in stuck_tree_indexes
            for tree_index, tree in enumerate(trees)
        ):
            return GrowthGroupResult(
                trees=tuple(trees),
                calibration=final_reference_calibration(trees),
                all_targets_reached=False,
                maximum_depth_processed=depth,
                rejection_counts=rejection_counts,
            )
        frontiers = next_frontiers
        depth += 1


def resumable_frontier_paper_ids(
    tree: GrowthTreeState,
    num_children_per_node: int,
) -> set[str]:
    """Return the shallowest papers that may continue a saved v5 tree."""
    candidates: list[str] = []
    for paper_id in tree.visible_paper_ids:
        if tree.child_count_by_paper.get(paper_id, 0) >= num_children_per_node:
            continue
        retirement_reason = tree.retirement_reason_by_paper.get(paper_id)
        if paper_id not in tree.processed_center_ids or retirement_reason in {
            None,
            "target_overshoot",
            "center_promotion_target_overshoot",
        }:
            candidates.append(paper_id)
    if not candidates:
        return set()
    minimum_depth = min(tree.paper_depth.get(paper_id, 0) for paper_id in candidates)
    return {
        paper_id
        for paper_id in candidates
        if tree.paper_depth.get(paper_id, 0) == minimum_depth
    }


def create_level_growth_session(
    initial_trees: Sequence[GrowthTreeState],
    target_kinds: Sequence[str],
    *,
    facts_reader: PaperFactsReader,
    include_author_identity: bool,
    num_children_per_node: int,
    random_seed: int | None,
    group_key: Any,
    blockers: GrowthBlockers | None = None,
    paper_candidate_is_allowed: Callable[[str], bool] | None = None,
    paper_candidate_filter: PaperCandidateFilter | None = None,
    candidate_pool_cache: CandidatePoolCache | None = None,
) -> LevelGrowthSession | None:
    """Create a persistent session while validating the initial tree group."""
    if not initial_trees or len(initial_trees) != len(target_kinds):
        raise ValueError("Every growth tree requires exactly one target kind.")
    if (
        not isinstance(num_children_per_node, int)
        or isinstance(num_children_per_node, bool)
        or num_children_per_node <= 0
    ):
        raise ValueError("num_children_per_node must be positive.")
    targets = [GrowthTarget(kind, 0) for kind in target_kinds]
    for target in targets:
        target.validate()
    trees = tuple(tree.clone() for tree in initial_trees)
    group = QuestionGrowthGroup(
        trees=list(trees),
        targets=targets,
        facts_reader=facts_reader,
        include_author_identity=include_author_identity,
        blockers=blockers,
    )
    if group.rebuild_registries() is not None:
        return None
    return LevelGrowthSession(
        trees=trees,
        target_kinds=tuple(target_kinds),
        facts_reader=facts_reader,
        include_author_identity=include_author_identity,
        num_children_per_node=num_children_per_node,
        random_seed=random_seed,
        group_key=group_key,
        blockers=blockers,
        paper_candidate_is_allowed=paper_candidate_is_allowed,
        paper_candidate_filter=paper_candidate_filter,
        candidate_pool_cache=candidate_pool_cache,
        rejection_counts={},
    )


def advance_level_growth_session(
    session: LevelGrowthSession,
    target_value: float,
    *,
    progress_callback: Callable[[], None] | None = None,
) -> GrowthGroupResult | None:
    """Continue one v5 session toward a common larger target."""
    targets = [
        GrowthTarget(target_kind, target_value) for target_kind in session.target_kinds
    ]
    result = grow_question_group(
        session.trees,
        targets,
        facts_reader=session.facts_reader,
        include_author_identity=session.include_author_identity,
        num_children_per_node=session.num_children_per_node,
        random_seed=session.random_seed,
        group_key=(session.group_key, "advance", session.advance_count),
        blockers=session.blockers,
        progress_callback=progress_callback,
        paper_candidate_is_allowed=session.paper_candidate_is_allowed,
        paper_candidate_filter=session.paper_candidate_filter,
        candidate_pool_cache=session.candidate_pool_cache,
        resume_existing_trees=session.advance_count > 0,
    )
    session.advance_count += 1
    if result is None:
        return None
    session.trees = tuple(tree.clone() for tree in result.trees)
    session.maximum_depth_processed = max(
        session.maximum_depth_processed,
        result.maximum_depth_processed,
    )
    for reason, count in result.rejection_counts.items():
        session.rejection_counts[reason] = (
            session.rejection_counts.get(reason, 0) + count
        )
    return GrowthGroupResult(
        trees=tuple(tree.clone() for tree in session.trees),
        calibration=final_reference_calibration(session.trees),
        all_targets_reached=result.all_targets_reached,
        maximum_depth_processed=session.maximum_depth_processed,
        rejection_counts=dict(session.rejection_counts),
    )


__all__ = [
    "DIRECTION_CITED_BY",
    "DIRECTION_CITES",
    "TARGET_COMPONENT",
    "TARGET_MATCHING",
    "TARGET_YES",
    "CandidatePool",
    "CandidatePoolCache",
    "GrowthBlockers",
    "GrowthGroupResult",
    "GrowthTarget",
    "GrowthTreeState",
    "LevelGrowthSession",
    "PaperCandidateFilter",
    "PaperFacts",
    "PaperFactsReader",
    "QuestionGrowthGroup",
    "ReferenceCalibration",
    "accepted_child_count",
    "advance_level_growth_session",
    "candidate_identity_batch_size",
    "candidate_pool_for_paper",
    "create_level_growth_session",
    "final_reference_calibration",
    "frontier_growth_stop_reason",
    "grow_question_group",
    "initialize_growth_tree",
    "plan_growth_addition",
    "resumable_frontier_paper_ids",
    "validate_rendered_growth_tree",
]
