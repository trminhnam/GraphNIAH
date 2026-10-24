"""Rendered-edge, depth-level growth for v4 reachability questions.

Only selected citation edges belong to the question graph. Candidate
admission uses visible identity, while citation adjacency is loaded only when
the paper is processed as a center.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.questions.reachability_level_growth import (
    DIRECTION_CITED_BY,
    DIRECTION_CITES,
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
    TARGET_COMPONENT,
    TARGET_MATCHING,
    TARGET_YES,
    addition_target_value,
    apply_growth_addition,
    expansion_state_for_paper,
    initialize_growth_tree as initialize_base_growth_tree,
    reference_calibration,
    stable_growth_seed,
    validate_incremental_complexity,
)


IDENTITY_PREFETCH_BATCH_SIZE = 128
FACTS_PREFETCH_BATCH_SIZE = 128


@dataclass(frozen=True, slots=True)
class CenterPromotion:
    """One terminal-to-block promotion ready to commit."""

    tree_index: int
    paper_id: str
    author_ids: frozenset[str]
    anchor_index: int
    resulting_source_cost_units: int
    resulting_target_cost_units: int
    resulting_component_size: int


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
) -> GrowthTreeState | None:
    """Build a thin tree from rendered edges without inspecting hidden edges."""
    # These identities are all needed by initialization. Unlike speculative
    # neighbor prefetch, this batch does not read unused candidate metadata.
    facts_reader.prime_identities(expanded_paper_ids)
    return initialize_base_growth_tree(
        cache_key=cache_key,
        path_id=path_id,
        orientation_index=orientation_index,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        witness=witness,
        source_wrong_path=source_wrong_path,
        target_wrong_path=target_wrong_path,
        gold_paper_ids=gold_paper_ids,
        expanded_paper_ids=expanded_paper_ids,
        authors_by_paper=authors_by_paper,
        references_by_paper=references_by_paper,
        protected_reference_pairs=protected_reference_pairs,
        facts_reader=facts_reader,
        include_author_identity=include_author_identity,
        rendered_edges_only=True,
    )


def candidate_conflict_reason(
    group: QuestionGrowthGroup,
    candidate_facts: PaperFacts,
) -> str | None:
    """Return a visible-identity conflict for one proposed new node."""
    blockers = group.blockers
    if blockers is not None:
        if candidate_facts.title_key in blockers.title_keys:
            return "shared_normalized_paper_title"
        if group.include_author_identity:
            if blockers.author_ids.intersection(candidate_facts.author_ids):
                return "shared_author_id"
            if blockers.author_name_keys.intersection(
                candidate_facts.author_name_keys
            ):
                return "shared_normalized_author_name"
    if candidate_facts.title_key in group.used_title_keys:
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
    return None


def anchor_can_improve_target(
    tree: GrowthTreeState,
    target: GrowthTarget,
    paper_id: str,
) -> bool:
    """Return whether descendants at this anchor can still raise the target."""
    if target.kind != TARGET_YES:
        return True
    anchor_index = tree.anchor_index_by_node.get(paper_id)
    if anchor_index is None:
        return False
    target_units = int(round(float(target.value) * 2.0))
    can_raise_source = (
        anchor_index < len(tree.witness) - 1
        and tree.source_cost_units < target_units
    )
    can_raise_target = (
        anchor_index > 0
        and tree.target_cost_units < target_units
    )
    return can_raise_source or can_raise_target


def plan_growth_addition(
    group: QuestionGrowthGroup,
    expansion: PaperExpansionState,
    *,
    direction: str,
    candidate_paper_id: str,
) -> tuple[GrowthAddition | None, str | None]:
    """Validate one rendered-edge candidate without loading its adjacency."""
    tree = group.trees[expansion.tree_index]
    target = group.targets[expansion.tree_index]
    if not anchor_can_improve_target(tree, target, expansion.paper_id):
        return None, "anchor_side_saturated"
    candidate_facts = group.facts_reader.get_identity(candidate_paper_id)
    if candidate_facts is None or not candidate_facts.title_key:
        return None, "missing_candidate_identity"
    reason = candidate_conflict_reason(group, candidate_facts)
    if reason is not None:
        return None, reason
    center_id = expansion.paper_id
    if center_id not in tree.expanded_paper_ids:
        return None, "center_block_not_rendered"
    if direction == DIRECTION_CITES:
        reference_pair = (center_id, candidate_paper_id)
    elif direction == DIRECTION_CITED_BY:
        reference_pair = (candidate_paper_id, center_id)
    else:
        return None, "unsupported_direction"
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


def prime_expansion_candidate_identities(
    group: QuestionGrowthGroup,
    expansion: PaperExpansionState,
    *,
    direction: str,
    batch_size: int = IDENTITY_PREFETCH_BATCH_SIZE,
) -> None:
    """Prefetch the next small candidate batch before individual checks.

    ``batch_size`` bounds the prefetch. Every identity beyond the candidates
    this scan actually pops is a wasted database round trip, so callers that
    know their own per-center child budget should pass it.
    """
    if direction == DIRECTION_CITES:
        candidates = expansion.outgoing_candidates
    else:
        candidates = expansion.incoming_candidates
    if (
        not candidates
        or candidates[-1] in group.facts_reader.identity_cache
    ):
        return
    upper = len(candidates)
    lower = max(0, upper - batch_size)
    group.facts_reader.prime_identities(candidates[lower:upper])


def primed_expansion_paper_ids(
    facts_reader: PaperFactsReader,
    paper_ids: Sequence[str],
    cached_candidate_paper_ids: Mapping[str, Any] | None = None,
) -> Iterable[str]:
    """Yield frontier papers after loading each bounded adjacency batch."""
    for start in range(0, len(paper_ids), FACTS_PREFETCH_BATCH_SIZE):
        chunk = paper_ids[start : start + FACTS_PREFETCH_BATCH_SIZE]
        if cached_candidate_paper_ids is None:
            facts_reader.prime_facts(chunk)
        else:
            facts_reader.prime_facts(
                [
                    paper_id
                    for paper_id in chunk
                    if paper_id not in cached_candidate_paper_ids
                ]
            )
        yield from chunk


def target_value_after_promotion(
    promotion: CenterPromotion,
    target_kind: str,
) -> float:
    """Return the target value after one center promotion."""
    if target_kind == TARGET_COMPONENT:
        return float(promotion.resulting_component_size)
    yes_value = float(
        min(
            promotion.resulting_source_cost_units,
            promotion.resulting_target_cost_units,
        )
    ) / 2.0
    if target_kind == TARGET_YES:
        return yes_value
    if target_kind == TARGET_MATCHING:
        return float(math.ceil(yes_value))
    raise ValueError(f"Unsupported growth target kind: {target_kind!r}.")


def plan_center_promotion(
    group: QuestionGrowthGroup,
    tree_index: int,
    paper_id: str,
) -> tuple[CenterPromotion | None, str | None]:
    """Plan rendering a scheduled terminal paper as a full block."""
    tree = group.trees[tree_index]
    if paper_id in tree.expanded_paper_ids:
        return None, None
    facts = group.facts_reader.get_identity(paper_id)
    if facts is None or not facts.title_key:
        return None, "missing_center_identity"
    author_ids = (
        frozenset(facts.author_ids)
        if group.include_author_identity
        else frozenset()
    )
    added_author_ids = {
        author_id
        for author_id in author_ids
        if author_id not in tree.graph_author_ids
    }
    anchor_index = tree.anchor_index_by_node[paper_id]
    source_units = tree.source_cost_units
    target_units = tree.target_cost_units
    if anchor_index < len(tree.witness) - 1:
        source_units += len(added_author_ids)
    if anchor_index > 0:
        target_units += len(added_author_ids)
    return (
        CenterPromotion(
            tree_index=tree_index,
            paper_id=paper_id,
            author_ids=author_ids,
            anchor_index=anchor_index,
            resulting_source_cost_units=source_units,
            resulting_target_cost_units=target_units,
            resulting_component_size=(
                tree.component_size + len(added_author_ids)
            ),
        ),
        None,
    )


def apply_center_promotion(
    group: QuestionGrowthGroup,
    promotion: CenterPromotion,
) -> None:
    """Commit one planned terminal-to-block promotion."""
    tree = group.trees[promotion.tree_index]
    paper_id = promotion.paper_id
    tree.expanded_paper_ids.add(paper_id)
    if paper_id not in tree.expanded_paper_seen_ids:
        tree.expanded_paper_order.append(paper_id)
        tree.expanded_paper_seen_ids.add(paper_id)
    tree.authors_by_paper[paper_id] = set(promotion.author_ids)
    tree.references_by_paper.setdefault(paper_id, set())
    tree.graph_author_ids.update(promotion.author_ids)
    for author_id in promotion.author_ids:
        tree.anchor_index_by_node[author_id] = promotion.anchor_index
    tree.source_cost_units = promotion.resulting_source_cost_units
    tree.target_cost_units = promotion.resulting_target_cost_units
    tree.component_size = promotion.resulting_component_size


def validate_rendered_growth_tree(
    tree: GrowthTreeState,
    *,
    facts_reader: PaperFactsReader,
    include_author_identity: bool,
) -> bool:
    """Validate v4 state using rendered nodes, edges, and identities only."""
    visible_ids = set(tree.visible_paper_ids)
    expanded_ids = set(tree.expanded_paper_ids)
    gold_ids = set(tree.gold_paper_ids)
    rendered_pairs = {
        (owner_id, cited_id)
        for owner_id, cited_ids in tree.references_by_paper.items()
        for cited_id in cited_ids
    }
    if (
        not validate_incremental_complexity(tree)
        or not gold_ids.issubset(expanded_ids)
        or not expanded_ids.issubset(visible_ids)
        or not tree.processed_center_ids.issubset(expanded_ids)
        or set(tree.references_by_paper) != expanded_ids
        or set(tree.authors_by_paper) != expanded_ids
        or set(tree.paper_depth) != visible_ids
        or set(tree.child_count_by_paper) != visible_ids
        or set(tree.parent_by_paper) != visible_ids - gold_ids
        or set(tree.discovery_pair_by_paper) != visible_ids - gold_ids
        or tree.optional_reference_pairs
        != set(tree.discovery_pair_by_paper.values())
        or rendered_pairs
        != tree.protected_reference_pairs | tree.optional_reference_pairs
        or any(
            cited_id not in visible_ids
            for _owner_id, cited_id in rendered_pairs
        )
        or tree.used_title_keys != set(tree.title_owner_by_key)
        or set(tree.title_owner_by_key.values()) != visible_ids
        or len(tree.title_owner_by_key) != len(visible_ids)
    ):
        return False

    expected_author_owner_by_id: dict[str, str] = {}
    expected_author_owner_by_name: dict[str, str] = {}
    for paper_id in visible_ids:
        facts = facts_reader.get_identity(paper_id)
        if (
            facts is None
            or not facts.title_key
            or tree.title_owner_by_key.get(facts.title_key) != paper_id
        ):
            return False
        if include_author_identity:
            for author_id in facts.author_ids:
                previous = expected_author_owner_by_id.get(author_id)
                if previous is not None and previous != paper_id:
                    return False
                expected_author_owner_by_id[author_id] = paper_id
            for name_key in facts.author_name_keys:
                previous = expected_author_owner_by_name.get(name_key)
                if previous is not None and previous != paper_id:
                    return False
                expected_author_owner_by_name[name_key] = paper_id

    for paper_id, parent_id in tree.parent_by_paper.items():
        pair = tree.discovery_pair_by_paper[paper_id]
        if (
            parent_id not in visible_ids
            or pair not in rendered_pairs
            or set(pair) != {paper_id, parent_id}
            or (pair[0] == paper_id and paper_id not in expanded_ids)
            or (
                paper_id in tree.processed_center_ids
                and paper_id not in expanded_ids
            )
        ):
            return False

    for paper_id in visible_ids - expanded_ids:
        owners = {
            owner_id
            for owner_id, cited_id in rendered_pairs
            if cited_id == paper_id
        }
        if (
            len(owners) != 1
            or tree.discovery_pair_by_paper.get(paper_id)
            != (next(iter(owners)), paper_id)
        ):
            return False

    rendered_author_ids = {
        author_id
        for author_ids in tree.authors_by_paper.values()
        for author_id in author_ids
    }
    if rendered_author_ids != tree.graph_author_ids:
        return False
    if not include_author_identity:
        return not tree.author_owner_by_id and not tree.author_owner_by_name
    return (
        tree.author_owner_by_id == expected_author_owner_by_id
        and tree.author_owner_by_name == expected_author_owner_by_name
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
    progress_callback: Callable[[], None] | None = None,
) -> GrowthGroupResult | None:
    """Grow rendered trees using randomized depth-level round robin."""
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
    if group.rebuild_registries() is not None:
        return None
    if any(tree.target_exceeded(target) for tree, target in zip(trees, targets)):
        return None

    frontiers: list[set[str]] = [set(tree.gold_paper_ids) for tree in trees]
    calibration = initial_calibration
    rejection_counts: dict[str, int] = {}
    depth = 0

    def report_growth_progress() -> None:
        if progress_callback is not None:
            progress_callback()

    while True:
        report_growth_progress()
        active: list[PaperExpansionState] = []
        next_frontiers: list[set[str]] = [set() for tree in trees]
        for tree_index, tree in enumerate(trees):
            target = targets[tree_index]
            if tree.target_reached(target):
                continue
            frontier_paper_ids: list[str] = []
            for paper_id in sorted(frontiers[tree_index]):
                if anchor_can_improve_target(tree, target, paper_id):
                    frontier_paper_ids.append(paper_id)
                    continue
                reason = "anchor_side_saturated"
                tree.retirement_reason_by_paper[paper_id] = reason
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            for paper_id in primed_expansion_paper_ids(
                facts_reader,
                frontier_paper_ids,
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
                    continue
                if promotion is not None:
                    proposed_value = target_value_after_promotion(
                        promotion,
                        target.kind,
                    )
                    if proposed_value > float(target.value):
                        reason = "center_promotion_target_overshoot"
                        rejection_counts[reason] = (
                            rejection_counts.get(reason, 0) + 1
                        )
                        tree.retirement_reason_by_paper[paper_id] = reason
                        continue
                    apply_center_promotion(group, promotion)
                    if tree.target_reached(target):
                        continue
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
                if expansion is None:
                    tree.retirement_reason_by_paper[paper_id] = (
                        "missing_center_expansion"
                    )
                    continue
                tree.processed_center_ids.add(paper_id)
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
                if not anchor_can_improve_target(
                    tree,
                    target,
                    expansion.paper_id,
                ):
                    reason = "anchor_side_saturated"
                    expansion.retired_reason = reason
                    tree.retirement_reason_by_paper[expansion.paper_id] = reason
                    rejection_counts[reason] = (
                        rejection_counts.get(reason, 0) + 1
                    )
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
                    prime_expansion_candidate_identities(
                        group,
                        expansion,
                        direction=DIRECTION_CITES,
                    )
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

                directions = [DIRECTION_CITES]
                if expansion.incoming_candidates:
                    directions.append(DIRECTION_CITED_BY)
                direction = random.Random(
                    stable_growth_seed(
                        random_seed,
                        (
                            "direction",
                            group_key,
                            depth,
                            sweep_index,
                            expansion.tree_index,
                            expansion.paper_id,
                            expansion.attempts,
                        ),
                    )
                ).choice(directions)
                expansion.attempts += 1
                if direction == DIRECTION_CITES:
                    expansion.outgoing_candidates.pop()
                    addition = outgoing_addition
                    reason = None
                else:
                    prime_expansion_candidate_identities(
                        group,
                        expansion,
                        direction=DIRECTION_CITED_BY,
                    )
                    candidate_id = expansion.incoming_candidates.pop()
                    addition, reason = plan_growth_addition(
                        group,
                        expansion,
                        direction=direction,
                        candidate_paper_id=candidate_id,
                    )
                    if addition is not None and addition_target_value(
                        addition,
                        target.kind,
                    ) > float(target.value):
                        addition = None
                        reason = "target_overshoot"
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
                tree.achieved_counts_by_paper[expansion.paper_id] = len(
                    tree.references_by_paper.get(expansion.paper_id, set())
                )
                if (
                    not tree.target_reached(target)
                    and not expansion.retire_if_needed(tree)
                ):
                    next_active.append(expansion)
                elif expansion.retired_reason:
                    tree.retirement_reason_by_paper[expansion.paper_id] = (
                        expansion.retired_reason
                    )
            active = next_active
            sweep_index += 1
            report_growth_progress()

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


__all__ = [
    "DIRECTION_CITED_BY",
    "DIRECTION_CITES",
    "GrowthBlockers",
    "GrowthGroupResult",
    "GrowthTarget",
    "GrowthTreeState",
    "PaperFacts",
    "PaperFactsReader",
    "QuestionGrowthGroup",
    "ReferenceCalibration",
    "TARGET_COMPONENT",
    "TARGET_MATCHING",
    "TARGET_YES",
    "apply_center_promotion",
    "candidate_conflict_reason",
    "grow_question_group",
    "initialize_growth_tree",
    "plan_center_promotion",
    "plan_growth_addition",
    "validate_rendered_growth_tree",
]
