"""Frontier growth that may close cycles inside a component (create_dataset v3-v6).

Every step draws a center by the v8 (alpha) or v10 (theta, discount) frontier
weights, then one real citation link of the center. By default (v3/v4) this
is the v8/v10 draw: a fair coin picks the direction (references or citing
papers), then one link in it is drawn uniformly. With uniform_links (v5/v6)
the references and citing papers form one list instead, so each direction's
chance is its share of the center's remaining links. The candidate is checked
as in v8/v10, with one case that differs. When the neighbor is already a
paper of the center's own component and the two are not linked yet, v8/v10
reject it; here the real citation between them is rendered instead. Such a
cycle link adds no paper, so growth goes on until the component has its N
papers, now with N - 1 + (cycle links) links.

A cycle link is kept only when the component's shortest source-target path
keeps its length: the k-paper gold path (k - 1 links, read in either
direction, as the questions read them) stays a shortest path. Another path of
the same length is allowed. A neighbor in another component is still
rejected, so components stay disjoint.

The selection weights read each paper's count of extra links: its links in
the component apart from the ones it started with (the path links of a path
paper) or entered with (the link to the paper that brought it in). In a tree
this is the number of accepted children, as in v8/v10; a cycle link adds one
to both of its papers, and parent/child is only about who brought a paper in.
Without the cycle links, a component is exactly a v8/v10 tree, which
validate_cycle_growth_tree checks.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType

from src.questions import reachability_rendered_growth_v8 as alpha_growth
from src.questions import reachability_rendered_growth_v10 as crp_growth
from src.questions.reachability_level_growth import (
    DIRECTION_CITED_BY,
    DIRECTION_CITES,
    TARGET_COMPONENT,
    GrowthAddition,
    GrowthTarget,
    GrowthTreeState,
    PaperFactsReader,
    QuestionGrowthGroup,
    addition_target_value,
    apply_growth_addition,
)
from src.questions.reachability_rendered_growth import (
    CenterPromotion,
    apply_center_promotion,
    plan_center_promotion,
    validate_rendered_growth_tree,
)

GrowthSession = (
    alpha_growth.PreferentialGrowthSession | crp_growth.PreferentialGrowthSession
)
GrowthResult = alpha_growth.GrowthAdvanceResult | crp_growth.GrowthAdvanceResult
# Each frontier module's session type; the module supplies the selection weights.
FRONTIER_MODULES = {
    alpha_growth.PreferentialGrowthSession: alpha_growth,
    crp_growth.PreferentialGrowthSession: crp_growth,
}


@dataclass(frozen=True, slots=True)
class CycleLink:
    """One validated link between two papers already in the same component."""

    tree_index: int
    center_paper_id: str
    candidate_paper_id: str
    reference_pair: tuple[str, str]  # (owner, cited): the owner's block lists it.
    promotion: CenterPromotion | None


def paper_neighbors(tree: GrowthTreeState) -> dict[str, set[str]]:
    """Rendered links of one component, in both directions."""
    neighbors: dict[str, set[str]] = {
        paper_id: set() for paper_id in tree.visible_paper_ids
    }
    for owner_id, cited_ids in tree.references_by_paper.items():
        for cited_id in cited_ids:
            neighbors[owner_id].add(cited_id)
            neighbors[cited_id].add(owner_id)
    return neighbors


def link_counts_from(neighbors: Mapping[str, set[str]], start: str) -> dict[str, int]:
    """Breadth-first search: fewest links from ``start`` to every reachable paper."""
    counts = {start: 0}
    queue = deque([start])
    while queue:
        paper_id = queue.popleft()
        for neighbor_id in neighbors[paper_id]:
            if neighbor_id not in counts:
                counts[neighbor_id] = counts[paper_id] + 1
                queue.append(neighbor_id)
    return counts


@dataclass(slots=True)
class PathLengths:
    """Fewest links from one component's source and from its target to each paper.

    With both tables, whether one more link a-b shortens the source-target
    path is a constant-time check: the shortest path that uses the new link
    has from_source[a] + 1 + from_target[b] links (or the same with a and b
    swapped). A new paper with one link changes no other distance, so the
    tables are measured again (two breadth-first searches) only after a cycle
    link is added.
    """

    required: int  # k - 1 links for a k-paper gold path
    from_source: dict[str, int]
    from_target: dict[str, int]

    @classmethod
    def measure(cls, tree: GrowthTreeState) -> "PathLengths":
        neighbors = paper_neighbors(tree)
        lengths = cls(
            len(tree.witness) - 1,
            link_counts_from(neighbors, tree.source_node_id),
            link_counts_from(neighbors, tree.target_node_id),
        )
        if lengths.from_source.get(tree.target_node_id) != lengths.required:
            raise ValueError(
                f"Component {tree.path_id}: the shortest source-target path does "
                f"not have the {len(tree.witness)} papers of its gold path."
            )
        return lengths

    def add_leaf(self, paper_id: str, parent_id: str) -> None:
        self.from_source[paper_id] = self.from_source[parent_id] + 1
        self.from_target[paper_id] = self.from_target[parent_id] + 1

    def shortens(self, first_id: str, second_id: str) -> bool:
        """Whether a link first-second makes the source-target path shorter."""
        return (
            min(
                self.from_source[first_id] + 1 + self.from_target[second_id],
                self.from_source[second_id] + 1 + self.from_target[first_id],
            )
            < self.required
        )


def plan_cycle_link(
    group: QuestionGrowthGroup,
    tree_index: int,
    center_id: str,
    edge: crp_growth.CandidateEdge,
    lengths: PathLengths,
) -> tuple[CycleLink | None, str | None]:
    """Validate a link to ``edge.paper_id``, a paper of the center's component."""
    tree = group.trees[tree_index]
    candidate_id = edge.paper_id
    if candidate_id in tree.references_by_paper.get(
        center_id, ()
    ) or center_id in tree.references_by_paper.get(candidate_id, ()):
        return None, "cycle_papers_already_linked"
    if edge.direction == DIRECTION_CITES:
        reference_pair = (center_id, candidate_id)
    elif edge.direction == DIRECTION_CITED_BY:
        reference_pair = (candidate_id, center_id)
    else:
        return None, "unsupported_direction"
    if lengths.shortens(center_id, candidate_id):
        return None, "cycle_shortens_question_path"
    # As for a new child, the center itself becomes a block.
    promotion, reason = plan_center_promotion(group, tree_index, center_id)
    if reason is not None:
        return None, reason
    link = CycleLink(tree_index, center_id, candidate_id, reference_pair, promotion)
    return link, None


def apply_cycle_link(group: QuestionGrowthGroup, link: CycleLink) -> None:
    """Commit one planned cycle link."""
    tree = group.trees[link.tree_index]
    if link.promotion is not None:
        apply_center_promotion(group, link.promotion)
    owner_id, cited_id = link.reference_pair
    if owner_id not in tree.expanded_paper_ids:
        # cited_by: the candidate lists the center, so a paper that appeared
        # only in reference lists gets its own block.
        tree.expanded_paper_ids.add(owner_id)
        if owner_id not in tree.expanded_paper_seen_ids:
            tree.expanded_paper_order.append(owner_id)
            tree.expanded_paper_seen_ids.add(owner_id)
        tree.authors_by_paper[owner_id] = set()
    tree.references_by_paper.setdefault(owner_id, set()).add(cited_id)
    tree.cycle_center_by_pair[link.reference_pair] = link.center_paper_id
    # One more extra link for both papers (the weights' child_count_by_paper).
    tree.child_count_by_paper[owner_id] += 1
    tree.child_count_by_paper[cited_id] += 1


def draw_citation_link(
    rng: random.Random, state: crp_growth.FrontierNodeState
) -> tuple[str, crp_growth.CandidateEdge]:
    """Remove one of the center's remaining citation links, all equally likely.

    Returns its direction and the link. A paper that both cites the center and
    is cited by it has two links, one per direction.
    """
    cites = state.remaining_by_direction[DIRECTION_CITES]
    cited_by = state.remaining_by_direction[DIRECTION_CITED_BY]
    index = rng.randrange(len(cites) + len(cited_by))
    if index < len(cites):
        return DIRECTION_CITES, cites.pop(index)
    return DIRECTION_CITED_BY, cited_by.pop(index - len(cites))


def choose_step(
    frontier: ModuleType,
    session: GrowthSession,
    lengths: list[PathLengths],
    node_key: tuple[int, str],
    uniform_links: bool,
) -> GrowthAddition | CycleLink | None:
    """The frontier module's choose_candidate, plus links inside the component.

    Without uniform_links the draws use the same random calls in the same
    order, so a component where no cycle link is possible grows exactly as in
    v8/v10; with it, draw_citation_link replaces the coin flip.
    """
    tree_index, center_id = node_key
    group = session.group
    target = group.targets[tree_index]
    state = frontier.frontier_state(session, node_key)
    while state.has_remaining():
        if uniform_links:
            direction, edge = draw_citation_link(session.random_generator, state)
        else:
            direction = frontier.choose_available_direction(session, state)
            candidates = state.remaining_by_direction[direction]
            edge_index = session.random_generator.randrange(len(candidates))
            edge = candidates.pop(edge_index)
        owner_index = group.paper_owner.get(edge.paper_id)
        if edge.paper_id != center_id and owner_index == tree_index:
            step, reason = plan_cycle_link(
                group, tree_index, center_id, edge, lengths[tree_index]
            )
        else:
            step, reason = frontier.plan_preferential_growth_addition(
                group, tree_index, center_id, edge
            )
            if step is not None and (
                addition_target_value(step, target.kind) > float(target.value)
            ):
                state.deferred_by_direction[direction].append(edge)
                frontier.record_rejection(session, "target_overshoot")
                continue
        if step is not None:
            return step
        frontier.record_rejection(session, reason or "candidate_rejected")
    if not state.has_deferred():
        state.permanently_exhausted = True
        session.discard_expandable_node(node_key)
        tree = group.trees[tree_index]
        tree.achieved_counts_by_paper[center_id] = tree.child_count_by_paper.get(
            center_id, 0
        )
        tree.retirement_reason_by_paper[center_id] = "citation_candidates_exhausted"
    return None


def advance_cycle_growth_session(
    session: GrowthSession, target_value: int, *, uniform_links: bool = False
) -> GrowthResult:
    """Grow every component to ``target_value`` papers; may close cycles.

    ``session`` comes from create_preferential_growth_session of the v8 or v10
    growth module, whose weights choose the centers. ``uniform_links`` draws
    each center's citation link uniformly over both directions instead of
    flipping a coin for the direction first. Like that module's
    advance_preferential_growth_session, this returns its GrowthAdvanceResult.
    """
    frontier = FRONTIER_MODULES[type(session)]
    group = session.group
    if group.include_author_identity or any(
        target.kind != TARGET_COMPONENT for target in group.targets
    ):
        raise ValueError("Cycle growth supports paper-only node-count targets.")
    targets = [GrowthTarget(TARGET_COMPONENT, target_value) for _tree in group.trees]
    for target in targets:
        target.validate()
    group.targets = targets
    frontier.reset_deferred_candidates(session)
    session.eligibility_signature = ()
    lengths = [PathLengths.measure(tree) for tree in group.trees]

    if any(tree.target_exceeded(target) for tree, target in zip(group.trees, targets)):
        frontier.record_rejection(session, "current_value_exceeds_target")
        return frontier.growth_snapshot(
            session, frontier.GROWTH_TARGET_BLOCKED, target_value
        )

    while True:
        if group.all_targets_reached():
            return frontier.growth_snapshot(
                session, frontier.GROWTH_REACHED, target_value
            )
        eligible = frontier.eligible_frontier_nodes(session)
        if not eligible:
            status = (
                frontier.GROWTH_TARGET_BLOCKED
                if frontier.deferred_candidate_exists(session)
                else frontier.GROWTH_SATURATED
            )
            frontier.record_rejection(session, status)
            return frontier.growth_snapshot(session, status, target_value)

        node_key = frontier.choose_weighted_frontier_node(session, eligible)
        step = choose_step(frontier, session, lengths, node_key, uniform_links)
        session.turn_count += 1
        if step is None:
            session.refresh_frontier_node(node_key)
            continue
        tree = group.trees[step.tree_index]
        if isinstance(step, CycleLink):
            apply_cycle_link(group, step)
            lengths[step.tree_index] = PathLengths.measure(tree)
            # Its count changed too; v8 keeps each eligible paper's weight.
            session.refresh_frontier_node((step.tree_index, step.candidate_paper_id))
        else:
            apply_growth_addition(group, step)
            lengths[step.tree_index].add_leaf(
                step.candidate_paper_id, step.center_paper_id
            )
        center_count = tree.child_count_by_paper[step.center_paper_id]
        session.refresh_frontier_node(node_key)
        tree.achieved_counts_by_paper[step.center_paper_id] = center_count
        tree.processed_center_ids.add(step.center_paper_id)
        if isinstance(step, GrowthAddition):
            session.add_expandable_node((step.tree_index, step.candidate_paper_id))
            session.maximum_depth = max(session.maximum_depth, step.depth)
        state = session.frontier_states[node_key]
        if (
            session.num_children_per_node != frontier.UNLIMITED_CHILDREN_PER_NODE
            and center_count >= session.num_children_per_node
        ):
            tree.retirement_reason_by_paper[step.center_paper_id] = (
                "num_children_budget_reached"
            )
            session.discard_expandable_node(node_key)
            state.clear_candidates()
            state.permanently_exhausted = True
        elif not state.has_remaining() and not state.has_deferred():
            state.permanently_exhausted = True
            tree.retirement_reason_by_paper[step.center_paper_id] = (
                "citation_candidates_exhausted"
            )
            session.discard_expandable_node(node_key)


def tree_part(tree: GrowthTreeState) -> GrowthTreeState:
    """A copy without the cycle links: the tree v8/v10 growth would have built."""
    part = tree.clone()
    for owner_id, cited_id in tree.cycle_center_by_pair:
        part.references_by_paper[owner_id].discard(cited_id)
        part.child_count_by_paper[owner_id] -= 1
        part.child_count_by_paper[cited_id] -= 1
    part.cycle_center_by_pair.clear()
    return part


def validate_cycle_growth_tree(
    tree: GrowthTreeState, *, facts_reader: PaperFactsReader
) -> bool:
    """Check a paper-only component grown by advance_cycle_growth_session.

    Each cycle link is rendered, joins two papers not linked otherwise, and
    names one of them as its center; without the cycle links the component
    passes the v8/v10 tree checks; and the shortest source-target path still
    has the papers of the gold path.
    """
    links = {
        frozenset(pair)
        for pair in tree.protected_reference_pairs | tree.optional_reference_pairs
    }
    for pair, center_id in tree.cycle_center_by_pair.items():
        owner_id, cited_id = pair
        if (
            center_id not in pair
            or owner_id == cited_id
            or cited_id not in tree.references_by_paper.get(owner_id, ())
            or frozenset(pair) in links
        ):
            return False
        links.add(frozenset(pair))
    if not validate_rendered_growth_tree(
        tree_part(tree), facts_reader=facts_reader, include_author_identity=False
    ):
        return False
    counts = link_counts_from(paper_neighbors(tree), tree.source_node_id)
    return counts.get(tree.target_node_id) == len(tree.witness) - 1


__all__ = [
    "CycleLink",
    "PathLengths",
    "advance_cycle_growth_session",
    "apply_cycle_link",
    "plan_cycle_link",
    "tree_part",
    "validate_cycle_growth_tree",
]
