"""Shared construction helpers for v5 reachability questions.

Thin-tree construction, rendered-edge identity checks, exact targeting, and
fallback pruning stay compatible with v4. V5 changes only the growth
scheduler and the statistics recorded for its final rendered paper blocks.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.questions import reachability_construction as yesno_base
from src.prompts.reachability_yesno_qa import is_paper_only_traversal_mode
from src.questions.reachability_crossed_trees import path_tree_cache_key
from src.questions import reachability_rendered_pipeline as v4
from src.questions.reachability_rendered_growth_v5 import (
    TARGET_COMPONENT,
    TARGET_YES,
    CandidatePoolCache,
    GrowthGroupResult,
    GrowthTarget,
    GrowthTreeState,
    LevelGrowthSession,
    PaperCandidateFilter,
    PaperFacts,
    PaperFactsReader,
    QuestionGrowthGroup,
    ReferenceCalibration,
    advance_level_growth_session,
    create_level_growth_session,
    grow_question_group,
    initialize_growth_tree,
    validate_rendered_growth_tree,
)
from src.questions.shared import (
    NEEDLE_REFERENCE_STATISTICS_FIELD,
    prepare_question_path,
)
from src.setup.constants import (
    EDGE_CITED_BY,
    EDGE_CITING,
    NODE_PAPER,
    nid_db_id,
    nid_type,
)
from src.utils.resume import load_json_output, write_json_output
from src.questions.explored import entry_is_explored

V5_GROWTH_FIELD = "level_growth"
V5_PAIR_CACHE_VERSION = 1
V5_TREE_CONSTRUCTION_STRATEGY = "rendered-edge-balanced-children-level-growth-v3"
V5_RENDERED_GRAPH_POLICY_VERSION = 1
V5_PAPER_IDENTITY_POLICY = "normalized-title-single-node-v1"
V5_CANDIDATE_COMMIT_POLICY = "render-before-next-depth-v1"
V5_NEGATIVE_DISTRACTOR_TREE_VERSION = 2
V5_NEGATIVE_DISTRACTOR_TREE_SUPPORTED_VERSIONS = frozenset({1, 2})
V5_PAPER_BLOCK_POLICY = "nonempty-selected-references-v1"
PERMANENT_PATH_FAILURE_REASONS_FIELD = "permanent_path_failure_reasons"

PreparedGrowthPath = v4.PreparedGrowthPath
GrowthPlan = v4.GrowthPlan
ReusableGrowthGroup = v4.ReusableGrowthGroup
PairDisjointnessCache = v4.PairDisjointnessCache

record_final_invariant_failure = v4.record_final_invariant_failure
stable_order = v4.stable_order
orientation_order = v4.orientation_order
paper_facts_reader = v4.paper_facts_reader
negative_component_target = v4.negative_component_target
paper_author_node_ids = v4.paper_author_node_ids
legal_target_values = v4.legal_target_values
select_uniform_target = v4.select_uniform_target
record_accepted_question = v4.record_accepted_question
pairwise_thin_disjointness_reason = v4.pairwise_thin_disjointness_reason
growth_state_variant = v4.growth_state_variant
combine_growth_variants = v4.combine_growth_variants
negative_distractor_tree_signature = v4.negative_distractor_tree_signature
plan_signature = v4.plan_signature


def validate_num_children_per_node(value: int) -> None:
    """Require one positive, non-Boolean child budget."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("num_children_per_node must be positive.")


def parse_cutoff_date(raw: str) -> date:
    """Parse one strict ``YYYY-MM-DD`` cutoff date."""
    text = str(raw).strip()
    if len(text) != 10:
        raise ValueError("cutoff date must use YYYY-MM-DD format")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"invalid cutoff date: {raw!r}") from error
    if parsed.isoformat() != text:
        raise ValueError("cutoff date must use YYYY-MM-DD format")
    return parsed


def resolve_cutoff_date(
    cutoff_date: str | date | None,
) -> tuple[date | None, str | None]:
    """Return the cutoff date and its normalized output value."""
    if cutoff_date is None:
        return None, None
    if type(cutoff_date) is date:
        return cutoff_date, cutoff_date.isoformat()
    if isinstance(cutoff_date, str):
        parsed = parse_cutoff_date(cutoff_date)
        return parsed, parsed.isoformat()
    raise ValueError("cutoff_date must be a date, YYYY-MM-DD string, or None.")


def parse_published_date(raw: Any) -> date | None:
    """Parse one stored paper publication value."""
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except (TypeError, ValueError):
        pass
    if len(text) < 10:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


@dataclass(slots=True)
class PaperDateCutoff:
    """Cache and enforce an inclusive minimum publication date."""

    db: Any
    cutoff_date: date | None
    published_dates: dict[str, date | None]

    def prime_published_dates(self, paper_ids: Sequence[str]) -> None:
        """Load uncached publication dates in batches when supported."""
        paper_ids_by_db_id: dict[str, list[str]] = {}
        for paper_id in paper_ids:
            normalized_id = str(paper_id)
            if normalized_id in self.published_dates:
                continue
            try:
                db_id = nid_db_id(normalized_id)
            except (IndexError, ValueError):
                self.published_dates[normalized_id] = None
                continue
            paper_ids_by_db_id.setdefault(db_id, []).append(normalized_id)

        if not paper_ids_by_db_id:
            return
        batch_loader = getattr(self.db, "get_paper_published_many", None)
        if callable(batch_loader):
            published_by_db_id = batch_loader(list(paper_ids_by_db_id))
        else:
            published_by_db_id = {
                db_id: self.db.get_paper_published(db_id)
                for db_id in paper_ids_by_db_id
            }
        for db_id, paper_ids_for_db_id in paper_ids_by_db_id.items():
            published_date = parse_published_date(
                published_by_db_id.get(db_id, "")
            )
            for paper_id in paper_ids_for_db_id:
                self.published_dates[paper_id] = published_date

    def allowed_paper_ids(self, paper_ids: Sequence[str]) -> frozenset[str]:
        """Return IDs that satisfy the configured publication cutoff."""
        normalized_ids = tuple(str(paper_id) for paper_id in paper_ids)
        if self.cutoff_date is None:
            return frozenset(normalized_ids)
        self.prime_published_dates(normalized_ids)
        return frozenset(
            paper_id for paper_id in normalized_ids if self.allows(paper_id)
        )

    def published_date(self, paper_id: str) -> date | None:
        """Return one cached paper date."""
        normalized_id = str(paper_id)
        if normalized_id not in self.published_dates:
            self.prime_published_dates((normalized_id,))
        return self.published_dates[normalized_id]

    def rejection_reason(self, paper_id: str) -> str | None:
        """Return why a paper fails the configured cutoff."""
        if self.cutoff_date is None:
            return None
        published = self.published_date(paper_id)
        if published is None:
            return "missing_or_invalid_paper_publication_date"
        if published < self.cutoff_date:
            return "paper_before_cutoff_date"
        return None

    def allows(self, paper_id: str) -> bool:
        """Return whether a paper satisfies the configured cutoff."""
        return self.rejection_reason(paper_id) is None


def initial_tree_paper_ids(entry: Mapping[str, Any]) -> tuple[str, ...]:
    """Return papers selected by a simple or profiled thin path."""
    raw_node_ids: list[Any] = []
    path = entry.get("path") or entry.get("steps") or []
    if isinstance(path, list):
        raw_node_ids.extend(
            step.get("node", {}).get("node_id")
            for step in path
            if isinstance(step, Mapping) and isinstance(step.get("node"), Mapping)
        )
    profile = entry.get("reachability_long_path_profile")
    if isinstance(profile, Mapping):
        for branch_name in ("source_wrong_branch", "target_wrong_branch"):
            branch = profile.get(branch_name)
            if isinstance(branch, Mapping):
                node_ids = branch.get("node_ids")
                if isinstance(node_ids, list):
                    raw_node_ids.extend(node_ids)
    paper_ids: list[str] = []
    for raw_node_id in raw_node_ids:
        node_id = str(raw_node_id or "")
        try:
            is_paper = nid_type(node_id) == NODE_PAPER
        except (IndexError, ValueError):
            is_paper = False
        if is_paper and node_id not in paper_ids:
            paper_ids.append(node_id)
    return tuple(paper_ids)


def entry_cutoff_rejection_reason(
    entry: Mapping[str, Any],
    paper_cutoff: PaperDateCutoff,
    *,
    explored_flags: Mapping[str, bool] | None = None,
) -> str | None:
    """Return the first explored-status or publication cutoff failure."""
    if not entry_is_explored(entry, paper_cutoff.db, explored_flags=explored_flags):
        return "paper_not_explored"
    paper_ids = initial_tree_paper_ids(entry)
    for paper_id in paper_ids:
        reason = paper_cutoff.rejection_reason(paper_id)
        if reason is not None:
            return reason
    return None


def load_permanent_path_failure_reasons(path: Path | None) -> dict[str, str]:
    """Return permanent path failures saved by an earlier v5 run."""
    if path is None:
        return {}
    payload = load_json_output(path)
    if not isinstance(payload, dict):
        return {}
    raw_reasons = payload.get(PERMANENT_PATH_FAILURE_REASONS_FIELD)
    if not isinstance(raw_reasons, dict):
        return {}
    return {
        str(path_id): str(reason)
        for path_id, reason in raw_reasons.items()
        if str(path_id) and str(reason)
    }


def write_run_state(
    path: Path | None,
    *,
    bucket_labels: Sequence[str],
    processed_path_ids: set[str],
    skipped_path_ids: set[str],
    skip_reason_counts: dict[str, int],
    permanent_path_failure_reasons: dict[str, str],
    indent: int,
) -> None:
    """Write resume state including permanent per-path failures."""
    if path is None:
        return
    write_json_output(
        path,
        {
            "bucket_labels": sorted(str(label) for label in bucket_labels),
            "total_processed_paths": len(processed_path_ids),
            "total_skipped_paths": len(skipped_path_ids),
            "processed_path_ids": sorted(processed_path_ids),
            "skipped_path_ids": sorted(skipped_path_ids),
            "skip_reason_counts": dict(sorted(skip_reason_counts.items())),
            PERMANENT_PATH_FAILURE_REASONS_FIELD: dict(
                sorted(permanent_path_failure_reasons.items())
            ),
        },
        indent,
    )


def build_simple_prepared_growth_path(
    entry: dict[str, Any],
    *,
    context: yesno_base.ReachabilityCheckerContext,
    facts_reader: PaperFactsReader,
    traversal_mode: str,
) -> PreparedGrowthPath | None:
    """Build one v5 thin tree from a plain sampled counting path."""
    prepared = prepare_question_path(
        entry.get("path") or entry.get("steps") or []
    )
    if prepared is None:
        return None
    nodes, edges = prepared
    node_ids = tuple(str(node.get("node_id") or "") for node in nodes)
    if not node_ids or any(not node_id for node_id in node_ids):
        return None
    paper_only = is_paper_only_traversal_mode(traversal_mode)
    if paper_only and any(nid_type(node_id) != NODE_PAPER for node_id in node_ids):
        return None
    paper_ids = list(dict.fromkeys(yesno_base.paper_node_ids_on_path(nodes)))
    if not paper_ids:
        return None
    references_by_paper = {paper_id: [] for paper_id in paper_ids}
    protected_reference_pairs: list[tuple[str, str]] = []
    for index in range(1, len(nodes)):
        edge = edges[index]
        if not isinstance(edge, dict):
            continue
        edge_type = str(edge.get("edge_type") or "")
        source_id = str(edge.get("from") or node_ids[index - 1])
        target_id = str(edge.get("to") or node_ids[index])
        if edge_type == EDGE_CITING:
            pair = (source_id, target_id)
        elif edge_type == EDGE_CITED_BY:
            pair = (target_id, source_id)
        else:
            continue
        references_by_paper.setdefault(pair[0], [])
        if pair[1] not in references_by_paper[pair[0]]:
            references_by_paper[pair[0]].append(pair[1])
        if pair not in protected_reference_pairs:
            protected_reference_pairs.append(pair)
    authors_by_paper = (
        {paper_id: [] for paper_id in paper_ids}
        if paper_only
        else paper_author_node_ids(paper_ids, context)
    )
    path_id = str(
        entry.get("path_id")
        or entry.get("path_index")
        or path_tree_cache_key(entry)
    )
    state = initialize_growth_tree(
        cache_key=f"{path_tree_cache_key(entry)}:simple:v5",
        path_id=path_id,
        orientation_index=0,
        source_node_id=node_ids[0],
        target_node_id=node_ids[-1],
        witness=node_ids,
        source_wrong_path=(node_ids[0],),
        target_wrong_path=(node_ids[-1],),
        gold_paper_ids=paper_ids,
        expanded_paper_ids=paper_ids,
        authors_by_paper=authors_by_paper,
        references_by_paper=references_by_paper,
        protected_reference_pairs=protected_reference_pairs,
        facts_reader=facts_reader,
        include_author_identity=not paper_only,
    )
    if state is None:
        return None
    return PreparedGrowthPath(entry=entry, state=state)


def citation_owner_paper_ids(
    paper_ids: Sequence[str],
    references_by_paper: Mapping[str, Sequence[str]],
) -> list[str]:
    """Return logical papers that own at least one rendered citation."""
    ordered_paper_ids = list(paper_ids)
    if any(not isinstance(paper_id, str) for paper_id in ordered_paper_ids):
        raise TypeError("Paper block IDs must be strings.")
    if len(set(ordered_paper_ids)) != len(ordered_paper_ids):
        raise ValueError("Paper block IDs must be unique.")

    normalized_references: dict[str, list[str]] = {}
    for paper_id in ordered_paper_ids:
        if paper_id not in references_by_paper:
            raise ValueError(f"Paper {paper_id!r} has no reference mapping.")
        raw_references = references_by_paper[paper_id]
        if isinstance(raw_references, (str, bytes)):
            raise TypeError(f"Paper {paper_id!r} has invalid references.")
        cited_ids = list(raw_references)
        if any(not isinstance(cited_id, str) for cited_id in cited_ids):
            raise TypeError(f"Paper {paper_id!r} has invalid references.")
        normalized_references[paper_id] = cited_ids

    paper_block_ids = [
        paper_id for paper_id in ordered_paper_ids if normalized_references[paper_id]
    ]
    mentioned_paper_ids = {
        cited_id
        for paper_id in paper_block_ids
        for cited_id in normalized_references[paper_id]
    }
    missing_paper_ids = [
        paper_id
        for paper_id in ordered_paper_ids
        if paper_id not in paper_block_ids and paper_id not in mentioned_paper_ids
    ]
    if missing_paper_ids:
        raise ValueError(
            "Logical papers without blocks must be cited by a rendered block: "
            + ", ".join(missing_paper_ids)
            + "."
        )
    return paper_block_ids


@dataclass(slots=True)
class PreparedPathStore:
    """Build and cache v5 thin-tree orientations lazily."""

    entries_by_path_id: dict[str, dict[str, Any]]
    context: yesno_base.ReachabilityCheckerContext
    facts_reader: PaperFactsReader
    traversal_mode: str
    max_author_answer_index: int | None
    cache: dict[str, list[PreparedGrowthPath]]
    cache_max_entries: int | None = None
    simple_paths: bool = False

    def get(self, path_id: str) -> list[PreparedGrowthPath]:
        """Return cached thin orientations with v5-specific cache keys."""
        if path_id in self.cache:
            prepared = self.cache.pop(path_id)
            self.cache[path_id] = prepared
            return prepared
        if self.simple_paths:
            simple_path = build_simple_prepared_growth_path(
                self.entries_by_path_id[path_id],
                context=self.context,
                facts_reader=self.facts_reader,
                traversal_mode=self.traversal_mode,
            )
            v4_prepared = [] if simple_path is None else [simple_path]
        else:
            v4_prepared = v4.build_prepared_growth_paths(
                self.entries_by_path_id[path_id],
                context=self.context,
                facts_reader=self.facts_reader,
                traversal_mode=self.traversal_mode,
                max_author_answer_index=self.max_author_answer_index,
            )
        prepared: list[PreparedGrowthPath] = []
        for item in v4_prepared:
            state = item.state.clone()
            if state.cache_key.endswith(":v4"):
                state.cache_key = f"{state.cache_key[:-3]}v5"
            elif state.cache_key.endswith(":v5"):
                pass
            else:
                state.cache_key = f"{state.cache_key}:v5"
            prepared.append(PreparedGrowthPath(entry=item.entry, state=state))
        self.cache[path_id] = prepared
        if (
            self.cache_max_entries is not None
            and len(self.cache) > self.cache_max_entries
        ):
            self.cache.pop(next(iter(self.cache)))
        return prepared


def cache_configuration(
    *,
    graph_path: Path,
    db_path: Path,
    traversal_mode: str,
    question_hops: int,
    max_author_answer_index: int | None,
    num_children_per_node: int,
    only_explored: int = 0,
) -> dict[str, Any]:
    """Return settings that make v5 pair results reusable."""
    validate_num_children_per_node(num_children_per_node)
    configuration = v4.cache_configuration(
        only_explored=only_explored,
        graph_path=graph_path,
        db_path=db_path,
        traversal_mode=traversal_mode,
        question_hops=question_hops,
        max_author_answer_index=max_author_answer_index,
    )
    configuration.update(
        {
            "version": V5_PAIR_CACHE_VERSION,
            "constructor_version": "v5",
            "num_children_per_node": num_children_per_node,
            "tree_construction_strategy": V5_TREE_CONSTRUCTION_STRATEGY,
            "rendered_graph_policy_version": (V5_RENDERED_GRAPH_POLICY_VERSION),
            "paper_identity_policy": V5_PAPER_IDENTITY_POLICY,
            "candidate_commit_policy": V5_CANDIDATE_COMMIT_POLICY,
        }
    )
    return configuration


def load_pair_disjointness_cache(
    path: Path | None,
    configuration: dict[str, Any],
    *,
    max_entries: int | None = None,
) -> PairDisjointnessCache:
    """Load a v5-configured pair cache."""
    return v4.load_pair_disjointness_cache(
        path,
        configuration,
        max_entries=max_entries,
        ignored_configuration_fields=("graph", "db"),
    )


def plan_new_growth_group(
    initial_trees: Sequence[GrowthTreeState],
    target_kinds: Sequence[str],
    *,
    sampled_target: float,
    current_bucket: tuple[str, float, float],
    fallback_bucket_specs: Sequence[tuple[str, float, float]],
    facts_reader: PaperFactsReader,
    include_author_identity: bool,
    num_children_per_node: int,
    random_seed: int | None,
    key: Any,
    progress_callback: Callable[[], None] | None = None,
    paper_candidate_is_allowed: Callable[[str], bool] | None = None,
    paper_candidate_filter: PaperCandidateFilter | None = None,
    candidate_pool_cache: CandidatePoolCache | None = None,
) -> GrowthPlan | None:
    """Try a v5 target, then saturate and prune on failure."""
    validate_num_children_per_node(num_children_per_node)
    targets = [
        GrowthTarget(target_kind, sampled_target) for target_kind in target_kinds
    ]
    direct = grow_question_group(
        initial_trees,
        targets,
        facts_reader=facts_reader,
        include_author_identity=include_author_identity,
        num_children_per_node=num_children_per_node,
        random_seed=random_seed,
        group_key=(key, "v5-growth"),
        progress_callback=progress_callback,
        paper_candidate_is_allowed=paper_candidate_is_allowed,
        paper_candidate_filter=paper_candidate_filter,
        candidate_pool_cache=candidate_pool_cache,
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
            removed_paper_counts=tuple(0 for tree in direct.trees),
            removed_reference_pairs=tuple(() for tree in direct.trees),
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
        or v4.exhausted_growth_can_be_reused(direct)
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
            num_children_per_node=num_children_per_node,
            random_seed=random_seed,
            group_key=(key, "v5-growth"),
            progress_callback=progress_callback,
            paper_candidate_is_allowed=paper_candidate_is_allowed,
            paper_candidate_filter=paper_candidate_filter,
            candidate_pool_cache=candidate_pool_cache,
        )
    if saturated is None:
        return None
    fallback_plan = v4.final_plan_from_ladders(
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
    if fallback_plan is None:
        return None
    return fallback_plan


def legal_ascending_targets(
    session: LevelGrowthSession,
    bucket_spec: tuple[str, float, float],
) -> list[float]:
    """Return bucket targets that do not shrink a continuing session."""
    values = legal_target_values(
        bucket_spec[1],
        bucket_spec[2],
        target_kind=session.target_kinds[0],
    )
    current_values = [
        tree.target_value(target_kind)
        for tree, target_kind in zip(session.trees, session.target_kinds)
    ]
    return [
        value
        for value in values
        if all(value >= current - 1e-9 for current in current_values)
    ]


def plan_from_session_result(
    result: GrowthGroupResult,
    *,
    sampled_target: float,
    bucket_spec: tuple[str, float, float],
    target_kinds: Sequence[str],
    facts_reader: PaperFactsReader,
    include_author_identity: bool,
) -> GrowthPlan | None:
    """Convert one session snapshot into a valid plan for its bucket."""
    if result.all_targets_reached:
        if not all(
            validate_rendered_growth_tree(
                tree,
                facts_reader=facts_reader,
                include_author_identity=include_author_identity,
            )
            for tree in result.trees
        ):
            return None
        return GrowthPlan(
            trees=result.trees,
            target_kinds=tuple(target_kinds),
            target_value=sampled_target,
            bucket_label=bucket_spec[0],
            calibration=result.calibration,
            used_fallback=False,
            reused_higher_bucket_tree=False,
            removed_paper_counts=tuple(0 for tree in result.trees),
            removed_reference_pairs=tuple(() for tree in result.trees),
            maximum_depth_processed=result.maximum_depth_processed,
            rejection_counts=result.rejection_counts,
        )
    return v4.final_plan_from_ladders(
        result.trees,
        target_kinds,
        bucket_specs=[bucket_spec],
        facts_reader=facts_reader,
        random_seed=None,
        key=("v5-session-snapshot", bucket_spec[0]),
        used_fallback=True,
        reused_higher_bucket_tree=False,
        maximum_depth_processed=result.maximum_depth_processed,
        rejection_counts=result.rejection_counts,
    )


def advance_session_for_bucket(
    session: LevelGrowthSession,
    bucket_spec: tuple[str, float, float],
    *,
    facts_reader: PaperFactsReader,
    include_author_identity: bool,
    random_seed: int | None,
    key: Any,
    progress_callback: Callable[[], None] | None = None,
) -> tuple[GrowthPlan | None, GrowthGroupResult | None, float | None]:
    """Advance one persistent v5 session into an ascending bucket."""
    legal_values = legal_ascending_targets(session, bucket_spec)
    if not legal_values:
        return None, None, None
    sampled_target = select_uniform_target(
        legal_values,
        random_seed=random_seed,
        key=("v5-ascending", key),
    )
    result = advance_level_growth_session(
        session,
        sampled_target,
        progress_callback=progress_callback,
    )
    if result is None:
        return None, None, sampled_target
    plan = plan_from_session_result(
        result,
        sampled_target=sampled_target,
        bucket_spec=bucket_spec,
        target_kinds=session.target_kinds,
        facts_reader=facts_reader,
        include_author_identity=include_author_identity,
    )
    return plan, result, sampled_target


def plan_reused_growth_group(
    reusable: ReusableGrowthGroup,
    *,
    bucket_spec: tuple[str, float, float],
    facts_reader: PaperFactsReader,
    random_seed: int | None,
    key: Any,
) -> GrowthPlan | None:
    """Prune a reusable v5 group to one exact value in the requested bucket."""
    return v4.plan_reused_growth_group(
        reusable,
        bucket_spec=bucket_spec,
        facts_reader=facts_reader,
        random_seed=random_seed,
        key=key,
    )


def optional_child_counts(
    trees: Sequence[GrowthTreeState],
) -> tuple[list[str], list[int]]:
    """Return final optional-child counts for expanded papers."""
    paper_ids: list[str] = []
    counts: list[int] = []
    for tree in trees:
        count_by_parent: dict[str, int] = {}
        for parent_id in tree.parent_by_paper.values():
            count_by_parent[parent_id] = count_by_parent.get(parent_id, 0) + 1
        for paper_id in tree.expanded_paper_order:
            if paper_id not in tree.expanded_paper_ids:
                continue
            paper_ids.append(paper_id)
            counts.append(count_by_parent.get(paper_id, 0))
    return paper_ids, counts


def population_statistics(values: Sequence[int]) -> tuple[float, float]:
    """Return population mean and standard deviation."""
    if not values:
        return 0.0, 0.0
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def add_growth_metadata(
    record: dict[str, Any],
    plan: GrowthPlan,
    *,
    num_children_per_node: int,
    reuse_growth_session_across_buckets: bool = False,
) -> None:
    """Attach v5 growth and final-block statistics metadata in place."""
    validate_num_children_per_node(num_children_per_node)
    reference_statistics = record.get(NEEDLE_REFERENCE_STATISTICS_FIELD)
    if isinstance(reference_statistics, dict):
        reference_statistics.update(
            {
                "scope": "final_rendered_paper_blocks",
                "calculated_after_pruning": True,
                "paper_block_policy": V5_PAPER_BLOCK_POLICY,
                "empty_reference_blocks_included": False,
            }
        )

    paper_ids, child_counts = optional_child_counts(plan.trees)
    child_mean, child_std = population_statistics(child_counts)
    record["children_per_node_statistics"] = {
        "version": 1,
        "scope": "final_expanded_papers",
        "calculated_after_pruning": True,
        "paper_count": len(paper_ids),
        "paper_ids": paper_ids,
        "child_counts": child_counts,
        "mean": child_mean,
        "std": child_std,
        "std_type": "population",
        "num_children_per_node_budget": num_children_per_node,
        "counts_optional_growth_children_only": True,
        "includes_expanded_leaf_papers": True,
        "includes_unexpanded_terminal_papers": False,
    }

    reachability = record.setdefault("reachability", {})
    visible_graph = reachability.get("visible_graph")
    if isinstance(visible_graph, dict):
        visible_graph["allow_hidden_citations"] = True
    reachability[V5_GROWTH_FIELD] = {
        "constructor_version": "v5",
        "strategy": V5_TREE_CONSTRUCTION_STRATEGY,
        "edge_scope": "rendered_only",
        "rendered_graph_policy_version": V5_RENDERED_GRAPH_POLICY_VERSION,
        "paper_identity_policy": V5_PAPER_IDENTITY_POLICY,
        "candidate_commit_policy": V5_CANDIDATE_COMMIT_POLICY,
        "num_children_per_node": num_children_per_node,
        "child_budget_counts_directions": ["cites", "cited_by"],
        "anchor_side_saturation_enabled": False,
        "reference_caps_enabled": False,
        "frontier_stopping_policy": (
            "per-tree-smaller-side-unreachable-at-depth-boundary-v1"
        ),
        "equal_counter_frontier_policy": "continue-nonempty-frontier-v1",
        "target_value": plan.target_value,
        "target_kinds": list(plan.target_kinds),
        "allocation_bucket": plan.bucket_label,
        "used_maximal_fallback": plan.used_fallback,
        "reused_higher_bucket_tree": plan.reused_higher_bucket_tree,
        "maximum_depth_processed": plan.maximum_depth_processed,
        "pruned_paper_counts": list(plan.removed_paper_counts),
        "rejection_counts": dict(plan.rejection_counts),
        "reference_statistics_scope": "final_rendered_paper_blocks",
        "growth_history_stored": False,
        "bucket_order": (
            "ascending-continuous-session-v1"
            if reuse_growth_session_across_buckets
            else "independent-session-per-bucket-v1"
        ),
        "reuse_growth_session_across_buckets": (reuse_growth_session_across_buckets),
        "target_block_policy": (
            "reconsider-at-larger-target-v1"
            if reuse_growth_session_across_buckets
            else "bucket-local-v1"
        ),
    }


def serialize_negative_distractor_tree(
    tree: GrowthTreeState,
    *,
    target_component_size: int,
    num_children_per_node: int,
) -> dict[str, Any]:
    """Serialize one complete v5 rendered-edge negative component."""
    validate_num_children_per_node(num_children_per_node)
    expanded = [
        paper_id
        for paper_id in tree.expanded_paper_order
        if paper_id in tree.expanded_paper_ids
    ]
    references_by_paper = {
        paper_id: sorted(tree.references_by_paper.get(paper_id, set()))
        for paper_id in expanded
    }
    paper_block_ids = citation_owner_paper_ids(
        expanded,
        references_by_paper,
    )
    rendered_references_by_paper = {
        paper_id: references_by_paper[paper_id] for paper_id in paper_block_ids
    }
    _paper_ids, child_counts = optional_child_counts([tree])
    payload: dict[str, Any] = {
        "version": V5_NEGATIVE_DISTRACTOR_TREE_VERSION,
        "constructor_version": "v5",
        "edge_scope": "rendered_only",
        "tree_construction_strategy": V5_TREE_CONSTRUCTION_STRATEGY,
        "paper_identity_policy": V5_PAPER_IDENTITY_POLICY,
        "num_children_per_node": num_children_per_node,
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
        "paper_block_policy": V5_PAPER_BLOCK_POLICY,
        "paper_block_ids": paper_block_ids,
        "paper_block_count": len(paper_block_ids),
        "rendered_references_by_paper": rendered_references_by_paper,
        "authors_by_paper": {
            paper_id: sorted(tree.authors_by_paper.get(paper_id, set()))
            for paper_id in expanded
        },
        "references_by_paper": references_by_paper,
        "protected_reference_pairs": [
            list(pair) for pair in sorted(tree.protected_reference_pairs)
        ],
        "paper_depth": dict(sorted(tree.paper_depth.items())),
        "accepted_child_counts": dict(zip(expanded, child_counts)),
        "component_size": tree.component_size,
        "target_component_size": target_component_size,
    }
    payload["tree_id"] = negative_distractor_tree_signature(payload)
    return payload


def negative_distractor_paper_blocks(
    payload: Mapping[str, Any],
) -> tuple[list[str], dict[str, list[str]]] | None:
    """Return validated paper blocks and references from a v5 payload."""
    version = payload.get("version")
    expanded = payload.get("expanded_paper_ids")
    references = payload.get("references_by_paper")
    if (
        version not in V5_NEGATIVE_DISTRACTOR_TREE_SUPPORTED_VERSIONS
        or not isinstance(expanded, list)
        or not expanded
        or not all(isinstance(value, str) for value in expanded)
        or len(set(expanded)) != len(expanded)
        or not isinstance(references, Mapping)
        or set(references) != set(expanded)
        or not all(isinstance(value, list) for value in references.values())
        or not all(
            isinstance(cited_id, str)
            for cited_ids in references.values()
            for cited_id in cited_ids
        )
    ):
        return None
    normalized_references = {
        str(paper_id): list(cited_ids) for paper_id, cited_ids in references.items()
    }
    try:
        expected_paper_block_ids = citation_owner_paper_ids(
            expanded,
            normalized_references,
        )
    except (TypeError, ValueError):
        return None
    expected_rendered_references = {
        paper_id: normalized_references[paper_id]
        for paper_id in expected_paper_block_ids
    }
    if version == 1:
        return expected_paper_block_ids, expected_rendered_references
    if (
        payload.get("paper_block_policy") != V5_PAPER_BLOCK_POLICY
        or payload.get("paper_block_ids") != expected_paper_block_ids
        or payload.get("rendered_references_by_paper") != expected_rendered_references
        or payload.get("paper_block_count") != len(expected_paper_block_ids)
    ):
        return None
    return expected_paper_block_ids, expected_rendered_references


def deserialize_negative_distractor_tree(
    payload: Mapping[str, Any],
    *,
    facts_reader: PaperFactsReader,
    include_author_identity: bool,
) -> GrowthTreeState | None:
    """Rebuild one signed v5 distractor using rendered-edge semantics."""
    child_budget = payload.get("num_children_per_node")
    if (
        payload.get("version") not in V5_NEGATIVE_DISTRACTOR_TREE_SUPPORTED_VERSIONS
        or payload.get("constructor_version") != "v5"
        or payload.get("edge_scope") != "rendered_only"
        or payload.get("tree_construction_strategy") != V5_TREE_CONSTRUCTION_STRATEGY
        or payload.get("paper_identity_policy") != V5_PAPER_IDENTITY_POLICY
        or not isinstance(child_budget, int)
        or isinstance(child_budget, bool)
        or child_budget <= 0
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


def static_question_growth_states(
    question: Mapping[str, Any],
    *,
    facts_reader: PaperFactsReader,
    include_author_identity: bool,
) -> tuple[GrowthTreeState, ...] | None:
    """Rebuild v5 main components using only their rendered edges."""
    states = v4.static_question_growth_states(
        question,
        facts_reader=facts_reader,
        include_author_identity=include_author_identity,
    )
    if states is None:
        return None
    for state in states:
        if state.cache_key.endswith(":v4"):
            state.cache_key = f"{state.cache_key[:-3]}v5"
    return states


__all__ = [
    "PERMANENT_PATH_FAILURE_REASONS_FIELD",
    "TARGET_COMPONENT",
    "TARGET_YES",
    "V5_CANDIDATE_COMMIT_POLICY",
    "V5_GROWTH_FIELD",
    "V5_PAPER_BLOCK_POLICY",
    "V5_PAPER_IDENTITY_POLICY",
    "V5_RENDERED_GRAPH_POLICY_VERSION",
    "V5_TREE_CONSTRUCTION_STRATEGY",
    "CandidatePoolCache",
    "GrowthPlan",
    "GrowthTarget",
    "GrowthTreeState",
    "LevelGrowthSession",
    "PairDisjointnessCache",
    "PaperCandidateFilter",
    "PaperDateCutoff",
    "PaperFacts",
    "PaperFactsReader",
    "PreparedGrowthPath",
    "PreparedPathStore",
    "QuestionGrowthGroup",
    "ReferenceCalibration",
    "ReusableGrowthGroup",
    "add_growth_metadata",
    "advance_session_for_bucket",
    "build_simple_prepared_growth_path",
    "cache_configuration",
    "citation_owner_paper_ids",
    "combine_growth_variants",
    "create_level_growth_session",
    "deserialize_negative_distractor_tree",
    "entry_cutoff_rejection_reason",
    "grow_question_group",
    "growth_state_variant",
    "initial_tree_paper_ids",
    "legal_ascending_targets",
    "legal_target_values",
    "load_pair_disjointness_cache",
    "load_permanent_path_failure_reasons",
    "negative_component_target",
    "negative_distractor_paper_blocks",
    "negative_distractor_tree_signature",
    "orientation_order",
    "pairwise_thin_disjointness_reason",
    "paper_facts_reader",
    "parse_cutoff_date",
    "parse_published_date",
    "plan_from_session_result",
    "plan_new_growth_group",
    "plan_reused_growth_group",
    "plan_signature",
    "record_accepted_question",
    "record_final_invariant_failure",
    "resolve_cutoff_date",
    "select_uniform_target",
    "serialize_negative_distractor_tree",
    "stable_order",
    "static_question_growth_states",
    "validate_rendered_growth_tree",
    "write_run_state",
]
