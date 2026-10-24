"""Shared construction helpers for v11 reachability questions."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.prompts.reachability_yesno_qa import is_paper_only_traversal_mode
from src.questions import reachability_construction as yesno_base
from src.questions import reachability_rendered_pipeline as v4
from src.questions import reachability_rendered_pipeline_v5 as v5
from src.questions.explored import entry_is_explored
from src.questions.reachability_crossed_trees import path_tree_cache_key
from src.questions.reachability_long_path_profiles import (
    LONG_PATH_PROFILE_FIELD,
)
from src.questions.reachability_rendered_growth_v11 import (
    GROWTH_REACHED,
    GROWTH_SATURATED,
    GROWTH_TARGET_BLOCKED,
    TARGET_COMPONENT,
    TARGET_MATCHING,
    TARGET_YES,
    UNLIMITED_CHILDREN_PER_NODE,
    CandidateEdge,
    GrowthAdvanceResult,
    GrowthTarget,
    GrowthTreeState,
    PaperFacts,
    PaperFactsReader,
    PreferentialGrowthSession,
    QuestionGrowthGroup,
    ReferenceCalibration,
    advance_preferential_growth_session,
    create_preferential_growth_session,
    equal_current_target_value,
    initialize_growth_tree,
    validate_discount,
    validate_rendered_growth_tree,
    validate_theta,
)
from src.questions.reachability_rendered_growth_v11 import (
    validate_num_children_per_node as validate_growth_num_children_per_node,
)
from src.questions.shared import prepare_question_path
from src.setup.constants import (
    EDGE_CITED_BY,
    EDGE_CITING,
    NODE_PAPER,
    nid_db_id,
    nid_type,
)
from src.utils.resume import load_json_output, write_json_output

V11_GROWTH_FIELD = "level_growth"
V11_PAIR_CACHE_VERSION = 1
V11_TREE_CONSTRUCTION_STRATEGY = "rendered-edge-uniform-root-subtree-degree-growth-v1"
V11_RENDERED_GRAPH_POLICY_VERSION = 1
V11_PAPER_IDENTITY_POLICY = "normalized-title-single-node-v1"
V11_CANDIDATE_COMMIT_POLICY = "atomic-child-with-delayed-terminal-promotion-v1"
V11_NEGATIVE_DISTRACTOR_TREE_VERSION = 2
V11_NEGATIVE_DISTRACTOR_TREE_SUPPORTED_VERSIONS = frozenset({2})
V11_PAPER_BLOCK_POLICY = "nonempty-selected-references-v1"
V11_TERMINAL_CITES_POLICY = "reference-only-unless-later-expanded-v1"
V11_TERMINAL_CITED_BY_POLICY = "render-full-owner-block-v1"
V11_FRONTIER_SELECTION_POLICY = "uniform-root-subtree-degree-weighted-v1"
V11_FRONTIER_SELECTION_PROBABILITY = (
    "1 / eligible_root_count * "
    "weight(subtree_degree - discount if subtree_degree > 1 "
    "else theta + discount * K) / sum_eligible_subtree(weight); "
    "K = eligible_subtree_nodes_with_degree_greater_than_one"
)
V11_ATTACHMENT_DEGREE = (
    "optional-paper-children-plus-parent-excluding-initial-path-edges"
)
V11_DIRECTION_SELECTION_POLICY = "fair-coin-among-available-directions-v1"
PERMANENT_PATH_FAILURE_REASONS_FIELD = "permanent_path_failure_reasons"

PreparedGrowthPath = v4.PreparedGrowthPath
GrowthPlan = v4.GrowthPlan
PairDisjointnessCache = v4.PairDisjointnessCache

record_final_invariant_failure = v4.record_final_invariant_failure
stable_order = v4.stable_order
orientation_order = v4.orientation_order
paper_facts_reader = v4.paper_facts_reader
negative_component_target = v4.negative_component_target
negative_distractor_tree_signature = v4.negative_distractor_tree_signature
paper_author_node_ids = v4.paper_author_node_ids
legal_target_values = v4.legal_target_values
select_uniform_target = v4.select_uniform_target
record_accepted_question = v4.record_accepted_question
pairwise_thin_disjointness_reason = v4.pairwise_thin_disjointness_reason
growth_state_variant = v4.growth_state_variant
combine_growth_variants = v4.combine_growth_variants
plan_signature = v4.plan_signature


def citation_owner_paper_ids(
    paper_ids: Sequence[str],
    references_by_paper: Mapping[str, Iterable[str]],
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


def parse_cutoff_date(raw: str) -> date:
    """Parse one strict ``YYYY-MM-DD`` cutoff date."""
    text = str(raw).strip()
    if len(text) != 10:
        raise ValueError("cutoff date must use YYYY-MM-DD format")
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d").date()  # noqa: DTZ007
    except ValueError as error:
        raise ValueError(f"invalid cutoff date: {raw!r}") from error
    if parsed.isoformat() != text:
        raise ValueError("cutoff date must use YYYY-MM-DD format")
    return parsed


def resolve_cutoff_date(
    cutoff_date: str | date | None,
) -> tuple[date | None, str | None]:
    """Return the cutoff date and its normalized JSON value."""
    if cutoff_date is None:
        return None, None
    if type(cutoff_date) is date:
        return cutoff_date, cutoff_date.isoformat()
    if isinstance(cutoff_date, str):
        parsed = parse_cutoff_date(cutoff_date)
        return parsed, parsed.isoformat()
    raise ValueError("cutoff_date must be a date, YYYY-MM-DD string, or None.")


def parse_published_date(raw: Any) -> date | None:
    """Parse a stored paper publication timestamp into a calendar date."""
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
        return datetime.strptime(  # noqa: DTZ007
            text[:10], "%Y-%m-%d"
        ).date()
    except ValueError:
        return None


@dataclass(slots=True)
class PaperDateCutoff:
    """Cache and enforce one inclusive minimum paper publication date."""

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
            published_date = parse_published_date(published_by_db_id.get(db_id, ""))
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
        """Return one cached canonical publication date."""
        normalized_id = str(paper_id)
        if normalized_id not in self.published_dates:
            self.prime_published_dates((normalized_id,))
        return self.published_dates[normalized_id]

    def rejection_reason(self, paper_id: str) -> str | None:
        """Return why one paper fails the cutoff, if it does."""
        if self.cutoff_date is None:
            return None
        published = self.published_date(paper_id)
        if published is None:
            return "missing_or_invalid_paper_publication_date"
        if published < self.cutoff_date:
            return "paper_before_cutoff_date"
        return None

    def allows(self, paper_id: str) -> bool:
        """Return whether one paper satisfies the configured cutoff."""
        return self.rejection_reason(paper_id) is None


def initial_tree_paper_ids(entry: Mapping[str, Any]) -> tuple[str, ...]:
    """Return papers selected for the thin path and both wrong branches."""
    raw_node_ids: list[Any] = []
    path = entry.get("path") or entry.get("steps") or []
    if isinstance(path, list):
        raw_node_ids.extend(
            step.get("node", {}).get("node_id")
            for step in path
            if isinstance(step, Mapping) and isinstance(step.get("node"), Mapping)
        )
    profile = entry.get(LONG_PATH_PROFILE_FIELD)
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


def load_permanent_path_failure_reasons(
    path: Path | None,
) -> dict[str, str]:
    """Return permanent per-path failures saved by a previous v11 run."""
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
    """Write v11 resume state, including permanent per-path failures."""
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


@dataclass(slots=True)
class PreparedPathStore:
    """Build and cache v11 thin-tree orientations lazily."""

    entries_by_path_id: dict[str, dict[str, Any]]
    context: yesno_base.ReachabilityCheckerContext
    facts_reader: PaperFactsReader
    traversal_mode: str
    max_author_answer_index: int | None
    cache: dict[str, list[PreparedGrowthPath]]
    cache_max_entries: int | None = None
    simple_paths: bool = False

    def get(self, path_id: str) -> list[PreparedGrowthPath]:
        """Return cached thin orientations with v11-specific cache keys."""
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
            prepared = [] if simple_path is None else [simple_path]
        else:
            base_prepared = v4.build_prepared_growth_paths(
                self.entries_by_path_id[path_id],
                context=self.context,
                facts_reader=self.facts_reader,
                traversal_mode=self.traversal_mode,
                max_author_answer_index=self.max_author_answer_index,
            )
            prepared = []
            for item in base_prepared:
                state = item.state.clone()
                for suffix in (":v4", ":v5"):
                    if state.cache_key.endswith(suffix):
                        state.cache_key = f"{state.cache_key[:-2]}v11"
                        break
                else:
                    state.cache_key = f"{state.cache_key}:v11"
                prepared.append(PreparedGrowthPath(entry=item.entry, state=state))
        self.cache[path_id] = prepared
        if (
            self.cache_max_entries is not None
            and len(self.cache) > self.cache_max_entries
        ):
            self.cache.pop(next(iter(self.cache)))
        return prepared


def build_simple_prepared_growth_path(
    entry: dict[str, Any],
    *,
    context: yesno_base.ReachabilityCheckerContext,
    facts_reader: PaperFactsReader,
    traversal_mode: str,
) -> PreparedGrowthPath | None:
    """Build one v11 tree from a plain sampled path without side branches."""
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
        cache_key=f"{path_tree_cache_key(entry)}:simple:v11",
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


def cache_configuration(
    *,
    graph_path: Path,
    db_path: Path,
    traversal_mode: str,
    question_hops: int,
    max_author_answer_index: int | None,
    num_children_per_node: int,
    theta: float,
    discount: float,
    only_explored: int = 0,
) -> dict[str, Any]:
    """Return settings that make v11 thin-pair results reusable."""
    validate_num_children_per_node(num_children_per_node)
    validate_theta(theta)
    validate_discount(discount)
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
            "version": V11_PAIR_CACHE_VERSION,
            "constructor_version": "v11",
            "num_children_per_node": num_children_per_node,
            "theta": float(theta),
            "discount": float(discount),
            "tree_construction_strategy": V11_TREE_CONSTRUCTION_STRATEGY,
            "rendered_graph_policy_version": V11_RENDERED_GRAPH_POLICY_VERSION,
            "paper_identity_policy": V11_PAPER_IDENTITY_POLICY,
            "candidate_commit_policy": V11_CANDIDATE_COMMIT_POLICY,
        }
    )
    return configuration


def load_pair_disjointness_cache(
    path: Path | None,
    configuration: dict[str, Any],
    *,
    max_entries: int | None = None,
) -> PairDisjointnessCache:
    """Load a v11-configured permanent thin-pair cache."""
    return v4.load_pair_disjointness_cache(
        path,
        configuration,
        max_entries=max_entries,
        ignored_configuration_fields=("graph", "db"),
    )


def legal_ascending_targets(
    session: PreferentialGrowthSession,
    bucket_spec: tuple[str, float, float],
) -> list[float]:
    """Return legal bucket targets that do not shrink any current tree."""
    target_kind = session.group.targets[0].kind
    values = legal_target_values(
        bucket_spec[1],
        bucket_spec[2],
        target_kind=target_kind,
    )
    current_values = [
        tree.target_value(target.kind)
        for tree, target in zip(session.group.trees, session.group.targets)
    ]
    return [
        value
        for value in values
        if all(value >= current - 1e-9 for current in current_values)
    ]


def plan_from_advance(
    result: GrowthAdvanceResult,
    *,
    bucket_spec: tuple[str, float, float],
    include_author_identity: bool,
    facts_reader: PaperFactsReader,
) -> GrowthPlan | None:
    """Convert a reached target or in-bucket true saturation to a plan."""
    if result.status == GROWTH_REACHED:
        final_value = result.target_value
        used_saturation = False
    elif result.status == GROWTH_SATURATED:
        common_value = equal_current_target_value(
            result.trees,
            result.target_kinds,
        )
        if common_value is None or not (
            bucket_spec[1] <= common_value < bucket_spec[2]
        ):
            return None
        final_value = common_value
        used_saturation = True
    else:
        return None
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
        target_kinds=result.target_kinds,
        target_value=final_value,
        bucket_label=bucket_spec[0],
        calibration=result.calibration,
        used_fallback=used_saturation,
        reused_higher_bucket_tree=False,
        removed_paper_counts=tuple(0 for tree in result.trees),
        removed_reference_pairs=tuple(() for tree in result.trees),
        maximum_depth_processed=result.maximum_depth_processed,
        rejection_counts=result.rejection_counts,
    )


def advance_session_for_bucket(
    session: PreferentialGrowthSession,
    bucket_spec: tuple[str, float, float],
    *,
    facts_reader: PaperFactsReader,
    include_author_identity: bool,
    random_seed: int | None,
    key: Any,
    progress_callback: Callable[[], None] | None = None,
) -> tuple[GrowthPlan | None, GrowthAdvanceResult | None, float | None]:
    """Sample and advance one bucket in an existing ascending session."""
    legal_values = legal_ascending_targets(session, bucket_spec)
    if not legal_values:
        return None, None, None
    sampled_target = select_uniform_target(
        legal_values,
        random_seed=random_seed,
        key=("v11-ascending", key),
    )
    result = advance_preferential_growth_session(
        session,
        sampled_target,
        progress_callback=progress_callback,
    )
    plan = plan_from_advance(
        result,
        bucket_spec=bucket_spec,
        include_author_identity=include_author_identity,
        facts_reader=facts_reader,
    )
    return plan, result, sampled_target


def optional_child_counts(
    trees: Sequence[GrowthTreeState],
) -> tuple[list[str], list[int]]:
    """Return final optional-child counts for expanded paper blocks."""
    return v5.optional_child_counts(trees)


def add_growth_metadata(
    record: dict[str, Any],
    plan: GrowthPlan,
    *,
    num_children_per_node: int,
    theta: float,
    discount: float,
    reuse_growth_session_across_buckets: bool,
) -> None:
    """Attach v11 preferential-frontier and rendering metadata."""
    validate_num_children_per_node(num_children_per_node)
    validate_theta(theta)
    validate_discount(discount)
    metadata_child_budget = (
        1
        if num_children_per_node == UNLIMITED_CHILDREN_PER_NODE
        else num_children_per_node
    )
    v5.add_growth_metadata(
        record,
        plan,
        num_children_per_node=metadata_child_budget,
    )
    reference_statistics = record.get("needle_reference_statistics")
    if isinstance(reference_statistics, dict):
        reference_statistics["calculated_after_pruning"] = False
        reference_statistics["calculated_from_final_snapshot"] = True
    child_statistics = record.get("children_per_node_statistics")
    if isinstance(child_statistics, dict):
        child_statistics["calculated_after_pruning"] = False
        child_statistics["calculated_from_final_snapshot"] = True
        child_statistics["num_children_per_node_budget"] = num_children_per_node
    growth = record.setdefault("reachability", {}).setdefault(
        V11_GROWTH_FIELD,
        {},
    )
    growth.update(
        {
            "constructor_version": "v11",
            "strategy": V11_TREE_CONSTRUCTION_STRATEGY,
            "rendered_graph_policy_version": (V11_RENDERED_GRAPH_POLICY_VERSION),
            "paper_identity_policy": V11_PAPER_IDENTITY_POLICY,
            "candidate_commit_policy": V11_CANDIDATE_COMMIT_POLICY,
            "num_children_per_node": num_children_per_node,
            "child_budget_policy": (
                "unlimited-v1"
                if num_children_per_node == UNLIMITED_CHILDREN_PER_NODE
                else "combined-cites-and-cited-by-hard-cap-v1"
            ),
            "growth_order": V11_FRONTIER_SELECTION_POLICY,
            "frontier_selection_probability": V11_FRONTIER_SELECTION_PROBABILITY,
            "theta": float(theta),
            "discount": float(discount),
            "attachment_degree": V11_ATTACHMENT_DEGREE,
            "candidate_pool": "separate-cites-and-cited-by-edge-pools-v1",
            "candidate_sampling": (
                "uniform-without-replacement-within-chosen-direction-v1"
            ),
            "direction_probability": V11_DIRECTION_SELECTION_POLICY,
            "candidate_cache": "process-local-full-adjacency-by-paper-id-v1",
            "bucket_order": (
                "ascending-continuous-session-v1"
                if reuse_growth_session_across_buckets
                else "independent-session-per-bucket-v1"
            ),
            "reuse_growth_session_across_buckets": (
                reuse_growth_session_across_buckets
            ),
            "target_block_policy": (
                "defer-until-larger-target-v1"
                if reuse_growth_session_across_buckets
                else "bucket-local-v1"
            ),
            "saturation_policy": "accept-current-value-only-in-current-bucket-v1",
            "terminal_cites_policy": V11_TERMINAL_CITES_POLICY,
            "terminal_cited_by_policy": V11_TERMINAL_CITED_BY_POLICY,
            "used_true_saturation": plan.used_fallback,
            "used_maximal_fallback": False,
            "reused_higher_bucket_tree": False,
            "pruned_paper_counts": [0 for tree in plan.trees],
            "maximum_discovery_depth": plan.maximum_depth_processed,
            "anchor_side_saturation_enabled": True,
            "frontier_stopping_policy": "continuous-smaller-side-saturation-v1",
            "equal_counter_frontier_policy": "either-side-can-improve-v1",
        }
    )


def negative_distractor_paper_blocks(
    payload: Mapping[str, Any],
) -> tuple[list[str], dict[str, list[str]]] | None:
    """Return validated paper blocks and references from a v11 payload."""
    version = payload.get("version")
    expanded = payload.get("expanded_paper_ids")
    references = payload.get("references_by_paper")
    if (
        version not in V11_NEGATIVE_DISTRACTOR_TREE_SUPPORTED_VERSIONS
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
    except ValueError:
        return None
    expected_rendered_references = {
        paper_id: normalized_references[paper_id]
        for paper_id in expected_paper_block_ids
    }
    if version == 1:
        return expected_paper_block_ids, expected_rendered_references

    paper_block_ids = payload.get("paper_block_ids")
    rendered_references = payload.get("rendered_references_by_paper")
    paper_block_count = payload.get("paper_block_count")
    if (
        payload.get("paper_block_policy") != V11_PAPER_BLOCK_POLICY
        or paper_block_ids != expected_paper_block_ids
        or rendered_references != expected_rendered_references
        or not isinstance(paper_block_count, int)
        or isinstance(paper_block_count, bool)
        or paper_block_count != len(expected_paper_block_ids)
    ):
        return None
    return expected_paper_block_ids, expected_rendered_references


def serialize_negative_distractor_tree(
    tree: GrowthTreeState,
    *,
    target_component_size: int,
    num_children_per_node: int,
    theta: float,
    discount: float,
) -> dict[str, Any]:
    """Serialize one complete v11 preferential negative component."""
    validate_num_children_per_node(num_children_per_node)
    validate_theta(theta)
    validate_discount(discount)
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
        "version": V11_NEGATIVE_DISTRACTOR_TREE_VERSION,
        "constructor_version": "v11",
        "edge_scope": "rendered_only",
        "tree_construction_strategy": V11_TREE_CONSTRUCTION_STRATEGY,
        "growth_order": V11_FRONTIER_SELECTION_POLICY,
        "frontier_selection_probability": V11_FRONTIER_SELECTION_PROBABILITY,
        "attachment_degree": V11_ATTACHMENT_DEGREE,
        "paper_identity_policy": V11_PAPER_IDENTITY_POLICY,
        "candidate_commit_policy": V11_CANDIDATE_COMMIT_POLICY,
        "terminal_cites_policy": V11_TERMINAL_CITES_POLICY,
        "terminal_cited_by_policy": V11_TERMINAL_CITED_BY_POLICY,
        "num_children_per_node": num_children_per_node,
        "theta": float(theta),
        "discount": float(discount),
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
        "paper_block_policy": V11_PAPER_BLOCK_POLICY,
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
        "parent_by_paper": dict(sorted(tree.parent_by_paper.items())),
        "discovery_pair_by_paper": {
            paper_id: list(pair)
            for paper_id, pair in sorted(tree.discovery_pair_by_paper.items())
        },
        "processed_center_ids": sorted(tree.processed_center_ids),
        "accepted_child_counts": dict(zip(expanded, child_counts)),
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
    """Rebuild one signed v11 distractor using rendered-edge semantics."""
    child_budget = payload.get("num_children_per_node")
    theta = payload.get("theta")
    discount = payload.get("discount")
    if (
        payload.get("version") not in V11_NEGATIVE_DISTRACTOR_TREE_SUPPORTED_VERSIONS
        or payload.get("constructor_version") != "v11"
        or payload.get("edge_scope") != "rendered_only"
        or payload.get("tree_construction_strategy") != V11_TREE_CONSTRUCTION_STRATEGY
        or payload.get("paper_identity_policy") != V11_PAPER_IDENTITY_POLICY
        or payload.get("candidate_commit_policy") != V11_CANDIDATE_COMMIT_POLICY
        or payload.get("terminal_cites_policy") != V11_TERMINAL_CITES_POLICY
        or payload.get("terminal_cited_by_policy") != V11_TERMINAL_CITED_BY_POLICY
        or not isinstance(child_budget, int)
        or isinstance(child_budget, bool)
        or child_budget == 0
        or child_budget < UNLIMITED_CHILDREN_PER_NODE
        or isinstance(theta, bool)
        or not isinstance(theta, (int, float))
        or not math.isfinite(float(theta))
        or float(theta) <= 0.0
        or isinstance(discount, bool)
        or not isinstance(discount, (int, float))
        or not math.isfinite(float(discount))
        or not 0.0 <= float(discount) < 1.0
        or payload.get("tree_id") != negative_distractor_tree_signature(payload)
    ):
        return None
    expanded = payload.get("expanded_paper_ids")
    authors = payload.get("authors_by_paper")
    references = payload.get("references_by_paper")
    protected = payload.get("protected_reference_pairs")
    raw_paper_depth = payload.get("paper_depth")
    raw_parents = payload.get("parent_by_paper")
    raw_discovery_pairs = payload.get("discovery_pair_by_paper")
    raw_processed_centers = payload.get("processed_center_ids")
    raw_child_counts = payload.get("accepted_child_counts")
    rendered_paper_blocks = negative_distractor_paper_blocks(payload)
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
        or not isinstance(raw_paper_depth, Mapping)
        or not isinstance(raw_parents, Mapping)
        or not isinstance(raw_discovery_pairs, Mapping)
        or not isinstance(raw_processed_centers, list)
        or not isinstance(raw_child_counts, Mapping)
        or rendered_paper_blocks is None
        or not all(
            isinstance(depth, int) and not isinstance(depth, bool) and depth >= 0
            for depth in raw_paper_depth.values()
        )
        or not all(isinstance(value, str) for value in raw_processed_centers)
        or not all(
            isinstance(count, int) and not isinstance(count, bool) and count >= 0
            for count in raw_child_counts.values()
        )
    ):
        return None
    try:
        paper_depth = {
            str(paper_id): int(depth) for paper_id, depth in raw_paper_depth.items()
        }
        parent_by_paper = {
            str(paper_id): str(parent_id) for paper_id, parent_id in raw_parents.items()
        }
        discovery_pair_by_paper = {
            str(paper_id): (str(pair[0]), str(pair[1]))
            for paper_id, pair in raw_discovery_pairs.items()
            if isinstance(pair, list) and len(pair) == 2
        }
        processed_center_ids = {str(paper_id) for paper_id in raw_processed_centers}
        accepted_child_counts = {
            str(paper_id): int(count) for paper_id, count in raw_child_counts.items()
        }
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
    visible_ids = set(state.visible_paper_ids)
    expanded_ids = set(state.expanded_paper_ids)
    non_gold_ids = visible_ids - set(state.gold_paper_ids)
    rendered_pairs = {
        (paper_id, cited_id)
        for paper_id, cited_ids in state.references_by_paper.items()
        for cited_id in cited_ids
    }
    if (
        set(paper_depth) != visible_ids
        or set(parent_by_paper) != non_gold_ids
        or set(discovery_pair_by_paper) != non_gold_ids
        or not processed_center_ids.issubset(expanded_ids)
        or set(accepted_child_counts) != expanded_ids
        or any(
            parent_id not in visible_ids
            or set(discovery_pair_by_paper[paper_id]) != {paper_id, parent_id}
            or discovery_pair_by_paper[paper_id] not in rendered_pairs
            or paper_depth[paper_id] != paper_depth[parent_id] + 1
            for paper_id, parent_id in parent_by_paper.items()
        )
    ):
        return None
    child_count_by_paper = {paper_id: 0 for paper_id in visible_ids}
    for parent_id in parent_by_paper.values():
        child_count_by_paper[parent_id] += 1
    if any(
        accepted_child_counts[paper_id] != child_count_by_paper[paper_id]
        for paper_id in expanded_ids
    ):
        return None
    state.paper_depth = paper_depth
    state.parent_by_paper = parent_by_paper
    state.child_count_by_paper = child_count_by_paper
    state.discovery_pair_by_paper = discovery_pair_by_paper
    state.optional_reference_pairs = set(discovery_pair_by_paper.values())
    state.processed_center_ids = processed_center_ids
    if not validate_rendered_growth_tree(
        state,
        facts_reader=facts_reader,
        include_author_identity=include_author_identity,
    ):
        return None
    return state


def static_question_growth_states(
    question: Mapping[str, Any],
    *,
    facts_reader: PaperFactsReader,
    include_author_identity: bool,
) -> tuple[GrowthTreeState, ...] | None:
    """Rebuild v11 main components using only their rendered edges."""
    states = v4.static_question_growth_states(
        question,
        facts_reader=facts_reader,
        include_author_identity=include_author_identity,
    )
    if states is None:
        return None
    for state in states:
        if state.cache_key.endswith(":v4"):
            state.cache_key = f"{state.cache_key[:-3]}:v11"
    return states


def session_is_permanently_saturated(result: GrowthAdvanceResult | None) -> bool:
    """Return whether no later bucket can make this session grow."""
    return result is not None and result.status == GROWTH_SATURATED


def validate_num_children_per_node(value: int) -> None:
    """Require ``-1`` for unlimited children or one positive v11 child cap."""
    validate_growth_num_children_per_node(value)


__all__ = [
    "GROWTH_REACHED",
    "GROWTH_SATURATED",
    "GROWTH_TARGET_BLOCKED",
    "TARGET_COMPONENT",
    "TARGET_MATCHING",
    "TARGET_YES",
    "UNLIMITED_CHILDREN_PER_NODE",
    "V11_ATTACHMENT_DEGREE",
    "V11_CANDIDATE_COMMIT_POLICY",
    "V11_DIRECTION_SELECTION_POLICY",
    "V11_FRONTIER_SELECTION_POLICY",
    "V11_FRONTIER_SELECTION_PROBABILITY",
    "V11_GROWTH_FIELD",
    "V11_NEGATIVE_DISTRACTOR_TREE_VERSION",
    "V11_PAPER_BLOCK_POLICY",
    "V11_PAPER_IDENTITY_POLICY",
    "V11_RENDERED_GRAPH_POLICY_VERSION",
    "V11_TERMINAL_CITED_BY_POLICY",
    "V11_TERMINAL_CITES_POLICY",
    "V11_TREE_CONSTRUCTION_STRATEGY",
    "CandidateEdge",
    "GrowthAdvanceResult",
    "GrowthPlan",
    "GrowthTarget",
    "GrowthTreeState",
    "PairDisjointnessCache",
    "PaperDateCutoff",
    "PaperFacts",
    "PaperFactsReader",
    "PreferentialGrowthSession",
    "PreparedGrowthPath",
    "PreparedPathStore",
    "QuestionGrowthGroup",
    "ReferenceCalibration",
    "add_growth_metadata",
    "advance_preferential_growth_session",
    "advance_session_for_bucket",
    "build_simple_prepared_growth_path",
    "cache_configuration",
    "citation_owner_paper_ids",
    "combine_growth_variants",
    "create_preferential_growth_session",
    "deserialize_negative_distractor_tree",
    "entry_cutoff_rejection_reason",
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
    "plan_from_advance",
    "plan_signature",
    "record_accepted_question",
    "record_final_invariant_failure",
    "resolve_cutoff_date",
    "select_uniform_target",
    "serialize_negative_distractor_tree",
    "session_is_permanently_saturated",
    "stable_order",
    "static_question_growth_states",
    "validate_discount",
    "validate_num_children_per_node",
    "validate_rendered_growth_tree",
    "validate_theta",
    "write_run_state",
]
