"""Construction helpers for scholarly-graph reachability multiple-choice questions.

Option-tree variants, correct-option and NONE-answer quotas, option choice,
and multiple-choice question and needle records.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from src.prompts.reachability_yesno_qa import (
    is_paper_only_traversal_mode,
)
from src.questions import construction_run
from src.questions import reachability_construction as reachability_base
from src.questions.reachability_crossed_trees import (
    CachedSearchTree,
    TreeBucketVariant,
    tree_variant_disjointness_reason,
)
from src.questions.reachability_multiple import (
    ANSWER_TYPE,
    CONSTRUCTION,
    DEFAULT_NONE_OPTION,
    NONE_OPTION_TEXT,
    QUESTION_TYPE,
    answer_variants,
    author_name_answer_variants,
    build_question_text,
    deduplicate_answer_texts,
    displayed_num_options,
    graph_invariant_holds,
    option_labels,
)
from src.questions.reachability_yesno import (
    build_visible_graph,
    build_visible_identity_registry,
    no_exhaustive_search_steps,
    rooted_expected_dfs_steps,
    serialize_visible_graph,
)
from src.questions.shared import (
    NEEDLE_REFERENCE_STATISTICS_FIELD,
    calculate_needle_reference_statistics,
)
from src.setup.constants import NODE_AUTHOR, nid_db_id, nid_type
from src.utils.resume import load_json_output

DEFAULT_NUM_OPTIONS = 4
DEFAULT_NEEDLE_TYPE = reachability_base.DEFAULT_NEEDLE_TYPE
SUPPORTED_NEEDLE_TYPES = reachability_base.SUPPORTED_NEEDLE_TYPES
DEFAULT_MAX_AUTHOR_ANSWER_INDEX = (
    reachability_base.DEFAULT_MAX_AUTHOR_ANSWER_INDEX
)


@dataclass(frozen=True, slots=True)
class CompactTreeVariant:
    """Small description of a variant that can be materialized when needed."""

    tree: CachedSearchTree
    selection: reachability_base.TreeBucketSelection

    @property
    def yes_complexity(self) -> float:
        """Return the selected YES complexity."""
        return self.selection.yes_complexity

    @property
    def component_size(self) -> int:
        """Return the selected connected-component size."""
        return self.selection.component_size

    @property
    def removed_reference_count(self) -> int:
        """Return the number of removed optional references."""
        return self.selection.removed_reference_count


TreeVariantDescriptor = TreeBucketVariant | CompactTreeVariant


def matching_complexity(raw_yes_complexity: float) -> int:
    """Return the integer complexity used to match all option trees."""
    value = float(raw_yes_complexity)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Primary YES complexity must be finite and non-negative.")
    return math.ceil(value)


def combine_tree_variants(
    variants: Sequence[TreeBucketVariant],
) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    """Combine pairwise-disjoint final variants without overwriting data."""
    paper_ids: list[str] = []
    authors_by_paper: dict[str, list[str]] = {}
    references_by_paper: dict[str, list[str]] = {}
    for variant in variants:
        final_paper_ids = (
            variant.paper_ids
            if variant.paper_ids is not None
            else variant.tree.paper_ids
        )
        final_authors_by_paper = (
            variant.authors_by_paper
            if variant.authors_by_paper is not None
            else variant.tree.authors_by_paper
        )
        paper_ids.extend(final_paper_ids)
        authors_by_paper.update(final_authors_by_paper)
        references_by_paper.update(variant.references_by_paper)
    return paper_ids, authors_by_paper, references_by_paper


def option_position_counts(
    records: dict[str, dict[str, Any]],
    labels: Sequence[str],
) -> dict[str, int]:
    """Count accepted correct positions among one bucket's records."""
    counts = {label: 0 for label in labels}
    for record in records.values():
        multiple_choice = record.get("multiple_choice")
        if not isinstance(multiple_choice, dict):
            continue
        label = str(multiple_choice.get("correct_option") or "").upper()
        if label in counts:
            counts[label] += 1
    return counts


def correct_option_quotas(
    target_questions: int | None,
    *,
    labels: Sequence[str],
    bucket_labels: Sequence[str],
    random_seed: int | None,
) -> dict[str, dict[str, int]] | None:
    """Return per-bucket quotas with globally rotating remainder slots."""
    if target_questions is None:
        return None
    label_count = len(labels)
    base_count, remainder = divmod(target_questions, label_count)
    start = construction_run.stable_path_seed(
        random_seed,
        ("multiple-choice-quota", tuple(labels), tuple(bucket_labels)),
    ) % label_count
    quotas: dict[str, dict[str, int]] = {}
    for bucket_index, bucket_label in enumerate(bucket_labels):
        bucket_quotas = {label: base_count for label in labels}
        remainder_start = start + bucket_index * remainder
        for offset in range(remainder):
            label = labels[(remainder_start + offset) % label_count]
            bucket_quotas[label] += 1
        quotas[bucket_label] = bucket_quotas
    return quotas


def option_quotas_are_balanced(counts: Mapping[str, int]) -> bool:
    """Return whether option quotas differ by at most one question."""
    values = list(counts.values())
    return not values or max(values) - min(values) <= 1


def grown_correct_option_quotas(
    target_questions: int,
    *,
    labels: Sequence[str],
    bucket_labels: Sequence[str],
    records_by_label: dict[str, dict[str, dict[str, Any]]],
    random_seed: int | None,
) -> dict[str, dict[str, int]]:
    """Extend existing option counts to balanced deterministic larger quotas."""
    quotas: dict[str, dict[str, int]] = {}
    for bucket_label in bucket_labels:
        counts = option_position_counts(records_by_label[bucket_label], labels)
        remaining = target_questions - sum(counts.values())
        if remaining < 0:
            raise ValueError(
                f"Existing bucket {bucket_label!r} exceeds the requested target."
            )
        stable_order = sorted(
            labels,
            key=lambda label: construction_run.stable_path_seed(
                random_seed,
                ("multiple-choice-growth-quota", bucket_label, label),
            ),
        )
        rank = {label: index for index, label in enumerate(stable_order)}
        bucket_quotas = dict(counts)
        for _index in range(remaining):
            minimum = min(bucket_quotas.values())
            option_label = min(
                (
                    label
                    for label in labels
                    if bucket_quotas[label] == minimum
                ),
                key=rank.__getitem__,
            )
            bucket_quotas[option_label] += 1
        if not option_quotas_are_balanced(bucket_quotas):
            details = ", ".join(
                f"{label}={bucket_quotas[label]}" for label in labels
            )
            raise ValueError(
                f"Existing bucket {bucket_label!r} cannot reach balanced "
                f"correct-option quotas at target {target_questions} ({details})."
            )
        quotas[bucket_label] = bucket_quotas
    return quotas


def saved_correct_option_quotas(
    bucket_labels: Sequence[str],
    *,
    multi_bucket: bool,
    needles_output: Path,
) -> dict[str, dict[str, int]] | None:
    """Return consistent saved correct-option quotas from existing outputs."""
    requested_labels = set(bucket_labels)
    saved_values: list[dict[str, dict[str, int]]] = []
    for bucket_label in bucket_labels:
        target = (
            construction_run.resolve_bucket_output_path(
                needles_output,
                bucket_label,
            )
            if multi_bucket
            else needles_output
        )
        payload = load_json_output(target)
        parameters = (payload or {}).get("parameters")
        raw_quotas = (
            parameters.get("correct_option_quotas")
            if isinstance(parameters, dict)
            else None
        )
        if not isinstance(raw_quotas, dict):
            continue
        saved_values.append(
            {
                str(label): {
                    str(option): int(count)
                    for option, count in counts.items()
                }
                for label, counts in raw_quotas.items()
                if label in requested_labels and isinstance(counts, dict)
            }
        )
    if not saved_values:
        return None
    merged: dict[str, dict[str, int]] = {}
    for saved_value in saved_values:
        for label, counts in saved_value.items():
            if label in merged and merged[label] != counts:
                raise ValueError(
                    "Existing bucket outputs have inconsistent option quotas."
                )
            merged[label] = counts
    return merged


def merge_saved_correct_option_quotas(
    requested_quotas: dict[str, dict[str, int]] | None,
    saved_quotas: dict[str, dict[str, int]],
    *,
    labels: Sequence[str],
    bucket_labels: Sequence[str],
) -> dict[str, dict[str, int]] | None:
    """Preserve saved quotas and add quotas for newly requested buckets."""
    if requested_quotas is None:
        if saved_quotas:
            raise ValueError("Existing correct-option quotas are incompatible.")
        return None
    requested_labels = set(bucket_labels)
    expected_options = set(labels)
    merged = {
        bucket_label: dict(counts)
        for bucket_label, counts in requested_quotas.items()
    }
    for bucket_label, counts in saved_quotas.items():
        if bucket_label not in requested_labels:
            continue
        if set(counts) != expected_options:
            raise ValueError("Existing correct-option quotas are incompatible.")
        if sum(counts.values()) != sum(requested_quotas[bucket_label].values()):
            raise ValueError("Existing correct-option quotas are incompatible.")
        if not option_quotas_are_balanced(counts):
            raise ValueError("Existing correct-option quotas are not balanced.")
        merged[bucket_label] = dict(counts)
    return merged


def choose_correct_option(
    records: dict[str, dict[str, Any]],
    *,
    labels: Sequence[str],
    target_counts: dict[str, int] | None = None,
    random_seed: int | None,
    path_key: Any,
) -> str:
    """Choose an open position with the largest remaining quota deficit."""
    counts = option_position_counts(records, labels)
    if target_counts is None:
        minimum = min(counts.values())
        candidates = [label for label in labels if counts[label] == minimum]
    else:
        deficits = {
            label: int(target_counts[label]) - counts[label] for label in labels
        }
        maximum = max(deficits.values())
        if maximum <= 0:
            raise ValueError("Every correct-option position quota is full.")
        candidates = [label for label in labels if deficits[label] == maximum]
    rng = random.Random(
        construction_run.stable_path_seed(
            random_seed,
            ("multiple-choice-position", path_key, len(records)),
        )
    )
    return rng.choice(candidates)


def record_none_answer_is_correct(record: Mapping[str, Any]) -> bool:
    """Return whether one stored multiple-choice record is None-correct."""
    multiple_choice = record.get("multiple_choice")
    if not isinstance(multiple_choice, Mapping):
        return False
    answer_kind = str(
        multiple_choice.get("correct_answer_kind") or ""
    ).strip()
    if answer_kind:
        return answer_kind == "none"
    correct_option = str(
        multiple_choice.get("correct_option") or ""
    ).strip().upper()
    return any(
        isinstance(option, Mapping)
        and option.get("option_kind") == "none"
        and str(option.get("label") or "").strip().upper() == correct_option
        for option in multiple_choice.get("options") or []
    )


def none_answer_is_correct_for_slot(
    slot_index: int,
    *,
    labels: Sequence[str],
    bucket_label: str,
    random_seed: int | None,
) -> bool:
    """Return the seeded cyclic None-correct assignment for one bucket slot."""
    if slot_index < 0:
        raise ValueError("slot_index must be non-negative.")
    if not labels:
        raise ValueError("At least one option label is required.")
    phase = construction_run.stable_path_seed(
        random_seed,
        ("multiple-choice-none-answer-schedule", bucket_label, tuple(labels)),
    ) % len(labels)
    return (slot_index + phase) % len(labels) == 0


def next_none_answer_is_correct(
    records: Mapping[str, Mapping[str, Any]],
    *,
    labels: Sequence[str],
    none_option: int,
    bucket_label: str,
    random_seed: int | None,
) -> bool:
    """Return whether the next accepted record should make None correct."""
    if not none_option:
        return False
    return none_answer_is_correct_for_slot(
        len(records),
        labels=labels,
        bucket_label=bucket_label,
        random_seed=random_seed,
    )


def validate_none_answer_schedule(
    records: Mapping[str, Mapping[str, Any]],
    *,
    labels: Sequence[str],
    none_option: int,
    bucket_label: str,
    random_seed: int | None,
) -> None:
    """Reject resumed records that do not follow the configured None schedule."""
    if not none_option:
        return
    for slot_index, record in enumerate(records.values()):
        expected = none_answer_is_correct_for_slot(
            slot_index,
            labels=labels,
            bucket_label=bucket_label,
            random_seed=random_seed,
        )
        if record_none_answer_is_correct(record) != expected:
            raise ValueError(
                f"Existing bucket {bucket_label!r} has an incompatible "
                f"None-answer assignment at slot {slot_index}."
            )


def none_option_position_counts(
    records: Mapping[str, Mapping[str, Any]],
    labels: Sequence[str],
) -> dict[str, int]:
    """Count how often the None choice occupies each displayed position."""
    counts = {label: 0 for label in labels}
    for record in records.values():
        multiple_choice = record.get("multiple_choice")
        if not isinstance(multiple_choice, Mapping):
            continue
        for option in multiple_choice.get("options") or []:
            if not isinstance(option, Mapping):
                continue
            if option.get("option_kind") != "none":
                continue
            label = str(option.get("label") or "").strip().upper()
            if label in counts:
                counts[label] += 1
            break
    return counts


def choose_none_option_label(
    records: Mapping[str, Mapping[str, Any]],
    *,
    labels: Sequence[str],
    correct_option: str,
    none_is_correct: bool,
    random_seed: int | None,
    path_key: Any,
) -> str:
    """Choose a balanced seeded position for the displayed None choice."""
    normalized_correct = str(correct_option or "").strip().upper()
    if normalized_correct not in labels:
        raise ValueError("The correct option is not a displayed label.")
    if none_is_correct:
        return normalized_correct
    counts = none_option_position_counts(records, labels)
    candidates = [label for label in labels if label != normalized_correct]
    minimum = min(counts[label] for label in candidates)
    candidates = [label for label in candidates if counts[label] == minimum]
    rng = random.Random(
        construction_run.stable_path_seed(
            random_seed,
            ("multiple-choice-none-position", path_key, len(records)),
        )
    )
    return rng.choice(candidates)


def variant_identity(variant: TreeVariantDescriptor) -> str:
    """Return the exact stable identity of one final bucket variant."""
    if isinstance(variant, CompactTreeVariant):
        return (
            f"{variant.tree.cache_key}:"
            f"{variant.selection.removed_reference_count}"
        )
    signature = str(variant.variant_signature or "").strip()
    if signature:
        return signature
    return f"{variant.tree.cache_key}:{variant.removed_reference_count}"


def exact_group_signature(
    primary: TreeVariantDescriptor,
    negatives: Sequence[TreeVariantDescriptor],
) -> tuple[str, tuple[str, ...]]:
    """Return an option-order-independent signature for a proposed group."""
    return (
        variant_identity(primary),
        tuple(sorted(variant_identity(variant) for variant in negatives)),
    )


def option_answer_aliases(
    node_id: str,
    option_text: str,
    db: Any,
) -> list[str]:
    """Return standalone paper text or canonical/stored author-name aliases."""
    aliases = [option_text]
    if nid_type(node_id) != NODE_AUTHOR:
        return aliases

    author_id = nid_db_id(node_id)
    names: list[str] = []
    get_author = getattr(db, "get_author", None)
    if callable(get_author):
        author = get_author(author_id)
        if author is not None:
            try:
                name = str(author.get("name") or "").strip()
            except AttributeError:
                name = str(author["name"] or "").strip()
            if name:
                names.append(name)
    if not names and option_text.startswith("Author "):
        names.append(option_text[len("Author ") :].strip())

    get_variants = getattr(db, "author_name_variants_for_author", None)
    if callable(get_variants):
        for row in get_variants(author_id):
            try:
                variant = str(row.get("variant_name") or "").strip()
            except AttributeError:
                variant = str(row["variant_name"] or "").strip()
            if variant:
                names.append(variant)

    for name in deduplicate_answer_texts(names):
        for variant in author_name_answer_variants(name):
            aliases.extend((variant, f"Author {variant}"))
    return deduplicate_answer_texts(aliases)


def build_multiple_choice_record(
    variants: Sequence[TreeBucketVariant],
    *,
    label: str,
    bucket_min: float,
    bucket_max: float,
    correct_option: str,
    num_target_options: int | None = None,
    db: Any,
    needle_type: str,
    traversal_mode: str,
    random_seed: int | None,
    none_option: int = DEFAULT_NONE_OPTION,
    none_is_correct: bool = False,
    none_option_label: str | None = None,
    paper_block_ids: Sequence[str] | None = None,
    qid_namespace: str = "",
    allowed_terminal_reference_owners: Mapping[
        str, Sequence[str]
    ] | None = None,
) -> dict[str, Any]:
    """Build one audited multiple-choice needle record."""
    if not isinstance(none_is_correct, bool):
        raise ValueError("none_is_correct must be a boolean.")
    if none_is_correct and not none_option:
        raise ValueError("None cannot be correct when none_option=0.")
    if none_option_label is not None and not none_option:
        raise ValueError("A None option label requires none_option=1.")
    normal_option_count = (
        len(variants) - int(none_is_correct)
        if num_target_options is None
        else num_target_options
    )
    displayed_option_count = displayed_num_options(
        normal_option_count,
        none_option,
    )
    expected_tree_count = normal_option_count + int(none_is_correct)
    if expected_tree_count != len(variants):
        raise ValueError("The tree count does not match the answer construction.")
    path_ids = [variant.tree.path_id for variant in variants]
    if len(set(path_ids)) != len(path_ids):
        raise ValueError("One path cannot provide two options in a question.")
    for first_index, first in enumerate(variants):
        for second in variants[first_index + 1 :]:
            reason = tree_variant_disjointness_reason(
                first,
                second,
            )
            if reason is not None:
                raise ValueError(
                    "Final option-tree variants are not disjoint: "
                    f"{reason}."
                )
    primary = variants[0]
    if none_is_correct:
        raw_primary_complexity = float(primary.component_size)
        exact_matching_complexity = int(primary.component_size)
    else:
        raw_primary_complexity = float(primary.yes_complexity)
        exact_matching_complexity = matching_complexity(raw_primary_complexity)
    if not bucket_min <= exact_matching_complexity < bucket_max:
        raise ValueError(
            "The quantized primary complexity falls outside its bucket."
        )
    alternatives = list(variants[1:])
    option_order_key: tuple[Any, ...] = (
        "multiple-choice-option-order",
        primary.tree.cache_key,
        label,
        correct_option,
    )
    if none_option:
        option_order_key = (
            *option_order_key,
            "none-option",
            none_is_correct,
            none_option_label,
        )
    option_rng = random.Random(
        construction_run.stable_path_seed(
            random_seed,
            option_order_key,
        )
    )
    option_rng.shuffle(alternatives)
    labels = option_labels(displayed_option_count)
    correct_index = labels.index(correct_option)
    normalized_none_label: str | None = None
    if none_option:
        normalized_none_label = str(none_option_label or "").strip().upper()
        if none_is_correct:
            if normalized_none_label and normalized_none_label != correct_option:
                raise ValueError(
                    "The correct None choice must use the correct option label."
                )
            normalized_none_label = correct_option
        elif not normalized_none_label:
            none_candidates = [label for label in labels if label != correct_option]
            normalized_none_label = option_rng.choice(none_candidates)
        if (
            normalized_none_label not in labels
            or (
                not none_is_correct
                and normalized_none_label == correct_option
            )
        ):
            raise ValueError("The None choice has an invalid option label.")

    if none_is_correct:
        ordered_variants = [primary, *alternatives]
        target_labels = [
            value for value in labels if value != normalized_none_label
        ]
        labeled_variants = list(zip(target_labels, alternatives))
    else:
        target_labels = [
            value for value in labels if value != normalized_none_label
        ]
        ordered_variants = list(alternatives)
        ordered_variants.insert(target_labels.index(correct_option), primary)
        labeled_variants = list(zip(target_labels, ordered_variants))

    paper_ids, authors_by_paper, references_by_paper = combine_tree_variants(
        ordered_variants
    )
    if paper_block_ids is None:
        rendered_paper_ids = paper_ids
    else:
        requested_paper_block_ids = list(paper_block_ids)
        if (
            len(set(requested_paper_block_ids))
            != len(requested_paper_block_ids)
            or not set(requested_paper_block_ids).issubset(paper_ids)
        ):
            raise ValueError("Explicit paper block IDs are invalid.")
        requested_paper_block_id_set = set(requested_paper_block_ids)
        rendered_paper_ids = [
            paper_id
            for paper_id in paper_ids
            if paper_id in requested_paper_block_id_set
        ]
    visible = build_visible_graph(
        expanded_paper_ids=paper_ids,
        authors_by_paper=authors_by_paper,
        references_by_paper=references_by_paper,
    )
    source_component_size = len(
        nx.node_connected_component(visible.graph, primary.tree.source_node_id)
    )
    if none_is_correct and source_component_size != exact_matching_complexity:
        raise ValueError(
            "The source component does not have the requested exhaustive "
            "search complexity."
        )
    if none_is_correct and any(
        int(variant.component_size) != exact_matching_complexity
        for variant in variants
    ):
        raise ValueError(
            "Every None-option tree must have the source component's exact "
            "exhaustive-search complexity."
        )
    for variant_index, variant in enumerate(variants):
        expected_role = (
            reachability_base.TREE_ROLE_POSITIVE
            if variant_index == 0
            else reachability_base.TREE_ROLE_TARGET_NEGATIVE
        )
        if variant.tree.gold_depth > 0 and (
            variant.tree.tree_role != expected_role
            or not reachability_base.wrong_branches_are_valid(
                visible=visible,
                tree_role=variant.tree.tree_role,
                source_node_id=variant.tree.source_node_id,
                target_node_id=variant.tree.target_node_id,
                witness=variant.tree.witness,
                gold_depth=variant.tree.gold_depth,
                wrong_branch_paths=variant.tree.wrong_branch_paths,
            )
        ):
            raise ValueError(
                "An option tree is missing its required exact-depth wrong "
                "branch."
            )
    paper_cache: dict[str, dict[str, Any] | None] = {}
    paper_titles = reachability_base.paper_titles_for_visible(
        paper_ids,
        references_by_paper,
        db,
        paper_cache,
    )
    author_names = reachability_base.author_names_for_visible(
        authors_by_paper,
        db,
    )
    include_author_identity = not is_paper_only_traversal_mode(
        traversal_mode
    )
    (
        identity_author_ids_by_paper,
        identity_author_names,
        allowed_reference_pairs,
        citation_neighbors_by_paper,
    ) = reachability_base.final_tree_identity_inputs(
        paper_ids=paper_ids,
        references_by_paper=references_by_paper,
        trees=[variant.tree for variant in ordered_variants],
        db=db,
        include_author_identity=include_author_identity,
    )
    identity_author_names.update(author_names)
    identity_registry = build_visible_identity_registry(
        visible,
        paper_titles_by_paper=paper_titles,
        author_names_by_author=identity_author_names,
        citation_owners_by_paper={},
        paper_author_ids_by_paper=identity_author_ids_by_paper,
        allowed_reference_pairs=allowed_reference_pairs,
        allowed_terminal_reference_owners=(
            allowed_terminal_reference_owners
        ),
        citation_neighbors_by_paper=citation_neighbors_by_paper,
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
            "Combined option-tree graph has ambiguous identities: "
            + ", ".join(conflict_fields)
            + "."
        )

    option_rows: list[dict[str, Any]] = []
    for option_label, variant in labeled_variants:
        is_correct = not none_is_correct and option_label == correct_option
        search_complexity = (
            float(exact_matching_complexity)
            if is_correct
            else no_exhaustive_search_steps(
                visible,
                primary.tree.source_node_id,
                variant.tree.target_node_id,
            )
        )
        if search_complexity is None:
            raise ValueError("An incorrect option is not a valid negative tree.")
        if float(search_complexity) != float(exact_matching_complexity):
            raise ValueError(
                "Every option must have the primary tree's exact matched "
                "complexity."
            )
        option_rows.append(
            {
                "label": option_label,
                "option_kind": "target",
                "node_id": variant.tree.target_node_id,
                "text": reachability_base.node_label(
                    variant.tree.target_node_id,
                    db,
                    paper_cache,
                ),
                "path_id": variant.tree.path_id,
                "tree_cache_key": variant.tree.cache_key,
                "variant_signature": variant_identity(variant),
                "search_kind": "yes" if is_correct else "no",
                "complexity": float(search_complexity),
                "tree_yes_complexity": variant.yes_complexity,
                "component_size": variant.component_size,
                "tree_role": variant.tree.tree_role,
                "gold_depth": variant.tree.gold_depth,
                "wrong_branch_paths": variant.tree.wrong_branch_paths,
                "removed_optional_reference_count": (
                    variant.removed_reference_count
                ),
                "removed_optional_reference_pairs": [
                    list(pair) for pair in variant.removed_reference_pairs
                ],
            }
        )
    if none_option:
        assert normalized_none_label is not None
        option_rows.insert(
            labels.index(normalized_none_label),
            {
                "label": normalized_none_label,
                "option_kind": "none",
                "text": NONE_OPTION_TEXT,
                "search_kind": "none",
                "complexity": float(exact_matching_complexity),
                "component_size": source_component_size,
            },
        )
    option_nodes = [
        (str(row["label"]), str(row["node_id"]))
        for row in option_rows
        if row.get("option_kind") != "none"
    ]
    if not graph_invariant_holds(
        visible,
        source_node_id=primary.tree.source_node_id,
        options=option_nodes,
        correct_option=correct_option,
        none_option_label=(normalized_none_label if none_is_correct else None),
    ):
        raise ValueError("Combined graph does not have exactly one correct option.")

    proposed_group_signature = exact_group_signature(primary, alternatives)
    group_basis_parts = [QUESTION_TYPE]
    if qid_namespace:
        group_basis_parts.append(qid_namespace)
    group_basis_parts.append(label)
    if none_option:
        group_basis_parts.extend(
            (
                "none_option=1",
                f"none_is_correct={int(none_is_correct)}",
                f"none_option_label={normalized_none_label}",
            )
        )
    group_basis_parts.extend(
        [
            correct_option,
            *(variant_identity(variant) for variant in ordered_variants),
        ]
    )
    group_basis = ":".join(group_basis_parts)
    qid = hashlib.sha256(group_basis.encode("utf-8")).hexdigest()
    needle_reference_statistics = (
        calculate_needle_reference_statistics(
            rendered_paper_ids,
            references_by_paper,
        )
    )
    needles = reachability_base.render_needles(
        paper_node_ids=rendered_paper_ids,
        authors_by_paper=authors_by_paper,
        references_by_paper=references_by_paper,
        db=db,
        needle_type=needle_type,
        random_seed=random_seed,
        path_key=qid,
        display_all_database_authors=not include_author_identity,
    )
    source_label = reachability_base.node_label(
        primary.tree.source_node_id,
        db,
        paper_cache,
    )
    question_text = build_question_text(
        source_label=source_label,
        options=[
            (str(row["label"]), str(row["text"])) for row in option_rows
        ],
        include_authorship=include_author_identity,
    )
    correct_row = option_rows[correct_index]
    if none_is_correct:
        source_work = float(source_component_size)
        target_work = None
        visible_raw_complexity = source_work
    else:
        source_work = rooted_expected_dfs_steps(
            visible.graph,
            primary.tree.source_node_id,
            primary.tree.target_node_id,
        )
        target_work = rooted_expected_dfs_steps(
            visible.graph,
            primary.tree.target_node_id,
            primary.tree.source_node_id,
        )
        visible_raw_complexity = (
            None
            if source_work is None or target_work is None
            else min(float(source_work), float(target_work))
        )
        if visible_raw_complexity is None or not math.isclose(
            visible_raw_complexity,
            raw_primary_complexity,
        ):
            raise ValueError(
                "The primary tree's raw YES complexity changed during "
                "materialization."
            )
    multiple_choice = {
        "num_options": len(option_rows),
        "num_target_options": normal_option_count,
        "none_option": none_option,
        "none_option_label": normalized_none_label,
        "correct_answer_kind": "none" if none_is_correct else "target",
        "correct_option": correct_option,
        "options": option_rows,
    }
    reachability = {
        "source_node_id": primary.tree.source_node_id,
        "target_node_id": (
            None if none_is_correct else primary.tree.target_node_id
        ),
        "option_target_node_ids": {
            str(row["label"]): str(row["node_id"])
            for row in option_rows
            if row.get("option_kind") != "none"
        },
        "correct_option": correct_option,
        "author_identity_checks": include_author_identity,
        "complexity": {
            "measure": (
                "source_component_exhaustive_search_complexity"
                if none_is_correct
                else "ceiling_matched_option_search_complexity"
            ),
            "value": exact_matching_complexity,
            "matching_complexity": exact_matching_complexity,
            "correct_raw_expected_dfs": (
                None if none_is_correct else raw_primary_complexity
            ),
            "source_component_size": source_component_size,
            "positive_quantization": (
                "none" if none_is_correct else "ceiling"
            ),
            "source_endpoint_work": source_work,
            "target_endpoint_work": target_work,
            "option_tree_values": {
                str(row["label"]): float(row["complexity"])
                for row in option_rows
            },
            "option_search_kinds": {
                str(row["label"]): str(row["search_kind"])
                for row in option_rows
            },
        },
        "construction": {
            "strategy": CONSTRUCTION,
            "path_ids": [variant.tree.path_id for variant in ordered_variants],
            "group_signature": [
                proposed_group_signature[0],
                *proposed_group_signature[1],
            ],
            "pairwise_disjointness_validated": True,
            "source_tree_target_node_id": primary.tree.target_node_id,
        },
        "visible_graph": serialize_visible_graph(
            visible,
            paper_titles_by_paper=paper_titles,
            author_names_by_author=identity_author_names,
            paper_author_ids_by_paper=identity_author_ids_by_paper,
            allowed_reference_pairs=allowed_reference_pairs,
            allowed_terminal_reference_owners=(
                allowed_terminal_reference_owners
            ),
            citation_neighbors_by_paper=citation_neighbors_by_paper,
        ),
    }
    return {
        "qid": qid,
        "context_group_id": qid,
        "path_ids": [variant.tree.path_id for variant in ordered_variants],
        "question_type": QUESTION_TYPE,
        "question": question_text,
        "needles": needles,
        "needle_paper_ids": rendered_paper_ids,
        NEEDLE_REFERENCE_STATISTICS_FIELD: (
            needle_reference_statistics
        ),
        "answer": answer_variants(
            correct_option,
            str(correct_row["text"]),
            (
                [NONE_OPTION_TEXT]
                if none_is_correct
                else option_answer_aliases(
                    str(correct_row["node_id"]),
                    str(correct_row["text"]),
                    db,
                )
            ),
        ),
        "answer_type": ANSWER_TYPE,
        "multiple_choice": multiple_choice,
        "reachability": reachability,
    }
