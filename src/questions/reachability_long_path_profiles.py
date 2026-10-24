"""Deterministically slice reachability questions from longer sampled paths."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Sequence

import orjson

from src.questions.shared import prepare_question_path
from src.setup.constants import EDGE_CITED_BY, EDGE_CITING

LONG_PATH_PROFILE_FIELD = "reachability_long_path_profile"
LONG_PATH_PROFILE_POLICY = "three-segment-shared-roots-v1"


def invert_edge_type(edge_type: str) -> str:
    """Return the traversal type for the reverse direction of one edge."""
    if edge_type == EDGE_CITING:
        return EDGE_CITED_BY
    if edge_type == EDGE_CITED_BY:
        return EDGE_CITING
    return edge_type


def long_path_source_index(
    entry: dict[str, Any],
    *,
    question_hops: int,
    random_seed: int | None,
) -> int:
    """Choose a stable uniform gold-source index for one long input path."""
    steps = list(entry.get("path") or entry.get("steps") or [])
    lower = question_hops
    upper = len(steps) - 2 * question_hops
    if lower > upper:
        raise ValueError(
            f"Input path has {len(steps)} nodes, but question_hops="
            f"{question_hops} requires at least {3 * question_hops} nodes."
        )
    path_value = entry.get("path") or entry.get("steps") or []
    encoded = orjson.dumps([random_seed, question_hops, LONG_PATH_PROFILE_POLICY, path_value], option=orjson.OPT_SORT_KEYS)
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    return random.Random(seed).randint(lower, upper)


def sliced_step(
    step: dict[str, Any],
    *,
    position: int,
    first: bool,
) -> dict[str, Any]:
    """Copy one sampled-path step into a newly indexed gold path."""
    result = dict(step)
    result["step"] = position
    result["edge_taken"] = None if first else result.get("edge_taken")
    return result


def branch_node_ids(
    steps: Sequence[dict[str, Any]],
    indexes: Sequence[int],
) -> list[str]:
    """Return normalized node IDs for a selected branch."""
    return [str(steps[index]["node"]["node_id"]) for index in indexes]


def slice_long_path_entry(
    entry: dict[str, Any],
    *,
    question_hops: int,
    random_seed: int | None,
    source_index: int | None = None,
) -> dict[str, Any]:
    """Return one gold path plus exact-depth endpoint branches from a long path.

    The source position is drawn from ``random_seed`` unless ``source_index``
    fixes it; a fixed position may use the path's first and last nodes.
    """
    if question_hops < 2:
        raise ValueError("question_hops must be at least 2 nodes.")
    raw_steps = entry.get("path") or entry.get("steps") or []
    if not isinstance(raw_steps, list):
        raise ValueError("Sampled path must be a list of steps.")
    prepared = prepare_question_path(raw_steps)
    if prepared is None:
        raise ValueError("Sampled path is not usable for question construction.")
    _nodes, edges = prepared
    if any(
        not isinstance(step, dict)
        or not isinstance(step.get("node"), dict)
        or not str(step["node"].get("node_id") or "")
        for step in raw_steps
    ):
        raise ValueError("Every sampled-path step must contain a node_id.")
    if any(not isinstance(edge, dict) for edge in edges[1:]):
        raise ValueError("Every non-initial sampled-path step needs an edge.")

    if source_index is None:
        source_index = long_path_source_index(
            entry,
            question_hops=question_hops,
            random_seed=random_seed,
        )
    elif not question_hops - 1 <= source_index <= len(raw_steps) - 2 * question_hops + 1:
        raise ValueError(
            f"source_index={source_index} leaves no room for both "
            f"{question_hops}-node branches in a {len(raw_steps)}-node path."
        )
    target_index = source_index + question_hops - 1
    source_indexes = list(
        range(source_index, source_index - question_hops, -1)
    )
    gold_indexes = list(range(source_index, target_index + 1))
    target_indexes = list(range(target_index, target_index + question_hops))
    selected_indexes = set(source_indexes + gold_indexes + target_indexes)
    selected_node_ids = branch_node_ids(raw_steps, sorted(selected_indexes))
    if len(set(selected_node_ids)) != len(selected_node_ids):
        raise ValueError("Selected long-path segments repeat a node.")

    source_edge_types = [
        invert_edge_type(str(edges[index].get("edge_type") or ""))
        for index in source_indexes[:-1]
    ]
    target_edge_types = [
        str(edges[index].get("edge_type") or "")
        for index in target_indexes[1:]
    ]
    original_path_id = str(
        entry.get("path_id") or entry.get("path_index") or ""
    )
    result = dict(entry)
    result["input_path_id"] = original_path_id
    if "moves" in entry:
        result["input_moves"] = entry["moves"]
    result["path_id"] = (
        f"{original_path_id}:question_hops={question_hops}:"
        f"source_index={source_index}"
    )
    result["path"] = [
        sliced_step(
            raw_steps[index],
            position=position,
            first=position == 0,
        )
        for position, index in enumerate(gold_indexes)
    ]
    result["start_node"] = str(raw_steps[source_index]["node"]["node_id"])
    result["moves"] = question_hops - 1
    result.pop("steps", None)
    result[LONG_PATH_PROFILE_FIELD] = {
        "policy": LONG_PATH_PROFILE_POLICY,
        "input_path_id": original_path_id,
        "input_node_count": len(raw_steps),
        "question_hops": question_hops,
        "question_edge_depth": question_hops - 1,
        "source_index": source_index,
        "target_index": target_index,
        "source_wrong_branch": {
            "node_ids": branch_node_ids(raw_steps, source_indexes),
            "edge_types": source_edge_types,
        },
        "target_wrong_branch": {
            "node_ids": branch_node_ids(raw_steps, target_indexes),
            "edge_types": target_edge_types,
        },
    }
    return result


def whole_path_entry(entry: dict[str, Any], question_hops: int) -> dict[str, Any]:
    """Use a path of exactly 3k - 2 nodes whole: branch, gold path, branch.

    With that length there is a single layout, so nothing is drawn: the source
    is node k - 1 and the target node 2k - 2 (counting from 0).
    """
    steps = entry.get("path") or entry.get("steps") or []
    if len(steps) != 3 * question_hops - 2:
        raise ValueError(
            f"A question of {question_hops} nodes needs a path of exactly "
            f"{3 * question_hops - 2} nodes, not {len(steps)}."
        )
    return slice_long_path_entry(
        entry,
        question_hops=question_hops,
        random_seed=None,
        source_index=question_hops - 1,
    )


def slice_long_path_entries(
    entries: Sequence[dict[str, Any]],
    *,
    question_hops: int,
    random_seed: int | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Slice all usable paths and count deterministic rejection reasons."""
    if question_hops < 2:
        raise ValueError("question_hops must be at least 2 nodes.")
    sliced: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    for entry in entries:
        try:
            sliced.append(
                slice_long_path_entry(
                    entry,
                    question_hops=question_hops,
                    random_seed=random_seed,
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            reason = str(error) or type(error).__name__
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    return sliced, rejection_counts
