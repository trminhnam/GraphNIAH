"""Optional explored-paper filtering shared by reachability constructors."""

from collections.abc import Iterable, Mapping
from copy import copy
from dataclasses import dataclass
from typing import Any

from src.questions.reachability_long_path_profiles import LONG_PATH_PROFILE_FIELD
from src.setup.constants import nid_db_id


def validate_only_explored(value: int) -> None:
    if value not in (0, 1):
        raise ValueError("only_explored must be 0 or 1.")


def configure_only_explored(db: Any, value: int) -> None:
    """Enable the filter on the reader owned by this construction worker."""
    validate_only_explored(value)
    if value:
        db.only_explored = True


@dataclass
class ExploredAdjacency:
    """Filter citation and authorship neighbors before any growth sampling."""

    source: Any
    db: Any

    def get_many(self, node_ids: Iterable[str]) -> dict[str, list[Any]]:
        ids = tuple(dict.fromkeys(node_ids))
        batch = getattr(self.source, "get_many", None)
        rows = (
            batch(ids)
            if callable(batch)
            else {node_id: self.source.get(node_id, []) for node_id in ids}
        )
        papers = {node_id for node_id in ids if node_id.startswith("paper:")}
        papers.update(
            neighbor
            for edges in rows.values()
            for neighbor, _attrs in edges
            if neighbor.startswith("paper:")
        )
        allowed = allowed_explored_paper_ids(self.db, sorted(papers))
        return {
            node_id: [
                (neighbor, attrs)
                for neighbor, attrs in rows.get(node_id, [])
                if not neighbor.startswith("paper:") or neighbor in allowed
            ]
            if not node_id.startswith("paper:") or node_id in allowed
            else []
            for node_id in ids
        }

    def get(self, node_id: str, default: Any = None) -> list[Any]:
        return self.get_many((node_id,))[node_id]


def configure_explored_context(context: Any, value: int) -> None:
    """Keep canonical validation intact while filtering the construction graph."""
    validate_only_explored(value)
    if not value:
        return
    configure_only_explored(context.db, value)
    context.graph = copy(context.graph)
    context.graph.adj = ExploredAdjacency(context.graph.adj, context.db)


def allowed_explored_paper_ids(db: Any, paper_ids: Iterable[str]) -> frozenset[str]:
    """Batch-check explored flags; disabled filtering performs no DB reads."""
    ids = tuple(dict.fromkeys(paper_ids))
    if not getattr(db, "only_explored", False):
        return frozenset(ids)
    db_ids = [nid_db_id(paper_id) for paper_id in ids]
    batch = getattr(db, "get_paper_explored_many", None)
    if callable(batch):
        flags = batch(db_ids)
    else:
        flags = {db_id: db.get_paper_explored(db_id) for db_id in db_ids}
    return frozenset(
        paper_id for paper_id, db_id in zip(ids, db_ids) if flags.get(db_id) == 1
    )


def paper_ids_in(value: Any) -> set[str]:
    """Collect paper IDs from a graph/path payload, including reference keys."""
    if isinstance(value, str):
        return {value} if value.startswith("paper:") else set()
    if isinstance(value, Mapping):
        return paper_ids_in(list(value)) | paper_ids_in(list(value.values()))
    if isinstance(value, (list, tuple, set, frozenset)):
        result: set[str] = set()
        for item in value:
            result.update(paper_ids_in(item))
        return result
    return set()


def entry_paper_ids(entry: Mapping[str, Any]) -> set[str]:
    """Collect the papers checked in a sampled path and its wrong branches."""
    return paper_ids_in(
        [
            [
                step.get("node", {}).get("node_id")
                for step in (entry.get("path") or entry.get("steps") or [])
            ],
            entry.get(LONG_PATH_PROFILE_FIELD, {}),
        ]
    )


def input_path_explored_flags(
    entries: Iterable[Mapping[str, Any]], db: Any
) -> dict[str, bool] | None:
    """Keep one check pass's flags outside the reader's bounded LRU cache."""
    if not getattr(db, "only_explored", False):
        return None
    ids: set[str] = set()
    for entry in entries:
        ids.update(entry_paper_ids(entry))
    allowed = allowed_explored_paper_ids(db, ids)
    return {paper_id: paper_id in allowed for paper_id in ids}


def entry_is_explored(
    entry: Mapping[str, Any],
    db: Any,
    *,
    explored_flags: Mapping[str, bool] | None = None,
) -> bool:
    """Check a path using pass-local flags when supplied, otherwise the DB."""
    if not getattr(db, "only_explored", False):
        return True
    ids = entry_paper_ids(entry)
    if explored_flags is not None:
        return all(explored_flags.get(paper_id, False) for paper_id in ids)
    return ids.issubset(allowed_explored_paper_ids(db, ids))


def validate_explored_records(records: Iterable[Mapping[str, Any]], db: Any) -> None:
    """Fail before reusing a saved question or distractor with ineligible papers."""
    if not getattr(db, "only_explored", False):
        return
    ids: set[str] = set()
    for record in records:
        ids.update(
            paper_ids_in(
                [
                    record.get("needle_paper_ids", []),
                    record.get("node_ids", []),
                    record.get("reachability", {}).get("visible_graph", {}),
                    record.get("paper_negative_trees", []),
                ]
            )
        )
    rejected = ids - allowed_explored_paper_ids(db, sorted(ids))
    if rejected:
        example = min(rejected)
        raise ValueError(
            f"only_explored=1 rejects {len(rejected)} unexplored or missing "
            f"papers in saved questions/distractors (for example {example}). "
            "Regenerate these outputs using fresh output/state paths."
        )
