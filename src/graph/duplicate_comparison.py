"""Picklable workers for the notebook's duplicate-paper identity checks."""

from __future__ import annotations

from itertools import combinations

from src.questions.reachability_counting import normalize_author_list
from src.questions.reachability_yesno import normalize_visible_identity


def normalize_title_batch(batch: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(paper_id, normalize_visible_identity(title)) for paper_id, title in batch]


def compare_title_batch(batch: list[tuple[list[dict], dict[str, list[str]]]]) -> list:
    # Imported here because the shared merge module also calls these workers.
    from merge_graph_databases import (
        DuplicatePaperGroup,
        author_lists_match,
        identity_columns_conflict,
        published_years_conflict,
    )

    parent: dict[str, str] = {}

    def find(paper_id: str) -> str:
        parent.setdefault(paper_id, paper_id)
        while parent[paper_id] != paper_id:
            parent[paper_id] = parent[parent[paper_id]]
            paper_id = parent[paper_id]
        return paper_id

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    groups: list[DuplicatePaperGroup] = []
    for rows, raw_authors in batch:
        authors_by_paper = {
            paper_id: normalize_author_list(names)
            for paper_id, names in raw_authors.items()
        }
        ids = [str(row["id"]) for row in rows]
        for paper_id in ids:
            find(paper_id)

        compatible_pairs: set[frozenset[str]] = set()
        for row_a, row_b in combinations(rows, 2):
            if identity_columns_conflict(row_a, row_b):
                continue
            if published_years_conflict(row_a, row_b):
                continue
            id_a, id_b = str(row_a["id"]), str(row_b["id"])
            if author_lists_match(
                authors_by_paper.get(id_a, ()), authors_by_paper.get(id_b, ())
            ):
                compatible_pairs.add(frozenset((id_a, id_b)))
                union(id_a, id_b)

        members_by_root: dict[str, list[str]] = {}
        for paper_id in ids:
            members_by_root.setdefault(find(paper_id), []).append(paper_id)

        title = str(rows[0]["title"])
        for members in members_by_root.values():
            if len(members) < 2:
                continue
            expected_pairs = len(members) * (len(members) - 1) // 2
            actual_pairs = sum(
                1
                for a, b in combinations(members, 2)
                if frozenset((a, b)) in compatible_pairs
            )
            groups.append(
                DuplicatePaperGroup(
                    paper_ids=tuple(sorted(members)),
                    title=title,
                    is_clique=actual_pairs == expected_pairs,
                )
            )

    return groups
