"""Integration tests for balanced-child v5 reachability pipelines."""

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest

from src.questions import construction_run
from src.questions import reachability_rendered_pipeline_v5 as v5
from src.setup.constants import EDGE_CITED_BY, EDGE_CITING


class FakePaperDB:
    """Return unique paper titles and no graph authors."""

    def get_paper(self, db_id: str):
        """Return minimal paper metadata."""
        return {
            "title": f"Title {db_id}",
            "published": "",
            "venue": "",
            "arxiv_id": "",
        }

    def get_expanded_authors_for_paper(self, db_id: str):
        """Return no authors for one paper."""
        del db_id
        return []

    def get_expanded_authors_for_papers(self, db_ids):
        """Return no authors for a paper batch."""
        return {db_id: [] for db_id in db_ids}

    def close(self) -> None:
        """Match the real database lifecycle."""


def make_prefixed_graph(
    prefixes: tuple[str, ...],
    *,
    add_growth_leaves: bool = False,
):
    """Return disconnected nine-paper citation chains."""
    adjacency: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for prefix in prefixes:
        for index in range(9):
            adjacency[f"paper:{prefix}{index}"] = []
        for index in range(8):
            owner_id = f"paper:{prefix}{index}"
            cited_id = f"paper:{prefix}{index + 1}"
            adjacency[owner_id].append((cited_id, {"edge_type": EDGE_CITING}))
            adjacency[cited_id].append((owner_id, {"edge_type": EDGE_CITED_BY}))
        if add_growth_leaves:
            for side, owner_index in (("left", 3), ("right", 5)):
                owner_id = f"paper:{prefix}{owner_index}"
                for leaf_index in range(6):
                    leaf_id = f"paper:{prefix}-{side}-{leaf_index}"
                    adjacency[leaf_id] = [
                        (owner_id, {"edge_type": EDGE_CITED_BY})
                    ]
                    adjacency[owner_id].append(
                        (leaf_id, {"edge_type": EDGE_CITING})
                    )
    return SimpleNamespace(adj=adjacency)


def make_long_entry(prefix: str) -> dict:
    """Return one metadata-complete forward paper path."""
    return {
        "path_id": f"linear-{prefix}",
        "path": [
            {
                "step": index,
                "node": {
                    "node_id": f"paper:{prefix}{index}",
                    "node_type": "paper",
                    "title": f"Title {prefix}{index}",
                },
                "edge_taken": (
                    None
                    if index == 0
                    else {
                        "from": f"paper:{prefix}{index - 1}",
                        "to": f"paper:{prefix}{index}",
                        "edge_type": EDGE_CITING,
                    }
                ),
            }
            for index in range(9)
        ],
    }


def write_paths(path: Path, entries: list[dict]) -> None:
    """Write a small paper-only sampled-path input."""
    path.write_text(
        orjson.dumps({
                "metadata": {"traversal_mode": "papers-only"},
                "paths": entries,
            }).decode("utf-8"),
        encoding="utf-8",
    )


def patch_runtime(module, monkeypatch, tmp_path: Path, context) -> None:
    """Replace graph resolution and opening with one in-memory fixture."""
    constructor_base = (
        module.yesno_base if hasattr(module, "yesno_base") else module.base
    )
    monkeypatch.setattr(
        construction_run,
        "resolve_question_graph_paths",
        lambda *args, **kwargs: (
            tmp_path / "graph.db",
            tmp_path / "papers.db",
            "test",
        ),
    )
    monkeypatch.setattr(
        constructor_base,
        "open_lazy_caching_checker",
        lambda *args, **kwargs: context,
    )
    monkeypatch.setattr(
        construction_run,
        "collect_path_edge_validation_errors",
        lambda *args, **kwargs: (0, []),
    )


def test_v5_pair_cache_ignores_database_fingerprint_changes(
    tmp_path: Path,
) -> None:
    """V5 reuses pair results after graph database storage changes."""
    cache_path = tmp_path / "pairs.json"
    saved_configuration = {
        "version": v5.V5_PAIR_CACHE_VERSION,
        "constructor_version": "v5",
        "graph": {"path": "old-graph", "size": 1, "mtime_ns": 2},
        "db": {"path": "old-db", "size": 3, "mtime_ns": 4},
        "traversal_mode": "papers-only",
        "question_hops": 4,
        "max_author_answer_index": 10,
        "num_children_per_node": 8,
    }
    cache = v5.load_pair_disjointness_cache(cache_path, saved_configuration)
    cache.put("first", "second", "compatible")
    cache.write(indent=2)

    current_configuration = {
        **saved_configuration,
        "graph": {"path": "new-graph", "size": 5, "mtime_ns": 6},
        "db": {"path": "new-db", "size": 7, "mtime_ns": 8},
    }
    restored = v5.load_pair_disjointness_cache(cache_path, current_configuration)

    assert restored.get("first", "second") == "compatible"
    assert restored.configuration == current_configuration

    incompatible_configuration = {
        **current_configuration,
        "question_hops": 6,
    }
    with pytest.raises(ValueError, match="configuration.question_hops"):
        v5.load_pair_disjointness_cache(cache_path, incompatible_configuration)


def test_v5_cutoff_date_is_strict_and_inclusive() -> None:
    cutoff = v5.parse_cutoff_date("2025-09-01")
    assert cutoff.isoformat() == "2025-09-01"
    with pytest.raises(ValueError, match="cutoff date"):
        v5.parse_cutoff_date("09/01/2025")


def test_v5_paper_cutoff_primes_dates_with_one_batch() -> None:
    class PublishedPaperDB:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def get_paper_published_many(self, db_ids):
            self.calls.append(list(db_ids))
            return {
                "old": "2024-12-31",
                "new": "2025-01-01",
                "missing": "",
            }

        def get_paper_published(self, db_id: str) -> str:
            raise AssertionError(f"Unexpected single-paper query: {db_id}")

    db = PublishedPaperDB()
    cutoff = v5.PaperDateCutoff(
        db=db,
        cutoff_date=date(2025, 1, 1),
        published_dates={},
    )

    allowed = cutoff.allowed_paper_ids(
        ("paper:old", "paper:new", "paper:missing", "paper:new")
    )

    assert allowed == frozenset({"paper:new"})
    assert db.calls == [["old", "new", "missing"]]
    assert cutoff.allowed_paper_ids(("paper:new",)) == frozenset({"paper:new"})
    assert len(db.calls) == 1


def test_v5_paper_block_policy_rejects_an_unmentioned_logical_paper() -> None:
    with pytest.raises(
        ValueError,
        match="Logical papers without blocks must be cited",
    ):
        v5.citation_owner_paper_ids(
            ["paper:a", "paper:b"],
            {"paper:a": [], "paper:b": []},
        )


def test_v5_reuses_structurally_stuck_growth_for_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    growth_result = SimpleNamespace(
        all_targets_reached=False,
        trees=("partial-tree",),
        rejection_counts={"frontier_cannot_increase_target": 1},
        maximum_depth_processed=2,
    )
    growth_calls: list[object] = []

    def grow_once(*args, **kwargs):
        growth_calls.append((args, kwargs))
        return growth_result

    initial_calibration = v5.ReferenceCalibration((), (), 0.0, 0.0)
    fallback_plan = v5.GrowthPlan(
        trees=("partial-tree",),
        target_kinds=(v5.TARGET_COMPONENT,),
        target_value=5,
        bucket_label="4-8",
        calibration=initial_calibration,
        used_fallback=True,
        reused_higher_bucket_tree=False,
        removed_paper_counts=(0,),
        removed_reference_pairs=((),),
        maximum_depth_processed=2,
        rejection_counts=growth_result.rejection_counts,
    )
    monkeypatch.setattr(v5, "grow_question_group", grow_once)
    monkeypatch.setattr(
        v5.v4,
        "final_plan_from_ladders",
        lambda *args, **kwargs: fallback_plan,
    )

    plan = v5.plan_new_growth_group(
        [object()],
        [v5.TARGET_COMPONENT],
        sampled_target=5,
        current_bucket=("4-8", 4, 8),
        fallback_bucket_specs=[("4-8", 4, 8)],
        facts_reader=object(),
        include_author_identity=False,
        num_children_per_node=1,
        random_seed=1,
        key="reuse-structurally-stuck-v5",
    )

    assert plan is fallback_plan
    assert len(growth_calls) == 1


