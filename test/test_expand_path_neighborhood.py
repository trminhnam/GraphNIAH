from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import orjson
import pytest

from endpoints import expand_papers_in_paths as path_module
from endpoints import expand_path_neighborhood as module
from src.graph import path_expansion_levels as levels
from src.graph.storage import DatabaseManager
from test.test_expand_papers_in_paths import (
    FakeBulkClient,
    add_paper_authors,
    db_id,
    dropped_records,
    expansion,
    paper,
    path_node,
    saved_paper,
    write_paths,
    write_walks,
)
from test.test_expand_papers_in_paths import (
    graph as graph,  # noqa: PLC0414 -- shared pytest fixture
)
from test.test_expand_papers_in_paths import (
    index_root as index_root,  # noqa: PLC0414 -- shared pytest fixture
)


def write_seed_paths(
    path: Path, database, ids: list[str], cutoff: str | None = "2000-01-01"
) -> Path:
    write_paths(path, database.db_path, [path_node(pid) for pid in ids])
    payload = orjson.loads(path.read_bytes())
    if cutoff is not None:
        payload["metadata"]["cutoff_date"] = cutoff
    path.write_bytes(orjson.dumps(payload))
    return path


def linked_paper(sid: str, references: tuple = (), citations: tuple = ()) -> tuple:
    source = paper(sid)
    source["referenceCount"] = len(references)
    return (
        source,
        [{"citingPaper": paper(neighbor)} for neighbor in citations],
        [{"citedPaper": paper(neighbor)} for neighbor in references],
    )


@dataclass
class CutoffBulkClient(FakeBulkClient):
    cutoffs: list[str] = field(default_factory=list)

    def iter_expansion_subgraph_batches(self, ids, **kwargs):
        cutoff = kwargs["min_publication_date"]
        self.cutoffs.append(cutoff)
        assert kwargs["include_citations"] and kwargs["include_references"]
        assert kwargs["edge_budget"] > 0
        batch = self.get_papers_with_citations_and_references_batch(
            ids,
            fields=kwargs["fields"],
            max_results=kwargs["max_results"],
            include_citations=True,
            include_references=True,
        )
        yield {
            sid: (
                source,
                [
                    row
                    for row in citations
                    if (row["citingPaper"].get("publicationDate") or "") >= cutoff
                ],
                [
                    row
                    for row in references
                    if (row["citedPaper"].get("publicationDate") or "") >= cutoff
                ],
            )
            for sid, (source, citations, references) in batch.items()
        }


def run(paths: list[Path], index: Path, *options: str) -> int:
    return module.main(
        [
            *(str(path) for path in paths),
            "--bulk-index-root",
            str(index),
            "--cutoff-date",
            "2000-01-01",
            *options,
        ]
    )


def test_neighbor_scan_batches_metadata_and_skips_visited_papers(
    graph, monkeypatch
) -> None:
    seeds = [graph.insert_paper(title=f"Seed {index}") for index in range(3)]
    neighbors = {
        name: graph.insert_paper(
            title=name, semantic_scholar_id=name, published=published
        )
        for name, published in (
            ("boundary", "2000-01-01"),
            ("timestamp", "2000-01-02T12:00:00Z"),
            ("recent", "2001-01-01"),
            ("old", "1999-12-31"),
            ("unknown", ""),
            ("invalid", "invalid date"),
        )
    }
    for seed in seeds:
        for neighbor in [*seeds, *neighbors.values()]:
            graph.insert_citation(citing_paper_id=seed, cited_paper_id=neighbor)
            graph.insert_citation(citing_paper_id=neighbor, cited_paper_id=seed)

    batches = []
    original = levels.collect_neighbor_metadata

    def record_metadata(connection, neighbor_ids, cutoff_date):
        if neighbor_ids:
            batches.append(set(neighbor_ids))
        return original(connection, neighbor_ids, cutoff_date)

    monkeypatch.setattr(levels, "SQL_ID_BATCH_SIZE", 2)
    monkeypatch.setattr(levels, "collect_neighbor_metadata", record_metadata)
    visited = set(seeds)
    assert levels.collect_next_path_level(
        graph.conn, seeds, visited, date(2000, 1, 1)
    ) == {neighbors[name]: name for name in ("boundary", "timestamp", "recent")}
    assert visited == set(seeds)
    assert batches
    assert all(len(batch) <= 2 and batch.isdisjoint(visited) for batch in batches)
    assert all(graph.get_paper(pid)["explored"] == 0 for pid in neighbors.values())


@pytest.mark.parametrize("boundary_state", [None, (1, None), (1, 1)])
def test_four_level_boundary_both_directions_cycles_and_shared_seeds(
    graph, index_root, tmp_path, monkeypatch, boundary_state
) -> None:
    seed = graph.insert_paper(title="Seed", semantic_scholar_id="seed")
    outside = graph.insert_paper(title="Outside", semantic_scholar_id="outside")
    if boundary_state is not None:
        boundary_id = graph.insert_paper(title="Four", semantic_scholar_id="four")
        graph.update_paper(
            boundary_id,
            explored=boundary_state[0],
            explored_after_cutoff=boundary_state[1],
        )
    paths = [
        write_seed_paths(tmp_path / name, graph, [seed])
        for name in ("one.json", "two.json")
    ]
    client = CutoffBulkClient(
        {
            "seed": linked_paper("seed", ("one",), ("incoming",)),
            "one": linked_paper("one", ("two",), ("seed",)),
            "incoming": linked_paper("incoming", ("seed", "two")),
            "two": linked_paper("two", ("three",), ("one", "incoming")),
            "three": linked_paper("three", ("four",), ("two",)),
            "four": linked_paper("four", ("five",)),
        }
    )
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    assert run(paths, index_root, "--bulk-batch-api-group-size", "2") == 0
    assert [set(group) for group in client.calls] == [
        {"seed"},
        {"one", "incoming"},
        {"two"},
        {"three"},
    ]
    assert client.opened == client.closed == 1
    assert client.cutoffs == ["2000-01-01"] * 4
    for sid in ("seed", "one", "incoming", "two", "three"):
        row = graph.get_paper_by_semantic_scholar_id(sid)
        assert row["explored"] == row["explored_after_cutoff"] == 1
        assert row["valid_references"] is None
        assert graph.get_expansion_lease("paper", row["id"])["state"] == "complete"
    boundary = graph.get_paper_by_semantic_scholar_id("four")
    assert (boundary["explored"], boundary["explored_after_cutoff"]) == (
        boundary_state or (0, None)
    )
    assert graph.get_expansion_lease("paper", boundary["id"]) is None
    assert graph.get_paper_by_semantic_scholar_id("five") is None
    assert graph.get_paper(outside)["explored"] == 0
    assert list(graph.conn.execute("PRAGMA foreign_key_check")) == []


def test_reuses_full_expansions_but_filters_stored_neighbors(
    graph, index_root, tmp_path, monkeypatch
) -> None:
    seed = graph.insert_paper(title="Seed", semantic_scholar_id="seed", explored=True)
    child = graph.insert_paper(
        title="Child", semantic_scholar_id="child", published="2000-01-01"
    )
    graph.insert_citation(citing_paper_id=seed, cited_paper_id=child)
    for name, published in (("old", "1999-12-31"), ("unknown", "")):
        pid = graph.insert_paper(
            title=name, semantic_scholar_id=name, published=published
        )
        graph.insert_citation(citing_paper_id=seed, cited_paper_id=pid)
        graph.insert_citation(citing_paper_id=pid, cited_paper_id=seed)
    paths = write_seed_paths(tmp_path / "paths.json", graph, [seed])
    data = linked_paper("child", ("old", "unknown", "boundary"))
    data[2][0]["citedPaper"]["publicationDate"] = "1999-12-31"
    data[2][1]["citedPaper"]["publicationDate"] = None
    client = CutoffBulkClient({"child": data})
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    assert run([paths], index_root, "--max-level", "2") == 0
    assert client.calls == [["child"]]
    assert graph.get_paper(seed)["explored_after_cutoff"] is None
    assert graph.get_paper_by_semantic_scholar_id("old")["explored"] == 0
    assert graph.get_paper_by_semantic_scholar_id("unknown")["explored"] == 0
    assert graph.get_paper_by_semantic_scholar_id("boundary")["explored"] == 0
    assert graph.get_paper(child)["explored_after_cutoff"] == 1


def test_reopens_unknown_cutoff_and_full_path_expansion_clears_marker(
    graph, index_root, tmp_path, monkeypatch
) -> None:
    seed = graph.insert_paper(title="Seed", semantic_scholar_id="seed", explored=True)
    graph.update_paper(seed, explored_after_cutoff=1)
    graph.claim_expansion("paper", seed, "previous", now=0, lease_until=1)
    graph.complete_expansion("paper", seed, "previous")
    paths = write_seed_paths(tmp_path / "paths.json", graph, [seed])
    client = CutoffBulkClient({"seed": linked_paper("seed", ("boundary",))})
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    assert run([paths], index_root, "--max-level", "1") == 0
    assert client.calls == [["seed"]]
    assert graph.get_paper(seed)["explored_after_cutoff"] == 1
    assert run([paths], index_root, "--max-level", "1") == 0
    assert client.calls == [["seed"]]
    assert run([paths], index_root, "--max-level", "1", "--restart") == 0
    assert client.calls == [["seed"], ["seed"]]

    full_client = FakeBulkClient({"seed": expansion("seed")})
    monkeypatch.setattr(path_module, "SemanticScholarBulkClient", full_client.create)
    assert path_module.main([str(paths), "--bulk-index-root", str(index_root)]) == 0
    assert full_client.calls == [["seed"]]
    assert graph.get_paper(seed)["explored"] == 1
    assert graph.get_paper(seed)["explored_after_cutoff"] is None


@pytest.mark.parametrize("failure", ["not_found", "fetch", "write"])
@pytest.mark.parametrize("previous_cutoff", [False, True])
def test_failed_paper_preserves_state_and_can_retry(
    graph, index_root, tmp_path, monkeypatch, failure, previous_cutoff
) -> None:
    seed = graph.insert_paper(title="Seed", semantic_scholar_id="seed")
    if previous_cutoff:
        graph.update_paper(seed, explored=1, explored_after_cutoff=1)
    paths = write_seed_paths(tmp_path / "paths.json", graph, [seed])
    data = {"seed": linked_paper("seed", ("child",)), "child": expansion("child")}
    client = CutoffBulkClient({} if failure == "not_found" else data)
    client.fail_fetch = failure == "fetch"
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)

    def fail_write(self, edges):
        raise RuntimeError("write failed")

    with monkeypatch.context() as patch:
        if failure == "write":
            patch.setattr(module.PathExpansionWriter, "write_edges", fail_write)
        assert run([paths], index_root, "--max-level", "2") == 1
    assert client.calls == [["seed"]]
    assert graph.get_paper(seed)["explored"] == int(previous_cutoff)
    assert graph.get_paper(seed)["explored_after_cutoff"] == (
        1 if previous_cutoff else None
    )
    assert graph.get_expansion_lease("paper", seed)["state"] == "pending"
    assert graph.get_paper_by_semantic_scholar_id("child") is None
    client.responses = data
    client.fail_fetch = False
    if failure == "not_found":
        # A miss stays cached across reruns of the same bulk snapshot.
        assert run([paths], index_root, "--max-level", "2") == 1
        assert client.calls == [["seed"]]
        metadata_path = index_root / "metadata.json"
        metadata = orjson.loads(metadata_path.read_bytes())
        metadata["createdAt"] = "rebuilt-index"
        metadata_path.write_bytes(orjson.dumps(metadata))
    retry_options = () if failure == "not_found" else ("--restart",)
    assert run([paths], index_root, "--max-level", "2", *retry_options) == 0
    assert graph.get_paper_by_semantic_scholar_id("child")["explored_after_cutoff"] == 1


def test_cached_misses_reuse_stored_edges_and_continue_to_boundary(
    graph, index_root, tmp_path, monkeypatch
) -> None:
    seed = graph.insert_paper(title="Seed", semantic_scholar_id="seed")
    child = graph.insert_paper(
        title="Child", semantic_scholar_id="child", published="2000-01-01"
    )
    graph.insert_citation(citing_paper_id=child, cited_paper_id=seed)
    path_module.record_bulk_lookup_misses(graph, [seed], "")
    # A failure from an older index must not prevent a fresh lookup.
    path_module.record_bulk_lookup_misses(graph, [child], "old-index")
    paths = write_seed_paths(tmp_path / "paths.json", graph, [seed])
    client = CutoffBulkClient(
        {
            "child": linked_paper("child", ("two",)),
            "two": linked_paper("two", ("three",)),
            "three": linked_paper("three", ("four",)),
        }
    )
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    assert run([paths], index_root) == 1
    assert client.calls == [["child"], ["two"], ["three"]]
    assert graph.get_paper(seed)["explored"] == 0
    assert graph.get_paper(seed)["explored_after_cutoff"] is None
    assert graph.get_expansion_lease("paper", seed) is None
    assert graph.get_paper_by_semantic_scholar_id("four")["explored"] == 0


def test_only_cached_misses_do_not_start_spark(
    graph, index_root, tmp_path, monkeypatch
) -> None:
    seeds = [graph.insert_paper(title=f"Missing {index}") for index in range(5)]
    path_module.record_bulk_lookup_misses(graph, seeds, "")
    paths = write_seed_paths(tmp_path / "paths.json", graph, seeds)

    def unexpected_client(**kwargs):
        pytest.fail("Cached misses must not start Spark")

    monkeypatch.setattr(module, "SemanticScholarBulkClient", unexpected_client)
    monkeypatch.setattr(levels, "SQL_ID_BATCH_SIZE", 2)
    assert run([paths], index_root) == 1
    assert all(graph.get_paper(seed)["explored"] == 0 for seed in seeds)


@pytest.mark.parametrize("interruption", ["after_commit", "inside_transaction", "scan"])
def test_resume_saved_level_and_committed_papers(
    graph, index_root, tmp_path, monkeypatch, interruption
) -> None:
    seed = graph.insert_paper(title="Seed", semantic_scholar_id="seed", explored=True)
    children = [
        graph.insert_paper(title=name, semantic_scholar_id=name, published="2000-01-02")
        for name in ("a", "b")
    ]
    for child in children:
        graph.insert_citation(citing_paper_id=seed, cited_paper_id=child)
    paths = write_seed_paths(tmp_path / "paths.json", graph, [seed])
    client = CutoffBulkClient(
        {
            "a": linked_paper("a", ("two",)),
            "b": linked_paper("b", ("two",)),
            "two": linked_paper("two", ("three",)),
            "three": linked_paper("three", ("four",)),
        }
    )
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    original_ingest = module.PathExpansionWriter.ingest
    original_success = module.PathExpansionCheckpoint.record_success
    original_scan = module.collect_next_path_level

    def interrupt_after_commit(self, node_id, source, citations, references):
        original_ingest(self, node_id, source, citations, references)
        raise KeyboardInterrupt

    def interrupt_in_transaction(self, paper_id):
        original_success(self, paper_id)
        raise KeyboardInterrupt

    def interrupt_scan(connection, source_ids, visited, cutoff):
        if seed not in source_ids:
            raise KeyboardInterrupt
        return original_scan(connection, source_ids, visited, cutoff)

    with monkeypatch.context() as patch:
        if interruption == "after_commit":
            patch.setattr(module.PathExpansionWriter, "ingest", interrupt_after_commit)
        elif interruption == "inside_transaction":
            patch.setattr(
                module.PathExpansionCheckpoint,
                "record_success",
                interrupt_in_transaction,
            )
        else:
            patch.setattr(module, "collect_next_path_level", interrupt_scan)
        assert run([paths], index_root, "--bulk-batch-api-group-size", "2") == 130

    state = graph.conn.execute(
        "SELECT level, finished FROM path_expansion_runs"
    ).fetchone()
    assert tuple(state) == (1, 0)
    assert {
        row[0]
        for row in graph.conn.execute(
            "SELECT paper_id FROM path_expansion_frontiers WHERE level = 1"
        )
    } == set(children)
    completed = {
        row[0]
        for row in graph.conn.execute(
            "SELECT paper_id FROM path_expansion_results WHERE status = 'expanded'"
        )
    }
    assert (
        len(completed)
        == {"after_commit": 1, "inside_transaction": 0, "scan": 2}[interruption]
    )
    assert {pid for pid in children if graph.get_paper(pid)["explored"]} == completed
    if interruption == "inside_transaction":
        assert graph.get_paper_by_semantic_scholar_id("two") is None

    def resumed_scan(connection, source_ids, visited, cutoff):
        assert seed not in source_ids, "Resume must not rebuild earlier level lists"
        return original_scan(connection, source_ids, visited, cutoff)

    monkeypatch.setattr(module, "collect_next_path_level", resumed_scan)
    client.calls.clear()
    # Changing batch size must reuse the same checkpoint.
    assert run([paths], index_root, "--bulk-batch-api-group-size", "1") == 0
    expected = {
        graph.get_paper(pid)["semantic_scholar_id"] for pid in set(children) - completed
    }
    assert {sid for group in client.calls for sid in group} == expected | {
        "two",
        "three",
    }
    assert graph.get_paper_by_semantic_scholar_id("four")["explored"] == 0
    assert tuple(
        graph.conn.execute("SELECT level, finished FROM path_expansion_runs").fetchone()
    ) == (4, 1)
    calls = list(client.calls)
    assert run([paths], index_root) == 0
    assert client.calls == calls


def test_checkpoint_isolated_by_cutoff_and_invalidated_by_changed_seeds(
    graph, index_root, tmp_path, monkeypatch
) -> None:
    seed = graph.insert_paper(title="Seed", semantic_scholar_id="seed")
    paths = write_seed_paths(tmp_path / "paths.json", graph, [seed])
    client = CutoffBulkClient({"seed": linked_paper("seed", ("child",))})
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    assert run([paths], index_root, "--max-level", "1") == 0
    assert (
        run([paths], index_root, "--max-level", "1", "--cutoff-date", "2000-01-02") == 0
    )
    assert client.calls == [["seed"], ["seed"]]
    assert (
        graph.conn.execute("SELECT COUNT(*) FROM path_expansion_runs").fetchone()[0]
        == 2
    )
    other = graph.insert_paper(title="Other", semantic_scholar_id="other")
    write_seed_paths(paths, graph, [other])
    client.responses["other"] = linked_paper("other")
    assert run([paths], index_root, "--max-level", "1") == 0
    assert client.calls[-1] == ["other"]


@pytest.mark.parametrize(
    "failure", ["not_found", "duplicate_conflict", "fetch", "write"]
)
def test_unresolved_papers_do_not_block_later_levels(
    graph, index_root, tmp_path, monkeypatch, failure
) -> None:
    good = graph.insert_paper(title="Good", semantic_scholar_id="good")
    bad = graph.insert_paper(
        title="Ambiguous",
        semantic_scholar_id="" if failure == "duplicate_conflict" else "bad",
    )
    child = graph.insert_paper(
        title="Stored Child", semantic_scholar_id="stored-child", published="2000-01-02"
    )
    graph.insert_citation(citing_paper_id=bad, cited_paper_id=child)
    graph.insert_paper(
        title="Different Title", semantic_scholar_id="owner", explored=True
    )
    paths = write_seed_paths(tmp_path / "paths.json", graph, [bad, good])
    client = CutoffBulkClient(
        {
            "good": linked_paper("good", ("one",)),
            "one": linked_paper("one", ("two", "missing-at-level-2")),
            "two": linked_paper("two", ("three",)),
            "three": linked_paper("three", ("four",)),
            "stored-child": linked_paper("stored-child", ("two",)),
            **({"bad": expansion("bad")} if failure in {"fetch", "write"} else {}),
        },
        {"Ambiguous": {"paperId": "owner", "matchScore": 100}},
    )
    original_fetch = client.get_papers_with_citations_and_references_batch
    original_ingest = module.PathExpansionWriter.ingest

    def fetch(ids, **kwargs):
        if failure == "fetch" and ids == ["bad"]:
            raise RuntimeError("temporary fetch failure")
        return original_fetch(ids, **kwargs)

    def ingest(self, node_id, source, citations, references):
        if failure == "write" and source["paperId"] == "bad":
            raise RuntimeError("temporary write failure")
        return original_ingest(self, node_id, source, citations, references)

    monkeypatch.setattr(client, "get_papers_with_citations_and_references_batch", fetch)
    monkeypatch.setattr(module.PathExpansionWriter, "ingest", ingest)
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    assert run([paths], index_root, "--bulk-batch-api-group-size", "1") == 1
    assert graph.get_paper(bad)["explored"] == 0
    assert graph.get_paper(bad)["explored_after_cutoff"] is None
    for sid in ("good", "one", "two", "three", "stored-child"):
        row = graph.get_paper_by_semantic_scholar_id(sid)
        assert row["explored"] == row["explored_after_cutoff"] == 1
        assert client.calls.count([sid]) == 1
    assert graph.get_paper_by_semantic_scholar_id("four")["explored"] == 0
    assert graph.get_paper_by_semantic_scholar_id("missing-at-level-2")["explored"] == 0
    assert client.opened == client.closed == 1


def test_duplicate_seed_remaps_and_traverses_winner(
    graph, index_root, tmp_path, monkeypatch
) -> None:
    owner = graph.insert_paper(title="Same Title", semantic_scholar_id="resolved")
    loser = graph.insert_paper(title="Same Title")
    for pid in (owner, loser):
        add_paper_authors(graph, pid, ["First Author"])
    paths = write_seed_paths(tmp_path / "paths.json", graph, [loser])
    data = linked_paper("resolved", ("child",))
    data[0]["title"] = "Same Title"
    client = CutoffBulkClient(
        {"resolved": data, "child": expansion("child")},
        {"Same Title": {"paperId": "resolved", "matchScore": 100}},
    )
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    assert run([paths], index_root, "--max-level", "2") == 0
    assert client.calls == [["resolved"], ["child"]]
    assert graph.get_paper(loser) is None
    payload = orjson.loads(paths.read_bytes())
    assert payload["paths"][0]["path"][0]["node"]["node_id"] == f"paper:{owner}"
    assert run([paths], index_root, "--max-level", "2") == 0
    assert client.calls == [["resolved"], ["child"]]
    assert (
        graph.conn.execute("SELECT COUNT(*) FROM path_expansion_runs").fetchone()[0]
        == 1
    )


@pytest.mark.parametrize(
    ("options", "expected_cutoff", "expected_neighbors"),
    [
        ([], "2025-09-01", {"boundary", "recent"}),
        (["--cutoff-date", "2025-09-10"], "2025-09-10", {"recent"}),
        (["--cutoff-date", "2025-09"], "2025-09-01", {"boundary", "recent"}),
        (["--cutoff-date", "2025"], "2025-01-01", {"old", "boundary", "recent"}),
    ],
)
def test_cli_cutoff_filters_fetched_and_stored_neighbors_without_path_metadata(
    graph,
    index_root,
    tmp_path,
    monkeypatch,
    options,
    expected_cutoff,
    expected_neighbors,
) -> None:
    seed = graph.insert_paper(title="Seed", semantic_scholar_id="seed")
    stored = graph.insert_paper(
        title="Stored", semantic_scholar_id="stored", explored=True
    )
    data = linked_paper("seed", ("old", "boundary", "recent"))
    responses = {"seed": data}
    dates = {"old": "2025-08-31", "boundary": "2025-09-01", "recent": "2025-09-10"}
    for item in data[2]:
        neighbor = item["citedPaper"]
        sid = neighbor["paperId"]
        neighbor["publicationDate"] = dates[sid]
        pid = graph.insert_paper(
            title=sid, semantic_scholar_id=sid, published=dates[sid]
        )
        graph.insert_citation(citing_paper_id=stored, cited_paper_id=pid)
        responses[sid] = expansion(sid)
        responses[sid][0]["publicationDate"] = dates[sid]
    paths = [
        write_seed_paths(
            tmp_path / f"paths-{index}.json", graph, [seed, stored], cutoff
        )
        for index, cutoff in enumerate((None, "invalid", "2030"))
    ]
    client = CutoffBulkClient(responses)
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    assert (
        module.main(
            [
                *(str(path) for path in paths),
                "--bulk-index-root",
                str(index_root),
                "--max-level",
                "2",
                *options,
            ]
        )
        == 0
    )
    assert [set(group) for group in client.calls] == [{"seed"}, expected_neighbors]
    assert client.cutoffs == [expected_cutoff] * 2
    fetched_ids = {
        row[0]
        for row in graph.conn.execute(
            "SELECT p.semantic_scholar_id FROM citations c "
            "JOIN papers p ON p.id = c.cited_paper_id WHERE c.citing_paper_id = ?",
            (seed,),
        )
    }
    assert fetched_ids == expected_neighbors
    assert graph.get_paper(seed)["explored_after_cutoff"] == 1
    assert graph.get_paper(stored)["explored_after_cutoff"] is None


@pytest.mark.parametrize("cutoff", ["invalid", "2025-02-30", ""])
def test_rejects_invalid_cli_cutoff_before_expansion(
    graph, index_root, tmp_path, monkeypatch, cutoff
) -> None:
    seed = graph.insert_paper(title="Seed", semantic_scholar_id="seed")
    paths = write_seed_paths(tmp_path / "paths.json", graph, [seed], None)
    client = CutoffBulkClient({})
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    assert run([paths], index_root, "--cutoff-date", cutoff) == 2
    assert client.opened == 0
    assert graph.get_paper(seed)["explored"] == 0


def test_full_graph_without_cutoff_metadata_needs_no_spark(
    graph, tmp_path, monkeypatch
) -> None:
    seed = graph.insert_paper(title="Seed", semantic_scholar_id="seed", explored=True)
    paths = [write_seed_paths(tmp_path / "paths.json", graph, [seed], None)]
    client = CutoffBulkClient({})
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    assert run(paths, tmp_path / "absent-index") == 0
    assert client.opened == 0


def dated(sid: str, published: str = "2000-01-02") -> dict:
    record = paper(sid)
    record["publicationDate"] = published
    return record


def bulk_entry(
    sid: str, published: str = "2000-01-02", references=(), citations=()
) -> tuple:
    """Bulk record; neighbors are (Semantic Scholar ID, publication date) pairs."""
    source = dated(sid, published)
    source["referenceCount"] = len(references)
    return (
        source,
        [{"citingPaper": dated(*neighbor)} for neighbor in citations],
        [{"citedPaper": dated(*neighbor)} for neighbor in references],
    )


def run_seeded(
    paths: list[Path], index: Path, db_path: Path, max_level: str, *options: str
) -> int:
    return run(
        paths,
        index,
        "--db-path",
        str(db_path),
        "--seed-missing-papers",
        "--max-level",
        max_level,
        *options,
    )


def test_seed_missing_papers_builds_database_and_expands_neighborhood(
    index_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "new" / "graph.db"
    first, second = saved_paper("a"), saved_paper("b")
    paths = write_walks(
        tmp_path / "paths.json", db_path, [[(first, ""), (second, "citing")]]
    )
    saved = paths.read_bytes()
    assert run([paths], index_root, "--db-path", str(db_path)) == 2
    assert not db_path.exists()
    client = CutoffBulkClient(
        {
            "a": bulk_entry("a", references=[("b", "2000-01-02"), ("n1", "2000-01-02")]),
            "b": bulk_entry("b", citations=[("a", "2000-01-02")]),
            "n1": bulk_entry("n1", references=[("n2", "2000-01-02")]),
        }
    )
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)

    assert run_seeded([paths], index_root, db_path, "2") == 0
    assert paths.read_bytes() == saved
    graph = DatabaseManager(db_path)
    try:
        for node in (first, second):
            row = graph.get_paper(db_id(node))
            assert row["depth"] == 0
            assert row["explored"] == row["explored_after_cutoff"] == 1
        assert graph.get_paper_by_semantic_scholar_id("n1")["explored"] == 1
        assert graph.get_paper_by_semantic_scholar_id("n2")["explored"] == 0
        assert (db_id(first), db_id(second)) in {
            (row["citing_paper_id"], row["cited_paper_id"])
            for row in graph.all_citations()
        }
    finally:
        graph.close()


def test_seed_drops_disproved_paths_before_the_next_level(
    index_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "graph.db"
    source, target = saved_paper("a"), saved_paper("b")
    lost, unlinked = saved_paper("lost"), saved_paper("c")
    paths = write_walks(
        tmp_path / "paths.json",
        db_path,
        [
            [(source, ""), (target, "citing")],
            [(source, ""), (lost, "citing")],
            [(source, ""), (unlinked, "citing")],
        ],
    )
    client = CutoffBulkClient(
        {
            "a": bulk_entry("a", references=[("b", "2000-01-02")]),
            "b": bulk_entry("b", citations=[("a", "2000-01-02")]),
            "c": bulk_entry("c", references=[("c-only", "2000-01-02")]),
            "c-only": bulk_entry("c-only"),
        }
    )
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)

    # The not-found paper leaves with its path, so the run is complete.
    assert run_seeded([paths], index_root, db_path, "2", "--drop-disproved-paths") == 0
    kept = orjson.loads(paths.read_bytes())["paths"]
    assert [entry["path_id"] for entry in kept] == ["path-0"]
    assert {
        record["path_id"]: [reason["reason"] for reason in record["reasons"]]
        for record in dropped_records(paths)
    } == {"path-1": ["missing_node"], "path-2": ["missing_edge"]}
    graph = DatabaseManager(db_path)
    try:
        # Level 1 starts from the kept seeds, so the dropped path's neighbor waits.
        assert graph.get_paper_by_semantic_scholar_id("c-only")["explored"] == 0
    finally:
        graph.close()
    calls = list(client.calls)
    # The checkpoint follows the rewritten files instead of starting over.
    assert run_seeded([paths], index_root, db_path, "2", "--drop-disproved-paths") == 0
    assert client.calls == calls
    assert len(dropped_records(paths)) == 2


def test_seed_keeps_paths_whose_edge_the_cutoff_may_hide(
    index_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "graph.db"
    old_source, old_target = saved_paper("old-a"), saved_paper("old-b")
    for node in (old_source, old_target):
        node["published"] = "1999-01-01"
    new_source, new_target = saved_paper("new-a"), saved_paper("new-b")
    paths = write_walks(
        tmp_path / "paths.json",
        db_path,
        [
            [(old_source, ""), (old_target, "citing")],
            [(new_source, ""), (new_target, "citing")],
        ],
    )
    client = CutoffBulkClient(
        {
            # The cutoff filter removes this older edge from both sides.
            "old-a": bulk_entry("old-a", "1999-01-01", references=[("old-b", "1999-01-01")]),
            "old-b": bulk_entry("old-b", "1999-01-01", citations=[("old-a", "1999-01-01")]),
            "new-a": bulk_entry("new-a"),
            "new-b": bulk_entry("new-b"),
        }
    )
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)

    assert run_seeded([paths], index_root, db_path, "1", "--drop-disproved-paths") == 0
    kept = orjson.loads(paths.read_bytes())["paths"]
    assert [entry["path_id"] for entry in kept] == ["path-0"]
    assert [record["path_id"] for record in dropped_records(paths)] == ["path-1"]
