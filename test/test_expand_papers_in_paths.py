from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import orjson
import pytest

from endpoints import expand_papers_in_paths as module
from src.graph.path_expansion_writer import PathExpansionWriter
from src.graph.path_seeding import dropped_paths_file
from src.graph.sampled_paths import collect_path_papers
from src.graph.storage import DatabaseManager
from src.setup.constants import paper_nid


def write_paths(path: Path, db_path: Path, nodes: list[dict]) -> Path:
    path.write_bytes(
        orjson.dumps(
            {
                "metadata": {"db_path": str(db_path)},
                "paths": [{"path": [{"node": node} for node in nodes]}],
            }
        )
    )
    return path


def path_node(paper_id: str, semantic_id: str = "") -> dict:
    return {
        "node_id": f"paper:{paper_id}",
        "node_type": "paper",
        "semantic_scholar_id": semantic_id,
    }


def paper(semantic_id: str) -> dict:
    return {
        "paperId": semantic_id,
        "title": semantic_id.title(),
        "externalIds": {"ArXiv": f"arxiv-{semantic_id}"},
        "abstract": f"Abstract for {semantic_id}",
        "publicationDate": "2000-01-02",
        "venue": "Test venue",
        "openAccessPdf": {"url": f"https://example.org/{semantic_id}.pdf"},
        "authors": [
            {"authorId": "a1", "name": "First Author"},
            {"authorId": "a2", "name": "Second Author"},
        ],
        "referenceCount": 0,
    }


def expansion(semantic_id: str, *, with_neighbors: bool = False) -> tuple:
    metadata = paper(semantic_id)
    metadata["referenceCount"] = int(with_neighbors)
    return (
        metadata,
        [{"citingPaper": paper("citing")}] if with_neighbors else [],
        [{"citedPaper": paper("cited")}] if with_neighbors else [],
    )


@dataclass
class FakeBulkClient:
    responses: dict
    matches: dict = field(default_factory=dict)
    calls: list[list[str]] = field(default_factory=list)
    title_calls: list[list[str]] = field(default_factory=list)
    opened: int = 0
    closed: int = 0
    fail_fetch: bool = False

    def iter_expansion_subgraph_batches(
        self,
        paper_ids,
        *,
        fields,
        max_results,
        edge_budget,
        include_citations=True,
        include_references=True,
        min_publication_date=None,
    ):
        assert edge_budget > 0
        yield self.get_papers_with_citations_and_references_batch(
            paper_ids,
            fields=fields,
            max_results=max_results,
            include_citations=True,
            include_references=True,
        )

    def create(self, **kwargs):
        assert kwargs["use_index"] is True
        assert kwargs["paper_cache_max_bytes"] == module.DEFAULT_PAPER_CACHE_BYTES
        self.opened += 1
        return self

    def search_by_titles_batch(self, titles: list[str], *, fields: list[str]) -> list:
        assert fields == ["paperId"]
        self.title_calls.append(titles)
        return [self.matches.get(title) for title in titles]

    def get_papers_with_citations_and_references_batch(
        self, paper_ids: list[str], **kwargs
    ) -> dict:
        assert kwargs == {
            "fields": module.PAPER_FIELDS,
            "max_results": 100_000,
            "include_citations": True,
            "include_references": True,
        }
        self.calls.append(paper_ids)
        if self.fail_fetch:
            raise RuntimeError("fetch failed")
        fields = {*kwargs["fields"], "referenceCount", "citationCount"}

        def select_metadata(record):
            return (
                {key: value for key, value in record.items() if key in fields}
                if record
                else None
            )

        results = {}
        for paper_id in paper_ids:
            source, citations, references = self.responses.get(paper_id, (None, [], []))
            results[paper_id] = (
                select_metadata(source),
                [
                    {"citingPaper": select_metadata(item.get("citingPaper"))}
                    for item in citations
                ],
                [
                    {"citedPaper": select_metadata(item.get("citedPaper"))}
                    for item in references
                ],
            )
        return results

    def close(self, *, stop_spark: bool = False) -> None:
        assert stop_spark
        self.closed += 1


@pytest.fixture
def graph(tmp_path: Path):
    db_path = tmp_path / "graph.db"
    database = DatabaseManager(db_path)
    yield database
    database.close()


@pytest.fixture
def index_root(tmp_path: Path) -> Path:
    root = tmp_path / "index"
    root.mkdir()
    for table in module.REQUIRED_INDEX_TABLES:
        (root / table).mkdir()
        (root / table / "_SUCCESS").touch()
    (root / "metadata.json").write_bytes(
        orjson.dumps({"tables": module.REQUIRED_INDEX_TABLES})
    )
    return root


def run_expansion(paths: list[Path], index_root: Path, *options: str) -> int:
    return module.main(
        [*(str(path) for path in paths), "--bulk-index-root", str(index_root), *options]
    )


def test_expands_unique_unexplored_papers_across_all_files(
    graph,
    index_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = graph.insert_paper(
        title="Source",
        semantic_scholar_id="source",
        text_path="local.txt",
        depth=2,
        abstract="Existing abstract",
    )
    second = graph.insert_paper(title="Second", semantic_scholar_id="second")
    done = graph.insert_paper(title="Done", semantic_scholar_id="done", explored=True)
    outside = graph.insert_paper(title="Outside", semantic_scholar_id="outside")
    first_file = write_paths(
        tmp_path / "first.json",
        graph.db_path,
        [
            path_node(source, "source"),
            {"node_id": f"author:{uuid.uuid4()}", "node_type": "author"},
            path_node(done, "done"),
            path_node(second, "second"),
        ],
    )
    second_file = write_paths(
        tmp_path / "second.json", graph.db_path, [path_node(second), path_node(source)]
    )
    saved_files = [first_file.read_bytes(), second_file.read_bytes()]
    client = FakeBulkClient(
        {
            "source": expansion("source", with_neighbors=True),
            "second": expansion("second"),
        }
    )
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)

    assert run_expansion([first_file, second_file, first_file], index_root) == 0
    assert client.calls == [["source", "second"]]
    assert client.opened == client.closed == 1
    row = graph.get_paper(source)
    assert row["id"] == source and row["explored"] == 1
    assert row["abstract"] == "Existing abstract"
    assert "abstract" not in module.PAPER_FIELDS
    assert row["published"] == "2000-01-02"  # No date cutoff for path papers.
    assert row["arxiv_id"] == "arxiv-source"
    assert row["venue"] == "Test venue"
    assert row["pdf_url"] == "https://example.org/source.pdf"
    assert row["reference_count"] == 1 and row["valid_references"] == 1
    assert row["text_path"] == "local.txt" and row["depth"] == 2
    for paper_id in (source, second):
        assert graph.get_paper(paper_id)["explored"] == 1
        assert graph.get_expansion_lease("paper", paper_id)["state"] == "complete"
    assert graph.get_paper(outside)["explored"] == 0
    citing = graph.get_paper_by_semantic_scholar_id("citing")
    cited = graph.get_paper_by_semantic_scholar_id("cited")
    assert citing["explored"] == cited["explored"] == 0
    assert cited["abstract"] == ""
    assert {
        (row["citing_paper_id"], row["cited_paper_id"]) for row in graph.all_citations()
    } == {
        (citing["id"], source),
        (source, cited["id"]),
    }
    authorships = graph.authorships_for_paper(source)
    assert sorted(
        (graph.get_author(row["author_id"])["name"], row["position"])
        for row in authorships
    ) == [
        ("First Author", 1),
        ("Second Author", 2),
    ]
    assert [first_file.read_bytes(), second_file.read_bytes()] == saved_files
    # Completed papers do not restart Spark, even if its index is unavailable.
    assert run_expansion([first_file, second_file], tmp_path / "missing-index") == 0
    assert client.opened == 1


@pytest.mark.parametrize("failure", ["fetch", "write", "interrupt"])
def test_failed_expansion_rolls_back_releases_lease_and_can_resume(
    graph,
    index_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    source = graph.insert_paper(title="Source", semantic_scholar_id="source")
    paths = write_paths(tmp_path / "paths.json", graph.db_path, [path_node(source)])
    client = FakeBulkClient({"source": expansion("source", with_neighbors=True)})
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)

    def fail_write(*args) -> None:
        if failure == "interrupt":
            raise KeyboardInterrupt()
        raise RuntimeError("write failed")

    with monkeypatch.context() as failures:
        if failure == "fetch":
            client.fail_fetch = True
        else:
            failures.setattr(PathExpansionWriter, "write_edges", fail_write)
        assert run_expansion([paths], index_root) == (
            130 if failure == "interrupt" else 1
        )

    assert graph.get_paper(source)["explored"] == 0
    assert graph.get_paper(source)["abstract"] == ""
    assert graph.count_papers() == 1
    assert graph.count_authors() == 0
    assert not graph.all_citations()
    assert graph.get_expansion_lease("paper", source)["state"] == "pending"
    assert client.opened == client.closed == 1
    client.fail_fetch = False
    assert run_expansion([paths], index_root) == 0
    assert graph.get_paper(source)["explored"] == 1
    assert client.opened == client.closed == 2


def test_batches_title_resolution_and_rejects_weak_matches(
    graph,
    index_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = [graph.insert_paper(title=title) for title in ("First", "Second", "Weak")]
    paths = write_paths(
        tmp_path / "paths.json",
        graph.db_path,
        [path_node(paper_id) for paper_id in ids],
    )
    client = FakeBulkClient(
        {"first": expansion("first"), "second": expansion("second")},
        {
            "First": {"paperId": "first", "matchScore": 100},
            "Second": {"paperId": "second", "matchScore": 95},
            "Weak": {"paperId": "wrong", "matchScore": 89},
        },
    )
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    assert (
        run_expansion([paths], index_root, "--bulk-title-search-batch-size", "2") == 1
    )
    assert client.title_calls == [["First", "Second"], ["Weak"]]
    assert client.calls == [["first", "second"]]
    assert [graph.get_paper(paper_id)["explored"] for paper_id in ids] == [1, 1, 0]


def test_title_duplicates_preserve_path_uuids_and_share_one_fetch(
    graph,
    index_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = [graph.insert_paper(title="Same Title") for _ in range(2)]
    paths = write_paths(
        tmp_path / "paths.json",
        graph.db_path,
        [path_node(paper_id) for paper_id in ids],
    )
    client = FakeBulkClient(
        {"resolved": expansion("resolved")},
        {"Same Title": {"paperId": "resolved", "matchScore": 100}},
    )
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    assert run_expansion([paths], index_root) == 1
    assert client.calls == [["resolved"]]
    assert graph.count_papers() == 2
    assert [graph.get_paper(paper_id)["explored"] for paper_id in ids] == [1, 0]
    assert graph.get_expansion_lease("paper", ids[1])["state"] == "pending"


def add_paper_authors(graph, paper_id: str, names: list[str]) -> None:
    for name in names:
        graph.insert_authorship(
            paper_id=paper_id, author_id=graph.insert_author(name=name)
        )


@pytest.mark.parametrize("explored", [False, True])
def test_merges_audited_duplicate_and_expands_survivor(
    graph, index_root, tmp_path, monkeypatch, explored
) -> None:
    owner = graph.insert_paper(
        title="Same Title", semantic_scholar_id="resolved", explored=explored
    )
    loser = graph.insert_paper(title="same title", text_path="saved.txt")
    outside = graph.insert_paper(title="Outside")
    for paper_id in (owner, loser):
        add_paper_authors(graph, paper_id, ["First Author"])
    graph.insert_citation(citing_paper_id=outside, cited_paper_id=loser)
    paths = [
        write_paths(tmp_path / f"paths-{i}.json", graph.db_path, [path_node(loser)])
        for i in range(2)
    ]
    payload = orjson.loads(paths[0].read_bytes())
    payload["paths"][0]["start_node"] = paper_nid(loser)
    payload["paths"][0]["path"][0]["edge_taken"] = {
        "from": paper_nid(outside),
        "to": paper_nid(loser),
    }
    paths[0].write_bytes(orjson.dumps(payload))
    client = FakeBulkClient(
        {"resolved": expansion("resolved", with_neighbors=True)},
        {"same title": {"paperId": "resolved", "matchScore": 100}},
    )
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    assert run_expansion(paths, index_root) == 0
    assert client.calls == ([] if explored else [["resolved"]])
    assert graph.get_paper(loser) is None
    assert graph.get_paper(owner)["explored"] == 1
    assert graph.get_paper(owner)["text_path"] == "saved.txt"
    assert graph.conn.execute(
        "SELECT 1 FROM citations WHERE citing_paper_id=? AND cited_paper_id=?",
        (outside, owner),
    ).fetchone()
    for path in paths:
        node = orjson.loads(path.read_bytes())["paths"][0]["path"][0]["node"]
        assert node["node_id"] == paper_nid(owner)
        assert node["semantic_scholar_id"] == "resolved"
    payload = orjson.loads(paths[0].read_bytes())["paths"][0]
    assert payload["start_node"] == paper_nid(owner)
    assert payload["path"][0]["edge_taken"]["to"] == paper_nid(owner)
    assert list(graph.conn.execute("PRAGMA foreign_key_check")) == []
    assert run_expansion(paths, index_root) == 0
    assert client.opened == 1  # Rerun needs no Spark session.


@pytest.mark.parametrize(
    "reason", ["authors", "year", "arxiv", "title", "chain", "oversized"]
)
def test_duplicate_audit_rejects_unsafe_merges(
    graph, index_root, tmp_path, monkeypatch, reason
) -> None:
    owner = graph.insert_paper(
        title="Same Title",
        semantic_scholar_id="resolved",
        published="2000",
        arxiv_id="owner-arxiv",
        explored=True,
    )
    loser = graph.insert_paper(
        title="Different Title" if reason == "title" else "Same Title",
        published="2001" if reason == "year" else "2000",
        arxiv_id="different-arxiv" if reason == "arxiv" else "",
    )
    add_paper_authors(graph, owner, ["First Author", "Second Author"])
    add_paper_authors(
        graph, loser, ["Other Author" if reason == "authors" else "First Author"]
    )
    if reason == "chain":
        third = graph.insert_paper(title="Same Title")
        add_paper_authors(graph, third, ["Second Author"])
    if reason == "oversized":
        for _ in range(49):
            graph.insert_paper(title="Same Title")
    paths = write_paths(tmp_path / "paths.json", graph.db_path, [path_node(loser)])
    before = paths.read_bytes()
    client = FakeBulkClient(
        {},
        {graph.get_paper(loser)["title"]: {"paperId": "resolved", "matchScore": 100}},
    )
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    assert run_expansion([paths], index_root) == 1
    assert graph.get_paper(loser)["explored"] == 0
    assert paths.read_bytes() == before
    assert client.calls == []


def test_merge_remap_recovers_after_path_write_failure(
    graph, index_root, tmp_path, monkeypatch
) -> None:
    from src.graph import path_duplicate_merger

    owner = graph.insert_paper(
        title="Same Title", semantic_scholar_id="resolved", explored=True
    )
    loser = graph.insert_paper(title="Same Title")
    for paper_id in (owner, loser):
        add_paper_authors(graph, paper_id, ["First Author"])
    paths = write_paths(tmp_path / "paths.json", graph.db_path, [path_node(loser)])
    client = FakeBulkClient(
        {}, {"Same Title": {"paperId": "resolved", "matchScore": 100}}
    )
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    original = path_duplicate_merger.write_json_atomic

    def fail_write(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(path_duplicate_merger, "write_json_atomic", fail_write)
    assert run_expansion([paths], index_root) == 2
    assert graph.get_paper(loser) is None
    assert collect_path_papers([paths])[1] == {loser: ""}
    monkeypatch.setattr(path_duplicate_merger, "write_json_atomic", original)
    assert run_expansion([paths], index_root) == 0
    assert collect_path_papers([paths])[1] == {owner: "resolved"}
    assert client.opened == 1


def test_merge_and_remap_roll_back_together_on_interrupt(
    graph, index_root, tmp_path, monkeypatch
) -> None:
    from src.graph import path_duplicate_merger

    owner = graph.insert_paper(
        title="Same Title", semantic_scholar_id="resolved", explored=True
    )
    loser = graph.insert_paper(title="Same Title")
    for paper_id in (owner, loser):
        add_paper_authors(graph, paper_id, ["First Author"])
    paths = write_paths(tmp_path / "paths.json", graph.db_path, [path_node(loser)])
    client = FakeBulkClient(
        {}, {"Same Title": {"paperId": "resolved", "matchScore": 100}}
    )
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    original = path_duplicate_merger.apply_duplicate_paper_groups

    def interrupt_merge(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(
        path_duplicate_merger, "apply_duplicate_paper_groups", interrupt_merge
    )
    assert run_expansion([paths], index_root) == 130
    assert graph.get_paper(loser) is not None
    assert list(graph.conn.execute("SELECT * FROM path_paper_merges")) == []
    assert collect_path_papers([paths])[1] == {loser: ""}


def test_same_batch_duplicates_share_fetch_and_audit(
    graph, index_root, tmp_path, monkeypatch
) -> None:
    from src.graph import path_duplicate_merger

    ids = [graph.insert_paper(title="Same Title") for _ in range(3)]
    for paper_id in ids:
        add_paper_authors(graph, paper_id, ["First Author"])
    paths = write_paths(
        tmp_path / "paths.json", graph.db_path, [path_node(pid) for pid in ids]
    )
    data = expansion("resolved")
    data[0]["title"] = "Same Title"
    client = FakeBulkClient(
        {"resolved": data}, {"Same Title": {"paperId": "resolved", "matchScore": 100}}
    )
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    original = path_duplicate_merger.find_duplicate_paper_groups
    audits = []

    def record_audit(conn):
        audits.append(True)
        return original(conn)

    monkeypatch.setattr(
        path_duplicate_merger, "find_duplicate_paper_groups", record_audit
    )
    assert run_expansion([paths], index_root) == 0
    assert client.calls == [["resolved"]]
    assert len(audits) == 1
    assert graph.count_papers() == 1
    assert collect_path_papers([paths])[1] == {ids[0]: "resolved"}


@pytest.mark.parametrize("busy", [False, True])
def test_merge_invalidates_cached_neighbors_and_respects_active_owner(
    graph, tmp_path, busy
) -> None:
    import time

    from merge_graph_databases import inspect_duplicate_paper_group

    owner = graph.insert_paper(title="Same Title", semantic_scholar_id="resolved")
    loser = graph.insert_paper(title="Same Title")
    for paper_id in (owner, loser):
        add_paper_authors(graph, paper_id, ["First Author"])
    network = module.SQLiteCitationNetwork(graph, owner_id="test-worker")
    writer = PathExpansionWriter(network)
    writer.committed_neighbors[b"fingerprint"] = loser
    paths = write_paths(tmp_path / "paths.json", graph.db_path, [path_node(loser)])
    merger = module.PathDuplicateMerger(writer, [paths])
    merger.prepare()
    if busy:
        graph.claim_expansion(
            "paper",
            owner,
            "another-worker",
            now=time.time(),
            lease_until=time.time() + 60,
        )
    group = inspect_duplicate_paper_group(graph.conn, (owner, loser))
    assert merger.apply(group, owner) is not busy
    if busy:
        assert graph.get_paper(loser) is not None
        assert graph.get_expansion_lease("paper", owner)["owner_id"] == "another-worker"
    else:
        assert not writer.committed_neighbors
        assert merger.remaps == {loser: owner}


def test_retries_sampler_completed_lease_when_paper_is_unexplored(
    graph,
    index_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = graph.insert_paper(title="Source", semantic_scholar_id="source")
    graph.claim_expansion("paper", source, "sampler", now=0, lease_until=1)
    graph.complete_expansion("paper", source, "sampler")
    paths = write_paths(tmp_path / "paths.json", graph.db_path, [path_node(source)])
    client = FakeBulkClient({"source": expansion("source")})
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    assert run_expansion([paths], index_root) == 0
    assert client.calls == [["source"]]
    assert graph.get_paper(source)["explored"] == 1


def test_retry_leases_preserves_active_owners_and_explored_papers(graph) -> None:
    active = graph.insert_paper(title="Active")
    done = graph.insert_paper(title="Done", explored=True)
    graph.claim_expansion("paper", active, "other-worker", now=0, lease_until=999)
    graph.claim_expansion("paper", done, "other-worker", now=0, lease_until=999)
    graph.complete_expansion("paper", done, "other-worker")
    network = module.SQLiteCitationNetwork(graph, owner_id="test")
    module.retry_completed_unexplored_leases(network, [active, done])
    assert graph.get_expansion_lease("paper", active)["owner_id"] == "other-worker"
    assert graph.get_expansion_lease("paper", active)["state"] == "expanding"
    assert graph.get_expansion_lease("paper", done)["state"] == "complete"


def test_missing_or_conflicting_nodes_are_reported_while_valid_papers_expand(
    graph,
    index_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = graph.insert_paper(title="Valid", semantic_scholar_id="valid")
    conflict = graph.insert_paper(title="Conflict", semantic_scholar_id="stored-id")
    paths = write_paths(
        tmp_path / "paths.json",
        graph.db_path,
        [
            path_node(valid),
            path_node(conflict, "different-id"),
            path_node(str(uuid.uuid4())),
        ],
    )
    client = FakeBulkClient({"valid": expansion("valid")})
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    assert run_expansion([paths], index_root) == 1
    assert client.calls == [["valid"]]
    assert graph.get_paper(conflict)["explored"] == 0


def test_requires_complete_indexes_before_starting_spark(
    graph,
    index_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = graph.insert_paper(title="Source", semantic_scholar_id="source")
    paths = write_paths(tmp_path / "paths.json", graph.db_path, [path_node(source)])
    (index_root / "citations_by_citing" / "_SUCCESS").unlink()
    client = FakeBulkClient({})
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    assert run_expansion([paths], index_root) == 2
    assert client.opened == 0
    assert graph.get_paper(source)["explored"] == 0


def test_collect_paths_rejects_mixed_databases_and_supports_override(
    graph,
    tmp_path: Path,
) -> None:
    other = tmp_path / "other.db"
    other.touch()
    paper_id = str(uuid.uuid4())
    first = write_paths(tmp_path / "first.json", graph.db_path, [path_node(paper_id)])
    second = write_paths(tmp_path / "second.json", other, [path_node(paper_id)])
    with pytest.raises(ValueError, match="different databases"):
        collect_path_papers([first, second])
    assert collect_path_papers([first, second], graph.db_path) == (
        graph.db_path.resolve(),
        {paper_id: ""},
        2,
    )


@pytest.mark.parametrize(
    "nodes",
    [
        [{"node_id": "paper:123"}],
        [{"node_id": f"paper:{uuid.uuid4()}", "node_type": "author"}],
        [{"node_id": "unknown:123"}],
    ],
)
def test_collect_paths_rejects_invalid_paper_identity(
    graph, tmp_path: Path, nodes: list[dict]
) -> None:
    paths = write_paths(tmp_path / "paths.json", graph.db_path, nodes)
    with pytest.raises(ValueError):
        collect_path_papers([paths])


def test_collect_paths_rejects_conflicting_ids_across_files(
    graph, tmp_path: Path
) -> None:
    paper_id = str(uuid.uuid4())
    first = write_paths(
        tmp_path / "first.json", graph.db_path, [path_node(paper_id, "one")]
    )
    second = write_paths(
        tmp_path / "second.json", graph.db_path, [path_node(paper_id, "two")]
    )
    with pytest.raises(ValueError, match="conflicting Semantic Scholar IDs"):
        collect_path_papers([first, second])


def test_reuses_client_across_bounded_batches(graph, index_root, tmp_path, monkeypatch):
    semantic_ids = ["first", "second", "third"]
    paper_ids = [
        graph.insert_paper(title=sid, semantic_scholar_id=sid) for sid in semantic_ids
    ]
    paths = write_paths(
        tmp_path / "paths.json", graph.db_path, [path_node(pid) for pid in paper_ids]
    )
    client = FakeBulkClient({sid: expansion(sid) for sid in semantic_ids})
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)

    assert run_expansion([paths], index_root, "--bulk-batch-api-group-size", "2") == 0
    assert client.calls == [["first", "second"], ["third"]]
    assert client.opened == client.closed == 1


@pytest.mark.parametrize(
    ("launcher", "endpoint", "defaults"),
    [
        (
            "expand_path_neighborhood.sh",
            "expand_path_neighborhood.py",
            [
                "--max-level",
                "20",
                "--bulk-batch-api-group-size",
                "10000",
                "--bulk-title-search-batch-size",
                "10000",
                "--bulk-edge-budget",
                "500000",
                "--seed-missing-papers",
                "--cutoff-date",
                "2026-04-01",
            ],
        ),
    ],
)
def test_shell_launcher_collects_sampling_grid_in_one_invocation(
    tmp_path: Path, launcher: str, endpoint: str, defaults: list[str]
):
    repository = Path(__file__).resolve().parents[1]
    scripts = tmp_path / "scripts" / "path_sampling.new" / "2026-04-01"
    scripts.mkdir(parents=True)
    for filename in (launcher, "path_sampling_settings.sh"):
        shutil.copyfile(
            repository / "scripts" / "path_sampling.new" / "2026-04-01" / filename, scripts / filename
        )
    # The real hop grid changes per experiment; pin two hop counts so one file
    # is still left to expand after the missing-file check below.
    with (scripts / "path_sampling_settings.sh").open("a") as settings_file:
        settings_file.write("HOP_COUNTS=(3 4)\n")
    assert (
        "path_sampling_settings.sh"
        in (repository / "scripts/path_sampling.new/2026-04-01/sample_paths_bulk_graph.sh").read_text()
    )
    settings = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\n" "${HOP_COUNTS[@]}"',
            "bash",
            str(scripts / "path_sampling_settings.sh"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data_root = tmp_path / "data with spaces"
    path_files = []
    for hops in settings.stdout.splitlines():
        path = data_root / f"hops={hops}" / "papers-only" / "sampled_paths.json"
        path.parent.mkdir(parents=True)
        path.touch()
        path_files.append(str(path))
    binaries = tmp_path / "bin"
    binaries.mkdir()
    fake_python = binaries / "python"
    fake_python.write_text(
        '#!/bin/bash\nprintf "%s\\n" invocation "$@" >> "$ARGUMENTS_FILE"\n'
    )
    fake_python.chmod(0o755)
    arguments_file = tmp_path / "arguments.txt"
    db_path = str(tmp_path / "graph with spaces.db")
    environment = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "DATA_DIR": str(data_root),
        "SHARED_DB_PATH": db_path,
        "ARGUMENTS_FILE": str(arguments_file),
    }
    command = [
        "bash",
        str(scripts / launcher),
        "--bulk-index-root",
        "custom index",
    ]
    subprocess.run(command, env=environment, check=True, capture_output=True, text=True)
    assert arguments_file.read_text().splitlines() == [
        "invocation",
        "-u",
        f"endpoints/{endpoint}",
        *path_files,
        "--db-path",
        db_path,
        *defaults,
        "--bulk-index-root",
        "custom index",
    ]

    # Missing outputs are reported and skipped; remaining files share one run.
    arguments_file.unlink()
    Path(path_files[-1]).unlink()
    result = subprocess.run(
        command, env=environment, check=False, capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "Skipping missing sampled paths file" in result.stderr
    assert path_files[-1] in result.stderr
    assert arguments_file.read_text().splitlines() == [
        "invocation",
        "-u",
        f"endpoints/{endpoint}",
        *path_files[:-1],
        "--db-path",
        db_path,
        *defaults,
        "--bulk-index-root",
        "custom index",
    ]

    # No available inputs is a successful no-op, without starting Python.
    arguments_file.unlink()
    for path_file in path_files[:-1]:
        Path(path_file).unlink()
    result = subprocess.run(
        command, env=environment, check=False, capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "nothing to expand" in result.stdout
    assert not arguments_file.exists()


def test_late_chunk_failure_keeps_committed_papers_and_releases_remaining_leases(
    graph,
    index_root,
    tmp_path,
    monkeypatch,
):
    ids = [
        graph.insert_paper(title=sid, semantic_scholar_id=sid)
        for sid in ("first", "second")
    ]
    paths = write_paths(
        tmp_path / "paths.json", graph.db_path, [path_node(pid) for pid in ids]
    )
    client = FakeBulkClient({})
    closed = []

    def chunks(*args, **kwargs):
        try:
            yield {"first": expansion("first", with_neighbors=True)}
            raise RuntimeError("second chunk unavailable")
        finally:
            closed.append(True)

    monkeypatch.setattr(client, "iter_expansion_subgraph_batches", chunks)
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)
    assert run_expansion([paths], index_root) == 1
    assert [graph.get_paper(pid)["explored"] for pid in ids] == [1, 0]
    assert graph.get_expansion_lease("paper", ids[0])["state"] == "complete"
    assert graph.get_expansion_lease("paper", ids[1])["state"] == "pending"
    assert closed == [True]


def test_writer_reuses_neighbors_and_invalidates_cache_after_external_changes(graph):
    network = module.SQLiteCitationNetwork(graph, owner_id="test")
    writer = PathExpansionWriter(network)
    sql = []
    graph.conn.set_trace_callback(sql.append)

    def expand(sid):
        pid = graph.insert_paper(title=sid, semantic_scholar_id=sid)
        node = paper_nid(pid)
        network.acquire_expansion(node)
        sql.clear()
        writer.ingest(node, *expansion(sid, with_neighbors=True))
        return [query for query in sql if query.startswith("INSERT INTO authorship")]

    assert len(expand("first")) == 6  # Source + two neighbors, two authors each.
    assert len(expand("second")) == 2  # Only the source's authorship is revisited.
    assert writer.cache_hits == 2
    neighbor = graph.get_paper_by_semantic_scholar_id("cited")
    other = DatabaseManager(graph.db_path)
    try:
        other.update_paper(neighbor["id"], abstract="")
    finally:
        other.close()
    assert len(expand("third")) == 6
    assert graph.get_paper(neighbor["id"])["abstract"] == "Abstract for cited"
    graph.conn.set_trace_callback(None)


def test_writer_does_not_cache_rolled_back_neighbor_ids(graph, monkeypatch):
    network = module.SQLiteCitationNetwork(graph, owner_id="test")
    writer = PathExpansionWriter(network, cache_limit=1)
    pid = graph.insert_paper(title="source", semantic_scholar_id="source")
    node = paper_nid(pid)
    network.acquire_expansion(node)

    def fail(edges):
        raise RuntimeError("write failed")

    with monkeypatch.context() as failures:
        failures.setattr(writer, "write_edges", fail)
        with pytest.raises(RuntimeError, match="write failed"):
            writer.ingest(node, *expansion("source", with_neighbors=True))
    assert not writer.committed_neighbors
    assert graph.count_papers() == 1
    writer.ingest(node, *expansion("source", with_neighbors=True))
    assert graph.count_papers() == 3
    assert graph.get_paper(pid)["explored"] == 1
    assert len(writer.committed_neighbors) == 1


def write_walks(path: Path, db_path: Path, walks: list[list[tuple[dict, str]]]) -> Path:
    """Write sampled walks; each step is (node, edge type from the previous node)."""
    entries = []
    for index, walk in enumerate(walks):
        steps = []
        for step_index, (node, edge_type) in enumerate(walk):
            edge = None
            if step_index:
                edge = {
                    "from": walk[step_index - 1][0]["node_id"],
                    "to": node["node_id"],
                    "edge_type": edge_type,
                }
            steps.append({"step": step_index, "node": node, "edge_taken": edge})
        entries.append(
            {
                "path_id": f"path-{index}",
                "path_index": index,
                "start_node": walk[0][0]["node_id"],
                "moves": len(walk) - 1,
                "path": steps,
            }
        )
    metadata = {"db_path": str(db_path), "n": len(entries), "complete_paths": len(entries)}
    path.write_bytes(orjson.dumps({"metadata": metadata, "paths": entries}))
    return path


def saved_paper(semantic_id: str, paper_id: str | None = None) -> dict:
    return {
        "node_id": f"paper:{paper_id or uuid.uuid4()}",
        "node_type": "paper",
        "semantic_scholar_id": semantic_id,
        "title": semantic_id.title(),
        "published": "2001-01-01",
        "explored": True,
    }


def saved_author(semantic_id: str, name: str) -> dict:
    return {
        "node_id": f"author:{uuid.uuid4()}",
        "node_type": "author",
        "name": name,
        "semantic_scholar_id": semantic_id,
        "orcid": "",
    }


def linked(semantic_id: str, *, cites=(), cited_by=()) -> tuple:
    """Bulk record for one paper, the papers it cites, and the papers citing it."""
    return (
        paper(semantic_id),
        [{"citingPaper": paper(other)} for other in cited_by],
        [{"citedPaper": paper(other)} for other in cites],
    )


def db_id(node: dict) -> str:
    return node["node_id"].partition(":")[2]


def dropped_records(paths: Path) -> list[dict]:
    lines = dropped_paths_file(paths).read_bytes().splitlines()
    return [orjson.loads(line) for line in lines]


def run_seeded(paths: list[Path], index_root: Path, db_path: Path, *options: str) -> int:
    return run_expansion(
        paths, index_root, "--db-path", str(db_path), "--seed-missing-papers", *options
    )


def test_seed_missing_papers_builds_new_database_with_path_uuids(
    index_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "new" / "graph.db"
    first, middle, last = saved_paper("a"), saved_paper("b"), saved_paper("c")
    paths = write_walks(
        tmp_path / "paths.json",
        db_path,
        [[(first, ""), (middle, "citing"), (last, "cited_by")]],
    )
    saved = paths.read_bytes()
    client = FakeBulkClient(
        {
            "a": linked("a", cites=["b"]),
            "b": linked("b", cited_by=["a", "c"]),
            "c": linked("c", cites=["b"]),
        }
    )
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)

    assert run_seeded([paths], index_root, db_path) == 0
    assert paths.read_bytes() == saved
    assert not dropped_paths_file(paths).exists()
    graph = DatabaseManager(db_path)
    try:
        for node in (first, middle, last):
            row = graph.get_paper(db_id(node))
            assert row["explored"] == 1 and row["depth"] == 0
            assert row["semantic_scholar_id"] == node["semantic_scholar_id"]
        citations = {
            (row["citing_paper_id"], row["cited_paper_id"])
            for row in graph.all_citations()
        }
        assert citations == {
            (db_id(first), db_id(middle)),
            (db_id(last), db_id(middle)),
        }
    finally:
        graph.close()
    # A rerun finds nothing to expand and does not restart Spark.
    assert run_seeded([paths], index_root, db_path) == 0
    assert client.opened == 1


def test_missing_database_requires_seed_flag(index_root: Path, tmp_path: Path) -> None:
    db_path = tmp_path / "missing.db"
    paths = write_walks(tmp_path / "paths.json", db_path, [[(saved_paper("a"), "")]])
    assert run_expansion([paths], index_root, "--db-path", str(db_path)) == 2
    # Creating a database needs an explicit --db-path, not the path metadata.
    assert run_expansion([paths], index_root, "--seed-missing-papers") == 2
    assert not db_path.exists()


def test_seed_drops_paths_disproved_by_bulk_data(
    index_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "graph.db"
    source, target, lost, unlinked = (
        saved_paper("a"),
        saved_paper("b"),
        saved_paper("lost"),
        saved_paper("c"),
    )
    paths = write_walks(
        tmp_path / "paths.json",
        db_path,
        [
            [(source, ""), (target, "citing")],
            [(source, ""), (lost, "citing")],
            [(source, ""), (unlinked, "citing")],
        ],
    )
    client = FakeBulkClient(
        {
            "a": linked("a", cites=["b"]),
            "b": linked("b", cited_by=["a"]),
            "c": linked("c"),
        }
    )
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)

    assert run_seeded([paths], index_root, db_path, "--drop-disproved-paths") == 0
    payload = orjson.loads(paths.read_bytes())
    assert [entry["path_id"] for entry in payload["paths"]] == ["path-0"]
    assert payload["metadata"]["n"] == payload["metadata"]["complete_paths"] == 1
    records = dropped_records(paths)
    assert {
        record["path_id"]: [reason["reason"] for reason in record["reasons"]]
        for record in records
    } == {"path-1": ["missing_node"], "path-2": ["missing_edge"]}
    assert records[0]["entry"]["path"][1]["node"]["node_id"] == lost["node_id"]
    assert records[1]["reasons"][0] == {
        "reason": "missing_edge",
        "from": source["node_id"],
        "to": unlinked["node_id"],
        "edge_type": "citing",
    }
    # Dropped papers stay out of later runs; saved drops are not duplicated.
    assert run_seeded([paths], index_root, db_path, "--drop-disproved-paths") == 0
    assert len(dropped_records(paths)) == 2


def test_seed_keeps_unverified_paths_when_expansion_fails(
    index_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "graph.db"
    paths = write_walks(
        tmp_path / "paths.json",
        db_path,
        [[(saved_paper("a"), ""), (saved_paper("b"), "citing")]],
    )
    saved = paths.read_bytes()
    client = FakeBulkClient({"a": linked("a", cites=["b"])}, fail_fetch=True)
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)

    assert run_seeded([paths], index_root, db_path, "--drop-disproved-paths") == 1
    assert paths.read_bytes() == saved
    assert not dropped_paths_file(paths).exists()


def test_seed_remaps_path_uuid_to_existing_row_and_seeds_the_rest(
    graph, index_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = graph.insert_paper(title="A", semantic_scholar_id="a")
    duplicate, new = saved_paper("a"), saved_paper("b")
    paths = write_walks(
        tmp_path / "paths.json", graph.db_path, [[(duplicate, ""), (new, "citing")]]
    )
    client = FakeBulkClient(
        {"a": linked("a", cites=["b"]), "b": linked("b", cited_by=["a"])}
    )
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)

    assert run_seeded([paths], index_root, graph.db_path) == 0
    assert graph.get_paper(db_id(duplicate)) is None
    assert graph.get_paper(db_id(new))["depth"] == 0
    assert graph.get_paper(owner)["explored"] == 1
    steps = orjson.loads(paths.read_bytes())["paths"][0]["path"]
    assert steps[0]["node"]["node_id"] == f"paper:{owner}"
    assert steps[1]["edge_taken"]["from"] == f"paper:{owner}"
    assert steps[1]["node"]["node_id"] == new["node_id"]


def test_seed_reuses_author_node_uuids_for_authorship_edges(
    index_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "graph.db"
    source = saved_paper("a")
    author, stranger = saved_author("a1", "First Author"), saved_author("zz", "Stranger")
    paths = write_walks(
        tmp_path / "paths.json",
        db_path,
        [[(source, ""), (author, "authorship")], [(source, ""), (stranger, "authorship")]],
    )
    client = FakeBulkClient({"a": linked("a")})
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)

    assert run_seeded([paths], index_root, db_path, "--drop-disproved-paths") == 0
    graph = DatabaseManager(db_path)
    try:
        assert graph.get_author(db_id(author))["semantic_scholar_id"] == "a1"
        assert graph.get_authorship(db_id(source), db_id(author)) is not None
    finally:
        graph.close()
    assert [entry["path_id"] for entry in orjson.loads(paths.read_bytes())["paths"]] == [
        "path-0"
    ]
    assert [record["path_id"] for record in dropped_records(paths)] == ["path-1"]


def test_seed_alone_never_removes_paths(
    index_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "graph.db"
    source, lost, unlinked = saved_paper("a"), saved_paper("lost"), saved_paper("c")
    paths = write_walks(
        tmp_path / "paths.json",
        db_path,
        [[(source, ""), (lost, "citing")], [(source, ""), (unlinked, "citing")]],
    )
    saved = paths.read_bytes()
    client = FakeBulkClient({"a": linked("a"), "c": linked("c")})
    monkeypatch.setattr(module, "SemanticScholarBulkClient", client.create)

    # The not-found paper stays unexplored, so the run reports it with status 1.
    assert run_seeded([paths], index_root, db_path) == 1
    assert paths.read_bytes() == saved
    assert not dropped_paths_file(paths).exists()
