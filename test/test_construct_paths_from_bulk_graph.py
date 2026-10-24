"""End-to-end runs of construct_paths_from_bulk_graph.py on a cached window graph."""

import sqlite3
import sys
from datetime import date

import numpy as np
import orjson
import pytest

import construct_paths_from_bulk_graph as cli
from src.graph.bulk_window_graph import (
    save_window_graph_cache,
    window_graph_cache_key,
    window_graph_cache_path,
)
from src.graph.long_path_sampler import canonical_path_key

START = date(2026, 6, 1)


def chain_with_branch():
    # Chain 1..12 (each paper cites the previous one), a mutual citation
    # 5 <-> 6, and a branch 6 -> 20 -> 21.
    edges = [(node + 1, node) for node in range(1, 12)]
    edges += [(5, 6), (6, 20), (20, 21)]
    return edges


def paper_record(corpus_id: int, title: str | None = None) -> dict:
    return {
        "paperId": f"{corpus_id:040x}",
        "externalIds": {},
        "title": title or f"Paper {corpus_id}",
        "authors": [{"authorId": str(1000 + corpus_id), "name": f"Author {corpus_id}"}],
        "year": 2026,
        "publicationDate": "2026-07-01",
        "venue": "Venue",
        "openAccessPdf": None,
        "url": "",
    }


class FakeBulkClient:
    def __init__(self, missing=(), titles=None):
        self.missing = set(missing)
        self.titles = titles or {}
        self.requests: list[list[int]] = []

    def collect_indexed_papers_by_corpus_ids(self, corpus_ids):
        self.requests.append(list(corpus_ids))
        return {
            corpus_id: paper_record(corpus_id, self.titles.get(corpus_id))
            for corpus_id in corpus_ids
            if corpus_id not in self.missing
        }


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    index_root = tmp_path / "index"
    index_root.mkdir()
    (index_root / "metadata.json").write_bytes(orjson.dumps({"createdAt": "test-index"}))
    cache_dir = tmp_path / "cache"
    edges = chain_with_branch()
    key = window_graph_cache_key("test-index", "all", START, None)
    save_window_graph_cache(
        window_graph_cache_path(cache_dir, "all", START, None),
        np.array([edge[0] for edge in edges]),
        np.array([edge[1] for edge in edges]),
        {**key, "num_eligible_papers": 40},
    )
    client = FakeBulkClient()
    monkeypatch.setattr(cli.BulkSession, "client", lambda self: client)
    monkeypatch.setattr(cli.BulkSession, "close", lambda self: None)

    def run(*extra):
        argv = [
            "construct_paths_from_bulk_graph.py",
            "--start-date", "2026-06-01",
            "--bulk-index-root", str(index_root),
            "--graph-cache-dir", str(cache_dir),
            "--output-dir", str(tmp_path / "out"),
            "--db-path", str(tmp_path / "graph.db"),
            "--random-seed", "0",
            "--no-tqdm-bar", "1",
            "--max-failed-attempts", "200",
            *extra,
        ]
        monkeypatch.setattr(sys, "argv", argv)
        cli.main()
        return orjson.loads((tmp_path / "out" / "sampled_paths.json").read_bytes())

    return {"run": run, "client": client, "db": tmp_path / "graph.db"}


def neighbors_of(edges):
    adjacency = {}
    for citing, cited in edges:
        adjacency.setdefault(citing, set()).add(cited)
        adjacency.setdefault(cited, set()).add(citing)
    return adjacency


def test_run_writes_valid_paths_and_database_rows(workspace):
    payload = workspace["run"]("--num-hops", "5", "--target-num-paths", "6")
    edges = chain_with_branch()
    adjacency = neighbors_of(edges)
    citing_pairs = set(edges)

    assert payload["metadata"]["n"] == 6
    assert payload["metadata"]["path_length"] == 5
    assert payload["metadata"]["traversal_mode"] == "papers-only"
    assert payload["parameters"]["num_hops"] == 4
    keys = set()
    db = sqlite3.connect(workspace["db"])
    for index, entry in enumerate(payload["paths"]):
        corpus_ids = entry["corpus_ids"]
        assert entry["path_index"] == index
        assert len(entry["path"]) == 5 and len(set(corpus_ids)) == 5
        keys.add(canonical_path_key(corpus_ids))
        for step, corpus_id in zip(entry["path"], corpus_ids):
            assert step["node"]["title"] == f"Paper {corpus_id}"
            assert step["node"]["authors"] == [f"Author {corpus_id}"]
            assert step["node"]["semantic_scholar_id"] == f"{corpus_id:040x}"
        for (a, b), step in zip(zip(corpus_ids, corpus_ids[1:]), entry["path"][1:]):
            assert b in adjacency[a]
            expected = "citing" if (a, b) in citing_pairs else "cited_by"
            assert step["edge_taken"]["edge_type"] == expected
            citing_id, cited_id = (
                (step["edge_taken"]["from"], step["edge_taken"]["to"])
                if expected == "citing"
                else (step["edge_taken"]["to"], step["edge_taken"]["from"])
            )
            row = db.execute(
                "SELECT 1 FROM citations WHERE citing_paper_id = ? AND cited_paper_id = ?",
                (citing_id.split(":")[1], cited_id.split(":")[1]),
            ).fetchone()
            assert row is not None
    assert len(keys) == 6
    # One metadata request for all path papers.
    assert len(workspace["client"].requests) == 1


def test_rerun_extends_saved_paths_and_rejects_other_settings(workspace):
    first = workspace["run"]("--num-hops", "5", "--target-num-paths", "4")
    second = workspace["run"]("--num-hops", "5", "--target-num-paths", "7")

    assert second["paths"][:4] == first["paths"]
    assert len({canonical_path_key(p["corpus_ids"]) for p in second["paths"]}) == 7
    with pytest.raises(SystemExit, match="different settings"):
        workspace["run"]("--num-hops", "6", "--target-num-paths", "7")


def test_max_paths_per_paper_holds_across_reruns(workspace):
    first = workspace["run"](
        "--num-hops", "3", "--target-num-paths", "2", "--max-paths-per-paper", "1"
    )
    second = workspace["run"](
        "--num-hops", "3", "--target-num-paths", "20", "--max-paths-per-paper", "1"
    )

    assert second["paths"][:2] == first["paths"]
    assert second["metadata"]["max_paths_per_paper"] == 1
    used = [corpus_id for entry in second["paths"] for corpus_id in entry["corpus_ids"]]
    assert len(used) == len(set(used))
    # 14 papers hold at most four disjoint 3-paper paths.
    assert 2 < len(second["paths"]) <= 4


def test_papers_without_usable_metadata_are_replaced(workspace):
    client = workspace["client"]
    client.missing = {8}
    client.titles = {10: "<mml:math><mml:mi>x</mml:mi></mml:math>"}
    payload = workspace["run"]("--num-hops", "4", "--target-num-paths", "8")

    # Without papers 8 and 10, exactly seven 4-paper paths remain: four along
    # 1..7 and three through the 6-20-21 branch. Rejected paths are replaced
    # until all of them are found.
    found = {canonical_path_key(entry["corpus_ids"]) for entry in payload["paths"]}
    assert found == {
        (1, 2, 3, 4),
        (2, 3, 4, 5),
        (3, 4, 5, 6),
        (4, 5, 6, 7),
        (4, 5, 6, 20),
        (5, 6, 20, 21),
        (7, 6, 20, 21),
    }
    assert payload["metadata"]["n"] == 7
    assert len(client.requests) >= 2


def test_run_without_any_long_path_exits(workspace):
    with pytest.raises(SystemExit, match="No path of 30 papers"):
        workspace["run"]("--num-hops", "30", "--target-num-paths", "2")


@pytest.fixture(scope="module")
def spark_session():
    pyspark_sql = pytest.importorskip("pyspark.sql")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("PYSPARK_PYTHON", sys.executable)
        spark = (
            pyspark_sql.SparkSession.builder.master("local[1]")
            .appName("bulk-graph-paths-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
        yield spark
        spark.stop()


def test_run_on_a_real_bulk_index(spark_session, tmp_path, monkeypatch):
    import build_semantic_scholar_bulk_api_index as index_builder

    def write_jsonl(path, records):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"\n".join(orjson.dumps(record) for record in records))

    def sha_for(corpus_id):
        return f"a{corpus_id:039x}"

    # Chain 1..12; paper 6 has a MathML title and paper 13 predates the
    # cutoff, so neither may appear, and 13 must not join the two halves.
    titles = {corpus_id: f"Paper {corpus_id}" for corpus_id in range(1, 14)}
    titles[6] = "<mml:math><mml:mi>x</mml:mi></mml:math> bound"
    raw_root = tmp_path / "raw_datasets"
    write_jsonl(
        raw_root / "papers" / "papers.json",
        [
            {
                "corpusid": corpus_id,
                "title": title,
                "url": f"https://www.semanticscholar.org/paper/{sha_for(corpus_id)}",
                "publicationdate": "2025-01-01" if corpus_id == 13 else "2026-07-01",
                "year": 2025 if corpus_id == 13 else 2026,
                "authors": [{"authorId": str(500 + corpus_id), "name": f"Author {corpus_id}"}],
            }
            for corpus_id, title in titles.items()
        ],
    )
    write_jsonl(
        raw_root / "paper-ids" / "paper-ids.json",
        [{"sha": sha_for(corpus_id), "corpusid": corpus_id, "primary": True} for corpus_id in titles],
    )
    edges = [(node + 1, node) for node in range(1, 12)] + [(5, 13), (7, 13)]
    write_jsonl(
        raw_root / "citations" / "citations.json",
        [
            {"citationid": number, "citingcorpusid": citing, "citedcorpusid": cited}
            for number, (citing, cited) in enumerate(edges, start=1)
        ],
    )
    index_root = tmp_path / "index"
    index_builder.build_semantic_scholar_bulk_index(
        raw_root=raw_root,
        index_root=index_root,
        num_buckets=4,
        overwrite=True,
        spark_session=spark_session,
    )
    # The fixture's session is shared with other tests; keep it running.
    monkeypatch.setattr(cli.BulkSession, "close", lambda self: self.bulk_client and self.bulk_client.close())

    def run(target):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "construct_paths_from_bulk_graph.py",
                "--num-hops", "5",
                "--target-num-paths", str(target),
                "--start-date", "2026-06-01",
                "--bulk-data-root", str(raw_root),
                "--bulk-index-root", str(index_root),
                "--output-dir", str(tmp_path / "out"),
                "--db-path", str(tmp_path / "graph.db"),
                "--random-seed", "0",
                "--no-tqdm-bar", "1",
                "--max-failed-attempts", "100",
            ],
        )
        cli.main()
        return orjson.loads((tmp_path / "out" / "sampled_paths.json").read_bytes())

    payload = run(10)

    found = {canonical_path_key(entry["corpus_ids"]) for entry in payload["paths"]}
    assert found == {(1, 2, 3, 4, 5), (7, 8, 9, 10, 11), (8, 9, 10, 11, 12)}
    assert (tmp_path / "window_graphs" / "all.2026-06-01.open.npz").is_file()
    entry = payload["paths"][0]
    assert [step["node"]["title"] for step in entry["path"]] == [
        f"Paper {corpus_id}" for corpus_id in entry["corpus_ids"]
    ]
    assert entry["path"][0]["node"]["semantic_scholar_id"] == sha_for(entry["corpus_ids"][0])
    assert entry["path"][0]["node"]["authors"] == [f"Author {entry['corpus_ids'][0]}"]
