"""endpoints/export_examples.py: one line per unique example, growth search skipped."""

from __future__ import annotations

from pathlib import Path

import orjson

from endpoints import export_examples


def pool_record(qid: str, question_type: str, **extra) -> dict:
    return {
        "qid": qid,
        "compatibility": {"generator_version": "test"},
        "config": {"question_format": question_type, "question_depth": 4},
        "question_type": question_type,
        "answer_type": "TYPE",
        "question": f"question {qid}",
        "answer": ["YES"],
        "system_message": "system",
        "user_message": f"user {qid}",
        "token_counts": {"gpt": 10},
        "generator_version": "test",
        **extra,
    }


def write_experiment(folder: Path, qids: list[str], vary: str) -> None:
    folder.mkdir(parents=True)
    (folder / "experiment.jsonl").write_bytes(
        b"".join(orjson.dumps({"qid": qid}) + b"\n" for qid in qids)
    )
    identity = {"pool": "pool.jsonl", "vary": vary, "context_length": 200, "random_seed": 1}
    (folder / "experiment.jsonl.settings.json").write_bytes(
        orjson.dumps({"identity": identity, "slots": len(qids)})
    )


def test_export_examples_deduplicates_and_skips_growth_search(tmp_path, monkeypatch):
    monkeypatch.setattr(export_examples, "REPO_ROOT", tmp_path)
    records = [
        pool_record("a", "reachability-yesno"),
        pool_record(
            "b",
            "reachability-multiple",
            reachability={"options": [{"label": "A", "node_id": "1", "text": "Paper"}]},
        ),
        pool_record("g", "reachability-counting"),
    ]
    (tmp_path / "pool.jsonl").write_bytes(
        b"".join(orjson.dumps(record) + b"\n" for record in records)
    )
    root = tmp_path / "experiments"
    write_experiment(root / "seed=1" / "vary=question-depth", ["a", "b"], "question-depth")
    write_experiment(root / "seed=1" / "vary=node-count", ["b"], "node-count")
    write_experiment(root / "growth_search" / "vary=question-format", ["g"], "question-format")

    output = tmp_path / "examples.jsonl"
    export_examples.export_examples(root, output)
    lines = [orjson.loads(line) for line in output.read_bytes().splitlines()]

    assert [line["qid"] for line in lines] == ["b", "a"]
    by_qid = {line["qid"]: line for line in lines}
    assert by_qid["a"]["reachability_task"] == "yes-no"
    assert by_qid["a"]["system_prompt"] == "system"
    assert by_qid["a"]["user_prompt"] == "user a"
    assert by_qid["a"]["answer"] == "YES"
    assert "options" not in by_qid["a"]
    assert by_qid["b"]["reachability_task"] == "multiple-choice"
    assert by_qid["b"]["options"] == [{"label": "A", "text": "Paper"}]
    assert [e["name"] for e in by_qid["b"]["experiments"]] == ["node-count", "question-depth"]
    assert by_qid["b"]["experiments"][1]["slot"] == 1
