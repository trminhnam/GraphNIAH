"""Shared synthetic fixtures for dataset v1 tests."""

import sys
from types import SimpleNamespace

import pytest

from test.test_build_full_prompts import FakeTokenizer
from test.test_reachability_v5_pipeline import FakePaperDB, make_long_entry


class PaperDB(FakePaperDB):
    def get_paper_published(self, db_id):
        return "2025-10-01"

    def get_paper_explored(self, db_id):
        return 1


def depth_entry(prefix, depth=3):
    """A fixture path cut to the 3k - 2 papers of question depth k."""
    entry = make_long_entry(prefix)
    entry["path"] = entry["path"][: 3 * depth - 2]
    return entry


def check_components_only_needs_no_distractors(module, monkeypatch, tmp_path):
    """With no distractor folders, a full run stops at startup; a components-only
    run starts, and its pool key leaves the distractor locations out."""
    checker = SimpleNamespace(
        context=SimpleNamespace(db=SimpleNamespace(close=lambda: None)),
        graph_path=tmp_path / "graph.json",
        db_path=tmp_path / "graph.db",
        facts=None,
        cutoff=None,
    )
    monkeypatch.setattr(module, "load_paths", lambda paths: {})
    monkeypatch.setattr(module, "open_graph_checker", lambda *args: checker)
    monkeypatch.setattr(module, "open_pools", lambda *args: {})
    monkeypatch.setattr(module, "bootstrap_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module.packing,
        "load_prompt_packing_tokenizers",
        lambda context_length, only_slugs=None: [("fake", FakeTokenizer())],
    )
    keys = []

    def run_experiment(pool, output, configs, settings, compatibility, *args, **kwargs):
        keys.append(compatibility)
        return {"retries": {}, "created": 0, "reused": 0, "resumed": 0}

    monkeypatch.setattr(module, "run_experiment", run_experiment)
    argv = [
        "create_dataset",
        "--paths",
        str(tmp_path / "paths.json"),
        "--vary",
        "question-format",
        "--examples-per-value",
        "1",
        "--pool",
        str(tmp_path / "pool.jsonl"),
        "--output",
        str(tmp_path / "experiment.jsonl"),
        "--graph-db",
        str(tmp_path / "graph.db"),
    ]
    for kind in module.FILE_DISTRACTOR_TYPES:
        argv += [f"--{kind}-dir", str(tmp_path / kind)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="No nonempty code files"):
        module.main()
    monkeypatch.setattr(sys, "argv", [*argv, "--components-only"])
    module.main()
    assert len(keys) == 1
    assert "distractor_sources" not in keys[0]
    assert keys[0]["tokenizers"] == ["fake"]


def check_worker_needs_the_main_tokenizers(module, monkeypatch):
    """A worker loads only the main process's tokenizers (they are in the pool
    key); if it gets others it stops the run, and otherwise opens its inputs once."""
    opened, closed = [], []

    def open_construction(args, tokenizer_names=None):
        opened.append((args, tokenizer_names))
        return SimpleNamespace(
            tokenizers=[("fake", FakeTokenizer())],
            close=lambda: closed.append(True),
            construct=lambda config, seed: {"seed": seed},
        )

    monkeypatch.setattr(module, "open_construction", open_construction)
    monkeypatch.setattr(module, "bootstrap_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "WORKER_CONSTRUCTION", {})
    with pytest.raises(
        RuntimeError, match=r"loaded the tokenizers \['fake'\], but the run uses"
    ):
        module.construct_in_worker("args", ["fake", "other"], {}, 1)
    assert closed == [True] and module.WORKER_CONSTRUCTION == {}
    assert module.construct_in_worker("args", ["fake"], {}, 2) == {"seed": 2}
    assert module.construct_in_worker("args", ["fake"], {}, 3) == {"seed": 3}
    assert opened == [("args", ["fake", "other"]), ("args", ["fake"])]
