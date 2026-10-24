"""run_inference.py: export from pool + experiment, request once, resume."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from pathlib import Path

import litellm
import orjson
import pytest

import inference
import run_inference
from src.questions import dataset_pool
from test.test_inference import build_response

SETTINGS = {"vary": "question-depth", "random_seed": 1, "context_length": 4000}


def make_config(index: int) -> dict:
    return {"question_format": "reachability-yesno", "question_depth": 2 + index}


def construct(config, seed):
    return {
        "config": config,
        "question_type": "reachability-yesno",
        "answer_type": "yes_no",
        "answer": ["YES"],
        "system_message": f"System for depth {config['question_depth']}",
        "user_message": f"Question {seed}",
    }


def build_experiment(
    tmp_path: Path,
    count: int,
    *,
    vary: str = "question-depth",
    num_values: int | None = None,
) -> tuple[Path, Path]:
    pool, experiment = tmp_path / "pool.jsonl", tmp_path / "experiment.jsonl"
    dataset_pool.run_experiment(
        pool,
        experiment,
        [
            make_config(index if num_values is None else index % num_values)
            for index in range(count)
        ],
        {**SETTINGS, "vary": vary},
        {"generator_version": "test"},
        construct,
        seed=1,
        max_attempts=2,
    )
    return pool, experiment


def build_args(pool: Path, experiment: Path, output: Path, **overrides) -> argparse.Namespace:
    args = run_inference.build_parser().parse_args(
        ["--pool", str(pool), "--experiment", str(experiment), "--output", str(output)]
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.fixture
def sent(monkeypatch):
    """Record every request; a user message containing FAIL is rejected once."""
    requests = []

    async def fake_call_litellm_async(**kwargs):
        requests.append(kwargs["messages"])
        if "FAIL" in kwargs["messages"][-1]["content"]:
            raise litellm.AuthenticationError(
                message="bad key", llm_provider="openai", model="test"
            )
        return build_response("\\answer{YES}")

    monkeypatch.setattr(inference, "call_litellm_async", fake_call_litellm_async)
    monkeypatch.setattr(litellm, "cost_per_token", lambda **_kwargs: (0.01, 0.02))
    return requests


def saved(output: Path) -> dict:
    return orjson.loads(output.read_bytes())


@pytest.mark.parametrize(
    "vary,key,values",
    [
        ("question-depth", "question_depth", [4, 8, 16]),
        ("node-count", "component_node_count", [100, 150, 200]),
        ("question-format", "question_format", ["yesno", "multiple", "counting"]),
        ("distractor-type", "distractor_type", ["code", "books", "essays"]),
    ],
)
def test_max_samples_limits_each_value_even_with_uneven_order(vary, key, values):
    examples = [
        {"qid": str(index), "config": {key: values[value], "other_control": index}}
        for index, value in enumerate([0, 1, 0, 0, 1, 2])
    ]
    selected = run_inference.select_examples(examples, vary, 2)
    assert [example["qid"] for example in selected] == ["0", "1", "2", "4", "5"]
    assert run_inference.select_examples(examples, vary, None) == examples
    assert run_inference.select_examples(examples, vary, 10) == examples


@pytest.mark.parametrize("limit,expected_per_depth", [(None, 120), (100, 100)])
def test_question_depth_filter_preserves_order_and_limits_each_depth(
    limit, expected_per_depth
):
    examples = [
        {"qid": str(i), "config": {"question_depth": [4, 16, 8, 32][i % 4]}}
        for i in range(480)
    ]
    selected = run_inference.select_examples(
        examples, "question-depth", limit, [16, 32]
    )
    assert Counter(item["config"]["question_depth"] for item in selected) == {
        16: expected_per_depth,
        32: expected_per_depth,
    }
    assert [int(item["qid"]) for item in selected] == list(
        range(1, expected_per_depth * 4, 2)
    )


def test_depth_selection_resumes_reuses_pool_and_protects_saved_answers(
    tmp_path, sent, monkeypatch
):
    original_config = make_config

    def depth_config(index):
        return {**original_config(index), "question_depth": [4, 8, 16, 32][index % 4]}

    monkeypatch.setattr(__name__ + ".make_config", depth_config)
    pool, experiment = build_experiment(tmp_path, 12)
    output = tmp_path / "filtered.json"
    args = build_args(
        pool, experiment, output, question_depths=[16, 32], max_samples=100
    )
    asyncio.run(run_inference.run(args))
    assert len(sent) == 6
    assert {request[0]["content"] for request in sent} == {
        "System for depth 16",
        "System for depth 32",
    }
    payload = saved(output)
    assert payload["metadata"]["question_depths"] == [16, 32]
    assert "question_depths" not in payload["metadata"]["generation_settings"]

    sent.clear()
    asyncio.run(run_inference.run(args))
    assert sent == []
    assert saved(output)["metadata"]["resumed_examples"] == 6

    cached_output = tmp_path / "cached.json"
    asyncio.run(
        run_inference.run(
            build_args(pool, experiment, cached_output, question_depths=[32])
        )
    )
    assert sent == []
    assert saved(cached_output)["metadata"]["cache_hits"] == 3
    assert {
        item["config"]["question_depth"] for item in saved(cached_output)["records"]
    } == {32}

    before = saved(output)["records"]
    asyncio.run(
        run_inference.run(build_args(pool, experiment, output, question_depths=[32]))
    )
    assert saved(output)["records"] == before
    assert saved(output)["metadata"]["selected_examples"] == 3
    assert saved(output)["metadata"]["preserved_examples"] == 3
    assert saved(output)["metadata"]["total_examples"] == 6
    assert saved(output)["metadata"]["selected_answered_examples"] == 3
    assert sent == []

    asyncio.run(run_inference.run(build_args(pool, experiment, output)))
    assert len(sent) == 6
    assert len(saved(output)["records"]) == 12
    assert saved(output)["metadata"]["resumed_examples"] == 6


@pytest.mark.parametrize("limit,expected_per_size", [(None, 120), (100, 100)])
def test_component_size_filter_preserves_order_and_limits_each_size(limit, expected_per_size):
    examples = [
        {
            "qid": str(i),
            "config": {
                "component_node_count": [100, 200, 150, 250][i % 4],
                "question_depth": 16 if i % 4 == 1 else 32,
            },
        }
        for i in range(480)
    ]
    selected = run_inference.select_examples(
        examples, "node-count", limit, component_node_counts=[200, 250]
    )
    assert Counter(item["config"]["component_node_count"] for item in selected) == {
        200: expected_per_size,
        250: expected_per_size,
    }
    assert [int(item["qid"]) for item in selected] == list(
        range(1, expected_per_size * 4, 2)
    )
    combined = run_inference.select_examples(
        examples,
        "node-count",
        limit,
        question_depths=[16],
        component_node_counts=[200, 250],
    )
    assert [int(item["qid"]) for item in combined] == list(
        range(1, expected_per_size * 4, 4)
    )


def test_component_size_selection_reuses_pool_and_protects_saved_answers(
    tmp_path, sent, monkeypatch
):
    original_config = make_config

    def size_config(index: int) -> dict:
        return {
            **original_config(index),
            "component_node_count": [100, 150, 200, 250][index % 4],
        }

    monkeypatch.setattr(__name__ + ".make_config", size_config)
    pool, experiment = build_experiment(tmp_path, 12, vary="node-count")
    output = tmp_path / "filtered.json"
    args = build_args(
        pool, experiment, output, component_node_counts=[200, 250], max_samples=100
    )
    asyncio.run(run_inference.run(args))
    assert len(sent) == 6
    payload = saved(output)
    assert Counter(
        item["config"]["component_node_count"] for item in payload["records"]
    ) == {200: 3, 250: 3}
    assert payload["metadata"]["component_node_counts"] == [200, 250]
    assert "component_node_counts" not in payload["metadata"]["generation_settings"]
    sent.clear()
    asyncio.run(run_inference.run(args))
    assert sent == []
    assert saved(output)["metadata"]["resumed_examples"] == 6
    cached_output = tmp_path / "cached.json"
    asyncio.run(
        run_inference.run(
            build_args(pool, experiment, cached_output, component_node_counts=[250])
        )
    )
    assert sent == []
    assert saved(cached_output)["metadata"]["cache_hits"] == 3
    before = saved(output)["records"]
    asyncio.run(
        run_inference.run(
            build_args(pool, experiment, output, component_node_counts=[250])
        )
    )
    assert saved(output)["records"] == before
    assert saved(output)["metadata"]["selected_examples"] == 3
    assert saved(output)["metadata"]["preserved_examples"] == 3
    assert sent == []


@pytest.mark.parametrize("vary", ["question-depth", "none"])
def test_larger_max_samples_reuses_completed_examples_per_value(tmp_path, sent, vary):
    pool, experiment = build_experiment(tmp_path, 500, vary=vary, num_values=2)
    output = tmp_path / "predictions.json"
    groups = 1 if vary == "none" else 2
    asyncio.run(run_inference.run(build_args(pool, experiment, output, max_samples=10)))
    first_qids = [record["qid"] for record in saved(output)["records"]]
    assert len(sent) == 10 * groups

    sent.clear()
    asyncio.run(
        run_inference.run(build_args(pool, experiment, output, max_samples=200))
    )
    payload = saved(output)
    assert len(sent) == 190 * groups
    assert payload["metadata"]["resumed_examples"] == 10 * groups
    assert payload["metadata"]["new_requests"] == 190 * groups
    assert payload["metadata"]["answered_examples"] == 200 * groups
    assert [
        record["qid"] for record in payload["records"][: len(first_qids)]
    ] == first_qids
    if vary != "none":
        assert Counter(
            record["config"]["question_depth"] for record in payload["records"]
        ) == {2: 200, 3: 200}


@pytest.mark.parametrize(
    "vary,filter_name,values",
    [
        ("question-depth", "question_depths", [16, 32]),
        ("node-count", "component_node_counts", [200, 250]),
    ],
)
def test_filtered_run_adds_only_selected_answers_and_preserves_others(
    tmp_path, sent, monkeypatch, vary, filter_name, values
):
    def config(index: int) -> dict:
        return {
            "question_format": "reachability-yesno",
            "question_depth": [4, 8, 16, 32][index % 4],
            "component_node_count": [100, 150, 200, 250][index % 4],
        }

    monkeypatch.setattr(__name__ + ".make_config", config)
    pool, experiment = build_experiment(tmp_path, 12, vary=vary)
    output = tmp_path / "predictions.json"
    asyncio.run(run_inference.run(build_args(pool, experiment, output, max_samples=1)))
    payload = saved(output)
    # Even a failed record outside the selection must be preserved without retrying.
    payload["records"][0].update(raw_answer=None, error="server error")
    payload["records"][0]["cost"]["total_cost"] = 1.0
    output.write_bytes(orjson.dumps(payload))
    preserved = payload["records"][:2]
    sent.clear()

    asyncio.run(
        run_inference.run(
            build_args(pool, experiment, output, max_samples=2, **{filter_name: values})
        )
    )
    payload = saved(output)
    assert len(sent) == 2
    assert {request[0]["content"] for request in sent} == {
        "System for depth 16",
        "System for depth 32",
    }
    assert payload["records"][:2] == preserved
    examples, _, _ = run_inference.load_experiment(pool, experiment)
    assert [record["qid"] for record in payload["records"]] == [
        examples[index]["qid"] for index in [0, 1, 2, 3, 6, 7]
    ]
    metadata = payload["metadata"]
    assert metadata["selected_examples"] == 4
    assert metadata["preserved_examples"] == 2
    assert metadata["total_examples"] == 6
    assert metadata["selected_answered_examples"] == 4
    assert metadata["answered_examples"] == 5
    assert metadata["failed_examples"] == 1
    assert metadata["new_requests"] == 2
    assert metadata["run_accounting"]["records_with_cost"] == 2
    assert metadata["run_accounting"]["total_total_cost"] == pytest.approx(
        sum(record["cost"]["total_cost"] for record in payload["records"][-2:])
    )
    assert metadata["records_with_cost"] == 6
    assert metadata["total_total_cost"] == pytest.approx(
        sum(record["cost"]["total_cost"] for record in payload["records"])
    )

    sent.clear()
    asyncio.run(run_inference.run(build_args(pool, experiment, output, max_samples=3)))
    payload = saved(output)
    assert len(sent) == 6
    assert payload["metadata"]["resumed_examples"] == 5
    assert payload["metadata"]["cache_hits"] == 1
    assert payload["metadata"]["answered_examples"] == 12
    assert payload["metadata"]["preserved_examples"] == 0
    assert [record["qid"] for record in payload["records"]] == [
        example["qid"] for example in examples
    ]


def test_reasoning_efforts_share_examples_and_resume_separately(tmp_path, sent):
    pool, experiment = build_experiment(tmp_path, 12, vary="none", num_values=3)
    for limit, new_per_effort in ((3, 3), (5, 2)):
        expected_qids = None
        for effort in ("low", "medium", "high"):
            output = tmp_path / f"{effort}.json"
            sent.clear()
            asyncio.run(
                run_inference.run(
                    build_args(
                        pool,
                        experiment,
                        output,
                        max_samples=limit,
                        reasoning_effort=effort,
                    )
                )
            )
            assert len(sent) == new_per_effort
            payload = saved(output)
            assert payload["metadata"]["max_samples_scope"] == "total"
            qids = [record["qid"] for record in payload["records"]]
            assert len(qids) == limit
            if expected_qids is None:
                expected_qids = qids
            assert qids == expected_qids


def test_sends_pool_messages_in_experiment_order_with_gold_fields(tmp_path, sent):
    pool, experiment = build_experiment(tmp_path, 3)
    output = tmp_path / "predictions.json"
    asyncio.run(run_inference.run(build_args(pool, experiment, output)))

    qids = [line["qid"] for line in map(orjson.loads, experiment.read_bytes().splitlines())]
    with pool.open("rb") as handle:
        examples = [record for _offset, record in dataset_pool.read_jsonl(handle, repair=False)]
    by_qid = {example["qid"]: example for example in examples}
    assert sent == [
        [
            {"role": "system", "content": by_qid[qid]["system_message"]},
            {"role": "user", "content": by_qid[qid]["user_message"]},
        ]
        for qid in qids
    ]
    payload = saved(output)
    assert [record["qid"] for record in payload["records"]] == qids
    first = payload["records"][0]
    assert first["raw_answer"] == "\\answer{YES}"
    assert first["answer"] == ["YES"]
    assert first["config"] == make_config(0)
    assert "turn_index" not in first
    assert payload["metadata"]["answered_examples"] == 3


def test_resume_sends_only_failed_and_new_examples(tmp_path, sent, monkeypatch, capsys):
    pool, experiment = build_experiment(tmp_path, 2)
    output = tmp_path / "predictions.json"
    with pool.open("rb") as handle:
        examples = [
            record for _offset, record in dataset_pool.read_jsonl(handle, repair=False)
        ]
    failing = examples[1]["user_message"]
    original_call = inference.call_litellm_async
    calls = {"failed": False}

    async def fail_second_once(**kwargs):
        if kwargs["messages"][-1]["content"] == failing and not calls["failed"]:
            calls["failed"] = True
            kwargs["messages"][-1] = {"role": "user", "content": "FAIL"}
        return await original_call(**kwargs)

    monkeypatch.setattr(inference, "call_litellm_async", fail_second_once)
    asyncio.run(run_inference.run(build_args(pool, experiment, output)))
    captured = capsys.readouterr()
    assert "FAILED after 1 attempts" in captured.err
    assert "Requests" in captured.err and "2/2" in captured.err
    payload = saved(output)
    assert payload["metadata"]["failed_examples"] == 1
    assert payload["records"][1]["error"]
    cache_path = tmp_path / "inference_pool.jsonl"
    assert len(cache_path.read_bytes().splitlines()) == 1

    sent.clear()
    asyncio.run(run_inference.run(build_args(pool, experiment, output)))
    assert [messages[-1]["content"] for messages in sent] == [failing]
    retried = saved(output)["records"][1]
    assert retried["raw_answer"] and retried["error"] is None
    assert retried["attempts"] == 2
    assert len(retried["failed_attempts"]) == 1
    assert len(cache_path.read_bytes().splitlines()) == 2
    assert (
        saved(output)["metadata"]["run_accounting"]["retry_accounting"][
            "total_attempts"
        ]
        == 0
    )

    sent.clear()
    build_experiment(tmp_path, 3)  # The experiment grows by one slot.
    asyncio.run(run_inference.run(build_args(pool, experiment, output)))
    assert len(sent) == 1
    assert saved(output)["metadata"]["answered_examples"] == 3


def test_changed_generation_settings_stop(tmp_path, sent):
    pool, experiment = build_experiment(tmp_path, 1)
    output = tmp_path / "predictions.json"
    asyncio.run(run_inference.run(build_args(pool, experiment, output)))
    with pytest.raises(ValueError, match="temperature changed"):
        asyncio.run(
            run_inference.run(build_args(pool, experiment, output, temperature=0.0))
        )


@pytest.mark.parametrize("vary", ["question-depth", "none"])
def test_smaller_max_samples_preserves_answers_but_drops_stale_qids(
    tmp_path, sent, vary
):
    pool, experiment = build_experiment(tmp_path, 3, vary=vary, num_values=1)
    output = tmp_path / "predictions.json"
    asyncio.run(run_inference.run(build_args(pool, experiment, output)))
    sent.clear()
    asyncio.run(run_inference.run(build_args(pool, experiment, output, max_samples=2)))
    assert sent == []
    payload = saved(output)
    assert len(payload["records"]) == 3
    assert payload["metadata"]["selected_examples"] == 2
    assert payload["metadata"]["selected_answered_examples"] == 2
    assert payload["metadata"]["preserved_examples"] == 1
    assert payload["metadata"]["answered_examples"] == 3
    payload["records"].append({**payload["records"][0], "qid": "left-the-experiment"})
    output.write_bytes(orjson.dumps(payload))
    sent.clear()
    asyncio.run(run_inference.run(build_args(pool, experiment, output)))
    assert sent == []
    assert "left-the-experiment" not in {
        record["qid"] for record in saved(output)["records"]
    }


def test_components_only_experiment_is_rejected(tmp_path, sent):
    pool, experiment = tmp_path / "pool.jsonl", tmp_path / "experiment.jsonl"
    dataset_pool.run_experiment(
        pool,
        experiment,
        [{**make_config(0), "components_only": True}],
        SETTINGS,
        {"generator_version": "test"},
        lambda config, seed: {**construct(config, seed), "content_key": [seed]},
        seed=1,
        max_attempts=2,
    )
    with pytest.raises(ValueError, match="components-only"):
        asyncio.run(
            run_inference.run(build_args(pool, experiment, tmp_path / "out.json"))
        )
    assert sent == []


def test_experiment_from_another_pool_is_rejected(tmp_path, sent):
    _pool, experiment = build_experiment(tmp_path, 1)
    other = tmp_path / "other-pool.jsonl"
    other.write_bytes(b"")
    with pytest.raises(ValueError, match="was built with pool"):
        asyncio.run(
            run_inference.run(build_args(other, experiment, tmp_path / "out.json"))
        )


def test_pool_reuses_overlapping_experiments_in_order_and_counts_only_new_cost(
    tmp_path, sent
):
    pool, first = build_experiment(tmp_path, 3)
    first_output = tmp_path / "first.json"
    asyncio.run(run_inference.run(build_args(pool, first, first_output)))
    second = tmp_path / "second.jsonl"
    dataset_pool.run_experiment(
        pool,
        second,
        [make_config(i) for i in (2, 0, 3)],
        {**SETTINGS, "vary": "none"},
        {"generator_version": "test"},
        construct,
        seed=2,
        max_attempts=2,
    )
    sent.clear()
    second_output = tmp_path / "second.json"
    asyncio.run(
        run_inference.run(build_args(pool, second, second_output, max_workers=3))
    )
    assert len(sent) == 1
    payload = saved(second_output)
    expected = [orjson.loads(line)["qid"] for line in second.read_bytes().splitlines()]
    assert [record["qid"] for record in payload["records"]] == expected
    assert [record["inference_source"] for record in payload["records"]] == [
        "pool",
        "pool",
        "request",
    ]
    assert [record["config"] for record in payload["records"]] == [
        make_config(i) for i in (2, 0, 3)
    ]
    metadata = payload["metadata"]
    assert metadata["cache_hits"] == 2 and metadata["new_requests"] == 1
    assert metadata["resumed_examples"] == 0
    assert metadata["records_with_cost"] == 3
    assert metadata["run_accounting"]["records_with_cost"] == 1
    assert (
        metadata["run_accounting"]["total_total_cost"]
        == payload["records"][-1]["cost"]["total_cost"]
    )


@pytest.mark.parametrize(
    "changed",
    [
        {"model": "openai/another-model"},
        {"reasoning_effort": "medium"},
        {"reasoning_effort": "high"},
        {"temperature": 0.1},
        {"top_p": 0.5},
        {"top_k": 10},
        {"frequency_penalty": 0.5},
        {"presence_penalty": 0.5},
        {"stop": ["END"]},
    ],
)
def test_changed_request_parameters_do_not_reuse_cached_answers(
    tmp_path, sent, changed
):
    pool, experiment = build_experiment(tmp_path, 1)
    asyncio.run(
        run_inference.run(build_args(pool, experiment, tmp_path / "first.json"))
    )
    output = tmp_path / "changed.json"
    asyncio.run(run_inference.run(build_args(pool, experiment, output, **changed)))
    assert len(sent) == 2
    assert saved(output)["metadata"]["cache_hits"] == 0
    assert len((tmp_path / "inference_pool.jsonl").read_bytes().splitlines()) == 2


def test_keys_are_explicit_and_aliases_defaults_and_credentials_do_not_change_reuse(
    tmp_path, sent, monkeypatch
):
    pool, experiment = build_experiment(tmp_path, 1)
    cache = tmp_path / "custom" / "responses.jsonl"
    args = build_args(
        pool,
        experiment,
        tmp_path / "first.json",
        api_key="test-secret",
        inference_pool=cache,
    )
    asyncio.run(run_inference.run(args))
    original_cache = cache.read_bytes()
    [entry] = [orjson.loads(line) for line in cache.read_bytes().splitlines()]
    assert entry["key"] == {
        "qid": saved(args.output)["records"][0]["qid"],
        "model": args.model,
        "reasoning_effort": "low",
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": None,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": None,
    }
    assert b"test-secret" not in cache.read_bytes()
    assert "config" not in entry["response"]
    monkeypatch.setitem(
        inference.MODEL_SLUG_TO_API_MODEL_NAME, "test-alias", args.model
    )
    output = tmp_path / "alias.json"
    args = build_args(
        pool,
        experiment,
        output,
        model="test-alias",
        api_key="other-secret",
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=[],
        max_workers=2,
        inference_pool=cache,
    )
    sent.clear()
    asyncio.run(run_inference.run(args))
    assert sent == []
    assert saved(output)["metadata"]["cache_hits"] == 1
    assert saved(output)["metadata"]["new_requests"] == 0
    assert saved(output)["metadata"]["run_accounting"]["total_total_cost"] == 0
    assert cache.read_bytes() == original_cache


def test_existing_output_is_imported_then_reused_without_new_requests(tmp_path, sent):
    pool, experiment = build_experiment(tmp_path, 1)
    output = tmp_path / "legacy.json"
    asyncio.run(run_inference.run(build_args(pool, experiment, output)))
    cache = tmp_path / "inference_pool.jsonl"
    cache.unlink()
    payload = saved(output)
    payload["records"][0].pop("inference_source")
    for optional in ("top_k", "stop"):
        payload["metadata"]["generation_settings"].pop(optional)
    output.write_bytes(orjson.dumps(payload))
    sent.clear()
    asyncio.run(run_inference.run(build_args(pool, experiment, output)))
    assert saved(output)["metadata"]["resumed_examples"] == 1
    assert saved(output)["metadata"]["cache_hits"] == 0
    assert len(cache.read_bytes().splitlines()) == 1
    imported = tmp_path / "imported.json"
    asyncio.run(run_inference.run(build_args(pool, experiment, imported)))
    assert sent == []
    assert saved(imported)["metadata"]["cache_hits"] == 1


def test_response_survives_interrupted_output_write(tmp_path, sent, monkeypatch):
    pool, experiment = build_experiment(tmp_path, 1)
    output = tmp_path / "interrupted.json"
    original = run_inference.write_output

    def fail_after_answer(path, payload):
        if payload["records"]:
            raise OSError("interrupted output write")
        original(path, payload)

    with monkeypatch.context() as patch:
        patch.setattr(run_inference, "write_output", fail_after_answer)
        with pytest.raises(OSError, match="interrupted output write"):
            asyncio.run(run_inference.run(build_args(pool, experiment, output)))
    assert len(sent) == 1
    assert saved(output)["records"] == []
    asyncio.run(run_inference.run(build_args(pool, experiment, output)))
    assert len(sent) == 1
    assert saved(output)["metadata"]["cache_hits"] == 1


@pytest.mark.parametrize("destination", ["pool", "experiment", "settings", "output"])
def test_inference_pool_cannot_overwrite_inputs_or_output(tmp_path, sent, destination):
    pool, experiment = build_experiment(tmp_path, 1)
    output = tmp_path / "predictions.json"
    paths = {
        "pool": pool,
        "experiment": experiment,
        "settings": dataset_pool.settings_path_for(experiment),
        "output": output,
    }
    original = {path: path.read_bytes() for path in paths.values() if path.exists()}
    with pytest.raises(ValueError, match="must differ"):
        asyncio.run(
            run_inference.run(
                build_args(pool, experiment, output, inference_pool=paths[destination])
            )
        )
    assert all(path.read_bytes() == data for path, data in original.items())
    assert sent == []


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_cli_accepts_reasoning_experiment_efforts(effort):
    args = run_inference.build_parser().parse_args(
        [
            "--pool",
            "pool.jsonl",
            "--experiment",
            "experiment.jsonl",
            "--output",
            "predictions.json",
            "--reasoning-effort",
            effort,
            "--inference-pool",
            "responses.jsonl",
            "--question-depths",
            "16",
            "32",
            "--component-node-counts",
            "200",
            "250",
        ]
    )
    assert args.reasoning_effort == effort
    assert args.inference_pool == Path("responses.jsonl")
    assert args.question_depths == [16, 32]
    assert args.component_node_counts == [200, 250]
