from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import litellm
import orjson
import pytest
import requests

import inference
from src.prompts import FORMAT_REMINDER


def build_args(input_path: Path, output_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input=str(input_path),
        output=str(output_path),
        model="openai/gpt-5.4-nano",
        api_key=None,
        temperature=0.7,
        top_p=1.0,
        top_k=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        reasoning_effort="low",
        stop=None,
        question_position="first",
        max_samples=None,
        resume=False,
        max_workers=1,
        num_turns=1,
    )


def build_response(
    content: str,
    *,
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
    total_tokens: int = 18,
    response_cost: float = 0.03,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
        _hidden_params={"response_cost": response_cost},
    )


def test_build_prompt_record_preserves_existing_user_message() -> None:
    existing_user_message = (
        "# Question\nWho wrote the paper?\n\n# Context\n\npaper text\n\n"
        f"{FORMAT_REMINDER}"
    )
    record = inference.build_prompt_record(
        {
            "qid": "question-1",
            "user_message": existing_user_message,
            "system_prompt": "reachability system prompt",
        },
        0,
    )

    assert record["qid"] == "question-1"
    assert record["user_message"] == existing_user_message
    assert record["system_prompt"] == "reachability system prompt"


def test_build_prompt_record_preserves_no_system_message_mode() -> None:
    record = inference.build_prompt_record(
        {
            "qid": "question-1",
            "user_message": "# Instruction\nDo the task.\n\n# Question\nQ?",
            "no_system_message": True,
        },
        0,
    )

    assert record["no_system_message"] is True
    assert "system_prompt" not in record


def test_build_prompt_record_requires_prebuilt_user_message() -> None:
    with pytest.raises(ValueError, match="user_message"):
        inference.build_prompt_record(
            {
                "qid": "question-2",
                "question": "Who wrote the paper?",
                "context": "paper text",
            },
            0,
        )


def test_resolve_model_name_uses_api_model_name_for_known_slug() -> None:
    assert inference.resolve_model_name("gpt-5.3") == "openai/gpt-5.3-chat-latest"
    assert inference.resolve_model_name("qwen3.6-27b") == "custom/qwen3.6:27b"


def test_resolve_model_name_preserves_original_when_override_is_missing_or_unknown(
) -> None:
    assert (
        inference.resolve_model_name("definitely-unknown-model")
        == "definitely-unknown-model"
    )
    assert inference.resolve_model_name("openai/gpt-5.4-nano") == "openai/gpt-5.4-nano"


def test_run_inference_sends_user_only_for_no_system_message(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "questions.json"
    output_path = tmp_path / "predictions.json"
    user_message = "# Instruction\nDo the task.\n\n# Question\nQ?"
    input_path.write_text(
        orjson.dumps({
                "questions": [
                    {
                        "qid": "question-1",
                        "user_message": user_message,
                        "no_system_message": True,
                    }
                ]
            }).decode("utf-8"),
        encoding="utf-8",
    )
    captured_messages: list[dict[str, str]] = []

    async def fake_call_litellm_async(**kwargs):
        captured_messages.extend(kwargs["messages"])
        return build_response("\\answer{A}")

    monkeypatch.setattr(inference, "call_litellm_async", fake_call_litellm_async)
    monkeypatch.setattr(
        litellm,
        "cost_per_token",
        lambda **_kwargs: (0.01, 0.02),
    )

    asyncio.run(inference.run_inference(build_args(input_path, output_path)))

    assert captured_messages == [{"role": "user", "content": user_message}]


def test_run_inference_routes_resolved_custom_model_and_preserves_reasoning_effort(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "questions.json"
    output_path = tmp_path / "predictions.json"
    input_path.write_text(
        orjson.dumps({
                "questions": [
                    {
                        "qid": "question-1",
                        "user_message": "# Question\nQ?\n\n# Context\nC.",
                    }
                ]
            }).decode("utf-8"),
        encoding="utf-8",
    )
    captured_kwargs: dict[str, object] = {}

    async def fake_call_llm_custom_async(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            json=lambda: {
                "choices": [
                    {
                        "message": {
                            "content": "reasoning\n\\answer{A}",
                            "reasoning_content": "reasoning",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            }
        )

    async def fail_call_litellm_async(**_kwargs):
        raise AssertionError("LiteLLM must not handle CUSTOM models")

    monkeypatch.setattr(inference, "call_llm_custom_async", fake_call_llm_custom_async)
    monkeypatch.setattr(inference, "call_litellm_async", fail_call_litellm_async)
    args = build_args(input_path, output_path)
    args.model_slug = "qwen3.6-27b"
    args.model = inference.resolve_model_name(args.model_slug)
    args.api_key = "test-key"
    args.reasoning_effort = "high"

    asyncio.run(inference.run_inference(args))

    payload = orjson.loads(output_path.read_text(encoding="utf-8"))
    assert captured_kwargs["model"] == "custom/qwen3.6:27b"
    assert captured_kwargs["api_key"] == "test-key"
    assert captured_kwargs["reasoning_effort"] == "high"
    assert payload["parameters"]["model"] == "qwen3.6-27b"
    assert payload["records"][0]["raw_answer"] == "reasoning\n\\answer{A}"
    assert payload["records"][0]["reasoning_content"] == "reasoning"
    assert payload["records"][0]["usage"] == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }


def test_completion_retry_uses_retry_after_for_custom(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    sleep_calls: list[float] = []
    rate_limit_error = requests.HTTPError(
        response=SimpleNamespace(
            status_code=429,
            headers={"Retry-After": "7"},
        )
    )

    async def fake_call_llm_custom_async(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise rate_limit_error
        return SimpleNamespace(
            json=lambda: {
                "choices": [{"message": {"content": "\\answer{A}"}}],
                "usage": {},
            }
        )

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    async def run_retry():
        return await inference.call_completion_with_retry(
            {
                "model": "custom/qwen3.6:27b",
                "messages": [{"role": "user", "content": "Q"}],
                "reasoning_effort": "high",
            },
            use_custom_router=True,
            model="custom/qwen3.6:27b",
            question_id="question-1",
            semaphore=asyncio.Semaphore(1),
        )

    monkeypatch.setattr(inference, "call_llm_custom_async", fake_call_llm_custom_async)
    monkeypatch.setattr(inference.asyncio, "sleep", fake_sleep)

    result = asyncio.run(run_retry())

    assert result["content"] == "\\answer{A}"
    assert len(calls) == 2
    assert calls[1]["reasoning_effort"] == "high"
    assert "num_retries" not in calls[1]
    assert sleep_calls == [7.0]


def test_completion_retry_handles_null_custom_response_body(monkeypatch) -> None:
    calls = 0

    async def fake_call_llm_custom_async(**_kwargs):
        nonlocal calls
        calls += 1
        payload = (
            None
            if calls == 1
            else {"choices": [{"message": {"content": "\\answer{A}"}}]}
        )
        return SimpleNamespace(json=lambda: payload)

    async def run_retry():
        return await inference.call_completion_with_retry(
            {"model": "custom/qwen3.6:27b", "messages": []},
            use_custom_router=True,
            model="custom/qwen3.6:27b",
            question_id="question-1",
            semaphore=asyncio.Semaphore(1),
        )

    monkeypatch.setattr(inference, "call_llm_custom_async", fake_call_llm_custom_async)
    monkeypatch.setattr(inference, "calculate_retry_delay", lambda *args: 0.0)

    result = asyncio.run(run_retry())

    assert calls == 2
    assert result["content"] == "\\answer{A}"


def test_completion_retry_handles_null_litellm_response(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    sleep_calls: list[float] = []

    async def fake_call_litellm_async(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return None
        return build_response("\\answer{A}")

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    async def run_retry():
        return await inference.call_completion_with_retry(
            {"model": "test-model", "messages": []},
            use_custom_router=False,
            model="test-model",
            question_id="question-1",
            semaphore=asyncio.Semaphore(1),
        )

    monkeypatch.setattr(inference, "call_litellm_async", fake_call_litellm_async)
    monkeypatch.setattr(inference, "calculate_retry_delay", lambda *args: 0.0)
    monkeypatch.setattr(inference.asyncio, "sleep", fake_sleep)

    result = asyncio.run(run_retry())

    assert result["content"] == "\\answer{A}"
    assert len(calls) == 2
    assert all(call["num_retries"] == 0 for call in calls)
    assert sleep_calls == [0.0]


def test_completion_retry_does_not_retry_authentication_error(monkeypatch) -> None:
    calls = 0
    authentication_error = litellm.AuthenticationError(
        message="bad key",
        llm_provider="openai",
        model="test-model",
    )

    async def fake_call_litellm_async(**_kwargs):
        nonlocal calls
        calls += 1
        raise authentication_error

    async def fail_sleep(_delay: float) -> None:
        raise AssertionError("Authentication errors must not be retried")

    async def run_retry():
        return await inference.call_completion_with_retry(
            {"model": "test-model", "messages": []},
            use_custom_router=False,
            model="test-model",
            question_id="question-1",
            semaphore=asyncio.Semaphore(1),
        )

    monkeypatch.setattr(inference, "call_litellm_async", fake_call_litellm_async)
    monkeypatch.setattr(inference.asyncio, "sleep", fail_sleep)

    with pytest.raises(inference.CompletionRequestError) as error_info:
        asyncio.run(run_retry())

    assert calls == 1
    assert error_info.value.attempts == 1


def test_completion_retry_stops_after_max_attempts(monkeypatch) -> None:
    calls = 0
    sleep_calls: list[float] = []

    async def fake_call_litellm_async(**_kwargs):
        nonlocal calls
        calls += 1
        return None

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    async def run_retry():
        return await inference.call_completion_with_retry(
            {"model": "test-model", "messages": []},
            use_custom_router=False,
            model="test-model",
            question_id="question-1",
            semaphore=asyncio.Semaphore(1),
        )

    monkeypatch.setattr(inference, "LLM_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(inference, "call_litellm_async", fake_call_litellm_async)
    monkeypatch.setattr(inference, "calculate_retry_delay", lambda *args: 0.0)
    monkeypatch.setattr(inference.asyncio, "sleep", fake_sleep)

    with pytest.raises(inference.CompletionRequestError) as error_info:
        asyncio.run(run_retry())

    assert calls == 3
    assert len(sleep_calls) == 2
    assert error_info.value.attempts == 3


def test_completion_retry_times_only_the_successful_request(monkeypatch) -> None:
    clock = {"now": 100.0}
    calls = 0
    rate_limit_error = requests.HTTPError(
        response=SimpleNamespace(status_code=429, headers={})
    )

    async def fake_call_llm_custom_async(*, on_send, **_kwargs):
        nonlocal calls
        calls += 1
        clock["now"] += 30.0  # CUSTOM rate-limit wait, before the request is sent
        on_send()
        clock["now"] += 5.0 if calls == 1 else 2.5
        if calls == 1:
            raise rate_limit_error
        return SimpleNamespace(
            json=lambda: {"choices": [{"message": {"content": "\\answer{A}"}}]}
        )

    async def fake_sleep(_delay: float) -> None:
        return None

    async def run_retry():
        return await inference.call_completion_with_retry(
            {"model": "custom/qwen3.6:27b", "messages": []},
            use_custom_router=True,
            model="custom/qwen3.6:27b",
            question_id="question-1",
            semaphore=asyncio.Semaphore(1),
        )

    monkeypatch.setattr(inference, "call_llm_custom_async", fake_call_llm_custom_async)
    monkeypatch.setattr(inference, "calculate_retry_delay", lambda *args: 0.0)
    monkeypatch.setattr(inference.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(inference.time, "perf_counter", lambda: clock["now"])

    result = asyncio.run(run_retry())

    assert result["latency_seconds"] == 2.5
    assert result["attempts"] == 2
    started_at = datetime.fromisoformat(result["started_at"])
    finished_at = datetime.fromisoformat(result["finished_at"])
    assert started_at.utcoffset() == timedelta(0)
    assert started_at <= finished_at


def test_run_inference_records_request_timing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "questions.json"
    output_path = tmp_path / "predictions.json"
    input_path.write_text(
        orjson.dumps({
                "questions": [
                    {"qid": "question-1", "user_message": "# Question\nQ1?"},
                    {"qid": "question-2", "user_message": "# Question\nQ2?"},
                ]
            }).decode("utf-8"),
        encoding="utf-8",
    )
    authentication_error = litellm.AuthenticationError(
        message="bad key",
        llm_provider="openai",
        model="test-model",
    )

    async def fake_call_litellm_async(**kwargs):
        if "Q2?" in kwargs["messages"][-1]["content"]:
            raise authentication_error
        return build_response("\\answer{A}")

    monkeypatch.setattr(inference, "call_litellm_async", fake_call_litellm_async)
    monkeypatch.setattr(
        litellm,
        "cost_per_token",
        lambda **_kwargs: (0.01, 0.02),
    )

    asyncio.run(inference.run_inference(build_args(input_path, output_path)))

    payload = orjson.loads(output_path.read_text(encoding="utf-8"))
    records = {record["qid"]: record for record in payload["records"]}
    answered = records["question-1"]
    assert answered["attempts"] == 1
    assert answered["latency_seconds"] >= 0
    assert datetime.fromisoformat(answered["started_at"]) <= datetime.fromisoformat(
        answered["finished_at"]
    )
    failed = records["question-2"]
    assert failed["raw_answer"] is None
    assert failed["attempts"] == 1
    assert failed["latency_seconds"] is None
    assert failed["started_at"] is None
    assert failed["finished_at"] is None


def test_run_inference_automatically_skips_qids_from_existing_output(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    input_path = tmp_path / "questions.json"
    output_path = tmp_path / "predictions.json"
    input_path.write_text(
        orjson.dumps({
                "questions": [
                    {
                        "qid": "question-1",
                        "user_message": (
                            "# Question\nWho wrote the paper?\n\n"
                            "# Context\n\npaper text\n\n"
                            f"{FORMAT_REMINDER}"
                        ),
                    },
                    {
                        "qid": "question-2",
                        "user_message": (
                            "# Question\nWho edited the paper?\n\n"
                            "# Context\n\npaper text\n\n"
                            f"{FORMAT_REMINDER}"
                        ),
                    },
                ]
            }).decode("utf-8"),
        encoding="utf-8",
    )
    output_path.write_text(
        orjson.dumps({
                "metadata": {"total_samples": 1},
                "records": [
                    {
                        "qid": "question-1",
                        "raw_answer": "reasoning\n\\answer{Alice}",
                        "usage": {
                            "input_tokens": 11,
                            "output_tokens": 7,
                            "total_tokens": 18,
                        },
                        "cost": {
                            "input_cost": 0.01,
                            "output_cost": 0.02,
                            "total_cost": 0.03,
                        },
                    }
                ],
            }).decode("utf-8"),
        encoding="utf-8",
    )

    captured_prompts: list[str] = []

    async def fake_call_litellm_async(**kwargs):
        captured_prompts.append(kwargs["messages"][1]["content"])
        return build_response("reasoning\n\\answer{Bob}", response_cost=0.06)

    monkeypatch.setattr(inference, "call_litellm_async", fake_call_litellm_async)
    monkeypatch.setattr(
        litellm,
        "cost_per_token",
        lambda **_kwargs: (0.02, 0.04),
    )

    asyncio.run(inference.run_inference(build_args(input_path, output_path)))

    payload = orjson.loads(output_path.read_text(encoding="utf-8"))
    stdout = capsys.readouterr().out

    assert len(captured_prompts) == 1
    assert "Who edited the paper?" in captured_prompts[0]
    assert "1 successful turns reused; 1 examples complete" in stdout
    assert "[1/1 | 100%] OK example 2/2, turn 1/1 | successful 2/2" in stdout
    assert payload["metadata"]["total_samples"] == 2
    assert payload["parameters"] == {
        "num_hops": None,
        "needle_type": None,
        "distracting_content_type": None,
        "needle_order": None,
        "target_context_length": None,
        "model": "openai/gpt-5.4-nano",
    }
    assert [record["qid"] for record in payload["records"]] == [
        "question-1",
        "question-2",
    ]


def build_yesno_item(qid: str, label: str) -> dict[str, object]:
    """Return one reachability yes/no prompt item with a categorical answer."""
    return {
        "qid": qid,
        "user_message": (
            f"# Question\nIs the target reachable? ({qid})\n\n"
            f"# Context\n\npaper text\n\n{FORMAT_REMINDER}"
        ),
        "answer": [label],
        "answer_type": "REACHABILITY_YESNO",
        "question_type": "reachability-yesno",
    }


def test_stratify_by_label_interleaves_a_label_grouped_file() -> None:
    items = [build_yesno_item(f"yes-{index}", "YES") for index in range(3)]
    items += [build_yesno_item(f"no-{index}", "NO") for index in range(3)]

    ordered = inference.stratify_by_label(items, inference.stratification_label)

    assert [item["qid"] for item in ordered] == [
        "yes-0",
        "no-0",
        "yes-1",
        "no-1",
        "yes-2",
        "no-2",
    ]


def test_stratify_by_label_rotates_multiple_choice_options() -> None:
    items = [
        {
            "qid": f"{label}-{index}",
            "user_message": "prompt",
            "answer": [label, f"{label}."],
            "answer_type": "REACHABILITY_MULTIPLE",
        }
        for label in ("A", "B", "C", "D")
        for index in range(2)
    ]

    ordered = inference.stratify_by_label(items, inference.stratification_label)

    assert [item["qid"] for item in ordered] == [
        "A-0",
        "B-0",
        "C-0",
        "D-0",
        "A-1",
        "B-1",
        "C-1",
        "D-1",
    ]


def test_stratify_by_label_resumes_on_the_label_that_is_behind() -> None:
    pending = [build_yesno_item(f"yes-{index}", "YES") for index in range(5, 8)]
    pending += [build_yesno_item(f"no-{index}", "NO") for index in range(4, 7)]

    ordered = inference.stratify_by_label(
        pending,
        inference.stratification_label,
        {"YES": 5, "NO": 4},
    )

    # NO is one behind, so it goes first; each label then continues from the
    # first item still pending in its own queue.
    assert [item["qid"] for item in ordered] == [
        "no-4",
        "yes-5",
        "no-5",
        "yes-6",
        "no-6",
        "yes-7",
    ]


def test_stratify_by_label_drains_a_label_whose_queue_runs_out() -> None:
    items = [build_yesno_item("yes-0", "YES")]
    items += [build_yesno_item(f"no-{index}", "NO") for index in range(3)]

    ordered = inference.stratify_by_label(items, inference.stratification_label)

    assert [item["qid"] for item in ordered] == ["yes-0", "no-0", "no-1", "no-2"]


def test_stratify_by_label_preserves_order_for_free_form_answer_types() -> None:
    items = [
        {
            "qid": f"question-{index}",
            "user_message": "prompt",
            "answer": [f"Some Paper Title {index}"],
            "answer_type": "COUNT_PAPER_CITATIONS",
        }
        for index in range(4)
    ]

    ordered = inference.stratify_by_label(items, inference.stratification_label)

    assert [item["qid"] for item in ordered] == [item["qid"] for item in items]


def test_run_inference_max_samples_keeps_a_label_balanced_prefix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "full_prompts.json"
    output_path = tmp_path / "inference.json"
    questions = [build_yesno_item(f"yes-{index}", "YES") for index in range(3)]
    questions += [build_yesno_item(f"no-{index}", "NO") for index in range(3)]
    input_path.write_text(
        orjson.dumps({"questions": questions}).decode("utf-8"),
        encoding="utf-8",
    )

    async def fake_call_litellm_async(**_kwargs):
        return build_response("reasoning\n\\answer{YES}")

    monkeypatch.setattr(inference, "call_litellm_async", fake_call_litellm_async)
    monkeypatch.setattr(
        litellm,
        "cost_per_token",
        lambda **_kwargs: (0.01, 0.02),
    )

    args = build_args(input_path, output_path)
    args.max_samples = 4
    asyncio.run(inference.run_inference(args))

    payload = orjson.loads(output_path.read_text(encoding="utf-8"))

    # A raw prefix of this file would be 3 YES + 1 NO.
    assert [record["qid"] for record in payload["records"]] == [
        "yes-0",
        "no-0",
        "yes-1",
        "no-1",
    ]


def test_run_inference_resume_rebalances_a_label_skewed_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "full_prompts.json"
    output_path = tmp_path / "inference.json"
    questions = [build_yesno_item(f"yes-{index}", "YES") for index in range(3)]
    questions += [build_yesno_item(f"no-{index}", "NO") for index in range(3)]
    input_path.write_text(
        orjson.dumps({"questions": questions}).decode("utf-8"),
        encoding="utf-8",
    )
    # An earlier non-stratified run finished every YES and only one NO.
    output_path.write_text(
        orjson.dumps({
                "metadata": {"total_samples": 4},
                "records": [
                    {"qid": "yes-0", "raw_answer": "\\answer{YES}"},
                    {"qid": "yes-1", "raw_answer": "\\answer{YES}"},
                    {"qid": "yes-2", "raw_answer": "\\answer{YES}"},
                    {"qid": "no-0", "raw_answer": "\\answer{NO}"},
                ],
            }).decode("utf-8"),
        encoding="utf-8",
    )

    requested_qids: list[str] = []

    async def fake_call_litellm_async(**kwargs):
        requested_qids.append(kwargs["messages"][1]["content"])
        return build_response("reasoning\n\\answer{NO}")

    monkeypatch.setattr(inference, "call_litellm_async", fake_call_litellm_async)
    monkeypatch.setattr(
        litellm,
        "cost_per_token",
        lambda **_kwargs: (0.01, 0.02),
    )

    asyncio.run(inference.run_inference(build_args(input_path, output_path)))

    payload = orjson.loads(output_path.read_text(encoding="utf-8"))

    assert len(requested_qids) == 2
    assert "(no-1)" in requested_qids[0]
    assert "(no-2)" in requested_qids[1]
    assert [record["qid"] for record in payload["records"]] == [
        "yes-0",
        "no-0",
        "yes-1",
        "no-1",
        "yes-2",
        "no-2",
    ]


@pytest.fixture
def turn_job(tmp_path: Path, monkeypatch):
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_bytes(
        orjson.dumps(
            [
                {"qid": "q1", "user_message": "First question"},
                {"qid": "q2", "user_message": "Second question"},
            ]
        )
    )
    monkeypatch.setattr(litellm, "completion_cost", lambda **kwargs: 0.03)
    monkeypatch.setattr(litellm, "cost_per_token", lambda **kwargs: (0.01, 0.02))
    return build_args(input_path, output_path)


@pytest.mark.parametrize(
    ("cost", "expected"),
    [
        (
            {"input_cost": 0.01, "output_cost": 0.02, "total_cost": 0.03},
            "Input Cost: $0.020 | Output Cost: $0.040 | Total Cost: $0.060",
        ),
        (
            {"input_cost": None, "output_cost": None, "total_cost": 0.03},
            "Input Cost: unavailable | Output Cost: unavailable | Total Cost: $0.060",
        ),
        (
            {"input_cost": None, "output_cost": None, "total_cost": None},
            "Input Cost: unavailable | Output Cost: unavailable | Total Cost: unavailable",
        ),
    ],
)
def test_cost_summary_separates_input_output_and_total(
    turn_job, monkeypatch, capsys, cost, expected
):
    async def complete(**kwargs):
        return build_response("answer")

    monkeypatch.setattr(inference, "call_litellm_async", complete)
    monkeypatch.setattr(
        inference, "response_accounting", lambda *args, **kwargs: {"cost": cost}
    )
    asyncio.run(inference.run_inference(turn_job))
    stdout = capsys.readouterr().out
    assert expected in stdout
    assert "Known response cost:" not in stdout
    if cost["input_cost"] is None:
        assert "Cost coverage (known turns): Input: 0/2 | Output: 0/2" in stdout
    else:
        assert "Cost coverage" not in stdout


def test_progress_logs_failures_and_resumed_work(turn_job, monkeypatch, capsys):
    wait = asyncio.wait
    waiting_reported = False

    async def wait_with_initial_timeout(futures, *, timeout):
        nonlocal waiting_reported
        assert timeout == 30
        if not waiting_reported:
            waiting_reported = True
            return set(), futures
        return await wait(futures, timeout=timeout)

    monkeypatch.setattr(inference.asyncio, "wait", wait_with_initial_timeout)

    async def complete(**kwargs):
        if kwargs["messages"][-1]["content"] == "Second question":
            raise litellm.AuthenticationError(
                message="bad key", llm_provider="openai", model="test-model"
            )
        return build_response("answer")

    monkeypatch.setattr(inference, "call_litellm_async", complete)
    asyncio.run(inference.run_inference(turn_job))
    stdout = capsys.readouterr().out
    assert "[Inference run]" in stdout
    assert "[Waiting] openai/gpt-5.4-nano | 0/2 turns saved" in stdout
    assert "Model           : GPT-5.4-Nano" in stdout
    assert "Reasoning effort: low (requested)" in stdout
    assert (
        "[2/2 | 100%] FAILED example 2/2, turn 1/1 | successful 1/2 | failed 1"
        in stdout
    )
    assert "elapsed " in stdout and "ETA ~" in stdout
    assert "Turns: 1 successful, 1 failed, 2 requested" in stdout
    assert "First question" not in stdout

    async def recovered(**kwargs):
        return build_response("answer")

    monkeypatch.setattr(inference, "call_litellm_async", recovered)
    asyncio.run(inference.run_inference(turn_job))
    stdout = capsys.readouterr().out
    assert "1 successful turns reused; 1 examples complete" in stdout
    assert "[1/1 | 100%] OK example 2/2, turn 1/1 | successful 2/2 | failed 0" in stdout

    asyncio.run(inference.run_inference(turn_job))
    stdout = capsys.readouterr().out
    assert "Remaining       : 0 turns" in stdout
    assert "[Run complete]" in stdout
    assert "ETA" not in stdout


def test_resume_completed_four_turns_then_extend_to_five(turn_job, monkeypatch):
    calls = []

    async def complete(**kwargs):
        calls.append(orjson.loads(orjson.dumps(kwargs["messages"])))
        return build_response(f"answer-{len(calls)}")

    monkeypatch.setattr(inference, "call_litellm_async", complete)
    turn_job.num_turns = 4
    output = Path(turn_job.output)
    asyncio.run(inference.run_inference(turn_job))
    first = orjson.loads(output.read_bytes())
    assert len(calls) == 8
    assert calls[:4] == [calls[0]] * 4
    assert calls[4:] == [calls[4]] * 4
    assert [(r["qid"], r["turn_index"]) for r in first["records"]] == [
        (qid, turn) for qid in ("q1", "q2") for turn in range(1, 5)
    ]
    # Same target makes no requests, even when concurrency changes.
    turn_job.max_workers = 2
    asyncio.run(inference.run_inference(turn_job))
    assert len(calls) == 8
    turn_job.num_turns = 5
    asyncio.run(inference.run_inference(turn_job))
    extended = orjson.loads(output.read_bytes())
    assert len(calls) == 10
    assert [r for r in extended["records"] if r["turn_index"] <= 4] == first["records"]
    assert [r["inference_id"] for r in extended["records"] if r["turn_index"] == 5] == [
        "q1::turn-5",
        "q2::turn-5",
    ]
    assert (
        extended["metadata"]["request_fingerprint"]
        == first["metadata"]["request_fingerprint"]
    )
    assert extended["metadata"]["completed_examples"] == 2
    assert extended["metadata"]["successful_turns"] == 10
    asyncio.run(inference.run_inference(turn_job))
    assert len(calls) == 10


@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("cut_input", [False, True])
def test_resume_extends_max_samples_from_four_to_one_hundred(
    turn_job, monkeypatch, interrupted, cut_input
):
    from cut_full_prompts import select_samples

    items = [
        {
            "qid": f"q{i}",
            "user_message": f"Question {i}",
            "answer_type": "REACHABILITY_YESNO",
            "answer": ["YES" if i < 50 else "NO"],
        }
        for i in range(100)
    ]
    input_path = Path(turn_job.input)
    input_path.write_bytes(
        orjson.dumps(select_samples(items, 4) if cut_input else items)
    )
    calls = []
    interrupt_run = interrupted

    async def complete(**kwargs):
        if interrupt_run and len(calls) >= 8:
            raise asyncio.CancelledError()
        calls.append(kwargs["messages"][-1]["content"])
        if interrupt_run and len(calls) == 8:
            raise asyncio.CancelledError()
        return build_response(f"answer-{len(calls)}")

    monkeypatch.setattr(inference, "call_litellm_async", complete)
    turn_job.num_turns = 5
    turn_job.max_samples = 4
    if interrupted:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(inference.run_inference(turn_job))
    else:
        asyncio.run(inference.run_inference(turn_job))
    output = Path(turn_job.output)
    first = orjson.loads(output.read_bytes())
    assert len(first["records"]) == (7 if interrupted else 20)
    saved_ids = {record["inference_id"] for record in first["records"]}
    previous_calls = len(calls)
    interrupt_run = False

    if cut_input:
        input_path = input_path.with_name("input.max_samples=100.json")
        input_path.write_bytes(orjson.dumps(select_samples(items, 100)))
        turn_job.input = str(input_path)
    turn_job.max_samples = 100
    asyncio.run(inference.run_inference(turn_job))
    resumed = orjson.loads(output.read_bytes())
    assert len(calls) - previous_calls == 500 - len(saved_ids)
    assert len(resumed["records"]) == 500
    assert [
        record for record in resumed["records"] if record["inference_id"] in saved_ids
    ] == first["records"]
    assert resumed["metadata"]["completed_examples"] == 100
    assert resumed["metadata"]["successful_turns"] == 500
    assert resumed["metadata"]["max_samples"] == 100
    assert resumed["metadata"]["legacy_checkpoint_unverified"] is False
    assert (
        resumed["metadata"]["request_fingerprint"]
        != first["metadata"]["request_fingerprint"]
    )
    previous_calls = len(calls)
    asyncio.run(inference.run_inference(turn_job))
    assert len(calls) == previous_calls


@pytest.mark.parametrize(
    "change", ["prompt", "system_prompt", "model", "temperature", "remove", "reorder"]
)
def test_extend_max_samples_rejects_incompatible_checkpoint(
    turn_job, monkeypatch, change
):
    items = [{"qid": f"q{i}", "user_message": f"Question {i}"} for i in range(5)]
    input_path = Path(turn_job.input)
    input_path.write_bytes(orjson.dumps(items))
    calls = []

    async def complete(**kwargs):
        if len(calls) >= 2:
            raise asyncio.CancelledError()
        calls.append(kwargs)
        if len(calls) == 2:
            raise asyncio.CancelledError()
        return build_response("answer")

    monkeypatch.setattr(inference, "call_litellm_async", complete)
    turn_job.max_samples = 4
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(inference.run_inference(turn_job))
    output = Path(turn_job.output)
    before = output.read_bytes()
    turn_job.max_samples = 100
    # q3 was selected but never requested; its prompt must still be verified.
    if change in ("prompt", "system_prompt"):
        items[3]["user_message" if change == "prompt" else "system_prompt"] = "changed"
    elif change == "remove":
        items.pop(3)
    elif change == "reorder":
        items[2], items[3] = items[3], items[2]
    else:
        setattr(turn_job, change, "different" if change == "model" else 0.9)
    input_path.write_bytes(orjson.dumps(items))
    with pytest.raises(ValueError, match="Cannot resume"):
        asyncio.run(inference.run_inference(turn_job))
    assert len(calls) == 2
    assert output.read_bytes() == before


def test_resume_after_interruption_saves_each_turn(turn_job, monkeypatch):
    calls = []

    async def complete(**kwargs):
        calls.append(kwargs["messages"][-1]["content"])
        if len(calls) == 3:
            raise asyncio.CancelledError()
        return build_response(f"answer-{len(calls)}")

    monkeypatch.setattr(inference, "call_litellm_async", complete)
    turn_job.num_turns = 4
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(inference.run_inference(turn_job))
    output = Path(turn_job.output)
    saved = orjson.loads(output.read_bytes())["records"]
    assert [r["turn_index"] for r in saved if r["qid"] == "q1"] == [1, 2]
    asyncio.run(inference.run_inference(turn_job))
    resumed = orjson.loads(output.read_bytes())
    saved_ids = {record["inference_id"] for record in saved}
    assert [
        record for record in resumed["records"] if record["inference_id"] in saved_ids
    ] == saved
    assert len(resumed["records"]) == 8
    assert (
        len(calls) == 9
    )  # Includes the interrupted request, whose answer was never saved.
    assert resumed["metadata"]["completed_examples"] == 2


def test_resume_fills_failed_turn_without_repeating_later_turns(turn_job, monkeypatch):
    calls = []

    async def complete(**kwargs):
        calls.append(kwargs["messages"][-1]["content"])
        return build_response("" if len(calls) == 2 else f"answer-{len(calls)}")

    monkeypatch.setattr(inference, "call_litellm_async", complete)
    monkeypatch.setattr(inference, "LLM_MAX_ATTEMPTS", 1)
    turn_job.num_turns = 3
    asyncio.run(inference.run_inference(turn_job))
    output = Path(turn_job.output)
    initial = orjson.loads(output.read_bytes())
    failed = initial["records"][1]
    assert failed["raw_answer"] is None
    assert failed["failed_attempts"][0]["cost"]["total_cost"] == 0.03
    assert initial["metadata"]["failed_turns"] == 1
    asyncio.run(inference.run_inference(turn_job))
    resumed = orjson.loads(output.read_bytes())
    assert len(calls) == 7
    assert len(resumed["records"]) == 6
    assert resumed["records"][1]["attempts"] == 2
    assert resumed["records"][1]["failed_attempts"] == failed["failed_attempts"]
    assert resumed["records"][2:] == initial["records"][2:]
    assert resumed["metadata"]["retry_accounting"]["total_total_cost"] == 0.03


def test_workers_keep_examples_until_all_turns_finish(turn_job, monkeypatch):
    Path(turn_job.input).write_bytes(
        orjson.dumps([{"qid": str(i), "user_message": str(i)} for i in range(3)])
    )
    turn_job.num_turns, turn_job.max_workers = 3, 2
    finished = {str(i): 0 for i in range(3)}
    active = set()
    max_active = 0

    async def complete(**kwargs):
        nonlocal max_active
        qid = kwargs["messages"][-1]["content"]
        assert qid not in active
        if qid == "2":
            assert finished["0"] == 3 or finished["1"] == 3
        active.add(qid)
        max_active = max(max_active, len(active))
        await asyncio.sleep(0)
        active.remove(qid)
        finished[qid] += 1
        return build_response("answer")

    monkeypatch.setattr(inference, "call_litellm_async", complete)
    asyncio.run(inference.run_inference(turn_job))
    assert max_active == 2
    assert finished == {"0": 3, "1": 3, "2": 3}


@pytest.mark.parametrize(
    "change",
    ["prompt", "model", "temperature", "system_prompt", "duplicate", "lower_turns"],
)
def test_resume_rejects_incompatible_requests_without_writing(
    turn_job, monkeypatch, change
):
    async def complete(**kwargs):
        return build_response("answer")

    monkeypatch.setattr(inference, "call_litellm_async", complete)
    turn_job.num_turns = 2
    asyncio.run(inference.run_inference(turn_job))
    output = Path(turn_job.output)
    before = output.read_bytes()
    if change in ("prompt", "system_prompt", "duplicate"):
        items = orjson.loads(Path(turn_job.input).read_bytes())
        if change == "duplicate":
            items[1]["qid"] = items[0]["qid"]
        else:
            items[0]["user_message" if change == "prompt" else "system_prompt"] = (
                "changed"
            )
        Path(turn_job.input).write_bytes(orjson.dumps(items))
    elif change == "lower_turns":
        turn_job.num_turns = 1
    else:
        setattr(turn_job, change, "different" if change == "model" else 0.9)
    with pytest.raises(ValueError):
        asyncio.run(inference.run_inference(turn_job))
    assert output.read_bytes() == before


def test_legacy_checkpoint_extends_from_turn_one(turn_job, monkeypatch):
    output = Path(turn_job.output)
    output.write_bytes(
        orjson.dumps({"records": [{"qid": "q1", "raw_answer": "original"}]})
    )
    calls = []

    async def complete(**kwargs):
        calls.append(kwargs["messages"][-1]["content"])
        return build_response("new")

    monkeypatch.setattr(inference, "call_litellm_async", complete)
    turn_job.num_turns = 2
    asyncio.run(inference.run_inference(turn_job))
    result = orjson.loads(output.read_bytes())
    assert len(calls) == 3
    assert result["records"][0]["raw_answer"] == "original"
    assert result["records"][0]["turn_index"] == 1
    assert result["metadata"]["legacy_checkpoint_unverified"] is True


@pytest.mark.parametrize(
    "checkpoint",
    [
        b"{bad",
        b'{"records": {}}',
        b'{"records": [{"qid": "q1", "turn_index": 0}]}',
        b'{"records": [{"qid": "q1"}, {"qid": "q1"}]}',
    ],
)
def test_invalid_checkpoint_is_never_overwritten(turn_job, checkpoint):
    output = Path(turn_job.output)
    output.write_bytes(checkpoint)
    with pytest.raises(ValueError):
        asyncio.run(inference.run_inference(turn_job))
    assert output.read_bytes() == checkpoint


def test_atomic_checkpoint_failure_preserves_old_file(turn_job, monkeypatch):
    output = Path(turn_job.output)
    before = orjson.dumps({"records": []})
    output.write_bytes(before)

    def fail_replace(*args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(inference.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        asyncio.run(inference.run_inference(turn_job))
    assert output.read_bytes() == before
    assert list(output.parent.glob(f".{output.name}.*")) == []


@pytest.mark.parametrize("value", ["0", "-1", "1.5"])
def test_num_turns_must_be_positive_integer(value):
    with pytest.raises(SystemExit):
        inference.arg_parser().parse_args(
            ["--input", "in", "--output", "out", "--num-turns", value]
        )
