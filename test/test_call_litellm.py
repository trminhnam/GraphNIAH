from __future__ import annotations

import asyncio
import importlib

import pytest

from src.llm.model_info import MODEL_SLUG_TO_API_MODEL_NAME

call_litellm_module = importlib.import_module("src.llm.call_litellm")


def test_call_litellm_forwards_prepared_kwargs(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_completion(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return "response"

    monkeypatch.setattr(call_litellm_module.litellm, "completion", fake_completion)

    result = call_litellm_module.call_litellm(
        "positional", model="test-model", messages=[]
    )

    assert result == "response"
    assert calls == {
        "args": ("positional",),
        "kwargs": {"model": "test-model", "messages": []},
    }


def test_call_litellm_forwards_configured_base_url(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_completion(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return "response"

    monkeypatch.setattr(call_litellm_module.litellm, "completion", fake_completion)
    monkeypatch.setattr(
        call_litellm_module,
        "MODEL_NAME_TO_BASE_URL",
        {"test-model": "https://llm.example.com/v1"},
    )

    result = call_litellm_module.call_litellm(
        "positional", model="test-model", messages=[]
    )

    assert result == "response"
    assert calls == {
        "args": ("positional",),
        "kwargs": {
            "model": "test-model",
            "messages": [],
            "base_url": "https://llm.example.com/v1",
        },
    }


def test_call_litellm_async_forwards_prepared_kwargs(monkeypatch) -> None:
    calls: dict[str, object] = {}

    async def fake_acompletion(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return "async response"

    monkeypatch.setattr(call_litellm_module.litellm, "acompletion", fake_acompletion)

    result = asyncio.run(
        call_litellm_module.call_litellm_async(
            "positional", model="test-model", messages=[]
        )
    )

    assert result == "async response"
    assert calls == {
        "args": ("positional",),
        "kwargs": {"model": "test-model", "messages": []},
    }


def build_service_unavailable_error():
    return call_litellm_module.litellm.ServiceUnavailableError(
        message="This model is currently experiencing high demand.",
        llm_provider="gemini",
        model="gemini-3.8-flash",
    )


def test_call_litellm_async_reraises_503_without_flex(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    async def fake_acompletion(*args, **kwargs):
        calls.append(kwargs)
        raise build_service_unavailable_error()

    monkeypatch.setattr(call_litellm_module.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(call_litellm_module, "MODEL_NAME_TO_BASE_URL", {})

    with pytest.raises(call_litellm_module.litellm.ServiceUnavailableError):
        asyncio.run(
            call_litellm_module.call_litellm_async(
                model="deepseek/deepseek-v4-pro", messages=[]
            )
        )

    assert len(calls) == 1


@pytest.mark.parametrize(
    "model",
    sorted(
        {
            name
            for name in MODEL_SLUG_TO_API_MODEL_NAME.values()
            if name and name.startswith("gemini/")
        }
    ),
)
def test_call_litellm_async_keeps_gemini_on_flex_after_503(monkeypatch, model) -> None:
    calls: list[dict[str, object]] = []

    async def fake_acompletion(*args, **kwargs):
        calls.append(kwargs)
        raise build_service_unavailable_error()

    monkeypatch.setattr(call_litellm_module.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(call_litellm_module, "MODEL_NAME_TO_BASE_URL", {})

    with pytest.raises(call_litellm_module.litellm.ServiceUnavailableError):
        asyncio.run(
            call_litellm_module.call_litellm_async(
                model=model, messages=[]
            )
        )

    assert [call["service_tier"] for call in calls] == ["flex"]


def test_call_litellm_async_falls_back_to_standard_for_gpt(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    async def fake_acompletion(*args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("service_tier") == "flex":
            raise build_service_unavailable_error()
        return "standard response"

    monkeypatch.setattr(call_litellm_module.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(call_litellm_module, "MODEL_NAME_TO_BASE_URL", {})

    result = asyncio.run(
        call_litellm_module.call_litellm_async(model="gpt-test", messages=[])
    )

    assert result == "standard response"
    assert [call.get("service_tier") for call in calls] == ["flex", None]
