from types import SimpleNamespace

import pytest

from src.llm import call_custom


def test_custom_endpoint_requires_configuration(monkeypatch):
    monkeypatch.setattr(call_custom, "CUSTOM_CHAT_COMPLETIONS_URL", "")
    with pytest.raises(ValueError, match="CUSTOM_CHAT_COMPLETIONS_URL"):
        call_custom.call_llm_custom([], api_key="test-key")


def test_custom_endpoint_uses_configured_url_and_strips_prefix(monkeypatch):
    sent = []
    monkeypatch.setattr(
        call_custom,
        "CUSTOM_CHAT_COMPLETIONS_URL",
        "https://example.org/chat/completions",
    )
    monkeypatch.setattr(call_custom, "wait_for_custom_rate_limit", lambda: None)

    def post(url, **kwargs):
        sent.append((url, kwargs))
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(call_custom.requests, "post", post)
    call_custom.call_llm_custom(
        [{"role": "user", "content": "Question"}],
        model="custom/test-model",
        api_key="test-key",
    )
    assert sent[0][0] == "https://example.org/chat/completions"
    assert sent[0][1]["json"]["model"] == "test-model"
    assert sent[0][1]["headers"]["Authorization"] == "Bearer test-key"
