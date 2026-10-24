"""Call LiteLLM.

```
python -m src.llm.call_litellm
```
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import litellm

from .model_info import MODEL_NAME_TO_BASE_URL


def prepare_litellm_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Prepare LiteLLM kwargs before every completion call.

    Returns a fresh dict so callers' kwargs are never mutated. OpenAI models
    read OPENAI_API_KEY from the environment through LiteLLM."""

    kwargs = dict(kwargs)
    model_name = kwargs.get("model")
    normalized_model_name = model_name.strip() if isinstance(model_name, str) else ""

    if not kwargs.get("base_url"):
        base_url = MODEL_NAME_TO_BASE_URL.get(normalized_model_name)
        if base_url is not None:
            kwargs["base_url"] = base_url

    if (
        normalized_model_name.lower().startswith("openai")
        or "gpt" in normalized_model_name.lower()
        or "gemini" in normalized_model_name.lower()
    ):
        kwargs["service_tier"] = "flex"
        kwargs["timeout"] = 900

    if normalized_model_name.lower().startswith("huggingface"):
        kwargs["api_key"] = "sk-..."

    if "qwen3.5" in normalized_model_name.lower():
        kwargs["temperature"] = 1
        kwargs["top_p"] = 0.95
        kwargs["top_k"] = 20
        kwargs["min_p"] = 0.0
        kwargs["presence_penalty"] = 1.5
        kwargs["repetition_penalty"] = 1.0
        kwargs["timeout"] = 600
        kwargs["max_tokens"] = 40000
        kwargs["max_completion_tokens"] = 40000

    if "openrouter" in normalized_model_name.lower():
        kwargs["provider"] = {
            "order": [
                "alibaba",
                "deepinfra/fp8",
            ]
            # "sort": "price",
        }

    if "deepseek" in normalized_model_name.lower():
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

    # Prepare reasoning effort for OpenRouter models
    # https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#reasoning-mode
    if "openrouter" in normalized_model_name.lower():
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        if reasoning_effort is None or reasoning_effort == "none":
            kwargs["reasoning"] = {"type": "disabled", "effort": "none"}
        else:
            kwargs["reasoning"] = {"type": "enabled", "effort": reasoning_effort}

    return kwargs


def standard_tier_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of flex-tier kwargs that requests the standard tier.

    Flex returns HTTP 503 when its capacity is full, and the provider never
    upgrades the request to the standard tier by itself.
    https://ai.google.dev/gemini-api/docs/flex-inference
    """
    print(
        f"[Flex fallback] {kwargs.get('model')}: flex returned HTTP 503; "
        "retrying at the standard tier.",
        flush=True,
    )
    return {key: value for key, value in kwargs.items() if key != "service_tier"}


def allows_standard_tier_fallback(kwargs: dict[str, Any]) -> bool:
    """Return whether a flex 503 may be retried at the standard tier.

    Gemini models always stay on flex, so their 503 reaches the caller's retry
    loop instead of being billed at the standard price.
    """
    model_name = str(kwargs.get("model") or "").lower()
    return kwargs.get("service_tier") == "flex" and "gemini" not in model_name


def call_litellm(*args: Any, **kwargs: Any) -> Any:
    prepared_kwargs = prepare_litellm_kwargs(kwargs)
    try:
        return litellm.completion(*args, **prepared_kwargs)
    except litellm.ServiceUnavailableError:
        if not allows_standard_tier_fallback(prepared_kwargs):
            raise
    return litellm.completion(*args, **standard_tier_kwargs(prepared_kwargs))


async def call_litellm_async(
    *args: Any, on_send: Callable[[], None] | None = None, **kwargs: Any
) -> Any:
    """Call LiteLLM; ``on_send`` runs right before a standard-tier fallback."""
    prepared_kwargs = prepare_litellm_kwargs(kwargs)
    try:
        return await litellm.acompletion(*args, **prepared_kwargs)
    except litellm.ServiceUnavailableError:
        if not allows_standard_tier_fallback(prepared_kwargs):
            raise
    fallback_kwargs = standard_tier_kwargs(prepared_kwargs)
    if on_send is not None:
        on_send()
    return await litellm.acompletion(*args, **fallback_kwargs)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    messages = [
        {
            "role": "user",
            "content": "This is a hard math problem, which required reasoning to solve. Please solve it step by step.\n\nDo all positive integer sequences end at 1 if you divide even numbers by 2 and multiply odd numbers by 3 and add 1?do",
        },
    ]
    response = call_litellm(
        model="openai/gpt-5.4-nano",
        messages=messages,
        reasoning_effort="low",
    )
    print(response)
