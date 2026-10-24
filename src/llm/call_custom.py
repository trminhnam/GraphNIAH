"""Call a custom chat-completions endpoint without LiteLLM."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
import os
from threading import Lock
import time
from typing import Any

import requests


CUSTOM_CHAT_COMPLETIONS_URL = os.getenv("CUSTOM_CHAT_COMPLETIONS_URL", "")
CUSTOM_MODEL_PREFIX = "custom/"

# Edit these constants if Custom endpoint changes its request limit.
CUSTOM_REQUESTS_PER_MINUTE = 20
CUSTOM_RATE_LIMIT_WINDOW_SECONDS = 60.0

custom_request_timestamps: deque[float] = deque()
custom_rate_limit_lock = Lock()


def reset_custom_rate_limit() -> None:
    """Clear this process's CUSTOM request history."""
    with custom_rate_limit_lock:
        custom_request_timestamps.clear()


def wait_for_custom_rate_limit() -> None:
    """Wait until one request fits inside the CUSTOM rolling window."""
    if CUSTOM_REQUESTS_PER_MINUTE <= 0:
        raise ValueError("CUSTOM_REQUESTS_PER_MINUTE must be greater than zero")
    if CUSTOM_RATE_LIMIT_WINDOW_SECONDS <= 0:
        raise ValueError("CUSTOM_RATE_LIMIT_WINDOW_SECONDS must be greater than zero")

    while True:
        with custom_rate_limit_lock:
            now = time.monotonic()
            cutoff = now - CUSTOM_RATE_LIMIT_WINDOW_SECONDS
            while custom_request_timestamps and custom_request_timestamps[0] <= cutoff:
                custom_request_timestamps.popleft()

            if len(custom_request_timestamps) < CUSTOM_REQUESTS_PER_MINUTE:
                custom_request_timestamps.append(now)
                return

            wait_seconds = CUSTOM_RATE_LIMIT_WINDOW_SECONDS - (
                now - custom_request_timestamps[0]
            )

        if wait_seconds > 0:
            time.sleep(wait_seconds)


def call_llm_custom(
    messages: list[dict[str, Any]],
    model: str = "llama3.1:latest",
    api_key: str | None = None,
    timeout: float = 120,
    on_send: Callable[[], None] | None = None,
    **body_kwargs: Any,
) -> requests.Response:
    """Call CUSTOM chat completions and return the HTTP response.

    Extra keyword arguments, including ``reasoning_effort``, are sent in the
    request body. The routing-only ``custom/`` model prefix is removed before the
    request is sent. ``on_send`` runs after the rate-limit wait, right before
    the request goes out, so callers can time the request alone.
    """
    if not CUSTOM_CHAT_COMPLETIONS_URL:
        raise ValueError("Set CUSTOM_CHAT_COMPLETIONS_URL for the custom backend")
    if api_key is None:
        api_key = os.getenv("CUSTOM_API_KEY")
    if not api_key:
        raise ValueError("CUSTOM_API_KEY must be set when api_key is not provided")

    request_model = model.removeprefix(CUSTOM_MODEL_PREFIX)
    if not request_model:
        raise ValueError("CUSTOM model name must not be empty")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": request_model,
        "messages": messages,
        **body_kwargs,
    }
    wait_for_custom_rate_limit()
    if on_send is not None:
        on_send()
    response = requests.post(
        CUSTOM_CHAT_COMPLETIONS_URL,
        headers=headers,
        json=body,
        timeout=timeout,
    )
    if response is None:
        raise requests.ConnectionError("CUSTOM returned a null HTTP response")
    if response.status_code != 200:
        raise requests.HTTPError(
            f"CUSTOM error: {response.status_code}, {response.text}",
            response=response,
        )
    return response


async def call_llm_custom_async(*args: Any, **kwargs: Any) -> requests.Response:
    """Run ``call_llm_custom`` without blocking the async event loop."""
    return await asyncio.to_thread(call_llm_custom, *args, **kwargs)


__all__ = [
    "CUSTOM_RATE_LIMIT_WINDOW_SECONDS",
    "CUSTOM_REQUESTS_PER_MINUTE",
    "call_llm_custom",
    "call_llm_custom_async",
    "reset_custom_rate_limit",
    "wait_for_custom_rate_limit",
]
