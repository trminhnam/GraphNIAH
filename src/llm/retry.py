"""Shared retry policy for LLM inference requests."""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import random
from typing import Any

import litellm
import requests


# Edit these constants to change the inference retry policy.
LLM_MAX_ATTEMPTS = 5
LLM_INITIAL_BACKOFF_SECONDS = 2.0
LLM_BACKOFF_MULTIPLIER = 2.0
LLM_MAX_BACKOFF_SECONDS = 60.0
LLM_BACKOFF_JITTER_SECONDS = 1.0
RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


class InvalidCompletionResponseError(RuntimeError):
    """Raised when an LLM response has no usable assistant content."""


class CompletionRequestError(RuntimeError):
    """Raised after a completion request cannot produce a valid result."""

    def __init__(
        self,
        attempts: int,
        last_error: Exception,
        failed_attempts: list[dict[str, Any]] | None = None,
    ) -> None:
        self.attempts = attempts
        self.last_error = last_error
        self.failed_attempts = failed_attempts or []
        super().__init__(summarize_llm_error(last_error))


def get_http_status_code(error: Exception) -> int | None:
    """Return an HTTP status code carried by an exception, when available."""
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    try:
        return int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        return None


def read_retry_after_seconds(
    error: Exception,
    *,
    now: datetime | None = None,
) -> float | None:
    """Return the server's Retry-After delay in seconds, when present."""
    headers: Any = getattr(error, "headers", None)
    if not headers:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
    if not headers:
        return None

    retry_after = headers.get("Retry-After")
    if retry_after is None:
        retry_after = headers.get("retry-after")
    if retry_after is None:
        return None

    try:
        return max(0.0, float(retry_after))
    except (TypeError, ValueError):
        pass

    try:
        retry_at = parsedate_to_datetime(str(retry_after))
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    current_time = now or datetime.now(timezone.utc)
    return max(0.0, (retry_at - current_time).total_seconds())


def calculate_retry_delay(
    attempt: int,
    retry_after_seconds: float | None = None,
) -> float:
    """Return the delay after one failed attempt."""
    if retry_after_seconds is not None:
        return max(0.0, retry_after_seconds)

    exponent = max(0, attempt - 1)
    base_delay = min(
        LLM_INITIAL_BACKOFF_SECONDS * (LLM_BACKOFF_MULTIPLIER**exponent),
        LLM_MAX_BACKOFF_SECONDS,
    )
    jitter = random.uniform(0.0, LLM_BACKOFF_JITTER_SECONDS)
    return min(base_delay + jitter, LLM_MAX_BACKOFF_SECONDS)


def is_retryable_llm_error(error: Exception) -> bool:
    """Return whether an inference failure is safe to retry."""
    if isinstance(error, InvalidCompletionResponseError):
        return True

    if isinstance(error, requests.HTTPError):
        return get_http_status_code(error) in RETRYABLE_HTTP_STATUS_CODES
    if isinstance(error, (requests.Timeout, requests.ConnectionError)):
        return True

    if isinstance(error, (litellm.AuthenticationError, litellm.BadRequestError)):
        return False
    if isinstance(
        error,
        (
            litellm.RateLimitError,
            litellm.Timeout,
            litellm.APIConnectionError,
            litellm.InternalServerError,
        ),
    ):
        return True

    return get_http_status_code(error) in RETRYABLE_HTTP_STATUS_CODES


def summarize_llm_error(error: Exception) -> str:
    """Return a short error reason without request messages or credentials."""
    error_name = type(error).__name__
    status_code = get_http_status_code(error)
    if status_code is not None:
        return f"{error_name} (HTTP {status_code})"
    if isinstance(error, InvalidCompletionResponseError):
        return f"{error_name}: {error}"
    return error_name


__all__ = [
    "CompletionRequestError",
    "InvalidCompletionResponseError",
    "LLM_BACKOFF_JITTER_SECONDS",
    "LLM_BACKOFF_MULTIPLIER",
    "LLM_INITIAL_BACKOFF_SECONDS",
    "LLM_MAX_ATTEMPTS",
    "LLM_MAX_BACKOFF_SECONDS",
    "RETRYABLE_HTTP_STATUS_CODES",
    "calculate_retry_delay",
    "get_http_status_code",
    "is_retryable_llm_error",
    "read_retry_after_seconds",
    "summarize_llm_error",
]
