from __future__ import annotations

from typing import Callable

from src.logging import configure_logging


def bootstrap_runtime(
    log_level: str = "INFO",
    *,
    setup_litellm_runtime: bool = True,
    log_filter: Callable[[dict], bool] | None = None,
) -> None:
    """Configure shared runtime services for an entrypoint."""
    from .suppress_warnings import suppress_warnings

    # Import LiteLLM before configure_logging(): LiteLLM attaches its own
    # DEBUG console handler on import, and configure_logging() only strips
    # handlers from loggers that already exist.
    if setup_litellm_runtime:
        from .litellm import setup_litellm

        setup_litellm()
    configure_logging(log_level, record_filter=log_filter)
    suppress_warnings()
