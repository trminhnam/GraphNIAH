"""
Loguru logging configuration for the benchmark pipeline.

All modules should use ``from loguru import logger`` directly.
Entry-point scripts must call ``configure_logging()`` once at startup so that:
  - loguru is configured with the project format and colours, and
  - *every* stdlib ``logging`` call — from our code, imported modules, and
    third-party libraries — is transparently forwarded through loguru.

How global interception works
------------------------------
Python's logging machinery is hierarchical.  Messages propagate upward from
child loggers to the root logger unless ``propagate=False`` is set on a child.
The root logger is the single choke-point, so installing ``InterceptHandler``
there catches everything — **provided** child loggers haven't been given their
own private handlers or had propagation disabled.

Many popular libraries (``requests``, ``httpx``, ``arxiv``, …) only call
``logging.getLogger(__name__)`` and never attach handlers themselves, so they
already propagate to root.  A minority of libraries do attach handlers or set
``propagate = False`` before our code runs.  ``configure_logging()`` handles both
cases:

1. Root handler — ``InterceptHandler`` replaces all root handlers (via
   ``force=True``), so every new logger created afterwards reaches loguru.
2. Existing child loggers — the full logger registry
   (``logging.root.manager.loggerDict``) is walked at startup: any private
   handlers are removed and ``propagate`` is restored to ``True``.  This
   covers libraries that were imported before ``configure_logging()`` was called.
"""

from __future__ import annotations

import logging
import sys
from typing import Callable

from loguru import logger
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Log format
# ---------------------------------------------------------------------------

# "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
_FORMAT = (
    "<green>{time:YY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)


def _write_console_message(message: str) -> None:
    if sys.stderr.isatty():
        tqdm.write(message, end="", file=sys.stderr)
        return

    sys.stderr.write(message)
    sys.stderr.flush()

# ---------------------------------------------------------------------------
# stdlib → loguru bridge
# ---------------------------------------------------------------------------


class InterceptHandler(logging.Handler):
    """Redirect every stdlib ``logging`` record into loguru.

    Installed on the root stdlib logger so that all child loggers whose
    messages propagate to root are automatically forwarded to loguru.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk the call stack to find the real caller (not the logging module
        # itself), so loguru reports the correct file / function / line.
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(
    level: str = "INFO",
    *,
    record_filter: Callable[[dict], bool] | None = None,
) -> None:
    """Configure loguru and intercept *all* stdlib logging globally.

    This is safe to call multiple times (idempotent): each call fully
    replaces the previous loguru sink and re-walks the logger registry.

    Parameters
    ----------
    level:
        Minimum log level for console output.  Accepts any loguru / stdlib
        level name: ``"TRACE"``, ``"DEBUG"``, ``"INFO"`` (default),
        ``"SUCCESS"``, ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``.
    record_filter:
        Optional loguru-style filter: a callable receiving the record dict
        and returning ``False`` to drop that record from the console sink.
        Default ``None`` keeps every record at or above *level* (unchanged
        behaviour); callers that need to silence a specific noisy message
        can pass their own predicate without affecting other entry points.
    """
    # ------------------------------------------------------------------
    # 1. Configure loguru
    # ------------------------------------------------------------------
    logger.remove()
    logger.add(
        _write_console_message,
        format=_FORMAT,
        level=level,
        colorize=sys.stderr.isatty(),
        backtrace=True,
        diagnose=True,
        filter=record_filter,
    )

    # ------------------------------------------------------------------
    # 2. Install InterceptHandler on the root stdlib logger.
    #    force=True removes any existing root handlers first (Python ≥ 3.8).
    # ------------------------------------------------------------------
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    logging.captureWarnings(True)

    # ------------------------------------------------------------------
    # 3. Walk every named logger that already exists in the registry.
    #
    #    Libraries imported *before* this function is called may have
    #    attached handlers directly to their own loggers or set
    #    propagate=False.  Either prevents messages reaching the root
    #    InterceptHandler.  We strip private handlers and re-enable
    #    propagation so everything flows through loguru.
    # ------------------------------------------------------------------
    for name in list(logging.root.manager.loggerDict):
        child = logging.getLogger(name)
        for handler in child.handlers[:]:
            child.removeHandler(handler)
        child.propagate = True
        child.disabled = False

    # ------------------------------------------------------------------
    # 4. Quieten known verbose third-party loggers at WARNING+ unless the
    #    caller explicitly asked for DEBUG/TRACE output.
    # ------------------------------------------------------------------
    if level not in ("TRACE", "DEBUG"):
        _QUIET = (
            "urllib3",
            "requests",
            "httpx",
            "httpcore",
            "hpack",
            "h2",
            "h11",
            "LiteLLM",
            "LiteLLM Router",
            "LiteLLM Proxy",
        )
        for name in _QUIET:
            logging.getLogger(name).setLevel(logging.WARNING)
