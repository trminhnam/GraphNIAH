"""Console helpers shared by the pipeline stage scripts.

Pipeline stages print their summaries to stdout while ``tqdm`` draws progress
bars on stderr. A plain ``print`` collides with a live bar: the bar redraws with
a carriage return, which only returns to the start of the last terminal row, so
it overwrites the middle of any long line that shares that row. Routing every
message through :func:`log_line` hands the write to ``tqdm``, which clears the
bars first and redraws them afterwards.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

PATH_ELLIPSIS = "..."
DEFAULT_PATH_HEAD_PARTS = 3
DEFAULT_PATH_TAIL_PARTS = 4

PROGRESS_QUIET = False


def set_progress_quiet(quiet: bool) -> None:
    """Silence every progress bar this process creates.

    Spawned workers share the parent's terminal, so bars drawn by several
    workers at once interleave into an unreadable stream of repeats. Only
    the parent, which knows the whole job, draws a bar.
    """
    global PROGRESS_QUIET
    PROGRESS_QUIET = quiet


def progress_bar(*args: Any, **kwargs: Any) -> tqdm:
    """Create a progress bar that draws nothing in a silenced process."""
    return tqdm(*args, disable=PROGRESS_QUIET, **kwargs)


def log_line(message: str = "") -> None:
    """Print one message to stdout without breaking a live progress bar."""
    sys.stderr.flush()
    tqdm.write(str(message), file=sys.stdout)
    sys.stdout.flush()


def startup_logger(label: str) -> Callable[[str], None]:
    """Create a flushed startup logger with elapsed time for one run."""
    started = time.monotonic()

    def log_startup(message: str) -> None:
        elapsed = time.monotonic() - started
        log_line(f"[{label} startup +{elapsed:.1f}s] {message}")

    return log_startup


def format_display_path(
    path: str | Path,
    *,
    head_parts: int = DEFAULT_PATH_HEAD_PARTS,
    tail_parts: int = DEFAULT_PATH_TAIL_PARTS,
) -> str:
    """Return one path with its middle directories elided for logging.

    Experiment paths repeat every condition key, so the full string is several
    hundred characters and wraps over multiple terminal rows. The head names the
    experiment and the tail carries the condition that actually varies.
    """
    parts = Path(path).parts
    if len(parts) <= head_parts + tail_parts:
        return str(path)

    head = str(Path(*parts[:head_parts]))
    tail = str(Path(*parts[-tail_parts:]))
    return f"{head}/{PATH_ELLIPSIS}/{tail}"
