"""Shared batching and rejection checkpoints for input-path checks."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.logging.console import progress_bar
from src.questions.explored import input_path_explored_flags

REJECTION_CHECKPOINT_SECONDS = 30.0


def input_path_id(entry: dict[str, Any]) -> str:
    """Return the stable id an input-path entry is keyed by."""
    return str(entry.get("path_id") or entry.get("path_index") or "")


@dataclass
class RejectionCheckpoint:
    """Save dirty rejection state periodically, with an explicit final flush."""

    write: Callable[[], None]
    last_write: float = field(default_factory=time.monotonic)
    dirty: bool = False

    def record(self) -> None:
        self.dirty = True
        if time.monotonic() - self.last_write >= REJECTION_CHECKPOINT_SECONDS:
            self.flush()

    def flush(self) -> None:
        if self.dirty:
            self.write()
            self.dirty = False
            self.last_write = time.monotonic()


def filter_input_paths(
    entries: Sequence[dict[str, Any]],
    *,
    paper_cutoff: Any,
    rejection_reason: Callable[..., str | None],
    desc: str,
    on_rejection: Callable[[dict[str, Any], str], None] | None = None,
    checkpoint: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    """Batch explored reads, preserve order, and save rejections on exit."""
    eligible = []
    with progress_bar(
        total=len(entries), desc=desc, unit="path", mininterval=1.0
    ) as progress:
        progress.set_postfix_str("loading explored flags")
        explored_flags = input_path_explored_flags(entries, paper_cutoff.db)
        progress.set_postfix_str("checking paths")
        try:
            for entry in entries:
                reason = rejection_reason(
                    entry, paper_cutoff, explored_flags=explored_flags
                )
                if reason is None:
                    eligible.append(entry)
                elif on_rejection is not None:
                    on_rejection(entry, reason)
                progress.update()
        finally:
            if checkpoint is not None:
                checkpoint()
    return eligible
