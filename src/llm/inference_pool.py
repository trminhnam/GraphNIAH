"""Shared inference responses, indexed by explicit request parameter values."""

from __future__ import annotations

import fcntl
import os
import shutil
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, BinaryIO

import orjson

from src.questions.dataset_pool import append_record, read_jsonl

POOL_SCHEMA_VERSION = 1


def successful_response(record: dict[str, Any]) -> bool:
    answer = record.get("raw_answer")
    return isinstance(answer, str) and bool(answer.strip()) and not record.get("error")


def key_bytes(key: dict[str, Any]) -> bytes:
    """Canonical JSON for the in-memory index; no key is hashed."""
    return orjson.dumps(key, option=orjson.OPT_SORT_KEYS)


def validate_pool_entry(entry: dict[str, Any]) -> None:
    """Validate the fields required to reuse a cached response."""
    if entry.get("schema_version") != POOL_SCHEMA_VERSION:
        raise ValueError("Not a supported inference-pool record.")
    key = entry.get("key")
    if not isinstance(key, dict) or any(
        not isinstance(key.get(field), str) or not key[field]
        for field in ("qid", "model", "reasoning_effort")
    ):
        raise ValueError("Inference-pool key needs qid, model, and reasoning_effort.")
    response = entry.get("response")
    if not isinstance(response, dict) or not successful_response(response):
        raise ValueError("Inference pool must contain successful responses only.")


@dataclass
class InferencePool:
    handle: BinaryIO
    offsets: dict[bytes, int] = field(default_factory=dict)

    def scan(self) -> None:
        for offset, entry in read_jsonl(self.handle):
            validate_pool_entry(entry)
            key = key_bytes(entry["key"])
            if key in self.offsets:
                raise ValueError("Duplicate request key in inference pool.")
            self.offsets[key] = offset

    def get(self, key: dict[str, Any]) -> dict[str, Any] | None:
        offset = self.offsets.get(key_bytes(key))
        if offset is None:
            return None
        self.handle.seek(offset)
        return orjson.loads(self.handle.readline())["response"]

    def add(
        self, key: dict[str, Any], response: dict[str, Any], source_output: str
    ) -> None:
        """Keep the first successful response; errors remain in experiment outputs."""
        index = key_bytes(key)
        if index in self.offsets or not successful_response(response):
            return
        self.offsets[index] = append_record(
            self.handle,
            {
                "schema_version": POOL_SCHEMA_VERSION,
                "key": key,
                "response": response,
                "source_output": source_output,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            },
        )


@contextmanager
def open_inference_pool(path: Path) -> Iterator[InferencePool]:
    """Hold the pool lock throughout inference to prevent duplicate paid requests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                f"Another process is writing this inference pool ({path}). "
                "Retry when it finishes."
            ) from None
        pool = InferencePool(handle)
        pool.scan()
        yield pool


def read_merge_entries(handle: BinaryIO) -> Iterator[dict[str, Any]]:
    """Read a source strictly; never repair or silently skip an unfinished entry."""
    handle.seek(0)
    seen = set()
    for line_number, line in enumerate(handle, start=1):
        try:
            entry = orjson.loads(line)
            if not isinstance(entry, dict):
                raise TypeError("Expected an object.")
            validate_pool_entry(entry)
            key = key_bytes(entry["key"])
            if key in seen:
                raise ValueError("Duplicate request key in inference pool.")
        except (ValueError, TypeError) as error:
            raise ValueError(f"{handle.name}, line {line_number}: {error}") from error
        seen.add(key)
        yield entry


def merge_inference_pools(
    sources: list[Path], target: Path, *, copy_back: bool = False
) -> dict[str, int]:
    """Merge entries, optionally replacing/creating sources with the merged pool."""
    if not sources:
        raise ValueError("At least one source pool is required.")
    sources = [path.resolve() for path in sources]
    target = target.resolve()
    for first, second in combinations([*sources, target], 2):
        if first == second or (
            first.exists() and second.exists() and first.samefile(second)
        ):
            raise ValueError("Source pools and target must be distinct files.")
    with ExitStack() as stack:
        handles = []
        for source in sources:
            if copy_back:
                source.parent.mkdir(parents=True, exist_ok=True)
                # Open or create without truncation or append-only writes.
                descriptor = os.open(source, os.O_RDWR | os.O_CREAT, 0o666)
                handle = stack.enter_context(os.fdopen(descriptor, "r+b"))
            else:
                handle = stack.enter_context(source.open("rb"))
            try:
                lock = fcntl.LOCK_EX if copy_back else fcntl.LOCK_SH
                fcntl.flock(handle, lock | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(
                    f"Inference pool is in use ({source}). Retry when inference finishes."
                ) from None
            handles.append(handle)
        # Validate every source before creating or appending to the target.
        for handle in handles:
            for _entry in read_merge_entries(handle):
                pass
        pool = stack.enter_context(open_inference_pool(target))
        summary = {"read": 0, "added": 0, "skipped": 0, "target_total": 0}
        for handle in handles:
            for entry in read_merge_entries(handle):
                summary["read"] += 1
                key = key_bytes(entry["key"])
                if key in pool.offsets:
                    summary["skipped"] += 1
                    continue
                pool.offsets[key] = append_record(pool.handle, entry)
                summary["added"] += 1
        summary["target_total"] = len(pool.offsets)
        if copy_back:
            # Persist the full union before overwriting any source. Keep the
            # same file inodes so inference's file locks remain effective.
            pool.handle.flush()
            os.fsync(pool.handle.fileno())
            for handle in handles:
                pool.handle.seek(0)
                handle.seek(0)
                shutil.copyfileobj(pool.handle, handle)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
            summary["copied_back"] = len(handles)
        return summary
