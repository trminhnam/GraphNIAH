"""Bounded, ordered process tasks with shutdown before caller recovery."""

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from itertools import islice
from multiprocessing import get_context
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def batches(items: Iterable[T], size: int) -> Iterator[list[T]]:
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


@contextmanager
def process_results(
    function: Callable[[T], R], items: Iterable[T], num_workers: int
) -> Iterator[Iterator[R]]:
    """Keep at most twice the worker count queued; never inherit SQLite handles."""
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if num_workers == 1:
        yield map(function, items)
        return
    executor = ProcessPoolExecutor(
        max_workers=num_workers, mp_context=get_context("spawn")
    )

    def results() -> Iterator[R]:
        iterator = iter(items)
        pending = deque(
            executor.submit(function, item)
            for item in islice(iterator, 2 * num_workers)
        )
        while pending:
            yield pending.popleft().result()
            for item in islice(iterator, 1):
                pending.append(executor.submit(function, item))

    try:
        yield results()
    finally:
        # Running tasks must finish before the caller restores any backups.
        executor.shutdown(wait=True, cancel_futures=True)
