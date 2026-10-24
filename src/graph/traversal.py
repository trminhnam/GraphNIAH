"""Citation-graph traversal strategies.

Threading model for the parallel traversals
-------------------------------------------
``expand_graph_parallel`` and ``expand_graph_parallel_priority`` follow a
**single-driver-thread** design:

* Only the main thread mutates the shared queue, the ``claimed`` /
  ``in_flight_*`` sets, and the ``CitationNetwork`` (graph + DB).
* Worker threads run ``fetch_paper_data`` only — they download PDFs, parse
  references / authors, and return a result bundle. They never touch the
  shared graph state.

The ``lock`` defined inside each parallel function therefore guards
**graph + DB mutations performed from the driver thread**. It exists to make
the contract explicit and to keep checkpoint saves consistent with the
neighbour-enqueue step. Worker code does not acquire this lock because workers
do not touch the protected state. Any future refactor that moves graph or DB
access into a worker thread MUST take ``lock`` for those calls.
"""

from __future__ import annotations

import heapq
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from itertools import count
from pathlib import Path
from typing import Any

from loguru import logger

from src.setup.constants import nid_db_id, normalize_text

from .checkpoints import build_graph_stem, extract_base_stem, prune_checkpoints
from .expansion import (
    PaperExpander,
    apply_fetched_paper,
    enqueue_citation_neighbors,
    fetch_paper_data,
    get_paper_depth,
)
from .network import CitationNetwork
from .persistence import save_graph


class DepthPriorityQueue:
    """Queue API backed by a heap that pops the smallest depth first."""

    def __init__(self) -> None:
        self.heap: list[tuple[int, int, str]] = []
        self.counter = count()

    def append(self, item: tuple[str, int]) -> None:
        nid, depth = item
        heapq.heappush(self.heap, (depth, next(self.counter), nid))

    def popleft(self) -> tuple[str, int]:
        depth, _, nid = heapq.heappop(self.heap)
        return nid, depth

    def __bool__(self) -> bool:
        return bool(self.heap)

    def __len__(self) -> int:
        return len(self.heap)


class ThreadLocalPaperParsers:
    """Provide parser instances scoped to each worker thread."""

    def __init__(
        self,
        pdf_parser: Any,
        author_parser: Any,
        ref_parser: Any,
        tex_ref_parser: Any,
        *,
        pdf_parser_factory: Callable[[], Any] | None = None,
        author_parser_factory: Callable[[], Any] | None = None,
        ref_parser_factory: Callable[[], Any] | None = None,
        tex_ref_parser_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.local = threading.local()
        self.pdf_parser = pdf_parser
        self.author_parser = author_parser
        self.ref_parser = ref_parser
        self.tex_ref_parser = tex_ref_parser
        self.pdf_parser_factory = pdf_parser_factory
        self.author_parser_factory = author_parser_factory
        self.ref_parser_factory = ref_parser_factory
        self.tex_ref_parser_factory = tex_ref_parser_factory

    def get_parsers(self) -> tuple[Any, Any, Any, Any]:
        parsers = getattr(self.local, "parsers", None)
        if parsers is None:
            parsers = (
                self.pdf_parser_factory()
                if self.pdf_parser_factory is not None
                else self.pdf_parser,
                self.author_parser_factory()
                if self.author_parser_factory is not None
                else self.author_parser,
                self.ref_parser_factory()
                if self.ref_parser_factory is not None
                else self.ref_parser,
                self.tex_ref_parser_factory()
                if self.tex_ref_parser_factory is not None
                else self.tex_ref_parser,
            )
            self.local.parsers = parsers
        return parsers


def has_exploration_capacity(
    network: CitationNetwork,
    *,
    pending_count: int = 0,
) -> bool:
    """Return whether another paper expansion can be started.

    In parallel mode, pending workers reserve possible explored-paper slots so
    ``--max-papers`` is not exceeded just because several workers finish at
    once.
    """
    max_papers = network.limits.max_papers
    if not max_papers:
        return True
    return network.explored_count() + pending_count < max_papers


def expand_bfs_node(
    expander: PaperExpander,
    nid: str,
    depth: int,
) -> tuple[str, bool, bool, bool]:
    """Attempt to expand one BFS node.

    Returns ``(canonical_nid, success, was_new, capacity_skip)``.
    """
    db_id = nid_db_id(nid)
    if expander.net.is_explored(db_id):
        return nid, True, False, False

    row = expander.net.db.get_paper(db_id)
    arxiv_id = row["arxiv_id"] if row else ""
    title = row["title"] if row else ""

    if not arxiv_id and not title:
        logger.debug("Node {} has no arxiv_id or title; skipping.", nid)
        expander.net.mark_explored(nid)
        return nid, True, False, False

    explored_count_before = expander.net.explored_count()
    try:
        result_nid = (
            expander.expand_by_arxiv_id(arxiv_id, depth=depth)
            if arxiv_id
            else expander.expand_by_title(title, depth=depth)
        )
    except Exception as exc:
        logger.warning(
            "Expansion of node {} raised an unexpected exception; leaving unexplored for retry: {}",
            nid,
            exc,
        )
        if expander.raise_error:
            raise
        return nid, False, False, False

    if result_nid is None:
        if expander.net.at_paper_capacity():
            logger.debug(
                "Expansion skipped for {} due to capacity; not marking explored.",
                nid,
            )
            return nid, False, False, True
        logger.debug(
            "Expansion returned None for {} (transient failure); not marking explored.",
            nid,
        )
        return nid, False, False, False

    was_new = expander.net.explored_count() > explored_count_before
    if nid != result_nid:
        expander.net.mark_explored(nid)
    return result_nid, True, was_new, False


def expand_graph(
    expander: PaperExpander,
    seeds: str | list[str],
    *,
    max_depth: int | None = 1,
    graph_dir: Path | None = None,
    run_stem: str = "citation_network",
    checkpoint_every: int = 0,
    max_num_checkpoints: int = 2,
) -> int:
    """Expand the citation graph via BFS up to ``max_depth`` hops."""
    if isinstance(seeds, str):
        seeds = [seeds]

    queue: deque[tuple[str, int]] = deque()
    visited: set[str] = set()
    for nid in seeds:
        if nid in visited:
            continue
        visited.add(nid)
        queue.append((nid, get_paper_depth(expander.net.db, nid)))

    if not queue:
        logger.info("expand_graph: no seeds provided; nothing to do.")
        return 0

    logger.info(
        "Starting {} BFS expansion; max_depth: {}",
        "multi-seed" if len(queue) > 1 else "seeded",
        max_depth,
    )

    newly_explored = 0
    partial_failures = 0
    # Cadence counter for --checkpoint-every; only newly-explored papers count.
    papers_since_checkpoint = 0

    while queue:
        nid, depth = queue.popleft()
        if expander.net.at_paper_capacity():
            logger.info(
                "Explored-paper capacity ({}) reached; stopping BFS.",
                expander.net.limits.max_papers,
            )
            break

        canonical_nid, success, was_new, capacity_skip = expand_bfs_node(
            expander,
            nid,
            depth,
        )
        if not success:
            if not capacity_skip:
                partial_failures += 1
            continue

        if was_new:
            newly_explored += 1
            papers_since_checkpoint += 1

        if (
            checkpoint_every > 0
            and graph_dir is not None
            and papers_since_checkpoint >= checkpoint_every
        ):
            ckpt_stem = build_graph_stem(run_stem)
            logger.info(
                "BFS checkpoint; {} papers explored so far, saving {}",
                newly_explored,
                ckpt_stem,
            )
            save_graph(expander.net, graph_dir, stem=ckpt_stem, write_gexf=False)
            prune_checkpoints(
                graph_dir,
                base_stem=extract_base_stem(ckpt_stem),
                max_keep=max_num_checkpoints,
            )
            papers_since_checkpoint = 0

        visited.add(canonical_nid)
        enqueue_citation_neighbors(
            expander.net,
            canonical_nid,
            depth,
            max_depth,
            visited,
            queue,
        )

        logger.debug(
            "BFS queue size: {}, visited: {}, depth: {}/{}",
            len(queue),
            len(visited),
            depth,
            max_depth,
        )

    if partial_failures:
        logger.info(
            "BFS expansion done; {} papers newly explored, {} failures | {}",
            newly_explored,
            partial_failures,
            expander.net.stats(),
        )
    else:
        logger.info(
            "BFS expansion done; {} papers newly explored | {}",
            newly_explored,
            expander.net.stats(),
        )
    return newly_explored


def expand_graph_priority(
    expander: PaperExpander,
    seeds: str | list[str],
    *,
    max_depth: int | None = 1,
    graph_dir: Path | None = None,
    run_stem: str = "citation_network",
    checkpoint_every: int = 0,
    max_num_checkpoints: int = 2,
) -> int:
    """Expand the citation graph, always selecting the smallest-depth node next."""
    if isinstance(seeds, str):
        seeds = [seeds]

    queue = DepthPriorityQueue()
    visited: set[str] = set()
    for nid in seeds:
        if nid in visited:
            continue
        visited.add(nid)
        queue.append((nid, get_paper_depth(expander.net.db, nid)))

    if not queue:
        logger.info("expand_graph_priority: no seeds provided; nothing to do.")
        return 0

    logger.info(
        "Starting {} priority expansion; max_depth: {}",
        "multi-seed" if len(queue) > 1 else "seeded",
        max_depth,
    )

    newly_explored = 0
    partial_failures = 0
    papers_since_checkpoint = 0

    while queue:
        nid, depth = queue.popleft()
        if expander.net.at_paper_capacity():
            logger.info(
                "Explored-paper capacity ({}) reached; stopping priority expansion.",
                expander.net.limits.max_papers,
            )
            break

        canonical_nid, success, was_new, capacity_skip = expand_bfs_node(
            expander,
            nid,
            depth,
        )
        if not success:
            if not capacity_skip:
                partial_failures += 1
            continue

        if was_new:
            newly_explored += 1
            papers_since_checkpoint += 1

        if (
            checkpoint_every > 0
            and graph_dir is not None
            and papers_since_checkpoint >= checkpoint_every
        ):
            ckpt_stem = build_graph_stem(run_stem)
            logger.info(
                "Priority checkpoint; {} papers explored so far, saving {}",
                newly_explored,
                ckpt_stem,
            )
            save_graph(expander.net, graph_dir, stem=ckpt_stem, write_gexf=False)
            prune_checkpoints(
                graph_dir,
                base_stem=extract_base_stem(ckpt_stem),
                max_keep=max_num_checkpoints,
            )
            papers_since_checkpoint = 0

        visited.add(canonical_nid)
        enqueue_citation_neighbors(
            expander.net,
            canonical_nid,
            depth,
            max_depth,
            visited,
            queue,
        )

        logger.debug(
            "Priority queue size: {}, visited: {}, depth: {}/{}",
            len(queue),
            len(visited),
            depth,
            max_depth,
        )

    if partial_failures:
        logger.info(
            "Priority expansion done; {} papers newly explored, {} failures | {}",
            newly_explored,
            partial_failures,
            expander.net.stats(),
        )
    else:
        logger.info(
            "Priority expansion done; {} papers newly explored | {}",
            newly_explored,
            expander.net.stats(),
        )
    return newly_explored


def expand_graph_parallel(
    network: CitationNetwork,
    seeds: str | list[str],
    *,
    max_depth: int | None = 1,
    workers: int = 4,
    arxiv_client: Any,
    pdf_parser: Any,
    author_parser: Any,
    ref_parser: Any,
    tex_ref_parser: Any,
    pdf_dir: Path,
    text_dir: Path,
    source_dir: Path,
    pdf_parser_factory: Callable[[], Any] | None = None,
    author_parser_factory: Callable[[], Any] | None = None,
    ref_parser_factory: Callable[[], Any] | None = None,
    tex_ref_parser_factory: Callable[[], Any] | None = None,
    graph_dir: Path | None = None,
    run_stem: str = "citation_network",
    checkpoint_every: int = 0,
    max_num_checkpoints: int = 2,
    dblp_client: Any = None,
    semantic_scholar_client: Any = None,
    resolve_et_al_authors: bool = False,
    raise_error: bool = False,
) -> int:
    """Expand the citation graph via parallel BFS up to ``max_depth`` hops."""
    if isinstance(seeds, str):
        seeds = [seeds]

    lock = threading.Lock()
    queue: deque[tuple[str, int]] = deque()
    claimed: set[str] = set()
    in_flight_arxiv_ids: set[str] = set()
    in_flight_titles: set[str] = set()
    file_locks_mu = threading.Lock()
    file_locks: dict[str, threading.Lock] = {}

    def get_file_lock(key: str) -> threading.Lock:
        with file_locks_mu:
            if key not in file_locks:
                file_locks[key] = threading.Lock()
            return file_locks[key]

    parser_provider = ThreadLocalPaperParsers(
        pdf_parser,
        author_parser,
        ref_parser,
        tex_ref_parser,
        pdf_parser_factory=pdf_parser_factory,
        author_parser_factory=author_parser_factory,
        ref_parser_factory=ref_parser_factory,
        tex_ref_parser_factory=tex_ref_parser_factory,
    )

    def fetch_with_thread_parsers(
        arxiv_id: str,
        title: str,
        nid: str,
        depth: int,
    ) -> Any:
        (
            worker_pdf_parser,
            worker_author_parser,
            worker_ref_parser,
            worker_tex_ref_parser,
        ) = parser_provider.get_parsers()
        return fetch_paper_data(
            arxiv_id,
            title,
            nid,
            depth,
            arxiv_client,
            worker_pdf_parser,
            worker_author_parser,
            worker_ref_parser,
            worker_tex_ref_parser,
            pdf_dir,
            text_dir,
            source_dir,
            get_file_lock,
            dblp_client,
            semantic_scholar_client,
            raise_error,
            resolve_et_al_authors=resolve_et_al_authors,
        )

    for nid in seeds:
        if nid in claimed:
            continue
        claimed.add(nid)
        queue.append((nid, get_paper_depth(network.db, nid)))

    if not queue:
        logger.info("expand_graph_parallel: no seeds; nothing to do.")
        return 0

    logger.info(
        "Starting parallel BFS; {} seed(s), max_depth: {}, workers: {}",
        len(queue),
        max_depth,
        workers,
    )

    newly_explored = 0
    partial_failures = 0
    papers_since_checkpoint = 0

    def try_submit_one(
        pool: ThreadPoolExecutor,
        pending: dict[Future, tuple[str, int, str, str]],
    ) -> bool:
        deferred = 0
        queue_len_at_entry = len(queue)

        while queue:
            if not has_exploration_capacity(network, pending_count=len(pending)):
                return False

            nid, depth = queue.popleft()
            if network.is_explored(nid_db_id(nid)):
                with lock:
                    enqueue_citation_neighbors(
                        network,
                        nid,
                        depth,
                        max_depth,
                        claimed,
                        queue,
                    )
                continue

            row = network.db.get_paper(nid_db_id(nid))
            arxiv_id = row["arxiv_id"] if row else ""
            title = row["title"] if row else ""
            if not arxiv_id and not title:
                with lock:
                    network.mark_explored(nid)
                continue

            norm_title = normalize_text(title) if title else ""
            if arxiv_id and arxiv_id in in_flight_arxiv_ids:
                queue.append((nid, depth))
                deferred += 1
                if deferred >= queue_len_at_entry:
                    return False
                continue
            if not arxiv_id and norm_title and norm_title in in_flight_titles:
                queue.append((nid, depth))
                deferred += 1
                if deferred >= queue_len_at_entry:
                    return False
                continue

            if arxiv_id:
                in_flight_arxiv_ids.add(arxiv_id)
            if not arxiv_id and norm_title:
                in_flight_titles.add(norm_title)

            future = pool.submit(
                fetch_with_thread_parsers,
                arxiv_id,
                title,
                nid,
                depth,
            )
            pending[future] = (nid, depth, arxiv_id, norm_title)
            return True

        return False

    def fill_pool(
        pool: ThreadPoolExecutor,
        pending: dict[Future, tuple[str, int, str, str]],
    ) -> None:
        while len(pending) < workers:
            if not try_submit_one(pool, pending):
                break

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: dict[Future, tuple[str, int, str, str]] = {}
        fill_pool(pool, pending)

        while pending:
            done = wait(pending, return_when=FIRST_COMPLETED).done
            for future in done:
                nid, depth, fut_arxiv_id, fut_norm_title = pending.pop(future)
                if fut_arxiv_id:
                    in_flight_arxiv_ids.discard(fut_arxiv_id)
                if fut_norm_title:
                    in_flight_titles.discard(fut_norm_title)

                try:
                    bundle = future.result()
                except Exception as exc:
                    logger.warning(
                        "Worker for {} raised; leaving unexplored for retry: {}",
                        nid,
                        exc,
                    )
                    if raise_error:
                        for queued_future in list(pending):
                            queued_future.cancel()
                        raise
                    partial_failures += 1
                    fill_pool(pool, pending)
                    continue

                try:
                    with lock:
                        canonical_nid, was_new = apply_fetched_paper(network, bundle)
                except Exception as exc:
                    logger.warning("Merge of {} failed: {}", bundle.original_nid, exc)
                    if raise_error:
                        for queued_future in list(pending):
                            queued_future.cancel()
                        raise
                    with lock:
                        network.mark_explored(bundle.original_nid)
                    partial_failures += 1
                    fill_pool(pool, pending)
                    continue

                if not was_new and canonical_nid is None:
                    if bundle.error:
                        partial_failures += 1
                    fill_pool(pool, pending)
                    continue

                if canonical_nid and canonical_nid != nid:
                    claimed.add(canonical_nid)

                if was_new:
                    newly_explored += 1
                    papers_since_checkpoint += 1

                if (
                    checkpoint_every > 0
                    and graph_dir is not None
                    and papers_since_checkpoint >= checkpoint_every
                ):
                    ckpt_stem = build_graph_stem(run_stem)
                    logger.info(
                        "BFS checkpoint; {} papers explored, saving {}",
                        newly_explored,
                        ckpt_stem,
                    )
                    with lock:
                        save_graph(
                            network,
                            graph_dir,
                            stem=ckpt_stem,
                            write_gexf=False,
                        )
                    prune_checkpoints(
                        graph_dir,
                        base_stem=extract_base_stem(ckpt_stem),
                        max_keep=max_num_checkpoints,
                    )
                    papers_since_checkpoint = 0

                effective_nid = canonical_nid or nid
                with lock:
                    enqueue_citation_neighbors(
                        network,
                        effective_nid,
                        depth,
                        max_depth,
                        claimed,
                        queue,
                    )
                fill_pool(pool, pending)

            logger.debug(
                "Parallel BFS; pending: {}, queue: {}, claimed: {}",
                len(pending),
                len(queue),
                len(claimed),
            )

    if partial_failures:
        logger.info(
            "Parallel BFS done; {} newly explored, {} failures | {}",
            newly_explored,
            partial_failures,
            network.stats(),
        )
    else:
        logger.info(
            "Parallel BFS done; {} newly explored | {}",
            newly_explored,
            network.stats(),
        )
    return newly_explored


def expand_graph_parallel_priority(
    network: CitationNetwork,
    seeds: str | list[str],
    *,
    max_depth: int | None = 1,
    workers: int = 4,
    arxiv_client: Any,
    pdf_parser: Any,
    author_parser: Any,
    ref_parser: Any,
    tex_ref_parser: Any,
    pdf_dir: Path,
    text_dir: Path,
    source_dir: Path,
    pdf_parser_factory: Callable[[], Any] | None = None,
    author_parser_factory: Callable[[], Any] | None = None,
    ref_parser_factory: Callable[[], Any] | None = None,
    tex_ref_parser_factory: Callable[[], Any] | None = None,
    graph_dir: Path | None = None,
    run_stem: str = "citation_network",
    checkpoint_every: int = 0,
    max_num_checkpoints: int = 2,
    dblp_client: Any = None,
    semantic_scholar_client: Any = None,
    resolve_et_al_authors: bool = False,
    raise_error: bool = False,
) -> int:
    """Expand the citation graph in parallel, submitting smallest-depth nodes first."""
    if isinstance(seeds, str):
        seeds = [seeds]

    lock = threading.Lock()
    queue = DepthPriorityQueue()
    claimed: set[str] = set()
    in_flight_arxiv_ids: set[str] = set()
    in_flight_titles: set[str] = set()
    file_locks_mu = threading.Lock()
    file_locks: dict[str, threading.Lock] = {}

    def get_file_lock(key: str) -> threading.Lock:
        with file_locks_mu:
            if key not in file_locks:
                file_locks[key] = threading.Lock()
            return file_locks[key]

    parser_provider = ThreadLocalPaperParsers(
        pdf_parser,
        author_parser,
        ref_parser,
        tex_ref_parser,
        pdf_parser_factory=pdf_parser_factory,
        author_parser_factory=author_parser_factory,
        ref_parser_factory=ref_parser_factory,
        tex_ref_parser_factory=tex_ref_parser_factory,
    )

    def fetch_with_thread_parsers(
        arxiv_id: str,
        title: str,
        nid: str,
        depth: int,
    ) -> Any:
        (
            worker_pdf_parser,
            worker_author_parser,
            worker_ref_parser,
            worker_tex_ref_parser,
        ) = parser_provider.get_parsers()
        return fetch_paper_data(
            arxiv_id,
            title,
            nid,
            depth,
            arxiv_client,
            worker_pdf_parser,
            worker_author_parser,
            worker_ref_parser,
            worker_tex_ref_parser,
            pdf_dir,
            text_dir,
            source_dir,
            get_file_lock,
            dblp_client,
            semantic_scholar_client,
            raise_error,
            resolve_et_al_authors=resolve_et_al_authors,
        )

    for nid in seeds:
        if nid in claimed:
            continue
        claimed.add(nid)
        queue.append((nid, get_paper_depth(network.db, nid)))

    if not queue:
        logger.info("expand_graph_parallel_priority: no seeds; nothing to do.")
        return 0

    logger.info(
        "Starting parallel priority expansion; {} seed(s), max_depth: {}, workers: {}",
        len(queue),
        max_depth,
        workers,
    )

    newly_explored = 0
    partial_failures = 0
    papers_since_checkpoint = 0

    def try_submit_one(
        pool: ThreadPoolExecutor,
        pending: dict[Future, tuple[str, int, str, str]],
    ) -> bool:
        deferred: list[tuple[str, int]] = []

        try:
            while queue:
                if not has_exploration_capacity(
                    network,
                    pending_count=len(pending),
                ):
                    return False

                nid, depth = queue.popleft()
                if network.is_explored(nid_db_id(nid)):
                    with lock:
                        enqueue_citation_neighbors(
                            network,
                            nid,
                            depth,
                            max_depth,
                            claimed,
                            queue,
                        )
                    continue

                row = network.db.get_paper(nid_db_id(nid))
                arxiv_id = row["arxiv_id"] if row else ""
                title = row["title"] if row else ""
                if not arxiv_id and not title:
                    with lock:
                        network.mark_explored(nid)
                    continue

                norm_title = normalize_text(title) if title else ""
                if arxiv_id and arxiv_id in in_flight_arxiv_ids:
                    deferred.append((nid, depth))
                    continue
                if not arxiv_id and norm_title and norm_title in in_flight_titles:
                    deferred.append((nid, depth))
                    continue

                if arxiv_id:
                    in_flight_arxiv_ids.add(arxiv_id)
                if not arxiv_id and norm_title:
                    in_flight_titles.add(norm_title)

                future = pool.submit(
                    fetch_with_thread_parsers,
                    arxiv_id,
                    title,
                    nid,
                    depth,
                )
                pending[future] = (nid, depth, arxiv_id, norm_title)
                return True

            return False
        finally:
            for item in deferred:
                queue.append(item)

    def fill_pool(
        pool: ThreadPoolExecutor,
        pending: dict[Future, tuple[str, int, str, str]],
    ) -> None:
        while len(pending) < workers:
            if not try_submit_one(pool, pending):
                break

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: dict[Future, tuple[str, int, str, str]] = {}
        fill_pool(pool, pending)

        while pending:
            done = wait(pending, return_when=FIRST_COMPLETED).done
            for future in done:
                nid, depth, fut_arxiv_id, fut_norm_title = pending.pop(future)
                if fut_arxiv_id:
                    in_flight_arxiv_ids.discard(fut_arxiv_id)
                if fut_norm_title:
                    in_flight_titles.discard(fut_norm_title)

                try:
                    bundle = future.result()
                except Exception as exc:
                    logger.warning(
                        "Worker for {} raised; leaving unexplored for retry: {}",
                        nid,
                        exc,
                    )
                    if raise_error:
                        for queued_future in list(pending):
                            queued_future.cancel()
                        raise
                    partial_failures += 1
                    fill_pool(pool, pending)
                    continue

                try:
                    with lock:
                        canonical_nid, was_new = apply_fetched_paper(network, bundle)
                except Exception as exc:
                    logger.warning("Merge of {} failed: {}", bundle.original_nid, exc)
                    if raise_error:
                        for queued_future in list(pending):
                            queued_future.cancel()
                        raise
                    with lock:
                        network.mark_explored(bundle.original_nid)
                    partial_failures += 1
                    fill_pool(pool, pending)
                    continue

                if not was_new and canonical_nid is None:
                    if bundle.error:
                        partial_failures += 1
                    fill_pool(pool, pending)
                    continue

                if canonical_nid and canonical_nid != nid:
                    claimed.add(canonical_nid)

                if was_new:
                    newly_explored += 1
                    papers_since_checkpoint += 1

                if (
                    checkpoint_every > 0
                    and graph_dir is not None
                    and papers_since_checkpoint >= checkpoint_every
                ):
                    ckpt_stem = build_graph_stem(run_stem)
                    logger.info(
                        "Priority checkpoint; {} papers explored, saving {}",
                        newly_explored,
                        ckpt_stem,
                    )
                    with lock:
                        save_graph(
                            network,
                            graph_dir,
                            stem=ckpt_stem,
                            write_gexf=False,
                        )
                    prune_checkpoints(
                        graph_dir,
                        base_stem=extract_base_stem(ckpt_stem),
                        max_keep=max_num_checkpoints,
                    )
                    papers_since_checkpoint = 0

                effective_nid = canonical_nid or nid
                with lock:
                    enqueue_citation_neighbors(
                        network,
                        effective_nid,
                        depth,
                        max_depth,
                        claimed,
                        queue,
                    )
                fill_pool(pool, pending)

            logger.debug(
                "Parallel priority expansion; pending: {}, queue: {}, claimed: {}",
                len(pending),
                len(queue),
                len(claimed),
            )

    if partial_failures:
        logger.info(
            "Parallel priority expansion done; {} newly explored, {} failures | {}",
            newly_explored,
            partial_failures,
            network.stats(),
        )
    else:
        logger.info(
            "Parallel priority expansion done; {} newly explored | {}",
            newly_explored,
            network.stats(),
        )
    return newly_explored
