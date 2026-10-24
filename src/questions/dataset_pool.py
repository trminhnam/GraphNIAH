"""Shared example pool and resumable experiment files for create_dataset.v2.py.

The pool is an append-only JSONL file of complete examples. An experiment file
has one ``{"qid": ...}`` line per planned slot, in plan order, pointing into the
pool. ``run_experiment`` reuses pool examples whose config and pool key match
exactly, builds the missing ones (optionally in worker processes), and on a
rerun resumes, extends, or restarts the experiment file. ``run_experiments``
does the same for several experiments, each with its own pool, sharing one set
of worker processes.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import os
import random
import sys
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
from typing import Any, BinaryIO

import orjson

from src.logging.console import log_line, progress_bar, set_progress_quiet

REPO_ROOT = Path(__file__).resolve().parents[2]
PROGRESS_FIELDS = (
    ("question_format", "format"),
    ("question_depth", "depth"),
    ("component_node_count", "nodes"),
    ("distractor_type", "distractor"),
)
REFRESH_SECONDS = 1.0

# The slot this process is building, prefixed to every message it reports.
CURRENT_SLOT = ""
# A worker sends this through the message queue once it has opened its inputs;
# it is counted on the bar, never printed (a printed line has no NUL).
WORKER_READY = "\0worker-ready"
# In a worker process: the queue to the main process.
WORKER_MESSAGES: Any = None


def report(message: str) -> None:
    """Print one message above the progress bar, from the main or a worker process."""
    if CURRENT_SLOT:
        message = f"[{CURRENT_SLOT}] {message}"
    log_line(message)


class QueueWriter(io.TextIOBase):
    """A worker's stdout/stderr: each complete line goes to the main process."""

    def __init__(self, queue: Any) -> None:
        self.queue = queue
        self.pending = ""
        # While a list, complete lines are kept here instead of sent.
        self.held: list[str] | None = None

    def write(self, text: str) -> int:
        self.pending += text
        *lines, self.pending = self.pending.split("\n")
        for line in lines:
            if not line.strip():
                continue
            if self.held is not None:
                self.held.append(line)
            else:
                self.queue.put(line)
        return len(text)


def start_worker(queue: Any) -> None:
    """Worker initializer: only the main process draws the bar, so a worker
    sends everything it prints (messages, logs, warnings) to the main process,
    which prints it above the bar."""
    global WORKER_MESSAGES
    WORKER_MESSAGES = queue
    sys.stdout = sys.stderr = QueueWriter(queue)
    set_progress_quiet(True)


@contextmanager
def held_output() -> Iterator[list[str]]:
    """In a worker, keep the lines printed inside the block instead of sending
    them; the caller decides whether to show them. Elsewhere, print as usual."""
    writer = sys.stdout
    if not isinstance(writer, QueueWriter):
        yield []
        return
    writer.held = held = []
    try:
        yield held
    finally:
        writer.held = None


def worker_ready() -> None:
    """Tell the main process that this worker has opened its inputs."""
    if WORKER_MESSAGES is not None:
        WORKER_MESSAGES.put(WORKER_READY)


def describe(config: dict[str, Any]) -> str:
    return " ".join(
        f"{label}={str(config[key]).removeprefix('reachability-')}"
        for key, label in PROGRESS_FIELDS
        if key in config
    )


class RetryableConstructionError(RuntimeError):
    """One attempt failed by chance; a new seed can succeed with the same config."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), (self.reason, self.detail)


def digest(value: Any) -> str:
    return hashlib.sha256(orjson.dumps(value, option=orjson.OPT_SORT_KEYS)).hexdigest()


def seed_for(seed: int, *parts: Any) -> int:
    return int(digest([seed, *parts])[:16], 16)


def portable_path(path: Path) -> str:
    """Name a file the same way on every machine: repo-relative, else its name.

    Only the parent directory is resolved, so a symlink inside the repo keeps
    its repo path instead of the machine-specific target.
    """
    path = Path(path)
    absolute = Path(os.path.realpath(path.parent)) / path.name
    try:
        return absolute.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def settings_path_for(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".settings.json")


def read_jsonl(
    handle: BinaryIO, *, repair: bool = True
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Read records; an interrupted final append is cut off (repair) or skipped.

    Readers that do not own the file (``repair=False``) never modify it, so they
    can read a pool or experiment that another process is still appending to.
    """
    handle.seek(0)
    while line := handle.readline():
        offset = handle.tell() - len(line)
        try:
            record = orjson.loads(line)
        except orjson.JSONDecodeError:
            if not line.endswith(b"\n"):
                if repair:
                    handle.seek(offset)
                    handle.truncate()
                    handle.flush()
                    os.fsync(handle.fileno())
                    log_line(
                        "Removed an unfinished last line (an interrupted write) "
                        f"at byte {offset} of {getattr(handle, 'name', 'a JSONL file')}."
                    )
                return
            raise ValueError(f"Corrupt JSONL record at byte {offset}.") from None
        if not isinstance(record, dict):
            raise TypeError(f"Expected an object at byte {offset}.")
        if repair and not line.endswith(b"\n"):
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield offset, record


def append_record(handle: BinaryIO, record: dict[str, Any]) -> int:
    handle.seek(0, os.SEEK_END)
    offset = handle.tell()
    handle.write(orjson.dumps(record) + b"\n")
    handle.flush()
    os.fsync(handle.fileno())
    return offset


@dataclass
class ExamplePool:
    handle: BinaryIO
    offsets: dict[str, int] = field(default_factory=dict)
    # A built config can differ from the slot's planned one (a resampled node
    # count), so each qid keeps both keys.
    actual_keys: dict[str, str] = field(default_factory=dict)
    planned_keys: dict[str, str] = field(default_factory=dict)
    matches: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def scan(self, *, repair: bool = True) -> None:
        for offset, record in read_jsonl(self.handle, repair=repair):
            self.register(offset, record)

    def register(self, offset: int, record: dict[str, Any]) -> None:
        qid = record["qid"]
        if qid in self.offsets:
            raise ValueError(f"Duplicate qid in pool: {qid}")
        compatibility = record["compatibility"]
        actual = digest([compatibility, record["config"]])
        self.offsets[qid] = offset
        self.actual_keys[qid] = actual
        self.planned_keys[qid] = digest(
            [compatibility, record.get("planned_config", record["config"])]
        )
        # Reuse goes by the config actually built, never the planned one.
        self.matches[actual].append(qid)

    def get(self, qid: str) -> dict[str, Any]:
        self.handle.seek(self.offsets[qid])
        return orjson.loads(self.handle.readline())

    def can_resume_slot(
        self, qid: str, config: dict[str, Any], compatibility: dict[str, Any]
    ) -> bool:
        """Whether a saved qid still fits the slot that holds it.

        Besides an exact match, a slot keeps the example it built itself after
        resampling (its planned config). New slots never get such examples.
        """
        key = digest([compatibility, config])
        return key in (self.actual_keys.get(qid), self.planned_keys.get(qid))

    def choose(
        self,
        config: dict[str, Any],
        compatibility: dict[str, Any],
        used: set[str],
        rng: random.Random,
    ) -> str | None:
        candidates = [
            qid
            for qid in self.matches.get(digest([compatibility, config]), [])
            if qid not in used
        ]
        return rng.choice(candidates) if candidates else None

    def add(self, record: dict[str, Any]) -> None:
        if record["qid"] in self.offsets:
            raise ValueError("Cannot append an existing qid.")
        self.register(append_record(self.handle, record), record)


def read_experiment_qids(output_path: Path, *, repair: bool) -> list[str]:
    """Return the qids of an experiment file in slot order."""
    if not output_path.exists():
        return []
    with output_path.open("r+b" if repair else "rb") as handle:
        qids = []
        for _offset, record in read_jsonl(handle, repair=repair):
            if set(record) != {"qid"}:
                raise ValueError(f"{output_path}: every line must be {{'qid': ...}}.")
            qids.append(record["qid"])
    if len(set(qids)) != len(qids):
        raise ValueError(f"{output_path}: an experiment cannot repeat a qid.")
    return qids


def restart_experiment(
    output_path: Path, settings_path: Path, reason: str, label: str = ""
) -> None:
    """Keep the old experiment as *.previous; its examples stay in the pool."""
    for path in (output_path, settings_path):
        if path.exists():
            os.replace(path, path.with_name(path.name + ".previous"))
    log_line(
        (f"[{label}] " if label else "")
        + f"{reason}: kept the old experiment as {output_path.name}.previous and "
        "started a new one. Pool examples whose built config matches a slot are "
        "reused."
    )


def resume_or_restart(
    output_path: Path,
    settings_path: Path,
    identity: dict[str, Any],
    configs: list[dict[str, Any]],
    pool: ExamplePool,
    label: str = "",
) -> list[str]:
    """Return the saved qids to keep; restart when the experiment itself changed."""
    if not settings_path.exists():
        if output_path.exists() and output_path.stat().st_size:
            raise ValueError("Existing experiment has no settings checkpoint.")
        return []
    saved = orjson.loads(settings_path.read_bytes()).get("identity") or {}
    if saved != identity:
        changed = sorted(
            key
            for key in identity.keys() | saved.keys()
            if identity.get(key) != saved.get(key)
        )
        restart_experiment(
            output_path,
            settings_path,
            f"Experiment settings changed ({', '.join(changed)})",
            label,
        )
        return []
    qids = read_experiment_qids(output_path, repair=True)
    if len(qids) > len(configs):
        raise ValueError(
            f"{output_path} already has {len(qids)} examples, more than the "
            f"{len(configs)} planned now. Nothing was changed; use a larger "
            "--examples-per-value or a new --output."
        )
    for position, qid in enumerate(qids):
        if qid not in pool.offsets:
            raise ValueError(f"Experiment qid {qid} is missing from the pool.")
        if not pool.can_resume_slot(qid, configs[position], identity["compatibility"]):
            restart_experiment(
                output_path,
                settings_path,
                f"The planned config of slot {position + 1} changed",
                label,
            )
            return []
    return qids


def write_settings(settings_path: Path, identity: dict[str, Any], slots: int) -> None:
    temporary = settings_path.with_suffix(settings_path.suffix + ".tmp")
    temporary.write_bytes(
        orjson.dumps({"identity": identity, "slots": slots}, option=orjson.OPT_INDENT_2)
    )
    temporary.replace(settings_path)


@dataclass(frozen=True)
class SlotTask:
    """Everything a (possibly separate) process needs to build one slot."""

    construct: Callable[[dict[str, Any], int], dict[str, Any]]
    config: dict[str, Any]
    slot_seed: int
    compatibility: dict[str, Any]
    max_attempts: int
    label: str
    start_attempt: int = 0


def build_slot(task: SlotTask) -> tuple[dict[str, Any], int, dict[str, int]]:
    """Try new seeds for one fixed config; only chance failures are retried."""
    global CURRENT_SLOT
    retries: Counter = Counter()
    for attempt in range(task.start_attempt, task.max_attempts):
        CURRENT_SLOT = task.label
        try:
            example = task.construct(
                task.config, seed_for(task.slot_seed, "construct", attempt)
            )
        except RetryableConstructionError as error:
            retries[error.reason] += 1
            report(
                f"attempt {attempt + 1}/{task.max_attempts} failed ({error}); "
                "retrying with a new seed"
            )
            continue
        finally:
            CURRENT_SLOT = ""
        example["compatibility"] = task.compatibility
        # Content identity: the same prompt and answer always get the same qid.
        # An example without a prompt (components only) names its own content.
        content = example.get("content_key") or [
            example["system_message"],
            example["user_message"],
        ]
        example["qid"] = digest(
            [task.compatibility, task.config, *content, example["answer"]]
        )
        return example, attempt, dict(retries)
    raise RuntimeError(
        f"Could not complete {task.label} after {task.max_attempts} attempts with "
        f"config {task.config}: {dict(retries)}"
    )


def submit_slot(executor: ProcessPoolExecutor | None, task: SlotTask) -> Future:
    if executor is not None:
        return executor.submit(build_slot, task)
    future: Future = Future()
    try:
        future.set_result(build_slot(task))
    except Exception as error:
        future.set_exception(error)
    return future


def run_experiment(
    pool_path: Path,
    output_path: Path,
    configs: list[dict[str, Any]],
    settings: dict[str, Any],
    compatibility: dict[str, Any],
    construct: Callable[[dict[str, Any], int], dict[str, Any]],
    *,
    seed: int,
    max_attempts: int,
    workers: int = 1,
) -> dict[str, Any]:
    """Fill every slot in order; each pool record is saved before its qid.

    ``settings`` and ``compatibility`` identify the experiment. When they change,
    or the planned config of a saved slot changes, the old file is kept as
    ``*.previous`` and the experiment starts again, reusing pool examples. A
    longer plan only appends slots. With ``workers > 1``, ``construct`` must be
    picklable; it runs in spawned processes, never inheriting SQLite handles.
    """
    experiment = ExperimentSpec(pool_path, output_path, configs, settings, compatibility)
    return run_experiments(
        [experiment], construct, seed=seed, max_attempts=max_attempts, workers=workers
    )[0]


@dataclass(frozen=True)
class ExperimentSpec:
    """One experiment to fill: its files, planned slots, and identity."""

    pool_path: Path
    output_path: Path
    configs: list[dict[str, Any]]
    settings: dict[str, Any]
    compatibility: dict[str, Any]
    # Names the experiment in messages when one run fills several.
    label: str = ""


@dataclass
class ExperimentFill:
    """An open experiment: its locked pool, its output, and its progress."""

    spec: ExperimentSpec
    pool: ExamplePool
    output: BinaryIO
    done: list[str]
    seed: int
    retries: Counter = field(default_factory=Counter)
    ready: dict[int, str] = field(default_factory=dict)
    stats: dict[str, Any] = field(init=False)
    used: set[str] = field(init=False)
    next_position: int = field(init=False)

    def __post_init__(self) -> None:
        self.stats = {"resumed": len(self.done), "reused": 0, "created": 0}
        self.used = set(self.done)
        self.next_position = len(self.done)

    @property
    def complete(self) -> bool:
        return self.next_position == len(self.spec.configs)

    def slot_seed(self, position: int) -> int:
        return seed_for(self.seed, "slot", self.spec.settings, position)

    def write_ready(self) -> None:
        """Write every finished slot whose earlier slots are all written."""
        while self.next_position in self.ready:
            append_record(self.output, {"qid": self.ready.pop(self.next_position)})
            self.next_position += 1


def run_experiments(
    experiments: list[ExperimentSpec],
    construct: Callable[[dict[str, Any], int], dict[str, Any]],
    *,
    seed: int,
    max_attempts: int,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Fill several experiments with one set of workers; one stats dict each.

    Each experiment keeps its own pool, experiment file, and settings, exactly
    as ``run_experiment`` writes them, so the two resume each other. The
    workers start once, and a free worker takes the next experiment's slots
    instead of waiting for the slowest slot of the current one.
    """
    if workers < 1 or max_attempts < 1:
        raise ValueError("workers and max_attempts must be positive.")
    files = [
        path.resolve()
        for experiment in experiments
        for path in (
            experiment.pool_path,
            experiment.output_path,
            settings_path_for(experiment.output_path),
        )
    ]
    if len(set(files)) != len(files):
        raise ValueError(
            "Pool, experiment output, and settings must be different files, and "
            "each experiment needs its own pool."
        )
    with ExitStack() as stack:
        # Lock every pool before changing any experiment file.
        pools = []
        for experiment in experiments:
            experiment.pool_path.parent.mkdir(parents=True, exist_ok=True)
            experiment.output_path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(experiment.pool_path.open("a+b"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(
                    f"Another process is writing this pool ({experiment.pool_path}). "
                    "Retry when it finishes."
                ) from None
            pools.append(ExamplePool(handle))
        fills = []
        for experiment, pool in zip(experiments, pools):
            pool.scan()
            settings_path = settings_path_for(experiment.output_path)
            identity = {
                **experiment.settings,
                "compatibility": experiment.compatibility,
                "pool": portable_path(experiment.pool_path),
            }
            done = resume_or_restart(
                experiment.output_path,
                settings_path,
                identity,
                experiment.configs,
                pool,
                experiment.label,
            )
            write_settings(settings_path, identity, len(experiment.configs))
            output = stack.enter_context(experiment.output_path.open("a+b"))
            fills.append(ExperimentFill(experiment, pool, output, done, seed))
        return fill_experiments(fills, construct, max_attempts=max_attempts, workers=workers)


def fill_experiments(
    fills: list[ExperimentFill],
    construct: Callable[[dict[str, Any], int], dict[str, Any]],
    *,
    max_attempts: int,
    workers: int,
) -> list[dict[str, Any]]:
    """Reserve pool matches, build the rest, and write qids in slot order.

    One bar counts finished slots (reused or built) of every experiment, in both
    serial and worker mode; a slot is written to its experiment once all slots
    before it are. Slots are submitted experiment by experiment.
    """
    several = len(fills) > 1
    queue: deque[tuple[ExperimentFill, int, SlotTask]] = deque()
    for fill in fills:
        configs = fill.spec.configs
        # Reserving every pool match first equals the serial order: a new
        # example is only ever used by the slot that built it.
        for position in range(len(fill.done), len(configs)):
            config = configs[position]
            qid = fill.pool.choose(
                config,
                fill.spec.compatibility,
                fill.used,
                random.Random(fill.slot_seed(position)),
            )
            if qid is None:
                label = f"slot {position + 1}/{len(configs)}"
                if fill.spec.label:
                    label = f"{fill.spec.label} | {label}"
                queue.append(
                    (
                        fill,
                        position,
                        SlotTask(
                            construct,
                            config,
                            fill.slot_seed(position),
                            fill.spec.compatibility,
                            max_attempts,
                            label,
                        ),
                    )
                )
            else:
                fill.used.add(qid)
                fill.ready[position] = qid
                fill.stats["reused"] += 1
        fill.write_ready()

    def total(key: str) -> int:
        return sum(fill.stats[key] for fill in fills)

    workers_text = f"{workers} worker{'s' if workers > 1 else ''}"
    if several:
        log_line(
            f"Plan: {len(fills)} experiments, "
            f"{sum(len(fill.spec.configs) for fill in fills):,} slots; "
            f"{total('resumed'):,} already written, {total('reused'):,} reused from "
            f"the pools, {len(queue):,} to build with {workers_text}."
        )
    else:
        log_line(
            f"Plan: {len(fills[0].spec.configs)} slots; {total('resumed')} already "
            f"written, {total('reused')} reused from the pool, {len(queue)} to build "
            f"with {workers_text}."
        )
    context = get_context("spawn")
    messages = context.Queue() if workers > 1 and queue else None
    starting = min(workers, len(queue)) if messages is not None else 0
    if starting:
        log_line(
            f"Starting {starting} worker processes. Each one loads the code, opens the "
            "database, and loads the tokenizers once before its first example, so "
            "the first examples can take a few minutes."
        )
    started = 0
    executor = (
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=start_worker,
            initargs=(messages,),
        )
        if messages is not None
        else None
    )

    def print_worker_messages() -> None:
        nonlocal started
        while messages is not None:
            try:
                message = messages.get_nowait()
            except Empty:
                return
            if message == WORKER_READY:
                started += 1
            else:
                log_line(message)

    bar = progress_bar(
        total=sum(len(fill.spec.configs) for fill in fills),
        initial=total("resumed") + total("reused"),
        desc="Examples",
        unit="ex",
        dynamic_ncols=True,
        miniters=1,
    )
    running: dict[Future, tuple[ExperimentFill, int, SlotTask]] = {}
    failure: BaseException | None = None
    last = ""
    try:
        while queue or running:
            while queue and failure is None and len(running) < workers:
                fill, position, task = queue.popleft()
                running[submit_slot(executor, task)] = (fill, position, task)
            if not running:
                break
            # The timeout keeps worker messages and elapsed time live.
            finished, _pending = wait(
                running, timeout=REFRESH_SECONDS, return_when=FIRST_COMPLETED
            )
            print_worker_messages()
            for future in finished:
                fill, position, task = running.pop(future)
                try:
                    example, attempt, slot_retries = future.result()
                except Exception as error:
                    # Save what the other workers finish, then stop.
                    failure = failure or error
                    continue
                fill.retries.update(slot_retries)
                qid = example["qid"]
                if qid in fill.used:
                    queue.appendleft(
                        (fill, position, replace(task, start_attempt=attempt + 1))
                    )
                    continue
                if qid in fill.pool.offsets:
                    fill.stats["reused"] += 1
                else:
                    fill.pool.add(example)
                    fill.stats["created"] += 1
                fill.used.add(qid)
                fill.ready[position] = qid
                last = describe(example["config"])
                bar.update(1)
                was_complete = fill.complete
                fill.write_ready()
                if several and fill.complete and not was_complete:
                    log_line(
                        f"[{fill.spec.label}] finished: {len(fill.spec.configs):,} slots "
                        f"({fill.stats['created']:,} new, {fill.stats['reused']:,} reused "
                        f"from the pool, {fill.stats['resumed']:,} already there)."
                    )
            retried = sum(sum(fill.retries.values()) for fill in fills)
            completed = sum(fill.complete for fill in fills)
            bar.set_postfix_str(
                (f"workers ready={started}/{starting} " if started < starting else "")
                + (f"experiments done={completed}/{len(fills)} " if several else "")
                + f"new={total('created')} reused={total('reused')} "
                f"retries={retried} last: {last}"
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        print_worker_messages()
        bar.close()
    for fill in fills:
        fill.stats["retries"] = dict(fill.retries)
    if failure is not None:
        log_line(
            f"Stopped: {failure}. Finished slots are saved; rerun the same command "
            "to continue."
        )
        raise failure
    return [fill.stats for fill in fills]
