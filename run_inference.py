#!/usr/bin/env python3
"""Send each example of a create_dataset.v1 experiment to a model once.

The export step happens here: the experiment's qids are read in order, each is
looked up in the shared pool, and that example's system and user messages are
sent as stored. Answers are saved to --output after every response, together
with the gold answer and config for evaluation. Rerun the same command to
resume: saved answers are kept, failed requests are sent again, and examples
appended to the experiment are picked up. Saved answers whose qid left the
experiment (for example after it restarted) are dropped and counted.

--max-samples limits examples per studied value, preserving their saved order.
For the reasoning-effort dataset (--vary none), it limits the shared dataset
as a whole, so every effort uses the same first N examples.
--question-depths optionally selects depths before applying --max-samples.
--component-node-counts likewise selects component sizes before limiting examples.

Successful answers are also stored in --inference-pool (by default,
inference_pool.jsonl beside --pool) for reuse across experiments. Keys contain
explicit qid, resolved model, and generation parameter values, without hashes.
Existing successful output records are imported on resume. One process holds
the inference-pool lock for its entire run; --max-workers still allows
concurrent requests inside that process. Metadata's run_accounting reports
new response costs separately from the historical costs in the output records.

Example (activate the niah environment first)::

    python run_inference.py --pool output/datasets/pool.jsonl \
        --experiment output/datasets/depth.jsonl \
        --output output/predictions/depth.gpt-5.6-sol.json --model gpt-5.6-sol
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import os
import sys
import tempfile
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import orjson
from tqdm import tqdm

from inference import (
    build_completion_kwargs,
    build_empty_result,
    call_completion_with_retry,
    format_duration,
    positive_int,
    resolve_model_name,
)
from src.llm.accounting import aggregate_accounting, aggregate_retry_accounting
from src.llm.call_custom import CUSTOM_MODEL_PREFIX
from src.llm.inference_pool import (
    InferencePool,
    open_inference_pool,
    successful_response,
)
from src.llm.model_info import MODEL_INFO_BY_SLUG
from src.llm.retry import CompletionRequestError
from src.logging.console import log_line, progress_bar
from src.questions.dataset_pool import (
    ExamplePool,
    portable_path,
    read_experiment_qids,
    settings_path_for,
)
from src.setup import bootstrap_runtime

OUTPUT_SCHEMA_VERSION = 1
GOLD_FIELDS = ("question_type", "answer_type", "answer", "config")
VARY_KEYS = {
    "question-depth": "question_depth",
    "node-count": "component_node_count",
    "question-format": "question_format",
    "distractor-type": "distractor_type",
}


def normalize_generation_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Make omitted defaults, numeric values, and model aliases compare equally."""
    normalized = {
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": None,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "reasoning_effort": "low",
        "stop": None,
        **settings,
    }
    normalized.pop("api_key", None)
    normalized.pop("messages", None)
    normalized["model"] = resolve_model_name(normalized["model"])
    normalized["stop"] = normalized["stop"] or None
    for key in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
        normalized[key] = float(normalized[key])
    return normalized


def inference_key(qid: str, generation_settings: dict[str, Any]) -> dict[str, Any]:
    return {"qid": qid, **normalize_generation_settings(generation_settings)}


def response_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Cache the response, without experiment-specific gold fields or reuse flags."""
    excluded = {"qid", *GOLD_FIELDS, "planned_config", "inference_source"}
    return {key: value for key, value in record.items() if key not in excluded}


def load_experiment(
    pool_path: Path, experiment_path: Path
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    """Return the experiment's examples in slot order, their pool offsets, and identity.

    Only the small gold fields are kept in memory; messages are read from the
    pool when each request is sent. Neither file is modified, so this also works
    while create_dataset.v1.py is still appending to them.
    """
    settings_path = settings_path_for(experiment_path)
    if not settings_path.exists():
        raise ValueError(
            f"{settings_path} is missing; --experiment must be a create_dataset.v1 output."
        )
    identity = orjson.loads(settings_path.read_bytes()).get("identity") or {}
    if identity.get("pool") != portable_path(pool_path):
        raise ValueError(
            f"{experiment_path} was built with pool {identity.get('pool')!r}, "
            f"not {portable_path(pool_path)!r}."
        )
    qids = read_experiment_qids(experiment_path, repair=False)
    with pool_path.open("rb") as handle:
        pool = ExamplePool(handle)
        pool.scan(repair=False)
        missing = [qid for qid in qids if qid not in pool.offsets]
        if missing:
            raise ValueError(f"{len(missing)} experiment qids are missing from the pool.")
        examples = []
        for qid in qids:
            record = pool.get(qid)
            if record["config"].get("components_only"):
                raise ValueError(
                    f"{experiment_path} was built with --components-only (for the "
                    "growth-parameter search); it has no prompts to send."
                )
            examples.append(
                {
                    "qid": qid,
                    **{key: record[key] for key in GOLD_FIELDS},
                    "planned_config": record.get("planned_config", record["config"]),
                }
            )
    return examples, {qid: pool.offsets[qid] for qid in qids}, identity


def select_examples(
    examples: list[dict[str, Any]],
    vary: str,
    max_samples: int | None,
    question_depths: list[int] | None = None,
    component_node_counts: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Keep the first N examples of each studied value in experiment order."""
    if question_depths is not None:
        examples = [
            example
            for example in examples
            if example["config"]["question_depth"] in question_depths
        ]
    if component_node_counts is not None:
        examples = [
            example
            for example in examples
            if example["config"]["component_node_count"] in component_node_counts
        ]
    if max_samples is None:
        return examples
    if vary == "none":
        return examples[:max_samples]
    studied_key = VARY_KEYS[vary]
    counts: Counter[Any] = Counter()
    selected = []
    for example in examples:
        value = example["config"][studied_key]
        if counts[value] < max_samples:
            selected.append(example)
            counts[value] += 1
    return selected


def load_saved_records(
    output_path: Path,
    generation_settings: dict[str, Any],
    all_qids: set[str],
) -> dict[str, dict[str, Any]]:
    """Keep saved answers that still belong to the experiment."""
    if not output_path.exists():
        return {}
    payload = orjson.loads(output_path.read_bytes())
    if payload.get("schema_version") != OUTPUT_SCHEMA_VERSION:
        raise ValueError(f"{output_path} is not a run_inference.py output.")
    saved_settings = normalize_generation_settings(
        payload["metadata"]["generation_settings"]
    )
    if saved_settings != generation_settings:
        changed = sorted(
            key
            for key in saved_settings.keys() | generation_settings.keys()
            if saved_settings.get(key) != generation_settings.get(key)
        )
        raise ValueError(
            f"Cannot resume {output_path}: {', '.join(changed)} changed. "
            "Use a new --output."
        )
    records, dropped = {}, 0
    for record in payload["records"]:
        if record["qid"] in all_qids:
            records[record["qid"]] = record
        else:
            dropped += 1
    if dropped:
        log_line(
            f"Dropped {dropped} saved answers whose qid is no longer in the experiment."
        )
    return records


class BarSafeStdout(io.TextIOBase):
    """Print lines from shared helpers (e.g. retry notices) above the progress bar."""

    def __init__(self) -> None:
        self.pending = ""

    def write(self, text: str) -> int:
        self.pending += text
        *lines, self.pending = self.pending.split("\n")
        for line in lines:
            # The bar draws on stderr; writing there lets tqdm clear and redraw it.
            tqdm.write(line, file=sys.stderr)
        return len(text)

    def flush(self) -> None:
        sys.stderr.flush()


def write_output(output_path: Path, payload: dict[str, Any]) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent, prefix=f".{output_path.name}.", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


async def run(args: argparse.Namespace) -> None:
    """Lock the shared pool before reading saved answers or making requests."""
    args = argparse.Namespace(**vars(args))
    args.model_slug = getattr(args, "model_slug", args.model)
    args.model = resolve_model_name(args.model)
    pool_path, experiment_path = Path(args.pool), Path(args.experiment)
    inference_path = args.inference_pool or pool_path.parent / "inference_pool.jsonl"
    args.inference_pool = Path(inference_path)
    protected = {
        path.resolve()
        for path in (pool_path, experiment_path, settings_path_for(experiment_path))
    }
    destinations = [Path(args.output).resolve(), args.inference_pool.resolve()]
    if any(path in protected for path in destinations) or len(set(destinations)) != 2:
        raise ValueError(
            "--output and --inference-pool must differ from each other and from "
            "--pool, --experiment, and its settings file."
        )
    with open_inference_pool(args.inference_pool) as inference_pool:
        await run_with_pool(args, inference_pool)


async def run_with_pool(
    args: argparse.Namespace, inference_pool: InferencePool
) -> None:
    """Request only examples missing from both saved output and the shared pool."""
    started = time.perf_counter()
    pool_path, experiment_path = Path(args.pool), Path(args.experiment)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    examples, offsets, identity = load_experiment(pool_path, experiment_path)
    selected = select_examples(
        examples,
        identity["vary"],
        args.max_samples,
        args.question_depths,
        args.component_node_counts,
    )
    generation_settings = normalize_generation_settings(
        build_completion_kwargs(args, "")
    )
    records = load_saved_records(
        output_path,
        generation_settings,
        {example["qid"] for example in examples},
    )
    selected_qids = {example["qid"] for example in selected}
    output_examples = [
        example
        for example in examples
        if example["qid"] in selected_qids or example["qid"] in records
    ]
    preserved = len(output_examples) - len(selected)
    resumed, cache_hits = 0, 0
    source_output = portable_path(output_path)
    for example in selected:
        qid = example["qid"]
        key = inference_key(qid, generation_settings)
        if successful_response(records.get(qid, {})):
            inference_pool.add(key, response_fields(records[qid]), source_output)
            records[qid] = {**records[qid], **example, "inference_source": "output"}
            resumed += 1
        else:
            cached = inference_pool.get(key)
            if cached is not None:
                records[qid] = {**cached, **example, "inference_source": "pool"}
                cache_hits += 1
    pending = [
        example
        for example in selected
        if not successful_response(records.get(example["qid"], {}))
    ]
    model_slug = getattr(args, "model_slug", args.model)
    model_info = MODEL_INFO_BY_SLUG.get(model_slug)
    litellm_version = version("litellm")
    log_line(f"[Inference] {model_slug} ({args.model})")
    log_line(
        f"  experiment: {experiment_path} ({len(examples)} examples, "
        f"{len(selected)} selected)"
    )
    if args.max_samples is not None:
        scope = (
            "total (shared across efforts)"
            if identity["vary"] == "none"
            else "per studied value"
        )
        log_line(f"  max samples: {args.max_samples} {scope}")
    log_line(
        f"  resumed: {resumed}; pool hits: {cache_hits}; to request: "
        f"{len(pending)} with up to {args.max_workers} at a time"
    )
    log_line(f"  saved records preserved outside selection: {preserved}")
    log_line(f"  output: {output_path}")
    if args.question_depths is not None:
        log_line(f"  question depths: {args.question_depths}")
    if args.component_node_counts is not None:
        log_line(f"  component node counts: {args.component_node_counts}")
    log_line(f"  inference pool: {args.inference_pool}")
    context_length = identity.get("context_length")
    if (
        model_info
        and context_length
        and model_info.native_context_length < context_length
    ):
        log_line(
            f"  WARNING: prompts have {context_length:,} tokens but the model "
            f"accepts {model_info.native_context_length:,}."
        )

    new_responses: list[dict[str, Any]] = []

    def flush() -> dict[str, Any]:
        ordered = [
            records[example["qid"]]
            for example in output_examples
            if example["qid"] in records
        ]
        answered = sum(successful_response(record) for record in ordered)
        selected_answered = sum(
            successful_response(record)
            for record in ordered
            if record["qid"] in selected_qids
        )
        run_accounting = aggregate_accounting(new_responses)
        if not new_responses:
            run_accounting["total_total_cost"] = 0.0
        metadata = {
            "generation_settings": generation_settings,
            "model": model_slug,
            "pool": portable_path(pool_path),
            "experiment": portable_path(experiment_path),
            "inference_pool": portable_path(args.inference_pool),
            "resumed_examples": resumed,
            "cache_hits": cache_hits,
            "new_requests": len(new_responses),
            "run_accounting": {
                **run_accounting,
                "retry_accounting": aggregate_retry_accounting(new_responses),
            },
            "max_samples": args.max_samples,
            "question_depths": args.question_depths,
            "component_node_counts": args.component_node_counts,
            "max_samples_scope": "total"
            if identity["vary"] == "none"
            else "per_studied_value",
            "litellm_version": litellm_version,
            "total_examples": len(output_examples),
            "selected_examples": len(selected),
            "preserved_examples": preserved,
            "selected_answered_examples": selected_answered,
            "answered_examples": answered,
            "failed_examples": len(ordered) - answered,
            **aggregate_accounting(ordered),
            "retry_accounting": aggregate_retry_accounting(ordered),
        }
        write_output(
            output_path,
            {
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "metadata": metadata,
                "records": ordered,
            },
        )
        return metadata

    semaphore = asyncio.Semaphore(args.max_workers)
    failed = 0
    bar = progress_bar(
        total=len(selected),
        initial=len(selected) - len(pending),
        desc="Requests",
        unit="ex",
        dynamic_ncols=True,
        miniters=1,
    )
    with (
        pool_path.open("rb") as pool_handle,
        contextlib.redirect_stdout(BarSafeStdout()),
    ):

        async def process(example: dict[str, Any]) -> None:
            nonlocal failed
            async with semaphore:
                qid = example["qid"]
                pool_handle.seek(offsets[qid])
                stored = orjson.loads(pool_handle.readline())
                previous = records.get(qid, {})
                error_message = None
                try:
                    result = await call_completion_with_retry(
                        build_completion_kwargs(
                            args, stored["user_message"], stored["system_message"]
                        ),
                        use_custom_router=args.model.startswith(CUSTOM_MODEL_PREFIX),
                        model=args.model,
                        question_id=qid,
                    )
                except CompletionRequestError as error:
                    error_message = str(error)
                    result = {
                        **build_empty_result(),
                        "attempts": error.attempts,
                        "failed_attempts": error.failed_attempts,
                    }
                prior_attempts = previous.get("attempts") or 0
                response = {
                    "raw_answer": result.pop("content"),
                    **result,
                    "error": error_message,
                }
                new_responses.append(response)
                records[qid] = {
                    **example,
                    **response,
                    "inference_source": "request",
                    "attempts": prior_attempts + result["attempts"],
                    "failed_attempts": previous.get("failed_attempts", [])
                    + [
                        {**attempt, "attempt": prior_attempts + attempt["attempt"]}
                        for attempt in result["failed_attempts"]
                    ],
                }
                # Persist before updating the experiment, so an interrupted
                # output write can recover the paid response from the pool.
                inference_pool.add(
                    inference_key(qid, generation_settings),
                    response_fields(records[qid]),
                    source_output,
                )
                summary = flush()
                if error_message:
                    failed += 1
                    # stdout is BarSafeStdout here, so this prints above the bar.
                    print(
                        f"[{qid[:12]}] FAILED after {records[qid]['attempts']} "
                        f"attempts: {error_message}"
                    )
                cost = summary["run_accounting"]["total_total_cost"]
                bar.update(1)
                bar.set_postfix_str(
                    f"answered={summary['selected_answered_examples']} failed={failed} new_cost="
                    f"{'?' if cost is None else f'${cost:.2f}'}"
                )

        flush()
        tasks = [asyncio.create_task(process(example)) for example in pending]
        group = asyncio.gather(*tasks)
        try:
            # Short waits keep the elapsed time on the bar live.
            while not group.done():
                await asyncio.wait({group}, timeout=1)
                bar.refresh()
            await group
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            bar.close()

    summary = flush()
    cost = summary["run_accounting"]["total_total_cost"]
    log_line(
        f"Done: {summary['selected_answered_examples']}/{len(selected)} selected answered "
        f"({failed} failed this run) in {format_duration(time.perf_counter() - started)}; "
        f"{summary['answered_examples']} answered in saved output; "
        f"pool hits {cache_hits}; new response cost "
        f"{'unavailable' if cost is None else f'${cost:.3f}'}. "
        f"Output: {output_path}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument(
        "--inference-pool",
        type=Path,
        help="Shared response JSONL; defaults to inference_pool.jsonl beside --pool.",
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
        help="Experiment file (qids only) written by create_dataset.v1.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Cumulative predictions; saved records outside this run's selection are kept.",
    )
    parser.add_argument("--model", type=str, default="openai/gpt-5.4-nano")
    parser.add_argument(
        "--api-key",
        dest="api_key",
        type=str,
        default=None,
        help="API key. Defaults to the provider's usual environment variable.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", dest="top_p", type=float, default=1.0)
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=None,
        help="Top-k sampling. Ignored by providers that don't support it (e.g. OpenAI).",
    )
    parser.add_argument(
        "--reasoning-effort",
        dest="reasoning_effort",
        choices=["none", "minimal", "low", "medium", "high", "max"],
        default="low",
    )
    parser.add_argument(
        "--frequency-penalty", dest="frequency_penalty", type=float, default=0.0
    )
    parser.add_argument(
        "--presence-penalty", dest="presence_penalty", type=float, default=0.0
    )
    parser.add_argument("--stop", nargs="*", default=None, help="Stop sequences.")
    parser.add_argument(
        "--max-samples",
        dest="max_samples",
        type=positive_int,
        default=None,
        help=(
            "Use up to N examples per studied value, in saved order. For the "
            "reasoning-effort dataset (--vary none), use N total examples shared "
            "across efforts. Omit to use all available examples."
        ),
    )
    parser.add_argument(
        "--question-depths",
        type=positive_int,
        nargs="+",
        default=None,
        help="Select only these question depths before limiting samples (e.g. 16 32).",
    )
    parser.add_argument(
        "--component-node-counts",
        type=positive_int,
        nargs="+",
        default=None,
        help="Select only these component sizes before limiting samples (e.g. 200 250).",
    )
    parser.add_argument(
        "--max-workers",
        dest="max_workers",
        type=positive_int,
        default=1,
        help="Number of concurrent requests.",
    )
    return parser


def main() -> None:
    bootstrap_runtime()
    args = build_parser().parse_args()
    args.model_slug = args.model
    args.model = resolve_model_name(args.model)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run(args))
    # Answers are already saved. LiteLLM can leave background work that makes
    # loop.close() wait forever, so flush stdio and exit immediately instead.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
