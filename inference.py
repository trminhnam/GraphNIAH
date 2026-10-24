"""Run independent repeated turns and resume by (qid, turn_index).

--num-turns is a target total: changing 4 to 5 adds only turn 5. Successful
records are preserved, while failed or missing turns are requested again.
Increasing --max-samples adds examples while preserving saved turns. Previously
selected prompts must remain an unchanged prefix (after label stratification),
with the same generation settings and output file.

Each record stores normalized usage plus provider_usage, cache-aware cost,
and failed_attempts for rejected requests. Unknown prices are null. Summary
costs cover accepted responses; retry_accounting reports rejected requests
separately. Neither includes charges for requests whose usage was not returned.
Legacy total_samples counts turn records, not unique examples.
"""

import argparse
import asyncio
import hashlib
import os
import sys
import tempfile
import time
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import orjson
import requests

from build_full_prompts import (
    NO_SYSTEM_MESSAGE_FIELD,
    QUESTION_ID_FIELD,
    SYSTEM_MESSAGE_FIELD,
    USER_MESSAGE_FIELD,
)
from src.llm.accounting import (
    aggregate_accounting,
    aggregate_retry_accounting,
    response_accounting,
)
from src.llm.call_litellm import call_litellm_async
from src.llm.call_custom import CUSTOM_MODEL_PREFIX, call_llm_custom_async
from src.llm.model_info import MODEL_INFO_BY_SLUG, MODEL_SLUG_TO_API_MODEL_NAME
from src.llm.retry import (
    LLM_MAX_ATTEMPTS,
    CompletionRequestError,
    InvalidCompletionResponseError,
    calculate_retry_delay,
    is_retryable_llm_error,
    read_retry_after_seconds,
    summarize_llm_error,
)
from src.prompts import SYSTEM_PROMPT
from src.setup import bootstrap_runtime
from src.setup.pipeline_parameters import (
    MODEL_PARAMETER_FIELD,
    PARAMETERS_FIELD,
    get_parameters_from_payload,
    merge_parameters,
)
from src.utils.stratify import (
    format_label_counts,
    stratification_label,
    stratify_by_label,
)

RAW_ANSWER_FIELD = "raw_answer"
REASONING_CONTENT_FIELD = "reasoning_content"
OUTPUT_SCHEMA_VERSION = 2


def build_empty_result() -> dict[str, Any]:
    """Return the canonical empty LLM result shape."""
    return {
        "content": None,
        "reasoning_content": None,
        "usage": None,
        "cost": None,
        "latency_seconds": None,
        "attempts": None,
        "started_at": None,
        "finished_at": None,
        "provider_usage": None,
        "service_tier": None,
        "failed_attempts": [],
    }


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def build_prompt_record(
    item: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    """Return ``qid`` plus the final user message for one full-prompt item."""
    existing_user_message = item.get(USER_MESSAGE_FIELD)
    qid = item.get(QUESTION_ID_FIELD)

    if not isinstance(qid, str) or not qid.strip():
        raise ValueError(
            f"Input item at index {index} must contain a non-empty '{QUESTION_ID_FIELD}' string."
        )
    if not isinstance(existing_user_message, str) or not existing_user_message.strip():
        raise ValueError(
            f"Input item at index {index} must contain a non-empty '{USER_MESSAGE_FIELD}' string."
        )

    record = {
        QUESTION_ID_FIELD: qid.strip(),
        USER_MESSAGE_FIELD: existing_user_message.strip(),
    }
    system_prompt = item.get(SYSTEM_MESSAGE_FIELD)
    if isinstance(system_prompt, str) and system_prompt.strip():
        record[SYSTEM_MESSAGE_FIELD] = system_prompt.strip()
    if item.get(NO_SYSTEM_MESSAGE_FIELD) is True:
        record[NO_SYSTEM_MESSAGE_FIELD] = True
    return record


def build_completion_kwargs(
    args: argparse.Namespace,
    user_prompt: str,
    system_prompt: str | None = SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Translate CLI args into one async completion request."""
    messages = [{"role": "user", "content": user_prompt}]
    if system_prompt is not None:
        messages.insert(0, {"role": "system", "content": system_prompt})
    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty,
        "reasoning_effort": args.reasoning_effort,
    }
    if args.api_key is not None:
        kwargs["api_key"] = args.api_key
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    if args.stop:
        kwargs["stop"] = args.stop
    return kwargs


def resolve_model_name(model_name: str) -> str:
    """Return the API model name for a known slug, else preserve the input."""
    api_model_name = MODEL_SLUG_TO_API_MODEL_NAME.get(model_name)
    if api_model_name:
        return api_model_name
    return model_name


def get_completion_field(value: Any, field: str) -> Any:
    """Return one field from a response object or dictionary."""
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def validate_completion_content(content: Any) -> str:
    """Return usable assistant text or raise a retryable response error."""
    if not isinstance(content, str) or not content.strip():
        raise InvalidCompletionResponseError(
            "completion response has null or empty assistant content"
        )
    return content


def extract_completion_result(
    response: Any,
    *,
    model: str,
) -> dict[str, Any]:
    """Return assistant content plus normalized usage/cost blocks."""
    if response is None:
        raise InvalidCompletionResponseError("LiteLLM returned a null response")
    choices = get_completion_field(response, "choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        raise InvalidCompletionResponseError("LiteLLM response has no choices")
    message = get_completion_field(choices[0], "message")
    if message is None:
        raise InvalidCompletionResponseError(
            "LiteLLM response has no assistant message"
        )
    content = validate_completion_content(get_completion_field(message, "content"))
    # Only some providers (e.g. DeepSeek, Gemini) return reasoning traces;
    # LiteLLM normalises them onto ``message.reasoning_content`` when present.
    reasoning_content = get_completion_field(message, "reasoning_content")

    return {
        "content": content,
        "reasoning_content": reasoning_content,
        **response_accounting(response, model),
    }


def extract_custom_completion_result(response: requests.Response) -> dict[str, Any]:
    """Return normalized content and usage from a custom HTTP response."""
    if response is None:
        raise InvalidCompletionResponseError("CUSTOM returned a null response")
    try:
        payload = response.json()
    except ValueError as error:
        raise InvalidCompletionResponseError(
            "CUSTOM response body is not valid JSON"
        ) from error
    if not isinstance(payload, dict):
        raise InvalidCompletionResponseError("CUSTOM response body is null or malformed")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise InvalidCompletionResponseError("CUSTOM response has no choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise InvalidCompletionResponseError("CUSTOM response choice is malformed")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise InvalidCompletionResponseError("CUSTOM response has no assistant message")
    content = validate_completion_content(message.get("content"))
    return {
        "content": content,
        "reasoning_content": message.get("reasoning_content"),
        **response_accounting(payload, "custom/"),
    }


async def call_completion_with_retry(
    completion_kwargs: dict[str, Any],
    *,
    use_custom_router: bool,
    model: str,
    question_id: str,
    semaphore: asyncio.Semaphore | None = None,
) -> dict[str, Any]:
    """Call one LLM route and retry transient or invalid responses.

    The result also times the successful attempt. Its clock starts when that
    request is sent, so waits for a worker slot, the CUSTOM rate-limit wait,
    a flex request rejected before the standard-tier fallback (not used for
    Gemini, which stays on flex) and failed attempts are not counted in
    ``latency_seconds``.
    """
    failed_attempts = []
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        response = None
        try:
            async with semaphore if semaphore is not None else nullcontext():
                sent: dict[str, Any] = {}

                def mark_sent(sent: dict[str, Any] = sent) -> None:
                    sent["at"] = utc_now_iso()
                    sent["clock"] = time.perf_counter()

                mark_sent()
                if use_custom_router:
                    # call_llm_custom marks again after its rate-limit wait.
                    response = await call_llm_custom_async(**completion_kwargs, on_send=mark_sent)
                else:
                    litellm_kwargs = dict(completion_kwargs)
                    litellm_kwargs["num_retries"] = 0
                    # call_litellm marks again before a flex-to-standard fallback
                    # (GPT only; a Gemini flex 503 is retried here, still on flex).
                    response = await call_litellm_async(**litellm_kwargs, on_send=mark_sent)
                latency_seconds = time.perf_counter() - sent["clock"]
                finished_at = utc_now_iso()

            if use_custom_router:
                result = extract_custom_completion_result(response)
            else:
                result = extract_completion_result(response, model=model)
            return {
                **result,
                "latency_seconds": latency_seconds,
                "attempts": attempt,
                "failed_attempts": failed_attempts,
                "started_at": sent["at"],
                "finished_at": finished_at,
            }
        except Exception as error:
            failed = {"attempt": attempt, "error": summarize_llm_error(error)}
            if response is not None:
                if use_custom_router:
                    try:
                        payload = response.json()
                    except ValueError:
                        payload = None
                    failed.update(response_accounting(payload, model))
                else:
                    failed.update(response_accounting(response, model))
            failed_attempts.append(failed)
            should_retry = is_retryable_llm_error(error)
            if not should_retry or attempt >= LLM_MAX_ATTEMPTS:
                raise CompletionRequestError(attempt, error, failed_attempts) from error

            retry_after = read_retry_after_seconds(error)
            delay = calculate_retry_delay(attempt, retry_after)
            reason = summarize_llm_error(error)
            print(
                f"[Retry] {question_id}: {reason}; "
                f"retrying attempt {attempt + 1}/{LLM_MAX_ATTEMPTS} "
                f"in {delay:.1f}s.",
                flush=True,
            )
            await asyncio.sleep(delay)

    raise RuntimeError("LLM retry loop ended unexpectedly")


def load_items(path: Path) -> list[dict]:
    """Load items from either a bare list or a ``{"questions": [...]}`` wrapper."""
    data = orjson.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    raise ValueError(
        f"Unrecognised JSON structure at {path}: "
        "expected a list or an object with a 'questions' key."
    )


def load_checkpoint(out_path: Path) -> tuple[list[dict], dict]:
    """Never silently overwrite a malformed checkpoint."""
    if not out_path.exists():
        return [], {}
    data = orjson.loads(out_path.read_bytes())
    if isinstance(data, list):
        return data, {}
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        schema_version = data.get("schema_version", 1)
        if schema_version not in (1, OUTPUT_SCHEMA_VERSION):
            raise ValueError(f"Unsupported checkpoint schema version: {schema_version}")
        return data["records"], data.get("metadata") or {}
    raise ValueError(f"Invalid inference checkpoint: {out_path}")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def has_non_empty_raw_answer(record: dict[str, Any]) -> bool:
    """Return whether an output record has a completed raw answer."""
    raw_answer = record.get(RAW_ANSWER_FIELD)
    return isinstance(raw_answer, str) and bool(raw_answer.strip())


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run LLM inference on a full-prompt JSON file. Each prediction record "
            "stores the question qid and the raw model answer."
        ),
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Input JSON from build_full_prompts.py. Accepts either a bare list "
            "or {'questions': [...]} where each item contains 'qid' and "
            "'user_message'."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Output JSON path. The payload contains metadata plus one record per "
            "turn with its qid, turn_index, and raw model answer."
        ),
    )

    parser.add_argument(
        "--num-turns",
        "--num_turns",
        type=positive_int,
        default=1,
        help="Target number of independent turns per example. Resume runs only missing turns.",
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
        "--reasoning_effort",
        dest="reasoning_effort",
        choices=["none", "minimal", "low", "high", "max"],
        default="low",
        help="Reasoning effort passed to the model request. Default 'low'.",
    )
    parser.add_argument("--frequency-penalty", dest="frequency_penalty", type=float, default=0.0)
    parser.add_argument("--presence-penalty", dest="presence_penalty", type=float, default=0.0)
    parser.add_argument(
        "--stop",
        nargs="*",
        default=None,
        help="Stop sequences, space-separated.",
    )

    parser.add_argument(
        "--question-position",
        "--question_position",
        dest="question_position",
        choices=["first", "last"],
        default="first",
        help=(
            "Deprecated. Prompt ordering must already be decided in "
            "build_full_prompts.py; inference.py always uses the provided "
            "'user_message' as-is and only appends the reminder."
        ),
    )

    parser.add_argument(
        "--max-samples",
        "--max_samples",
        dest="max_samples",
        type=int,
        default=None,
        help="Run inference only on the first N items after label stratification. Increasing N resumes saved turns.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Deprecated. Checkpointing now happens automatically when --output " "already exists."
        ),
    )
    parser.add_argument(
        "--max-workers",
        "--max_workers",
        dest="max_workers",
        type=positive_int,
        default=1,
        help=(
            "Number of concurrent examples. Each worker finishes all turns of one "
            "example before starting another. Default 1 (serial)."
        ),
    )
    return parser


def format_duration(seconds: float) -> str:
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


async def run_inference(args: argparse.Namespace) -> None:
    """Run independent turns, checkpointing each (qid, turn_index)."""
    started = time.perf_counter()
    num_turns = getattr(args, "num_turns", 1)
    if num_turns < 1 or args.max_workers < 1:
        raise ValueError("num_turns and max_workers must be positive")
    in_path, out_path = Path(args.input), Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    input_payload = orjson.loads(in_path.read_bytes())
    items = stratify_by_label(load_items(in_path), stratification_label)
    if args.max_samples is not None:
        items = items[: args.max_samples]
    prompt_records = [
        build_prompt_record(item, index) for index, item in enumerate(items)
    ]
    record_order = {
        record[QUESTION_ID_FIELD]: index for index, record in enumerate(prompt_records)
    }
    if len(record_order) != len(prompt_records):
        raise ValueError("Input contains duplicate qid values")
    labels_by_qid = {
        prompt[QUESTION_ID_FIELD]: stratification_label(item)
        for prompt, item in zip(prompt_records, items)
    }
    selected_label_counts = Counter(
        label for label in labels_by_qid.values() if label is not None
    )
    generation_settings = build_completion_kwargs(args, "")
    generation_settings.pop("messages")
    generation_settings.pop("api_key", None)
    generation_settings.update(top_k=args.top_k, stop=args.stop)
    existing_records, existing_metadata = load_checkpoint(out_path)
    saved_fingerprint = existing_metadata.get("request_fingerprint")
    saved_example_count = existing_metadata.get("total_examples")
    fingerprint = hashlib.sha256(
        orjson.dumps(generation_settings, option=orjson.OPT_SORT_KEYS)
    )
    prefix_fingerprint = fingerprint.hexdigest() if saved_example_count == 0 else None
    for index, prompt in enumerate(prompt_records, start=1):
        system_prompt = (
            None
            if prompt.get(NO_SYSTEM_MESSAGE_FIELD)
            else prompt.get(SYSTEM_MESSAGE_FIELD, SYSTEM_PROMPT)
        )
        fingerprint.update(
            orjson.dumps(
                [prompt[QUESTION_ID_FIELD], system_prompt, prompt[USER_MESSAGE_FIELD]]
            )
        )
        # Verify all previously selected prompts, including unfinished examples.
        # This also supports existing checkpoints without changing their schema.
        if index == saved_example_count:
            prefix_fingerprint = fingerprint.hexdigest()
    request_fingerprint = fingerprint.hexdigest()
    if saved_fingerprint is not None and saved_fingerprint not in (
        request_fingerprint,
        prefix_fingerprint,
    ):
        raise ValueError(
            "Cannot resume: prompts or generation settings differ from the checkpoint"
        )
    if existing_records and saved_fingerprint is None:
        for field, value in generation_settings.items():
            if field in existing_metadata and existing_metadata[field] != value:
                raise ValueError(f"Cannot resume: checkpoint {field} differs")
        print(
            "Legacy checkpoint: full prompt/settings compatibility cannot be verified; historical costs are unchanged."
        )

    records: dict[tuple[str, int], dict] = {}
    for original in existing_records:
        record = dict(original)
        qid = record.get(QUESTION_ID_FIELD)
        if not qid and isinstance(record.get("index"), int):
            index = record["index"]
            if 0 <= index < len(prompt_records):
                qid = prompt_records[index][QUESTION_ID_FIELD]
        qid = str(qid or "").strip()
        if qid not in record_order:
            raise ValueError(f"Checkpoint qid is not in the selected input: {qid!r}")
        turn = record.get("turn_index", 1)
        if type(turn) is not int or turn < 1:
            raise ValueError(f"Invalid turn_index for {qid}")
        if turn > num_turns:
            raise ValueError(
                f"Cannot resume with num_turns={num_turns}: checkpoint contains turn {turn}"
            )
        key = (qid, turn)
        if key in records:
            raise ValueError(f"Duplicate checkpoint turn: {key}")
        record.update(qid=qid, turn_index=turn, inference_id=f"{qid}::turn-{turn}")
        if RAW_ANSWER_FIELD not in record and "model_output" in record:
            record[RAW_ANSWER_FIELD] = record["model_output"]
        records[key] = record

    done = {key for key, record in records.items() if has_non_empty_raw_answer(record)}
    completed_qids = {
        qid
        for qid in record_order
        if all((qid, turn) in done for turn in range(1, num_turns + 1))
    }
    completed_counts = Counter(
        labels_by_qid[qid] for qid in completed_qids if labels_by_qid[qid] is not None
    )
    pending = stratify_by_label(
        [
            prompt
            for prompt in prompt_records
            if prompt[QUESTION_ID_FIELD] not in completed_qids
        ],
        lambda prompt: labels_by_qid[prompt[QUESTION_ID_FIELD]],
        completed_counts,
    )
    model_slug = getattr(args, "model_slug", args.model)
    model_info = MODEL_INFO_BY_SLUG.get(model_slug) or next(
        (
            info
            for info in MODEL_INFO_BY_SLUG.values()
            if info.api_model_name == args.model
        ),
        None,
    )
    total_turns = len(prompt_records) * num_turns
    print("\n[Inference run]")
    print(f"  Model           : {model_info.model if model_info else model_slug}")
    print(f"  API model       : {args.model}")
    if model_info is not None:
        print(f"  Provider        : {model_info.provider or 'unspecified'}")
        print(f"  Context limit   : {model_info.native_context_length:,} tokens")
    print(f"  Reasoning effort: {args.reasoning_effort} (requested)")
    print(f"  Examples        : {len(prompt_records)}")
    if selected_label_counts:
        print(f"  Labels          : {format_label_counts(selected_label_counts)}")
    print(f"  Target          : {num_turns} turns/example; {total_turns} turns total")
    print(f"  Workers         : {args.max_workers} concurrent examples (maximum)")
    print(
        f"  Resume          : {len(done)} successful turns reused; "
        f"{len(completed_qids)} examples complete"
    )
    print(f"  Remaining       : {total_turns - len(done)} turns")
    print(f"  Output          : {out_path}\n", flush=True)
    litellm_version = version("litellm")
    output_parameters = merge_parameters(
        get_parameters_from_payload(input_payload),
        **{MODEL_PARAMETER_FIELD: model_slug},
    )
    legacy_unverified = existing_metadata.get(
        "legacy_checkpoint_unverified", False
    ) or bool(existing_records and saved_fingerprint is None)

    def ordered_records() -> list[dict]:
        return sorted(
            records.values(),
            key=lambda record: (
                record_order[record[QUESTION_ID_FIELD]],
                record["turn_index"],
            ),
        )

    def aggregate_metadata(ordered: list[dict]) -> dict[str, Any]:
        successful = {
            (record[QUESTION_ID_FIELD], record["turn_index"])
            for record in ordered
            if has_non_empty_raw_answer(record)
        }
        return {
            **generation_settings,
            "input": str(in_path),
            "request_fingerprint": request_fingerprint,
            "legacy_checkpoint_unverified": legacy_unverified,
            "litellm_version": litellm_version,
            "max_samples": args.max_samples,
            "max_workers": args.max_workers,
            "num_turns": num_turns,
            "total_examples": len(prompt_records),
            "completed_examples": sum(
                all((qid, turn) in successful for turn in range(1, num_turns + 1))
                for qid in record_order
            ),
            "requested_turns": len(prompt_records) * num_turns,
            "successful_turns": len(successful),
            "failed_turns": len(ordered) - len(successful),
            "total_samples": len(ordered),  # Legacy name: number of turn records.
            **aggregate_accounting(ordered),
            "retry_accounting": aggregate_retry_accounting(ordered),
        }

    def flush() -> None:
        ordered = ordered_records()
        payload = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "metadata": aggregate_metadata(ordered),
            PARAMETERS_FIELD: output_parameters,
            "records": ordered,
        }
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=out_path.parent, prefix=f".{out_path.name}.", delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, out_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    semaphore = asyncio.Semaphore(args.max_workers)
    pending_turns = total_turns - len(done)
    processed_turns = 0
    failed_turns = 0

    async def process_one(prompt_record: dict[str, Any]) -> None:
        nonlocal processed_turns, failed_turns
        async with semaphore:
            qid = prompt_record[QUESTION_ID_FIELD]
            system_prompt = (
                None
                if prompt_record.get(NO_SYSTEM_MESSAGE_FIELD)
                else prompt_record.get(SYSTEM_MESSAGE_FIELD, SYSTEM_PROMPT)
            )
            for turn in range(1, num_turns + 1):
                key = (qid, turn)
                if key in done:
                    continue
                inference_id = f"{qid}::turn-{turn}"
                # Fresh messages protect subsequent turns from provider-side mutation.
                completion_kwargs = build_completion_kwargs(
                    args, prompt_record[USER_MESSAGE_FIELD], system_prompt
                )
                previous = records.get(key, {})
                error_message = None
                try:
                    result = await call_completion_with_retry(
                        completion_kwargs,
                        use_custom_router=args.model.startswith(CUSTOM_MODEL_PREFIX),
                        model=args.model,
                        question_id=inference_id,
                    )
                except CompletionRequestError as error:
                    error_message = str(error)
                    print(
                        f"[Failed] example {record_order[qid] + 1}, turn {turn}: "
                        f"{error.attempts} attempt(s): {error}.",
                        flush=True,
                    )
                    result = {
                        **build_empty_result(),
                        "attempts": error.attempts,
                        "failed_attempts": error.failed_attempts,
                    }
                prior_attempts = previous.get("attempts") or 0
                failed_attempts = previous.get("failed_attempts", []) + [
                    {**attempt, "attempt": prior_attempts + attempt["attempt"]}
                    for attempt in result["failed_attempts"]
                ]
                records[key] = {
                    QUESTION_ID_FIELD: qid,
                    "turn_index": turn,
                    "inference_id": inference_id,
                    "litellm_version": litellm_version,
                    RAW_ANSWER_FIELD: result.pop("content"),
                    **result,
                    "attempts": prior_attempts + result["attempts"],
                    "failed_attempts": failed_attempts,
                    "error": error_message,
                }
                flush()
                processed_turns += 1
                failed_turns += int(error_message is not None)
                elapsed = time.perf_counter() - started
                remaining = (
                    elapsed / processed_turns * (pending_turns - processed_turns)
                )
                status = "FAILED" if error_message else "OK"
                print(
                    f"  [{processed_turns}/{pending_turns} | {processed_turns / pending_turns:.0%}] "
                    f"{status} example {record_order[qid] + 1}/{len(prompt_records)}, "
                    f"turn {turn}/{num_turns} | "
                    f"successful {len(done) + processed_turns - failed_turns}/{total_turns} | "
                    f"failed {failed_turns} | elapsed {format_duration(elapsed)} | "
                    f"ETA ~{format_duration(remaining)}",
                    flush=True,
                )

    # Persist migration and the new target even if all requested turns already exist.
    flush()
    tasks = [asyncio.create_task(process_one(prompt)) for prompt in pending]
    group = asyncio.gather(*tasks)
    try:
        while not group.done():
            await asyncio.wait({group}, timeout=30)
            if not group.done():
                print(
                    f"  [Waiting] {model_slug} | {processed_turns}/{pending_turns} turns saved "
                    f"this run | elapsed {format_duration(time.perf_counter() - started)} "
                    "| requests running or waiting to retry",
                    flush=True,
                )
        await group
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    summary = aggregate_metadata(ordered_records())
    print(
        f"\n[Run complete] {model_slug} | "
        f"elapsed {format_duration(time.perf_counter() - started)}"
    )
    print(
        f"Examples complete: {summary['completed_examples']}/{summary['total_examples']}"
    )
    print(
        f"Turns: {summary['successful_turns']} successful, {summary['failed_turns']} failed, {summary['requested_turns']} requested"
    )
    cost_fields = (
        ("Input", "input_cost"),
        ("Output", "output_cost"),
        ("Total", "total_cost"),
    )
    cost_parts = []
    for label, field in cost_fields:
        amount = summary[f"total_{field}"]
        cost_text = "unavailable" if amount is None else f"${amount:.3f}"
        cost_parts.append(f"{label} Cost: {cost_text}")
    print(" | ".join(cost_parts))
    if any(
        summary[f"records_with_{field}"] < summary["total_samples"]
        for _, field in cost_fields
    ):
        print(
            "Cost coverage (known turns): "
            + " | ".join(
                f"{label}: {summary[f'records_with_{field}']}/{summary['total_samples']}"
                for label, field in cost_fields
            )
            + ". Amounts include only known costs."
        )
    retry_summary = summary["retry_accounting"]
    retry_cost = retry_summary["total_total_cost"]
    retry_cost_text = "unavailable" if retry_cost is None else f"${retry_cost:.3f}"
    if retry_summary["total_attempts"] == 0:
        retry_cost_text = "$0.000"
    print(
        f"Rejected requests: {retry_summary['total_attempts']}; known cost: {retry_cost_text}; unknown cost: {retry_summary['records_without_cost']}"
    )
    print(f"Output: {out_path}\n", flush=True)


def main() -> None:
    bootstrap_runtime()
    args = arg_parser().parse_args()
    args.model_slug = args.model
    args.model = resolve_model_name(args.model)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_inference(args))

    # All records are already durably written to --output by run_inference()'s
    # flush(). Do not close the loop here: LiteLLM can leave background work
    # that makes asyncio.run() or loop.close() wait forever after inference has
    # already finished. Flush stdio and exit immediately instead.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
