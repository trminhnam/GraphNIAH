from __future__ import annotations

from threading import Lock
from typing import Any

from loguru import logger

usage_lock = Lock()


def litellm_cost_to_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def litellm_count_to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def normalize_litellm_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(usage, dict):
        usage = {}
    total_cost = litellm_cost_to_float(usage.get("total_cost_usd", 0))
    call_count = litellm_count_to_int(usage.get("call_count", 0))
    successful_call_count = litellm_count_to_int(usage.get("successful_call_count", call_count))

    cost_by_model: dict[str, float] = {}
    raw_cost_by_model = usage.get("cost_by_model_usd", {}) or {}
    if isinstance(raw_cost_by_model, dict):
        for model, cost in raw_cost_by_model.items():
            cost_by_model[str(model)] = round(litellm_cost_to_float(cost), 10)

    return {
        "total_cost_usd": round(total_cost, 10),
        "call_count": call_count,
        "successful_call_count": successful_call_count,
        "cost_by_model_usd": cost_by_model,
    }


# Initialize with the canonical normalized shape so record_litellm_usage can
# mutate in place without defensive key checks.
litellm_usage_baseline: dict[str, Any] = normalize_litellm_usage(None)
litellm_usage_delta: dict[str, Any] = normalize_litellm_usage(None)


def combine_litellm_usage(*usages: dict[str, Any] | None) -> dict[str, Any]:
    total_cost = 0.0
    call_count = 0
    successful_call_count = 0
    cost_by_model: dict[str, float] = {}

    for usage in usages:
        normalized = normalize_litellm_usage(usage)
        total_cost += normalized["total_cost_usd"]
        call_count += normalized["call_count"]
        successful_call_count += normalized["successful_call_count"]
        for model, cost in normalized["cost_by_model_usd"].items():
            cost_by_model[model] = cost_by_model.get(model, 0.0) + cost

    return {
        "total_cost_usd": round(total_cost, 10),
        "call_count": call_count,
        "successful_call_count": successful_call_count,
        "cost_by_model_usd": {
            model: round(cost, 10) for model, cost in sorted(cost_by_model.items())
        },
    }


def reset_litellm_usage() -> None:
    """Zero both the baseline and the running delta."""
    global litellm_usage_baseline, litellm_usage_delta
    with usage_lock:
        litellm_usage_baseline = normalize_litellm_usage(None)
        litellm_usage_delta = normalize_litellm_usage(None)


def seed_litellm_usage(usage: dict[str, Any] | None) -> None:
    """Start tracking from saved usage and clear this run's delta."""
    global litellm_usage_baseline, litellm_usage_delta
    with usage_lock:
        litellm_usage_baseline = normalize_litellm_usage(usage)
        litellm_usage_delta = normalize_litellm_usage(None)


def get_litellm_usage_delta() -> dict[str, Any]:
    with usage_lock:
        return normalize_litellm_usage(litellm_usage_delta)


def get_litellm_usage() -> dict[str, Any]:
    with usage_lock:
        return combine_litellm_usage(litellm_usage_baseline, litellm_usage_delta)


def record_litellm_usage(
    model: str | None,
    response_cost: Any,
    *,
    successful: bool,
) -> dict[str, Any]:
    cost = litellm_cost_to_float(response_cost)
    model_key = model or "unknown"

    with usage_lock:
        delta = litellm_usage_delta
        delta["total_cost_usd"] = round(
            litellm_cost_to_float(delta.get("total_cost_usd", 0.0)) + cost, 10
        )
        delta["call_count"] = litellm_count_to_int(delta.get("call_count", 0)) + 1
        delta["successful_call_count"] = litellm_count_to_int(delta.get("successful_call_count", 0))
        if successful:
            delta["successful_call_count"] += 1
        cost_by_model = delta.setdefault("cost_by_model_usd", {})
        cost_by_model[model_key] = round(
            litellm_cost_to_float(cost_by_model.get(model_key, 0.0)) + cost, 10
        )
        return combine_litellm_usage(litellm_usage_baseline, delta)


def track_cost_callback(
    kwargs,  # kwargs to completion
    completion_response,  # response from completion
    start_time,
    end_time,  # start/end time
):
    try:
        response_cost = kwargs.get("response_cost", 0)
        model = kwargs.get("model")
        is_failure = kwargs.get("log_event_type") == "failed_api_call" or "exception" in kwargs
        usage = record_litellm_usage(
            model,
            response_cost,
            successful=not is_failure,
        )
        duration = end_time - start_time
        duration_seconds = (
            duration.total_seconds() if hasattr(duration, "total_seconds") else duration
        )
        logger.info(
            "LiteLLM {} from model {} with cost ${:.6f} in {:.2f} seconds. Total: ${:.6f} across {} successful request(s).",
            "failed call" if is_failure else "response",
            model,
            litellm_cost_to_float(response_cost),
            duration_seconds,
            usage["total_cost_usd"],
            usage["successful_call_count"],
        )
    except Exception:
        logger.exception("LiteLLM cost callback failed")


def setup_litellm() -> None:
    """Configure LiteLLM callbacks and custom model metadata."""
    import litellm

    reset_litellm_usage()
    # Register both paths. LiteLLM usually reports failure cost as zero, but if
    # it can calculate one we still want the total to include it.
    litellm.success_callback = [track_cost_callback]
    litellm.failure_callback = [track_cost_callback]
    litellm.suppress_debug_info = True
    litellm.drop_params = True
    litellm.turn_off_message_logging = True
    litellm.set_verbose = False

    # Custom-priced model. `gpt-5.4-nano` is an internal alias served by the
    # LiteLLM proxy in front of OpenAI; pricing here mirrors the proxy's billing
    # so cost-tracking matches the actual invoice. If a run fails with
    # "model not found", check the proxy's alias map — the alias must be
    # configured upstream as well as registered here.
    # model_configs = {
    #     "gpt-5.4": {
    #         "input_cost_per_token": 2.5 / 1_000_000,
    #         "output_cost_per_token": 15 / 1_000_000,
    #         "cache_read_input_token_cost": 0.25 / 1_000_000,
    #         "litellm_provider": "openai",
    #         "max_tokens": 1_050_000,
    #         "max_output_tokens": 128_000,
    #     },
    #     "gpt-5.4-mini": {
    #         "input_cost_per_token": 0.75 / 1_000_000,
    #         "output_cost_per_token": 4.5 / 1_000_000,
    #         "cache_read_input_token_cost": 0.075 / 1_000_000,
    #         "litellm_provider": "openai",
    #         "max_tokens": 400_000,
    #         "max_output_tokens": 128_000,
    #     },
    #     "gpt-5.4-nano": {
    #         "input_cost_per_token": 0.20 / 1_000_000,
    #         "output_cost_per_token": 1.25 / 1_000_000,
    #         "cache_read_input_token_cost": 0.02 / 1_000_000,
    #         "litellm_provider": "openai",
    #         "max_tokens": 400_000,
    #         "max_output_tokens": 128_000,
    #     },
    # }
    # for model, config in model_configs.items():
    #     litellm.register_model({model: config})
