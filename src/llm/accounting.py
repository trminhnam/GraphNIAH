"""Preserve response usage and report known costs without treating unknowns as zero."""

from __future__ import annotations

import math
from typing import Any


def response_field(value: Any, field: str) -> Any:
    return value.get(field) if isinstance(value, dict) else getattr(value, field, None)


def usage_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return vars(value).copy() if hasattr(value, "__dict__") else {}


def known_number(value: Any) -> float | None:
    """Accept only finite, nonnegative numeric measurements, including zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def response_accounting(response: Any, model: str) -> dict[str, Any]:
    """Price the full response; a pricing error must never invalidate an answer."""
    provider_usage = usage_dict(response_field(response, "usage"))
    usage = {
        target: provider_usage.get(source)
        for target, source in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        )
    }
    prompt_details = provider_usage.get("prompt_tokens_details") or {}
    completion_details = provider_usage.get("completion_tokens_details") or {}
    usage.update(
        cache_read_tokens=response_field(prompt_details, "cached_tokens"),
        cache_write_tokens=provider_usage.get("cache_creation_input_tokens"),
        reasoning_tokens=response_field(completion_details, "reasoning_tokens"),
    )
    if usage["cache_read_tokens"] is None:
        usage["cache_read_tokens"] = provider_usage.get("cache_read_input_tokens")
    if usage["cache_write_tokens"] is None:
        usage["cache_write_tokens"] = response_field(
            prompt_details, "cache_write_tokens"
        )
    if usage["cache_write_tokens"] is None:
        usage["cache_write_tokens"] = response_field(
            prompt_details, "cache_creation_tokens"
        )
    usage = {
        key: value for key, value in usage.items() if known_number(value) is not None
    }
    if (
        "total_tokens" not in usage
        and {"input_tokens", "output_tokens"} <= usage.keys()
    ):
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    tier = response_field(response, "service_tier") or provider_usage.get(
        "service_tier"
    )
    cost = {
        "input_cost": None,
        "output_cost": None,
        "total_cost": None,
        "source": "unavailable",
    }
    accounting = {
        "usage": usage or None,
        "provider_usage": provider_usage or None,
        "service_tier": tier,
        "cost": cost,
    }
    if model.startswith("custom/"):
        cost["error"] = "No CUSTOM pricing rule configured"
        return accounting

    import litellm

    try:
        # Without usage, completion_cost may estimate tokens from response text.
        # That is not a reliable measurement of this request's billable usage.
        if not {"input_tokens", "output_tokens"} <= usage.keys():
            raise ValueError("Missing response token counts")
        total = known_number(
            litellm.completion_cost(completion_response=response, model=model)
        )
        if total is None:
            raise ValueError("Invalid completion cost")
        cost.update(total_cost=total, source="litellm_completion_cost")
    except Exception as error:  # noqa: BLE001 - pricing failures must not retry a paid request
        hidden = response_field(response, "_hidden_params") or {}
        total = known_number(response_field(hidden, "response_cost"))
        cost["error"] = type(error).__name__  # Never log prompts or credentials.
        if total is not None:
            cost.update(total_cost=total, source="litellm_response_cost")
        else:
            return accounting

    try:
        input_cost, output_cost = litellm.cost_per_token(
            model=model,
            usage_object=litellm.Usage(**provider_usage),
            service_tier=tier,
            response=response,
        )
        input_cost, output_cost = known_number(input_cost), known_number(output_cost)
        if (
            input_cost is not None
            and output_cost is not None
            and math.isclose(
                input_cost + output_cost, total, rel_tol=1e-7, abs_tol=1e-12
            )
        ):
            cost.update(input_cost=input_cost, output_cost=output_cost)
        else:
            cost["breakdown_error"] = "Breakdown does not match total"
    except Exception as error:  # noqa: BLE001 - pricing failures must not retry a paid request
        cost["breakdown_error"] = type(error).__name__
    return accounting


def aggregate_accounting(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep legacy summary names, using a separate denominator for each field."""
    summary: dict[str, Any] = {
        "samples_with_usage": sum(bool(record.get("usage")) for record in records),
    }
    for block, fields in (
        (
            "usage",
            (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "reasoning_tokens",
            ),
        ),
        ("cost", ("input_cost", "output_cost", "total_cost")),
    ):
        for field in fields:
            values = [
                value
                for record in records
                if (value := known_number((record.get(block) or {}).get(field)))
                is not None
            ]
            summary[f"total_{field}"] = sum(values) if values else None
            summary[f"avg_{field}"] = sum(values) / len(values) if values else None
            summary[f"records_with_{field}"] = len(values)
    known = summary["records_with_total_cost"]
    summary.update(
        records_with_cost=known,
        records_without_cost=len(records) - known,
        cost_complete=known == len(records),
    )
    return summary


def aggregate_retry_accounting(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report rejected requests separately from accepted responses."""
    attempts = [
        attempt for record in records for attempt in record.get("failed_attempts", [])
    ]
    return {"total_attempts": len(attempts), **aggregate_accounting(attempts)}
