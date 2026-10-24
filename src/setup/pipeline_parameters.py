from __future__ import annotations

from typing import Any


PARAMETERS_FIELD = "parameters"
MODEL_PARAMETER_FIELD = "model"
NUM_HOPS_PARAMETER_FIELD = "num_hops"
NEEDLE_TYPE_PARAMETER_FIELD = "needle_type"
DISTRACTING_CONTENT_TYPE_PARAMETER_FIELD = "distracting_content_type"
NEEDLE_ORDER_PARAMETER_FIELD = "needle_order"
TARGET_CONTEXT_LENGTH_PARAMETER_FIELD = "target_context_length"
QUESTION_TYPE_PARAMETER_FIELD = "question_type"
TRAVERSAL_MODE_PARAMETER_FIELD = "traversal_mode"
NUM_OPTIONS_PARAMETER_FIELD = "num_options"
CUTOFF_DATE_PARAMETER_FIELD = "cutoff_date"
ONLY_EXPLORED_PARAMETER_FIELD = "only_explored"

PIPELINE_PARAMETER_FIELDS = (
    NUM_HOPS_PARAMETER_FIELD,
    NEEDLE_TYPE_PARAMETER_FIELD,
    DISTRACTING_CONTENT_TYPE_PARAMETER_FIELD,
    NEEDLE_ORDER_PARAMETER_FIELD,
    TARGET_CONTEXT_LENGTH_PARAMETER_FIELD,
)

OPTIONAL_PIPELINE_PARAMETER_FIELDS = (
    QUESTION_TYPE_PARAMETER_FIELD,
    TRAVERSAL_MODE_PARAMETER_FIELD,
    NUM_OPTIONS_PARAMETER_FIELD,
    CUTOFF_DATE_PARAMETER_FIELD,
    ONLY_EXPLORED_PARAMETER_FIELD,
)


def build_empty_parameters() -> dict[str, Any]:
    return {field: None for field in PIPELINE_PARAMETER_FIELDS}


def normalize_parameters(raw_parameters: Any) -> dict[str, Any]:
    parameters = build_empty_parameters()
    if not isinstance(raw_parameters, dict):
        return parameters

    for field in PIPELINE_PARAMETER_FIELDS:
        if field in raw_parameters:
            parameters[field] = raw_parameters[field]

    for field in OPTIONAL_PIPELINE_PARAMETER_FIELDS:
        if field in raw_parameters:
            parameters[field] = raw_parameters[field]

    if MODEL_PARAMETER_FIELD in raw_parameters:
        parameters[MODEL_PARAMETER_FIELD] = raw_parameters[MODEL_PARAMETER_FIELD]

    return parameters


def get_parameters_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return build_empty_parameters()
    return normalize_parameters(payload.get(PARAMETERS_FIELD))


def merge_parameters(raw_parameters: Any, **updates: Any) -> dict[str, Any]:
    parameters = normalize_parameters(raw_parameters)
    for field, value in updates.items():
        if (
            field in PIPELINE_PARAMETER_FIELDS
            or field in OPTIONAL_PIPELINE_PARAMETER_FIELDS
            or field == MODEL_PARAMETER_FIELD
        ):
            parameters[field] = value
    return parameters
