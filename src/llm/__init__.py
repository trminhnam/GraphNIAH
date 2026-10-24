from __future__ import annotations

import importlib


MODEL_INFO_EXPORTS = {
    "DEFAULT_LLM_MANAGER",
    "DEFAULT_LLM_MODELS_PATH",
    "LlmInfo",
    "LlmManager",
    "MODEL_ID_TO_MODEL_SLUG",
    "MODEL_SLUG_TO_API_MODEL_NAME",
    "MODEL_INFO_BY_SLUG",
    "MODEL_MAX_LENGTHS",
    "MODEL_SLUG_TO_MODEL_ID",
}
CALL_LITELLM_EXPORTS = {
    "call_litellm",
    "call_litellm_async",
    "prepare_litellm_kwargs",
}
CALL_CUSTOM_EXPORTS = {"call_llm_custom", "call_llm_custom_async"}


def __getattr__(name: str):
    if name in MODEL_INFO_EXPORTS:
        from . import model_info

        return getattr(model_info, name)
    if name in CALL_LITELLM_EXPORTS:
        call_litellm_module = importlib.import_module(".call_litellm", __name__)
        return getattr(call_litellm_module, name)
    if name in CALL_CUSTOM_EXPORTS:
        call_custom_module = importlib.import_module(".call_custom", __name__)
        return getattr(call_custom_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    *sorted(MODEL_INFO_EXPORTS),
    *sorted(CALL_LITELLM_EXPORTS),
    *sorted(CALL_CUSTOM_EXPORTS),
]
