"""LLM model metadata loaded from a YAML resource file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - exercised when OmegaConf is absent.
    OmegaConf = None


def resolve_default_llm_models_path() -> Path:
    return Path(__file__).resolve().parents[2] / "resources" / "llm_models.yaml"


def normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None

    return text


def parse_optional_price(value: Any) -> float | None:
    text = normalize_optional_string(value)
    if text is None:
        return None

    return float(text.replace(" ", "").replace(",", ""))


def parse_context_length(value: Any) -> int:
    return int(str(value).replace(",", "").strip())


def parse_io_pricing(value: Any) -> tuple[float | None, float | None]:
    text = normalize_optional_string(value)
    if text is None:
        return None, None

    compact_text = text.upper().replace(" ", "")
    if compact_text == "N/A/N/A":
        return None, None

    normalized_text = (
        text.replace(" / ", "|").replace(" /", "|").replace("/ ", "|")
    )
    separator = "|" if "|" in normalized_text else "/"
    parts = [part.strip() for part in normalized_text.split(separator, maxsplit=1)]
    if len(parts) != 2:
        raise ValueError(f"Invalid io_pricing value: {value!r}")

    return parse_optional_price(parts[0]), parse_optional_price(parts[1])


@dataclass(slots=True, frozen=True)
class LlmInfo:
    slug: str
    model: str
    api_model_name: str | None
    hf_model_id: str
    knowledge_cutoff_date: str | None
    native_context_length: int
    input_price: float | None
    output_price: float | None
    provider: str | None = None
    base_url: str | None = None

    @classmethod
    def from_dict(cls, slug: str, raw_data: dict[str, Any]) -> "LlmInfo":
        if "io_pricing" in raw_data:
            input_price, output_price = parse_io_pricing(raw_data["io_pricing"])
        else:
            input_price = parse_optional_price(raw_data.get("input_price"))
            output_price = parse_optional_price(raw_data.get("output_price"))

        return cls(
            slug=slug,
            model=str(raw_data["model"]).strip(),
            api_model_name=normalize_optional_string(raw_data.get("api_model_name")),
            hf_model_id="" if raw_data.get("hf_model_id") is None else str(raw_data["hf_model_id"]).strip(),
            knowledge_cutoff_date=normalize_optional_string(
                raw_data.get("knowledge_cutoff_date")
            ),
            native_context_length=parse_context_length(
                raw_data["native_context_length"]
            ),
            input_price=input_price,
            output_price=output_price,
            provider=normalize_optional_string(raw_data.get("provider")),
            base_url=normalize_optional_string(raw_data.get("base_url")),
        )


class LlmManager:
    def __init__(self, resource_path: str | Path | None = None) -> None:
        self.resource_path = (
            Path(resource_path)
            if resource_path is not None
            else resolve_default_llm_models_path()
        )
        self.models = self.load_models()

    def load_models(self) -> dict[str, LlmInfo]:
        if OmegaConf is not None:
            raw_config = OmegaConf.to_container(
                OmegaConf.load(self.resource_path), resolve=True
            )
        else:
            with self.resource_path.open(encoding="utf-8") as fh:
                raw_config = yaml.safe_load(fh)
        if not isinstance(raw_config, dict):
            raise ValueError(
                f"Expected mapping data in {self.resource_path}, got {type(raw_config)}"
            )

        raw_models = raw_config.get("models", raw_config)
        if not isinstance(raw_models, dict):
            raise ValueError(
                f"Expected 'models' mapping in {self.resource_path}, got {type(raw_models)}"
            )

        models: dict[str, LlmInfo] = {}
        for slug, raw_model in raw_models.items():
            if (
                raw_model.get("io_pricing") is None
                and "input_price" not in raw_model
                and "output_price" not in raw_model
            ):
                continue
            models[slug] = LlmInfo.from_dict(slug, raw_model)
        return models

    def get(self, slug: str) -> LlmInfo:
        if slug in LEGACY_LLM_INFO_BY_SLUG:
            return LEGACY_LLM_INFO_BY_SLUG[slug]
        return self.models[slug]


DEFAULT_LLM_MODELS_PATH = resolve_default_llm_models_path()
LEGACY_LLM_INFO_BY_SLUG = {
    "gpt-5": LlmInfo(
        slug="gpt-5",
        model="GPT-5",
        api_model_name="openai/gpt-5-chat-latest",
        hf_model_id="",
        knowledge_cutoff_date="September 30, 2024",
        native_context_length=400000,
        input_price=1.25,
        output_price=10.0,
        provider="OpenAI",
        base_url=None,
    ),
    "gpt-5.3": LlmInfo(
        slug="gpt-5.3",
        model="GPT-5.3",
        api_model_name="openai/gpt-5.3-chat-latest",
        hf_model_id="",
        knowledge_cutoff_date="August 31, 2025",
        native_context_length=128000,
        input_price=1.75,
        output_price=14.0,
        provider="OpenAI",
        base_url=None,
    ),
}
DEFAULT_LLM_MANAGER = LlmManager(DEFAULT_LLM_MODELS_PATH)
MODEL_INFO_BY_SLUG = DEFAULT_LLM_MANAGER.models
MODEL_SLUG_TO_MODEL_ID = {
    slug: llm_info.model for slug, llm_info in MODEL_INFO_BY_SLUG.items()
}
MODEL_SLUG_TO_API_MODEL_NAME = {
    slug: llm_info.api_model_name for slug, llm_info in MODEL_INFO_BY_SLUG.items()
}
MODEL_SLUG_TO_MODEL_ID.update(
    {slug: llm_info.model for slug, llm_info in LEGACY_LLM_INFO_BY_SLUG.items()}
)
MODEL_SLUG_TO_API_MODEL_NAME.update(
    {
        slug: llm_info.api_model_name
        for slug, llm_info in LEGACY_LLM_INFO_BY_SLUG.items()
    }
)
MODEL_NAME_TO_BASE_URL = {
    model_name: llm_info.base_url
    for slug, llm_info in MODEL_INFO_BY_SLUG.items()
    for model_name in (slug, llm_info.api_model_name)
    if model_name and llm_info.base_url is not None
}
MODEL_ID_TO_MODEL_SLUG = {
    llm_info.model: slug for slug, llm_info in MODEL_INFO_BY_SLUG.items()
}
MODEL_ID_TO_MODEL_SLUG.update(
    {llm_info.model: slug for slug, llm_info in LEGACY_LLM_INFO_BY_SLUG.items()}
)
MODEL_MAX_LENGTHS = {
    slug: llm_info.native_context_length
    for slug, llm_info in MODEL_INFO_BY_SLUG.items()
}
MODEL_MAX_LENGTHS.update(
    {
        slug: llm_info.native_context_length
        for slug, llm_info in LEGACY_LLM_INFO_BY_SLUG.items()
    }
)


__all__ = [
    "DEFAULT_LLM_MANAGER",
    "DEFAULT_LLM_MODELS_PATH",
    "LlmInfo",
    "LlmManager",
    "MODEL_ID_TO_MODEL_SLUG",
    "MODEL_SLUG_TO_API_MODEL_NAME",
    "MODEL_INFO_BY_SLUG",
    "MODEL_MAX_LENGTHS",
    "MODEL_SLUG_TO_MODEL_ID",
]
