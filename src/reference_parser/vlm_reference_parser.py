"""
VLM-based reference parser that uses an LLM to extract structured reference data
from raw text.

The parser accepts either the full document text or a pre-extracted references
section, automatically isolating the references block before sending it to the
model.

To run the parser for testing:
```bash
python -m src.reference_parser.vlm_reference_parser
```
"""

from __future__ import annotations

from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from src.llm import call_litellm
from src.reference_parser.prompts import (
    build_batch_reference_system_prompt,
    build_single_reference_system_prompt,
)
from src.setup.constants import (
    preprocess_reference_authors,
    preprocess_reference_title,
)
from src.utils.preprocessing.extract_reference_section import extract_references_section
from src.utils.preprocessing.preprocess_references_section import (
    preprocess_references_section,
)

from .reference_parser import Reference, ReferenceParser

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class ExtractedReference(BaseModel):
    title: str = Field(
        ...,
        description=(
            "The full title of the referenced work, exactly as it appears in the "
            "reference entry. Do not abbreviate or paraphrase. "
            "If no title can be identified, set to an empty string."
        ),
    )
    authors: list[str] = Field(
        ...,
        description=(
            "An ordered list of author name strings, preserving the sequence in "
            "which they appear in the reference. Each name should be formatted as "
            "written (e.g. 'J. Smith' or 'John Smith'). "
            "Strip any index markers or symbols attached to names "
            "(e.g. 'Smith\u00b9' \u2192 'Smith'). "
            "Do not include conjunctions such as 'and' at the start of an author "
            "name (e.g. return 'Naor, A.', not 'and Naor, A.'). "
            "If no authors are identifiable, return an empty list."
        ),
    )
    year: str = Field(
        ...,
        description=(
            "The four-digit publication year as it appears in the reference "
            "(e.g. '2023'). Include a trailing letter if present (e.g. '2023a'). "
            "If the year cannot be determined, set to an empty string."
        ),
    )
    venue: str = Field(
        ...,
        description=(
            "The publication venue: journal name, conference name, workshop title, "
            "or 'arXiv preprint'. Use the name exactly as it appears in the "
            "reference; do not expand abbreviations unless the full name is also "
            "present. If no venue can be identified, set to an empty string."
        ),
    )


class ExtractedReferenceList(BaseModel):
    references: list[ExtractedReference] = Field(
        ...,
        description=(
            "The complete list of reference entries parsed from the references "
            "section, in the order they appear in the source text. "
            "Every entry present in the section must have a corresponding element "
            "in this list, even if some fields cannot be determined."
        ),
    )

    def to_dict(self) -> dict:
        return self.model_dump()

    def to_json(self, **kwargs) -> str:
        return self.model_dump_json(**kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractedReferenceList":
        return cls.model_validate(data)


SYSTEM_PROMPT_BATCH = build_batch_reference_system_prompt(
    ExtractedReferenceList.model_json_schema()
)
SYSTEM_PROMPT_SINGLE = build_single_reference_system_prompt(
    ExtractedReference.model_json_schema()
)

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class VLMReferenceParser(ReferenceParser):
    """LLM-powered reference parser.

    Accepts either the full document text or a pre-extracted references block.
    The references section is isolated with :func:`extract_references_section`
    and cleaned with :func:`preprocess_references_section` before being sent to
    the model.

    :meth:`parse` issues a **single** LLM call for the entire references
    section, which is far more efficient than calling :meth:`parse_one` per
    entry. :meth:`parse_one` is also available for parsing individual reference
    strings on demand.

    Args:
        model: LiteLLM model identifier (e.g. ``"openai/gpt-4o"``).
        api_key: Optional API key override; falls back to the environment.
        num_trials: Number of retry attempts on failure before raising.
        service_tier: OpenAI service tier (``"default"``, ``"flex"``,
            ``"priority"``). Ignored for non-OpenAI models.
        reasoning_effort: Model reasoning effort hint. Set to ``"default"``
            to omit the parameter entirely from the request.
    """

    _OPENAI_MODEL_KEYWORDS = ("openai", "gpt")

    def __init__(
        self,
        model: str = "openai/gpt-5.4-mini",
        api_key: str | None = None,
        num_trials: int = 3,
        service_tier: Literal["default", "flex", "priority"] = "default",
        reasoning_effort: Literal["none", "low", "medium", "high", "default"] = "low",
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.num_trials = num_trials
        self.service_tier = service_tier
        self.reasoning_effort = reasoning_effort

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, text: str) -> list[Reference]:
        """Parse a full document or raw references section into Reference objects.

        Pipeline:
        1. Extract the references block via :func:`extract_references_section`.
        2. Clean the block via :func:`preprocess_references_section`.
        3. Send the cleaned text to the LLM in **one** request.
        4. Deserialise the JSON response into :class:`Reference` objects.

        Args:
            text: Full document text or a raw references-section string.

        Returns:
            A list of :class:`Reference` objects, one per parsed entry.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        references_text = extract_references_section(text)
        if not references_text:
            # If extraction found nothing, treat the whole input as the section.
            references_text = text

        references_text = preprocess_references_section(references_text)

        last_exc: Exception | None = None
        for trial in range(self.num_trials):
            try:
                raw_content = self._call_llm(
                    system_prompt=SYSTEM_PROMPT_BATCH,
                    user_message=(
                        "Please parse all reference entries from the following references section:\n\n"
                        + references_text
                    ),
                    response_format=self._build_response_format_batch(),
                )
                parsed = ExtractedReferenceList.model_validate_json(raw_content)
                return [self._extracted_to_reference(ref) for ref in parsed.references]
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[VLMReferenceParser] parse() trial {}/{} failed: {}",
                    trial + 1,
                    self.num_trials,
                    exc,
                )

        raise RuntimeError(
            f"All {self.num_trials} trial(s) failed during parse()."
        ) from last_exc

    def parse_one(self, text: str) -> Reference | None:
        """Parse a single reference string into a :class:`Reference` object.

        Satisfies the :class:`ReferenceParser` ABC contract and is useful when
        individual reference strings are already split out. For bulk parsing,
        prefer :meth:`parse` to minimise LLM round-trips.

        Args:
            text: A single raw reference string.

        Returns:
            A :class:`Reference` on success, or ``None`` after all retries fail.
        """
        last_exc: Exception | None = None
        for trial in range(self.num_trials):
            try:
                raw_content = self._call_llm(
                    system_prompt=SYSTEM_PROMPT_SINGLE,
                    user_message=(
                        "Please parse the following reference entry:\n\n" + text
                    ),
                    response_format=self._build_response_format_single(),
                )
                parsed = ExtractedReference.model_validate_json(raw_content)
                return self._extracted_to_reference(parsed)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[VLMReferenceParser] parse_one() trial {}/{} failed: {}",
                    trial + 1,
                    self.num_trials,
                    exc,
                )

        logger.warning(
            "[VLMReferenceParser] parse_one() giving up after {} trial(s): {}",
            self.num_trials,
            last_exc,
        )
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        response_format: dict,
    ) -> str:
        """Issue a text-only completion request and return the raw content string.

        Args:
            system_prompt: The system-role instruction message.
            user_message: The user-role message carrying the text to parse.
            response_format: The ``response_format`` dict for structured output.

        Returns:
            The raw string from ``response.choices[0].message.content``.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        extra_kwargs: dict = {}
        if self.api_key:
            extra_kwargs["api_key"] = self.api_key
        if self._is_openai_model():
            extra_kwargs["service_tier"] = self.service_tier
        if self.reasoning_effort != "default":
            extra_kwargs["reasoning_effort"] = self.reasoning_effort

        # If we use service_tier, and it value is flex, then we try catch the error and retry the call if it fails and remove that service_tier from the kwargs
        if self.service_tier == "flex":
            try:
                response = call_litellm(
                    model=self.model,
                    messages=messages,
                    response_format=response_format,
                    **extra_kwargs,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.info("Flex service tier failed: {}", e)
                extra_kwargs.pop("service_tier")

        response = call_litellm(
            model=self.model,
            messages=messages,
            response_format=response_format,
            **extra_kwargs,
        )
        return response.choices[0].message.content

    def _is_openai_model(self) -> bool:
        """Return True if the model identifier refers to an OpenAI/GPT model."""
        model_lower = self.model.lower()
        return any(kw in model_lower for kw in self._OPENAI_MODEL_KEYWORDS)

    @staticmethod
    def _build_response_format_batch() -> dict:
        """Build a fully-inlined, OpenAI-compatible JSON schema for batch parsing.

        Avoids Pydantic's ``$defs`` / ``additionalProperties: true`` which some
        API providers reject. The schema wraps a list of reference objects under
        the ``"references"`` key.
        """
        reference_item_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "authors": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "year": {"type": "string"},
                "venue": {"type": "string"},
            },
            "required": ["title", "authors", "year", "venue"],
            "additionalProperties": False,
        }
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "ExtractedReferenceList",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "references": {
                            "type": "array",
                            "items": reference_item_schema,
                        }
                    },
                    "required": ["references"],
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def _build_response_format_single() -> dict:
        """Build a fully-inlined, OpenAI-compatible JSON schema for single-reference parsing.

        Avoids Pydantic's ``$defs`` / ``additionalProperties: true`` which some
        API providers reject.
        """
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "ExtractedReference",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "authors": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "year": {"type": "string"},
                        "venue": {"type": "string"},
                    },
                    "required": ["title", "authors", "year", "venue"],
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def _extracted_to_reference(info: ExtractedReference) -> Reference:
        """Convert an :class:`ExtractedReference` Pydantic model to a :class:`Reference`."""
        return Reference(
            title=preprocess_reference_title(info.title),
            authors=preprocess_reference_authors(info.authors),
            year=info.year,
            venue=info.venue,
            raw="",
        )


# ---------------------------------------------------------------------------
# Smoke-test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from pathlib import Path

    from src.setup import bootstrap_runtime

    bootstrap_runtime()
    parser = VLMReferenceParser(
        model="openai/gpt-5.4-mini",
        service_tier="default",
    )

    sample_dirs = [
        "output/parsed_pdfs/docling/icml",
        "output/parsed_pdfs/docling/neurips",
        "output/parsed_pdfs/docling/aaai",
    ]

    os.makedirs("test_output", exist_ok=True)
    f_out = open("test_output/vlm_reference_parser.txt", "w")
    for sample_dir in sample_dirs:
        for md_path in sorted(Path(sample_dir).glob("*.md")):
            logger.info("Parsing references from {} ...", md_path)
            document_text = md_path.read_text(encoding="utf-8")
            results = parser.parse(document_text)
            logger.info("  -> {} reference(s) extracted", len(results))
            for ref in results[:3]:
                logger.info("{}", ref)
            logger.info("-" * 80)

            f_out.write(f"{md_path}\n")
            f_out.write(f"  -> {len(results)} reference(s) extracted\n")
            for ref in results:
                f_out.write(f"{ref}\n")
            f_out.write("-" * 80 + "\n")

    f_out.close()
