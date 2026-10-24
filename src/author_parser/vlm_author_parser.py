"""
This module contains the VLMAuthorParser class, which is used to parse the author information from a PDF file.

To run the parser for testing, run the following command:
```bash
python -m src.author_parser.vlm_author_parser
```
"""

import base64
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import fitz  # PyMuPDF
from loguru import logger
from pydantic import BaseModel, Field

from src.author_parser.prompts import build_author_system_prompt
from src.llm import call_litellm


class ExtractedAuthorInfo(BaseModel):
    name: str = Field(
        ...,
        description=(
            "The full name of the author as it appears on the paper, including first name, "
            "middle name or initial (if present), and last name. "
            "Do not include titles, honorifics, or superscript markers (e.g. write 'John A. Smith', not 'John A. Smith¹')."
        ),
    )
    organization: str = Field(
        ...,
        description=(
            "The full organizational affiliation of the author. "
            "If the affiliation spans multiple lines or hierarchical levels (e.g. department, university, country), "
            "join each line with a comma and a space. "
            "Example: 'Department of Computer Science, Example University'. "
            "If an author has multiple affiliations, join them with ' | ' as a separator. "
            "Example: 'MIT CSAIL, Cambridge, MA, USA | Google DeepMind, London, UK'. "
            "Do not abbreviate institution names unless the paper itself uses only the abbreviation."
        ),
    )
    email: str = Field(
        ...,
        description=(
            "The email address of the author as it appears in the paper. "
            "If no email is explicitly listed for this author, set to an empty string. "
            "Do not infer or construct email addresses; only extract ones that are explicitly shown. "
            "Example: 'author@example.org'."
        ),
    )
    orcid: str = Field(
        default="",
        description=(
            "The ORCID identifier of the author, formatted as a 16-digit string separated by hyphens. "
            "Example: '0000-0002-1825-0097'. "
            "If not present in the paper, set to an empty string. "
            "Do not infer or guess ORCIDs."
        ),
    )
    position: int = Field(
        ...,
        description=(
            "The 1-based position of the author in the paper's author list, reflecting the order "
            "in which they appear. The first author is position 1, the second is position 2, and so on. "
            "Corresponding or equal-contribution authors should still be assigned their listed order position."
        ),
    )


class ExtractedAuthorInfoList(BaseModel):
    authors: list[ExtractedAuthorInfo] = Field(
        ...,
        description=(
            "The complete list of authors extracted from the first page of the paper, "
            "ordered by their position in the author list (position 1 first). "
            "Every author listed on the page should have an entry, even if some fields "
            "such as email or ORCID are not available."
        ),
    )

    def to_dict(self) -> dict:
        return self.model_dump()

    def to_json(self, **kwargs) -> str:
        return self.model_dump_json(**kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractedAuthorInfoList":
        return cls.model_validate(data)


class VLMAuthorParser:
    SYSTEM_PROMPT = build_author_system_prompt(
        ExtractedAuthorInfoList.model_json_schema()
    )

    _OPENAI_MODEL_KEYWORDS = ("openai", "gpt")

    def __init__(
        self,
        model: str = "openai/gpt-5.4-mini",
        api_key: str | None = None,
        media_type: Literal["image", "pdf"] = "pdf",
        num_trials: int = 3,
        service_tier: Literal["default", "flex", "priority"] = "default",
        reasoning_effort: Literal["none", "low", "medium", "high", "default"] = "low",
    ):
        self.model = model
        self.api_key = api_key
        self.media_type = media_type
        self.num_trials = num_trials
        self.service_tier = service_tier
        self.reasoning_effort = reasoning_effort

    def _is_openai_model(self) -> bool:
        """Returns True if the model name contains 'openai' or 'gpt' (case-insensitive)."""
        model_lower = self.model.lower()
        return any(keyword in model_lower for keyword in self._OPENAI_MODEL_KEYWORDS)

    def parse(self, pdf_path: str) -> ExtractedAuthorInfoList:
        last_exc: Exception | None = None

        for trial in range(self.num_trials):
            page_kwargs: dict[str, Any] = {}

            try:
                page_kwargs = self.get_trial_page_kwargs(pdf_path, trial)
                messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                ]

                if self.media_type == "image":
                    images_b64 = self._get_pdf_page_as_b64_image(
                        pdf_path, **page_kwargs
                    )
                    content = [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        }
                        for image_b64 in images_b64
                    ]
                    content.append(
                        {
                            "type": "text",
                            "text": "Please extract the list of authors with their information from the given images.",
                        }
                    )
                    messages.append({"role": "user", "content": content})

                elif self.media_type == "pdf":
                    pdf_b64 = self._get_pdf_page_as_b64_pdf(pdf_path, **page_kwargs)
                    content = [
                        {
                            "type": "file",
                            "file": {
                                "file_data": f"data:application/pdf;base64,{pdf_b64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "Please extract the list of authors with their information from the given PDF.",
                        },
                    ]
                    messages.append({"role": "user", "content": content})

                else:
                    raise ValueError(f"Invalid input type: {self.media_type}")

                # Build optional kwargs conditionally
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
                            response_format=self._build_response_format(),
                            **extra_kwargs,
                        )
                        return ExtractedAuthorInfoList.model_validate_json(
                            response.choices[0].message.content
                        )
                    except Exception as e:
                        logger.info("Flex service tier failed: {}", e)
                        extra_kwargs.pop("service_tier")

                response = call_litellm(
                    model=self.model,
                    messages=messages,
                    response_format=self._build_response_format(),
                    **extra_kwargs,
                )
                return ExtractedAuthorInfoList.model_validate_json(
                    response.choices[0].message.content
                )

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[VLMAuthorParser] Trial {}/{} failed (page selection={}): {}",
                    trial + 1,
                    self.num_trials,
                    page_kwargs or "unresolved",
                    exc,
                )

        raise RuntimeError(
            f"All {self.num_trials} trial(s) failed for '{pdf_path}'."
        ) from last_exc

    def get_trial_page_kwargs(self, pdf_path: str, trial: int) -> dict[str, Any]:
        with fitz.open(pdf_path) as doc:
            return self.resolve_trial_page_kwargs(len(doc), trial)

    def _get_pdf_page_as_b64_image(
        self,
        pdf_path: str,
        page_index: int | None = None,
        page_range: tuple[int, int] | None = None,
    ) -> list[str]:
        """
        Renders one or more pages of a PDF and returns base64-encoded PNG strings.

        Args:
            pdf_path:    Path to the PDF file.
            page_index:  Index of the page to render (0-based).
            page_range:  Inclusive (start, end) page indices to render.
                         If both are None, all pages are rendered.

        Returns:
            List of base64-encoded PNG strings.
        """
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            if total_pages == 0:
                return []

            page_indices = self._resolve_page_indices(
                total_pages, page_index, page_range
            )

            images_b64: list[str] = []
            for idx in page_indices:
                page = doc[idx]
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png")
                images_b64.append(base64.b64encode(img_bytes).decode("utf-8"))

            return images_b64

    def _get_pdf_page_as_b64_pdf(
        self,
        pdf_path: str,
        page_index: int | None = None,
        page_range: tuple[int, int] | None = None,
    ) -> str:
        """
        Exports selected pages to a temporary PDF and returns its base64 string.

        Args:
            pdf_path:    Path to the source PDF file.
            page_index:  Single page index to keep (0-based).
            page_range:  Inclusive (start, end) page indices to keep.
                         If both are None, all pages are kept.

        Returns:
            Base64-encoded string of the truncated PDF.
        """
        source_path = Path(pdf_path)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            copied_pdf_path = temp_dir_path / source_path.name
            shutil.copy2(source_path, copied_pdf_path)

            with fitz.open(copied_pdf_path) as copied_doc:
                total_pages = len(copied_doc)
                if total_pages == 0:
                    return ""

                selected_pages = set(
                    self._resolve_page_indices(total_pages, page_index, page_range)
                )

                for idx in range(total_pages - 1, -1, -1):
                    if idx not in selected_pages:
                        copied_doc.delete_page(idx)

                truncated_pdf_path = temp_dir_path / "truncated.pdf"
                copied_doc.save(truncated_pdf_path)

            pdf_bytes = truncated_pdf_path.read_bytes()
            return base64.b64encode(pdf_bytes).decode("utf-8")

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _build_response_format() -> dict:
        """
        Builds a fully-inlined, OpenAI-compatible JSON schema for structured outputs.
        Avoids Pydantic's $defs / additionalProperties:true which the API rejects.
        """
        author_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "organization": {"type": "string"},
                "email": {"type": "string"},
                "orcid": {"type": "string"},
                "position": {"type": "integer"},
            },
            "required": ["name", "organization", "email", "orcid", "position"],
            "additionalProperties": False,
        }
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "ExtractedAuthorInfoList",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "authors": {
                            "type": "array",
                            "items": author_schema,
                        }
                    },
                    "required": ["authors"],
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def resolve_trial_page_kwargs(total_pages: int, trial: int) -> dict[str, Any]:
        if trial == 0 or total_pages <= 1:
            return {"page_index": 0}
        return {"page_range": (0, min(trial, total_pages - 1))}

    @staticmethod
    def _resolve_page_indices(
        total_pages: int,
        page_index: int | None,
        page_range: tuple[int, int] | None,
    ) -> range:
        """
        Resolves the desired page selection into a range of indices.
        Centralises the validation logic shared by both helper methods.
        """
        if page_range is not None:
            if len(page_range) != 2:
                raise ValueError("page_range must be a tuple of (start, end).")
            start, end = page_range
            if not (0 <= start < total_pages and 0 <= end < total_pages):
                raise ValueError(
                    f"page_range {page_range} out of range for document with "
                    f"{total_pages} page(s)."
                )
            if start > end:
                raise ValueError(
                    f"Invalid page_range {page_range}: start must be <= end."
                )
            return range(start, end + 1)

        if page_index is not None:
            if not (0 <= page_index < total_pages):
                raise ValueError(
                    f"page_index {page_index} out of range for document with "
                    f"{total_pages} page(s)."
                )
            return range(page_index, page_index + 1)

        return range(total_pages)


if __name__ == "__main__":
    from src.setup import bootstrap_runtime

    bootstrap_runtime()
    parser = VLMAuthorParser(
        model="openai/gpt-5.4-mini", media_type="pdf", service_tier="flex"
    )
    pdf_dir = "resources/samples/aaai"
    for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
        logger.info("Parsing {} ...", pdf_path)
        result = parser.parse(str(pdf_path))
        logger.info("{}", result)
        logger.info("-" * 100)
