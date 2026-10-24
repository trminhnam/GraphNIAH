from __future__ import annotations

import html
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.setup.constants import preprocess_reference_authors, preprocess_reference_title


@dataclass
class Reference:
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: str = ""
    venue: str = ""
    raw: str = ""

    def __setattr__(self, name: str, value: object) -> None:
        if name == "title" and isinstance(value, str):
            value = preprocess_reference_title(value)
        elif name == "authors" and isinstance(value, (list, tuple)):
            value = preprocess_reference_authors(
                [author for author in value if isinstance(author, str)]
            )
        super().__setattr__(name, value)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "raw": self.raw,
        }

    def __str__(self) -> str:
        return f"""Reference(
    title={self.title},
    authors={self.authors},
    year={self.year},
    venue={self.venue},
    raw={self.raw}
)
"""


_BLANK_LINE_RE = re.compile(r"\n\s*\n")
_LIST_MARKER_RE = re.compile(r"^[\-\*\u2022]\s+", re.MULTILINE)
_LIST_ITEM_SPLIT_RE = re.compile(r"\n(?=[\-\*\u2022]\s+)")
_NUMBERED_REF_SPLIT_RE = re.compile(
    r"(?:^|\n)\s*(?=(?:\[\s*\d+\s*\]|\d+\.\s+))",
    re.MULTILINE,
)
_HEADING_RE = re.compile(r"^#+\s+.*$", re.MULTILINE)
_LIKELY_COMPLETE_REFERENCE_LINE_RE = re.compile(
    r"\b(?:19|20)\d{2}[a-z]?\.\s+.+(?:[.?!]|[.?!]\)|\.)\s*$"
)


class ReferenceParser(ABC):
    """Base class for bibliography reference parsers."""

    @abstractmethod
    def parse_one(self, text: str) -> Reference | None:
        """Parse a single reference string into a Reference object."""
        ...

    @staticmethod
    def _prepare_chunks(text: str) -> list[str]:
        """Split a raw references-section string into normalised single-reference chunks.

        Handles blank-line splitting, numbered-list marker splitting, bullet-list
        splitting, markdown heading removal, HTML-entity unescaping, and whitespace
        normalisation.  The returned strings are ready to pass to ``parse_one``.

        Extracted as a static method so that :class:`UnifiedReferenceParser` can
        reuse the same logic during its grammar-detection probe phase without
        duplicating code.
        """
        text = _HEADING_RE.sub("", text).strip()

        chunks = _BLANK_LINE_RE.split(text)
        # If no blank-line structure exists, many extracted reference blocks are
        # still separable by numbered list markers such as "[12]" or "12.".
        if len(chunks) <= 1:
            numbered_chunks = [
                c.strip() for c in _NUMBERED_REF_SPLIT_RE.split(text) if c.strip()
            ]
            if len(numbered_chunks) > 1:
                chunks = numbered_chunks
            else:
                line_chunks = [
                    line.strip() for line in text.splitlines() if line.strip()
                ]
                if len(line_chunks) > 1 and all(
                    _LIKELY_COMPLETE_REFERENCE_LINE_RE.search(line)
                    for line in line_chunks
                ):
                    chunks = line_chunks

        result: list[str] = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            sub_chunks = _LIST_ITEM_SPLIT_RE.split(chunk)
            for sub in sub_chunks:
                sub = _LIST_MARKER_RE.sub("", sub).strip()
                sub = html.unescape(sub)
                sub = " ".join(sub.split())
                if sub:
                    result.append(sub)
        return result

    def parse(self, text: str) -> list[Reference]:
        """Parse a full references section into individual Reference objects.

        Handles splitting by blank lines, splitting bullet-listed items,
        stripping markdown artefacts, and delegating each chunk to
        ``parse_one``.
        """
        results: list[Reference] = []
        for sub in self._prepare_chunks(text):
            ref = self.parse_one(sub)
            if ref is not None:
                results.append(ref)
        return results
