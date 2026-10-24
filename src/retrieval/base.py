"""Common types for Wikipedia article retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedItem:
    """A retrieval result, ranked by `score` (higher = more relevant).

    `path` is populated for file-backed retrievers and may be empty for
    retrievers that read directly from a prebuilt external index.
    """

    id: int | str
    path: str
    content: str
    score: float
    title: str | None = None


class Retriever(ABC):
    """Common interface for article retrievers.

    Concrete subclasses may either build local indexes or wrap prebuilt ones.
    Each retriever exposes `retrieve(query, k)` and returns the top-k results
    for a free-form text query, plus `get_document_by_id(docid)` for direct
    passage lookup by external collection id.
    """

    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> list[RetrievedItem]:
        """Return up to k RetrievedItems ranked by descending relevance score."""
        raise NotImplementedError

    @abstractmethod
    def get_document_by_id(self, docid: int | str) -> RetrievedItem | None:
        """Return the document for an external collection id, if present."""
        raise NotImplementedError
