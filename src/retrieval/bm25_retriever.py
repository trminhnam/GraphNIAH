"""BM25 retriever backed by a prebuilt Pyserini Lucene Wikipedia index."""

from __future__ import annotations

import orjson
from pathlib import Path
from typing import Any

from .base import RetrievedItem, Retriever
from .prebuilt_index import pyserini_runtime_environment, resolve_cache_dir


DEFAULT_PREBUILT_INDEX_NAME = "wikipedia-dpr-100w"


def import_pyserini_lucene():
    try:
        from pyserini.search.lucene import LuceneSearcher
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyserini is required for BM25Retriever. "
            "Install the project dependencies before using BM25 retrieval."
        ) from exc
    return LuceneSearcher


class PrebuiltWikipediaBM25Retriever(Retriever):
    """BM25 Wikipedia retriever that uses only a prebuilt Pyserini index."""

    def __init__(
        self,
        index_name: str = DEFAULT_PREBUILT_INDEX_NAME,
        cache_dir: str | Path | None = None,
    ):
        self.index_name = index_name
        self.cache_dir = resolve_cache_dir(cache_dir)
        self.searcher = None

    def build_index(
        self,
        tsv_path: str | Path | None = None,
        index_dir: str | Path | None = None,
        progress_every: int = 100_000,
        rebuild: bool = False,
    ) -> None:
        del tsv_path, progress_every, rebuild
        if index_dir is not None:
            self.cache_dir = resolve_cache_dir(index_dir)
        self.load_index()

    def load_index(self, index_name: str | Path | None = None) -> None:
        if index_name is not None:
            resolved = Path(index_name)
            if resolved.exists():
                raise ValueError(
                    "BM25Retriever now uses Pyserini prebuilt index names, "
                    "not local index directories."
                )
            self.index_name = str(index_name)

        with pyserini_runtime_environment(self.cache_dir):
            LuceneSearcher = import_pyserini_lucene()
            self.searcher = LuceneSearcher.from_prebuilt_index(self.index_name)

    def ensure_searcher_loaded(self) -> None:
        if self.searcher is None:
            self.load_index()

    def parse_document(self, raw_doc: str) -> dict[str, Any]:
        try:
            parsed = orjson.loads(raw_doc)
        except orjson.JSONDecodeError:
            return {"contents": raw_doc}
        if isinstance(parsed, dict):
            return parsed
        return {"contents": raw_doc}

    def coerce_docid(self, docid: Any) -> str:
        return str(docid)

    def build_retrieved_item(
        self,
        docid: Any,
        raw_doc: str,
        score: float = 0.0,
    ) -> RetrievedItem:
        doc = self.parse_document(raw_doc)
        content = doc.get("contents", doc.get("text", raw_doc))
        title = doc.get("title")
        return RetrievedItem(
            id=self.coerce_docid(docid),
            path="",
            content=str(content),
            score=float(score),
            title=str(title) if title else None,
        )

    def get_document_by_id(self, docid: int | str) -> RetrievedItem | None:
        self.ensure_searcher_loaded()
        normalized_docid = self.coerce_docid(docid)
        document = self.searcher.doc(normalized_docid)
        if document is None:
            return None
        raw_doc = document.raw() or ""
        return self.build_retrieved_item(normalized_docid, raw_doc)

    def retrieve(self, query: str, k: int = 10) -> list[RetrievedItem]:
        self.ensure_searcher_loaded()
        hits = self.searcher.search(query, k=k)

        results: list[RetrievedItem] = []
        for hit in hits:
            document = self.searcher.doc(hit.docid)
            raw_doc = ""
            if document is not None:
                raw_doc = document.raw() or ""
            results.append(self.build_retrieved_item(hit.docid, raw_doc, hit.score))
        return results


BM25Retriever = PrebuiltWikipediaBM25Retriever


def main() -> None:
    retriever = BM25Retriever()
    query = "Who developed the theory of general relativity?"

    print(f"query: {query}")
    for rank, hit in enumerate(retriever.retrieve(query, k=5), start=1):
        print(f"\nRank: {rank}")
        print(f"DocID: {hit.id}")
        print(f"Score: {hit.score:.4f}")
        print(f"Title: {hit.title}")
        print(f"Text: {hit.content[:700]}")


if __name__ == "__main__":
    main()
