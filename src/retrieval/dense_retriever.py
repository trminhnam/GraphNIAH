"""Dense retriever backed by a prebuilt Pyserini DPR Wikipedia index."""

from __future__ import annotations

import orjson
from pathlib import Path
from typing import Any

from .base import RetrievedItem, Retriever
from .prebuilt_index import (
    pyserini_cache_environment,
    pyserini_cache_paths,
    pyserini_runtime_environment,
    resolve_cache_dir,
)


DEFAULT_PREBUILT_INDEX_NAME = "wikipedia-dpr-100w.dpr-multi"
DEFAULT_ENCODER_NAME = "facebook/dpr-question_encoder-multiset-base"
DEFAULT_FAISS_DEVICE = "cpu"


def import_pyserini_faiss():
    try:
        from pyserini.search.faiss import FaissSearcher
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyserini is required for DenseRetriever. "
            "Install the project dependencies before using dense retrieval."
        ) from exc

    try:
        from pyserini.search.faiss import DprQueryEncoder
    except ImportError:
        from pyserini.encode import DprQueryEncoder

    return FaissSearcher, DprQueryEncoder


def import_pyserini_prebuilt_download():
    try:
        from pyserini.util import download_prebuilt_index
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyserini is required for DenseRetriever. "
            "Install the project dependencies before using dense retrieval."
        ) from exc
    return download_prebuilt_index


def import_pyserini_dense_document_lookup():
    try:
        from pyserini.search.lucene import LuceneSearcher
        from pyserini.util import get_sparse_index
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyserini is required for DenseRetriever. "
            "Install the project dependencies before using dense retrieval."
        ) from exc
    return LuceneSearcher, get_sparse_index


class PrebuiltWikipediaDPRRetriever(Retriever):
    """Dense Wikipedia retriever that uses only a prebuilt Pyserini index."""

    def __init__(
        self,
        index_name: str = DEFAULT_PREBUILT_INDEX_NAME,
        encoder_name: str = DEFAULT_ENCODER_NAME,
        faiss_device: str = DEFAULT_FAISS_DEVICE,
        cache_dir: str | Path | None = None,
    ):
        self.index_name = index_name
        self.encoder_name = encoder_name
        self.faiss_device = faiss_device
        self.cache_dir = resolve_cache_dir(cache_dir)

        self.encoder = None
        self.document_searcher = None
        self.searcher = None

    def build_index(
        self,
        tsv_path: str | Path | None = None,
        index_dir: str | Path | None = None,
        batch_size: int | None = None,
        progress_every: int = 50_000,
        rebuild: bool = False,
    ) -> None:
        del tsv_path, batch_size, progress_every, rebuild
        if index_dir is not None:
            self.cache_dir = resolve_cache_dir(index_dir)
        self.load_index()

    def load_index(self, index_name: str | Path | None = None) -> None:
        if index_name is not None:
            resolved = Path(index_name)
            if resolved.exists():
                raise ValueError(
                    "DenseRetriever now uses Pyserini prebuilt index names, "
                    "not local index directories."
                )
            self.index_name = str(index_name)

        with pyserini_runtime_environment(self.cache_dir):
            FaissSearcher, DprQueryEncoder = import_pyserini_faiss()
            self.encoder = DprQueryEncoder(self.encoder_name, device=self.faiss_device)
            try:
                self.searcher = FaissSearcher.from_prebuilt_index(
                    self.index_name,
                    self.encoder,
                )
            except MemoryError as exc:
                raise MemoryError(
                    "Loading the prebuilt dense FAISS index exceeded available "
                    "memory. Downloading the index succeeded, but creating a live "
                    "dense searcher requires substantially more RAM. Run "
                    "`build_wiki_index.py` without `--query` to download only, or "
                    "use a machine with more memory for dense retrieval."
                ) from exc

    def download_prebuilt_index(self) -> Path:
        download_prebuilt_index = import_pyserini_prebuilt_download()
        with pyserini_cache_environment(self.cache_dir):
            index_path = download_prebuilt_index(self.index_name)
        return Path(index_path)

    def ensure_searcher_loaded(self) -> None:
        if self.searcher is None:
            self.load_index()

    def ensure_document_searcher_loaded(self) -> None:
        if self.document_searcher is not None:
            return

        document_cache_dir = self.resolve_document_cache_dir()
        with pyserini_runtime_environment(document_cache_dir):
            LuceneSearcher, get_sparse_index = import_pyserini_dense_document_lookup()
            sparse_index_name = get_sparse_index(self.index_name)
            self.document_searcher = LuceneSearcher.from_prebuilt_index(
                sparse_index_name
            )
            if self.document_searcher is None:
                raise RuntimeError(
                    "Unable to load the backing sparse index for dense document "
                    "lookup. Prepare the BM25 index first or ensure the sparse "
                    "index can be downloaded."
                )

    def resolve_document_cache_dir(self) -> Path | None:
        if self.cache_dir is None:
            return None

        sibling_bm25_cache_dir = self.cache_dir.parent / "bm25"
        if sibling_bm25_cache_dir != self.cache_dir and any(
            path.exists() for path in pyserini_cache_paths(sibling_bm25_cache_dir)
        ):
            return sibling_bm25_cache_dir

        return self.cache_dir

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
        self.ensure_document_searcher_loaded()
        normalized_docid = self.coerce_docid(docid)
        document = self.document_searcher.doc(normalized_docid)
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


DenseRetriever = PrebuiltWikipediaDPRRetriever


def main() -> None:
    retriever = DenseRetriever()
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
