"""Local paper retrievers backed by indexes built by build_paper_index.py."""

from __future__ import annotations

import orjson
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .base import RetrievedItem, Retriever
from .bm25_retriever import import_pyserini_lucene
from .prebuilt_index import pyserini_runtime_environment


DEFAULT_PAPER_CORPUS_FILENAME = "corpus.jsonl"
DEFAULT_PAPER_PARAGRAPH_CORPUS_FILENAME = "paragraph_corpus.jsonl"
DEFAULT_PAPER_DENSE_CORPUS_FILENAME = DEFAULT_PAPER_PARAGRAPH_CORPUS_FILENAME
DEFAULT_PAPER_DENSE_ENCODER_NAME = "BAAI/bge-base-en-v1.5"
DEFAULT_PAPER_DENSE_FAISS_DEVICE = "cpu"
DEFAULT_PAPER_DENSE_QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)
DEFAULT_PAPER_DENSE_BATCH_SIZE = 32
PAPER_PARAGRAPH_CHUNK_ID_SEPARATOR = "#chunk:"
PAPER_DENSE_CHUNK_ID_SEPARATOR = PAPER_PARAGRAPH_CHUNK_ID_SEPARATOR
INDEX_FILENAME = "index"
DOCID_FILENAME = "docid"


def import_faiss():
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "faiss is required for PaperDenseRetriever. "
            "Install the project dependencies before using dense paper retrieval."
        ) from exc
    return faiss


def import_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "sentence-transformers is required for PaperDenseRetriever. "
            "Install the project dependencies before using dense paper retrieval."
        ) from exc
    return SentenceTransformer


def encode_paper_dense_texts(
    encoder: Any,
    texts: list[str],
    prefix: str | None = None,
    batch_size: int = DEFAULT_PAPER_DENSE_BATCH_SIZE,
    max_length: int | None = None,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """Encode one batch of paper texts for dense retrieval."""
    normalized_texts = texts
    if prefix:
        normalized_texts = [f"{prefix}{text}" for text in texts]

    encode_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "show_progress_bar": show_progress_bar,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }
    # if max_length is not None:
    #     encode_kwargs["max_length"] = max_length

    embeddings = encoder.encode(normalized_texts, **encode_kwargs)
    return np.ascontiguousarray(embeddings, dtype="float32")


def load_paper_dense_docids(docids_path: str | Path) -> list[str]:
    """Load dense-index docids in insertion order."""
    with open(docids_path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def paper_dense_parent_id(docid: int | str) -> str:
    normalized_docid = str(docid)
    parent_id, separator, _remainder = normalized_docid.partition(
        PAPER_PARAGRAPH_CHUNK_ID_SEPARATOR
    )
    if separator and parent_id:
        return parent_id
    return normalized_docid


@dataclass(frozen=True)
class PaperDocumentMetadata:
    id: str
    path: str
    title: str | None = None
    text: str = ""


class PaperDocumentStore:
    """Load lightweight id/path/title metadata from the paper corpus JSONL."""

    def __init__(self, corpus_path: str | Path):
        self.corpus_path = Path(corpus_path)
        self.metadata_cache: dict[str, PaperDocumentMetadata] | None = None

    def load_metadata(self) -> dict[str, PaperDocumentMetadata]:
        if self.metadata_cache is not None:
            return self.metadata_cache

        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Paper corpus JSONL not found: {self.corpus_path}")

        metadata: dict[str, PaperDocumentMetadata] = {}
        with open(self.corpus_path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = orjson.loads(line)
                doc_id = str(obj.get("id", "") or "").strip()
                if not doc_id:
                    continue
                title = str(obj.get("title", "") or "").strip() or None
                path = str(obj.get("path", "") or "").strip()
                text = str(obj.get("text", "") or "")
                metadata[doc_id] = PaperDocumentMetadata(
                    id=doc_id,
                    path=path,
                    title=title,
                    text=text,
                )

        self.metadata_cache = metadata
        return metadata

    def metadata_for_id(self, docid: int | str) -> PaperDocumentMetadata | None:
        return self.load_metadata().get(str(docid))

    def read_content(self, docid: int | str) -> str:
        metadata = self.metadata_for_id(docid)
        if metadata is None:
            return ""
        if metadata.text:
            return metadata.text
        if not metadata.path:
            return ""

        path = Path(metadata.path)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")


class PaperRetrieverBase(Retriever):
    """Shared helpers for local paper BM25 and dense retrievers."""

    def __init__(self, corpus_path: str | Path):
        self.corpus_path = Path(corpus_path)
        self.document_store = PaperDocumentStore(self.corpus_path)

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
        raw_doc: str | None = None,
        score: float = 0.0,
    ) -> RetrievedItem:
        normalized_docid = self.coerce_docid(docid)
        parsed_doc = self.parse_document(raw_doc) if raw_doc is not None else {}
        metadata = self.document_store.metadata_for_id(normalized_docid)

        raw_path = str(parsed_doc.get("path", "") or "").strip()
        if not raw_path and metadata is not None:
            raw_path = metadata.path

        title = parsed_doc.get("title")
        if not title and metadata is not None:
            title = metadata.title

        content = parsed_doc.get("contents", parsed_doc.get("text", ""))
        content_text = str(content) if content else ""
        if not content_text and metadata is not None and metadata.text:
            content_text = metadata.text
        if not content_text:
            content_text = self.document_store.read_content(normalized_docid)

        return RetrievedItem(
            id=normalized_docid,
            path=raw_path,
            content=content_text,
            score=float(score),
            title=str(title) if title else None,
        )


class PaperBM25Retriever(PaperRetrieverBase):
    """BM25 retriever over a local paper Lucene index."""

    def __init__(
        self,
        index_dir: str | Path,
        corpus_path: str | Path | None = None,
    ):
        self.index_dir = Path(index_dir).resolve()
        if corpus_path is None:
            corpus_path = (
                self.index_dir.parent / DEFAULT_PAPER_PARAGRAPH_CORPUS_FILENAME
            )
        super().__init__(corpus_path)
        self.searcher = None

    def load_index(self) -> None:
        with pyserini_runtime_environment(self.index_dir.parent):
            LuceneSearcher = import_pyserini_lucene()
            self.searcher = LuceneSearcher(str(self.index_dir))

    def ensure_searcher_loaded(self) -> None:
        if self.searcher is None:
            self.load_index()

    def get_document_by_id(self, docid: int | str) -> RetrievedItem | None:
        self.ensure_searcher_loaded()
        normalized_docid = self.coerce_docid(docid)
        document = self.searcher.doc(normalized_docid)
        if document is None:
            metadata = self.document_store.metadata_for_id(normalized_docid)
            if metadata is None:
                return None
            return self.build_retrieved_item(normalized_docid)

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


class PaperDenseRetriever(PaperRetrieverBase):
    """Dense retriever over a local paper FAISS index."""

    def __init__(
        self,
        index_dir: str | Path,
        corpus_path: str | Path | None = None,
        encoder_name: str = DEFAULT_PAPER_DENSE_ENCODER_NAME,
        faiss_device: str = DEFAULT_PAPER_DENSE_FAISS_DEVICE,
    ):
        self.index_dir = Path(index_dir).resolve()
        if corpus_path is None:
            corpus_path = (
                self.index_dir.parent / DEFAULT_PAPER_PARAGRAPH_CORPUS_FILENAME
            )
        super().__init__(corpus_path)
        self.encoder_name = encoder_name
        self.faiss_device = faiss_device
        self.encoder = None
        self.index = None
        self.docids: list[str] | None = None

    def load_index(self) -> None:
        index_path = self.index_dir / INDEX_FILENAME
        if not index_path.exists():
            raise FileNotFoundError(f"Paper dense index file not found: {index_path}")

        docids_path = self.index_dir / DOCID_FILENAME
        if not docids_path.exists():
            raise FileNotFoundError(f"Paper dense docid file not found: {docids_path}")

        SentenceTransformer = import_sentence_transformer()
        faiss = import_faiss()
        self.encoder = SentenceTransformer(
            self.encoder_name,
            device=self.faiss_device,
        )
        self.index = faiss.read_index(str(index_path))
        self.docids = load_paper_dense_docids(docids_path)

    def ensure_searcher_loaded(self) -> None:
        if self.encoder is None or self.index is None or self.docids is None:
            self.load_index()

    def get_document_by_id(self, docid: int | str) -> RetrievedItem | None:
        normalized_docid = self.coerce_docid(docid)
        metadata = self.document_store.metadata_for_id(normalized_docid)
        if metadata is None:
            return None
        return self.build_retrieved_item(normalized_docid)

    def retrieve(self, query: str, k: int = 10) -> list[RetrievedItem]:
        self.ensure_searcher_loaded()
        if k <= 0:
            return []

        query_embeddings = encode_paper_dense_texts(
            self.encoder,
            [query],
            prefix=DEFAULT_PAPER_DENSE_QUERY_PREFIX,
            batch_size=1,
        )
        scores, indices = self.index.search(query_embeddings, k)
        assert self.docids is not None

        results: list[RetrievedItem] = []
        for doc_index, score in zip(indices[0], scores[0]):
            if doc_index < 0 or doc_index >= len(self.docids):
                continue
            results.append(
                self.build_retrieved_item(self.docids[doc_index], score=float(score))
            )
        return results
