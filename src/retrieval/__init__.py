"""Retrievers for Wikipedia passages and locally indexed paper corpora.

- `BM25Retriever`  — sparse, lexical, backed by Pyserini's prebuilt Lucene
  Wikipedia index.
- `DenseRetriever` — dense, semantic, backed by Pyserini's prebuilt DPR
  Wikipedia FAISS index.
- `PaperBM25Retriever` — sparse, lexical, backed by a local paper Lucene index.
- `PaperDenseRetriever` — dense, semantic, backed by a local paper FAISS index.

Retriever surface:
  retriever.retrieve(query, k) -> list[RetrievedItem]
  retriever.get_document_by_id(docid) -> RetrievedItem | None

Optional index lifecycle:
  bm25.build_index(index_dir)   # prepare/download into a cache root
  bm25.load_index(index_name)
  dense.build_index(index_dir)  # prepare/download into a cache root
  dense.load_index(index_name)
"""

from .base import RetrievedItem, Retriever
from .corpus import Paragraph, ParagraphCorpus, ParagraphMetadata
from .bm25_retriever import BM25Retriever, PrebuiltWikipediaBM25Retriever
from .dense_retriever import DenseRetriever, PrebuiltWikipediaDPRRetriever
from .paper_retriever import PaperBM25Retriever, PaperDenseRetriever

__all__ = [
    "RetrievedItem",
    "Retriever",
    "Paragraph",
    "ParagraphMetadata",
    "ParagraphCorpus",
    "BM25Retriever",
    "PrebuiltWikipediaBM25Retriever",
    "DenseRetriever",
    "PrebuiltWikipediaDPRRetriever",
    "PaperBM25Retriever",
    "PaperDenseRetriever",
]
