"""Download and use prebuilt Wikipedia retrieval indexes.

Default outputs:

    resources/distracting_content/wikipedia/bm25/
    resources/distracting_content/wikipedia/dense/

Each cache root is passed to Pyserini so that prebuilt indexes are downloaded
under this repository instead of the user's default home cache.
"""

from __future__ import annotations

import argparse
import orjson
from pathlib import Path

from src.retrieval import BM25Retriever, DenseRetriever
from src.retrieval.prebuilt_index import pyserini_cache_paths


DEFAULT_BM25_TEST_QUERY = "Machine learning"
DEFAULT_BM25_TEST_PRINT_TOP_K = 10
DEFAULT_BM25_TEST_SAVE_TOP_K = 100
DEFAULT_BM25_TEST_OUTPUT_FILENAME = "bm25_machine_learning_top100.json"


def project_root() -> Path:
    return Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    root = project_root()
    wiki_dir = root / "resources" / "distracting_content" / "wikipedia"

    parser = argparse.ArgumentParser(
        description=(
            "Download Pyserini's prebuilt Wikipedia BM25 and DPR indexes into "
            "the repository."
        )
    )
    parser.add_argument(
        "--wiki-dir",
        type=Path,
        default=wiki_dir,
        help="Output directory for local Pyserini cache roots.",
    )
    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Do not prepare the prebuilt BM25 index.",
    )
    parser.add_argument(
        "--skip-bm25-index",
        dest="skip_bm25",
        action="store_true",
        help="Alias for --skip-bm25.",
    )
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="Do not prepare the prebuilt dense index.",
    )
    parser.add_argument(
        "--bm25-index-name",
        default="wikipedia-dpr-100w",
        help="Pyserini prebuilt BM25 index name.",
    )
    parser.add_argument(
        "--dense-index-name",
        default="wikipedia-dpr-100w.dpr-multi",
        help="Pyserini prebuilt dense index name.",
    )
    parser.add_argument(
        "--dense-encoder-name",
        default="facebook/dpr-question_encoder-multiset-base",
        help="Pyserini DPR question encoder name.",
    )
    parser.add_argument(
        "--dense-faiss-device",
        default="cpu",
        help="FAISS device for the prebuilt dense retriever.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional query to run against the prepared retrievers.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of query results to print when --query is set.",
    )
    return parser.parse_args()


def print_paths(
    wiki_dir: Path,
    bm25_cache_dir: Path,
    dense_cache_dir: Path,
    bm25_index_name: str,
    dense_index_name: str,
) -> None:
    print(f"wiki dir:         {wiki_dir}")
    print(f"BM25 index name:  {bm25_index_name}")
    print(f"BM25 cache root:  {bm25_cache_dir}")
    for path in pyserini_cache_paths(bm25_cache_dir):
        print(f"BM25 cache path:  {path}")
    print(f"dense index name: {dense_index_name}")
    print(f"dense cache root: {dense_cache_dir}")
    for path in pyserini_cache_paths(dense_cache_dir):
        print(f"dense cache path: {path}")


def prepare_bm25_retriever(
    index_name: str,
    cache_dir: Path,
) -> BM25Retriever:
    bm25 = BM25Retriever(
        index_name=index_name,
        cache_dir=cache_dir,
    )
    print(f"Ensuring BM25 index is downloaded: {index_name!r}")
    bm25.load_index()
    print(f"BM25 index ready: {index_name!r}")
    return bm25


def prepare_dense_retriever(
    index_name: str,
    encoder_name: str,
    faiss_device: str,
    cache_dir: Path,
    load_searcher: bool = False,
) -> DenseRetriever:
    dense = DenseRetriever(
        index_name=index_name,
        encoder_name=encoder_name,
        faiss_device=faiss_device,
        cache_dir=cache_dir,
    )
    if load_searcher:
        print(
            "Ensuring dense retriever index is downloaded: "
            f"{index_name!r} (faiss_device={faiss_device})"
        )
        dense.load_index()
        print(f"Dense retriever index ready: {index_name!r}")
    else:
        print(f"Ensuring dense retriever index is downloaded: {index_name!r}")
        dense.download_prebuilt_index()
        print(f"Dense retriever index downloaded: {index_name!r}")
    return dense


def print_hits(label: str, hits) -> None:
    from tqdm.auto import tqdm

    print(f"\n{label}")
    for hit in tqdm(hits, desc=f"Printing {label}", unit="hit"):
        print(f"[{hit.score:7.3f}] id={hit.id}  title={hit.title!r}")
        print(f"          {hit.content!r}\n")


def save_hits_to_json(
    output_path: Path,
    query: str,
    retriever_name: str,
    index_name: str,
    hits,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": query,
        "retriever": retriever_name,
        "index_name": index_name,
        "top_k": len(hits),
        "results": [
            {
                "rank": rank,
                "docid": hit.id,
                "score": hit.score,
                "title": hit.title,
                "content": hit.content,
            }
            for rank, hit in enumerate(hits, start=1)
        ],
    }
    output_path.write_text(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8"),
        encoding="utf-8",
    )


def run_bm25_test_query(
    bm25_retriever: BM25Retriever,
    index_name: str,
    output_path: Path,
    print_top_k: int = DEFAULT_BM25_TEST_PRINT_TOP_K,
    save_top_k: int = DEFAULT_BM25_TEST_SAVE_TOP_K,
) -> None:
    print(f"\nRunning BM25 test query: {DEFAULT_BM25_TEST_QUERY!r}")
    hits = bm25_retriever.retrieve(DEFAULT_BM25_TEST_QUERY, k=save_top_k)
    print_hits("BM25 results", hits[:print_top_k])
    save_hits_to_json(
        output_path=output_path,
        query=DEFAULT_BM25_TEST_QUERY,
        retriever_name="BM25Retriever",
        index_name=index_name,
        hits=hits,
    )
    print(f"Saved {len(hits)} BM25 passages to {output_path}")


def run_query(
    query: str,
    top_k: int,
    bm25_retriever: BM25Retriever | None,
    dense_retriever: DenseRetriever | None,
) -> None:
    if bm25_retriever is not None:
        print_hits("BM25 results", bm25_retriever.retrieve(query, k=top_k))

    if dense_retriever is not None:
        print_hits("Dense results", dense_retriever.retrieve(query, k=top_k))


def main() -> None:
    args = parse_args()
    wiki_dir = args.wiki_dir.resolve()
    bm25_cache_dir = wiki_dir / "bm25"
    dense_cache_dir = wiki_dir / "dense"

    wiki_dir.mkdir(parents=True, exist_ok=True)
    bm25_cache_dir.mkdir(parents=True, exist_ok=True)
    dense_cache_dir.mkdir(parents=True, exist_ok=True)

    print_paths(
        wiki_dir=wiki_dir,
        bm25_cache_dir=bm25_cache_dir,
        dense_cache_dir=dense_cache_dir,
        bm25_index_name=args.bm25_index_name,
        dense_index_name=args.dense_index_name,
    )

    bm25_retriever: BM25Retriever | None = None
    dense_retriever: DenseRetriever | None = None

    if not args.skip_bm25:
        bm25_retriever = prepare_bm25_retriever(
            index_name=args.bm25_index_name,
            cache_dir=bm25_cache_dir,
        )

    if not args.skip_dense:
        dense_retriever = prepare_dense_retriever(
            index_name=args.dense_index_name,
            encoder_name=args.dense_encoder_name,
            faiss_device=args.dense_faiss_device,
            cache_dir=dense_cache_dir,
            load_searcher=args.query is not None,
        )

    if args.query is not None:
        run_query(
            query=args.query,
            top_k=args.top_k,
            bm25_retriever=bm25_retriever,
            dense_retriever=dense_retriever,
        )
    elif bm25_retriever is not None:
        run_bm25_test_query(
            bm25_retriever=bm25_retriever,
            index_name=args.bm25_index_name,
            output_path=project_root() / "temp" / DEFAULT_BM25_TEST_OUTPUT_FILENAME,
        )


if __name__ == "__main__":
    main()
