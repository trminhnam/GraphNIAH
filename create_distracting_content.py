#!/usr/bin/env python3
"""
create_distracting_content.py - Add distracting content to needle questions.

This script consumes the JSON output from create_needles.py. It preserves the
input metadata and question fields, then adds:

- metadata["distracting_content"]: metadata for this distractor pass.
- question["distracting_content_items"]: distractor items for each question.

Distractor items are serialized references with a small prefix:

- "path:<path>" for file-backed distractors.
- "repeat:<content>" for synthetic repeated-content controls.
- "wiki_bm25:<paragraph_id>" for BM25-retrieved Wikipedia passages.
- "wiki_dense:<paragraph_id>" for dense-retrieved Wikipedia passages.

Arguments
---------

--questions
  Required path to the JSON produced by create_needles.py.

--graph
  Graph JSON path, graphs directory, or output root. Required for
  --distracting-content-type=paper-random, paper-hard-sparse, and
  paper-hard-dense. Optional for Wikipedia distractors, where it helps
  resolve the graph DB for query expansion.

--db-path
  Optional explicit SQLite DB path. Used as an override when the script needs
  DB-backed information such as paper titles or author names.

--distracting-content-type
  Selects the distractor source or no-distractor control. Supported values are
  needles-only, repeat-needles, books, code, essays, paper-random,
  paper-related, paper-hard-sparse, paper-hard-dense, wiki-dense, wiki-sparse,
  repeat-sentence, repeat-symbol, plus the mixed placeholder.
  Each type uses its built-in distractor count from
  NUM_DISTRACTING_ITEMS_BY_TYPE.

--distracting-source-dir
  Optional override for the distractor source directory. For Wikipedia and
  hard-paper distractors, pass either the cache root containing bm25/ and
  dense/, or one cache directory directly.

--code-source-dir
  Optional compatibility override for code distractors when
  --distracting-source-dir is not provided.

--random-seed
  Optional seed for deterministic sampling of file-backed distractors.

--repeat-sentence
  Content stored in the repeat item when
  --distracting-content-type=repeat-sentence.

--repeat-symbols
  Content stored in the repeat item when
  --distracting-content-type=repeat-symbol.

--output
  Optional output JSON path. If omitted, the input file is overwritten.

--indent
  JSON indentation width for the written output file.

Usage
-----

python create_distracting_content.py \
    --questions output/needles/logitscope_qa_1000.json \
    --distracting-content-type wiki-sparse \
    --distracting-source-dir resources/distracting_content/wikipedia \
    --output output/needles/logitscope_qa_1000_with_distractors.json
"""

from __future__ import annotations

import argparse
import orjson
import random
from collections.abc import Collection
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from src.graph.readers import CitationGraphLoader, DBReader
from src.graph.resolution import (
    resolve_db_path,
    resolve_graph_argument,
    resolve_graph_db_paths,
)
from src.retrieval import BM25Retriever, DenseRetriever
from src.retrieval.paper_retriever import (
    DEFAULT_PAPER_PARAGRAPH_CORPUS_FILENAME,
    PaperBM25Retriever,
    PaperDenseRetriever,
    paper_dense_parent_id,
)
from src.setup.pipeline_parameters import (
    DISTRACTING_CONTENT_TYPE_PARAMETER_FIELD,
    PARAMETERS_FIELD,
    merge_parameters,
)
from src.setup.constants import (
    DB_PATH_FIELD,
    DB_RESOLUTION_FIELD,
    CODE_FILE_EXTS,
    DEFAULT_BOOKS_DISTRACTING_DIR,
    DEFAULT_CODE_DISTRACTING_DIR,
    DEFAULT_ESSAYS_DISTRACTING_DIR,
    DEFAULT_MAX_DISTRACTING_PATHS,
    DEFAULT_PAPER_DISTRACTING_DIR,
    DEFAULT_REPEAT_SENTENCE,
    DEFAULT_REPEAT_SYMBOLS,
    DEFAULT_WIKIPEDIA_DISTRACTING_DIR,
    DISTRACTING_CONTENT_PATHS_FIELD,
    DISTRACTING_CONTENT_FIELD,
    DISTRACTING_CONTENT_TYPE_BOOKS,
    DISTRACTING_CONTENT_TYPE_CODE,
    DISTRACTING_CONTENT_TYPE_ESSAYS,
    DISTRACTING_CONTENT_TYPE_FIELD,
    DISTRACTING_CONTENT_TYPE_MIXED,
    DISTRACTING_CONTENT_TYPE_NEEDLES_ONLY,
    DISTRACTING_CONTENT_TYPE_PAPER_HARD_DENSE,
    DISTRACTING_CONTENT_TYPE_PAPER_HARD_SPARSE,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V4,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V5,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V6,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V7,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V8,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V10,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V11,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V9,
    DISTRACTING_CONTENT_TYPE_PAPER_RELATED,
    DISTRACTING_CONTENT_TYPE_PAPER_RANDOM,
    DISTRACTING_CONTENT_TYPE_REPEAT_SENTENCE,
    DISTRACTING_CONTENT_TYPE_REPEAT_NEEDLES,
    DISTRACTING_CONTENT_TYPE_REPEAT_SYMBOL,
    DISTRACTING_CONTENT_TYPE_WIKI_DENSE,
    DISTRACTING_CONTENT_TYPE_WIKI_SPARSE,
    DISTRACTING_RELATED_EDGE_TYPES,
    EDGE_TYPE_FIELD,
    GRAPH_JSON_FIELD,
    NODE_AUTHOR,
    MAX_DISTRACTING_PATHS_FIELD,
    METADATA_FIELD,
    NODE_PAPER,
    NODE_IDS_FIELD,
    PAPER_SORT_FALLBACK_DB_ID,
    PATH_ITEM_PREFIX,
    QUESTIONS_FIELD,
    RANDOM_SEED_FIELD,
    REPEAT_ITEM_PREFIX,
    REPEAT_VALUE_FIELD,
    SOURCE_DIR_FIELD,
    SOURCE_QUESTIONS_FIELD,
    SUPPORTED_DISTRACTING_CONTENT_TYPES,
    TEXT_FILE_EXTS,
    TEXT_PATH_FIELD,
    TOTAL_QUESTIONS_FIELD,
    WIKI_BM25_ITEM_PREFIX,
    WIKI_DENSE_ITEM_PREFIX,
    nid_db_id,
    nid_type,
)


CLI_NAME = "create_distracting_content"

UNSUPPORTED_DISTRACTING_CONTENT_TYPE_ERROR = "Unsupported distracting content type: {content_type}"
EMPTY_SOURCE_DIR_ERROR = "No {content_type} distractor files found in {dir}"
GRAPH_REQUIRED_ERROR = "--graph is required when --distracting-content-type={type}"
INVALID_METADATA_ERROR = "Input metadata must be a JSON object"
INVALID_QUESTIONS_DATA_ERROR = (
    "Input file must be a create_needles JSON object with a questions list"
)
INVALID_QUESTION_ENTRY_ERROR = "Each question entry in the input file must be a JSON object"
INDEX_NAME_FIELD = "index_name"
ENCODER_NAME_FIELD = "encoder_name"
FAISS_DEVICE_FIELD = "faiss_device"
CORPUS_PATH_FIELD = "corpus_path"
WIKI_BM25_CACHE_SUBDIR = "bm25"
WIKI_DENSE_CACHE_SUBDIR = "dense"
PAPER_BM25_CACHE_SUBDIR = "bm25"
PAPER_DENSE_CACHE_SUBDIR = "dense"
PAPER_CORPUS_FILENAME = "corpus.jsonl"
PAPER_PARAGRAPH_CORPUS_FILENAME = DEFAULT_PAPER_PARAGRAPH_CORPUS_FILENAME
HARD_PAPER_RETRIEVAL_OVERSAMPLE = 4


NUM_DISTRACTING_ITEMS_BY_TYPE = {
    DISTRACTING_CONTENT_TYPE_BOOKS: 128,
    DISTRACTING_CONTENT_TYPE_CODE: 192,
    DISTRACTING_CONTENT_TYPE_ESSAYS: 32,
    DISTRACTING_CONTENT_TYPE_NEEDLES_ONLY: 0,
    DISTRACTING_CONTENT_TYPE_PAPER_RANDOM: 8_192,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE: 8_192,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V4: 8_192,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V5: 8_192,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V6: 8_192,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V7: 8_192,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V8: 8_192,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V10: 8_192,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V11: 8_192,
    DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V9: 8_192,
    DISTRACTING_CONTENT_TYPE_PAPER_RELATED: 8_192,
    DISTRACTING_CONTENT_TYPE_WIKI_DENSE: 16_384,
    DISTRACTING_CONTENT_TYPE_WIKI_SPARSE: 16_384,
    DISTRACTING_CONTENT_TYPE_PAPER_HARD_DENSE: 16_384,
    DISTRACTING_CONTENT_TYPE_PAPER_HARD_SPARSE: 16_384,
    DISTRACTING_CONTENT_TYPE_REPEAT_SENTENCE: 1,
    DISTRACTING_CONTENT_TYPE_REPEAT_NEEDLES: 0,
    DISTRACTING_CONTENT_TYPE_REPEAT_SYMBOL: 1,
}


def paper_sort_key(node_id: str) -> tuple[int, str]:
    """Sort paper node IDs by DB ID when possible."""
    try:
        return nid_db_id(node_id), node_id
    except (IndexError, ValueError):
        return PAPER_SORT_FALLBACK_DB_ID, node_id


def make_path_item(path: str | Path) -> str:
    """Return the serialized item for a file-backed distractor."""
    return f"{PATH_ITEM_PREFIX}{path}"


def make_repeat_item(content: str) -> str:
    """Return the serialized item for a repeated-content distractor."""
    return f"{REPEAT_ITEM_PREFIX}{content}"


def make_wiki_bm25_item(paragraph_id: str | int) -> str:
    """Return the serialized item for a BM25 Wikipedia distractor."""
    return f"{WIKI_BM25_ITEM_PREFIX}{paragraph_id}"


def make_wiki_dense_item(paragraph_id: str | int) -> str:
    """Return the serialized item for a dense Wikipedia distractor."""
    return f"{WIKI_DENSE_ITEM_PREFIX}{paragraph_id}"


def make_wiki_item(
    paragraph_id: str | int,
    distracting_content_type: str,
) -> str:
    """Return the serialized item for a Wikipedia distractor."""
    if distracting_content_type == DISTRACTING_CONTENT_TYPE_WIKI_SPARSE:
        return make_wiki_bm25_item(paragraph_id)
    if distracting_content_type == DISTRACTING_CONTENT_TYPE_WIKI_DENSE:
        return make_wiki_dense_item(paragraph_id)
    raise ValueError(
        UNSUPPORTED_DISTRACTING_CONTENT_TYPE_ERROR.format(
            content_type=distracting_content_type,
        )
    )


def get_related_node_ids(graph: CitationGraphLoader, node_ids: set[str]) -> set[str]:
    """Return nodes directly connected to any node in *node_ids*."""
    related_ids: set[str] = set()
    for source_id, neighbours in graph.adj.items():
        for target_id, edge_attrs in neighbours:
            if edge_attrs.get(EDGE_TYPE_FIELD) not in DISTRACTING_RELATED_EDGE_TYPES:
                continue
            if source_id in node_ids:
                related_ids.add(target_id)
            if target_id in node_ids:
                related_ids.add(source_id)
    return related_ids


def get_paper_text_path(node_id: str, db: DBReader) -> str | None:
    """Return a paper text path only when the DB row and file path are usable."""
    paper = db.get_paper(nid_db_id(node_id))
    if paper is None:
        return None

    text_path = paper.get(TEXT_PATH_FIELD)
    if not text_path:
        return None

    path = Path(text_path)
    if not path.exists():
        return None
    return str(path.resolve())


def normalize_paper_text_path(path: str | Path) -> str:
    """Return one normalized absolute paper text path."""
    return str(Path(path).resolve())


def build_paper_text_path_node_map(
    graph: CitationGraphLoader,
    db: DBReader,
) -> dict[str, str]:
    """Map resolved paper text paths back to canonical graph node ids."""
    path_to_node_id: dict[str, str] = {}
    for node_id in sorted(graph.nodes):
        if nid_type(node_id) != NODE_PAPER:
            continue
        text_path = get_paper_text_path(node_id, db)
        if text_path is None:
            continue
        path_to_node_id[normalize_paper_text_path(text_path)] = node_id
    return path_to_node_id


def resolve_hard_paper_hit_node_id(
    hit_id: str,
    hit_path: str,
    graph: CitationGraphLoader,
    paper_text_path_node_ids: dict[str, str] | None,
) -> str | None:
    """Resolve one retrieved paper hit back to a canonical graph node id."""
    if hit_id in graph.nodes:
        return hit_id
    parent_paper_id = paper_dense_parent_id(hit_id)
    if parent_paper_id in graph.nodes:
        return parent_paper_id
    if not hit_path or paper_text_path_node_ids is None:
        return None
    return paper_text_path_node_ids.get(normalize_paper_text_path(hit_path))


def collect_paper_distracting_content_paths(
    question: dict,
    db: DBReader,
    graph: CitationGraphLoader,
    max_paths: int = DEFAULT_MAX_DISTRACTING_PATHS,
) -> list[str]:
    """Collect unrelated paper text paths for one question."""
    path_node_ids = set(question.get(NODE_IDS_FIELD, []))
    avoidance_ids = path_node_ids | get_related_node_ids(graph, path_node_ids)
    all_paper_ids = {node_id for node_id in graph.nodes if nid_type(node_id) == NODE_PAPER}
    distracting_candidate_ids = all_paper_ids - avoidance_ids

    distracting_content_paths: list[str] = []
    for candidate_id in sorted(distracting_candidate_ids, key=paper_sort_key):
        text_path = get_paper_text_path(candidate_id, db)
        if text_path is None:
            continue

        distracting_content_paths.append(make_path_item(text_path))
        if len(distracting_content_paths) >= max_paths:
            break

    return distracting_content_paths


def list_distractor_paths(
    source_dir: Path,
    file_exts: Collection[str],
) -> list[Path]:
    """Return all candidate files in *source_dir* sorted for determinism."""
    if not source_dir.exists():
        return []
    return sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in file_exts)


def collect_file_distracting_content_paths(
    candidate_paths: list[Path],
    rng: random.Random,
    max_paths: int = DEFAULT_MAX_DISTRACTING_PATHS,
) -> list[str]:
    """Return sampled file distractors as path-prefixed items."""
    if max_paths == 0:
        return []

    expanded_paths = list(candidate_paths)
    while len(expanded_paths) < max_paths:
        expanded_paths.extend(candidate_paths)

    sampled = rng.sample(expanded_paths, max_paths)
    return [make_path_item(p.resolve()) for p in sampled]


def collect_repeat_distracting_content_paths(repeat_value: str) -> list[str]:
    """Return one repeat-prefixed item; downstream code expands it to length."""
    return [make_repeat_item(repeat_value)]


def get_wikipedia_query_for_node(node_id: str, db: DBReader) -> str | None:
    """Return the canonical Wikipedia query string for one graph node."""
    node_type = nid_type(node_id)
    db_id = nid_db_id(node_id)

    if node_type == NODE_AUTHOR:
        author = db.get_author(db_id)
        if author is None:
            return None
        query = str(author.get("name", "") or "").strip()
        return query or None

    if node_type == NODE_PAPER:
        paper = db.get_paper(db_id)
        if paper is None:
            return None
        query = str(paper.get("title", "") or "").strip()
        return query or None

    return None


def collect_wikipedia_queries(
    question: dict,
    db: DBReader | None = None,
) -> list[str]:
    """Return Wikipedia queries from path entities plus the question text."""
    wikipedia_queries: list[str] = []
    if db is not None:
        for node_id in question.get(NODE_IDS_FIELD, []):
            if not isinstance(node_id, str):
                continue
            query = get_wikipedia_query_for_node(node_id, db)
            if query is not None:
                wikipedia_queries.append(query)

    question_query = str(question.get("question", "") or "").strip()
    if question_query:
        wikipedia_queries.append(question_query)

    return wikipedia_queries


def open_distractor_db_reader(
    graph_json: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
    db_path_hint: str | Path | None,
    input_metadata: Any,
) -> DBReader | None:
    """Return a DB reader for distractor query expansion when one is available."""
    if db_path_hint is not None and graph_json is None:
        db_path = Path(db_path_hint)
        if not db_path.exists():
            raise FileNotFoundError(f"Explicitly provided database path does not exist: {db_path}")
        return DBReader(db_path)

    if graph_json is not None:
        _, resolved_db, _ = resolve_graph_db_paths(graph_json, db_path_hint)
        return DBReader(resolved_db)

    if not isinstance(input_metadata, dict):
        return None

    metadata_db_path = input_metadata.get(DB_PATH_FIELD)
    if metadata_db_path:
        db_path = Path(metadata_db_path)
        if db_path.exists():
            return DBReader(db_path)

    metadata_graph_json = input_metadata.get(GRAPH_JSON_FIELD)
    if not metadata_graph_json:
        return None

    graph_path = Path(metadata_graph_json)
    if not graph_path.exists():
        return None

    try:
        resolved_db, _ = resolve_db_path(
            graph_path,
            None,
            meta_db_path=str(metadata_db_path or ""),
        )
    except FileNotFoundError:
        return None

    return DBReader(resolved_db)


def collect_wiki_distracting_content_paths(
    question: dict,
    retriever: BM25Retriever | DenseRetriever,
    distracting_content_type: str,
    max_paths: int = DEFAULT_MAX_DISTRACTING_PATHS,
    db: DBReader | None = None,
) -> list[str]:
    """Return Wikipedia distractor IDs for one question."""
    if max_paths == 0:
        return []

    wikipedia_queries = collect_wikipedia_queries(question, db)
    if not wikipedia_queries:
        return []

    max_paths_per_query = max_paths // len(wikipedia_queries)
    if max_paths_per_query == 0:
        return []

    seen_ids: set[str] = set()
    distracting_content_paths: list[str] = []
    for query in wikipedia_queries:
        query_hits = 0
        for hit in retriever.retrieve(query, k=max_paths_per_query):
            paragraph_id = str(hit.id)
            if paragraph_id in seen_ids:
                continue
            seen_ids.add(paragraph_id)
            distracting_content_paths.append(make_wiki_item(paragraph_id, distracting_content_type))
            query_hits += 1
            if query_hits >= max_paths_per_query:
                break

    return distracting_content_paths


def collect_hard_paper_distracting_content_paths(
    question: dict,
    retriever: PaperBM25Retriever | PaperDenseRetriever,
    graph: CitationGraphLoader,
    max_paths: int = DEFAULT_MAX_DISTRACTING_PATHS,
    db: DBReader | None = None,
    paper_text_path_node_ids: dict[str, str] | None = None,
) -> list[str]:
    """Return graph-filtered hard-paper distractor paths for one question."""
    if max_paths == 0:
        return []

    paper_queries = collect_wikipedia_queries(question, db)
    if not paper_queries:
        return []

    path_node_ids = {
        node_id for node_id in question.get(NODE_IDS_FIELD, []) if isinstance(node_id, str)
    }
    avoidance_ids = path_node_ids | get_related_node_ids(graph, path_node_ids)

    max_paths_per_query = max_paths // len(paper_queries)
    if max_paths_per_query == 0:
        return []

    seen_ids: set[str] = set()
    distracting_content_paths: list[str] = []
    for query in paper_queries:
        query_hits = 0
        retrieval_k = max_paths
        if isinstance(retriever, (PaperBM25Retriever, PaperDenseRetriever)):
            retrieval_k *= HARD_PAPER_RETRIEVAL_OVERSAMPLE
        for hit in retriever.retrieve(query, k=retrieval_k):
            candidate_id = str(hit.id)
            text_path = str(hit.path or "").strip()
            resolved_node_id = resolve_hard_paper_hit_node_id(
                candidate_id,
                text_path,
                graph,
                paper_text_path_node_ids,
            )
            if resolved_node_id is None:
                continue
            if resolved_node_id in seen_ids or resolved_node_id in avoidance_ids:
                continue
            if nid_type(resolved_node_id) != NODE_PAPER:
                continue

            if not text_path and db is not None:
                resolved_text_path = get_paper_text_path(resolved_node_id, db)
                if resolved_text_path is not None:
                    text_path = resolved_text_path
            if not text_path:
                continue

            path = Path(text_path)
            if not path.exists():
                continue

            seen_ids.add(resolved_node_id)
            distracting_content_paths.append(make_path_item(path.resolve()))
            query_hits += 1
            if query_hits >= max_paths_per_query:
                break

    return distracting_content_paths


def add_distracting_content_paths(question: dict, content_paths: list[str]) -> dict:
    """Return a copy of *question* with distracting content items attached."""
    output_question = dict(question)
    output_question[DISTRACTING_CONTENT_PATHS_FIELD] = content_paths
    return output_question


def process_paper_question(
    question: dict,
    db: DBReader,
    graph: CitationGraphLoader,
    max_paths: int = DEFAULT_MAX_DISTRACTING_PATHS,
) -> dict:
    """Return a question with graph-filtered paper distractors attached."""
    return add_distracting_content_paths(
        question,
        collect_paper_distracting_content_paths(question, db, graph, max_paths),
    )


def process_file_question(
    question: dict,
    candidate_paths: list[Path],
    rng: random.Random,
    max_paths: int = DEFAULT_MAX_DISTRACTING_PATHS,
) -> dict:
    """Return a question with sampled file-backed distractors attached."""
    return add_distracting_content_paths(
        question,
        collect_file_distracting_content_paths(candidate_paths, rng, max_paths),
    )


def process_repeat_question(question: dict, repeat_value: str) -> dict:
    """Return a question with one synthetic repeat distractor attached."""
    return add_distracting_content_paths(
        question,
        collect_repeat_distracting_content_paths(repeat_value),
    )


def process_wiki_question(
    question: dict,
    retriever: BM25Retriever | DenseRetriever,
    distracting_content_type: str,
    max_paths: int = DEFAULT_MAX_DISTRACTING_PATHS,
    db: DBReader | None = None,
) -> dict:
    """Return a question with retrieved Wikipedia distractors attached."""
    return add_distracting_content_paths(
        question,
        collect_wiki_distracting_content_paths(
            question,
            retriever,
            distracting_content_type,
            max_paths,
            db,
        ),
    )


def process_hard_paper_question(
    question: dict,
    retriever: PaperBM25Retriever | PaperDenseRetriever,
    graph: CitationGraphLoader,
    max_paths: int = DEFAULT_MAX_DISTRACTING_PATHS,
    db: DBReader | None = None,
    paper_text_path_node_ids: dict[str, str] | None = None,
) -> dict:
    """Return a question with graph-filtered hard-paper distractors attached."""
    return add_distracting_content_paths(
        question,
        collect_hard_paper_distracting_content_paths(
            question,
            retriever,
            graph,
            max_paths,
            db,
            paper_text_path_node_ids,
        ),
    )


def validate_distracting_content_type(distracting_content_type: str) -> None:
    """Raise when *distracting_content_type* is unknown."""
    if distracting_content_type not in SUPPORTED_DISTRACTING_CONTENT_TYPES:
        raise ValueError(
            UNSUPPORTED_DISTRACTING_CONTENT_TYPE_ERROR.format(
                content_type=distracting_content_type,
            )
        )


def get_num_distracting_items(distracting_content_type: str) -> int:
    """Return the built-in per-question distractor cap for one type."""
    return NUM_DISTRACTING_ITEMS_BY_TYPE.get(
        distracting_content_type,
        DEFAULT_MAX_DISTRACTING_PATHS,
    )


def raise_not_implemented_distractor(distracting_content_type: str) -> None:
    """Raise the placeholder error for retrieval-based distractors."""
    raise NotImplementedError(
        f"{distracting_content_type} distracting content is not implemented yet."
    )


def build_base_distracting_metadata(
    questions_path: Path,
    distracting_content_type: str,
    max_distracting_paths: int,
) -> dict[str, Any]:
    """Return metadata for this distracting-content pass."""
    del questions_path
    return {
        DISTRACTING_CONTENT_TYPE_FIELD: distracting_content_type,
        MAX_DISTRACTING_PATHS_FIELD: max_distracting_paths,
    }


def add_distracting_metadata(
    questions_path: Path,
    distracting_metadata: dict[str, Any],
    total_questions: int,
) -> dict[str, Any]:
    """Return the minimal metadata block for one distractor pass."""
    return {
        SOURCE_QUESTIONS_FIELD: str(questions_path),
        TOTAL_QUESTIONS_FIELD: total_questions,
        DISTRACTING_CONTENT_FIELD: distracting_metadata,
    }


def validate_questions_data(questions_data: Any) -> list[dict[str, Any]]:
    """Return the question list from a create_needles output payload."""
    if not isinstance(questions_data, dict):
        raise ValueError(INVALID_QUESTIONS_DATA_ERROR)

    questions = questions_data.get(QUESTIONS_FIELD)
    if not isinstance(questions, list):
        raise ValueError(INVALID_QUESTIONS_DATA_ERROR)

    for question in questions:
        if not isinstance(question, dict):
            raise ValueError(INVALID_QUESTION_ENTRY_ERROR)

    return questions


def resolve_file_source_dir(
    default_source_dir: str | Path,
    distracting_source_dir: str | Path | None,
) -> Path:
    """Return the source directory for a non-code file distractor."""
    if distracting_source_dir is not None:
        return Path(distracting_source_dir)
    return Path(default_source_dir)


def resolve_wiki_cache_dir(
    distracting_source_dir: str | Path | None,
    cache_subdir: str,
) -> Path:
    """Return the Wikipedia retriever cache dir."""
    if distracting_source_dir is None:
        return Path(DEFAULT_WIKIPEDIA_DISTRACTING_DIR) / cache_subdir

    source_dir = Path(distracting_source_dir)
    if source_dir.name == cache_subdir:
        return source_dir
    return source_dir / cache_subdir


def resolve_paper_root_dir(
    distracting_source_dir: str | Path | None,
) -> Path:
    """Return the local paper index root directory."""
    if distracting_source_dir is None:
        return Path(DEFAULT_PAPER_DISTRACTING_DIR)

    source_dir = Path(distracting_source_dir)
    if source_dir.name in {PAPER_BM25_CACHE_SUBDIR, PAPER_DENSE_CACHE_SUBDIR}:
        return source_dir.parent
    return source_dir


def resolve_paper_index_dir(
    distracting_source_dir: str | Path | None,
    cache_subdir: str,
) -> Path:
    """Return the local paper BM25 or dense index directory."""
    if distracting_source_dir is not None:
        source_dir = Path(distracting_source_dir)
        if source_dir.name == cache_subdir:
            return source_dir

    return resolve_paper_root_dir(distracting_source_dir) / cache_subdir


def resolve_paper_corpus_path(
    distracting_source_dir: str | Path | None,
) -> Path:
    """Return the local paper corpus JSONL path."""
    return resolve_paper_root_dir(distracting_source_dir) / PAPER_CORPUS_FILENAME


def resolve_paper_dense_corpus_path(
    distracting_source_dir: str | Path | None,
) -> Path:
    """Return the local shared paper paragraph corpus JSONL path."""
    return resolve_paper_root_dir(distracting_source_dir) / PAPER_PARAGRAPH_CORPUS_FILENAME


def progress_questions(
    questions: list[dict[str, Any]],
    *,
    desc: str,
) -> Any:
    """Return a progress-wrapped question iterator."""
    return tqdm(questions, desc=desc, unit="question")


def create_sampled_file_distracting_questions(
    questions: list[dict],
    distracting_content_type: str,
    source_dir: Path,
    file_exts: Collection[str],
    random_seed: int | None,
    max_distracting_paths: int,
) -> tuple[list[dict], dict[str, Any]]:
    """Create questions with sampled file-backed distractors."""
    candidate_paths = list_distractor_paths(source_dir, file_exts)
    if not candidate_paths:
        raise ValueError(
            EMPTY_SOURCE_DIR_ERROR.format(
                content_type=distracting_content_type,
                dir=source_dir,
            )
        )

    rng = random.Random(random_seed)
    output_questions = [
        process_file_question(q, candidate_paths, rng, max_distracting_paths)
        for q in progress_questions(
            questions,
            desc=f"Adding {distracting_content_type} distractors",
        )
    ]
    metadata = {
        SOURCE_DIR_FIELD: str(source_dir),
    }
    if random_seed is not None:
        metadata[RANDOM_SEED_FIELD] = random_seed
    return output_questions, metadata


def create_books_distracting_questions(
    questions: list[dict],
    distracting_source_dir: str | Path | None,
    random_seed: int | None,
    max_distracting_paths: int,
) -> tuple[list[dict], dict[str, Any]]:
    """Create questions with sampled PG19 book distractors."""
    source_dir = resolve_file_source_dir(
        DEFAULT_BOOKS_DISTRACTING_DIR,
        distracting_source_dir,
    )
    return create_sampled_file_distracting_questions(
        questions,
        DISTRACTING_CONTENT_TYPE_BOOKS,
        source_dir,
        TEXT_FILE_EXTS,
        random_seed,
        max_distracting_paths,
    )


def create_code_distracting_questions(
    questions: list[dict],
    distracting_source_dir: str | Path | None,
    code_source_dir: str | Path | None,
    random_seed: int | None,
    max_distracting_paths: int,
) -> tuple[list[dict], dict[str, Any]]:
    """Create questions with sampled code distractors."""
    if distracting_source_dir is not None:
        source_dir = Path(distracting_source_dir)
    elif code_source_dir is not None:
        source_dir = Path(code_source_dir)
    else:
        source_dir = Path(DEFAULT_CODE_DISTRACTING_DIR)

    return create_sampled_file_distracting_questions(
        questions,
        DISTRACTING_CONTENT_TYPE_CODE,
        source_dir,
        CODE_FILE_EXTS,
        random_seed,
        max_distracting_paths,
    )


def create_essays_distracting_questions(
    questions: list[dict],
    distracting_source_dir: str | Path | None,
    random_seed: int | None,
    max_distracting_paths: int,
) -> tuple[list[dict], dict[str, Any]]:
    """Create questions with sampled essay distractors."""
    source_dir = resolve_file_source_dir(
        DEFAULT_ESSAYS_DISTRACTING_DIR,
        distracting_source_dir,
    )
    return create_sampled_file_distracting_questions(
        questions,
        DISTRACTING_CONTENT_TYPE_ESSAYS,
        source_dir,
        TEXT_FILE_EXTS,
        random_seed,
        max_distracting_paths,
    )


def create_graph_filtered_paper_distracting_questions(
    questions: list[dict],
    graph_json: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
    db_path_hint: str | Path | None,
    distracting_content_type: str,
    max_distracting_paths: int,
) -> tuple[list[dict], dict[str, Any]]:
    """Create questions with graph-filtered random paper distractors."""
    if graph_json is None:
        raise ValueError(GRAPH_REQUIRED_ERROR.format(type=distracting_content_type))

    graph_json, resolved_db, db_resolution = resolve_graph_db_paths(
        graph_json,
        db_path_hint,
    )
    db = DBReader(resolved_db)
    graph = CitationGraphLoader(graph_json)
    try:
        output_questions = [
            process_paper_question(q, db, graph, max_distracting_paths)
            for q in progress_questions(
                questions,
                desc=f"Adding {distracting_content_type} distractors",
            )
        ]
    finally:
        db.close()

    return output_questions, {}


def create_paper_random_distracting_questions(
    questions: list[dict],
    graph_json: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
    db_path_hint: str | Path | None,
    max_distracting_paths: int,
) -> tuple[list[dict], dict[str, Any]]:
    """Create questions with graph-filtered random paper distractors."""
    return create_graph_filtered_paper_distracting_questions(
        questions,
        graph_json,
        db_path_hint,
        DISTRACTING_CONTENT_TYPE_PAPER_RANDOM,
        max_distracting_paths,
    )


def create_repeat_sentence_distracting_questions(
    questions: list[dict],
    repeat_sentence: str,
) -> tuple[list[dict], dict[str, Any]]:
    """Create questions with a repeated-sentence control distractor."""
    output_questions = [
        process_repeat_question(q, repeat_sentence)
        for q in progress_questions(
            questions,
            desc=f"Adding {DISTRACTING_CONTENT_TYPE_REPEAT_SENTENCE} distractors",
        )
    ]
    return output_questions, {REPEAT_VALUE_FIELD: repeat_sentence}


def create_repeat_symbol_distracting_questions(
    questions: list[dict],
    repeat_symbols: str,
) -> tuple[list[dict], dict[str, Any]]:
    """Create questions with a repeated-symbol control distractor."""
    output_questions = [
        process_repeat_question(q, repeat_symbols)
        for q in progress_questions(
            questions,
            desc=f"Adding {DISTRACTING_CONTENT_TYPE_REPEAT_SYMBOL} distractors",
        )
    ]
    return output_questions, {REPEAT_VALUE_FIELD: repeat_symbols}


def create_empty_distracting_questions(
    questions: list[dict],
) -> tuple[list[dict], dict[str, Any]]:
    """Return questions with an explicit empty distractor-item list."""
    return [add_distracting_content_paths(question, []) for question in questions], {}


def create_wiki_distracting_questions(
    questions: list[dict],
    distracting_content_type: str,
    distracting_source_dir: str | Path | None,
    max_distracting_paths: int,
    graph_json: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    db_path_hint: str | Path | None = None,
    input_metadata: Any = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Create questions with retrieved Wikipedia distractors."""
    if distracting_content_type == DISTRACTING_CONTENT_TYPE_WIKI_SPARSE:
        source_dir = resolve_wiki_cache_dir(
            distracting_source_dir,
            WIKI_BM25_CACHE_SUBDIR,
        )
        retriever = BM25Retriever(cache_dir=source_dir)
        extra_metadata = {
            SOURCE_DIR_FIELD: str(source_dir),
            INDEX_NAME_FIELD: retriever.index_name,
        }
    elif distracting_content_type == DISTRACTING_CONTENT_TYPE_WIKI_DENSE:
        source_dir = resolve_wiki_cache_dir(
            distracting_source_dir,
            WIKI_DENSE_CACHE_SUBDIR,
        )
        retriever = DenseRetriever(cache_dir=source_dir)
        extra_metadata = {
            SOURCE_DIR_FIELD: str(source_dir),
            INDEX_NAME_FIELD: retriever.index_name,
            ENCODER_NAME_FIELD: retriever.encoder_name,
            FAISS_DEVICE_FIELD: retriever.faiss_device,
        }
    else:
        raise ValueError(
            UNSUPPORTED_DISTRACTING_CONTENT_TYPE_ERROR.format(
                content_type=distracting_content_type,
            )
        )

    db = open_distractor_db_reader(graph_json, db_path_hint, input_metadata)
    try:
        output_questions = [
            process_wiki_question(
                question,
                retriever,
                distracting_content_type,
                max_distracting_paths,
                db,
            )
            for question in progress_questions(
                questions,
                desc=f"Adding {distracting_content_type} distractors",
            )
        ]
    finally:
        if db is not None:
            db.close()

    return output_questions, extra_metadata


def create_wiki_dense_distracting_questions(
    questions: list[dict],
    distracting_source_dir: str | Path | None,
    max_distracting_paths: int,
    graph_json: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    db_path_hint: str | Path | None = None,
    input_metadata: Any = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Create questions with dense-retrieved Wikipedia distractors."""
    return create_wiki_distracting_questions(
        questions,
        DISTRACTING_CONTENT_TYPE_WIKI_DENSE,
        distracting_source_dir,
        max_distracting_paths,
        graph_json,
        db_path_hint,
        input_metadata,
    )


def create_wiki_sparse_distracting_questions(
    questions: list[dict],
    distracting_source_dir: str | Path | None,
    max_distracting_paths: int,
    graph_json: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    db_path_hint: str | Path | None = None,
    input_metadata: Any = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Create questions with BM25-retrieved Wikipedia distractors."""
    return create_wiki_distracting_questions(
        questions,
        DISTRACTING_CONTENT_TYPE_WIKI_SPARSE,
        distracting_source_dir,
        max_distracting_paths,
        graph_json,
        db_path_hint,
        input_metadata,
    )


def create_hard_paper_distracting_questions(
    questions: list[dict],
    graph_json: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
    db_path_hint: str | Path | None,
    distracting_content_type: str,
    distracting_source_dir: str | Path | None,
    max_distracting_paths: int,
) -> tuple[list[dict], dict[str, Any]]:
    """Create graph-filtered hard-paper distractors from local paper indexes."""
    if graph_json is None:
        raise ValueError(GRAPH_REQUIRED_ERROR.format(type=distracting_content_type))

    graph_json, resolved_db, db_resolution = resolve_graph_db_paths(
        graph_json,
        db_path_hint,
    )
    del db_resolution

    if distracting_content_type == DISTRACTING_CONTENT_TYPE_PAPER_HARD_SPARSE:
        corpus_path = resolve_paper_dense_corpus_path(distracting_source_dir)
        source_dir = resolve_paper_index_dir(
            distracting_source_dir,
            PAPER_BM25_CACHE_SUBDIR,
        )
        retriever = PaperBM25Retriever(
            index_dir=source_dir,
            corpus_path=corpus_path,
        )
        extra_metadata = {
            SOURCE_DIR_FIELD: str(source_dir),
            CORPUS_PATH_FIELD: str(corpus_path),
        }
    elif distracting_content_type == DISTRACTING_CONTENT_TYPE_PAPER_HARD_DENSE:
        corpus_path = resolve_paper_dense_corpus_path(distracting_source_dir)
        source_dir = resolve_paper_index_dir(
            distracting_source_dir,
            PAPER_DENSE_CACHE_SUBDIR,
        )
        retriever = PaperDenseRetriever(
            index_dir=source_dir,
            corpus_path=corpus_path,
        )
        extra_metadata = {
            SOURCE_DIR_FIELD: str(source_dir),
            CORPUS_PATH_FIELD: str(corpus_path),
            ENCODER_NAME_FIELD: retriever.encoder_name,
            FAISS_DEVICE_FIELD: retriever.faiss_device,
        }
    else:
        raise ValueError(
            UNSUPPORTED_DISTRACTING_CONTENT_TYPE_ERROR.format(
                content_type=distracting_content_type,
            )
        )

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Paper corpus JSONL not found: {corpus_path}. " "Run build_paper_index.py first."
        )
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Paper index directory not found: {source_dir}. " "Run build_paper_index.py first."
        )

    db = DBReader(resolved_db)
    graph = CitationGraphLoader(graph_json)
    try:
        paper_text_path_node_ids = build_paper_text_path_node_map(graph, db)
        output_questions = [
            process_hard_paper_question(
                question,
                retriever,
                graph,
                max_distracting_paths,
                db,
                paper_text_path_node_ids,
            )
            for question in progress_questions(
                questions,
                desc=f"Adding {distracting_content_type} distractors",
            )
        ]
    finally:
        db.close()

    return output_questions, extra_metadata


def create_paper_hard_dense_distracting_questions(
    questions: list[dict],
    graph_json: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
    db_path_hint: str | Path | None,
    distracting_source_dir: str | Path | None,
    max_distracting_paths: int,
) -> tuple[list[dict], dict[str, Any]]:
    """Create dense hard-paper distractors."""
    return create_hard_paper_distracting_questions(
        questions,
        graph_json,
        db_path_hint,
        DISTRACTING_CONTENT_TYPE_PAPER_HARD_DENSE,
        distracting_source_dir,
        max_distracting_paths,
    )


def create_paper_hard_sparse_distracting_questions(
    questions: list[dict],
    graph_json: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
    db_path_hint: str | Path | None,
    distracting_source_dir: str | Path | None,
    max_distracting_paths: int,
) -> tuple[list[dict], dict[str, Any]]:
    """Create sparse hard-paper distractors."""
    return create_hard_paper_distracting_questions(
        questions,
        graph_json,
        db_path_hint,
        DISTRACTING_CONTENT_TYPE_PAPER_HARD_SPARSE,
        distracting_source_dir,
        max_distracting_paths,
    )


def create_mixed_distracting_questions(
    questions: list[dict],
) -> tuple[list[dict], dict[str, Any]]:
    """Placeholder for mixed distractors."""
    raise_not_implemented_distractor(DISTRACTING_CONTENT_TYPE_MIXED)


def create_distracting_content(
    questions_path: Path,
    graph_json: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    db_path_hint: str | Path | None = None,
    distracting_content_type: str = DISTRACTING_CONTENT_TYPE_PAPER_RANDOM,
    distracting_source_dir: str | Path | None = None,
    code_source_dir: str | Path | None = DEFAULT_CODE_DISTRACTING_DIR,
    random_seed: int | None = None,
    repeat_sentence: str = DEFAULT_REPEAT_SENTENCE,
    repeat_symbols: str = DEFAULT_REPEAT_SYMBOLS,
) -> dict[str, Any]:
    """Load a create_needles output JSON and add distracting content."""
    validate_distracting_content_type(distracting_content_type)
    max_distracting_paths = get_num_distracting_items(distracting_content_type)

    with open(questions_path, encoding="utf-8") as fh:
        questions_data = orjson.loads(fh.read())

    questions = validate_questions_data(questions_data)
    distracting_metadata = build_base_distracting_metadata(
        questions_path,
        distracting_content_type,
        max_distracting_paths,
    )

    if distracting_content_type in {
        DISTRACTING_CONTENT_TYPE_NEEDLES_ONLY,
        DISTRACTING_CONTENT_TYPE_REPEAT_NEEDLES,
    }:
        output_questions, extra_metadata = create_empty_distracting_questions(questions)

    elif distracting_content_type == DISTRACTING_CONTENT_TYPE_BOOKS:
        output_questions, extra_metadata = create_books_distracting_questions(
            questions,
            distracting_source_dir,
            random_seed,
            max_distracting_paths,
        )

    elif distracting_content_type == DISTRACTING_CONTENT_TYPE_CODE:
        output_questions, extra_metadata = create_code_distracting_questions(
            questions,
            distracting_source_dir,
            code_source_dir,
            random_seed,
            max_distracting_paths,
        )

    elif distracting_content_type == DISTRACTING_CONTENT_TYPE_ESSAYS:
        output_questions, extra_metadata = create_essays_distracting_questions(
            questions,
            distracting_source_dir,
            random_seed,
            max_distracting_paths,
        )

    elif distracting_content_type == DISTRACTING_CONTENT_TYPE_PAPER_RANDOM:
        output_questions, extra_metadata = create_paper_random_distracting_questions(
            questions,
            graph_json,
            db_path_hint,
            max_distracting_paths,
        )

    elif distracting_content_type in {
        DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE,
        DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V4,
        DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V5,
        DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V6,
        DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V7,
        DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V8,
        DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V10,
        DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V11,
        DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V9,
    }:
        raise ValueError(
            "Negative-tree distractors require construction metadata; use "
            "construct_distracting_content.py."
        )

    elif distracting_content_type == DISTRACTING_CONTENT_TYPE_WIKI_DENSE:
        output_questions, extra_metadata = create_wiki_dense_distracting_questions(
            questions,
            distracting_source_dir,
            max_distracting_paths,
            graph_json,
            db_path_hint,
            questions_data.get(METADATA_FIELD, {}),
        )

    elif distracting_content_type == DISTRACTING_CONTENT_TYPE_WIKI_SPARSE:
        output_questions, extra_metadata = create_wiki_sparse_distracting_questions(
            questions,
            distracting_source_dir,
            max_distracting_paths,
            graph_json,
            db_path_hint,
            questions_data.get(METADATA_FIELD, {}),
        )

    elif distracting_content_type == DISTRACTING_CONTENT_TYPE_PAPER_HARD_DENSE:
        output_questions, extra_metadata = create_paper_hard_dense_distracting_questions(
            questions,
            graph_json,
            db_path_hint,
            distracting_source_dir,
            max_distracting_paths,
        )

    elif distracting_content_type == DISTRACTING_CONTENT_TYPE_PAPER_HARD_SPARSE:
        output_questions, extra_metadata = create_paper_hard_sparse_distracting_questions(
            questions,
            graph_json,
            db_path_hint,
            distracting_source_dir,
            max_distracting_paths,
        )

    elif distracting_content_type == DISTRACTING_CONTENT_TYPE_REPEAT_SENTENCE:
        output_questions, extra_metadata = create_repeat_sentence_distracting_questions(
            questions,
            repeat_sentence,
        )

    elif distracting_content_type == DISTRACTING_CONTENT_TYPE_REPEAT_SYMBOL:
        output_questions, extra_metadata = create_repeat_symbol_distracting_questions(
            questions,
            repeat_symbols,
        )

    elif distracting_content_type == DISTRACTING_CONTENT_TYPE_MIXED:
        output_questions, extra_metadata = create_mixed_distracting_questions(questions)

    else:
        raise ValueError(
            UNSUPPORTED_DISTRACTING_CONTENT_TYPE_ERROR.format(
                content_type=distracting_content_type,
            )
        )

    distracting_metadata.update(extra_metadata)

    output_data = {
        METADATA_FIELD: add_distracting_metadata(
            questions_path,
            distracting_metadata,
            len(output_questions),
        ),
        PARAMETERS_FIELD: merge_parameters(
            questions_data.get(PARAMETERS_FIELD),
            **{
                DISTRACTING_CONTENT_TYPE_PARAMETER_FIELD: distracting_content_type,
            },
        ),
        QUESTIONS_FIELD: output_questions,
    }
    return output_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--questions",
        required=True,
        metavar="JSON",
        help="Input create_needles JSON file.",
    )
    parser.add_argument(
        "--graph",
        required=False,
        nargs="+",
        metavar="JSON",
        help=(
            "Path to the graph JSON, a graphs directory, or an output root. "
            "When multiple files are given, the newest by modification time is used. "
            "Required when --distracting-content-type is "
            f"{DISTRACTING_CONTENT_TYPE_PAPER_RANDOM}, "
            f"{DISTRACTING_CONTENT_TYPE_PAPER_HARD_SPARSE}, or "
            f"{DISTRACTING_CONTENT_TYPE_PAPER_HARD_DENSE}."
        ),
    )
    parser.add_argument(
        "--db-path",
        default=None,
        metavar="DB",
        help="Optional explicit path to the SQLite DB.",
    )
    parser.add_argument(
        "--distracting-content-type",
        default=DISTRACTING_CONTENT_TYPE_PAPER_RANDOM,
        choices=list(SUPPORTED_DISTRACTING_CONTENT_TYPES),
        metavar="TYPE",
        help=(
            "Distracting content type. Each type uses its built-in distractor "
            "count. Choices: "
            f"{', '.join(SUPPORTED_DISTRACTING_CONTENT_TYPES)}."
        ),
    )
    parser.add_argument(
        "--distracting-source-dir",
        default=None,
        metavar="DIR",
        help=(
            "Optional source directory override for books, code, essays, "
            "Wikipedia, or hard-paper distractors. For Wikipedia and "
            "hard-paper indexes, pass either the root containing bm25/ and "
            "dense/ or one cache directory directly."
        ),
    )
    parser.add_argument(
        "--code-source-dir",
        default=str(DEFAULT_CODE_DISTRACTING_DIR),
        metavar="DIR",
        help=(
            "Code distractor source directory kept for compatibility "
            f"(default: {DEFAULT_CODE_DISTRACTING_DIR})."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        metavar="N",
        help="Optional seed for reproducible file-distractor sampling.",
    )
    parser.add_argument(
        "--repeat-sentence",
        default=DEFAULT_REPEAT_SENTENCE,
        help=(
            "Content stored in the single repeat item for repeat-sentence "
            "(default: %(default)r)."
        ),
    )
    parser.add_argument(
        "--repeat-symbols",
        default=DEFAULT_REPEAT_SYMBOLS,
        help=(
            "Content stored in the single repeat item for repeat-symbol " "(default: %(default)r)."
        ),
    )
    parser.add_argument(
        "--output",
        required=False,
        metavar="PATH",
        help="Path to write the output JSON file. Defaults to overwriting --questions.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation width (default: 2).",
    )
    return parser


def print_summary(out_path: Path, meta: dict[str, Any]) -> None:
    """Print a concise run summary."""
    distracting_meta = meta[DISTRACTING_CONTENT_FIELD]
    distracting_content_type = distracting_meta[DISTRACTING_CONTENT_TYPE_FIELD]
    print("Added distracting content for " f"{meta[TOTAL_QUESTIONS_FIELD]} question(s)")
    print(f"  Type   : {distracting_content_type}")
    print(f"  Max    : {distracting_meta[MAX_DISTRACTING_PATHS_FIELD]} items/question")
    if (
        distracting_content_type == DISTRACTING_CONTENT_TYPE_WIKI_DENSE
        or distracting_content_type == DISTRACTING_CONTENT_TYPE_WIKI_SPARSE
    ):
        print(f"  Source : {distracting_meta[SOURCE_DIR_FIELD]}")
        print(f"  Index  : {distracting_meta[INDEX_NAME_FIELD]}")
        if distracting_content_type == DISTRACTING_CONTENT_TYPE_WIKI_DENSE:
            print(f"  Encoder: {distracting_meta[ENCODER_NAME_FIELD]}")
            print(f"  Device : {distracting_meta[FAISS_DEVICE_FIELD]}")
    elif (
        distracting_content_type == DISTRACTING_CONTENT_TYPE_BOOKS
        or distracting_content_type == DISTRACTING_CONTENT_TYPE_CODE
        or distracting_content_type == DISTRACTING_CONTENT_TYPE_ESSAYS
    ):
        print(f"  Source : {distracting_meta[SOURCE_DIR_FIELD]}")
        if RANDOM_SEED_FIELD in distracting_meta:
            print(f"  Seed   : {distracting_meta[RANDOM_SEED_FIELD]}")
    elif (
        distracting_content_type == DISTRACTING_CONTENT_TYPE_PAPER_RANDOM
        or distracting_content_type == DISTRACTING_CONTENT_TYPE_PAPER_HARD_DENSE
        or distracting_content_type == DISTRACTING_CONTENT_TYPE_PAPER_HARD_SPARSE
    ):
        if SOURCE_DIR_FIELD in distracting_meta:
            print(f"  Source : {distracting_meta[SOURCE_DIR_FIELD]}")
        if CORPUS_PATH_FIELD in distracting_meta:
            print(f"  Corpus : {distracting_meta[CORPUS_PATH_FIELD]}")
        if ENCODER_NAME_FIELD in distracting_meta:
            print(f"  Encoder: {distracting_meta[ENCODER_NAME_FIELD]}")
        if FAISS_DEVICE_FIELD in distracting_meta:
            print(f"  Device : {distracting_meta[FAISS_DEVICE_FIELD]}")
    else:
        print(f"  Repeat : {distracting_meta[REPEAT_VALUE_FIELD]}")
    print(f"  Output : {out_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    graph_required = args.distracting_content_type in {
        DISTRACTING_CONTENT_TYPE_PAPER_RANDOM,
        DISTRACTING_CONTENT_TYPE_PAPER_HARD_DENSE,
        DISTRACTING_CONTENT_TYPE_PAPER_HARD_SPARSE,
    }
    if graph_required and not args.graph:
        parser.error(GRAPH_REQUIRED_ERROR.format(type=args.distracting_content_type))

    graph_json: Path | None = None
    if args.graph:
        graph_json = resolve_graph_argument(args.graph)
        if len(args.graph) > 1:
            print(
                f"[{CLI_NAME}] "
                f"{len(args.graph)} graph inputs matched; "
                f"using newest: {graph_json.name}"
            )

    result = create_distracting_content(
        questions_path=Path(args.questions),
        graph_json=graph_json,
        db_path_hint=args.db_path,
        distracting_content_type=args.distracting_content_type,
        distracting_source_dir=args.distracting_source_dir,
        code_source_dir=Path(args.code_source_dir),
        random_seed=args.random_seed,
        repeat_sentence=args.repeat_sentence,
        repeat_symbols=args.repeat_symbols,
    )

    out_path = Path(args.output) if args.output else Path(args.questions)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        orjson.dumps(result, option=orjson.OPT_INDENT_2).decode("utf-8"),
        encoding="utf-8",
    )

    print_summary(out_path, result[METADATA_FIELD])


if __name__ == "__main__":
    main()
