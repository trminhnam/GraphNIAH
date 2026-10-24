"""Shared constants and lightweight helpers used across entry-point scripts.

This module is intentionally import-light (only ``re`` and ``uuid`` from the
stdlib) so that ``visualize_network.py``, ``sample_paths.py``, and any other
consumer can import it without triggering the heavier ``create_graph`` module.
"""

from __future__ import annotations

import re
from uuid import UUID

# ─────────────────────────────────────────────────────────────────────────────
# Node / edge type constants
# ─────────────────────────────────────────────────────────────────────────────

NODE_PAPER = "paper"
NODE_AUTHOR = "author"

EDGE_AUTHORSHIP = "authorship"
EDGE_CITING = "citing"
EDGE_CITED_BY = "cited_by"

FUZZY_MATCHING_THRESHOLD = 0.9

REFERENCE_TITLE_EDGE_SEPARATORS = ".,;:"
REFERENCE_AUTHOR_EDGE_SEPARATORS = ",;:*•·†‡§¶-–—"
REFERENCE_TITLE_EDGE_QUOTES = "\"'`“”‘’´"
REFERENCE_TITLE_WRAPPER_PAIRS = (
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),
    ("‘", "’"),
    ("«", "»"),
    ("‹", "›"),
    ("(", ")"),
    ("[", "]"),
    ("{", "}"),
    ("<", ">"),
)

# ─────────────────────────────────────────────────────────────────────────────
# Common JSON field constants
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH_FIELD = "db_path"
DB_RESOLUTION_FIELD = "db_resolution"
EDGE_TYPE_FIELD = "edge_type"
GENERATED_AT_FIELD = "generated_at"
GRAPH_JSON_FIELD = "graph_json"
METADATA_FIELD = "metadata"
NODE_IDS_FIELD = "node_ids"
QUESTIONS_FIELD = "questions"
SOURCE_QUESTIONS_FIELD = "source_questions"
TEXT_PATH_FIELD = "text_path"
TOTAL_QUESTIONS_FIELD = "total_questions"

# ─────────────────────────────────────────────────────────────────────────────
# Distracting-content constants
# ─────────────────────────────────────────────────────────────────────────────

DISTRACTING_CONTENT_FIELD = "distracting_content"
DISTRACTING_CONTENT_TYPE_BOOKS = "books"
DISTRACTING_CONTENT_TYPE_CODE = "code"
DISTRACTING_CONTENT_TYPE_ESSAYS = "essays"
DISTRACTING_CONTENT_TYPE_MIXED = "mixed"
DISTRACTING_CONTENT_TYPE_NEEDLES_ONLY = "needles-only"
DISTRACTING_CONTENT_TYPE_PAPER_HARD_DENSE = "paper-hard-dense"
DISTRACTING_CONTENT_TYPE_PAPER_HARD_SPARSE = "paper-hard-sparse"
DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE = "paper-negative-tree"
DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V4 = "paper-negative-tree-v4"
DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V5 = "paper-negative-tree-v5"
DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V6 = "paper-negative-tree-v6"
DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V7 = "paper-negative-tree-v7"
DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V8 = "paper-negative-tree-v8"
DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V10 = "paper-negative-tree-v10"
DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V11 = "paper-negative-tree-v11"
DISTRACTING_CONTENT_TYPE_PAPER_NEGATIVE_TREE_V9 = "paper-negative-tree-v9"
DISTRACTING_CONTENT_TYPE_PAPER_RELATED = "paper-related"
DISTRACTING_CONTENT_TYPE_PAPER_RANDOM = "paper-random"
DISTRACTING_CONTENT_TYPE_REPEAT_SENTENCE = "repeat-sentence"
DISTRACTING_CONTENT_TYPE_REPEAT_NEEDLES = "repeat-needles"
DISTRACTING_CONTENT_TYPE_REPEAT_SYMBOL = "repeat-symbol"
DISTRACTING_CONTENT_TYPE_WIKI_DENSE = "wiki-dense"
DISTRACTING_CONTENT_TYPE_WIKI_SPARSE = "wiki-sparse"
DISTRACTING_CONTENT_TYPE_FIELD = "distracting_content_type"
DISTRACTING_CONTENT_PATHS_FIELD = "distracting_content_items"
PAPER_NEGATIVE_TREES_FIELD = "paper_negative_trees"
MAX_DISTRACTING_PATHS_FIELD = "max_distracting_paths"
RANDOM_SEED_FIELD = "random_seed"
REPEAT_VALUE_FIELD = "repeat_value"
SOURCE_DIR_FIELD = "source_dir"

PATH_ITEM_PREFIX = "path:"
PAPER_INDEX_ITEM_PREFIX = "paper_index:"
PAPER_NEGATIVE_TREE_ITEM_PREFIX = "paper_negative_tree:"
PAPER_RELATED_INDEX_ITEM_PREFIX = "paper_related_index:"
PAPER_RELATED_INDEX_LEVEL_ITEM_PREFIX = "paper_related_index_"
REPEAT_ITEM_PREFIX = "repeat:"
WIKI_BM25_ITEM_PREFIX = "wiki_bm25:"
WIKI_DENSE_ITEM_PREFIX = "wiki_dense:"

SUPPORTED_DISTRACTING_CONTENT_TYPES = (
    DISTRACTING_CONTENT_TYPE_BOOKS,
    DISTRACTING_CONTENT_TYPE_CODE,
    DISTRACTING_CONTENT_TYPE_ESSAYS,
    DISTRACTING_CONTENT_TYPE_NEEDLES_ONLY,
    DISTRACTING_CONTENT_TYPE_PAPER_RANDOM,
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
    DISTRACTING_CONTENT_TYPE_WIKI_DENSE,
    DISTRACTING_CONTENT_TYPE_WIKI_SPARSE,
    DISTRACTING_CONTENT_TYPE_PAPER_HARD_DENSE,
    DISTRACTING_CONTENT_TYPE_PAPER_HARD_SPARSE,
    DISTRACTING_CONTENT_TYPE_REPEAT_SENTENCE,
    DISTRACTING_CONTENT_TYPE_REPEAT_NEEDLES,
    DISTRACTING_CONTENT_TYPE_REPEAT_SYMBOL,
    DISTRACTING_CONTENT_TYPE_MIXED,
)
NOT_IMPLEMENTED_DISTRACTING_CONTENT_TYPES = frozenset(
    {
        DISTRACTING_CONTENT_TYPE_MIXED,
    }
)
FILE_DISTRACTING_CONTENT_TYPES = frozenset(
    {
        DISTRACTING_CONTENT_TYPE_BOOKS,
        DISTRACTING_CONTENT_TYPE_CODE,
        DISTRACTING_CONTENT_TYPE_ESSAYS,
    }
)
PAPER_RANDOM_DISTRACTING_CONTENT_TYPES = frozenset(
    {
        DISTRACTING_CONTENT_TYPE_PAPER_RANDOM,
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
    }
)

DEFAULT_MAX_DISTRACTING_PATHS = 384
PAPER_SORT_FALLBACK_DB_ID = 10**12

DEFAULT_BOOKS_DISTRACTING_DIR = "resources/distracting_content/books"
DEFAULT_CODE_DISTRACTING_DIR = "resources/distracting_content/code"
DEFAULT_ESSAYS_DISTRACTING_DIR = "resources/distracting_content/essays"
DEFAULT_PAPER_DISTRACTING_DIR = "resources/distracting_content/paper"
DEFAULT_WIKIPEDIA_DISTRACTING_DIR = "resources/distracting_content/wikipedia"
DEFAULT_REPEAT_SENTENCE = "The quick brown fox jumps over the lazy dog."
DEFAULT_REPEAT_SYMBOLS = "!@#$%^&*()[]{}<>?/|~"

TEXT_FILE_EXTS = frozenset({".txt", ".md"})
CODE_FILE_EXTS = frozenset({".c", ".h", ".cpp", ".hpp"})
DEFAULT_FILE_DISTRACTING_DIRS = {
    DISTRACTING_CONTENT_TYPE_BOOKS: DEFAULT_BOOKS_DISTRACTING_DIR,
    DISTRACTING_CONTENT_TYPE_CODE: DEFAULT_CODE_DISTRACTING_DIR,
    DISTRACTING_CONTENT_TYPE_ESSAYS: DEFAULT_ESSAYS_DISTRACTING_DIR,
}
FILE_DISTRACTING_EXTS = {
    DISTRACTING_CONTENT_TYPE_BOOKS: TEXT_FILE_EXTS,
    DISTRACTING_CONTENT_TYPE_CODE: CODE_FILE_EXTS,
    DISTRACTING_CONTENT_TYPE_ESSAYS: TEXT_FILE_EXTS,
}
DISTRACTING_RELATED_EDGE_TYPES = frozenset(
    {EDGE_CITING, EDGE_CITED_BY, EDGE_AUTHORSHIP}
)


# ─────────────────────────────────────────────────────────────────────────────
# Node-ID helpers
# ─────────────────────────────────────────────────────────────────────────────


def paper_nid(db_id: str | int) -> str:
    """Return the graph node ID for a paper with the given database ID."""
    return f"paper:{str(db_id)}"


def author_nid(db_id: str | int) -> str:
    """Return the graph node ID for an author with the given database ID."""
    return f"author:{str(db_id)}"


def nid_type(nid: str) -> str:
    """Return 'paper' or 'author' from a node ID string like 'paper:<uuid>'."""
    return nid.split(":")[0]


def nid_db_id(nid: str) -> str:
    """Return the DB id from a node ID string like 'paper:<uuid>'."""
    return nid.split(":", 1)[1]


def is_uuid_text(value: object) -> bool:
    """Return whether *value* is a valid UUID string."""
    text = str(value or "").strip()
    if not text:
        return False
    try:
        UUID(text)
    except (TypeError, ValueError, AttributeError):
        return False
    return True


def local_node_id_uses_uuid_db_id(node_id: object) -> bool:
    """Return whether *node_id* uses a UUID-backed local DB identifier."""
    if not isinstance(node_id, str) or ":" not in node_id:
        return False
    node_type = nid_type(node_id)
    if node_type not in {NODE_PAPER, NODE_AUTHOR}:
        return False
    try:
        db_id = nid_db_id(node_id)
    except (IndexError, ValueError):
        return False
    return is_uuid_text(db_id)


# ─────────────────────────────────────────────────────────────────────────────
# String normalisation
# ─────────────────────────────────────────────────────────────────────────────


def preprocess_paper_title(title: str) -> str:
    """Collapse embedded whitespace and drop any trailing period characters."""
    title = re.sub(r"\s+", " ", title).strip()
    if not title:
        return ""
    return title.rstrip(".").rstrip()


def preprocess_reference_title(title: str) -> str:
    """Trim edge punctuation commonly added by reference formatting or VLM parsing."""
    title = preprocess_paper_title(title)
    if not title:
        return ""

    previous = None
    while title and title != previous:
        previous = title
        title = title.strip()
        title = (
            title.lstrip(REFERENCE_TITLE_EDGE_SEPARATORS)
            .rstrip(REFERENCE_TITLE_EDGE_SEPARATORS)
            .strip()
        )
        title = title.strip(REFERENCE_TITLE_EDGE_QUOTES).strip()

        for opener, closer in REFERENCE_TITLE_WRAPPER_PAIRS:
            if title.startswith(opener) and title.endswith(closer):
                title = title[len(opener) : len(title) - len(closer)].strip()
                break

    return title


def preprocess_reference_author(author: str) -> str:
    """Trim wrapper punctuation around a parsed reference-author string."""
    author = re.sub(r"\s+", " ", author).strip()
    if not author:
        return ""

    previous = None
    while author and author != previous:
        previous = author
        author = re.sub(r"\s+", " ", author).strip()
        author = (
            author.lstrip(REFERENCE_AUTHOR_EDGE_SEPARATORS)
            .rstrip(REFERENCE_AUTHOR_EDGE_SEPARATORS)
            .strip()
        )
        author = author.strip(REFERENCE_TITLE_EDGE_QUOTES).strip()

        for opener, closer in REFERENCE_TITLE_WRAPPER_PAIRS:
            if author.startswith(opener) and author.endswith(closer):
                author = author[len(opener) : len(author) - len(closer)].strip()
                break

    return re.sub(r"\s+", " ", author).strip()


def preprocess_reference_authors(authors: list[str] | tuple[str, ...]) -> list[str]:
    """Normalize a parsed reference-author list and drop empty items."""
    cleaned_authors: list[str] = []
    for author in authors:
        cleaned_author = preprocess_reference_author(author)
        if cleaned_author:
            cleaned_authors.append(cleaned_author)
    return cleaned_authors


def normalize_text(text: str, max_len: int = 120) -> str:
    """Return a canonical form used for both index keys and node IDs.

    Steps:
      1. Lowercase and strip leading/trailing whitespace.
      2. Collapse all internal whitespace runs to a single space.
      3. Remove every character that is not alphanumeric, a hyphen, or a space.
      4. Replace spaces with underscores (filesystem/graph safe).
      5. Truncate to *max_len*.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\- ]", "", text)
    text = text.replace(" ", "_")
    return text[:max_len]


def normalize_text_aggressive(text: str, max_len: int = 120) -> str:
    """Return a looser canonical form for fuzzy alias matching.

    On top of ``normalize_text``, also strips common leading articles
    (a / an / the) and all hyphens, to catch title variants like
    "A Survey of …" vs "Survey of …" or "Self-Attention" vs "Self Attention".
    """
    text = normalize_text(text, max_len=max_len * 2)
    text = re.sub(r"^(a|an|the)_", "", text)
    text = text.replace("-", "")
    return text[:max_len]


def slugify(text: str, max_len: int = 120) -> str:
    """Return a filesystem-safe slug — used only for filenames, NOT for node IDs."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", text).strip("_")[:max_len]
