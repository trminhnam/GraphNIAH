#!/usr/bin/env python3
"""
create_needles.py — Build LLM needles from a question's paper chain
===================================================================

For every question in the input JSON this script:

1. Walks the question's ``node_ids`` list in order.
2. For each ``paper:<id>`` node, reads the paper from the SQLite DB and
   loads the text file stored at ``text_path``.  Author nodes are
   skipped — they do not carry document content.
3. Preserves the question path order when assembling paper texts, so the
   anchor-side paper appears first and the answer-side paper appears last.
4. For counting questions, optionally realizes an explicit
   ``counting_answer``. When ``counting_answer`` is omitted, the script
   samples the answer uniformly between ``path_support_count + 1`` and
   the number of candidate papers in the graph that can contribute
   needles under the selected mode. It then adds exactly that many
   support-paper needles to the context and uses the selected support
   set as the final numeric answer. Questions whose feasible range is
   empty are skipped.
5. Stores those texts according to ``needle_type``:
   - ``raw``: cleaned paper text strings with appendix sections removed
   - ``exact-describe``: exact per-paper metadata plus exact citation jumps
   - ``general-describe``: Semantic Scholar per-paper metadata plus reference lists
   - ``apa-describe``: structured per-paper metadata plus APA-style references

Needle Types
------------

``raw``
  Each paper node contributes one item: the cleaned paper text loaded from
  ``text_path``.  The final ``needles`` field is a ``list[str]`` in forward
  path order, so the anchor-side paper appears first.

``exact-describe``
  Each paper node contributes a metadata sentence:

    ``Paper <title>, written by <authors>, was published in <venue> on <published_date>.``

  Each paper-to-paper citation hop also contributes one exact relation
  sentence with inline metadata for the target paper when available:

  - ``citing``:
    ``Paper Pa references Paper Pb, which was written by <authors> and published in <venue> on <published_date>.``
  - ``cited_by``:
    ``Paper Pa is cited by Paper Pb, which was written by <authors> and published in <venue> on <published_date>.``

  If venue or date metadata is missing, the publication phrase is shortened.

``general-describe``
  Each paper node contributes a metadata sentence and a reference-list summary
  fetched from Semantic Scholar:

  ``Paper Pa, written by <authors>, was published in <venue> on <published_date>.``
  ``Paper Pa references the following papers:``
  ``- Paper Pb, written by <authors> and published in <venue> on <published_date>.``
  ``- Paper Pc, written by <authors> and published on <published_date>.``
  ``- ...``

``apa-describe``
  Each paper node contributes one structured block:

    ``Title: <title>.``
    ``Authors: <authors>.``
    ``Published in <venue> on <published_date>.``
    ``References:``
    ``- <APA-style reference>``

  arXiv-only papers use ``Uploaded to arXiv on <published_date>.`` instead of
  the venue publication line. Missing authors, publication venue, arXiv ID, or
  date fields are omitted or shortened. The ``Authors:`` line keeps full names;
  reference authors use the APA 7 list form (``Soto, C. J., & John, O. P.``).

Notes
-----

- Author nodes never contribute raw or descriptive document strings.
- In descriptive modes, a paper node may contribute one structured block or
  two strings: one metadata string and one citation/list string.
- Count questions may insert extra support-paper needles immediately before
  the target paper when the target is a paper, or after the path papers when
  the target is an author.
- ``exact-describe`` requires ``question["footprint"]["edge_types"]`` so the
  script can recover the forward path semantics.

The output file is a dictionary with a ``metadata`` block and a
``questions`` list.  Each question entry keeps only the fields needed
for downstream LLM evaluation: ``question``, ``needles``, ``answer``,
``answer_id``, ``answer_type``, ``node_ids``, and for counting questions
``counting_support_node_ids``.  The concrete answer is resolved here from
the generator's ``answer_type`` + ``answer_id`` pair.

Usage
-----

python create_needles.py \
    --questions questions_1000.json \
    --graph output/graphs/logitscope/graphs/LogitScope_*.json \
    --needle-type exact-describe \
    --counting-answer 3 \
    --output logitscope_qa_1000.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import orjson
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from src.clients.semantic_scholar_client import SemanticScholarClient
from src.graph.readers import CitationGraphLoader, DBReader
from src.graph.resolution import resolve_graph_argument, resolve_graph_db_paths
from src.setup.pipeline_parameters import (
    NEEDLE_TYPE_PARAMETER_FIELD,
    PARAMETERS_FIELD,
    merge_parameters,
)
from src.questions.shared import COUNT_AUTHOR_PAPERS, COUNT_PAPER_CITATIONS
from src.questions.text import quoted_title
from src.setup.constants import (
    EDGE_AUTHORSHIP,
    EDGE_CITED_BY,
    EDGE_CITING,
    NODE_PAPER,
    nid_db_id,
    nid_type,
)
from src.utils.string_matching import compact_alnum_string
from src.utils.preprocessing.clean_document import clean_document


APPENDIX_SECTION_HEADING_PATTERN = re.compile(
    r"^[ \t]*#{1,6}[ \t]*appendi(?:x|ces)\b[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)

SEMANTIC_SCHOLAR_PAPER_FIELDS = [
    "paperId",
    "externalIds",
    "title",
    "authors",
    "year",
    "publicationDate",
    "venue",
]
SEMANTIC_SCHOLAR_REFERENCE_FIELDS = [
    f"citedPaper.{field}" for field in SEMANTIC_SCHOLAR_PAPER_FIELDS
]
COUNTING_PATH_SUPPORT_LOWER_BOUND_PATH_LENGTH = 16

# APA 7 reference-list author names. Only lowercase particles join the surname:
# a capitalized "Van" or "De" is often a middle name (e.g. "Tuan Van Nguyen").
APA_SURNAME_PARTICLES = frozenset(
    {"da", "das", "de", "del", "della", "den", "der", "di", "dos", "du",
     "la", "le", "ten", "ter", "van", "von", "zu"}
)
APA_NAME_SUFFIXES = frozenset({"jr", "jr.", "sr", "sr.", "ii", "iii", "iv"})
APA_MAX_LISTED_AUTHORS = 20
# Convertible names hold only letters (with accents), spaces, and this
# punctuation. Other names (digits, parentheses, commas) are left as stored, and
# so are names not starting with an uppercase letter (group names like "the X
# team", CJK-script names).
APA_NAME_PUNCTUATION = frozenset(" .'’-‐‑")
APA_HYPHEN_PATTERN = re.compile(r"[-‐‑]")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_paper_text(text_path: str) -> str | None:
    """Return the contents of the paper's text file, or None if missing.

    ``text_path`` is stored in the DB as a project-root-relative path
    (e.g. ``output/graphs/logitscope/texts/2603_24929.md``).  Empty
    strings and missing files both return None so the caller can simply
    skip papers with no extracted text.
    """
    if not text_path:
        return None
    path = Path(text_path)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def remove_appendix_section(text: str) -> str:
    """Drop everything from the first markdown appendix heading onward."""
    match = APPENDIX_SECTION_HEADING_PATTERN.search(text)
    if match is None:
        return text.strip()
    return text[: match.start()].rstrip()


def preprocess_raw_paper_text(text: str | None) -> str | None:
    """Clean raw paper text and remove trailing appendix sections."""
    if text is None:
        return None
    cleaned_text = clean_document(text).strip()
    if not cleaned_text:
        return None
    cleaned_text = remove_appendix_section(cleaned_text).strip()
    return cleaned_text or None


def collect_paper_texts(node_ids: list[str], db: DBReader) -> list[str]:
    """Walk *node_ids* in order, returning the text of every paper node.

    Author nodes are ignored.  Paper nodes whose ``text_path`` is empty
    or points at a missing file are also ignored — they contribute
    nothing to the needles.
    """
    texts: list[str] = []
    for nid in node_ids:
        if nid_type(nid) != "paper":
            continue
        paper = db.get_paper(nid_db_id(nid))
        if paper is None:
            continue
        text = preprocess_raw_paper_text(load_paper_text(paper["text_path"]))
        if text is not None:
            texts.append(text)
    return texts


def build_raw_needles(paper_texts: list[str]) -> list[str]:
    """Return cleaned raw paper texts in forward traversal order."""
    return list(paper_texts)


def flatten_needle_groups(groups: list[list[str]]) -> list[str]:
    """Flatten grouped needle chunks into one list."""
    needles: list[str] = []
    for group in groups:
        needles.extend(group)
    return needles


def stable_reference_seed(*parts: Any) -> int:
    """Return a deterministic seed for reference-list ordering."""
    payload = orjson.dumps(parts, option=orjson.OPT_SORT_KEYS, default=str).decode("utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def reference_item_key(item: Any) -> Any:
    """Return a compact stable identity for a reference item."""
    if not isinstance(item, dict):
        return item
    external_ids = item.get("externalIds") or {}
    return {
        "paperId": item.get("paperId"),
        "title": item.get("title"),
        "externalIds": external_ids if isinstance(external_ids, dict) else {},
    }


def shuffled_reference_items(items: list[Any], *, owner_key: Any) -> list[Any]:
    """Return reference items in deterministic shuffled order."""
    shuffled_items = list(items)
    if len(shuffled_items) <= 1:
        return shuffled_items

    item_keys = [reference_item_key(item) for item in shuffled_items]
    rng = random.Random(stable_reference_seed("reference-order", owner_key, item_keys))
    rng.shuffle(shuffled_items)
    if shuffled_items == list(items):
        shuffled_items = shuffled_items[1:] + shuffled_items[:1]
    return shuffled_items


def get_question_edge_types(question: dict, node_ids: list[str]) -> list[str]:
    """Return the stored forward edge types for a generated question."""
    footprint = question.get("footprint") or {}
    edge_types = footprint.get("edge_types")
    if edge_types is None:
        if len(node_ids) <= 1:
            return []
        raise ValueError(
            "Descriptive needle types require question['footprint']['edge_types']."
        )

    expected = max(len(node_ids) - 1, 0)
    if len(edge_types) != expected:
        raise ValueError(
            "Question footprint edge_types length does not match node_ids length."
        )
    return [str(edge_type) for edge_type in edge_types]


def get_cached_paper(
    node_id: str,
    db: DBReader,
    paper_cache: dict[str, dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Return the cached paper row for *node_id*."""
    if node_id not in paper_cache:
        paper_cache[node_id] = db.get_paper(nid_db_id(node_id))
    return paper_cache[node_id]


def get_cached_paper_text(
    node_id: str,
    db: DBReader,
    paper_cache: dict[str, dict[str, Any] | None],
    paper_text_cache: dict[str, str | None],
) -> str | None:
    """Return the cached cleaned raw text for *node_id* if available."""
    if node_id not in paper_text_cache:
        paper = get_cached_paper(node_id, db, paper_cache)
        if paper is None:
            paper_text_cache[node_id] = None
        else:
            paper_text_cache[node_id] = preprocess_raw_paper_text(
                load_paper_text(str(paper.get("text_path") or ""))
            )
    return paper_text_cache[node_id]


def get_cached_authors(
    node_id: str,
    db: DBReader,
    authors_cache: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return the cached expanded authors for *node_id*."""
    if node_id not in authors_cache:
        authors_cache[node_id] = db.get_expanded_authors_for_paper(nid_db_id(node_id))
    return authors_cache[node_id]


def format_authors_text(author_names: list[str]) -> str:
    """Render author names as a comma-separated sequence in DB order."""
    names = [name.strip() for name in author_names if name and name.strip()]
    if not names:
        return ""
    return ", ".join(names)


def format_apa_initials(given_names: str) -> str:
    """Return APA initials, e.g. ``Dong-Wei`` -> ``D.-W.``, ``M.S.`` -> ``M. S.``."""
    initials: list[str] = []
    for token in unicodedata.normalize("NFC", given_names).split():
        hyphen_parts = []
        for part in APA_HYPHEN_PATTERN.split(token):
            letters = []
            for piece in part.split("."):
                letter = next((char for char in piece if char.isalpha()), "")
                if letter:
                    letters.append(f"{letter.upper()}.")
            if letters:
                hyphen_parts.append(" ".join(letters))
        if hyphen_parts:
            initials.append("-".join(hyphen_parts))
    return " ".join(initials)


def format_apa_author_name(name: str) -> str:
    """Return one author as ``Surname, I. I.``; keep names that cannot be split."""
    text = " ".join(str(name or "").split())
    tokens = text.split()
    if (
        len(tokens) < 2
        or not text[0].isupper()
        or not all(
            char.isalpha()
            or char in APA_NAME_PUNCTUATION
            or unicodedata.category(char).startswith("M")
            for char in text
        )
    ):
        return text
    suffix = ""
    if len(tokens) > 2 and tokens[-1].lower() in APA_NAME_SUFFIXES:
        suffix = tokens.pop()
    surname_start = len(tokens) - 1
    while surname_start > 1 and tokens[surname_start - 1] in APA_SURNAME_PARTICLES:
        surname_start -= 1
    initials = format_apa_initials(" ".join(tokens[:surname_start]))
    if not initials:
        return text
    apa_name = f"{' '.join(tokens[surname_start:])}, {initials}"
    return f"{apa_name}, {suffix}" if suffix else apa_name


def format_apa_authors_text(author_names: list[str]) -> str:
    """Render authors as an APA 7 reference list, keeping DB order.

    Two to twenty authors put ``&`` before the last one. More than twenty
    list the first nineteen, an ellipsis, then the last author (no ``&``).
    """
    names = [
        format_apa_author_name(name)
        for name in author_names
        if name and name.strip()
    ]
    if len(names) <= 1:
        return "".join(names)
    if len(names) > APA_MAX_LISTED_AUTHORS:
        listed = ", ".join(names[: APA_MAX_LISTED_AUTHORS - 1])
        return f"{listed}, . . . {names[-1]}"
    return ", ".join(names[:-1]) + f", & {names[-1]}"


def format_published_date_text(raw_published: str) -> str:
    """Normalize a stored published value to a readable date string."""
    from datetime import datetime

    text = str(raw_published or "").strip()
    if not text:
        return ""
    try:
        parsed_date = datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        return f"{parsed_date.strftime('%B')} {parsed_date.day}, {parsed_date.year}"
    except ValueError:
        pass
    if len(text) >= 10 and text[4:5] == "-" and text[7:8] == "-":
        try:
            parsed_date = datetime.fromisoformat(text[:10]).date()
            return f"{parsed_date.strftime('%B')} {parsed_date.day}, {parsed_date.year}"
        except ValueError:
            return text[:10]
    return text


def normalize_title_whitespace(title: str) -> str:
    """Collapse embedded whitespace (including line breaks) in a paper title."""
    return re.sub(r"\s+", " ", title).strip()


def get_paper_title_text(
    node_id: str,
    db: DBReader,
    paper_cache: dict[str, dict[str, Any] | None],
) -> str:
    """Return a display title for *node_id*."""
    paper = get_cached_paper(node_id, db, paper_cache)
    if paper is None:
        return node_id
    title = normalize_title_whitespace(str(paper.get("title") or ""))
    return title or node_id


def get_required_paper_title_text(
    node_id: str,
    db: DBReader,
    paper_cache: dict[str, dict[str, Any] | None],
) -> str:
    """Return a referenced paper title or reject missing title metadata."""
    paper = get_cached_paper(node_id, db, paper_cache) or {}
    title = normalize_title_whitespace(str(paper.get("title") or ""))
    if not title:
        raise ValueError(f"Referenced paper {node_id!r} has no title.")
    return title


def validate_rendered_reference_title(
    node_id: str,
    title: str,
    rendered_fragment: str,
) -> None:
    """Require one citation fragment to contain its referenced paper title."""
    if title in rendered_fragment:
        return
    normalized_title = compact_alnum_string(title)
    if normalized_title and normalized_title in compact_alnum_string(
        rendered_fragment
    ):
        return
    raise ValueError(
        f"Rendered reference for paper {node_id!r} does not contain its title "
        f"{title!r}."
    )


def get_paper_metadata_texts(
    node_id: str,
    db: DBReader,
    paper_cache: dict[str, dict[str, Any] | None],
    authors_cache: dict[str, list[dict[str, Any]]],
) -> tuple[str, str]:
    """Return normalized author and published strings for one paper node."""
    author_names = [
        author.get("name", "")
        for author in get_cached_authors(node_id, db, authors_cache)
    ]
    authors_text = format_authors_text([str(name) for name in author_names])
    paper = get_cached_paper(node_id, db, paper_cache) or {}
    published_text = format_published_date_text(str(paper.get("published") or ""))
    return authors_text, published_text


def get_paper_venue_text(
    node_id: str,
    db: DBReader,
    paper_cache: dict[str, dict[str, Any] | None],
) -> str:
    """Return a display venue for *node_id*, if one is stored."""
    paper = get_cached_paper(node_id, db, paper_cache) or {}
    return str(paper.get("venue") or "").strip()


def build_publication_text(venue_text: str, published_text: str) -> str:
    """Return a publication phrase from optional venue/date metadata."""
    if venue_text and published_text:
        return f"published in {venue_text} on {published_text}"
    if venue_text:
        return f"published in {venue_text}"
    if published_text:
        return f"published on {published_text}"
    return ""


def build_paper_metadata_clause(
    node_id: str,
    db: DBReader,
    paper_cache: dict[str, dict[str, Any] | None],
    authors_cache: dict[str, list[dict[str, Any]]],
    *,
    include_venue: bool = False,
) -> str:
    """Return an inline metadata clause for one paper node."""
    authors_text, published_text = get_paper_metadata_texts(
        node_id,
        db,
        paper_cache,
        authors_cache,
    )
    venue_text = (
        get_paper_venue_text(node_id, db, paper_cache) if include_venue else ""
    )
    publication_text = build_publication_text(venue_text, published_text)
    if authors_text and publication_text:
        return f"written by {authors_text} and {publication_text}"
    if authors_text:
        return f"written by {authors_text}"
    return publication_text


def build_metadata_needle(
    node_id: str,
    db: DBReader,
    paper_cache: dict[str, dict[str, Any] | None],
    authors_cache: dict[str, list[dict[str, Any]]],
    *,
    include_venue: bool = False,
) -> str:
    """Return the metadata description sentence for one paper node."""
    title = quoted_title(get_paper_title_text(node_id, db, paper_cache))
    authors_text, published_text = get_paper_metadata_texts(
        node_id,
        db,
        paper_cache,
        authors_cache,
    )
    venue_text = (
        get_paper_venue_text(node_id, db, paper_cache) if include_venue else ""
    )
    publication_text = build_publication_text(venue_text, published_text)

    if authors_text and publication_text:
        return f"Paper {title}, written by {authors_text}, was {publication_text}."
    if authors_text:
        return f"Paper {title}, written by {authors_text}."
    if publication_text:
        return f"Paper {title} was {publication_text}."
    return f"Paper {title}."


def build_exact_jump_needle(
    source_node_id: str,
    target_node_id: str,
    edge_type: str,
    db: DBReader,
    paper_cache: dict[str, dict[str, Any] | None],
    authors_cache: dict[str, list[dict[str, Any]]],
) -> str | None:
    """Return the exact citation-jump sentence for one paper→paper hop."""
    source_title_text = get_required_paper_title_text(
        source_node_id,
        db,
        paper_cache,
    )
    target_title_text = get_required_paper_title_text(
        target_node_id,
        db,
        paper_cache,
    )
    source_title = quoted_title(source_title_text)
    target_title = quoted_title(target_title_text)
    target_clause = build_paper_metadata_clause(
        target_node_id,
        db,
        paper_cache,
        authors_cache,
        include_venue=True,
    )

    if edge_type == EDGE_CITING:
        sentence = f"Paper {source_title} references Paper {target_title}"
        if target_clause:
            rendered_fragment = f"{sentence}, which was {target_clause}."
        else:
            rendered_fragment = f"{sentence}."
    elif edge_type == EDGE_CITED_BY:
        sentence = f"Paper {source_title} is cited by Paper {target_title}"
        if target_clause:
            rendered_fragment = f"{sentence}, which was {target_clause}."
        else:
            rendered_fragment = f"{sentence}."
    else:
        return None
    validate_rendered_reference_title(
        source_node_id,
        source_title_text,
        rendered_fragment,
    )
    validate_rendered_reference_title(
        target_node_id,
        target_title_text,
        rendered_fragment,
    )
    return rendered_fragment


def collect_cited_paper_ids(
    citing_node_id: str,
    graph: CitationGraphLoader,
) -> list[str]:
    """Return paper ids of every paper cited by *citing_node_id*."""
    cited_ids: list[str] = []
    seen: set[str] = set()
    for neighbour_id, edge_attrs in graph.adj.get(citing_node_id, []):
        if edge_attrs.get("edge_type") != EDGE_CITING:
            continue
        if nid_type(neighbour_id) != NODE_PAPER:
            continue
        if neighbour_id in seen:
            continue
        seen.add(neighbour_id)
        cited_ids.append(neighbour_id)
    return cited_ids


def build_general_jump_needle(
    source_node_id: str,
    target_node_id: str,
    edge_type: str,
    graph: CitationGraphLoader,
    db: DBReader,
    paper_cache: dict[str, dict[str, Any] | None],
    authors_cache: dict[str, list[dict[str, Any]]],
    *,
    include_venue: bool = False,
) -> str | None:
    """Return the expanded citation-list sentence for one paper→paper hop."""
    if edge_type == EDGE_CITING:
        citing_node_id = source_node_id
    elif edge_type == EDGE_CITED_BY:
        citing_node_id = target_node_id
    else:
        return None

    citing_title = quoted_title(get_paper_title_text(citing_node_id, db, paper_cache))
    cited_node_ids = collect_cited_paper_ids(citing_node_id, graph)
    if not cited_node_ids:
        return f"Paper {citing_title} references the following papers:"

    shuffled_cited_node_ids = shuffled_reference_items(
        cited_node_ids,
        owner_key=("general", citing_node_id),
    )
    bullets = "\n".join(
        build_general_reference_bullet(
            node_id,
            db,
            paper_cache,
            authors_cache,
            include_venue=include_venue,
        )
        for node_id in shuffled_cited_node_ids
    )
    return f"Paper {citing_title} references the following papers:\n{bullets}"


def build_general_reference_bullet(
    node_id: str,
    db: DBReader,
    paper_cache: dict[str, dict[str, Any] | None],
    authors_cache: dict[str, list[dict[str, Any]]],
    *,
    include_venue: bool = False,
) -> str:
    """Return one descriptive bullet for a referenced paper."""
    title_text = get_required_paper_title_text(node_id, db, paper_cache)
    title = quoted_title(title_text)
    clause = build_paper_metadata_clause(
        node_id,
        db,
        paper_cache,
        authors_cache,
        include_venue=include_venue,
    )
    rendered_fragment = (
        f"- Paper {title}, {clause}." if clause else f"- Paper {title}."
    )
    validate_rendered_reference_title(node_id, title_text, rendered_fragment)
    return rendered_fragment


def ensure_sentence_text(text: str) -> str:
    """Return text with sentence-ending punctuation when possible."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    if cleaned[-1] in ".?!":
        return cleaned
    return f"{cleaned}."


def extract_year_text(raw_published: str) -> str:
    """Return the first 4-digit year from a publication value."""
    match = re.search(r"\b(\d{4})\b", str(raw_published or ""))
    return match.group(1) if match else ""


def paper_is_arxiv_publication(venue_text: str, arxiv_id: str) -> bool:
    """Return whether publication metadata should be rendered as arXiv."""
    return (
        bool(str(arxiv_id or "").strip())
        or str(venue_text or "").strip().lower() == "arxiv"
    )


def build_apa_publication_line(
    venue_text: str,
    published_text: str,
    arxiv_id: str,
) -> str:
    """Return the structured APA-block publication line."""
    venue = str(venue_text or "").strip()
    published = str(published_text or "").strip()
    arxiv = str(arxiv_id or "").strip()
    venue_is_arxiv = venue.lower() == "arxiv"

    if venue and not venue_is_arxiv:
        if published:
            return f"Published in {venue} on {published}."
        return f"Published in {venue}."
    if paper_is_arxiv_publication(venue, arxiv):
        if published:
            return f"Uploaded to arXiv on {published}."
        return "Uploaded to arXiv."
    if published:
        return f"Published on {published}."
    return ""


def build_apa_source_text(venue_text: str, arxiv_id: str) -> str:
    """Return the APA reference source from venue/arXiv metadata."""
    venue = str(venue_text or "").strip()
    arxiv = str(arxiv_id or "").strip()
    if venue and venue.lower() != "arxiv":
        return venue
    if arxiv:
        return f"arXiv:{arxiv}"
    if venue.lower() == "arxiv":
        return "arXiv"
    return ""


def build_apa_reference_text(
    title_text: str,
    authors_text: str,
    raw_published: str,
    venue_text: str,
    arxiv_id: str,
) -> str:
    """Return one compact APA-style reference with missing fields omitted."""
    title = str(title_text or "").strip() or "Unknown title"
    authors = str(authors_text or "").strip()
    year = extract_year_text(raw_published)
    source = build_apa_source_text(venue_text, arxiv_id)

    if authors:
        lead = f"{authors} ({year})." if year else ensure_sentence_text(authors)
        parts = [lead, ensure_sentence_text(title)]
    elif year:
        parts = [ensure_sentence_text(f"{title} ({year})")]
    else:
        parts = [ensure_sentence_text(title)]

    if source:
        parts.append(ensure_sentence_text(source))
    return " ".join(part for part in parts if part)


def build_apa_reference_bullet_from_values(
    title_text: str,
    authors_text: str,
    raw_published: str,
    venue_text: str,
    arxiv_id: str,
) -> str:
    """Return one APA-style reference bullet."""
    reference = build_apa_reference_text(
        title_text,
        authors_text,
        raw_published,
        venue_text,
        arxiv_id,
    )
    return f"- {reference}"


def build_apa_block(
    title_text: str,
    authors_text: str,
    raw_published: str,
    venue_text: str,
    arxiv_id: str,
    reference_bullets: list[str],
) -> str:
    """Return the full structured APA needle block for one paper."""
    title = str(title_text or "").strip() or "Unknown title"
    authors = str(authors_text or "").strip()
    published = format_published_date_text(str(raw_published or ""))
    lines = [f"Title: {ensure_sentence_text(title)}"]
    if authors:
        lines.append(f"Authors: {ensure_sentence_text(authors)}")

    publication_line = build_apa_publication_line(venue_text, published, arxiv_id)
    if publication_line:
        lines.append(publication_line)

    lines.append("References:")
    lines.extend(reference_bullets)
    return "\n".join(lines)


def get_paper_author_names(
    node_id: str,
    db: DBReader,
    authors_cache: dict[str, list[dict[str, Any]]],
) -> list[str]:
    """Return cached author names for one paper node."""
    return [
        str(author.get("name") or "")
        for author in get_cached_authors(node_id, db, authors_cache)
    ]


def build_apa_reference_bullet(
    node_id: str,
    db: DBReader,
    paper_cache: dict[str, dict[str, Any] | None],
    authors_cache: dict[str, list[dict[str, Any]]],
) -> str:
    """Return one APA-style reference bullet for a DB-backed paper."""
    paper = get_cached_paper(node_id, db, paper_cache) or {}
    title = get_required_paper_title_text(node_id, db, paper_cache)
    authors_text = format_apa_authors_text(
        get_paper_author_names(node_id, db, authors_cache)
    )
    rendered_fragment = build_apa_reference_bullet_from_values(
        title,
        authors_text,
        str(paper.get("published") or ""),
        str(paper.get("venue") or ""),
        str(paper.get("arxiv_id") or ""),
    )
    validate_rendered_reference_title(node_id, title, rendered_fragment)
    return rendered_fragment


def build_apa_paper_needle(
    node_id: str,
    cited_node_ids: list[str],
    db: DBReader,
    paper_cache: dict[str, dict[str, Any] | None],
    authors_cache: dict[str, list[dict[str, Any]]],
    *,
    shuffle_references: bool = True,
) -> str:
    """Return one structured APA needle for a DB-backed paper."""
    paper = get_cached_paper(node_id, db, paper_cache) or {}
    authors_text = format_authors_text(
        get_paper_author_names(node_id, db, authors_cache)
    )
    reference_node_ids = (
        shuffled_reference_items(cited_node_ids, owner_key=("apa", node_id))
        if shuffle_references
        else list(cited_node_ids)
    )
    reference_bullets = [
        build_apa_reference_bullet(
            cited_node_id,
            db,
            paper_cache,
            authors_cache,
        )
        for cited_node_id in reference_node_ids
    ]
    return build_apa_block(
        get_paper_title_text(node_id, db, paper_cache),
        authors_text,
        str(paper.get("published") or ""),
        str(paper.get("venue") or ""),
        str(paper.get("arxiv_id") or ""),
        reference_bullets,
    )


def build_apa_paper_groups(
    node_ids: list[str],
    graph: CitationGraphLoader,
    db: DBReader,
    paper_cache: dict[str, dict[str, Any] | None],
    authors_cache: dict[str, list[dict[str, Any]]],
) -> list[list[str]]:
    """Return one APA block per paper node using graph reference edges."""
    paper_groups: list[list[str]] = []
    for node_id in node_ids:
        if nid_type(node_id) != NODE_PAPER:
            continue
        paper_groups.append(
            [
                build_apa_paper_needle(
                    node_id,
                    collect_cited_paper_ids(node_id, graph),
                    db,
                    paper_cache,
                    authors_cache,
                )
            ]
        )
    return paper_groups


def get_semantic_scholar_paper_lookup_id(
    node_id: str,
    db: DBReader,
    paper_cache: dict[str, dict[str, Any] | None],
) -> str:
    """Return the best Semantic Scholar-compatible id for one paper node."""
    paper = get_cached_paper(node_id, db, paper_cache)
    if paper is None:
        return ""

    semantic_scholar_id = str(paper.get("semantic_scholar_id") or "").strip()
    if semantic_scholar_id:
        return semantic_scholar_id

    arxiv_id = str(paper.get("arxiv_id") or "").strip()
    if arxiv_id:
        return arxiv_id

    return ""


def get_cached_semantic_scholar_paper(
    node_id: str,
    db: DBReader,
    semantic_scholar_client: Any,
    paper_cache: dict[str, dict[str, Any] | None],
    semantic_scholar_paper_cache: dict[str, dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Return Semantic Scholar metadata for *node_id*, cached by lookup id."""
    lookup_id = get_semantic_scholar_paper_lookup_id(node_id, db, paper_cache)
    if not lookup_id:
        return None
    if lookup_id not in semantic_scholar_paper_cache:
        semantic_scholar_paper_cache[lookup_id] = semantic_scholar_client.get_paper_by_id(
            lookup_id,
            fields=SEMANTIC_SCHOLAR_PAPER_FIELDS,
        )
    return semantic_scholar_paper_cache[lookup_id]


def collect_semantic_scholar_reference_papers(
    references: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return cited-paper dictionaries from Semantic Scholar reference entries."""
    cited_papers: list[dict[str, Any]] = []
    for reference in references:
        cited_paper = reference.get("citedPaper")
        if isinstance(cited_paper, dict):
            cited_papers.append(cited_paper)
        elif "title" in reference:
            cited_papers.append(reference)
    return cited_papers


def get_cached_semantic_scholar_reference_papers(
    node_id: str,
    db: DBReader,
    semantic_scholar_client: Any,
    paper_cache: dict[str, dict[str, Any] | None],
    semantic_scholar_references_cache: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return Semantic Scholar cited-paper metadata for *node_id*."""
    lookup_id = get_semantic_scholar_paper_lookup_id(node_id, db, paper_cache)
    if not lookup_id:
        return []
    if lookup_id not in semantic_scholar_references_cache:
        references = semantic_scholar_client.get_all_references(
            lookup_id,
            fields=SEMANTIC_SCHOLAR_REFERENCE_FIELDS,
        )
        semantic_scholar_references_cache[lookup_id] = (
            collect_semantic_scholar_reference_papers(references)
        )
    return semantic_scholar_references_cache[lookup_id]


def get_semantic_scholar_metadata_texts(
    paper: dict[str, Any] | None,
    fallback_title: str,
) -> tuple[str, str, str, str]:
    """Return title, authors, venue, and published text from S2 metadata."""
    if paper is None:
        return fallback_title, "", "", ""

    title = str(paper.get("title") or "").strip() or fallback_title
    names: list[str] = []
    for author in paper.get("authors") or []:
        if isinstance(author, dict):
            names.append(str(author.get("name") or ""))
        else:
            names.append(str(author))
    raw_published = (
        paper.get("publicationDate")
        or paper.get("published")
        or paper.get("year")
        or ""
    )
    return (
        title,
        format_authors_text(names),
        str(paper.get("venue") or "").strip(),
        format_published_date_text(str(raw_published)),
    )


def build_semantic_scholar_metadata_needle(
    paper: dict[str, Any] | None,
    fallback_title: str,
) -> str:
    """Return a metadata sentence from Semantic Scholar metadata."""
    title_text, authors_text, venue_text, published_text = (
        get_semantic_scholar_metadata_texts(paper, fallback_title)
    )
    title = quoted_title(title_text)
    publication_text = build_publication_text(venue_text, published_text)

    if authors_text and publication_text:
        return f"Paper {title}, written by {authors_text}, was {publication_text}."
    if authors_text:
        return f"Paper {title}, written by {authors_text}."
    if publication_text:
        return f"Paper {title} was {publication_text}."
    return f"Paper {title}."


def build_semantic_scholar_reference_bullet(paper: dict[str, Any]) -> str:
    """Return one descriptive bullet for a Semantic Scholar reference."""
    fallback_title = str(paper.get("paperId") or "unknown paper")
    title_text, authors_text, venue_text, published_text = (
        get_semantic_scholar_metadata_texts(paper, fallback_title)
    )
    title = quoted_title(title_text)
    publication_text = build_publication_text(venue_text, published_text)
    if authors_text and publication_text:
        clause = f"written by {authors_text} and {publication_text}"
    else:
        clause = f"written by {authors_text}" if authors_text else publication_text
    if clause:
        return f"- Paper {title}, {clause}."
    return f"- Paper {title}."


def build_semantic_scholar_references_needle(
    paper: dict[str, Any] | None,
    reference_papers: list[dict[str, Any]],
    fallback_title: str,
) -> str:
    """Return a Semantic Scholar reference-list needle for one paper."""
    title_text, _, _, _ = get_semantic_scholar_metadata_texts(paper, fallback_title)
    title = quoted_title(title_text)
    if not reference_papers:
        return f"Paper {title} references the following papers:"

    shuffled_reference_papers = shuffled_reference_items(
        reference_papers,
        owner_key=("semantic-general", title_text),
    )
    bullets = "\n".join(
        build_semantic_scholar_reference_bullet(reference_paper)
        for reference_paper in shuffled_reference_papers
    )
    return f"Paper {title} references the following papers:\n{bullets}"


def semantic_scholar_arxiv_id(paper: dict[str, Any] | None) -> str:
    """Return the arXiv ID from a Semantic Scholar paper payload."""
    if paper is None:
        return ""
    external_ids = paper.get("externalIds") or {}
    if not isinstance(external_ids, dict):
        return ""
    return str(external_ids.get("ArXiv") or "").strip()


def semantic_scholar_raw_published(paper: dict[str, Any] | None) -> str:
    """Return the best raw publication date/year from Semantic Scholar metadata."""
    if paper is None:
        return ""
    return str(
        paper.get("publicationDate")
        or paper.get("published")
        or paper.get("year")
        or ""
    )


def get_semantic_scholar_author_names(paper: dict[str, Any] | None) -> list[str]:
    """Return Semantic Scholar author names in payload order."""
    if paper is None:
        return []
    names: list[str] = []
    for author in paper.get("authors") or []:
        if isinstance(author, dict):
            names.append(str(author.get("name") or ""))
        else:
            names.append(str(author))
    return names


def get_semantic_scholar_authors_text(paper: dict[str, Any] | None) -> str:
    """Return Semantic Scholar authors as display text."""
    return format_authors_text(get_semantic_scholar_author_names(paper))


def build_semantic_scholar_apa_reference_bullet(
    paper: dict[str, Any],
) -> str:
    """Return one APA-style bullet from a Semantic Scholar paper payload."""
    fallback_title = str(paper.get("paperId") or "unknown paper")
    title_text, _, venue_text, _ = get_semantic_scholar_metadata_texts(
        paper,
        fallback_title,
    )
    return build_apa_reference_bullet_from_values(
        title_text,
        format_apa_authors_text(get_semantic_scholar_author_names(paper)),
        semantic_scholar_raw_published(paper),
        venue_text,
        semantic_scholar_arxiv_id(paper),
    )


def build_semantic_scholar_apa_paper_needle(
    paper: dict[str, Any] | None,
    reference_papers: list[dict[str, Any]],
    fallback_title: str,
) -> str:
    """Return one structured APA needle from Semantic Scholar metadata."""
    title_text, _, venue_text, _ = get_semantic_scholar_metadata_texts(
        paper,
        fallback_title,
    )
    shuffled_reference_papers = shuffled_reference_items(
        reference_papers,
        owner_key=("semantic-apa", title_text),
    )
    reference_bullets = [
        build_semantic_scholar_apa_reference_bullet(reference_paper)
        for reference_paper in shuffled_reference_papers
    ]
    return build_apa_block(
        title_text,
        get_semantic_scholar_authors_text(paper),
        semantic_scholar_raw_published(paper),
        venue_text,
        semantic_scholar_arxiv_id(paper),
        reference_bullets,
    )


def build_general_paper_groups(
    node_ids: list[str],
    db: DBReader,
    semantic_scholar_client: Any,
    paper_cache: dict[str, dict[str, Any] | None],
    semantic_scholar_paper_cache: dict[str, dict[str, Any] | None],
    semantic_scholar_references_cache: dict[str, list[dict[str, Any]]],
) -> list[list[str]]:
    """Return Semantic Scholar metadata/reference groups for paper nodes."""
    paper_groups: list[list[str]] = []
    for node_id in node_ids:
        if nid_type(node_id) != NODE_PAPER:
            continue

        fallback_title = get_paper_title_text(node_id, db, paper_cache)
        paper = get_cached_semantic_scholar_paper(
            node_id,
            db,
            semantic_scholar_client,
            paper_cache,
            semantic_scholar_paper_cache,
        )
        reference_papers = get_cached_semantic_scholar_reference_papers(
            node_id,
            db,
            semantic_scholar_client,
            paper_cache,
            semantic_scholar_references_cache,
        )
        paper_groups.append(
            [
                build_semantic_scholar_metadata_needle(paper, fallback_title),
                build_semantic_scholar_references_needle(
                    paper,
                    reference_papers,
                    fallback_title,
                ),
            ]
        )
    return paper_groups


def build_semantic_scholar_apa_paper_groups(
    node_ids: list[str],
    db: DBReader,
    semantic_scholar_client: Any,
    paper_cache: dict[str, dict[str, Any] | None],
    semantic_scholar_paper_cache: dict[str, dict[str, Any] | None],
    semantic_scholar_references_cache: dict[str, list[dict[str, Any]]],
) -> list[list[str]]:
    """Return Semantic Scholar APA blocks for paper nodes."""
    paper_groups: list[list[str]] = []
    for node_id in node_ids:
        if nid_type(node_id) != NODE_PAPER:
            continue

        fallback_title = get_paper_title_text(node_id, db, paper_cache)
        paper = get_cached_semantic_scholar_paper(
            node_id,
            db,
            semantic_scholar_client,
            paper_cache,
            semantic_scholar_paper_cache,
        )
        reference_papers = get_cached_semantic_scholar_reference_papers(
            node_id,
            db,
            semantic_scholar_client,
            paper_cache,
            semantic_scholar_references_cache,
        )
        paper_groups.append(
            [
                build_semantic_scholar_apa_paper_needle(
                    paper,
                    reference_papers,
                    fallback_title,
                )
            ]
        )
    return paper_groups


def build_descriptive_paper_groups(
    question: dict,
    db: DBReader,
    graph: CitationGraphLoader,
    needle_type: str,
    *,
    semantic_scholar_client: Any | None = None,
    paper_cache: dict[str, dict[str, Any] | None] | None = None,
    authors_cache: dict[str, list[dict[str, Any]]] | None = None,
    semantic_scholar_paper_cache: dict[str, dict[str, Any] | None] | None = None,
    semantic_scholar_references_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> list[list[str]]:
    """Return descriptive paper groups for one question."""
    node_ids = [str(node_id) for node_id in question.get("node_ids", [])]
    paper_cache = paper_cache if paper_cache is not None else {}
    authors_cache = authors_cache if authors_cache is not None else {}

    if needle_type == "general-describe":
        if semantic_scholar_client is None:
            raise ValueError("general-describe requires a Semantic Scholar client.")
        return build_general_paper_groups(
            node_ids,
            db,
            semantic_scholar_client,
            paper_cache,
            semantic_scholar_paper_cache
            if semantic_scholar_paper_cache is not None
            else {},
            semantic_scholar_references_cache
            if semantic_scholar_references_cache is not None
            else {},
        )
    if needle_type == "apa-describe":
        if semantic_scholar_client is None:
            raise ValueError("apa-describe requires a Semantic Scholar client.")
        return build_semantic_scholar_apa_paper_groups(
            node_ids,
            db,
            semantic_scholar_client,
            paper_cache,
            semantic_scholar_paper_cache
            if semantic_scholar_paper_cache is not None
            else {},
            semantic_scholar_references_cache
            if semantic_scholar_references_cache is not None
            else {},
        )

    edge_types = get_question_edge_types(question, node_ids)
    paper_groups: list[list[str]] = []

    for index, node_id in enumerate(node_ids):
        if nid_type(node_id) != NODE_PAPER:
            continue

        group = [
            build_metadata_needle(
                node_id,
                db,
                paper_cache,
                authors_cache,
                include_venue=(
                    needle_type in {"exact-describe", "general-describe"}
                ),
            )
        ]
        if index < len(edge_types):
            next_node_id = node_ids[index + 1]
            edge_type = edge_types[index]
            if nid_type(next_node_id) == NODE_PAPER:
                if needle_type == "exact-describe":
                    jump = build_exact_jump_needle(
                        node_id,
                        next_node_id,
                        edge_type,
                        db,
                        paper_cache,
                        authors_cache,
                    )
                else:
                    jump = build_general_jump_needle(
                        node_id,
                        next_node_id,
                        edge_type,
                        graph,
                        db,
                        paper_cache,
                        authors_cache,
                        include_venue=True,
                    )
                if jump:
                    group.append(jump)

        paper_groups.append(group)

    return paper_groups


def build_raw_paper_groups(
    node_ids: list[str],
    db: DBReader,
    paper_cache: dict[str, dict[str, Any] | None],
    paper_text_cache: dict[str, str | None],
) -> list[list[str]]:
    """Return one raw-text group per paper node in forward path order."""
    paper_groups: list[list[str]] = []
    for node_id in node_ids:
        if nid_type(node_id) != NODE_PAPER:
            continue
        text = get_cached_paper_text(node_id, db, paper_cache, paper_text_cache)
        if text is None:
            continue
        paper_groups.append([text])
    return paper_groups


def collect_graph_neighbours(
    graph: CitationGraphLoader,
    node_id: str,
    edge_type: str,
    *,
    neighbour_type: str | None = None,
) -> list[str]:
    """Return unique forward neighbours of *node_id* under *edge_type*."""
    matches: list[str] = []
    seen: set[str] = set()
    for neighbour_id, edge_attrs in graph.adj.get(node_id, []):
        if edge_attrs.get("edge_type") != edge_type:
            continue
        if neighbour_type is not None and nid_type(neighbour_id) != neighbour_type:
            continue
        if neighbour_id in seen:
            continue
        seen.add(neighbour_id)
        matches.append(neighbour_id)
    return matches


def collect_unique_path_paper_ids(node_ids: list[str]) -> list[str]:
    """Return unique paper ids from *node_ids* in left-to-right path order."""
    path_paper_ids: list[str] = []
    seen: set[str] = set()
    for node_id in node_ids:
        if nid_type(node_id) != NODE_PAPER:
            continue
        if node_id in seen:
            continue
        seen.add(node_id)
        path_paper_ids.append(node_id)
    return path_paper_ids


def paper_has_needle_content(
    node_id: str,
    needle_type: str,
    db: DBReader,
    paper_cache: dict[str, dict[str, Any] | None],
    paper_text_cache: dict[str, str | None],
) -> bool:
    """Return whether *node_id* can contribute a needle under *needle_type*."""
    if needle_type != "raw":
        return True
    return get_cached_paper_text(node_id, db, paper_cache, paper_text_cache) is not None


def collect_counting_candidate_ids(
    answer_type: str,
    answer_id: str,
    graph: CitationGraphLoader,
) -> list[str]:
    """Return paper ids eligible to support one counting question."""
    if answer_type == COUNT_PAPER_CITATIONS:
        return collect_graph_neighbours(
            graph,
            answer_id,
            EDGE_CITED_BY,
            neighbour_type=NODE_PAPER,
        )
    if answer_type == COUNT_AUTHOR_PAPERS:
        return collect_graph_neighbours(
            graph,
            answer_id,
            EDGE_AUTHORSHIP,
            neighbour_type=NODE_PAPER,
        )
    return []


def sample_counting_answer(
    counting_answer: int | None,
    path_support_count: int,
    available_candidate_count: int,
    rng: random.Random,
    *,
    allow_path_support_count: bool = False,
) -> int | None:
    """Return the realized counting answer for one counting question."""
    if counting_answer is not None:
        if counting_answer < path_support_count:
            return None
        if counting_answer > available_candidate_count:
            return None
        return counting_answer

    lower_bound = (
        path_support_count if allow_path_support_count else path_support_count + 1
    )
    upper_bound = available_candidate_count
    if lower_bound > upper_bound:
        return None
    return rng.randint(lower_bound, upper_bound)


def allows_counting_answer_from_path_support(question: dict) -> bool:
    return (
        len(question.get("node_ids", []))
        >= COUNTING_PATH_SUPPORT_LOWER_BOUND_PATH_LENGTH
    )


def build_counting_plan(
    question: dict,
    db: DBReader,
    graph: CitationGraphLoader,
    needle_type: str,
    counting_answer: int | None,
    rng: random.Random,
    paper_cache: dict[str, dict[str, Any] | None],
    paper_text_cache: dict[str, str | None],
) -> dict[str, list[str]]:
    """Return the selected support papers for one counting question."""
    answer_type = str(question.get("answer_type") or "")
    if answer_type not in {COUNT_PAPER_CITATIONS, COUNT_AUTHOR_PAPERS}:
        return {}

    answer_id = str(question.get("answer_id") or "")
    if not answer_id:
        return {}

    candidate_ids = collect_counting_candidate_ids(answer_type, answer_id, graph)
    available_candidate_ids = [
        node_id
        for node_id in candidate_ids
        if paper_has_needle_content(
            node_id, needle_type, db, paper_cache, paper_text_cache
        )
    ]
    available_candidate_set = set(available_candidate_ids)
    path_paper_ids = collect_unique_path_paper_ids(
        [str(node_id) for node_id in question.get("node_ids", [])]
    )
    path_support_node_ids = [
        node_id for node_id in path_paper_ids if node_id in available_candidate_set
    ]

    realized_counting_answer = sample_counting_answer(
        counting_answer,
        len(path_support_node_ids),
        len(available_candidate_ids),
        rng,
        allow_path_support_count=allows_counting_answer_from_path_support(question),
    )
    if realized_counting_answer is None:
        return {}

    path_support_set = set(path_support_node_ids)
    remaining_candidate_ids = [
        node_id
        for node_id in available_candidate_ids
        if node_id not in path_support_set
    ]
    sample_size = realized_counting_answer - len(path_support_node_ids)
    if sample_size > len(remaining_candidate_ids):
        return {}

    sampled_support_node_ids = (
        rng.sample(remaining_candidate_ids, sample_size) if sample_size > 0 else []
    )
    return {
        "counting_support_node_ids": path_support_node_ids + sampled_support_node_ids,
        "path_support_node_ids": path_support_node_ids,
        "sampled_support_node_ids": sampled_support_node_ids,
    }


def question_has_direct_citing_hop(
    question: dict,
    source_node_id: str,
    target_node_id: str,
) -> bool:
    """Return whether the question path already contains *source* citing *target*."""
    node_ids = [str(node_id) for node_id in question.get("node_ids", [])]
    if len(node_ids) <= 1:
        return False
    edge_types = get_question_edge_types(question, node_ids)
    for index, edge_type in enumerate(edge_types):
        if (
            node_ids[index] == source_node_id
            and node_ids[index + 1] == target_node_id
            and edge_type == EDGE_CITING
        ):
            return True
    return False


def build_citation_support_group(
    support_node_id: str,
    target_node_id: str,
    needle_type: str,
    graph: CitationGraphLoader,
    db: DBReader,
    paper_cache: dict[str, dict[str, Any] | None],
    authors_cache: dict[str, list[dict[str, Any]]],
    *,
    include_metadata: bool,
) -> list[str]:
    """Return the descriptive support group for one counted citing paper."""
    if needle_type == "apa-describe":
        return [
            build_apa_paper_needle(
                support_node_id,
                collect_cited_paper_ids(support_node_id, graph),
                db,
                paper_cache,
                authors_cache,
            )
        ]

    group: list[str] = []
    if include_metadata:
        group.append(
            build_metadata_needle(
                support_node_id,
                db,
                paper_cache,
                authors_cache,
                include_venue=(
                    needle_type in {"exact-describe", "general-describe"}
                ),
            )
        )

    if needle_type == "exact-describe":
        jump = build_exact_jump_needle(
            support_node_id,
            target_node_id,
            EDGE_CITING,
            db,
            paper_cache,
            authors_cache,
        )
    else:
        jump = build_general_jump_needle(
            support_node_id,
            target_node_id,
            EDGE_CITING,
            graph,
            db,
            paper_cache,
            authors_cache,
            include_venue=True,
        )
    if jump:
        group.append(jump)
    return group


def build_counting_support_groups(
    question: dict,
    counting_plan: dict[str, list[str]],
    db: DBReader,
    graph: CitationGraphLoader,
    needle_type: str,
    paper_cache: dict[str, dict[str, Any] | None],
    authors_cache: dict[str, list[dict[str, Any]]],
    paper_text_cache: dict[str, str | None],
    semantic_scholar_client: Any | None = None,
    semantic_scholar_paper_cache: dict[str, dict[str, Any] | None] | None = None,
    semantic_scholar_references_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> list[list[str]]:
    """Return extra needle groups required by the selected counting supports."""
    if not counting_plan:
        return []

    answer_type = str(question.get("answer_type") or "")
    answer_id = str(question.get("answer_id") or "")
    sampled_support_node_ids = list(counting_plan.get("sampled_support_node_ids") or [])
    path_support_node_ids = list(counting_plan.get("path_support_node_ids") or [])
    support_groups: list[list[str]] = []

    if (
        needle_type in {"general-describe", "apa-describe"}
        and semantic_scholar_client is not None
    ):
        if needle_type == "apa-describe":
            return build_semantic_scholar_apa_paper_groups(
                sampled_support_node_ids,
                db,
                semantic_scholar_client,
                paper_cache,
                semantic_scholar_paper_cache
                if semantic_scholar_paper_cache is not None
                else {},
                semantic_scholar_references_cache
                if semantic_scholar_references_cache is not None
                else {},
            )
        return build_general_paper_groups(
            sampled_support_node_ids,
            db,
            semantic_scholar_client,
            paper_cache,
            semantic_scholar_paper_cache
            if semantic_scholar_paper_cache is not None
            else {},
            semantic_scholar_references_cache
            if semantic_scholar_references_cache is not None
            else {},
        )

    if answer_type == COUNT_PAPER_CITATIONS and needle_type in {
        "exact-describe",
        "general-describe",
    }:
        for node_id in path_support_node_ids:
            if question_has_direct_citing_hop(question, node_id, answer_id):
                continue
            group = build_citation_support_group(
                node_id,
                answer_id,
                needle_type,
                graph,
                db,
                paper_cache,
                authors_cache,
                include_metadata=False,
            )
            if group:
                support_groups.append(group)

    for node_id in sampled_support_node_ids:
        if needle_type == "raw":
            text = get_cached_paper_text(node_id, db, paper_cache, paper_text_cache)
            if text is None:
                continue
            support_groups.append([text])
            continue

        if answer_type == COUNT_PAPER_CITATIONS:
            group = build_citation_support_group(
                node_id,
                answer_id,
                needle_type,
                graph,
                db,
                paper_cache,
                authors_cache,
                include_metadata=True,
            )
        elif needle_type == "apa-describe":
            group = [
                build_apa_paper_needle(
                    node_id,
                    collect_cited_paper_ids(node_id, graph),
                    db,
                    paper_cache,
                    authors_cache,
                )
            ]
        else:
            group = [
                build_metadata_needle(
                    node_id,
                    db,
                    paper_cache,
                    authors_cache,
                    include_venue=(
                        needle_type in {"exact-describe", "general-describe"}
                    ),
                )
            ]
        if group:
            support_groups.append(group)
    return support_groups


def build_needles(
    question: dict,
    db: DBReader,
    graph: CitationGraphLoader,
    needle_type: str = "raw",
    *,
    counting_plan: dict[str, list[str]] | None = None,
    paper_cache: dict[str, dict[str, Any] | None] | None = None,
    authors_cache: dict[str, list[dict[str, Any]]] | None = None,
    paper_text_cache: dict[str, str | None] | None = None,
    semantic_scholar_client: Any | None = None,
    semantic_scholar_paper_cache: dict[str, dict[str, Any] | None] | None = None,
    semantic_scholar_references_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> list[str]:
    """Return needles in the requested representation."""
    paper_cache = paper_cache if paper_cache is not None else {}
    authors_cache = authors_cache if authors_cache is not None else {}
    paper_text_cache = paper_text_cache if paper_text_cache is not None else {}
    semantic_scholar_paper_cache = (
        semantic_scholar_paper_cache
        if semantic_scholar_paper_cache is not None
        else {}
    )
    semantic_scholar_references_cache = (
        semantic_scholar_references_cache
        if semantic_scholar_references_cache is not None
        else {}
    )
    node_ids = [str(node_id) for node_id in question.get("node_ids", [])]

    if needle_type == "raw":
        path_groups = build_raw_paper_groups(
            node_ids,
            db,
            paper_cache,
            paper_text_cache,
        )
    elif needle_type in {"exact-describe", "general-describe", "apa-describe"}:
        path_groups = build_descriptive_paper_groups(
            question,
            db,
            graph,
            needle_type,
            semantic_scholar_client=semantic_scholar_client,
            paper_cache=paper_cache,
            authors_cache=authors_cache,
            semantic_scholar_paper_cache=semantic_scholar_paper_cache,
            semantic_scholar_references_cache=semantic_scholar_references_cache,
        )
    else:
        raise ValueError(f"Unsupported needle type: {needle_type}")

    support_groups = build_counting_support_groups(
        question,
        counting_plan or {},
        db,
        graph,
        needle_type,
        paper_cache,
        authors_cache,
        paper_text_cache,
        semantic_scholar_client=semantic_scholar_client,
        semantic_scholar_paper_cache=semantic_scholar_paper_cache,
        semantic_scholar_references_cache=semantic_scholar_references_cache,
    )

    ordered_groups = list(path_groups)
    answer_id = str(question.get("answer_id") or "")
    target_is_paper = bool(answer_id) and nid_type(answer_id) == NODE_PAPER
    if support_groups:
        if target_is_paper and ordered_groups:
            ordered_groups = [*ordered_groups[:-1], *support_groups, ordered_groups[-1]]
        else:
            ordered_groups = [*ordered_groups, *support_groups]

    return flatten_needle_groups(ordered_groups)


def count_graph_neighbours(
    graph: CitationGraphLoader,
    node_id: str,
    edge_type: str,
    *,
    neighbour_type: str | None = None,
) -> int:
    """Count unique forward neighbours of *node_id* under *edge_type*."""
    return len(
        collect_graph_neighbours(
            graph,
            node_id,
            edge_type,
            neighbour_type=neighbour_type,
        )
    )


def resolve_answer_value(
    question: dict,
    db: DBReader,
    graph: CitationGraphLoader,
    counting_plan: dict[str, list[str]] | None = None,
) -> str | int | None:
    """Construct the concrete answer from ``answer_type`` + ``answer_id``."""
    answer_id = question.get("answer_id")
    answer_type = question.get("answer_type")

    if not answer_id or not answer_type:
        return question.get("answer")

    if answer_type == "TITLE":
        paper = db.get_paper(nid_db_id(answer_id))
        return paper["title"] if paper else None

    if answer_type == "AUTHOR_NAME":
        author = db.get_author(nid_db_id(answer_id))
        return author["name"] if author else None

    if answer_type.startswith("AUTHOR_"):
        try:
            author_index = int(answer_type.split("_")[1])
        except (IndexError, ValueError):
            return None
        authors = db.get_expanded_authors_for_paper(nid_db_id(answer_id))
        if 1 <= author_index <= len(authors):
            return authors[author_index - 1]["name"]
        return None

    if answer_type == COUNT_PAPER_CITATIONS:
        if counting_plan:
            return len(counting_plan.get("counting_support_node_ids") or [])
        return count_graph_neighbours(
            graph,
            answer_id,
            "cited_by",
            neighbour_type="paper",
        )

    if answer_type == COUNT_AUTHOR_PAPERS:
        if counting_plan:
            return len(counting_plan.get("counting_support_node_ids") or [])
        return count_graph_neighbours(
            graph,
            answer_id,
            "authorship",
            neighbour_type="paper",
        )

    return question.get("answer")


def process_question(
    question: dict,
    db: DBReader,
    graph: CitationGraphLoader,
    needle_type: str = "raw",
    *,
    counting_answer: int | None,
    rng: random.Random,
    semantic_scholar_client: Any | None = None,
    semantic_scholar_paper_cache: dict[str, dict[str, Any] | None] | None = None,
    semantic_scholar_references_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> dict | None:
    """Return the trimmed output record for a single question."""
    paper_cache: dict[str, dict[str, Any] | None] = {}
    authors_cache: dict[str, list[dict[str, Any]]] = {}
    paper_text_cache: dict[str, str | None] = {}
    semantic_scholar_paper_cache = (
        semantic_scholar_paper_cache
        if semantic_scholar_paper_cache is not None
        else {}
    )
    semantic_scholar_references_cache = (
        semantic_scholar_references_cache
        if semantic_scholar_references_cache is not None
        else {}
    )
    counting_plan = build_counting_plan(
        question,
        db,
        graph,
        needle_type,
        counting_answer,
        rng,
        paper_cache,
        paper_text_cache,
    )
    answer_type = str(question.get("answer_type") or "")
    if (
        answer_type in {COUNT_PAPER_CITATIONS, COUNT_AUTHOR_PAPERS}
        and not counting_plan
    ):
        return None

    node_ids = question.get("node_ids", [])
    record = {
        "question": question["question"],
        "needles": build_needles(
            question,
            db,
            graph,
            needle_type=needle_type,
            counting_plan=counting_plan,
            paper_cache=paper_cache,
            authors_cache=authors_cache,
            paper_text_cache=paper_text_cache,
            semantic_scholar_client=semantic_scholar_client,
            semantic_scholar_paper_cache=semantic_scholar_paper_cache,
            semantic_scholar_references_cache=semantic_scholar_references_cache,
        ),
        "answer": resolve_answer_value(
            question, db, graph, counting_plan=counting_plan
        ),
        "answer_id": question.get("answer_id"),
        "answer_type": question.get("answer_type"),
        "node_ids": node_ids,
    }
    if counting_plan:
        record["counting_support_node_ids"] = list(
            counting_plan.get("counting_support_node_ids") or []
        )
    return record


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def create_needles(
    questions_path: Path,
    graph_json: str | Path | list[str | Path] | tuple[str | Path, ...],
    db_path_hint: str | Path | None = None,
    needle_type: str = "raw",
    counting_answer: int | None = None,
    random_seed: int | None = None,
    semantic_scholar_client: Any | None = None,
) -> dict[str, Any]:
    """Load the questions file, resolve the DB, and build needles for every question."""
    if counting_answer is not None and counting_answer <= 0:
        raise ValueError("counting_answer must be a positive integer.")

    with open(questions_path, encoding="utf-8") as fh:
        questions_data = orjson.loads(fh.read())

    graph_json, resolved_db, db_resolution = resolve_graph_db_paths(
        graph_json,
        db_path_hint,
    )

    db = DBReader(resolved_db)
    graph = CitationGraphLoader(graph_json)
    rng = random.Random(random_seed)
    created_semantic_scholar_client = False
    if (
        needle_type in {"general-describe", "apa-describe"}
        and semantic_scholar_client is None
    ):
        semantic_scholar_client = SemanticScholarClient()
        created_semantic_scholar_client = True

    semantic_scholar_paper_cache: dict[str, dict[str, Any] | None] = {}
    semantic_scholar_references_cache: dict[str, list[dict[str, Any]]] = {}
    try:
        output_questions: list[dict[str, Any]] = []
        skipped_questions = 0
        for question in tqdm(
            questions_data["questions"],
            desc="Building needles",
            unit="question",
        ):
            processed = process_question(
                question,
                db,
                graph,
                needle_type=needle_type,
                counting_answer=counting_answer,
                rng=rng,
                semantic_scholar_client=semantic_scholar_client,
                semantic_scholar_paper_cache=semantic_scholar_paper_cache,
                semantic_scholar_references_cache=semantic_scholar_references_cache,
            )
            if processed is None:
                skipped_questions += 1
                continue
            output_questions.append(processed)
    finally:
        db.close()
        if created_semantic_scholar_client and semantic_scholar_client is not None:
            close_client = getattr(semantic_scholar_client, "close", None)
            if close_client is not None:
                close_client()

    metadata = {
        "source_questions": str(questions_path),
        "graph_json": str(graph_json),
        "db_path": str(resolved_db),
        "db_resolution": db_resolution,
        "total_questions": len(output_questions),
        "skipped_questions": skipped_questions,
        "counting_answer": counting_answer,
    }
    if random_seed is not None:
        metadata["random_seed"] = random_seed

    return {
        "metadata": metadata,
        PARAMETERS_FIELD: merge_parameters(
            questions_data.get(PARAMETERS_FIELD),
            **{
                NEEDLE_TYPE_PARAMETER_FIELD: needle_type,
            },
        ),
        "questions": output_questions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="create_needles",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--questions",
        required=True,
        metavar="JSON",
        help="Input questions JSON file (e.g. questions_1000.json).",
    )
    p.add_argument(
        "--graph",
        required=True,
        nargs="+",
        metavar="JSON",
        help=(
            "Path to the graph JSON, a graphs directory, or an output root — "
            "used to locate the SQLite DB. "
            "When multiple files are given (e.g. via a shell glob "
            "``LogitScope_*.json``), the newest by modification time is used."
        ),
    )
    p.add_argument(
        "--db-path",
        default=None,
        metavar="DB",
        help=(
            "Optional explicit path to the SQLite DB "
            "(auto-detected from the graph JSON by default)."
        ),
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Path to write the output JSON file.",
    )
    p.add_argument(
        "--needle-type",
        default="raw",
        choices=["raw", "exact-describe", "general-describe", "apa-describe"],
        help=(
            "Representation stored in the needles field. "
            "'raw' returns raw paper texts; "
            "'exact-describe' returns exact metadata and citation jumps; "
            "'general-describe' returns Semantic Scholar metadata plus reference lists; "
            "'apa-describe' returns structured metadata plus APA-style references."
        ),
    )
    p.add_argument(
        "--counting-answer",
        type=int,
        default=None,
        help=(
            "Optional exact answer to realize for counting questions. "
            "When omitted, sample uniformly per question between "
            "path_support_count + 1 and the number of candidate support papers "
            "available under the selected needle type."
        ),
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional random seed for selecting counting-support papers.",
    )
    p.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation width (default: 2).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    graph_json = resolve_graph_argument(args.graph)
    if len(args.graph) > 1:
        print(
            "\n[create_needles]"
            f"\n  Graph inputs   : {len(args.graph)} matches"
            f"\n  Using newest   : {graph_json.name}"
        )

    result = create_needles(
        questions_path=Path(args.questions),
        graph_json=graph_json,
        db_path_hint=args.db_path,
        needle_type=args.needle_type,
        counting_answer=args.counting_answer,
        random_seed=args.random_seed,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        orjson.dumps(result, option=orjson.OPT_INDENT_2).decode("utf-8"),
        encoding="utf-8",
    )

    meta = result["metadata"]
    print(
        "\n[create_needles]"
        f"\n  Built          : {meta['total_questions']} question(s)"
        f"\n  Skipped        : {meta['skipped_questions']}"
        f"\n  Needle type    : {args.needle_type}"
        f"\n  Counting answer: {meta['counting_answer']}"
        f"\n  Graph          : {meta['graph_json']}"
        f"\n  DB             : {meta['db_path']}  [{meta['db_resolution']}]"
        f"\n  Output         : {out_path}"
    )


if __name__ == "__main__":
    main()
