from __future__ import annotations

import re
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, NamedTuple

import requests
from loguru import logger

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from src.author_parser import VLMAuthorParser
from src.clients import (
    ArxivFallbackClient,
    ArxivOnlineClient,
    DBLPClient,
    SemanticScholarClient,
)
from src.parsers.pdf_parser import PDFParser
from src.setup.constants import (
    EDGE_CITING,
    FUZZY_MATCHING_THRESHOLD,
    NODE_PAPER,
    nid_db_id,
    nid_type,
    normalize_text,
    preprocess_paper_title,
    preprocess_reference_title,
    slugify,
)
from src.utils.preprocessing import clean_document
from src.utils.string_matching import compact_alnum_string, search_strings

from .network import CitationNetwork, GraphLimits
from .reference_merge import apply_reference_batch
from .storage import DatabaseManager


ArxivGraphClient = ArxivFallbackClient | ArxivOnlineClient
ARXIV_MISSING_TITLE_LOG_PATH = (
    Path(__file__).resolve().parents[2] / "resources" / "missing_arxiv_titles.txt"
)
ARXIV_MISSING_TITLE_LOG_LOCK = threading.Lock()


class AuthorStub(NamedTuple):
    """Lightweight stand-in for VLMAuthorParser results when parsing fails."""

    name: str
    organization: str
    email: str
    orcid: str
    position: int


@dataclass
class PaperIOBundle:
    """All externally-fetched data for one paper, ready for merge."""

    original_nid: str | None
    depth: int
    meta: Any = None
    text_path: Path | None = None
    author_infos: list[Any] = field(default_factory=list)
    refs: list[Any] = field(default_factory=list)
    references_empty: bool = False
    error: str | None = None
    transient: bool = False


ET_AL_RE = re.compile(r"^\s*et\.?\s+al\.?\s*$", re.IGNORECASE)
ARXIV_ID_RE = re.compile(
    r"^(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z](?:[a-z\-]*)\/\d{7}(?:v\d+)?)$",
    re.IGNORECASE,
)


def is_et_al(name: str) -> bool:
    """Return True if *name* is a variant of 'et al.' (not a real author)."""
    return bool(ET_AL_RE.match(name.strip()))


def validate_author_positions(author_infos: list[Any]) -> None:
    """Validate that parsed author positions are consecutive starting at 1."""
    if all(author.position == index + 1 for index, author in enumerate(author_infos)):
        return
    raise ValueError("Author positions must start from 1 and be consecutive")


def build_fallback_author_infos(author_names: list[str]) -> list[AuthorStub]:
    """Build plain author records from metadata when VLM parsing fails."""
    return [
        AuthorStub(name=name, organization="", email="", orcid="", position=position)
        for position, name in enumerate(author_names, start=1)
    ]


def rebuild_author_infos(
    author_infos: list[Any],
    resolved_names: list[str],
) -> list[AuthorStub]:
    """Rebuild author infos from resolved names, preserving existing rich metadata."""
    existing_rich = {
        normalize_text(author.name): author
        for author in author_infos
        if not is_et_al(author.name)
    }
    return [
        AuthorStub(
            name=name,
            organization=getattr(
                existing_rich.get(normalize_text(name)),
                "organization",
                "",
            ),
            email=getattr(existing_rich.get(normalize_text(name)), "email", ""),
            orcid=getattr(existing_rich.get(normalize_text(name)), "orcid", ""),
            position=position,
        )
        for position, name in enumerate(resolved_names, start=1)
    ]


def resolve_full_author_list(
    paper_title: str,
    cur_author_list: list[str],
    dblp_client: DBLPClient,
    semantic_scholar_client: SemanticScholarClient,
) -> list[str] | None:
    """Resolve full author list for a paper via DBLP then Semantic Scholar."""
    paper_title = preprocess_paper_title(paper_title)
    logger.debug("Resolving full author list for {}", paper_title)

    current_authors = [author for author in cur_author_list if not is_et_al(author)]
    if not current_authors:
        return None

    def authors_match(known: list[str], candidates: list[str]) -> bool:
        target = min(3, len(known), len(candidates))
        if target == 0:
            return False
        used: set[int] = set()
        passed = 0
        for known_author in known[:target]:
            available = [
                (index, candidate)
                for index, candidate in enumerate(candidates)
                if index not in used
            ]
            match_index = search_strings(
                known_author,
                [candidate for _, candidate in available],
                FUZZY_MATCHING_THRESHOLD,
            )
            if match_index >= 0:
                used.add(available[match_index][0])
                passed += 1
        return passed == target

    dblp_query = " ".join(f"{word}$" for word in paper_title.split())
    try:
        dblp_publications = dblp_client.search_publications(dblp_query, max_hits=500)
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code is not None and status_code < 500 and status_code != 429:
            raise
        logger.info(
            "DBLP author-list lookup failed for {!r}; falling back to Semantic Scholar: {}",
            paper_title[:60],
            exc,
        )
        dblp_publications = []
    except requests.RequestException as exc:
        logger.info(
            "DBLP author-list lookup failed for {!r}; falling back to Semantic Scholar: {}",
            paper_title[:60],
            exc,
        )
        dblp_publications = []

    for publication in dblp_publications:
        if search_strings(
            paper_title,
            [publication.title],
            FUZZY_MATCHING_THRESHOLD,
        ) < 0:
            continue
        if authors_match(current_authors, publication.authors):
            logger.debug('Found full author list for "{}" using DBLP', paper_title)
            return publication.authors

    semantic_scholar_paper = semantic_scholar_client.search_by_title(paper_title)
    if semantic_scholar_paper:
        if search_strings(
            paper_title,
            [semantic_scholar_paper["title"]],
            FUZZY_MATCHING_THRESHOLD,
        ) < 0:
            return None
        authors = semantic_scholar_client.get_all_authors_of_paper(
            semantic_scholar_paper["paperId"]
        )
        author_names = [author["name"] for author in authors if author.get("name")]
        if authors_match(current_authors, author_names):
            logger.debug(
                'Found full author list for "{}" using Semantic Scholar',
                paper_title,
            )
            return author_names

    return None


def resolve_primary_author_infos(
    author_infos: list[Any],
    paper_title: str,
    dblp_client: DBLPClient | None,
    semantic_scholar_client: SemanticScholarClient | None,
) -> list[Any]:
    """Resolve ``et al.`` placeholders in the primary author list when possible."""
    if not dblp_client or not semantic_scholar_client:
        return author_infos
    if not any(is_et_al(author.name) for author in author_infos):
        return author_infos

    known_names = [author.name for author in author_infos if not is_et_al(author.name)]
    logger.debug(
        "Author list for {!r} contains 'et al.' ({} known name(s)) — attempting resolution via DBLP / Semantic Scholar.",
        paper_title[:60],
        len(known_names),
    )
    resolved_names = resolve_full_author_list(
        paper_title,
        known_names,
        dblp_client,
        semantic_scholar_client,
    )
    if resolved_names is None:
        logger.debug(
            "Et-al resolution failed for {!r}; keeping the unresolved author list and using only the {} known author(s).",
            paper_title[:60],
            len(known_names),
        )
        return author_infos

    rebuilt_author_infos = rebuild_author_infos(author_infos, resolved_names)
    logger.debug(
        "Et-al resolved for {!r}; final author list has {} author(s) total.",
        paper_title[:60],
        len(rebuilt_author_infos),
    )
    return rebuilt_author_infos


def resolve_reference_authors(
    refs: list[Any],
    dblp_client: DBLPClient | None,
    semantic_scholar_client: SemanticScholarClient | None,
) -> list[Any]:
    """Resolve ``et al.`` placeholders inside parsed references."""
    if not dblp_client or not semantic_scholar_client:
        return refs

    for ref in refs:
        parsed_authors = list(ref.authors or [])
        if not any(is_et_al(author) for author in parsed_authors):
            continue
        ref_title = preprocess_reference_title(ref.title or getattr(ref, "raw", "")[:80])
        if not ref_title.strip():
            continue
        known_names = [author for author in parsed_authors if not is_et_al(author)]
        logger.debug(
            "Reference author list for {!r} contains 'et al.' ({} known name(s)) — attempting resolution via DBLP / Semantic Scholar.",
            ref_title[:60],
            len(known_names),
        )
        resolved_names = resolve_full_author_list(
            ref_title,
            known_names,
            dblp_client,
            semantic_scholar_client,
        )
        if resolved_names is None:
            logger.debug(
                "Et-al resolution failed for reference {!r}; final author list remains at {} known author(s).",
                ref_title[:60],
                len(known_names),
            )
            ref.authors = known_names
            continue

        ref.authors = resolved_names
        logger.debug(
            "Et-al resolved for reference {!r}; final author list has {} author(s) total.",
            ref_title[:60],
            len(ref.authors),
        )
    return refs


def author_names_equivalent(left: str, right: str) -> bool:
    """Return whether two author-name strings are equivalent for variant storage."""
    return compact_alnum_string(left) == compact_alnum_string(right)


def resolve_pending_author_name_variants(
    network: CitationNetwork,
    paper_node: str,
    canonical_authors: list[tuple[str, str]],
    *,
    allow_partial_match: bool = False,
) -> None:
    """Attach pending reference-list author spellings to canonical author rows."""
    paper_id = nid_db_id(paper_node)
    rows = network.db.pending_reference_authors_for_paper(paper_id)
    if not rows:
        return

    pending_by_source: dict[str, list[str]] = {}
    for row in rows:
        source_paper_id = str(row["source_paper_id"])
        pending_by_source.setdefault(source_paper_id, []).append(row["name"])

    variants_added = 0
    partially_matched_groups = 0
    skipped_groups = 0
    with network.db.batch():
        for pending_names in pending_by_source.values():
            if len(pending_names) != len(canonical_authors) and not allow_partial_match:
                skipped_groups += 1
                continue

            compare_count = min(len(pending_names), len(canonical_authors))
            if compare_count == 0:
                skipped_groups += 1
                continue
            if len(pending_names) != len(canonical_authors):
                partially_matched_groups += 1

            for pending_name, (author_node, canonical_name) in zip(
                pending_names[:compare_count],
                canonical_authors[:compare_count],
            ):
                if author_names_equivalent(pending_name, canonical_name):
                    continue
                network.add_author_name_variant(author_node, pending_name)
                variants_added += 1

        network.db.delete_pending_reference_authors_for_paper(paper_id)

    logger.debug(
        "[merge] Resolved {} author-name variant(s) for {} from {} pending "
        "reference list(s); partially matched {} list(s); skipped {} "
        "length-mismatched list(s).",
        variants_added,
        paper_node,
        len(pending_by_source),
        partially_matched_groups,
        skipped_groups,
    )


def collect_arxiv_search_clients(arxiv_client: Any) -> list[tuple[str, Any]]:
    """Return the ordered arXiv search backends for the current client."""
    bulk_client = getattr(arxiv_client, "bulk_client", None)
    online_client = getattr(arxiv_client, "online_client", None)
    search_clients: list[tuple[str, Any]] = []
    if bulk_client is not None:
        search_clients.append(("bulk arXiv", bulk_client))
    if online_client is not None:
        search_clients.append(("online arXiv", online_client))
    if search_clients:
        return search_clients
    return [("arXiv", arxiv_client)]


def should_record_missing_arxiv_title(error: str, title: str) -> bool:
    """Return whether *error* means the title was unavailable on arXiv."""
    cleaned_title = preprocess_paper_title(title)
    return error in {
        f"No arXiv results for {cleaned_title!r}",
        f"No fuzzy match above threshold for {cleaned_title!r}",
    }


def record_missing_arxiv_title(title: str) -> None:
    """Append one unresolved arXiv title search to the shared resources log."""
    cleaned_title = preprocess_paper_title(title)
    if not cleaned_title:
        return

    with ARXIV_MISSING_TITLE_LOG_LOCK:
        try:
            ARXIV_MISSING_TITLE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with ARXIV_MISSING_TITLE_LOG_PATH.open("a+", encoding="utf-8") as handle:
                lock_acquired = False
                try:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                        lock_acquired = True

                    handle.seek(0)
                    logged_titles = {
                        line.strip()
                        for line in handle.read().splitlines()
                        if line.strip()
                    }
                    if cleaned_title in logged_titles:
                        return

                    handle.seek(0, os.SEEK_END)
                    handle.write(f"{cleaned_title}\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    if lock_acquired:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            logger.warning(
                "Failed to record missing arXiv title {!r}: {}",
                cleaned_title,
                exc,
            )
            return

    logger.info(
        "Recorded missing arXiv title {!r} in {}",
        cleaned_title,
        ARXIV_MISSING_TITLE_LOG_PATH,
    )


def search_paper_metadata_by_title(
    search_client: Any,
    backend_name: str,
    title: str,
) -> tuple[Any | None, str | None]:
    """Search one arXiv backend by title and return the first fuzzy-validated hit."""
    logger.debug("Searching {} for: {}", backend_name, title[:60])
    results = search_client.search(f'ti:"{title}"', max_results=50)
    if not results:
        return None, f"No {backend_name} results for {title!r}"

    match_index = search_strings(
        title,
        [result.title for result in results],
        FUZZY_MATCHING_THRESHOLD,
    )
    if match_index >= 0:
        result = results[match_index]
        logger.debug(
            "Fuzzy match found on {}: {} ({})",
            backend_name,
            result.title[:60],
            result.arxiv_id,
        )
        try:
            return search_client.get_paper(result.arxiv_id), None
        except Exception as exc:
            return None, f"{backend_name} fetch failed for {result.arxiv_id}: {exc}"

    if backend_name == "arXiv":
        return None, f"No fuzzy match above threshold for {title!r}"
    return None, f"No {backend_name} fuzzy match above threshold for {title!r}"


def summarize_arxiv_backend_errors(errors: list[str], title: str) -> str:
    """Collapse overlapping backend miss messages into one user-facing error."""
    unique_errors = list(dict.fromkeys(errors))
    if not unique_errors:
        return f"No arXiv results for {title!r}"

    no_results_errors = {
        f"No arXiv results for {title!r}",
        f"No bulk arXiv results for {title!r}",
        f"No online arXiv results for {title!r}",
    }
    no_fuzzy_match_errors = {
        f"No fuzzy match above threshold for {title!r}",
        f"No bulk arXiv fuzzy match above threshold for {title!r}",
        f"No online arXiv fuzzy match above threshold for {title!r}",
    }

    if all(error in no_results_errors for error in unique_errors):
        return f"No arXiv results for {title!r}"
    if all(error in no_fuzzy_match_errors for error in unique_errors):
        return f"No fuzzy match above threshold for {title!r}"
    if len(unique_errors) == 1:
        return unique_errors[0]
    return " ; ".join(unique_errors)


def resolve_paper_metadata(
    arxiv_client: ArxivGraphClient,
    *,
    arxiv_id: str = "",
    title: str = "",
) -> tuple[Any | None, str | None]:
    """Resolve canonical arXiv metadata from an ID or title query."""
    if arxiv_id:
        try:
            return arxiv_client.get_paper(arxiv_id), None
        except Exception as exc:
            return None, f"arXiv fetch failed for {arxiv_id}: {exc}"

    title = preprocess_paper_title(title)
    errors: list[str] = []
    for backend_name, search_client in collect_arxiv_search_clients(arxiv_client):
        meta, error = search_paper_metadata_by_title(
            search_client,
            backend_name,
            title,
        )
        if meta is not None:
            return meta, None
        if error:
            errors.append(error)

    error = summarize_arxiv_backend_errors(errors, title)
    if should_record_missing_arxiv_title(error, title):
        record_missing_arxiv_title(title)
    return None, error


def build_source_archive_path(source_dir: Path, arxiv_id: str) -> Path:
    """Return the canonical source-archive path for one arXiv paper."""
    return source_dir / f"{arxiv_id.replace('/', '_')}.tar.gz"


def download_source_archive(
    arxiv_client: ArxivGraphClient,
    arxiv_id: str,
    source_dir: Path,
) -> Path | None:
    """Download the source archive to ``source_dir`` and normalize its filename."""
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_path = build_source_archive_path(source_dir, arxiv_id)
    if archive_path.exists():
        logger.debug("[worker] Source cached: {}", archive_path)
        return archive_path

    downloaded = arxiv_client.download_source(
        arxiv_id,
        output_dir=source_dir,
    )
    if downloaded is None:
        return None

    downloaded_path = Path(downloaded)
    if downloaded_path != archive_path:
        downloaded_path.replace(archive_path)

    logger.debug("[worker] Saved source archive → {}", archive_path)
    return archive_path


def parse_references_from_source_archive(
    tex_reference_parser: Any,
    source_archive_path: Path,
) -> list[Any]:
    """Extract a source archive to a temp dir and parse references from it."""
    from src.reference_parser.tex_reference_parser import extract_archive

    with tempfile.TemporaryDirectory(
        prefix=f"paper_source_{slugify(source_archive_path.name)}_"
    ) as temp_dir:
        temp_source_dir = Path(temp_dir)
        extract_archive(source_archive_path, temp_source_dir)
        return tex_reference_parser.parse(temp_source_dir)


def resolve_paper_references(
    arxiv_client: ArxivGraphClient,
    tex_reference_parser: Any,
    ref_parser: Any,
    extracted_text: str,
    arxiv_id: str,
    source_dir: Path,
    file_lock_fn: Any,
    file_lock_key: str,
    raise_error: bool = False,
) -> list[Any]:
    """Resolve references from source first, then fall back to extracted text."""
    if tex_reference_parser is not None:
        try:
            with file_lock_fn(file_lock_key):
                source_archive_path = download_source_archive(
                    arxiv_client,
                    arxiv_id,
                    source_dir,
                )
        except Exception as exc:
            logger.warning(
                "[worker] Source download failed for {}: {}. Falling back to extracted text.",
                arxiv_id,
                exc,
            )
        else:
            if source_archive_path is None:
                logger.debug(
                    "[worker] No source available for {}; falling back to extracted text.",
                    arxiv_id,
                )
            else:
                try:
                    refs = parse_references_from_source_archive(
                        tex_reference_parser,
                        source_archive_path,
                    )
                    if refs:
                        logger.debug(
                            "[worker] Extracted {} references from source", len(refs)
                        )
                        return refs
                    logger.warning(
                        "[worker] Source reference parsing returned no references for {}; falling back to extracted text.",
                        arxiv_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "[worker] Source reference parsing failed for {}: {}. Falling back to extracted text.",
                        arxiv_id,
                        exc,
                    )

    try:
        refs = ref_parser.parse(extracted_text)
        logger.debug("[worker] Extracted {} references", len(refs))
        return refs
    except Exception as exc:
        logger.warning("[worker] Reference parsing failed: {}", exc)
        if raise_error:
            raise
        return []


def fetch_paper_data(
    arxiv_id: str,
    title: str,
    nid: str | None,
    depth: int,
    arxiv_client: ArxivGraphClient,
    pdf_parser: PDFParser,
    author_parser: VLMAuthorParser,
    ref_parser: Any,
    tex_reference_parser: Any,
    pdf_dir: Path,
    text_dir: Path,
    source_dir: Path,
    file_lock_fn: Any,
    dblp_client: DBLPClient | None = None,
    semantic_scholar_client: SemanticScholarClient | None = None,
    raise_error: bool = False,
    prefetched_meta: Any | None = None,
    *,
    resolve_et_al_authors: bool = False,
) -> PaperIOBundle:
    """Fetch arXiv metadata, download PDF, parse text/authors/references."""
    bundle = PaperIOBundle(original_nid=nid, depth=depth)

    if prefetched_meta is not None:
        bundle.meta = prefetched_meta
    else:
        bundle.meta, bundle.error = resolve_paper_metadata(
            arxiv_client,
            arxiv_id=arxiv_id,
            title=title,
        )
        if bundle.meta is None:
            bundle.transient = True
            if raise_error and bundle.error:
                raise RuntimeError(bundle.error)
            return bundle

    safe_id = slugify(bundle.meta.arxiv_id)
    logger.info(
        'START paper "{}" | depth={} | arxiv_id={}',
        bundle.meta.title,
        depth,
        bundle.meta.arxiv_id,
    )

    with file_lock_fn(safe_id):
        pdf_path = pdf_dir / f"{safe_id}.pdf"
        legacy_pdf_paths = [
            pdf_dir / f"arXiv-{bundle.meta.arxiv_id}.pdf",
            pdf_dir / f"arXiv-{bundle.meta.arxiv_id.replace('/', '_')}.pdf",
        ]
        cached_pdf_path = pdf_path if pdf_path.exists() else None
        if cached_pdf_path is None:
            cached_pdf_path = next(
                (
                    legacy_path
                    for legacy_path in legacy_pdf_paths
                    if legacy_path.exists()
                ),
                None,
            )
        if cached_pdf_path is not None:
            if cached_pdf_path != pdf_path:
                cached_pdf_path.replace(pdf_path)
            logger.debug("[worker] PDF cached: {}", pdf_path)
        else:
            logger.debug("[worker] Downloading PDF for {} ...", bundle.meta.arxiv_id)
            downloaded = arxiv_client.download_pdf(
                bundle.meta.arxiv_id,
                output_dir=pdf_dir,
            )
            if downloaded is None:
                bundle.error = f"PDF download failed for {bundle.meta.arxiv_id}"
                bundle.transient = True
                return bundle
            downloaded_path = Path(downloaded)
            if downloaded_path != pdf_path:
                if downloaded_path.exists():
                    downloaded_path.replace(pdf_path)
                else:
                    pdf_path = downloaded_path

    extracted_text = ""
    with file_lock_fn(safe_id):
        text_path = text_dir / f"{safe_id}.md"
        if text_path.exists():
            logger.debug("[worker] Using cached text: {}", text_path)
            extracted_text = text_path.read_text(encoding="utf-8")
            bundle.text_path = text_path
        else:
            logger.info("PDF parse start | {}", bundle.meta.arxiv_id)
            pdf_parse_started = time.perf_counter()
            try:
                extracted_text = pdf_parser.parse_single_pdf(str(pdf_path))
                extracted_text = clean_document(extracted_text)
                text_path.write_text(extracted_text, encoding="utf-8")
                bundle.text_path = text_path
                logger.info(
                    "PDF parse done | {} | {:.1f}s",
                    bundle.meta.arxiv_id,
                    time.perf_counter() - pdf_parse_started,
                )
                logger.debug("[worker] Saved extracted text → {}", text_path)
            except Exception as exc:
                bundle.error = f"PDF parsing failed: {exc}"
                bundle.transient = True
                if raise_error:
                    raise
                return bundle

    logger.info("Author parse start | {}", bundle.meta.arxiv_id)
    author_parse_started = time.perf_counter()
    try:
        bundle.author_infos = author_parser.parse(str(pdf_path)).authors
        validate_author_positions(bundle.author_infos)
        logger.info(
            "Author parse done | {} | {} author(s) | {:.1f}s",
            bundle.meta.arxiv_id,
            len(bundle.author_infos),
            time.perf_counter() - author_parse_started,
        )
    except Exception as exc:
        logger.warning("[worker] Author parsing failed — using arXiv metadata: {}", exc)
        if raise_error:
            raise
        bundle.author_infos = build_fallback_author_infos(bundle.meta.authors)

    if resolve_et_al_authors:
        bundle.author_infos = resolve_primary_author_infos(
            bundle.author_infos,
            bundle.meta.title,
            dblp_client,
            semantic_scholar_client,
        )

    logger.info("Reference parse start | {}", bundle.meta.arxiv_id)
    reference_parse_started = time.perf_counter()
    bundle.refs = resolve_paper_references(
        arxiv_client,
        tex_reference_parser,
        ref_parser,
        extracted_text,
        bundle.meta.arxiv_id,
        source_dir,
        file_lock_fn,
        safe_id,
        raise_error=raise_error,
    )
    logger.info(
        "Reference parse done | {} | {} ref(s) | {:.1f}s",
        bundle.meta.arxiv_id,
        len(bundle.refs),
        time.perf_counter() - reference_parse_started,
    )
    if not bundle.refs:
        message = (
            f"No references were parsed for {bundle.meta.arxiv_id}; leaving the "
            "paper unexplored so it can be retried."
        )
        logger.warning("[worker] {}", message)
        if raise_error:
            raise RuntimeError(message)
        bundle.references_empty = True

    # Resolve et al. author names for reference items
    if resolve_et_al_authors:
        bundle.refs = resolve_reference_authors(
            bundle.refs,
            dblp_client,
            semantic_scholar_client,
        )
    return bundle


def apply_fetched_paper(
    network: CitationNetwork,
    bundle: PaperIOBundle,
    *,
    pre_upserted_paper_nid: str | None = None,
) -> tuple[str | None, bool]:
    """Apply a fetched paper bundle to the shared ``CitationNetwork``."""
    original_label = bundle.original_nid or getattr(bundle.meta, "arxiv_id", "<seed>")
    if bundle.error:
        if not bundle.transient and bundle.original_nid is not None:
            network.mark_explored(bundle.original_nid)
        logger.warning(
            "Paper {} failed (transient={}): {}",
            original_label,
            bundle.transient,
            bundle.error,
        )
        return None, False

    meta = bundle.meta
    if pre_upserted_paper_nid is not None:
        paper_node = pre_upserted_paper_nid
    else:
        paper_node = network.upsert_paper(
            arxiv_id=meta.arxiv_id,
            title=meta.title,
            abstract=meta.abstract,
            published=meta.published,
            pdf_url=meta.pdf_url,
            depth=bundle.depth,
        )
        if paper_node is None:
            return None, False

    if network.is_explored(nid_db_id(paper_node)):
        logger.debug("[merge] Already explored — skipping {}", meta.arxiv_id)
        if bundle.original_nid is not None and bundle.original_nid != paper_node:
            network.mark_explored(bundle.original_nid)
        return paper_node, False

    if bundle.text_path:
        network.db.update_paper(nid_db_id(paper_node), text_path=str(bundle.text_path))

    has_et_al_author = any(is_et_al(author_info.name) for author_info in bundle.author_infos)
    canonical_authors: list[tuple[str, str]] = []
    for author_info in bundle.author_infos:
        if is_et_al(author_info.name):
            continue
        author_node = network.upsert_author(
            name=author_info.name,
            orcid=getattr(author_info, "orcid", ""),
        )
        if author_node is None:
            continue
        network.add_authorship(
            paper_node,
            author_node,
            position=author_info.position,
            organization=getattr(author_info, "organization", ""),
            email=getattr(author_info, "email", ""),
        )
        canonical_authors.append((author_node, author_info.name))

    apply_reference_batch(
        network=network,
        paper_node=paper_node,
        refs=bundle.refs,
        depth=bundle.depth,
        paper_label=meta.arxiv_id,
        skip_author=is_et_al,
    )

    if bundle.references_empty:
        network.mark_unexplored(paper_node)
        if bundle.original_nid is not None and bundle.original_nid != paper_node:
            network.mark_unexplored(bundle.original_nid)
        logger.warning(
            "[merge] No references found for {}; paper remains unexplored.",
            meta.arxiv_id,
        )
        logger.debug("[merge] {}", network.stats())
        return paper_node, False

    resolve_pending_author_name_variants(
        network,
        paper_node,
        canonical_authors,
        allow_partial_match=has_et_al_author,
    )

    network.mark_explored(paper_node)
    if bundle.original_nid is not None and bundle.original_nid != paper_node:
        network.mark_explored(bundle.original_nid)

    max_depth = (
        "unbounded"
        if network.max_depth is None
        else network.max_depth
        if network.max_depth > 0
        else "unknown (legacy graph)"
    )
    logger.info(
        'FINISH paper "{}" | depth={} | max_depth={} | explored={}',
        meta.title,
        bundle.depth,
        max_depth,
        network.explored_count(),
    )
    logger.debug("[merge] {}", network.stats())
    return paper_node, True


class PaperExpander:
    """Fetch a paper from arXiv, parse it, and merge it into a network."""

    def __init__(
        self,
        network: CitationNetwork,
        arxiv_client: ArxivGraphClient,
        pdf_parser: PDFParser,
        author_parser: VLMAuthorParser,
        reference_parser: Any,
        tex_reference_parser: Any,
        dblp_client: DBLPClient,
        semantic_scholar_client: SemanticScholarClient,
        pdf_dir: Path,
        text_dir: Path,
        source_dir: Path,
        resolve_et_al_authors: bool = False,
        raise_error: bool = False,
    ) -> None:
        self.net = network
        self.arxiv = arxiv_client
        self.pdf_parser = pdf_parser
        self.author_parser = author_parser
        self.ref_parser = reference_parser
        self.tex_ref_parser = tex_reference_parser
        self.pdf_dir = pdf_dir
        self.text_dir = text_dir
        self.source_dir = source_dir
        self.dblp_client = dblp_client
        self.semantic_scholar_client = semantic_scholar_client
        self.resolve_et_al_authors = resolve_et_al_authors
        self.raise_error = raise_error
        self.file_locks_mu = threading.Lock()
        self.file_locks: dict[str, threading.Lock] = {}

    @property
    def limits(self) -> GraphLimits:
        return self.net.limits

    def get_file_lock(self, key: str) -> threading.Lock:
        with self.file_locks_mu:
            if key not in self.file_locks:
                self.file_locks[key] = threading.Lock()
            return self.file_locks[key]

    def expand_by_title(self, title: str, depth: int = 0) -> str | None:
        meta, error = resolve_paper_metadata(self.arxiv, title=title)
        if meta is None:
            if error:
                logger.error(error)
            else:
                logger.error("Could not resolve title {!r}", title)
            if self.raise_error and error:
                raise RuntimeError(error)
            return None
        return self.expand_with_meta(meta, depth)

    def expand_by_arxiv_id(self, arxiv_id: str, depth: int = 0) -> str | None:
        meta, error = resolve_paper_metadata(self.arxiv, arxiv_id=arxiv_id)
        if meta is None:
            if error:
                logger.error(error)
            else:
                logger.error("Could not resolve arXiv ID {}", arxiv_id)
            if self.raise_error and error:
                raise RuntimeError(error)
            return None
        return self.expand_with_meta(meta, depth)

    def expand_with_meta(self, meta: Any, depth: int = 0) -> str | None:
        if self.net.at_paper_capacity():
            logger.info("Explored-paper capacity reached — skipping {}", meta.arxiv_id)
            return None

        paper_node = self.net.upsert_paper(
            arxiv_id=meta.arxiv_id,
            title=meta.title,
            abstract=meta.abstract,
            published=meta.published,
            pdf_url=meta.pdf_url,
            depth=depth,
        )
        if paper_node is None:
            return None

        if self.net.is_explored(nid_db_id(paper_node)):
            logger.debug("Already explored — skipping {}", meta.arxiv_id)
            return paper_node

        bundle = fetch_paper_data(
            arxiv_id=meta.arxiv_id,
            title=meta.title,
            nid=paper_node,
            depth=depth,
            arxiv_client=self.arxiv,
            pdf_parser=self.pdf_parser,
            author_parser=self.author_parser,
            ref_parser=self.ref_parser,
            tex_reference_parser=self.tex_ref_parser,
            pdf_dir=self.pdf_dir,
            text_dir=self.text_dir,
            source_dir=self.source_dir,
            file_lock_fn=self.get_file_lock,
            dblp_client=self.dblp_client,
            semantic_scholar_client=self.semantic_scholar_client,
            raise_error=self.raise_error,
            prefetched_meta=meta,
            resolve_et_al_authors=self.resolve_et_al_authors,
        )
        try:
            canonical_nid, _was_new = apply_fetched_paper(
                self.net,
                bundle,
                pre_upserted_paper_nid=paper_node,
            )
        except Exception as exc:
            logger.warning(
                "Expansion of {} failed during merge: {}", meta.arxiv_id, exc
            )
            if self.raise_error:
                raise
            return None
        return canonical_nid


def get_paper_depth(db: DatabaseManager, nid: str) -> int:
    """Return the stored BFS depth for a paper node, or 0 if unset."""
    row = db.get_paper(nid_db_id(nid))
    if row is None:
        return 0
    depth = row["depth"]
    return int(depth) if depth is not None and int(depth) >= 0 else 0


def enqueue_citation_neighbors(
    network: CitationNetwork,
    paper_nid: str,
    depth: int,
    max_depth: int | None,
    seen: set[str],
    queue: deque[tuple[str, int]],
) -> None:
    """Append unexplored citation neighbors to the BFS queue."""
    if max_depth is not None and depth >= max_depth:
        return
    if paper_nid not in network.G:
        return

    for neighbor_nid, edge_data in network.G[paper_nid].items():
        if edge_data.get("edge_type") != EDGE_CITING:
            continue
        if nid_type(neighbor_nid) != NODE_PAPER:
            continue
        if neighbor_nid in seen:
            continue
        if network.is_explored(nid_db_id(neighbor_nid)):
            continue
        seen.add(neighbor_nid)
        queue.append((neighbor_nid, depth + 1))
