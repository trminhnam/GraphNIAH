"""
Semantic Scholar API client.

```bash
python -m src.clients.semantic_scholar_client
```

Wraps the Semantic Scholar Graph API (https://api.semanticscholar.org/graph/v1)
to retrieve paper metadata, reference lists, and citation lists.

The API accepts multiple paper ID formats, including bare arXiv IDs::

    arXiv:2302.13971   (preferred for arxiv papers)
    DOI:10.18653/v1/P18-1205
    <s2_paper_id>      (40-char hex)

Usage::

    client = SemanticScholarClient()

    # Fetch metadata + references for one paper
    paper = client.get_paper_by_id("arXiv:2302.13971", fields=["title", "references"])

    # Get the full reference list (auto-paginates)
    refs = client.get_all_references("arXiv:2302.13971")
    for ref in refs:
        cited = ref["citedPaper"]
        print(cited.get("title"), cited.get("externalIds", {}).get("ArXiv"))

    # Batch-fetch multiple papers
    papers = client.get_papers_batch(["arXiv:1706.03762", "arXiv:2005.11401"])

    # Enrich a list of PaperMeta objects with their references
    metas = ArxivOnlineClient.load_jsonl("data/arxiv_quantum_2025_01.jsonl")
    enriched = client.enrich_with_references(metas)
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Optional, TypedDict
from urllib.parse import quote

import requests
from loguru import logger

from src.clients.local_sqlite_cache import LocalSqliteCache

# Default fields to include when fetching reference / citation entries
_DEFAULT_PAPER_FIELDS = [
    "paperId",
    "externalIds",
    "title",
    # "abstract",
    "authors",
    "year",
    "publicationDate",
    "venue",
    # "referenceCount",
    # "citationCount",
    # "isOpenAccess",
    "openAccessPdf",
    "url",
]

_DEFAULT_REF_FIELDS = [f"citedPaper.{f}" for f in _DEFAULT_PAPER_FIELDS]
_DEFAULT_CITE_FIELDS = [f"citingPaper.{f}" for f in _DEFAULT_PAPER_FIELDS]
HTTP_429_MAX_RETRIES = 10
HTTP_429_BACKOFF_SECONDS = (5.0, 10.0, 15.0)
HTTP_429_COOLDOWN_SECONDS = 30.0
HTTP_500_MAX_RETRIES = 3
HTTP_500_RETRY_WAIT_SECONDS = 15.0
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_DELAY_SECONDS = 3.0
DEFAULT_PAGE_LIMIT = 1000
DEFAULT_MAX_RESULTS = 10000
LEGACY_CACHE_PAGE_LIMITS = (500, 100)

_DEFAULT_AUTHOR_FIELDS = [
    "authorId",
    # "externalIds",
    "url",
    "name",
    "affiliations",
    # "homepage",
    # "paperCount",
    # "citationCount",
    # "hIndex",
]

_ARXIV_ID_RE = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$",
    re.IGNORECASE,
)


def comma_join(values: Optional[list[str] | tuple[str, ...] | str]) -> Optional[str]:
    """Return a comma-separated query value for Semantic Scholar list filters."""
    if values is None or isinstance(values, str):
        return values
    return ",".join(values)


def normalize_paper_id(paper_id: str) -> str:
    """Convert a bare arXiv ID to the Semantic Scholar ``arXiv:`` prefix format.

    Strips the version suffix (e.g. ``v1``) because S2 resolves to the latest
    version by default when no version is specified.

    Examples
    --------
    ``"2302.13971v2"`` → ``"arXiv:2302.13971"``
    ``"arXiv:2302.13971"`` → ``"arXiv:2302.13971"``  (idempotent)
    ``"https://arxiv.org/abs/2302.13971"`` → ``"arXiv:2302.13971"``
    """
    normalized = paper_id.strip()
    bare = normalized.split("/abs/")[-1]
    bare = bare.removeprefix("arXiv:").removeprefix("ARXIV:").removeprefix("arxiv:")
    if not _ARXIV_ID_RE.fullmatch(bare):
        return normalized
    bare = re.sub(r"v\d+$", "", bare, flags=re.IGNORECASE)
    return f"arXiv:{bare}"


def load_dotenv_if_available() -> None:
    """Load .env so SEMANTIC_SCHOLAR_API_KEY is visible to os.getenv."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def path_param(value: str) -> str:
    """Encode a value for use as one Semantic Scholar path segment."""
    return quote(value, safe="")


def paper_search_params(
    *,
    query: str,
    fields: Optional[list[str] | str] = None,
    publication_types: Optional[list[str] | str] = None,
    open_access_pdf: bool = False,
    min_citation_count: Optional[int | str] = None,
    publication_date_or_year: Optional[str] = None,
    year: Optional[str] = None,
    venue: Optional[list[str] | str] = None,
    fields_of_study: Optional[list[str] | str] = None,
) -> dict[str, Any]:
    """Build common paper search query parameters."""
    params: dict[str, Any] = {
        "query": query,
        "fields": comma_join(fields),
        "publicationTypes": comma_join(publication_types),
        "minCitationCount": min_citation_count,
        "publicationDateOrYear": publication_date_or_year,
        "year": year,
        "venue": comma_join(venue),
        "fieldsOfStudy": comma_join(fields_of_study),
    }
    if open_access_pdf:
        params["openAccessPdf"] = ""
    return params


def paper_side_status(
    paper: dict[str, Any],
    *,
    count_key: str,
    embedded_key: str,
    max_results: int,
) -> tuple[bool, list[dict[str, Any]] | None]:
    """Check-only completeness test for one side (citations or references) of
    a combined paper response -- no fetching.

    Only ``citations``/``references`` are ever capped by Semantic Scholar:
    every other requested field (title, venue, authors, externalIds, ...)
    has been confirmed to survive untouched even when a paper's citations
    list is truncated to zero, so this is the only pair of fields that ever
    needs checking.

    Returns ``(is_complete, embedded_list_or_None)``. When ``is_complete`` is
    True, ``embedded_list_or_None`` is the usable (already max_results-sliced)
    embedded list.
    """
    if max_results == 0:
        return True, []

    total_count = paper.get(count_key)
    embedded = paper.get(embedded_key)

    if isinstance(total_count, int) and total_count >= 0:
        if total_count == 0:
            return True, []
        required_count = min(total_count, max_results)
        if isinstance(embedded, list) and len(embedded) >= required_count:
            return True, embedded[:max_results]

    return False, embedded


def resolve_expansion_side(
    paper: dict[str, Any],
    *,
    embedded_key: str,
    count_key: str,
    envelope_key: str,
    paper_id: str,
    fetch_all: Callable[..., list[dict[str, Any]]],
    fields: Optional[list[str]],
    max_results: int,
) -> list[dict[str, Any]]:
    """Return up to max_results envelope-wrapped entries for one side
    (citations or references) of a combined paper expansion, falling back to
    the paginated endpoint when the embedded list is incomplete."""
    is_complete, embedded = paper_side_status(
        paper, count_key=count_key, embedded_key=embedded_key, max_results=max_results
    )
    if is_complete:
        return [{envelope_key: item} for item in embedded]

    logger.debug(
        "get_paper_with_citations_and_references({}) falling back to paginated {} "
        "(embedded={}, count={}, max_results={})",
        paper_id,
        embedded_key,
        len(embedded or []),
        paper.get(count_key),
        max_results,
    )
    return fetch_all(paper_id, fields=fields, max_results=max_results)


def build_combined_expansion_fields(
    fields: Optional[list[str]],
    *,
    include_citations: bool = True,
    include_references: bool = True,
) -> list[str]:
    """Build fields for a paper request with selected expansion sides.

    Citation and reference count/list fields are included only when their
    corresponding flag is true. The result is order-preserving and
    deduplicated, and never mutates ``fields``.
    """
    base_fields = list(dict.fromkeys(fields or _DEFAULT_PAPER_FIELDS))
    return list(
        dict.fromkeys(
            [
                *base_fields,
                *(["citationCount"] if include_citations else []),
                *(["referenceCount"] if include_references else []),
                *((f"citations.{f}" for f in base_fields) if include_citations else []),
                *(
                    (f"references.{f}" for f in base_fields)
                    if include_references
                    else []
                ),
            ]
        )
    )


class Error400(TypedDict, total=False):
    error: str


class SemanticScholarRateLimitError(requests.HTTPError):
    """Raised when Semantic Scholar keeps returning HTTP 429 after cooldown."""


class Error404(TypedDict, total=False):
    error: str


class NormalizedAffiliation(TypedDict, total=False):
    rorId: str
    rorDisplayName: str


class PublicationVenue(TypedDict, total=False):
    id: str
    name: str
    type: str
    alternate_names: list[str]
    url: str


class OpenAccessPdf(TypedDict, total=False):
    url: str
    status: str
    license: str
    disclaimer: str


class S2FieldOfStudy(TypedDict, total=False):
    category: str
    source: str


class Journal(TypedDict, total=False):
    name: str
    volume: str
    pages: str


class CitationStyles(TypedDict, total=False):
    bibtex: str


class Embedding(TypedDict, total=False):
    model: str
    vector: list[float]


class Tldr(TypedDict, total=False):
    model: str
    text: str


class AuthorInfo(TypedDict, total=False):
    authorId: str
    name: str


class AuthorInPaper(AuthorInfo, total=False):
    externalIds: dict[str, Any]
    url: str
    affiliations: list[str]
    homepage: str
    paperCount: int
    citationCount: int
    hIndex: int
    normalizedAffiliations: list[NormalizedAffiliation]


class BasePaper(TypedDict, total=False):
    paperId: str
    corpusId: int
    externalIds: dict[str, Any]
    url: str
    title: str
    abstract: str
    venue: str
    publicationVenue: PublicationVenue
    year: int
    referenceCount: int
    citationCount: int
    influentialCitationCount: int
    isOpenAccess: bool
    openAccessPdf: OpenAccessPdf
    fieldsOfStudy: list[str]
    s2FieldsOfStudy: list[S2FieldOfStudy]
    publicationTypes: list[str]
    publicationDate: str
    journal: Journal
    citationStyles: CitationStyles
    authors: list[AuthorInfo]


class PaperInfo(TypedDict, total=False):
    paperId: str
    corpusId: int
    url: str
    title: str
    venue: str
    publicationVenue: PublicationVenue
    year: int
    authors: list[AuthorInfo]


class FullPaper(BasePaper, total=False):
    authors: list[AuthorInPaper]
    citations: list[BasePaper]
    references: list[BasePaper]
    embedding: Embedding
    tldr: Tldr
    textAvailability: str


class PaperWithLinks(BasePaper, total=False):
    citations: list[PaperInfo]
    references: list[PaperInfo]


class TitleMatchPaper(FullPaper, total=False):
    matchScore: int | float


class AutocompletePaper(TypedDict, total=False):
    id: str
    title: str
    authorsYear: str


class PaperAutocomplete(TypedDict, total=False):
    matches: list[AutocompletePaper]


class PaperBatch(TypedDict, total=False):
    ids: list[str]


class PaperRelevanceSearchBatch(TypedDict, total=False):
    total: int
    offset: int
    next: int
    data: list[FullPaper]


class PaperBulkSearchBatch(TypedDict, total=False):
    total: int
    token: str
    data: list[FullPaper]


class PaperMatch(TypedDict, total=False):
    data: list[TitleMatchPaper]


class Reference(TypedDict, total=False):
    contexts: list[str]
    intents: list[str]
    contextsWithIntent: list[dict[str, Any]]
    isInfluential: bool
    citedPaper: BasePaper


class ReferenceBatch(TypedDict, total=False):
    offset: int
    next: int
    data: list[Reference]


class Citation(TypedDict, total=False):
    contexts: list[str]
    intents: list[str]
    contextsWithIntent: list[dict[str, Any]]
    isInfluential: bool
    citingPaper: BasePaper


class CitationBatch(TypedDict, total=False):
    offset: int
    next: int
    data: list[Citation]


class AuthorInPaperWithPapers(AuthorInPaper, total=False):
    papers: list[BasePaper]


class AuthorWithPapers(AuthorInPaper, total=False):
    papers: list[BasePaper]


class AuthorIdList(TypedDict, total=False):
    ids: list[str]


class AuthorBatch(TypedDict, total=False):
    offset: int
    next: int
    data: list[AuthorInPaperWithPapers]


class AuthorPaperBatch(TypedDict, total=False):
    offset: int
    next: int
    data: list[PaperWithLinks]


class AuthorSearchBatch(TypedDict, total=False):
    total: int
    offset: int
    next: int
    data: list[AuthorWithPapers]


class Sentence(TypedDict, total=False):
    start: int
    end: int


class RefMention(TypedDict, total=False):
    start: int
    end: int
    matchedPaperCorpusId: str


class SnippetOffset(TypedDict, total=False):
    start: int
    end: int


class SnippetAnnotations(TypedDict, total=False):
    sentences: list[Sentence]
    refMentions: list[RefMention]


class Snippet(TypedDict, total=False):
    text: str
    snippetKind: str
    section: str
    snippetOffset: SnippetOffset
    annotations: SnippetAnnotations


class OpenAccessInfo(TypedDict, total=False):
    license: str
    status: str
    disclaimer: str


class SnippetPaper(TypedDict, total=False):
    corpusId: str
    title: str
    authors: list[str]
    openAccessInfo: OpenAccessInfo


class SnippetMatchItem(TypedDict, total=False):
    snippet: Snippet
    score: float
    paper: SnippetPaper


class SnippetMatch(TypedDict, total=False):
    data: list[SnippetMatchItem]
    retrievalVersion: str


SEMANTIC_SCHOLAR_ENDPOINTS = {
    "POST /author/batch": "get_authors_batch",
    "GET /author/search": "search_authors",
    "GET /author/{author_id}": "get_author",
    "GET /author/{author_id}/papers": "get_author_papers",
    "GET /paper/autocomplete": "autocomplete_paper",
    "POST /paper/batch": "get_papers_batch",
    "GET /paper/search": "search_papers",
    "GET /paper/search/bulk": "search_papers_bulk",
    "GET /paper/search/match": "match_paper_title",
    "GET /paper/{paper_id}": "get_paper_by_id",
    "GET /paper/{paper_id}/authors": "get_paper_authors",
    "GET /paper/{paper_id}/citations": "get_citations",
    "GET /paper/{paper_id}/references": "get_references",
    "GET /snippet/search": "search_snippets",
}

SEMANTIC_SCHOLAR_ENDPOINT_SCHEMAS = {
    "POST /author/batch": "list[AuthorWithPapers]",
    "GET /author/search": "AuthorSearchBatch",
    "GET /author/{author_id}": "AuthorWithPapers",
    "GET /author/{author_id}/papers": "AuthorPaperBatch",
    "GET /paper/autocomplete": "PaperAutocomplete",
    "POST /paper/batch": "list[FullPaper]",
    "GET /paper/search": "PaperRelevanceSearchBatch",
    "GET /paper/search/bulk": "PaperBulkSearchBatch",
    "GET /paper/search/match": "PaperMatch",
    "GET /paper/{paper_id}": "FullPaper",
    "GET /paper/{paper_id}/authors": "AuthorBatch",
    "GET /paper/{paper_id}/citations": "CitationBatch",
    "GET /paper/{paper_id}/references": "ReferenceBatch",
    "GET /snippet/search": "SnippetMatch",
}


class SemanticScholarClient:
    """Client for the Semantic Scholar Graph API.

    Parameters
    ----------
    api_key:
        Optional API key for higher rate limits (10 req/s vs 1 req/s).
        An explicit value takes priority over ``SEMANTIC_SCHOLAR_API_KEY``.
        Register at https://www.semanticscholar.org/product/api.
    base_url:
        Override the default API base URL.
    timeout:
        Default request timeout in seconds.
    delay_seconds:
        Minimum seconds between requests (default 2.0 to stay safely under
        the 1 req/s unauthenticated limit).  Set lower if you have an API key.
    cache_path:
        SQLite cache file path. Defaults to
        ``resources/cache/semantic_scholar_client_cache.sqlite``. Pass
        ``None`` to disable caching.
    force_refresh:
        Skip cache reads and fetch every request from Semantic Scholar. Fresh
        responses are still written to the configured cache. Defaults to
        ``False``.
    proxy_url:
        Optional proxy URL used for both HTTP and HTTPS requests.
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    CACHE_PATH = (
        Path(__file__).resolve().parents[2]
        / "resources"
        / "cache"
        / "semantic_scholar_client_cache.sqlite"
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = BASE_URL,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        delay_seconds: float = DEFAULT_DELAY_SECONDS,
        cache_path: str | Path | None = CACHE_PATH,
        force_refresh: bool = False,
        proxy_url: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._delay = delay_seconds
        self._cache = LocalSqliteCache(cache_path)
        self._force_refresh = force_refresh
        self._proxy_url = proxy_url
        self._last_request_time: float = 0.0
        self._throttle_lock = threading.Lock()

        load_dotenv_if_available()
        api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        if not api_key:
            logger.warning("SEMANTIC_SCHOLAR_API_KEY is not set")
        self._session_headers = {"Accept": "application/json"}

        if api_key:
            self._session_headers["x-api-key"] = api_key
            logger.info("Loaded Semantic Scholar API key")

        self._session = self.create_request_session()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def create_request_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(self._session_headers)
        if self._proxy_url:
            session.trust_env = False
            session.proxies.update(
                {
                    "http": self._proxy_url,
                    "https": self._proxy_url,
                }
            )
        return session

    def refresh_request_session(self) -> None:
        self._session.close()
        self._session = self.create_request_session()

    def close(self) -> None:
        self._session.close()
        self._cache.close()

    def __enter__(self) -> "SemanticScholarClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def wait(self) -> None:
        """Enforce the minimum inter-request delay (thread-safe)."""
        with self._throttle_lock:
            now = time.monotonic()
            earliest = self._last_request_time + self._delay
            wait = earliest - now
            self._last_request_time = max(now, earliest)
        if wait > 0:
            time.sleep(wait)

    def send_request_once(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> requests.Response:
        if method == "GET":
            return self._session.get(endpoint, **kwargs)
        if method == "POST":
            return self._session.post(endpoint, **kwargs)
        raise ValueError(f"Unsupported HTTP method: {method}")

    def send_request_with_backoff(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> requests.Response:
        """Send one request, retrying transient Semantic Scholar failures."""
        rate_limit_retries = 0
        server_error_retries = 0
        while rate_limit_retries <= HTTP_429_MAX_RETRIES:
            self.wait()
            try:
                response = self.send_request_once(method, endpoint, **kwargs)
            except requests.RequestException as exc:
                if server_error_retries >= HTTP_500_MAX_RETRIES:
                    raise
                server_error_retries += 1
                logger.warning(
                    "Semantic Scholar request failed for {} {}; retrying in {:.1f}s ({}/{}): {}",
                    method,
                    endpoint,
                    HTTP_500_RETRY_WAIT_SECONDS,
                    server_error_retries,
                    HTTP_500_MAX_RETRIES,
                    exc,
                )
                time.sleep(HTTP_500_RETRY_WAIT_SECONDS)
                continue

            if 500 <= response.status_code < 600 and server_error_retries < HTTP_500_MAX_RETRIES:
                server_error_retries += 1
                logger.warning(
                    "Semantic Scholar returned HTTP {} for {} {}; retrying in {:.1f}s ({}/{})",
                    response.status_code,
                    method,
                    endpoint,
                    HTTP_500_RETRY_WAIT_SECONDS,
                    server_error_retries,
                    HTTP_500_MAX_RETRIES,
                )
                time.sleep(HTTP_500_RETRY_WAIT_SECONDS)
                continue

            if response.status_code != 429:
                return response
            self.refresh_request_session()
            if rate_limit_retries == HTTP_429_MAX_RETRIES:
                break

            retry_after = response.headers.get("Retry-After")
            wait_seconds = HTTP_429_BACKOFF_SECONDS[
                min(rate_limit_retries, len(HTTP_429_BACKOFF_SECONDS) - 1)
            ]
            if retry_after:
                try:
                    wait_seconds = max(wait_seconds, float(retry_after))
                except ValueError:
                    pass

            logger.warning(
                "Semantic Scholar returned HTTP 429 for {} {}; " "retrying in {:.1f}s ({}/{})",
                method,
                endpoint,
                wait_seconds,
                rate_limit_retries + 1,
                HTTP_429_MAX_RETRIES,
            )
            rate_limit_retries += 1
            time.sleep(wait_seconds)

        logger.warning(
            "Semantic Scholar still rate-limited after {} retries for {} {}; waiting {:.1f}s before one final attempt",
            HTTP_429_MAX_RETRIES,
            method,
            endpoint,
            HTTP_429_COOLDOWN_SECONDS,
        )
        time.sleep(HTTP_429_COOLDOWN_SECONDS)
        self.wait()
        response = self.send_request_once(method, endpoint, **kwargs)
        if response.status_code == 429:
            self.refresh_request_session()
            raise SemanticScholarRateLimitError(
                f"Semantic Scholar returned HTTP 429 after {HTTP_429_MAX_RETRIES} retries and cooldown",
                response=response,
            )
        return response

        raise RuntimeError("Unreachable retry state")

    def request_get(self, path: str, params: Optional[dict] = None) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        endpoint = f"{self._base_url}{path}"
        cache_key = LocalSqliteCache.make_key("GET", endpoint, params=params)
        if not self._force_refresh:
            cache_hit, cached_response = self._cache.get(cache_key)
            if cache_hit:
                logger.debug("Semantic Scholar cache hit for {}", endpoint)
                return cached_response

            compatible_hit, compatible_response = self.get_legacy_paginated_cache(
                endpoint,
                params,
            )
            if compatible_hit:
                logger.debug("Semantic Scholar legacy cache hit for {}", endpoint)
                self._cache.store(
                    cache_key,
                    "GET",
                    endpoint,
                    compatible_response,
                    params=params,
                )
                return compatible_response

        response = self.send_request_with_backoff(
            "GET",
            endpoint,
            params=params,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        self._cache.store(cache_key, "GET", endpoint, data, params=params)
        return data

    def request_post(
        self,
        path: str,
        json: Any = None,
        params: Optional[dict] = None,
    ) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        endpoint = f"{self._base_url}{path}"
        cache_key = LocalSqliteCache.make_key("POST", endpoint, params=params, body=json)
        if not self._force_refresh:
            cache_hit, cached_response = self._cache.get(cache_key)
            if cache_hit:
                logger.debug("Semantic Scholar cache hit for {}", endpoint)
                return cached_response

        response = self.send_request_with_backoff(
            "POST",
            endpoint,
            json=json,
            params=params,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        self._cache.store(
            cache_key,
            "POST",
            endpoint,
            data,
            params=params,
            body=json,
        )
        return data

    def get_legacy_paginated_cache(
        self,
        endpoint: str,
        params: dict[str, Any],
    ) -> tuple[bool, Any]:
        """Build a larger page from smaller legacy cached pages."""
        limit = params.get("limit")
        offset = params.get("offset")
        if not isinstance(limit, int) or not isinstance(offset, int):
            return False, None

        for legacy_limit in LEGACY_CACHE_PAGE_LIMITS:
            if (
                limit <= legacy_limit
                or limit % legacy_limit != 0
                or offset % legacy_limit != 0
            ):
                continue

            page_offsets = range(offset, offset + limit, legacy_limit)
            stitched_data: list[Any] = []
            first_page: dict[str, Any] | None = None
            next_offset: int | None = None
            valid = True

            for page_offset in page_offsets:
                page_params = dict(params)
                page_params["limit"] = legacy_limit
                page_params["offset"] = page_offset
                page_key = LocalSqliteCache.make_key("GET", endpoint, params=page_params)
                cache_hit, cached_page = self._cache.get(page_key)
                if not cache_hit or not isinstance(cached_page, dict):
                    valid = False
                    break

                page_data = cached_page.get("data")
                if not isinstance(page_data, list):
                    valid = False
                    break

                if first_page is None:
                    first_page = cached_page
                stitched_data.extend(page_data)

                page_next = cached_page.get("next")
                if page_next is None:
                    next_offset = None
                    break
                if len(page_data) != legacy_limit:
                    valid = False
                    break
                if page_next != page_offset + legacy_limit:
                    valid = False
                    break

                next_offset = page_next

            if not valid or first_page is None:
                continue
            if len(stitched_data) < limit and next_offset is not None:
                continue

            stitched_page = dict(first_page)
            stitched_page["offset"] = offset
            stitched_page["data"] = stitched_data[:limit]
            stitched_page["next"] = next_offset
            return True, stitched_page

        return False, None

    # ------------------------------------------------------------------
    # Author data endpoints
    # ------------------------------------------------------------------

    def get_authors_batch(
        self,
        author_ids: list[str],
        *,
        fields: Optional[list[str] | str] = None,
    ) -> list[AuthorWithPapers | None]:
        """POST /author/batch: fetch details for up to 1,000 authors."""
        return self.request_post(
            "/author/batch",
            json={"ids": author_ids},
            params={"fields": comma_join(fields)},
        )

    def search_authors(
        self,
        query: str,
        *,
        fields: Optional[list[str] | str] = None,
        offset: int = 0,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> AuthorSearchBatch:
        """GET /author/search: search for authors by name."""
        return self.request_get(
            "/author/search",
            {
                "query": query,
                "fields": comma_join(fields),
                "offset": offset,
                "limit": limit,
            },
        )

    def get_author(
        self,
        author_id: str,
        *,
        fields: Optional[list[str] | str] = None,
    ) -> AuthorWithPapers:
        """GET /author/{author_id}: fetch one author by Semantic Scholar ID."""
        return self.request_get(
            f"/author/{path_param(author_id)}",
            {"fields": comma_join(fields)},
        )

    def get_author_papers(
        self,
        author_id: str,
        *,
        fields: Optional[list[str] | str] = None,
        publication_date_or_year: Optional[str] = None,
        offset: int = 0,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> AuthorPaperBatch:
        """GET /author/{author_id}/papers: fetch papers written by an author."""
        # Drop abstract field to save bandwidth
        return self.request_get(
            f"/author/{path_param(author_id)}/papers",
            {
                "fields": comma_join(fields) if fields else comma_join(_DEFAULT_PAPER_FIELDS),
                "publicationDateOrYear": publication_date_or_year,
                "offset": offset,
                "limit": limit,
            },
        )

    def get_all_papers_of_author(
        self,
        author_id: str,
        *,
        fields: Optional[list[str] | str] = None,
        publication_date_or_year: Optional[str] = None,
        offset: int = 0,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> list[PaperWithLinks]:
        """GET /author/{author_id}/papers: fetch all papers written by an author."""
        papers: list[PaperWithLinks] = []
        while True:
            batch = self.get_author_papers(
                author_id,
                fields=fields,
                publication_date_or_year=publication_date_or_year,
                offset=offset,
                limit=limit,
            )
            data = batch.get("data") or []
            papers.extend(data)

            next_offset = batch.get("next")
            if not data or next_offset is None:
                break
            offset = next_offset
        return papers

    # ------------------------------------------------------------------
    # Paper data endpoints
    # ------------------------------------------------------------------

    def autocomplete_paper(self, query: str) -> PaperAutocomplete:
        """GET /paper/autocomplete: suggest paper query completions."""
        return self.request_get("/paper/autocomplete", {"query": query})

    def search_papers(
        self,
        query: str,
        *,
        fields: Optional[list[str] | str] = None,
        publication_types: Optional[list[str] | str] = None,
        open_access_pdf: bool = False,
        min_citation_count: Optional[int | str] = None,
        publication_date_or_year: Optional[str] = None,
        year: Optional[str] = None,
        venue: Optional[list[str] | str] = None,
        fields_of_study: Optional[list[str] | str] = None,
        offset: int = 0,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> PaperRelevanceSearchBatch:
        """GET /paper/search: relevance-ranked paper search."""
        params = paper_search_params(
            query=query,
            fields=fields,
            publication_types=publication_types,
            open_access_pdf=open_access_pdf,
            min_citation_count=min_citation_count,
            publication_date_or_year=publication_date_or_year,
            year=year,
            venue=venue,
            fields_of_study=fields_of_study,
        )
        params["offset"] = offset
        params["limit"] = limit
        return self.request_get("/paper/search", params)

    def search_papers_bulk(
        self,
        query: str,
        *,
        token: Optional[str] = None,
        fields: Optional[list[str] | str] = None,
        sort: Optional[str] = None,
        publication_types: Optional[list[str] | str] = None,
        open_access_pdf: bool = False,
        min_citation_count: Optional[int | str] = None,
        publication_date_or_year: Optional[str] = None,
        year: Optional[str] = None,
        venue: Optional[list[str] | str] = None,
        fields_of_study: Optional[list[str] | str] = None,
    ) -> PaperBulkSearchBatch:
        """GET /paper/search/bulk: bulk paper search with continuation tokens."""
        params = paper_search_params(
            query=query,
            fields=fields,
            publication_types=publication_types,
            open_access_pdf=open_access_pdf,
            min_citation_count=min_citation_count,
            publication_date_or_year=publication_date_or_year,
            year=year,
            venue=venue,
            fields_of_study=fields_of_study,
        )
        params["token"] = token
        params["sort"] = sort
        return self.request_get("/paper/search/bulk", params)

    def match_paper_title(
        self,
        query: str,
        *,
        fields: Optional[list[str] | str] = None,
        publication_types: Optional[list[str] | str] = None,
        open_access_pdf: bool = False,
        min_citation_count: Optional[int | str] = None,
        publication_date_or_year: Optional[str] = None,
        year: Optional[str] = None,
        venue: Optional[list[str] | str] = None,
        fields_of_study: Optional[list[str] | str] = None,
        return_none_on_404: bool = False,
    ) -> Optional[PaperMatch]:
        """GET /paper/search/match: fetch the closest title match."""
        params = paper_search_params(
            query=query,
            fields=fields,
            publication_types=publication_types,
            open_access_pdf=open_access_pdf,
            min_citation_count=min_citation_count,
            publication_date_or_year=publication_date_or_year,
            year=year,
            venue=venue,
            fields_of_study=fields_of_study,
        )
        clean_params = {k: v for k, v in params.items() if v is not None}
        endpoint = f"{self._base_url}/paper/search/match"
        cache_key = LocalSqliteCache.make_key("GET", endpoint, params=clean_params)
        cache_hit, cached_response = self._cache.get(cache_key)
        if cache_hit:
            return cached_response

        response = self.send_request_with_backoff(
            "GET",
            endpoint,
            params=clean_params,
            timeout=self._timeout,
        )
        if response.status_code == 404 and return_none_on_404:
            self._cache.store(cache_key, "GET", endpoint, None, params=clean_params)
            return None

        response.raise_for_status()
        data = response.json()
        self._cache.store(cache_key, "GET", endpoint, data, params=clean_params)
        return data

    # ------------------------------------------------------------------
    # Snippet text endpoint
    # ------------------------------------------------------------------

    def search_snippets(
        self,
        query: str,
        *,
        fields: Optional[list[str] | str] = None,
        paper_ids: Optional[list[str] | str] = None,
        authors: Optional[list[str] | str] = None,
        min_citation_count: Optional[int | str] = None,
        inserted_before: Optional[str] = None,
        publication_date_or_year: Optional[str] = None,
        year: Optional[str] = None,
        venue: Optional[list[str] | str] = None,
        fields_of_study: Optional[list[str] | str] = None,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> SnippetMatch:
        """GET /snippet/search: search paper text snippets."""
        return self.request_get(
            "/snippet/search",
            {
                "query": query,
                "fields": comma_join(fields),
                "paperIds": comma_join(paper_ids),
                "authors": comma_join(authors),
                "minCitationCount": min_citation_count,
                "insertedBefore": inserted_before,
                "publicationDateOrYear": publication_date_or_year,
                "year": year,
                "venue": comma_join(venue),
                "fieldsOfStudy": comma_join(fields_of_study),
                "limit": limit,
            },
        )

    # ------------------------------------------------------------------
    # Paper lookup
    # ------------------------------------------------------------------

    def get_paper_by_id(
        self,
        paper_id: str,
        *,
        fields: Optional[list[str]] = None,
    ) -> FullPaper:
        """Fetch metadata for a single paper.

        Parameters
        ----------
        paper_id:
            Any S2-accepted ID: bare arXiv ID (``"2302.13971"``), prefixed
            (``"arXiv:2302.13971"``), DOI, or 40-char S2 hex ID.
        fields:
            List of fields to return.  Defaults to :data:`_DEFAULT_PAPER_FIELDS`.

        Returns
        -------
        dict
            Raw API response for the paper.
        """
        s2_id = normalize_paper_id(paper_id)
        f = ",".join(fields or _DEFAULT_PAPER_FIELDS)
        data = self.request_get(f"/paper/{path_param(s2_id)}", {"fields": f})
        logger.debug("get_paper_by_id({}) → {}", s2_id, data.get("paperId"))
        return data

    def get_papers_batch(
        self,
        paper_ids: list[str],
        *,
        fields: Optional[list[str]] = None,
    ) -> list[FullPaper | None]:
        """Batch-fetch metadata for up to 500 papers in a single request.

        Parameters
        ----------
        paper_ids:
            List of S2-accepted IDs (arXiv IDs, DOIs, or S2 hex IDs).
        fields:
            Fields to include in each paper object.

        Returns
        -------
        list[dict]
            One entry per input ID in the same order (may contain ``null``
            for IDs not found in the S2 corpus).
        """
        ids = [normalize_paper_id(pid) for pid in paper_ids]
        f = ",".join(fields or _DEFAULT_PAPER_FIELDS)
        data = self.request_post(
            "/paper/batch",
            json={"ids": ids},
            params={"fields": f},
        )
        logger.info("get_papers_batch({} ids) → {} results", len(ids), len(data))
        return data

    # ------------------------------------------------------------------
    # References (papers that a given paper cites)
    # ------------------------------------------------------------------

    def get_references(
        self,
        paper_id: str,
        *,
        fields: Optional[list[str]] = None,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> ReferenceBatch:
        """Fetch one page of references from a paper.

        Parameters
        ----------
        paper_id:
            S2-accepted paper ID.
        fields:
            Fields to include on each ``citedPaper`` object.  Defaults to
            :data:`_DEFAULT_REF_FIELDS` (all standard fields prefixed with
            ``citedPaper.``).
        limit:
            Number of results per page (max 1 000).
        offset:
            Pagination offset.

        Returns
        -------
        dict
            ``{"offset": int, "next": int | None, "data": [{"citedPaper": {...}}]}``
        """
        s2_id = normalize_paper_id(paper_id)
        f = ",".join(fields or _DEFAULT_REF_FIELDS)
        data = self.request_get(
            f"/paper/{path_param(s2_id)}/references",
            {"fields": f, "limit": limit, "offset": offset},
        )
        logger.debug(
            "get_references({}, offset={}) → {} entries",
            s2_id,
            offset,
            len(data.get("data") or []),
        )
        return data

    def get_all_references(
        self,
        paper_id: str,
        *,
        fields: Optional[list[str]] = None,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> list[dict]:
        """Fetch the complete reference list, auto-paginating as needed.

        Parameters
        ----------
        paper_id:
            S2-accepted paper ID or bare arXiv ID.
        fields:
            Fields to include on each ``citedPaper`` object.
        max_results:
            Hard cap on total references to return.

        Returns
        -------
        list[dict]
            Each entry is ``{"citedPaper": {...}}``.
        """
        refs: list[dict] = []
        offset = 0
        page_size = min(DEFAULT_PAGE_LIMIT, max_results)

        while len(refs) < max_results:
            page = self.get_references(
                paper_id,
                fields=fields,
                limit=page_size,
                offset=offset,
            )
            batch = page.get("data") or []
            refs.extend(batch)

            next_offset = page.get("next")
            if not batch or next_offset is None:
                break
            offset = next_offset

        total = len(refs[:max_results])
        # logger.info(f"get_all_references({paper_id}) → {total} refs")
        return refs[:max_results]

    # ------------------------------------------------------------------
    # Citations (papers that cite a given paper)
    # ------------------------------------------------------------------

    def get_citations(
        self,
        paper_id: str,
        *,
        fields: Optional[list[str]] = None,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> CitationBatch:
        """Fetch one page of papers that cite the given paper.

        Returns
        -------
        dict
            ``{"offset": int, "next": int | None, "data": [{"citingPaper": {...}}]}``
        """
        s2_id = normalize_paper_id(paper_id)
        f = ",".join(fields or _DEFAULT_CITE_FIELDS)
        data = self.request_get(
            f"/paper/{path_param(s2_id)}/citations",
            {"fields": f, "limit": limit, "offset": offset},
        )
        logger.debug(
            "get_citations({}, offset={}) → {} entries",
            s2_id,
            offset,
            len(data.get("data") or []),
        )
        return data

    def get_all_citations(
        self,
        paper_id: str,
        *,
        fields: Optional[list[str]] = None,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> list[dict]:
        """Fetch all papers citing the given paper, auto-paginating.

        Returns
        -------
        list[dict]
            Each entry is ``{"citingPaper": {...}}``.
        """
        cites: list[dict] = []
        offset = 0
        page_size = min(DEFAULT_PAGE_LIMIT, max_results)

        while len(cites) < max_results:
            page = self.get_citations(
                paper_id,
                fields=fields,
                limit=page_size,
                offset=offset,
            )
            batch = page.get("data") or []
            cites.extend(batch)

            next_offset = page.get("next")
            if not batch or next_offset is None:
                break
            offset = next_offset

        total = len(cites[:max_results])
        # logger.info(f"get_all_citations({paper_id}) → {total} citations")
        return cites[:max_results]

    def get_paper_with_citations_and_references(
        self,
        paper_id: str,
        *,
        fields: Optional[list[str]] = None,
        max_results: int = DEFAULT_MAX_RESULTS,
        include_citations: bool = True,
        include_references: bool = True,
    ) -> tuple[FullPaper, list[Citation], list[Reference]]:
        """Fetch a paper plus up to ``max_results`` selected neighbors.

        Requests citations/references as embedded fields on a single
        ``GET /paper/{paper_id}`` call instead of three separate requests.
        Falls back independently, per side, to the paginated
        ``get_all_citations``/``get_all_references`` when the embedded list
        is shorter than ``citationCount``/``referenceCount`` (Semantic
        Scholar silently caps embedded lists well before the documented
        10 MB response limit). Returns at most ``max_results`` per side,
        same as the paginated methods — this is "up to max_results", not a
        full-list guarantee.

        ``fields`` must be flat paper fields (e.g. ``["title", "authors"]``);
        do not pass already-nested fields like ``"citations.title"``.

        Setting either include flag to false omits that side's count and
        embedded fields from the request and returns an empty list for it.
        """
        effective_max_results = max(0, max_results)
        combined_fields = build_combined_expansion_fields(
            fields,
            include_citations=include_citations,
            include_references=include_references,
        )
        paper = self.get_paper_by_id(paper_id, fields=combined_fields)
        resolved_id = paper.get("paperId") or paper_id

        citations = (
            resolve_expansion_side(
                paper,
                embedded_key="citations",
                count_key="citationCount",
                envelope_key="citingPaper",
                paper_id=resolved_id,
                fetch_all=self.get_all_citations,
                fields=fields,
                max_results=effective_max_results,
            )
            if include_citations
            else []
        )
        references = (
            resolve_expansion_side(
                paper,
                embedded_key="references",
                count_key="referenceCount",
                envelope_key="citedPaper",
                paper_id=resolved_id,
                fetch_all=self.get_all_references,
                fields=fields,
                max_results=effective_max_results,
            )
            if include_references
            else []
        )
        return paper, citations, references

    def get_papers_with_citations_and_references_batch(
        self,
        paper_ids: list[str],
        *,
        fields: Optional[list[str]] = None,
        max_results: int = DEFAULT_MAX_RESULTS,
        include_citations: bool = True,
        include_references: bool = True,
    ) -> dict[str, tuple[Optional[FullPaper], list[Citation], list[Reference]]]:
        """Batch version of ``get_paper_with_citations_and_references``.

        ``include_citations=False`` drops ``citationCount``/``citations.*``
        from the request entirely (not just from the returned tuple) and
        the ``citations`` half of every returned entry is always ``[]`` --
        use this when only ``references`` is needed, to keep requests and
        responses smaller.

        ``include_references=False`` provides the symmetric citations-only
        behavior.

        Fetches up to 500 papers' metadata plus embedded citations/references
        in one request. The embedded ``citations``/``references`` budget is
        shared across the *whole* batch response, not per paper: when a
        batch contains a heavily-cited "hub" paper, that paper -- or,
        arbitrarily, some other paper in the batch; the winner is not
        determined by request order or by citation count -- can consume the
        entire budget, leaving every other paper's embedded list empty, with
        no error (``HTTP 200``) to signal it.

        This recovers from that by re-batching whichever papers came back
        incomplete (on either side), repeating until a round makes no
        further progress, then resolving any still-incomplete papers
        individually via the paginated ``get_all_citations``/
        ``get_all_references`` -- reusing the metadata already fetched for
        each, without a redundant ``get_paper_by_id`` call.

        In the common case (mostly ordinary papers, at most a couple of
        hubs) this converges in roughly (number of hubs present) + 1 batch
        calls, since a batch containing no hub paper resolves everyone in
        one round. A batch that is *entirely* hub papers plateaus after one
        round and each paper falls back individually, same as calling
        ``get_paper_with_citations_and_references`` on each -- but without
        the wasted initial batch attempt's citations being thrown away for
        whichever paper happened to win it.

        Returns a dict keyed by the *original* input ID (whatever ID form it
        was given in), mapping to ``(paper, citations, references)``.
        ``paper`` is ``None`` when Semantic Scholar could not resolve that ID
        at all -- a different failure from truncation, not retried here.
        Assumes distinct ``paper_ids``; duplicates collapse to one entry.
        """
        effective_max_results = max(0, max_results)
        combined_fields = build_combined_expansion_fields(
            fields,
            include_citations=include_citations,
            include_references=include_references,
        )

        resolved: dict[
            str, tuple[Optional[FullPaper], list[Citation], list[Reference]]
        ] = {}
        last_seen: dict[str, FullPaper] = {}
        remaining = list(paper_ids)

        while remaining:
            batch = self.get_papers_batch(remaining, fields=combined_fields)

            if len(batch) != len(remaining):
                # get_papers_batch() is documented to return one entry per
                # input ID in the same order; if a malformed/short response
                # ever violates that, don't guess at a partial correspondence
                # -- treat the whole round as incomplete and let it either
                # succeed on retry or plateau into the per-paper fallback
                # below, which doesn't depend on this ordering guarantee.
                logger.warning(
                    "get_papers_batch returned {} results for {} requested ids; "
                    "treating this round as incomplete",
                    len(batch),
                    len(remaining),
                )
                still_starved = list(remaining)
            else:
                still_starved = []

                for paper_id, paper in zip(remaining, batch):
                    if paper is None:
                        resolved[paper_id] = (None, [], [])
                        continue

                    last_seen[paper_id] = paper
                    if include_citations:
                        citations_ok, citations = paper_side_status(
                            paper,
                            count_key="citationCount",
                            embedded_key="citations",
                            max_results=effective_max_results,
                        )
                    else:
                        citations_ok, citations = True, []
                    if include_references:
                        references_ok, references = paper_side_status(
                            paper,
                            count_key="referenceCount",
                            embedded_key="references",
                            max_results=effective_max_results,
                        )
                    else:
                        references_ok, references = True, []

                    if citations_ok and references_ok:
                        resolved[paper_id] = (
                            paper,
                            [{"citingPaper": item} for item in citations],
                            [{"citedPaper": item} for item in references],
                        )
                    else:
                        still_starved.append(paper_id)

            if len(still_starved) == len(remaining):
                logger.debug(
                    "get_papers_with_citations_and_references_batch: {} paper(s) "
                    "still incomplete after batching plateaued; resolving individually",
                    len(still_starved),
                )
                remaining = still_starved
                break
            remaining = still_starved

        for paper_id in remaining:
            paper = last_seen.get(paper_id)
            if paper is None:
                # This ID never appeared in any well-formed batch response
                # (only possible after a length-mismatch above) -- fetch its
                # metadata directly rather than leaving it unresolved.
                paper = self.get_paper_by_id(paper_id, fields=combined_fields)
            resolved_id = paper.get("paperId") or paper_id
            if include_citations:
                citations = resolve_expansion_side(
                    paper,
                    embedded_key="citations",
                    count_key="citationCount",
                    envelope_key="citingPaper",
                    paper_id=resolved_id,
                    fetch_all=self.get_all_citations,
                    fields=fields,
                    max_results=effective_max_results,
                )
            else:
                citations = []
            references = (
                resolve_expansion_side(
                    paper,
                    embedded_key="references",
                    count_key="referenceCount",
                    envelope_key="citedPaper",
                    paper_id=resolved_id,
                    fetch_all=self.get_all_references,
                    fields=fields,
                    max_results=effective_max_results,
                )
                if include_references
                else []
            )
            resolved[paper_id] = (paper, citations, references)

        return resolved

    # ------------------------------------------------------------------
    # Convenience: enrich PaperMeta list with references
    # ------------------------------------------------------------------

    def enrich_with_references(
        self,
        papers: "list[Any]",
        *,
        fields: Optional[list[str]] = None,
        max_refs_per_paper: int = 500,
    ) -> list[dict]:
        """Attach reference lists to a collection of ``PaperMeta`` objects.

        Parameters
        ----------
        papers:
            List of :class:`~src.clients.arxiv_client.PaperMeta` (or any
            objects with an ``arxiv_id`` attribute or ``"arxiv_id"`` key).
        fields:
            Fields to include on each cited paper.
        max_refs_per_paper:
            Maximum number of references to fetch per paper.

        Returns
        -------
        list[dict]
            Each entry is a dict with all original paper fields plus a
            ``"references"`` key containing the list of cited papers.
        """
        enriched: list[dict] = []
        for paper in papers:
            # Support both dataclass/object and plain dict
            if isinstance(paper, dict):
                meta = paper.copy()
                arxiv_id = meta.get("arxiv_id", "")
            else:
                meta = paper.to_dict() if hasattr(paper, "to_dict") else vars(paper)
                arxiv_id = getattr(paper, "arxiv_id", "")

            if not arxiv_id:
                logger.warning("Skipping paper with no arxiv_id: {}", meta.get("title"))
                meta["references"] = []
                enriched.append(meta)
                continue

            s2_id = normalize_paper_id(arxiv_id)
            logger.info(
                "Fetching references for {} ({}) …",
                arxiv_id,
                meta.get("title", "")[:60],
            )
            try:
                refs = self.get_all_references(
                    s2_id,
                    fields=fields,
                    max_results=max_refs_per_paper,
                )
                # Flatten: keep only the inner citedPaper dicts
                meta["references"] = [r["citedPaper"] for r in refs if r.get("citedPaper")]
            except requests.HTTPError as exc:
                logger.warning("Failed to fetch references for {}: {}", arxiv_id, exc)
                meta["references"] = []

            enriched.append(meta)

        return enriched

    # ------------------------------------------------------------------
    # Paper lookup
    # ------------------------------------------------------------------

    def search_by_title(
        self,
        title: str,
        *,
        fields: Optional[list[str]] = None,
        publication_types: Optional[list[str]] = None,
        open_access_pdf: bool = False,
        min_citation_count: Optional[int] = None,
        publication_date_or_year: Optional[str] = None,
        year: Optional[str] = None,
        venue: Optional[list[str]] = None,
        fields_of_study: Optional[list[str]] = None,
    ) -> Optional[dict]:
        """Find the single best-matching paper for a given title query.

        Calls ``GET /paper/search/match``, which returns the closest title match
        (plus its ``matchScore``) or a 404 if nothing is found.

        Parameters
        ----------
        title:
            Plain-text title query.  No special query syntax is supported.
        fields:
            Fields to return on the matched paper.  Defaults to
            :data:`_DEFAULT_PAPER_FIELDS`.  ``paperId`` and ``title`` are always
            returned regardless.
        publication_types:
            Restrict to papers with any of these publication types, e.g.
            ``["Conference", "JournalArticle"]``.
        open_access_pdf:
            If ``True``, restrict to papers with a publicly available PDF.
        min_citation_count:
            Only return a match if it has at least this many citations.
        publication_date_or_year:
            Date/year range string in ``<start>:<end>`` format, e.g.
            ``"2020:2024"`` or ``"2022-06:"``.
        year:
            Publication year or range, e.g. ``"2023"`` or ``"2020-2024"``.
        venue:
            Restrict to papers from these venues, e.g. ``["ACL", "EMNLP"]``.
        fields_of_study:
            Restrict to these S2 fields of study, e.g. ``["Computer Science"]``.

        Returns
        -------
        dict or None
            The matched paper dict (includes ``matchScore``), or ``None`` if no
            match was found (404).

        Raises
        ------
        requests.HTTPError
            For any non-404 HTTP error (e.g. 400 bad parameters, 5xx).

        Examples
        --------
        >>> client = SemanticScholarClient()
        >>> paper = client.search_by_title(
        ...     "Attention Is All You Need",
        ...     fields=["title", "authors", "year", "citationCount"],
        ...     fields_of_study=["Computer Science"],
        ... )
        >>> if paper:
        ...     print(paper["title"], paper["matchScore"])
        """
        data = self.match_paper_title(
            title,
            fields=fields or _DEFAULT_PAPER_FIELDS,
            publication_types=publication_types,
            open_access_pdf=open_access_pdf,
            min_citation_count=min_citation_count,
            publication_date_or_year=publication_date_or_year,
            year=year,
            venue=venue,
            fields_of_study=fields_of_study,
            return_none_on_404=True,
        )
        if data is None:
            logger.debug("search_by_title({!r}) → no match found", title)
            return None

        # The endpoint wraps the result in a {"data": [...]} envelope;
        # unwrap to return the single paper dict directly.
        papers = data.get("data") or []
        if not papers:
            logger.debug("search_by_title({!r}) → empty data list", title)
            return None

        paper = papers[0]
        # logger.debug(
        #     f"search_by_title({title}) → {paper.get('paperId')} (score={paper.get('matchScore', 0.0)})",
        # )
        return paper

    # ------------------------------------------------------------------
    # Authors (authors of a given paper)
    # ------------------------------------------------------------------

    def get_paper_authors(
        self,
        paper_id: str,
        *,
        fields: Optional[list[str]] = None,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> AuthorBatch:
        """Fetch one page of authors for a given paper.

        Parameters
        ----------
        paper_id:
            Any S2-accepted ID: bare arXiv ID, ``arXiv:``-prefixed, DOI, or
            40-char S2 hex ID.
        fields:
            Fields to return for each author.  ``authorId`` and ``name`` are
            always returned.  To include the author's paper list use
            ``"papers"``; subfields are requested with dot notation, e.g.
            ``"papers.year"``.  Defaults to :data:`_DEFAULT_AUTHOR_FIELDS`.
        limit:
            Number of authors per page (max 1 000).
        offset:
            Pagination offset.

        Returns
        -------
        dict
            ``{"offset": int, "next": int | None, "data": [{"authorId": ..., ...}]}``
        """
        s2_id = normalize_paper_id(paper_id)
        f = ",".join(fields or _DEFAULT_AUTHOR_FIELDS)
        data = self.request_get(
            f"/paper/{path_param(s2_id)}/authors",
            {"fields": f, "limit": limit, "offset": offset},
        )
        logger.debug(
            "get_paper_authors({}, offset={}) → {} entries",
            s2_id,
            offset,
            len(data.get("data") or []),
        )
        return data

    def get_all_authors_of_paper(
        self,
        paper_id: str,
        *,
        fields: Optional[list[str]] = None,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> list[dict]:
        """Fetch the complete author list for a paper, auto-paginating as needed.

        Parameters
        ----------
        paper_id:
            Any S2-accepted ID or bare arXiv ID.
        fields:
            Fields to return for each author.  Defaults to
            :data:`_DEFAULT_AUTHOR_FIELDS`.
        max_results:
            Hard cap on total authors to return.

        Returns
        -------
        list[dict]
            Each entry is an author dict, e.g.
            ``{"authorId": "1741101", "name": "Oren Etzioni", ...}``.

        Examples
        --------
        >>> client = SemanticScholarClient()
        >>> authors = client.get_all_authors_of_paper(
        ...     "arXiv:1706.03762",
        ...     fields=["name", "affiliations", "hIndex"],
        ... )
        >>> for author in authors:
        ...     print(author["name"], author.get("hIndex"))
        """
        authors: list[dict] = []
        offset = 0
        page_size = min(DEFAULT_PAGE_LIMIT, max_results)

        while len(authors) < max_results:
            page = self.get_paper_authors(
                paper_id,
                fields=fields,
                limit=page_size,
                offset=offset,
            )
            batch = page.get("data") or []
            authors.extend(batch)

            next_offset = page.get("next")
            if not batch or next_offset is None:
                break
            offset = next_offset

        total = len(authors[:max_results])
        # logger.info(f"get_all_authors_of_paper({paper_id}) → {total} authors")
        return authors[:max_results]


EndpointSmokeTest = tuple[str, str, Callable[[], Any]]


def build_endpoint_smoke_tests(client: SemanticScholarClient) -> list[EndpointSmokeTest]:
    """Build low-volume live checks for every Semantic Scholar endpoint."""
    paper_id = "649def34f8be52c8b66281af98ae884c09aef38b"
    author_id = "1741101"
    title_query = "Construction of the Literature Graph in Semantic Scholar"
    snippet_query = "literature graph is a property graph"

    return [
        (
            "POST /author/batch",
            "get_authors_batch",
            lambda: client.get_authors_batch([author_id], fields=["name"]),
        ),
        (
            "GET /author/search",
            "search_authors",
            lambda: client.search_authors("Oren Etzioni", fields=["name"], limit=1),
        ),
        (
            "GET /author/{author_id}",
            "get_author",
            lambda: client.get_author(author_id, fields=["name"]),
        ),
        (
            "GET /author/{author_id}/papers",
            "get_author_papers",
            lambda: client.get_author_papers(
                author_id,
                fields=["title"],
                limit=1,
            ),
        ),
        (
            "GET /paper/autocomplete",
            "autocomplete_paper",
            lambda: client.autocomplete_paper("construction of the literature graph"),
        ),
        (
            "POST /paper/batch",
            "get_papers_batch",
            lambda: client.get_papers_batch([paper_id], fields=["title"]),
        ),
        (
            "GET /paper/search",
            "search_papers",
            lambda: client.search_papers(
                title_query,
                fields=["title"],
                limit=1,
            ),
        ),
        (
            "GET /paper/search/bulk",
            "search_papers_bulk",
            lambda: client.search_papers_bulk(
                f'"{title_query}"',
                fields=["title"],
            ),
        ),
        (
            "GET /paper/search/match",
            "match_paper_title",
            lambda: client.match_paper_title(title_query, fields=["title"]),
        ),
        (
            "GET /paper/{paper_id}",
            "get_paper_by_id",
            lambda: client.get_paper_by_id(paper_id, fields=["title"]),
        ),
        (
            "GET /paper/{paper_id}/authors",
            "get_paper_authors",
            lambda: client.get_paper_authors(paper_id, fields=["name"], limit=1),
        ),
        (
            "GET /paper/{paper_id}/citations",
            "get_citations",
            lambda: client.get_citations(paper_id, fields=["title"], limit=1),
        ),
        (
            "GET /paper/{paper_id}/references",
            "get_references",
            lambda: client.get_references(paper_id, fields=["title"], limit=1),
        ),
        (
            "GET /snippet/search",
            "search_snippets",
            lambda: client.search_snippets(snippet_query, limit=1),
        ),
    ]


def summarize_response(data: Any) -> str:
    """Return a short summary for endpoint smoke-test output."""
    if isinstance(data, list):
        return f"list length={len(data)}"
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return f"dict data length={len(data['data'])}"
        if isinstance(data.get("matches"), list):
            return f"dict matches length={len(data['matches'])}"
        keys = ",".join(list(data)[:5])
        return f"dict keys={keys}"
    if data is None:
        return "none"
    return type(data).__name__


def run_endpoint_smoke_tests(
    client: SemanticScholarClient,
    *,
    max_retries: int = 2,
) -> bool:
    """Run one live request against every swagger endpoint."""
    tests = build_endpoint_smoke_tests(client)
    failures = 0

    print(f"Testing {len(tests)} Semantic Scholar endpoints")
    for endpoint, method_name, call_endpoint in tests:
        attempt = 0
        while True:
            try:
                data = call_endpoint()
            except requests.HTTPError as exc:
                response = exc.response
                status_code = getattr(response, "status_code", "unknown")
                if status_code == 429 and attempt < max_retries:
                    retry_after = response.headers.get("Retry-After", "10")
                    try:
                        wait_seconds = float(retry_after)
                    except ValueError:
                        wait_seconds = 10.0
                    attempt += 1
                    print(
                        f"RETRY {endpoint} {method_name}: HTTP 429, " f"waiting {wait_seconds:.1f}s"
                    )
                    time.sleep(wait_seconds)
                    continue

                failures += 1
                text = getattr(response, "text", "")
                message = text[:200] if text else str(exc)
                print(f"FAIL {endpoint} {method_name}: " f"HTTP {status_code} {message}")
            except requests.RequestException as exc:
                failures += 1
                print(f"FAIL {endpoint} {method_name}: request error {exc}")
            except Exception as exc:
                failures += 1
                print(f"FAIL {endpoint} {method_name}: {type(exc).__name__}: {exc}")
            else:
                print(f"PASS {endpoint} {method_name}: {summarize_response(data)}")
            break

    print(f"Completed with {failures} failure(s)")
    return failures == 0


def main() -> None:
    """Run live smoke tests for all Semantic Scholar API endpoints."""
    client = SemanticScholarClient(delay_seconds=2.0, cache_path=None)
    if not run_endpoint_smoke_tests(client):
        raise SystemExit(1)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()
