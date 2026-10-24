"""
arXiv crawler / client.

Example commands:
```
python -m src.clients.arxiv_client
```

Directly queries the arXiv Atom API (https://export.arxiv.org/api/query) using
``feedparser`` and ``requests``, providing structured metadata extraction,
incremental saving, and category/date helpers.

Usage::

    client = ArxivOnlineClient()

    # Fetch up to 200 recent RAG papers
    papers = client.search("retrieval augmented generation", max_results=200)

    # Fetch a single paper by arXiv ID
    paper = client.get_paper("2005.11401")

    # Crawl a category in a date range and stream results to JSONL
    client.crawl(
        category="cs.CL",
        date_from="2024-01-01",
        date_to="2024-03-31",
        max_results=1000,
        output_path="data/cs_cl_2024_q1.jsonl",
    )
"""

from __future__ import annotations

import orjson
import os
import re
import threading
import time
from calendar import timegm
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Iterator, Optional, TypeVar
from urllib.parse import urlencode

import feedparser
import requests
from loguru import logger
from pydantic import BaseModel, Field

from src.clients.local_json_cache import LocalJsonCache
from src.utils.string_matching import score_string_match
from src.utils.save_and_load import load_objects, save_objects

# ---------------------------------------------------------------------------
# Sort enums  (mirror the arXiv API's accepted values)
# ---------------------------------------------------------------------------


class SortCriterion(Enum):
    """Field by which search results are sorted."""

    Relevance = "relevance"
    LastUpdatedDate = "lastUpdatedDate"
    SubmittedDate = "submittedDate"


class SortOrder(Enum):
    """Direction in which search results are sorted."""

    Ascending = "ascending"
    Descending = "descending"


# ---------------------------------------------------------------------------
# Internal search specification
# ---------------------------------------------------------------------------


@dataclass
class _ArxivSearch:
    """Parameters for a single arXiv API query."""

    query: str = ""
    id_list: list[str] = field(default_factory=list)
    max_results: Optional[int] = None
    sort_by: SortCriterion = SortCriterion.Relevance
    sort_order: SortOrder = SortOrder.Descending


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class ArxivPaperMetadata(BaseModel):
    arxiv_id: str
    title: str
    abstract: str = Field(default="")
    authors: list[str] = Field(default_factory=list)
    primary_category: str = Field(default="")
    categories: list[str] = Field(default_factory=list)
    published: str = Field(default="")
    updated: str = Field(default="")
    pdf_url: str = Field(default="")
    doi: str = Field(default="")
    journal_ref: str = Field(default="")
    summary: str = Field(default="")
    comment: str = Field(default="")
    entry_url: str = Field(default="")

    @classmethod
    def from_dict(cls, data: dict) -> "ArxivPaperMetadata":
        return cls.model_validate(data)

    @classmethod
    def from_feed_entry(cls, entry: feedparser.FeedParserDict) -> "ArxivPaperMetadata":
        """Construct an ``ArxivPaperMetadata`` from a raw feedparser entry."""
        entry_id: str = entry.get("id", "")
        arxiv_id = entry_id.split("/abs/")[-1]

        raw_title = entry.get("title", "0") or "0"
        title = re.sub(r"\s+", " ", raw_title)

        authors = [a.get("name", "") for a in entry.get("authors", [])]

        updated_ts = entry.get("updated_parsed")
        updated = (
            datetime.fromtimestamp(timegm(updated_ts), tz=timezone.utc).isoformat()
            if updated_ts
            else ""
        )

        published_ts = entry.get("published_parsed")
        published = (
            datetime.fromtimestamp(timegm(published_ts), tz=timezone.utc).isoformat()
            if published_ts
            else ""
        )

        pdf_url = ""
        for link in entry.get("links", []):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
                break

        primary_category = entry.get("arxiv_primary_category", {}).get("term", "")
        categories = [tag.get("term", "") for tag in entry.get("tags", [])]

        return cls(
            arxiv_id=arxiv_id,
            title=title,
            abstract=entry.get("summary", ""),
            authors=authors,
            primary_category=primary_category,
            categories=categories,
            published=published,
            updated=updated,
            pdf_url=pdf_url,
            doi=entry.get("arxiv_doi", "") or "",
            journal_ref=entry.get("arxiv_journal_ref", "") or "",
            summary=entry.get("summary", ""),
            comment=entry.get("arxiv_comment", "") or "",
            entry_url=f"https://arxiv.org/abs/{arxiv_id}",
        )

    def __repr__(self) -> str:
        return f"""
ArxivPaperMetadata(
    arxiv_id={self.arxiv_id},
    title={self.title},
    abstract={self.abstract},
    authors={self.authors},
    primary_category={self.primary_category},
    categories={self.categories},
    published={self.published},
    updated={self.updated},
    pdf_url={self.pdf_url},
    doi={self.doi},
    journal_ref={self.journal_ref},
    summary={self.summary},
    comment={self.comment},
    entry_url={self.entry_url},
)
"""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_QUERY_URL = "https://export.arxiv.org/api/query?{}"
MAX_BACKOFF_SECONDS = 90.0
_DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "cache"
    / "arxiv_online_client_cache.json"
)
ResultT = TypeVar("ResultT")


class ArxivOnlineClient:
    """High-level arXiv crawler.

    Communicates directly with the arXiv Atom feed API without relying on
    any third-party arXiv client library.

    Parameters
    ----------
    page_size:
        Number of results fetched per API request (max 2 000 per arXiv TOS).
    delay_seconds:
        Seconds to wait between paginated requests.  arXiv asks for ≥ 3 s.
    num_retries:
        How many times to retry a failed request.
    backoff_base:
        Base wait (seconds) before first retry; doubles each attempt.
    backoff_max:
        Maximum wait (seconds) between retry attempts.
    cache_path:
        JSON cache file path.  Defaults to
        ``resources/cache/arxiv_online_client_cache.json``.  Pass ``None`` to
        disable caching.
    """

    _session: requests.Session

    def __init__(
        self,
        page_size: int = 1000,
        delay_seconds: float = 5.0,
        num_retries: int = 5,
        backoff_base: float = 5.0,
        backoff_max: float = MAX_BACKOFF_SECONDS,
        cache_path: str | Path | None = _DEFAULT_CACHE_PATH,
    ) -> None:
        self._page_size = page_size
        self._delay_seconds = delay_seconds
        self._num_retries = num_retries
        self._backoff_base = backoff_base
        self._backoff_max = min(backoff_max, MAX_BACKOFF_SECONDS)
        self._cache = LocalJsonCache(cache_path)
        self._request_queue = threading.Condition()
        self._next_request_ticket = 0
        self._active_request_ticket = 0
        self._next_allowed_request_time = 0.0

        self._session = requests.Session()
        # Retry/backoff is handled explicitly in this client so every wait is
        # visible in our logs and capped consistently.
        adapter = requests.adapters.HTTPAdapter(max_retries=0)
        self._session.mount("https://", adapter)
        self._session.headers.update({"User-Agent": "Mozilla/5.0"})

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def perform_queued_request(self, request: Callable[[], ResultT]) -> ResultT:
        """Execute one online arXiv request in strict FIFO order.

        All threads that share this client instance enter a single queue. The
        next request may not start until the previous request has finished and
        ``delay_seconds`` has elapsed, which keeps parallel workers and
        sequential callers on the same schedule.
        """
        with self._request_queue:
            ticket = self._next_request_ticket
            self._next_request_ticket += 1
            while ticket != self._active_request_ticket:
                self._request_queue.wait()
            wait = self._next_allowed_request_time - time.monotonic()

        if wait > 0:
            logger.debug("Rate-limit: sleeping {:.2f} s", wait)
            time.sleep(wait)

        try:
            return request()
        finally:
            with self._request_queue:
                self._next_allowed_request_time = (
                    time.monotonic() + max(self._delay_seconds, 0.0)
                )
                self._active_request_ticket += 1
                self._request_queue.notify_all()

    def _build_url(self, search: _ArxivSearch, start: int, page_size: int) -> str:
        """Construct a paginated arXiv API query URL."""
        params = {
            "search_query": search.query,
            "id_list": ",".join(search.id_list),
            "sortBy": search.sort_by.value,
            "sortOrder": search.sort_order.value,
            "start": str(start),
            "max_results": str(page_size),
        }
        return _QUERY_URL.format(urlencode(params))

    def _backoff_sleep(self, attempt: int) -> None:
        """Sleep with exponential backoff before the next retry."""
        wait = min(
            self._backoff_base * (2**attempt),
            self._backoff_max,
            MAX_BACKOFF_SECONDS,
        )
        logger.debug("Backoff: sleeping {:.1f} s before retry {}", wait, attempt + 1)
        time.sleep(wait)

    def _fetch_page(
        self, url: str, first_page: bool = True
    ) -> feedparser.FeedParserDict:
        """Fetch one page of results from the arXiv API with retry + backoff.

        Respects the shared FIFO rate limit before every attempt.
        On failure, sleeps with exponential backoff before retrying.

        Raises
        ------
        requests.exceptions.RequestException
            If every retry attempt fails due to a network or HTTP error.
        RuntimeError
            If a non-first page returns an unexpectedly empty entry list on
            every retry attempt.
        """
        cache_key = LocalJsonCache.make_key("GET", url)
        cache_hit, cached_response = self._cache.get(cache_key)
        if cache_hit:
            logger.debug("arXiv cache hit for {}", url)
            return feedparser.parse(cached_response)

        last_exc: Exception = RuntimeError(f"Failed to fetch: {url}")
        for attempt in range(self._num_retries + 1):
            if attempt > 0:
                self._backoff_sleep(attempt - 1)
                logger.debug("Retry {} for: {}", attempt, url)
            try:
                def fetch_request() -> requests.Response:
                    response = self._session.get(url, timeout=(10, 60))
                    response.raise_for_status()
                    return response

                resp = self.perform_queued_request(fetch_request)
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                logger.warning("Request error (attempt {}): {}", attempt + 1, exc)
                continue

            feed = feedparser.parse(resp.content)

            if feed.bozo:
                logger.warning(
                    "Malformed feed warning: {}",
                    feed.get("bozo_exception"),
                )

            if not first_page and not feed.entries:
                last_exc = RuntimeError(f"Unexpectedly empty page: {url}")
                logger.warning("Empty page on attempt {}; retrying …", attempt + 1)
                continue

            self._cache.store(cache_key, "GET", url, resp.text)
            return feed

        raise last_exc

    def _iter_results(self, search: _ArxivSearch) -> Iterator[ArxivPaperMetadata]:
        """Paginate through all results for *search*, yielding
        ``ArxivPaperMetadata`` objects and logging progress."""
        limit = search.max_results
        offset = 0
        fetched = 0
        first_page = True

        while True:
            page_size = (
                min(self._page_size, limit - fetched) if limit else self._page_size
            )
            url = self._build_url(search, offset, page_size)
            feed = self._fetch_page(url, first_page=first_page)
            first_page = False

            if not feed.entries:
                logger.info("No more results; stopping pagination")
                break

            if offset == 0:
                total = int(feed.feed.get("opensearch_totalresults", 0))
                logger.info(
                    "Got first page: {} of {} total results",
                    len(feed.entries),
                    total,
                )

            for entry in feed.entries:
                try:
                    yield ArxivPaperMetadata.from_feed_entry(entry)
                except Exception as exc:
                    logger.warning("Skipping unparseable entry: {}", exc)
                    continue
                fetched += 1
                if fetched % 100 == 0:
                    logger.info("Fetched {} papers so far …", fetched)
                if limit and fetched >= limit:
                    return

            offset += len(feed.entries)
            total_available = int(feed.feed.get("opensearch_totalresults", 0))
            if offset >= total_available:
                break

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        max_results: int = 100,
        sort_by: SortCriterion = SortCriterion.Relevance,
        sort_order: SortOrder = SortOrder.Descending,
    ) -> list[ArxivPaperMetadata]:
        """Free-text search across all arXiv fields.

        Parameters
        ----------
        query:
            arXiv query string.  Supports field prefixes such as
            ``ti:transformer``, ``au:vaswani``, ``cat:cs.LG``, etc.
        max_results:
            Upper bound on the number of papers returned.
        sort_by:
            ``SortCriterion.Relevance`` (default),
            ``SubmittedDate``, or ``LastUpdatedDate``.
        sort_order:
            ``SortOrder.Descending`` (default) or ``Ascending``.

        Returns
        -------
        list[ArxivPaperMetadata]
        """
        if not query.strip():
            raise ValueError("query must not be empty")

        search = _ArxivSearch(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        papers = list(self._iter_results(search))

        if "ti:" in query.lower():
            query_phrase = query.lower()
            parts = query_phrase.split("and")
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                if "ti:" in part:
                    # Remove the "ti:" prefix, and then remove the beginning and ending double quotes if they exist
                    part = part.replace("ti:", "").strip('"')
                    papers = sorted(
                        papers,
                        key=lambda x: score_string_match(part, x.title),
                        reverse=True,
                    )
                    break

        logger.info("search('{}') → {} papers", query, len(papers))
        return papers

    def get_paper(self, arxiv_id: str) -> ArxivPaperMetadata:
        """Fetch metadata for a single paper by its arXiv ID.

        Parameters
        ----------
        arxiv_id:
            Bare ID like ``"2005.11401"`` or ``"2005.11401v4"``, or a full
            URL like ``"https://arxiv.org/abs/2005.11401"``.
        """
        arxiv_id = arxiv_id.split("/abs/")[-1].split("arxiv.org/")[-1]

        search = _ArxivSearch(id_list=[arxiv_id])
        results = list(self._iter_results(search))
        if not results:
            raise ValueError(f"No paper found for id={arxiv_id!r}")
        return results[0]

    def get_papers(self, arxiv_ids: list[str]) -> list[ArxivPaperMetadata]:
        """Fetch metadata for a batch of arXiv IDs.

        The arXiv API accepts up to 100 IDs per request; this method
        automatically chunks larger lists.
        """
        chunk_size = 100
        papers: list[ArxivPaperMetadata] = []
        for start in range(0, len(arxiv_ids), chunk_size):
            chunk = arxiv_ids[start : start + chunk_size]
            clean = [i.split("/abs/")[-1] for i in chunk]
            search = _ArxivSearch(id_list=clean)
            papers.extend(self._iter_results(search))
        logger.info("get_papers() → {} papers", len(papers))
        return papers

    def crawl(
        self,
        *,
        category: Optional[str] = None,
        query: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        max_results: int = 1000,
        sort_by: SortCriterion = SortCriterion.SubmittedDate,
        sort_order: SortOrder = SortOrder.Descending,
        output_path: Optional[str | Path] = None,
    ) -> list[ArxivPaperMetadata]:
        """Crawl papers matching a category and/or date range.

        At least one of ``category`` or ``query`` must be provided.

        Parameters
        ----------
        category:
            arXiv category, e.g. ``"cs.CL"``, ``"cs.LG"``, ``"stat.ML"``.
        query:
            Additional free-text query to AND with the category filter.
        date_from:
            Earliest submission date (``"YYYY-MM-DD"``).
        date_to:
            Latest submission date (``"YYYY-MM-DD"``).
        max_results:
            Maximum number of papers to return.
        output_path:
            If given, results are streamed to this file in JSONL format
            (one JSON object per line) as they arrive.

        Returns
        -------
        list[ArxivPaperMetadata]
        """
        parts: list[str] = []

        if category:
            parts.append(f"cat:{category}")

        if query:
            parts.append(query)

        if date_from or date_to:
            lo = date_from.replace("-", "") if date_from else "00000000"
            hi = date_to.replace("-", "") if date_to else "99999999"
            parts.append(f"submittedDate:[{lo}0000 TO {hi}2359]")

        if not parts:
            raise ValueError(
                "Provide at least one of: category, query, date_from/date_to"
            )

        full_query = " AND ".join(parts)
        logger.info("crawl query: {}  (max_results={})", full_query, max_results)

        search = _ArxivSearch(
            query=full_query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        papers: list[ArxivPaperMetadata] = []
        out_file = None

        if output_path is not None:
            parent = os.path.dirname(output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            out_file = open(output_path, "w", encoding="utf-8")

        try:
            for paper in self._iter_results(search):
                papers.append(paper)
                if out_file is not None:
                    out_file.write(paper.model_dump_json() + "\n")
        finally:
            if out_file is not None:
                out_file.close()

        logger.info("crawl() → {} papers", len(papers))
        return papers

    # ------------------------------------------------------------------
    # Source download helpers
    # ------------------------------------------------------------------

    def check_source_available(self, arxiv_id: str) -> bool:
        """Check whether LaTeX source files are available for *arxiv_id*.
        Reference: https://github.com/takashiishida/arxiv-to-prompt/blob/main/src/arxiv_to_prompt/core.py

        Queries ``https://arxiv.org/format/<arxiv_id>`` and looks for the
        string ``'Download source'`` in the response body.

        Parameters
        ----------
        arxiv_id:
            Bare arXiv ID, e.g. ``"2005.11401"`` or ``"2005.11401v4"``.

        Returns
        -------
        bool
            ``True`` if a source tarball appears to be available.
        """
        url = f"https://arxiv.org/format/{arxiv_id}"
        last_exc: Exception = RuntimeError(
            f"Failed to check source availability for {arxiv_id}"
        )
        for attempt in range(self._num_retries + 1):
            if attempt > 0:
                self._backoff_sleep(attempt - 1)
                logger.debug("Retry {} for source availability: {}", attempt, url)
            try:
                def check_request() -> bool:
                    response = self._session.get(url, timeout=(5, 30))
                    response.raise_for_status()
                    return "Download source" in response.text

                return self.perform_queued_request(check_request)
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "Source availability check failed for {} (attempt {}/{}): {}",
                    arxiv_id,
                    attempt + 1,
                    self._num_retries + 1,
                    exc,
                )

        logger.error(
            "Error checking source availability for {} after {} attempt(s): {}",
            arxiv_id,
            self._num_retries + 1,
            last_exc,
        )
        return False

    def download_source(
        self,
        arxiv_id: str,
        output_dir: str | Path = ".",
        *,
        skip_if_exists: bool = True,
    ) -> Optional[Path]:
        """Download the LaTeX source tarball for *arxiv_id*.

        The file is saved as ``<output_dir>/<arxiv_id>.tar.gz``.  If the
        source is not available (determined via :meth:`check_source_available`)
        the method returns ``None`` without raising.

        Parameters
        ----------
        arxiv_id:
            Bare arXiv ID, e.g. ``"2005.11401"`` or ``"2005.11401v4"``.
        output_dir:
            Directory where the tarball will be saved.  Created automatically
            if it does not yet exist.
        skip_if_exists:
            When ``True`` (default), skip the download if the target file
            already exists and return its path immediately.

        Returns
        -------
        Path | None
            Path to the saved tarball, or ``None`` if the source is
            unavailable.

        Raises
        ------
        requests.exceptions.RequestException
            On any network / HTTP error during the actual download.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_id = arxiv_id.replace("/", "_")
        dest = Path(output_dir) / f"arXiv-{safe_id}.tar.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)

        if skip_if_exists and dest.exists():
            logger.info("Source already exists, skipping download: {}", dest)
            return dest

        url = f"https://arxiv.org/e-print/{arxiv_id}"

        last_exc: Exception = RuntimeError(f"Failed to download source {url}")
        for attempt in range(self._num_retries + 1):
            if attempt > 0:
                self._backoff_sleep(attempt - 1)
                logger.debug("Retry {} for source download: {}", attempt, url)
            try:
                def download_request() -> Optional[Path]:
                    response = self._session.get(url, timeout=(5, 120), stream=True)
                    response.raise_for_status()

                    content_type = response.headers.get("Content-Type", "")
                    if "application/pdf" in content_type or "text/html" in content_type:
                        logger.warning(
                            "Source (tar.gz) not available for {} (server returned '{}'), skipping. "
                            "Check https://arxiv.org/abs/{}",
                            arxiv_id,
                            content_type,
                            arxiv_id,
                        )
                        return None

                    with dest.open("wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    logger.info("Saved source tarball {} → {}", url, dest)
                    return dest

                return self.perform_queued_request(download_request)
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "Error downloading source {} (attempt {}/{}): {}",
                    url,
                    attempt + 1,
                    self._num_retries + 1,
                    exc,
                )
                if dest.exists():
                    dest.unlink()

        logger.error(
            "Error downloading source {} after {} attempt(s): {}",
            url,
            self._num_retries + 1,
            last_exc,
        )
        raise last_exc

    def download_pdf(
        self, arxiv_id: str, output_dir: str | Path = "."
    ) -> Optional[Path]:
        """Download the PDF for *arxiv_id*."""
        url = f"https://arxiv.org/pdf/{arxiv_id}"

        dest = Path(output_dir) / f"arXiv-{arxiv_id}.pdf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        last_exc: Exception = RuntimeError(f"Failed to download PDF {url}")
        for attempt in range(self._num_retries + 1):
            if attempt > 0:
                self._backoff_sleep(attempt - 1)
                logger.debug("Retry {} for PDF download: {}", attempt, url)
            try:
                def download_request() -> Path:
                    response = self._session.get(url, timeout=(5, 120), stream=True)
                    response.raise_for_status()

                    with dest.open("wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    logger.info("Saved PDF → {}", dest)
                    return dest

                return self.perform_queued_request(download_request)
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "Error downloading PDF for {} (attempt {}/{}): {}",
                    arxiv_id,
                    attempt + 1,
                    self._num_retries + 1,
                    exc,
                )
                if dest.exists():
                    dest.unlink()

        logger.error(
            "Error downloading PDF for {} after {} attempt(s): {}",
            arxiv_id,
            self._num_retries + 1,
            last_exc,
        )
        return None

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def get_result_schema() -> type[ArxivPaperMetadata]:
        return ArxivPaperMetadata

    @staticmethod
    def save_jsonl(papers: list[ArxivPaperMetadata], path: str | Path) -> None:
        """Write a list of papers to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for paper in papers:
                f.write(paper.model_dump_json() + "\n")
        logger.info("Saved {} papers → {}", len(papers), path)

    @staticmethod
    def load_jsonl(path: str | Path) -> list[ArxivPaperMetadata]:
        """Load papers previously saved with :meth:`save_jsonl`."""
        path = Path(path)
        papers: list[ArxivPaperMetadata] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    papers.append(ArxivPaperMetadata.model_validate(orjson.loads(line)))
        return papers

    @staticmethod
    def save_csv(papers: list[ArxivPaperMetadata], path: str | Path) -> None:
        """Write a list of papers to a CSV file (authors/categories as JSON arrays)."""
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not papers:
            logger.warning("save_csv: no papers to write")
            return

        rows = [p.model_dump() for p in papers]
        # Serialise list fields as JSON strings so CSV stays flat
        for row in rows:
            row["authors"] = orjson.dumps(row["authors"]).decode("utf-8")
            row["categories"] = orjson.dumps(row["categories"]).decode("utf-8")

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Saved {} papers → {}", len(papers), path)


ArxivClient = ArxivOnlineClient


if __name__ == "__main__":
    client = ArxivOnlineClient()

    papers = client.search(
        'ti:"Attention is all you need"',
        max_results=5,
    )
    print(papers)
    exit()

    papers = client.crawl(
        date_from="2026-01-01",
        date_to="2026-01-31",
        max_results=200_000,
        output_path="data/test_2026_01.jsonl",
    )

    save_objects(papers, "data/test_2026_01.jsonl")
    papers = load_objects("data/test_2026_01.jsonl", ArxivPaperMetadata)
    print(papers)

    # Test download source
    result = client.download_source(papers[0].arxiv_id)
    print(result)

    # Test download PDF
    result = client.download_pdf(papers[0].arxiv_id)
    print(result)
