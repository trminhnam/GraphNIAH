"""
DBLP search API client.

To run this script:
python -m src.clients.dblp_client

Wraps the three DBLP CompleteSearch endpoints:

- https://dblp.org/search/publ/api  — publications
- https://dblp.org/search/author/api — persons (authors / editors)
- https://dblp.org/search/venue/api  — venues (journals / conferences)

API reference: https://dblp.org/faq/How+to+use+the+dblp+search+API.html

Usage::

    client = DBLPClient()

    # Search publications
    pubs = client.search_publications("attention is all you need", max_hits=10)
    for p in pubs:
        print(p.title, p.year, p.venue)

    # Search authors
    authors = client.search_authors("Yoshua Bengio", max_hits=5)

    # Search venues
    venues = client.search_venues("NeurIPS", max_hits=5)

    # Fetch all pages of a large publication query
    all_pubs = client.search_all_publications("transformer language model", max_hits=500)
"""

from __future__ import annotations

import orjson
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests
from loguru import logger

from src.clients.local_json_cache import LocalJsonCache

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class DBLPPublication:
    """Metadata for a single DBLP publication hit."""

    key: str  # DBLP record key, e.g. "conf/nips/VaswaniSPUJGKP17"
    title: str
    authors: list[str]
    year: str
    venue: str  # Abbreviated venue name
    venue_type: str  # "Conference", "Journal", "Workshop", etc.
    pages: str
    doi: str
    url: str  # DBLP record URL
    ee: str  # Electronic edition URL (publisher / arXiv link)
    access: str  # "open" or ""
    type: str  # "inproceedings", "article", "book", …

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return orjson.dumps(self.to_dict()).decode("utf-8")

    @classmethod
    def from_hit(cls, hit: dict) -> "DBLPPublication":
        info = hit.get("info", {})
        authors_raw = info.get("authors", {}).get("author", [])
        if isinstance(authors_raw, dict):
            authors_raw = [authors_raw]
        authors = [
            a.get("text", "") if isinstance(a, dict) else str(a) for a in authors_raw
        ]
        ee_raw = info.get("ee", "")
        if isinstance(ee_raw, list):
            ee_raw = ee_raw[0] if ee_raw else ""
        elif isinstance(ee_raw, dict):
            ee_raw = ee_raw.get("text", "") or ee_raw.get("#text", "")
        return cls(
            key=info.get("key", ""),
            title=info.get("title", "").strip(),
            authors=authors,
            year=str(info.get("year", "")),
            venue=info.get("venue", ""),
            venue_type=info.get("type", ""),
            pages=info.get("pages", ""),
            doi=info.get("doi", ""),
            url=info.get("url", ""),
            ee=str(ee_raw),
            access=info.get("access", ""),
            type=info.get("type", ""),
        )

    def __str__(self) -> str:
        return f"""
DBLPPublication(
    title={self.title},
    authors={self.authors},
    year={self.year},
    venue={self.venue},
    venue_type={self.venue_type},
    pages={self.pages},
    doi={self.doi},
    url={self.url},
    ee={self.ee},
    access={self.access},
    type={self.type}
)
"""


@dataclass
class DBLPAuthor:
    """Metadata for a single DBLP author hit."""

    author: str
    url: str  # DBLP person page URL
    aliases: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return orjson.dumps(self.to_dict()).decode("utf-8")

    @classmethod
    def from_hit(cls, hit: dict) -> "DBLPAuthor":
        info = hit.get("info", {})
        aliases_raw = info.get("aliases", {}).get("alias", [])
        if isinstance(aliases_raw, str):
            aliases_raw = [aliases_raw]
        notes_raw = info.get("notes", {}).get("note", [])
        if isinstance(notes_raw, str):
            notes_raw = [notes_raw]
        elif isinstance(notes_raw, dict):
            notes_raw = [notes_raw.get("text", "")]
        return cls(
            author=info.get("author", ""),
            url=info.get("url", ""),
            aliases=aliases_raw,
            notes=[n if isinstance(n, str) else n.get("text", "") for n in notes_raw],
        )

    def __str__(self) -> str:
        return f"""
DBLPAuthor(
    author={self.author},
    url={self.url},
    aliases={self.aliases},
    notes={self.notes}
)
"""


@dataclass
class DBLPVenue:
    """Metadata for a single DBLP venue hit."""

    venue: str
    acronym: str
    type: str  # "Conference", "Journal", "Workshop", etc.
    url: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return orjson.dumps(self.to_dict()).decode("utf-8")

    @classmethod
    def from_hit(cls, hit: dict) -> "DBLPVenue":
        info = hit.get("info", {})
        return cls(
            venue=info.get("venue", ""),
            acronym=info.get("acronym", ""),
            type=info.get("type", ""),
            url=info.get("url", ""),
        )

    def __str__(self) -> str:
        return f"""
DBLPVenue(
    venue={self.venue},
    acronym={self.acronym},
    type={self.type},
    url={self.url}
)
"""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class RateLimitedError(requests.HTTPError):
    """Raised when DBLP returns 429 and all retry attempts are exhausted."""


class DBLPClient:
    """Client for the DBLP CompleteSearch API.

    Parameters
    ----------
    timeout:
        Request timeout in seconds.
    delay_seconds:
        Minimum wait between consecutive requests.  DBLP recommends at least
        1–2 seconds to avoid triggering the rate limiter.
    max_retries:
        How many times to retry transient failures (5xx / network errors).
    max_rate_limit_retries:
        How many times to honour a ``429 Too Many Requests`` response before
        giving up.  Each retry sleeps for the duration specified in the
        ``Retry-After`` response header (falling back to
        ``rate_limit_fallback_wait`` when the header is absent).
    rate_limit_fallback_wait:
        Seconds to wait on a 429 response when the server does not send a
        ``Retry-After`` header.
    cache_path:
        JSON cache file path.  Defaults to ``resources/cache/dblp_client_cache.json``.
        Pass ``None`` to disable caching.
    """

    _PUBL_URL = "https://dblp.org/search/publ/api"
    _AUTHOR_URL = "https://dblp.org/search/author/api"
    _VENUE_URL = "https://dblp.org/search/venue/api"
    _ERROR_RETRY_BACKOFF_SECONDS = (5.0, 10.0, 15.0)
    CACHE_PATH = (
        Path(__file__).resolve().parents[2]
        / "resources"
        / "cache"
        / "dblp_client_cache.json"
    )

    def __init__(
        self,
        timeout: int = 30,
        delay_seconds: float = 2.5,
        max_retries: int = 3,
        max_rate_limit_retries: int = 5,
        rate_limit_fallback_wait: float = 10.0,
        cache_path: str | Path | None = CACHE_PATH,
    ) -> None:
        self._timeout = timeout
        self._delay = delay_seconds
        self._max_retries = max_retries
        self._max_rate_limit_retries = max_rate_limit_retries
        self._rate_limit_fallback_wait = rate_limit_fallback_wait
        self._cache = LocalJsonCache(cache_path)
        self._last_request_time: float = 0.0
        self._throttle_lock = threading.Lock()
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait(self) -> None:
        """Enforce the minimum inter-request delay (thread-safe)."""
        with self._throttle_lock:
            now = time.monotonic()
            earliest = self._last_request_time + self._delay
            wait = earliest - now
            self._last_request_time = max(now, earliest)
        if wait > 0:
            time.sleep(wait)

    def _retry_after(self, response: requests.Response) -> float:
        """Parse the ``Retry-After`` header; fall back to the configured default."""
        try:
            retry_after = response.headers.get("Retry-After", "")
            return float(retry_after)
        except (ValueError, TypeError):
            return self._rate_limit_fallback_wait

    def _get(self, url: str, params: dict) -> Any:
        params = {k: v for k, v in params.items() if v is not None}
        params.setdefault("format", "json")
        params.setdefault("c", 0)  # skip completions by default
        cache_key = LocalJsonCache.make_key("GET", url, params=params)
        cache_hit, cached_response = self._cache.get(cache_key)
        if cache_hit:
            logger.debug("DBLP cache hit for {}", url)
            return cached_response

        # Two independent counters so that 429 back-offs never eat into the
        # budget reserved for genuine server errors (5xx / network failures).
        error_attempts = 0
        rate_limit_attempts = 0

        while True:
            self._wait()
            try:
                response = self._session.get(url, params=params, timeout=self._timeout)

                # --- 429 Rate-limited ----------------------------------------
                if response.status_code == 429:
                    rate_limit_attempts += 1
                    wait = self._retry_after(response)
                    logger.warning(
                        "429 Too Many Requests — waiting {:.0f} s before retry "
                        "(rate-limit attempt {}/{})",
                        wait,
                        rate_limit_attempts,
                        self._max_rate_limit_retries,
                    )
                    if rate_limit_attempts > self._max_rate_limit_retries:
                        response.raise_for_status()
                    time.sleep(wait)
                    continue  # replay without incrementing error_attempts

                response.raise_for_status()
                data = response.json()
                self._cache.store(
                    cache_key,
                    "GET",
                    url,
                    data,
                    params=params,
                )
                return data

            except requests.HTTPError as exc:
                # 4xx errors other than 429 are caller mistakes — raise at once.
                if exc.response is not None and exc.response.status_code < 500:
                    raise
                error_attempts += 1
                if error_attempts > self._max_retries:
                    raise
                wait = self._ERROR_RETRY_BACKOFF_SECONDS[
                    min(error_attempts, len(self._ERROR_RETRY_BACKOFF_SECONDS)) - 1
                ]
                logger.warning(
                    "Attempt {}/{} failed (HTTP {}); waiting {:.0f} s before retry: {}",
                    error_attempts,
                    self._max_retries,
                    getattr(exc.response, "status_code", "?"),
                    wait,
                    exc,
                )
                time.sleep(wait)

            except requests.RequestException as exc:
                error_attempts += 1
                if error_attempts > self._max_retries:
                    raise
                wait = self._ERROR_RETRY_BACKOFF_SECONDS[
                    min(error_attempts, len(self._ERROR_RETRY_BACKOFF_SECONDS)) - 1
                ]
                logger.warning(
                    "Attempt {}/{} failed (network); waiting {:.0f} s before retry: {}",
                    error_attempts,
                    self._max_retries,
                    wait,
                    exc,
                )
                time.sleep(wait)

    def _parse_result(self, data: dict) -> tuple[list[dict], int]:
        """Return ``(hits, total)`` from a raw API response."""
        result = data.get("result", {})
        hits_obj = result.get("hits", {})
        total = int(hits_obj.get("@total", 0))
        raw_hits = hits_obj.get("hit", [])
        if isinstance(raw_hits, dict):
            raw_hits = [raw_hits]
        return raw_hits, total

    # ------------------------------------------------------------------
    # Publication search
    # ------------------------------------------------------------------

    def search_publications(
        self, query: str, *, max_hits: int = 100, offset: int = 0
    ) -> list[DBLPPublication]:
        """Search DBLP for publications matching *query*.

        Parameters
        ----------
        query:
            Free-text query.  Supports field prefixes such as
            ``author:vaswani``, ``title:transformer``, etc.
        max_hits:
            Maximum results to return (capped by DBLP at 1 000).
        offset:
            First result index for pagination (0-based).

        Returns
        -------
        list[DBLPPublication]
        """

        if not query.strip():
            raise ValueError("query must not be empty")

        data = self._get(self._PUBL_URL, {"q": query, "h": max_hits, "f": offset})
        raw_hits, total = self._parse_result(data)
        pubs = [DBLPPublication.from_hit(h) for h in raw_hits]
        # logger.info(
        #     f"search_publications({query}, h={max_hits}, f={offset}) → {len(pubs)} / {total}",
        # )
        return pubs

    def search_all_publications(
        self, query: str, *, max_hits: int = 1000, page_size: int = 100
    ) -> list[DBLPPublication]:
        """Paginate through all publication results for *query*.

        Parameters
        ----------
        query:
            DBLP query string.
        max_hits:
            Hard cap on total results.
        page_size:
            Results per page (max 1 000 per DBLP TOS).

        Returns
        -------
        list[DBLPPublication]
        """
        pubs: list[DBLPPublication] = []
        offset = 0
        while len(pubs) < max_hits:
            batch_size = min(page_size, max_hits - len(pubs))
            batch = self.search_publications(query, max_hits=batch_size, offset=offset)
            pubs.extend(batch)
            if len(batch) < batch_size:
                break
            offset += batch_size
        total = len(pubs)
        # logger.info(
        #     f"search_all_publications({query}, h={max_hits}, f={offset}) → {total}"
        # )
        return pubs[:max_hits]

    # ------------------------------------------------------------------
    # Author search
    # ------------------------------------------------------------------

    def search_authors(
        self, query: str, *, max_hits: int = 30, offset: int = 0
    ) -> list[DBLPAuthor]:
        """Search DBLP for persons matching *query*.

        Returns
        -------
        list[DBLPAuthor]
        """
        if not query.strip():
            raise ValueError("query must not be empty")
        data = self._get(self._AUTHOR_URL, {"q": query, "h": max_hits, "f": offset})
        raw_hits, total = self._parse_result(data)
        authors = [DBLPAuthor.from_hit(h) for h in raw_hits]
        # logger.info(
        #     f"search_authors({query}, h={max_hits}, f={offset}) → {len(authors)} / {total}",
        # )
        return authors

    def search_all_authors(
        self, query: str, *, max_hits: int = 1000, page_size: int = 100
    ) -> list[DBLPAuthor]:
        """Paginate through all author results for *query*."""
        authors: list[DBLPAuthor] = []
        offset = 0
        while len(authors) < max_hits:
            batch_size = min(page_size, max_hits - len(authors))
            batch = self.search_authors(query, max_hits=batch_size, offset=offset)
            authors.extend(batch)
            if len(batch) < batch_size:
                break
            offset += batch_size
        total = len(authors)
        # logger.info(f"search_all_authors({query}, h={max_hits}, f={offset}) → {total}")
        return authors[:max_hits]

    # ------------------------------------------------------------------
    # Venue search
    # ------------------------------------------------------------------

    def search_venues(
        self, query: str, *, max_hits: int = 30, offset: int = 0
    ) -> list[DBLPVenue]:
        """Search DBLP for venues matching *query*.

        Returns
        -------
        list[DBLPVenue]
        """
        if not query.strip():
            raise ValueError("query must not be empty")
        data = self._get(self._VENUE_URL, {"q": query, "h": max_hits, "f": offset})
        raw_hits, total = self._parse_result(data)
        venues = [DBLPVenue.from_hit(h) for h in raw_hits]
        # logger.info(
        #     f"search_venues({query}, h={max_hits}, f={offset}) → {len(venues)} / {total}",
        # )
        return venues

    def search_all_venues(
        self, query: str, *, max_hits: int = 1000, page_size: int = 100
    ) -> list[DBLPVenue]:
        """Paginate through all venue results for *query*."""
        venues: list[DBLPVenue] = []
        offset = 0
        while len(venues) < max_hits:
            batch_size = min(page_size, max_hits - len(venues))
            batch = self.search_venues(query, max_hits=batch_size, offset=offset)
            venues.extend(batch)
            if len(batch) < batch_size:
                break
            offset += batch_size
        # logger.info(f"search_all_venues({query}, h={max_hits}, f={offset}) → {len(venues)} total")
        return venues[:max_hits]

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save_jsonl(
        records: list[DBLPPublication] | list[DBLPAuthor] | list[DBLPVenue],
        path: str | Path,
    ) -> None:
        """Write a list of DBLP records to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(rec.to_json() + "\n")
        # logger.info("Saved {} records → {}", len(records), path)

    @staticmethod
    def load_publications_jsonl(path: str | Path) -> list[DBLPPublication]:
        """Load publications previously saved with :meth:`save_jsonl`."""
        path = Path(path)
        pubs: list[DBLPPublication] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pubs.append(DBLPPublication(**orjson.loads(line)))
        return pubs


if __name__ == "__main__":
    from src.logging import configure_logging

    configure_logging()
    client = DBLPClient()

    print("=== Publications ===")
    title = "Attention is all you need"
    # add $ to the end of each word to make it a literal search
    title_query = " ".join([f"{word}$" for word in title.split()])
    pubs = client.search_publications(title_query, max_hits=10)
    for p in pubs:
        # print(f"  [{p.year}] {p.title} — {p.venue} - {p.authors}")
        print(p)

    # print("\n=== Authors ===")
    # authors = client.search_authors("Yoshua Bengio", max_hits=3)
    # for a in authors:
    #     # print(f"  {a.author}  {a.url}")
    #     print(a)

    # print("\n=== Venues ===")
    # venues = client.search_venues("NeurIPS", max_hits=3)
    # for v in venues:
    #     # print(f"  {v.venue} ({v.acronym})  {v.type}")
    #     print(v)
