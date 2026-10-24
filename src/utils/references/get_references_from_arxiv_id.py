"""
Fetch the references of an arXiv paper via the Semantic Scholar Graph API.

Given an arXiv ID, retrieves the structured list of papers that the target
paper cites, returning typed :class:`Reference` objects.

Usage::

    refs = get_references_from_arxiv_id("2005.11401")
    for ref in refs:
        print(ref.title, ref.arxiv_id, ref.year)

    # Only arXiv papers, sorted by citation count
    arxiv_only = [r for r in refs if r.arxiv_id]
    arxiv_only.sort(key=lambda r: r.citation_count or 0, reverse=True)
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# Ensure the project root (three levels up from this file) is on sys.path so
# that `src` is importable regardless of the working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.clients.semantic_scholar_client import SemanticScholarClient

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Reference:
    """A single paper referenced by the source paper."""

    # Semantic Scholar paper ID (40-char hex); None if not in S2 corpus.
    s2_paper_id: Optional[str]

    # Identifiers in other systems
    arxiv_id: Optional[str]
    doi: Optional[str]

    # Bibliographic metadata
    title: Optional[str]
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    publication_date: Optional[str] = None
    venue: Optional[str] = None

    # Impact metrics
    citation_count: Optional[int] = None
    reference_count: Optional[int] = None

    # Open-access
    is_open_access: Optional[bool] = None
    open_access_pdf_url: Optional[str] = None

    # Semantic Scholar page URL
    url: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_s2_paper(cls, paper: dict) -> "Reference":
        """Build a :class:`Reference` from a Semantic Scholar paper dict."""
        ext = paper.get("externalIds") or {}
        oa_pdf = paper.get("openAccessPdf") or {}
        authors = [a.get("name", "") for a in (paper.get("authors") or [])]
        return cls(
            s2_paper_id=paper.get("paperId"),
            arxiv_id=ext.get("ArXiv"),
            doi=ext.get("DOI"),
            title=paper.get("title"),
            authors=authors,
            year=paper.get("year"),
            publication_date=paper.get("publicationDate"),
            venue=paper.get("venue") or None,
            citation_count=paper.get("citationCount"),
            reference_count=paper.get("referenceCount"),
            is_open_access=paper.get("isOpenAccess"),
            open_access_pdf_url=oa_pdf.get("url"),
            url=paper.get("url"),
        )


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def get_references_from_arxiv_id(
    arxiv_id: str,
    *,
    max_results: int = 1000,
    api_key: Optional[str] = None,
    client: Optional[SemanticScholarClient] = None,
) -> list[Reference]:
    """Fetch the papers cited by an arXiv paper.

    Parameters
    ----------
    arxiv_id:
        Bare arXiv ID such as ``"2005.11401"`` or ``"2005.11401v4"``, a
        prefixed form ``"arXiv:2005.11401"``, or a full URL like
        ``"https://arxiv.org/abs/2005.11401"``.
    max_results:
        Maximum number of references to return (default 1 000, which covers
        the vast majority of academic papers).
    api_key:
        Optional Semantic Scholar API key for higher rate limits (10 req/s).
        If *client* is provided this parameter is ignored.
    client:
        An existing :class:`~src.clients.semantic_scholar_client.SemanticScholarClient`
        instance to reuse (useful when fetching references for many papers in
        a loop so the same session and rate-limiter are shared).

    Returns
    -------
    list[Reference]
        One :class:`Reference` per cited paper, in the order returned by
        Semantic Scholar.  Papers not yet indexed by S2 will not appear.

    Raises
    ------
    requests.HTTPError
        If the Semantic Scholar API returns a non-2xx response (e.g. 404 when
        the paper is not in the S2 corpus).
    ValueError
        If *arxiv_id* is empty.

    Examples
    --------
    >>> refs = get_references_from_arxiv_id("1706.03762")   # Attention Is All You Need
    >>> print(len(refs))
    >>> for r in refs[:5]:
    ...     print(r.title, r.year)
    """
    if not arxiv_id.strip():
        raise ValueError("arxiv_id must not be empty")

    s2 = client or SemanticScholarClient(
        api_key=api_key,
        delay_seconds=0.0 if api_key else 1.1,
    )

    raw = s2.get_all_references(arxiv_id, max_results=max_results)

    refs: list[Reference] = []
    for entry in raw:
        cited = entry.get("citedPaper")
        if not cited:
            continue
        refs.append(Reference.from_s2_paper(cited))

    return refs


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_references_from_arxiv_id.py <arxiv_id> [max_results]")
        sys.exit(1)

    _arxiv_id = sys.argv[1]
    _max = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    _refs = get_references_from_arxiv_id(_arxiv_id, max_results=_max)
    print(f"\nFound {len(_refs)} reference(s) for arXiv:{_arxiv_id}\n")
    for i, ref in enumerate(_refs, 1):
        arxiv_tag = f"arXiv:{ref.arxiv_id}" if ref.arxiv_id else (ref.doi or "—")
        print(f"{i:>4}. [{arxiv_tag}] {ref.title or 'N/A'} ({ref.year or '?'})")
