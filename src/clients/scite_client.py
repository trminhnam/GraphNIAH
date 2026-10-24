"""
Scite API client.

Wraps the scite.ai REST API (https://api.scite.ai).
Every public method corresponds to one API route.

Usage::

    client = SciteClient(api_token="your-token")
    tally  = client.get_tally("10.1016/j.biopsych.2005.08.012")
    paper  = client.get_paper("10.1016/j.biopsych.2005.08.012")
"""

from __future__ import annotations

from typing import Any, Optional

import requests


class SciteClient:
    """Client for the scite.ai REST API.

    Parameters
    ----------
    api_token:
        Bearer token for authenticated endpoints.  Pass ``None`` to use the
        client in unauthenticated mode (only public endpoints will work).
    base_url:
        Override the default API base URL.
    timeout:
        Default request timeout in seconds.
    """

    BASE_URL = "https://api.scite.ai"

    def __init__(
        self,
        api_token: Optional[str] = None,
        base_url: str = BASE_URL,
        timeout: int = 30,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        if api_token:
            self._session.headers.update({"Authorization": f"Bearer {api_token}"})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        params = _drop_nones(params or {})
        response = self._session.get(
            f"{self._base_url}{path}", params=params, timeout=self._timeout
        )
        response.raise_for_status()
        return response.json()

    def _post(
        self,
        path: str,
        json: Any = None,
        data: Any = None,
        files: Any = None,
        params: Optional[dict] = None,
    ) -> Any:
        params = _drop_nones(params or {})
        response = self._session.post(
            f"{self._base_url}{path}",
            json=json,
            data=data,
            files=files,
            params=params,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def _put(self, path: str, json: Any = None) -> Any:
        response = self._session.put(
            f"{self._base_url}{path}", json=json, timeout=self._timeout
        )
        response.raise_for_status()
        return response.json()

    def _delete(self, path: str, params: Optional[dict] = None) -> Any:
        params = _drop_nones(params or {})
        response = self._session.delete(
            f"{self._base_url}{path}", params=params, timeout=self._timeout
        )
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Journal / ISSN endpoints
    # ------------------------------------------------------------------

    def get_journal_sji(self, issn: str, year: Optional[int] = None) -> dict:
        """GET /issn-sji — Scite Journal Index for a single ISSN."""
        return self._get("/issn-sji", {"issn": issn, "year": year})

    def get_journal_sji_bulk(self, issns: list[str]) -> dict:
        """POST /issn-sji-bulk — Scite Journal Index for multiple ISSNs."""
        return self._post("/issn-sji-bulk", json=issns)

    def get_journal_editorial_notices(self, issn: str) -> dict:
        """GET /issn-editorial-notices — Editorial notices for a journal."""
        return self._get("/issn-editorial-notices", {"issn": issn})

    def get_journal_tallies(self, issn: str) -> dict:
        """GET /journal/{issn}/tallies — Smart Citation tallies for a journal."""
        return self._get(f"/journal/{issn}/tallies")

    def get_journal_yearly_si(
        self, issn: str, years: Optional[list[int]] = None
    ) -> dict:
        """GET /journal/{issn}/yearly-si — Yearly Scite Index for a journal."""
        params: dict[str, Any] = {}
        if years is not None:
            params["years"] = years
        return self._get(f"/journal/{issn}/yearly-si", params)

    def get_journal_article_count(self, slug_or_issn: str) -> dict:
        """GET /dashboards/journal/{slug_or_issn}/article-count — Publication count for a journal dashboard."""
        return self._get(f"/dashboards/journal/{slug_or_issn}/article-count")

    # ------------------------------------------------------------------
    # Papers
    # ------------------------------------------------------------------

    def get_paper(self, doi: str) -> dict:
        """GET /papers/{doi} — Metadata for a single paper."""
        return self._get(f"/papers/{doi}")

    def get_papers(self, dois: list[str]) -> dict:
        """POST /papers — Metadata for multiple papers by DOI list."""
        return self._post("/papers", json=dois)

    def get_papers_by_target(self, target_doi: str) -> dict:
        """GET /papers/sources/{target_doi} — Papers that cite a target DOI."""
        return self._get(f"/papers/sources/{target_doi}")

    # ------------------------------------------------------------------
    # Tallies
    # ------------------------------------------------------------------

    def get_tally(self, doi: str) -> dict:
        """GET /tallies/{doi} — Smart Citation tally for a single DOI."""
        return self._get(f"/tallies/{doi}")

    def get_tallies(self, dois: list[str]) -> dict:
        """POST /tallies — Smart Citation tallies for multiple DOIs."""
        return self._post("/tallies", json=dois)

    def get_tally_by_section(self, doi: str, use_view: Optional[bool] = None) -> dict:
        """GET /tallies/cited-by-sections/{doi} — Section-level tally for a DOI."""
        return self._get(f"/tallies/cited-by-sections/{doi}", {"use_view": use_view})

    def get_tallies_by_section(self, dois: list[str]) -> dict:
        """POST /tallies/cited-by-sections — Section-level tallies for multiple DOIs."""
        return self._post("/tallies/cited-by-sections", json=dois)

    def get_aggregate_tally(self, dois: list[str]) -> dict:
        """POST /tallies/aggregate — Aggregate tally across a list of DOIs."""
        return self._post("/tallies/aggregate", json=dois)

    # ------------------------------------------------------------------
    # Citations  (api_partner)
    # ------------------------------------------------------------------

    def get_citations_citing(
        self,
        doi: str,
        *,
        papers: Optional[bool] = None,
        tallies: Optional[bool] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> dict:
        """GET /api_partner/citations/citing/{doi} — Citations where *doi* is the target."""
        return self._get(
            f"/api_partner/citations/citing/{doi}",
            {"papers": papers, "tallies": tallies, "limit": limit, "offset": offset},
        )

    def get_citations_cited_by(
        self,
        doi: str,
        *,
        papers: Optional[bool] = None,
        tallies: Optional[bool] = None,
    ) -> dict:
        """GET /api_partner/citations/cited_by/{doi} — Citations where *doi* is the source."""
        return self._get(
            f"/api_partner/citations/cited_by/{doi}",
            {"papers": papers, "tallies": tallies},
        )

    # ------------------------------------------------------------------
    # References  (api_partner)
    # ------------------------------------------------------------------

    def get_references_to(
        self,
        doi: str,
        *,
        papers: Optional[bool] = None,
        tallies: Optional[bool] = None,
        source_dois: Optional[list[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> dict:
        """GET /api_partner/references/references_to/{doi} — Distinct references toward a DOI."""
        params: dict[str, Any] = {
            "papers": papers,
            "tallies": tallies,
            "limit": limit,
            "offset": offset,
        }
        if source_dois:
            params["source_doi[]"] = source_dois
        return self._get(f"/api_partner/references/references_to/{doi}", params)

    def get_references_from(
        self,
        doi: str,
        *,
        papers: Optional[bool] = None,
        tallies: Optional[bool] = None,
    ) -> dict:
        """GET /api_partner/references/references_from/{doi} — Distinct references from a DOI."""
        return self._get(
            f"/api_partner/references/references_from/{doi}",
            {"papers": papers, "tallies": tallies},
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        term: str,
        *,
        mode: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        sort: Optional[str] = None,
        sort_order: Optional[str] = None,
        title: Optional[str] = None,
        abstract: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        citation_types: Optional[str] = None,
        has_retraction: Optional[bool] = None,
        has_concern: Optional[bool] = None,
        has_correction: Optional[bool] = None,
        has_erratum: Optional[bool] = None,
        has_withdrawn: Optional[bool] = None,
        has_tally: Optional[bool] = None,
        supporting_from: Optional[int] = None,
        supporting_to: Optional[int] = None,
        mentioning_from: Optional[int] = None,
        mentioning_to: Optional[int] = None,
        contrasting_from: Optional[int] = None,
        contrasting_to: Optional[int] = None,
        citing_publications_from: Optional[int] = None,
        citing_publications_to: Optional[int] = None,
        author: Optional[str] = None,
        authors: Optional[str] = None,
        journal: Optional[str] = None,
        journals: Optional[str] = None,
        publisher: Optional[str] = None,
        section: Optional[str] = None,
        sections: Optional[str] = None,
        paper_type: Optional[str] = None,
        paper_types: Optional[str] = None,
        affiliation: Optional[str] = None,
        affiliations: Optional[str] = None,
        topic: Optional[str] = None,
        topics: Optional[str] = None,
        substances: Optional[str] = None,
        mesh_type: Optional[str] = None,
        compute_aggregations: Optional[bool] = None,
        aggregations: Optional[str] = None,
        format: Optional[str] = None,
    ) -> dict:
        """GET /api_partner/search — Full-text search over scite's publication index."""
        return self._get(
            "/api_partner/search",
            {
                "term": term,
                "mode": mode,
                "limit": limit,
                "offset": offset,
                "sort": sort,
                "sort_order": sort_order,
                "title": title,
                "abstract": abstract,
                "date_from": date_from,
                "date_to": date_to,
                "citation_types": citation_types,
                "has_retraction": has_retraction,
                "has_concern": has_concern,
                "has_correction": has_correction,
                "has_erratum": has_erratum,
                "has_withdrawn": has_withdrawn,
                "has_tally": has_tally,
                "supporting_from": supporting_from,
                "supporting_to": supporting_to,
                "mentioning_from": mentioning_from,
                "mentioning_to": mentioning_to,
                "contrasting_from": contrasting_from,
                "contrasting_to": contrasting_to,
                "citing_publications_from": citing_publications_from,
                "citing_publications_to": citing_publications_to,
                "author": author,
                "authors": authors,
                "journal": journal,
                "journals": journals,
                "publisher": publisher,
                "section": section,
                "sections": sections,
                "paper_type": paper_type,
                "paper_types": paper_types,
                "affiliation": affiliation,
                "affiliations": affiliations,
                "topic": topic,
                "topics": topics,
                "substances": substances,
                "mesh_type": mesh_type,
                "compute_aggregations": compute_aggregations,
                "aggregations": aggregations,
                "format": format,
            },
        )

    def search_v2(
        self,
        term: str,
        *,
        mode: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        sort: Optional[str] = None,
        sort_order: Optional[str] = None,
        title: Optional[str] = None,
        abstract: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        citation_types: Optional[str] = None,
        has_retraction: Optional[bool] = None,
        has_concern: Optional[bool] = None,
        has_correction: Optional[bool] = None,
        has_erratum: Optional[bool] = None,
        has_withdrawn: Optional[bool] = None,
        has_tally: Optional[bool] = None,
        supporting_from: Optional[int] = None,
        supporting_to: Optional[int] = None,
        mentioning_from: Optional[int] = None,
        mentioning_to: Optional[int] = None,
        contrasting_from: Optional[int] = None,
        contrasting_to: Optional[int] = None,
        citing_publications_from: Optional[int] = None,
        citing_publications_to: Optional[int] = None,
        author: Optional[str] = None,
        authors: Optional[str] = None,
        journal: Optional[str] = None,
        journals: Optional[str] = None,
        publisher: Optional[str] = None,
        section: Optional[str] = None,
        sections: Optional[str] = None,
        paper_type: Optional[str] = None,
        paper_types: Optional[str] = None,
        affiliation: Optional[str] = None,
        affiliations: Optional[str] = None,
        topic: Optional[str] = None,
        topics: Optional[str] = None,
        substances: Optional[str] = None,
        mesh_type: Optional[str] = None,
        compute_aggregations: Optional[bool] = None,
        aggregations: Optional[str] = None,
        format: Optional[str] = None,
    ) -> dict:
        """GET /search/v2 — Search v2 with identical filter surface to :meth:`search`."""
        return self._get(
            "/search/v2",
            {
                "term": term,
                "mode": mode,
                "limit": limit,
                "offset": offset,
                "sort": sort,
                "sort_order": sort_order,
                "title": title,
                "abstract": abstract,
                "date_from": date_from,
                "date_to": date_to,
                "citation_types": citation_types,
                "has_retraction": has_retraction,
                "has_concern": has_concern,
                "has_correction": has_correction,
                "has_erratum": has_erratum,
                "has_withdrawn": has_withdrawn,
                "has_tally": has_tally,
                "supporting_from": supporting_from,
                "supporting_to": supporting_to,
                "mentioning_from": mentioning_from,
                "mentioning_to": mentioning_to,
                "contrasting_from": contrasting_from,
                "contrasting_to": contrasting_to,
                "citing_publications_from": citing_publications_from,
                "citing_publications_to": citing_publications_to,
                "author": author,
                "authors": authors,
                "journal": journal,
                "journals": journals,
                "publisher": publisher,
                "section": section,
                "sections": sections,
                "paper_type": paper_type,
                "paper_types": paper_types,
                "affiliation": affiliation,
                "affiliations": affiliations,
                "topic": topic,
                "topics": topics,
                "substances": substances,
                "mesh_type": mesh_type,
                "compute_aggregations": compute_aggregations,
                "aggregations": aggregations,
                "format": format,
            },
        )

    # ------------------------------------------------------------------
    # Assistant  (api_partner)
    # ------------------------------------------------------------------

    def poll_assistant(
        self,
        user_input: str,
        turns: list[dict],
        *,
        session_id: Optional[str] = None,
        always_use_references: bool = False,
        never_use_references: bool = False,
        full_texts_only: bool = False,
        abstracts_only: bool = False,
        num_references: int = 25,
        year_from: Optional[str] = None,
        year_to: Optional[str] = None,
        rank_by: str = "all",
        topics: Optional[list[str]] = None,
        citation_sections: Optional[list[str]] = None,
        publication_types: Optional[list[str]] = None,
        journals: Optional[list[str]] = None,
        dashboards: Optional[list[str]] = None,
        reference_checks: Optional[list[str]] = None,
        answer_length: str = "medium",
        dois: Optional[list[str]] = None,
        use_structured_response: bool = False,
    ) -> dict:
        """POST /api_partner/assistant/poll — Start an assistant task and return its ``id``.

        Poll the returned task id with :meth:`get_assistant_task` until
        ``status == 'SUCCESS'``.
        """
        body: dict[str, Any] = {
            "userInput": user_input,
            "turns": turns,
            "alwaysUseReferences": always_use_references,
            "neverUseReferences": never_use_references,
            "fullTextsOnly": full_texts_only,
            "abstractsOnly": abstracts_only,
            "numReferences": num_references,
            "rankBy": rank_by,
            "answerLength": answer_length,
            "useStructuredResponse": use_structured_response,
            "topics": topics or [],
            "citationSections": citation_sections or [],
            "publicationTypes": publication_types or [],
            "journals": journals or [],
            "dashboards": dashboards or [],
            "referenceChecks": reference_checks or [],
            "dois": dois or [],
        }
        if session_id is not None:
            body["sessionId"] = session_id
        if year_from is not None:
            body["yearFrom"] = year_from
        if year_to is not None:
            body["yearTo"] = year_to
        return self._post("/api_partner/assistant/poll", json=body)

    def get_assistant_task(self, task_id: str) -> dict:
        """GET /api_partner/assistant/tasks/{task_id} — Poll an assistant task by id."""
        return self._get(f"/api_partner/assistant/tasks/{task_id}")

    # ------------------------------------------------------------------
    # Recommend papers  (api_partner)
    # ------------------------------------------------------------------

    def recommend_papers(self, doi: str) -> dict:
        """GET /api_partner/recommend-papers/{doi} — Papers recommended to read given a DOI."""
        return self._get(f"/api_partner/recommend-papers/{doi}")

    # ------------------------------------------------------------------
    # Collections / Dashboards  (api_partner)
    # ------------------------------------------------------------------

    def create_collection(
        self,
        name: str,
        email: str,
        *,
        description: Optional[str] = None,
        is_public: bool = False,
        is_hidden: bool = False,
        is_shared_with_org: bool = False,
        dois: Optional[list[str]] = None,
        query: Optional[str] = None,
        enable_citation_alerts: bool = False,
    ) -> dict:
        """POST /api_partner/collections/create — Create a custom collection (Dashboard)."""
        body: dict[str, Any] = {
            "name": name,
            "email": email,
            "isPublic": is_public,
            "isHidden": is_hidden,
            "isSharedWithOrg": is_shared_with_org,
            "enableDashboardCitationAlerts": enable_citation_alerts,
        }
        if description is not None:
            body["description"] = description
        if dois is not None:
            body["dois"] = dois
        if query is not None:
            body["query"] = query
        return self._post("/api_partner/collections/create", json=body)

    def get_collection(self, collection_slug: str) -> dict:
        """GET /api_partner/collections/{collection_slug} — Retrieve a collection by slug."""
        return self._get(f"/api_partner/collections/{collection_slug}")

    def update_collection(
        self,
        collection_slug: str,
        name: str,
        email: str,
        dois: list[str],
        *,
        description: Optional[str] = None,
        is_public: bool = False,
        is_shared_with_org: bool = False,
        enable_citation_alerts: bool = False,
    ) -> dict:
        """PUT /api_partner/collections/{collection_slug} — Update a collection by slug."""
        body: dict[str, Any] = {
            "name": name,
            "email": email,
            "dois": dois,
            "isPublic": is_public,
            "isSharedWithOrg": is_shared_with_org,
            "enableDashboardCitationAlerts": enable_citation_alerts,
        }
        if description is not None:
            body["description"] = description
        return self._put(f"/api_partner/collections/{collection_slug}", json=body)

    def delete_collection(self, collection_slug: str, email: str) -> dict:
        """DELETE /api_partner/collections/{collection_slug} — Delete a collection by slug."""
        return self._delete(
            f"/api_partner/collections/{collection_slug}", {"email": email}
        )

    # ------------------------------------------------------------------
    # Reference check
    # ------------------------------------------------------------------

    def submit_reference_check(
        self,
        *,
        file_url: Optional[str] = None,
        pdf_url: Optional[str] = None,
        upload_for: Optional[str] = None,
        file: Optional[Any] = None,
        pdf: Optional[Any] = None,
    ) -> dict:
        """POST /reference_check — Submit a document for reference checking.

        Pass either URL parameters (``file_url`` / ``pdf_url``) or open file
        objects (``file`` / ``pdf``) for multipart upload.
        """
        params = _drop_nones(
            {"file_url": file_url, "pdf_url": pdf_url, "upload_for": upload_for}
        )
        files: Optional[dict] = None
        if file is not None or pdf is not None:
            files = {}
            if file is not None:
                files["file"] = file
            if pdf is not None:
                files["pdf"] = pdf
        return self._post("/reference_check", files=files, params=params)

    def get_reference_check_result(self, task_id: str) -> dict:
        """GET /reference_check/tasks/{task_id} — Result for a reference check task."""
        return self._get(f"/reference_check/tasks/{task_id}")

    def cancel_reference_check(self, task_id: str) -> dict:
        """GET /reference_check/tasks/{task_id}/cancel — Cancel a running reference check."""
        return self._get(f"/reference_check/tasks/{task_id}/cancel")

    def get_reference_check_result_url(self, task_id: str) -> dict:
        """GET /reference_check/tasks/{task_id}/result_url — Download URL for reference check output."""
        return self._get(f"/reference_check/tasks/{task_id}/result_url")

    # ------------------------------------------------------------------
    # Authors
    # ------------------------------------------------------------------

    def get_author_papers(
        self,
        author_slug: str,
        *,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> dict:
        """GET /authors/{author_slug}/papers — Author metadata and their papers."""
        return self._get(
            f"/authors/{author_slug}/papers", {"offset": offset, "limit": limit}
        )

    def get_author_stats(self, author_slug: str) -> dict:
        """GET /authors/{author_slug}/stats — Aggregated citation statistics for an author."""
        return self._get(f"/authors/{author_slug}/stats")


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _drop_nones(d: dict) -> dict:
    """Return a copy of *d* with all ``None`` values removed."""
    return {k: v for k, v in d.items() if v is not None}
