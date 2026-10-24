"""Offline-first arXiv client with optional online fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from loguru import logger

from .arxiv_bulk_client import ArxivBulkClient
from .arxiv_client import (
    ArxivOnlineClient,
    ArxivPaperMetadata,
    SortCriterion,
    SortOrder,
)


def normalize_arxiv_id(raw: str) -> str:
    """Return a bare arXiv identifier without URL or version suffix."""
    bare = raw.split("/abs/")[-1].split("arxiv.org/")[-1]
    if "v" in bare.split(".")[-1]:
        bare = bare.rsplit("v", 1)[0]
    return bare


class ArxivFallbackClient:
    """Search the local snapshot first, then optionally fall back online."""

    def __init__(
        self,
        bulk_client: ArxivBulkClient,
        online_client: ArxivOnlineClient | None,
    ) -> None:
        self.bulk_client = bulk_client
        self.online_client = online_client

    def close(self) -> None:
        """Release resources held by the underlying clients."""
        close_fn = getattr(self.bulk_client, "close", None)
        if callable(close_fn):
            close_fn()

    def search(
        self,
        query: str,
        *,
        max_results: int = 100,
        sort_by: SortCriterion = SortCriterion.Relevance,
        sort_order: SortOrder = SortOrder.Descending,
    ) -> list[ArxivPaperMetadata]:
        """Search the snapshot first and use the online API when it is enabled."""
        results = self.bulk_client.search(
            query,
            max_results=max_results,
            sort_order=sort_order,
        )
        if results:
            return results

        if self.online_client is None:
            logger.info(
                "Bulk arXiv search returned no results for {!r}; online fallback is disabled.",
                query,
            )
            return []

        logger.info(
            "Bulk arXiv search returned no results for {!r}; falling back to the online client.",
            query,
        )
        return self.online_client.search(
            query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    def get_paper(self, arxiv_id: str) -> ArxivPaperMetadata:
        """Fetch metadata by arXiv ID, preferring the local snapshot."""
        try:
            return self.bulk_client.get_paper(arxiv_id)
        except Exception as exc:
            if self.online_client is None:
                logger.info(
                    "Bulk arXiv metadata lookup failed for {}: {}. Online fallback is disabled.",
                    arxiv_id,
                    exc,
                )
                raise
            logger.info(
                "Bulk arXiv metadata lookup failed for {}: {}. Falling back to the online client.",
                arxiv_id,
                exc,
            )
            return self.online_client.get_paper(arxiv_id)

    def get_papers(self, arxiv_ids: list[str]) -> list[ArxivPaperMetadata]:
        """Fetch a batch of arXiv papers, filling snapshot misses online."""
        papers = self.bulk_client.get_papers(arxiv_ids)
        papers_by_id = {normalize_arxiv_id(paper.arxiv_id): paper for paper in papers}

        missing_ids = [
            raw
            for raw in arxiv_ids
            if normalize_arxiv_id(raw) not in papers_by_id
        ]
        if missing_ids:
            if self.online_client is None:
                logger.info(
                    "Bulk arXiv metadata lookup missed {} paper(s); online fallback is disabled.",
                    len(missing_ids),
                )
            else:
                logger.info(
                    "Bulk arXiv metadata lookup missed {} paper(s); falling back to the online client.",
                    len(missing_ids),
                )
                for paper in self.online_client.get_papers(missing_ids):
                    papers_by_id.setdefault(normalize_arxiv_id(paper.arxiv_id), paper)

        ordered_ids = [normalize_arxiv_id(raw) for raw in arxiv_ids]
        return [
            papers_by_id[arxiv_id]
            for arxiv_id in ordered_ids
            if arxiv_id in papers_by_id
        ]

    def crawl(self, *args, **kwargs):
        """Delegate crawl operations to the local snapshot client."""
        return self.bulk_client.crawl(*args, **kwargs)

    def check_source_available(self, arxiv_id: str) -> bool:
        """Check source availability offline-first, then online."""
        if self.bulk_client.check_source_available(arxiv_id):
            return True
        if self.online_client is None:
            logger.info(
                "Bulk arXiv source check was unavailable for {}; online fallback is disabled.",
                arxiv_id,
            )
            return False
        logger.info(
            "Bulk arXiv source check was unavailable for {}; falling back to the online client.",
            arxiv_id,
        )
        return self.online_client.check_source_available(arxiv_id)

    def download_source(
        self,
        arxiv_id: str,
        output_dir: str | Path = ".",
        *,
        skip_if_exists: bool = True,
    ) -> Path | None:
        """Download source files offline-first, then online."""
        try:
            downloaded = self.bulk_client.download_source(
                arxiv_id,
                output_dir=output_dir,
                skip_if_exists=skip_if_exists,
            )
        except Exception as exc:
            if self.online_client is None:
                logger.info(
                    "Bulk arXiv source download failed for {}: {}. Online fallback is disabled.",
                    arxiv_id,
                    exc,
                )
                return None
            logger.info(
                "Bulk arXiv source download failed for {}: {}. Falling back to the online client.",
                arxiv_id,
                exc,
            )
        else:
            if downloaded is not None:
                return downloaded
            if self.online_client is None:
                logger.info(
                    "Bulk arXiv source download returned no file for {}; online fallback is disabled.",
                    arxiv_id,
                )
                return None
            logger.info(
                "Bulk arXiv source download returned no file for {}; falling back to the online client.",
                arxiv_id,
            )

        return self.online_client.download_source(
            arxiv_id,
            output_dir=output_dir,
            skip_if_exists=skip_if_exists,
        )

    def download_pdf(
        self,
        arxiv_id: str,
        output_dir: str | Path = ".",
        pdf_download_from: Literal["arxiv", "gcs"] = "arxiv",
    ) -> Path | None:
        """Download the PDF offline-first, then online when needed."""
        downloaded = self.bulk_client.download_pdf(
            arxiv_id,
            output_dir=output_dir,
            pdf_download_from=pdf_download_from,
        )
        if downloaded is not None:
            return downloaded
        if pdf_download_from != "arxiv":
            return None
        if self.online_client is None:
            logger.info(
                "Bulk arXiv PDF download returned no file for {}; online fallback is disabled.",
                arxiv_id,
            )
            return None

        logger.info(
            "Bulk arXiv PDF download returned no file for {}; falling back to the online client.",
            arxiv_id,
        )
        return self.online_client.download_pdf(arxiv_id, output_dir=output_dir)

    @staticmethod
    def get_result_schema() -> type[ArxivPaperMetadata]:
        return ArxivOnlineClient.get_result_schema()

    @staticmethod
    def save_jsonl(papers: list[ArxivPaperMetadata], path: str | Path) -> None:
        ArxivOnlineClient.save_jsonl(papers, path)

    @staticmethod
    def load_jsonl(path: str | Path) -> list[ArxivPaperMetadata]:
        return ArxivOnlineClient.load_jsonl(path)

    @staticmethod
    def save_csv(papers: list[ArxivPaperMetadata], path: str | Path) -> None:
        ArxivOnlineClient.save_csv(papers, path)
