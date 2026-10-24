from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from .author import Author
from .reference import Reference

if TYPE_CHECKING:
    import arxiv


class Paper(BaseModel):
    arxiv_id: str
    title: str
    abstract: str = Field(default="")
    authors: List[Author] = Field(default_factory=list)
    primary_category: str = Field(default="")
    categories: List[str] = Field(default_factory=list)
    published: str = Field(default="")
    updated: str = Field(default="")
    pdf_url: str = Field(default="")
    doi: str = Field(default="")
    journal_ref: str = Field(default="")
    comment: str = Field(default="")
    entry_url: str = Field(default="")
    venue: str = Field(default="")
    source_path: str = Field(
        default="", description="Path to the source code of the paper."
    )
    pdf_path: str = Field(default="", description="Path to the PDF of the paper.")
    extracted_text: Optional[Union[str, dict]] = Field(
        default=None,
        nullable=True,
        description="Extracted text of the paper. It can be markdown, json, or html.",
    )
    references: List[Reference] = Field(default_factory=list)

    # @field_validator("authors", mode="before")
    # @classmethod
    # def coerce_authors(cls, v: list) -> list:
    #     return [Author(name=item) if isinstance(item, str) else item for item in v]

    def to_dict(self) -> dict:
        return self.model_dump()

    def to_json(self, **kwargs) -> str:
        return self.model_dump_json(**kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "Paper":
        return cls.model_validate(data)

    def __repr__(self) -> str:
        return f"""
Paper(
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
    comment={self.comment},
    entry_url={self.entry_url},
    venue={self.venue},
    extracted_text={self.extracted_text},
    source_path={self.source_path},
    pdf_path={self.pdf_path}
)
"""
