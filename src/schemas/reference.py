from pydantic import BaseModel, Field


class Reference(BaseModel):
    title: str = Field(default="")
    authors: list[str] = Field(default_factory=list)
    year: str = Field(default="")
    venue: str = Field(default="")
    raw: str = Field(default="", description="Raw text of the reference.")

    def to_dict(self) -> dict:
        return self.model_dump()

    def to_json(self, **kwargs) -> str:
        return self.model_dump_json(**kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "Reference":
        return cls.model_validate(data)

    def __repr__(self) -> str:
        return f"""
Reference(
    title={self.title},
    authors={self.authors},
    year={self.year},
    venue={self.venue},
    raw={self.raw}
)
"""
