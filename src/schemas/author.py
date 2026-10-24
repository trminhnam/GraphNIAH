from pydantic import BaseModel, Field


class Author(BaseModel):
    name: str = Field(default="")
    url: str = Field(default="")
    aliases: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return self.model_dump()

    def to_json(self, **kwargs) -> str:
        return self.model_dump_json(**kwargs)

    @classmethod
    def from_hit(cls, hit: dict) -> "Author":
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
            name=info.get("name", ""),
            url=info.get("url", ""),
            aliases=aliases_raw,
            notes=[n if isinstance(n, str) else n.get("text", "") for n in notes_raw],
        )

    def __repr__(self) -> str:
        return f"""
Author(
    name={self.name},
    url={self.url},
    aliases={self.aliases},
    notes={self.notes}
)
"""
