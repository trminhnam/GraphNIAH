# @dataclass
# class DBLPVenue:
#     """Metadata for a single DBLP venue hit."""

#     venue: str
#     acronym: str
# type: str  # "Conference", "Journal", "Workshop", etc.
#     url: str

#     def to_dict(self) -> dict:
#         return asdict(self)

#     def to_json(self, **kwargs) -> str:
#         return orjson.dumps(self.to_dict()).decode("utf-8")

#     @classmethod
#     def from_hit(cls, hit: dict) -> "DBLPVenue":
#         info = hit.get("info", {})
#         return cls(
#             venue=info.get("venue", ""),
#             acronym=info.get("acronym", ""),
#             type=info.get("type", ""),
#             url=info.get("url", ""),
#         )

#     def __str__(self) -> str:
#         return f"""
# DBLPVenue(
#     venue={self.venue},
#     acronym={self.acronym},
#     type={self.type},
#     url={self.url}
# )
# """


from pydantic import BaseModel, Field


class Venue(BaseModel):
    venue: str
    acronym: str
    type: str  # "Conference", "Journal", "Workshop", etc.
    url: str

    def to_dict(self) -> dict:
        return self.model_dump()

    def to_json(self, **kwargs) -> str:
        return self.model_dump_json(**kwargs)

    @classmethod
    def from_hit(cls, hit: dict) -> "Venue":
        return cls(
            venue=hit.get("venue", ""),
            acronym=hit.get("acronym", ""),
            type=hit.get("type", ""),
            url=hit.get("url", ""),
        )

    def __repr__(self) -> str:
        return f"""
Venue(
    venue={self.venue},
    acronym={self.acronym},
    type={self.type},
    url={self.url}
)
"""
