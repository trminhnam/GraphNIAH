from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .text import join_names_text, ordinal, quoted_title, unpack_author_pair, unpack_author_triplet

if TYPE_CHECKING:
    from .ambiguity import AmbiguityChecker


MAX_AUTHOR_PAIR_CANDIDATES = 10
MAX_AUTHOR_TRIPLET_CANDIDATES = 10


def descriptor_text(kind: str, value: Any) -> str:
    """Render a descriptor in predicate form for backward relative clauses."""
    if kind == "title":
        return f"is titled {quoted_title(str(value))}"
    if kind == "venue":
        return f"appears in {value}"
    if kind == "published_year_month":
        return f"was published in {value}"
    if kind == "first_author":
        return f"has {value} as first author"
    if kind == "title_first_author":
        title = value.get("title", "") if isinstance(value, dict) else ""
        first_author = value.get("first_author", "") if isinstance(value, dict) else ""
        return f"is titled {quoted_title(title)} and has {first_author} as first author"
    if kind == "some_author":
        return f"has {value} as one of the authors"
    if kind == "some_author_pair":
        author_one, author_two = unpack_author_pair(value)
        return f"has {author_one} and {author_two} as two of the authors"
    if kind == "some_author_triplet":
        author_one, author_two, author_three = unpack_author_triplet(value)
        return (
            f"has {join_names_text([author_one, author_two, author_three])} "
            "as three of the authors"
        )
    if kind == "name":
        return f"is named {value}"
    if kind == "author_position":
        return f"is the {ordinal(int(value))} author"
    raise ValueError(f"Unsupported descriptor kind: {kind}")


def paper_venue_descriptor_value(node: dict) -> str | None:
    """Return a usable venue descriptor value for a paper node."""
    venue = str(node.get("venue") or "").strip()
    if not venue:
        return None
    if venue.lower() in {"arxiv", "arxiv.org"}:
        return None
    return venue


def paper_published_year_month_descriptor_value(node: dict) -> str | None:
    """Return a usable YYYY-MM publication descriptor value for a paper node."""
    published = str(node.get("published") or "").strip()
    if len(published) < 7:
        return None

    value = published[:7]
    if value[4] != "-":
        return None

    year = value[:4]
    month = value[5:7]
    if not year.isdigit() or not month.isdigit():
        return None
    if not 1 <= int(month) <= 12:
        return None
    return value


def describe_anchor(node: dict) -> tuple[str, str, str] | None:
    """Describe the anchor node with its strongest identifier."""
    if node["node_type"] == "paper":
        title = node.get("title", "")
        if title:
            return (descriptor_text("title", title), "title", title)
        return None
    name = node.get("name", "")
    if name:
        return (descriptor_text("name", name), "name", name)
    return None


def describe_intermediate(
    node: dict,
    avoid_author: str | None = None,
    *,
    allow_first_author: bool = True,
) -> tuple[str, str, str] | None:
    """Describe an intermediate node without revealing its title."""
    if node["node_type"] == "paper":
        authors = node.get("authors") or []
        if (
            allow_first_author
            and authors
            and authors[0]
            and (avoid_author is None or authors[0] != avoid_author)
        ):
            return (
                descriptor_text("first_author", authors[0]),
                "first_author",
                authors[0],
            )

        eligible = [author for author in authors if author and (avoid_author is None or author != avoid_author)]
        if eligible:
            return (
                descriptor_text("some_author", eligible[0]),
                "some_author",
                eligible[0],
            )
        return None

    name = node.get("name", "")
    if not name:
        return None
    if avoid_author and name == avoid_author:
        return None
    return (descriptor_text("name", name), "name", name)


def paper_descriptor_candidates(
    node: dict,
    *,
    allow_title: bool = True,
    allow_first_author: bool = True,
    allow_title_first_author: bool = True,
    allow_some_author: bool = False,
    allow_some_author_pair: bool = False,
    allow_some_author_triplet: bool = False,
    allow_venue: bool = False,
    allow_published_year_month: bool = False,
    avoid_first_author: str | None = None,
    avoid_author: str | None = None,
    prefer_title: bool = True,
) -> list[tuple[str, Any]]:
    """Return candidate descriptors for a paper node in preference order."""
    title = node.get("title", "")
    authors = node.get("authors") or []
    first_author = authors[0] if authors else ""
    venue = paper_venue_descriptor_value(node)
    published_year_month = paper_published_year_month_descriptor_value(node)

    candidates: list[tuple[str, Any]] = []
    strong_order = (
        ("title", "first_author", "title_first_author")
        if prefer_title
        else ("first_author", "title", "title_first_author")
    )

    for kind in strong_order:
        if kind == "title":
            if allow_title and title:
                candidates.append(("title", title))
            continue

        if kind == "first_author":
            if not allow_first_author or not first_author:
                continue
            if avoid_first_author is not None and first_author == avoid_first_author:
                continue
            if avoid_author is not None and first_author == avoid_author:
                continue
            candidates.append(("first_author", first_author))
            continue

        if kind == "title_first_author":
            if not allow_title_first_author or not title or not first_author:
                continue
            if avoid_first_author is not None and first_author == avoid_first_author:
                continue
            if avoid_author is not None and first_author == avoid_author:
                continue
            candidates.append(
                ("title_first_author", {"title": title, "first_author": first_author})
            )

    eligible_authors = [
        author
        for author in authors
        if author and (avoid_author is None or author != avoid_author)
    ]

    if allow_some_author:
        for author in eligible_authors:
            candidates.append(("some_author", author))

    if allow_some_author_pair and len(eligible_authors) >= 2:
        pair_count = 0
        for index, author_one in enumerate(eligible_authors):
            if pair_count >= MAX_AUTHOR_PAIR_CANDIDATES:
                break
            for author_two in eligible_authors[index + 1 :]:
                if pair_count >= MAX_AUTHOR_PAIR_CANDIDATES:
                    break
                candidates.append(("some_author_pair", (author_one, author_two)))
                pair_count += 1

    if allow_some_author_triplet and len(eligible_authors) >= 3:
        triplet_count = 0
        for first_index, author_one in enumerate(eligible_authors):
            if triplet_count >= MAX_AUTHOR_TRIPLET_CANDIDATES:
                break
            for second_index in range(first_index + 1, len(eligible_authors)):
                if triplet_count >= MAX_AUTHOR_TRIPLET_CANDIDATES:
                    break
                author_two = eligible_authors[second_index]
                for author_three in eligible_authors[second_index + 1 :]:
                    if triplet_count >= MAX_AUTHOR_TRIPLET_CANDIDATES:
                        break
                    candidates.append(
                        ("some_author_triplet", (author_one, author_two, author_three))
                    )
                    triplet_count += 1

    if allow_venue and venue is not None:
        candidates.append(("venue", venue))

    if allow_published_year_month and published_year_month is not None:
        candidates.append(("published_year_month", published_year_month))

    return candidates


def author_descriptor_candidates(
    node: dict,
    *,
    allow_name: bool = True,
    avoid_name: str | None = None,
    allow_position: bool = False,
    authorship_position: int | None = None,
    prefer_position: bool = False,
) -> list[tuple[str, Any]]:
    """Return candidate descriptors for an author node in preference order."""
    candidates: list[tuple[str, Any]] = []
    has_position = (
        allow_position
        and authorship_position is not None
        and int(authorship_position) > 0
    )

    if has_position and prefer_position:
        candidates.append(("author_position", int(authorship_position)))

    if allow_name:
        name = node.get("name", "")
        if name and (avoid_name is None or name != avoid_name):
            candidates.append(("name", name))

    if has_position and not prefer_position:
        candidates.append(("author_position", int(authorship_position)))

    return candidates


def choose_descriptor_candidate(
    candidates: list[tuple[str | None, Any | None]],
    *,
    checker: "AmbiguityChecker | None",
    anchor_node_id: str | None = None,
    source_node_id: str | None = None,
    edge_type: str | None = None,
    target_node_id: str | None = None,
) -> tuple[str | None, Any | None] | None:
    """Pick the first candidate that is unique under the available checker."""
    if not candidates:
        return None
    if checker is None:
        return candidates[0]

    for kind, value in candidates:
        if anchor_node_id is not None:
            if checker.descriptor_unique_for_anchor(anchor_node_id, kind, value):
                return kind, value
            continue

        if source_node_id is None or edge_type is None or target_node_id is None:
            raise ValueError(
                "Local descriptor selection requires source_node_id, edge_type, and target_node_id."
            )
        if checker.descriptor_unique_for_edge(
            source_node_id,
            edge_type,
            target_node_id,
            kind,
            value,
        ):
            return kind, value

    return None


def select_anchor_descriptor(
    node: dict,
    checker: "AmbiguityChecker | None" = None,
) -> tuple[str, Any] | None:
    """Choose the anchor descriptor, preferring globally unique identifiers."""
    if checker is None:
        result = describe_anchor(node)
        if result is None:
            return None
        return result[1], result[2]

    if node["node_type"] == "paper":
        candidates = paper_descriptor_candidates(
            node,
            allow_title=bool(node.get("title")),
            allow_first_author=bool(node.get("authors")),
            allow_title_first_author=True,
            allow_venue=False,
            allow_published_year_month=False,
            prefer_title=True,
        )
    else:
        candidates = author_descriptor_candidates(node)

    return choose_descriptor_candidate(
        candidates,
        checker=checker,
        anchor_node_id=node["node_id"],
    )


def select_local_descriptor(
    source_node_id: str,
    edge: dict,
    destination_node: dict,
    *,
    checker: "AmbiguityChecker | None",
    allow_title: bool = True,
    allow_first_author: bool = True,
    allow_title_first_author: bool = True,
    allow_some_author: bool = False,
    allow_some_author_pair: bool = False,
    allow_venue: bool = False,
    allow_published_year_month: bool = False,
    prefer_title: bool = True,
    avoid_first_author: str | None = None,
    avoid_author: str | None = None,
    allow_name: bool = True,
    avoid_name: str | None = None,
    allow_position: bool = False,
    prefer_position: bool = False,
    allow_none: bool = False,
) -> tuple[str | None, Any | None] | None:
    """Choose a locally unique descriptor for the destination of one hop."""
    if destination_node["node_type"] == "paper":
        candidates = paper_descriptor_candidates(
            destination_node,
            allow_title=allow_title,
            allow_first_author=allow_first_author,
            allow_title_first_author=allow_title_first_author,
            allow_some_author=allow_some_author,
            allow_some_author_pair=allow_some_author_pair,
            allow_venue=allow_venue,
            allow_published_year_month=allow_published_year_month,
            avoid_first_author=avoid_first_author,
            avoid_author=avoid_author,
            prefer_title=prefer_title,
        )
    else:
        position = edge.get("position") if edge.get("edge_type") == "authorship" else None
        candidates = author_descriptor_candidates(
            destination_node,
            allow_name=allow_name,
            avoid_name=avoid_name,
            allow_position=allow_position,
            authorship_position=position,
            prefer_position=prefer_position,
        )

    if allow_none:
        candidates.append((None, None))

    return choose_descriptor_candidate(
        candidates,
        checker=checker,
        source_node_id=source_node_id,
        edge_type=edge["edge_type"],
        target_node_id=destination_node["node_id"],
    )
