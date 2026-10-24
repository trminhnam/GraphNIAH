from __future__ import annotations

from typing import Any


ORDINALS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
}


def quoted_title(title: str) -> str:
    """Return a readable quoted title for surface text."""
    if '"' in title and "'" not in title:
        return f"'{title}'"
    if '"' in title and "'" in title:
        return title
    return f'"{title}"'


def join_names_text(names: list[str]) -> str:
    """Return author names joined as a natural-language list."""
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def unpack_author_pair(value: Any) -> tuple[str, str]:
    """Return the two author names carried by a ``some_author_pair`` value."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return str(value[0]), str(value[1])
    raise ValueError(f"some_author_pair requires a 2-element sequence, got {value!r}")


def unpack_author_triplet(value: Any) -> tuple[str, str, str]:
    """Return the three author names carried by a ``some_author_triplet`` value."""
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return str(value[0]), str(value[1]), str(value[2])
    raise ValueError(f"some_author_triplet requires a 3-element sequence, got {value!r}")


def ordinal(value: int) -> str:
    """Return an ordinal word or suffix form for *value*."""
    if value in ORDINALS:
        return ORDINALS[value]
    if 11 <= value % 100 <= 13:
        return f"{value}th"
    return f"{value}{({1: 'st', 2: 'nd', 3: 'rd'}).get(value % 10, 'th')}"
