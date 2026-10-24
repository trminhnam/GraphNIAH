"""
Extract the references section from the document.

python -m src.utils.preprocessing.extract_reference_section
"""

import re

REFERENCES_HEADING_PATTERN = re.compile(
    r"^#+\s*References\s*$", re.MULTILINE | re.IGNORECASE
)

_HEADING_KEYWORDS = (
    r"Appendix|Appendices|Acknowledgement|Acknowledgements|Supplementary"
    r"|Attention Visualizations|Contents|Abstract|Introduction"
    r"|Figure|Limitation|Ethic|Checklist|Author|License"
    r"|Broader Impact|NeurIPS|Paper Checklist|Competing Interests"
)

_FLOAT_CAPTIONS = (
    r"\n(?:Table|Figure|Fig\.|Fig|Algorithm|Listing|Equation|Example)\s+\d+"
)

NEXT_SECTION_PATTERN = re.compile(
    rf"\n#+\s*({_HEADING_KEYWORDS})"
    # Generic markdown heading — require a letter start so that
    # ACM article:page artifacts like "## 12:13" are not matched.
    r"|##\s+[A-Za-z]\w+"
    rf"|{_FLOAT_CAPTIONS}"
    r"|\n!\["
    # ACM-style received/accepted metadata line that follows references.
    r"|\nReceived\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}",
    re.IGNORECASE,
)

# ── fallback heuristic patterns ──────────────────────────────────────

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")
_PARENS_RE = re.compile(r"\([^)]*\)")
_VENUE_RE = re.compile(
    r"(?:proceedings|conference|journal|arxiv|workshop|symposium"
    r"|transactions|advances in neural|volume\s+\d|pages?\s+\d"
    r"|preprint|technical report)",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^#+\s")
_FLOAT_RE = re.compile(
    r"^(?:Table|Figure|Fig\.|Algorithm|Listing)\s+\d+", re.IGNORECASE
)

_MIN_CLUSTER_SIZE = 5
_MAX_GAP = 1


def split_references_section(document: str) -> tuple[str, str, str]:
    """
    Split *document* into ``(before_refs, refs_section, after_refs)``.

    The section boundary search follows the same heading and fallback logic as
    ``extract_references_section`` so downstream consumers can either keep or
    remove references using a single source of truth.
    """
    match = REFERENCES_HEADING_PATTERN.search(document)

    if not match:
        references_section = _fallback_find_references(document)
        if not references_section:
            return document, "", ""
        start_pos = document.find(references_section)
        if start_pos < 0:
            return document, "", ""
        end_pos = start_pos + len(references_section)
        return document[:start_pos], document[start_pos:end_pos], document[end_pos:]

    start_pos = match.start()
    remaining_text = document[start_pos:]

    lines = remaining_text.split("\n")
    if len(lines) > 1:
        rest_of_doc = "\n".join(lines[1:])
        next_section_match = NEXT_SECTION_PATTERN.search(rest_of_doc)
    else:
        next_section_match = None

    if next_section_match:
        end_pos = start_pos + len(lines[0]) + 1 + next_section_match.start()
        return (
            document[:start_pos],
            document[start_pos:end_pos].strip(),
            document[end_pos:],
        )

    return document[:start_pos], document[start_pos:].strip(), ""


def extract_references_section(document: str) -> str:
    """
    Extract the references section from the document.
    Returns the full text of the References section from the heading onwards.
    If no References section is found, falls back to a heuristic that
    locates the longest contiguous block of reference-like paragraphs.
    """
    _before_refs, references_section, _after_refs = split_references_section(document)
    return references_section


def remove_references_section(document: str) -> str:
    """Return *document* with the detected references section removed."""
    before_refs, _references_section, after_refs = split_references_section(document)
    return "\n\n".join(
        part.strip() for part in (before_refs, after_refs) if part.strip()
    )


# ── fallback: heading-less reference detection ───────────────────────


def _is_reference_like(paragraph: str) -> bool:
    """Return True if *paragraph* looks like one or more reference entries."""
    text = paragraph.strip()
    if not text or len(text) < 15:
        return False
    if _HEADING_RE.match(text):
        return False
    if _FLOAT_RE.match(text):
        return False
    if text.startswith("|") and text.endswith("|"):
        return False

    # Years that survive after stripping parenthetical citations are "bare"
    # years typical of reference listings (e.g. "Author. 2020. Title.").
    # Body text embeds years inside citation parens: "(Smith et al. 2020)".
    text_no_parens = _PARENS_RE.sub("", text)
    has_bare_year = bool(_YEAR_RE.search(text_no_parens))
    if not has_bare_year:
        return False

    has_venue = bool(_VENUE_RE.search(text))
    ends_period = text.rstrip().endswith(".")

    return has_venue or ends_period


def _fallback_find_references(document: str) -> str:
    """
    Scan for the longest contiguous block of reference-like paragraphs.
    Tolerates small gaps (up to _MAX_GAP non-matching paragraphs) to
    handle stray footnotes or page numbers between references.
    Returns empty string if no cluster of at least _MIN_CLUSTER_SIZE
    reference-like paragraphs is found.
    """
    paragraphs = re.split(r"\n\s*\n", document)
    is_ref = [_is_reference_like(p) for p in paragraphs]

    best_start, best_end = 0, 0
    i = 0
    while i < len(is_ref):
        if not is_ref[i]:
            i += 1
            continue

        start = i
        gap = 0
        while i < len(is_ref):
            if is_ref[i]:
                gap = 0
                i += 1
            elif gap < _MAX_GAP:
                gap += 1
                i += 1
            else:
                break

        end = i - gap

        ref_count = sum(is_ref[start:end])
        best_count = sum(is_ref[best_start:best_end])
        if ref_count > best_count:
            best_start, best_end = start, end

    if sum(is_ref[best_start:best_end]) < _MIN_CLUSTER_SIZE:
        return ""

    return "\n\n".join(paragraphs[best_start:best_end]).strip()


if __name__ == "__main__":
    import os
    import sys

    parsed_pdfs_dir = "./output/cleaned_pdfs/docling"
    parsed_references_section_dir = "./output/parsed_references/docling"
    os.makedirs(parsed_references_section_dir, exist_ok=True)
    for venue in os.listdir(parsed_pdfs_dir):
        venue_dir = os.path.join(parsed_pdfs_dir, venue)
        venue_references_section_dir = os.path.join(
            parsed_references_section_dir, venue
        )
        os.makedirs(venue_references_section_dir, exist_ok=True)
        if not os.path.isdir(venue_dir):
            continue

        for document_name in os.listdir(venue_dir):
            if not document_name.endswith(".md"):
                continue

            document_path = os.path.join(venue_dir, document_name)
            with open(document_path, "r") as f:
                document = f.read()

            references_section = extract_references_section(document)
            with open(
                os.path.join(venue_references_section_dir, document_name), "w"
            ) as f:
                f.write(references_section)
            print(f"Extracted references section from {document_name}")
