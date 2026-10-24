"""
Parse references from a PDF file.

Extracts the references/bibliography section from an academic PDF and returns
a list of individual reference strings.

Supports common reference numbering styles:
  - Numeric brackets:  [1], [2], ...
  - Numeric dots:      1., 2., ...
  - Parenthesised:     (1), (2), ...
  - Author-year keys:  [Vaswani2017], [LeCun et al., 2015], ...

Usage::

    refs = parse_references_from_pdf("path/to/paper.pdf")
    for ref in refs:
        print(ref)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SECTION_HEADERS = re.compile(
    r"^\s*(?:\d+[\.\s]+)?"  # optional leading section number
    r"(references?|bibliography|works\s+cited)"
    r"\s*$",
    re.IGNORECASE,
)

# Patterns that mark the start of a new reference entry.
# Ordered from most specific to most general.
_REF_START_PATTERNS: list[re.Pattern] = [
    # [1], [12], ...
    re.compile(r"^\s*\[\d{1,3}\]\s+\S"),
    # [AuthorYear] / [Author et al., 2020]
    re.compile(r"^\s*\[[A-Z][^\]]{1,40}\]\s+\S"),
    # (1), (12), ...
    re.compile(r"^\s*\(\d{1,3}\)\s+\S"),
    # 1.  / 12.
    re.compile(r"^\s*\d{1,3}\.\s{2,}\S"),
    # 1  (bare number + multiple spaces)
    re.compile(r"^\s*\d{1,3}\s{3,}\S"),
    # Author-style references (no numeric markers):
    # Match lines that look like they start a reference with author names.
    # Key insight: Author names have initials or specific patterns that distinguish
    # them from journal/conference titles.
    # Valid patterns:
    # - "Ahmed, T." or "Smith, J." (last name + comma + initial with period)
    # - "E. Akyürek," (initial + name + comma)
    # - "T. B. Brown," (multiple initials + name + comma)
    # - "OpenAI." (organization + period at start)
    # - "St¨oger, D." (names with diacritics like ¨, ´, `, ^, ~, etc.)
    # - "Chen An, Shansan Gong," (full names in first-last format, common in ACL-style)
    # - "Satanjeev Banerjee and Alon Lavie" (names with "and")
    # - "J Ding, S Ma," (single letter + last name, common in some ACL refs)
    #
    # IMPORTANT: We must NOT match journal/conference titles like:
    # - "Signal Processing, 12(6)"
    # - "Machine Learning, pp."
    # The key difference is that author names have initials (single letters followed by periods)
    # or multiple person names in sequence, while titles don't.
    #
    # Character class includes: a-zA-Z, accented letters (à-ÿ, À-Ÿ),
    # hyphens, apostrophes, and common diacritics (¨´`^~)
    re.compile(
        r"^(?:"
        r"[A-Z][a-zA-Zà-ÿÀ-Ÿ¨´`^~\-']+,\s+[A-Z]\.|"  # Last name, Initial. (e.g., "Ahmed, T." or "St¨oger, D.")
        r"[A-Z]\.\s+[A-Z][a-zA-Zà-ÿÀ-Ÿ¨´`^~\-]+[,\.]|"  # Initial + name + comma/period (e.g., "E. Akyürek,")
        r"[A-Z]\.\s+[A-Z]\.\s+[A-Z][a-zA-Zà-ÿÀ-Ÿ¨´`^~\-]+[,\.]|"  # Two initials + name + comma/period
        r"[A-Z][A-Za-z@\-]+\.\s+[A-Z]|"  # Organization + period + word (e.g., "OpenAI. GPT")
        r"[A-Z][a-z]+\s+[A-Z][a-z]+(?:,|\s+and)\s+[A-Z][a-z]+"  # Full names with comma or "and": "Chen An, Shansan" or "John Smith and Jane"
        r"|[A-Z]\s+[A-Z][a-z]{2,},\s+[A-Z]\s+[A-Z]"  # Single letter names: "J Ding, S Ma," (I Last, I Last,)
        r")"
    ),
]

# Regex to strip the leading marker from the first line of each entry.
_LEADING_MARKER = re.compile(
    r"^\s*(?:"
    r"\[\d{1,3}\]"  # [1]
    r"|\[[A-Z][^\]]{1,40}\]"  # [AuthorYear]
    r"|\(\d{1,3}\)"  # (1)
    r"|\d{1,3}\."  # 1.
    r")\s*"
)

# Lines that signal we have left the references section (page body, ML tokens, etc.).
_JUNK_LINE = re.compile(
    r"<[A-Za-z/][A-Za-z]*>"  # special tokens like <EOS>, <pad>
    r"|^\s*Figure\s+\d+\s*[:.–\-]"  # figure captions
    r"|^\s*Table\s+\d+\s*[:]"  # table captions
    r"|^\s*\d{4,}\s*$",  # standalone 4+ digit numbers (page numbers)
    re.IGNORECASE,
)

# Embedded page-number + section-title junk: "...2013. 12 Attention Visualizations..."
# Matches a period, then a standalone 1-3 digit number, then a Title-cased word.
_PAGE_BREAK_JUNK = re.compile(r"(?<=\.)\s+\d{1,3}\s+(?=[A-Z][a-z])")

# Publisher/journal footer boilerplate that can bleed into the last reference entry.
# e.g. "...389. Visit SAGE journals online journals.sagepub.com/..."
_PUBLISHER_FOOTER = re.compile(
    r"(?<=\.)\s+(?:Visit\s|(?:journals?|www)\.[a-zA-Z0-9]+\.)",
    re.IGNORECASE,
)


def _extract_full_text(pdf_path: Union[str, Path]) -> str:
    """
    Return the full text of a PDF, page-by-page, joined by newlines.

    Excludes header and footer regions (top and bottom 1-inch margins)
    to avoid capturing page numbers, running headers, and other
    repeated elements that may interfere with reference parsing.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.

    Returns
    -------
    str
        Full text content of the PDF with headers and footers removed.
    """
    doc = fitz.open(str(pdf_path))
    pages: list[str] = []

    # 1 inch = 72 points in PDF coordinate system
    margin_points = 72

    for page in doc:
        # Get the page dimensions
        page_rect = page.rect
        page_height = page_rect.height
        page_width = page_rect.width

        # Define a clip rectangle that excludes top and bottom 1-inch margins
        # fitz.Rect(x0, y0, x1, y1) where (x0, y0) is top-left, (x1, y1) is bottom-right
        clip_rect = fitz.Rect(
            0,  # left edge
            margin_points,  # top edge (skip 1 inch from top)
            page_width,  # right edge
            page_height - margin_points,  # bottom edge (skip 1 inch from bottom)
        )

        # Extract text only from the clipped region
        text = page.get_text("text", clip=clip_rect)
        pages.append(text)

    doc.close()
    return "\n".join(pages)


def _find_references_section(text: str) -> str:
    """
    Locate the references/bibliography section and return everything from
    its header to the end of the document (or the next top-level section).
    """
    lines = text.splitlines()

    # Find the last occurrence of a references header (some papers have
    # "References" in the body too, e.g. "see References section").
    ref_start_idx: int | None = None
    for i, line in enumerate(lines):
        if _SECTION_HEADERS.match(line):
            ref_start_idx = i

    if ref_start_idx is None:
        # Fallback: search case-insensitively for a line that IS the header
        for i, line in enumerate(lines):
            stripped = line.strip().lower()
            if stripped in ("references", "bibliography", "works cited"):
                ref_start_idx = i

    if ref_start_idx is None:
        raise ValueError(
            "Could not locate a 'References' or 'Bibliography' section in the PDF."
        )

    # Take everything after the header.
    ref_lines = lines[ref_start_idx + 1 :]

    # Drop trailing appendix / acknowledgements that may follow.
    # Appendix markers: single letter "A", "B", etc. on their own line, or full words
    _STOP_HEADER = re.compile(
        r"^\s*(?:\d+[\.\s]+)?"
        r"(appendix|acknowledgements?|about\s+the\s+author|author\s+contributions?|"
        r"supplementary|notes?|index)"  # Allow additional text after these keywords
        r"|^[A-Z]$",  # Just "A", "B", "C" etc. alone on a line (appendix markers)
        re.IGNORECASE,
    )
    stop_idx: int | None = None
    for i, line in enumerate(ref_lines):
        stripped = line.strip()
        if _STOP_HEADER.match(stripped):
            stop_idx = i
            break

    if stop_idx is not None:
        ref_lines = ref_lines[:stop_idx]

    return "\n".join(ref_lines)


def _detect_pattern(text: str) -> re.Pattern | None:
    """Return the first entry-start pattern that matches inside *text*."""
    for pat in _REF_START_PATTERNS:
        if pat.search(text):
            return pat
    return None


def _split_references(ref_text: str) -> list[str]:
    """
    Split the references block into individual entries.

    Strategy:
      1. Detect which numbering pattern is used.
      2. For author-style references: use pattern matching as primary signal,
         with blank lines as secondary signal when pattern is ambiguous.
      3. Walk line by line; when a new entry marker is found, flush the
         current buffer.
      4. Normalise whitespace within each entry.
    """
    lines = ref_text.splitlines()
    pattern = _detect_pattern(ref_text)

    if pattern is None:
        # No recognisable marker – return each non-blank line as a reference.
        return [ln.strip() for ln in lines if ln.strip()]

    entries: list[str] = []
    current: list[str] = []

    # Check if this is an author-style pattern (no numeric markers)
    is_author_style = pattern == _REF_START_PATTERNS[-1]

    for line in lines:
        stripped = line.strip()

        # Skip junk lines (page numbers, special tokens, etc.)
        if _JUNK_LINE.search(line):
            continue

        # Handle blank lines
        if not stripped:
            if current and is_author_style:
                # For author-style, blank line MAY signal end of entry
                # but we'll wait to see if next non-blank line starts a new entry
                pass
            continue

        # Check if this line starts a new reference
        is_ref_start = pattern.match(line)

        if is_author_style:
            # For author-style: use multiple heuristics to determine if this is truly a new entry
            if is_ref_start and current:
                current_text = " ".join(current)

                # Heuristic 1: Current entry should have substantial content
                has_enough_content = len(current_text) > 80 or re.search(
                    r"\b(19|20)\d{2}\b", current_text
                )

                # Heuristic 2: Previous line should end with sentence-ending punctuation
                # or end with a year (common pattern: "... 2020." or "... arXiv:2020.12345, 2024.")
                prev_line_complete = current[-1].rstrip().endswith(".") or re.search(
                    r"(19|20)\d{2}[a-z]?\.\s*$", current[-1].rstrip()
                )

                # Heuristic 3: This line should start with "FirstName LastName," or "Initial. LastName," pattern
                starts_with_full_author = re.match(
                    r"^[A-Z][a-z]+\s+[A-Z][a-zA-Zà-ÿÀ-Ÿ\-\']+[,\.]", stripped
                )

                # Decision: start new entry if content is sufficient AND
                # (previous line ended with period OR this clearly starts new with author+comma/period)
                if has_enough_content and (
                    prev_line_complete or starts_with_full_author
                ):
                    entries.append(_clean_entry(current))
                    current = [stripped]
                else:
                    # Continuation line
                    current.append(stripped)
            elif is_ref_start and not current:
                # First entry
                current = [stripped]
            else:
                # Continuation line
                if stripped:
                    current.append(stripped)
        else:
            # For numbered styles, pattern match is sufficient
            if is_ref_start:
                if current:
                    entries.append(_clean_entry(current))
                # Strip the leading marker
                first = _LEADING_MARKER.sub("", stripped, count=1).strip()
                current = [first] if first else []
            else:
                if stripped:
                    current.append(stripped)

    if current:
        entries.append(_clean_entry(current))

    return entries


def _clean_entry(lines: list[str]) -> str:
    """Join continuation lines, collapse internal whitespace, and trim trailing junk."""
    joined = " ".join(lines)
    joined = re.sub(r"\s{2,}", " ", joined)
    # Trim embedded page-number + section-title junk:
    # e.g. "...August 2013. 12 Attention Visualizations..." → "...August 2013."
    match = _PAGE_BREAK_JUNK.search(joined)
    if match:
        joined = joined[: match.start() + 1]
    # Trim publisher/journal footer boilerplate:
    # e.g. "...389. Visit SAGE journals online..." → "...389."
    match = _PUBLISHER_FOOTER.search(joined)
    if match:
        joined = joined[: match.start() + 1]

    # Handle cases where the next reference number appears at the end of the current entry
    # This happens when entries span page breaks. Patterns to catch:
    # - "...2018. 23. Signal Processing..." → "...2018."
    # - "...2022. 36. Machine Learning..." → "...2022."
    # Look for a period, optional whitespace, then a 1-3 digit number, then a period and capital letter
    embedded_ref_num = re.search(r"(?<=\.)\s*\d{1,3}\.\s+(?=[A-Z])", joined)
    if embedded_ref_num:
        joined = joined[: embedded_ref_num.start() + 1]

    # Strip trailing bare page numbers left by PDF column layout, e.g. "...2016. 10"
    joined = re.sub(r"\s+\d{1,3}\s*$", "", joined)
    return joined.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_references_from_pdf(pdf_path: Union[str, Path]) -> list[str]:
    """
    Parse all references from an academic PDF.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.

    Returns
    -------
    list[str]
        Each element is one reference entry, with internal whitespace
        normalised but otherwise unchanged from the source text.

    Raises
    ------
    FileNotFoundError
        If *pdf_path* does not exist.
    ValueError
        If no references section can be located.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    full_text = _extract_full_text(pdf_path)
    ref_section = _find_references_section(full_text)
    references = _split_references(ref_section)
    return references


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse_references_from_pdf.py <path_to_pdf>")
        sys.exit(1)

    refs = parse_references_from_pdf(sys.argv[1])
    print(f"Found {len(refs)} reference(s):\n")
    for i, ref in enumerate(refs, 1):
        print(f"{i:>3}. {ref}\n")
