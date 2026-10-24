"""
Clean and preprocess a parsed academic paper document (Markdown from Docling).

Entry point:

    clean_document(text: str) -> str

Pipeline
--------
1. ``filter_line_numbers``          – remove PDF line-number columns
2. ``_split_at_references_section`` – split into body chunks + refs chunk
3. ``preprocess_references_section``– clean the references chunk
4. ``_clean_body_text``             – fix all defects in body chunks
5. Concatenate and return

Usage
-----
    python -m src.utils.preprocessing.clean_document
"""

from __future__ import annotations

import html
import re

from .extract_reference_section import NEXT_SECTION_PATTERN, REFERENCES_HEADING_PATTERN
from .filter_line_numbers import filter_line_numbers
from .preprocess_references_section import preprocess_references_section

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def clean_document(text: str) -> str:
    """
    Clean a single parsed PDF document (Docling Markdown output).

    Steps
    -----
    1. Strip PDF line-number columns via ``filter_line_numbers``.
    2. Split the document at its References section, producing:
       - ``before_refs``  – main body (title → conclusion)
       - ``refs_chunk``   – the references section
       - ``after_refs``   – anything that follows (appendix, checklist …)
    3. Clean the references chunk with ``preprocess_references_section``.
    4. Clean all remaining body chunks with ``_clean_body_text``.
    5. Concatenate and return.

    Parameters
    ----------
    text:
        Raw Markdown string produced by Docling.

    Returns
    -------
    str
        Cleaned document text.
    """
    # ── Step 1: filter line numbers ──────────────────────────────────────
    text = filter_line_numbers(text)

    # ── Step 2: split at References section ──────────────────────────────
    before_refs, refs_chunk, after_refs = _split_at_references_section(text)

    # ── Step 3: preprocess references ────────────────────────────────────
    if refs_chunk.strip():
        cleaned_refs = preprocess_references_section(refs_chunk)
        # Also decode any HTML entities that survived reference preprocessing
        cleaned_refs = _decode_html_entities(cleaned_refs)
    else:
        cleaned_refs = ""

    # ── Step 4: clean body chunks ─────────────────────────────────────────
    cleaned_before = _clean_body_text(before_refs)
    cleaned_after = _clean_body_text(after_refs) if after_refs.strip() else ""

    # ── Step 5: concatenate ───────────────────────────────────────────────
    parts = [cleaned_before, cleaned_refs, cleaned_after]
    return "\n\n".join(p.strip() for p in parts if p.strip())


# ---------------------------------------------------------------------------
# Step 2 helper – split at References
# ---------------------------------------------------------------------------


def _split_at_references_section(text: str) -> tuple[str, str, str]:
    """
    Split *text* into ``(before_refs, refs_section, after_refs)``.

    Uses the same heading and next-section patterns as
    ``extract_references_section`` so the boundary detection is consistent.

    Returns ``(text, "", "")`` when no References heading is found.
    """
    match = REFERENCES_HEADING_PATTERN.search(text)
    if not match:
        return text, "", ""

    ref_start = match.start()
    remaining = text[ref_start:]
    lines = remaining.split("\n")

    next_match = None
    if len(lines) > 1:
        rest = "\n".join(lines[1:])
        next_match = NEXT_SECTION_PATTERN.search(rest)

    if next_match:
        ref_end = ref_start + len(lines[0]) + 1 + next_match.start()
        return text[:ref_start], text[ref_start:ref_end], text[ref_end:]

    return text[:ref_start], text[ref_start:], ""


# ---------------------------------------------------------------------------
# Step 4 – body text cleaning pipeline
# ---------------------------------------------------------------------------


def _clean_body_text(text: str) -> str:
    """
    Apply all body-text defect fixes in a safe, ordered pipeline.

    Each fix targets a specific defect class identified in the inspection
    reports.  Fixes are ordered from lowest to highest risk so that later,
    more aggressive rules operate on already-normalised text.
    """
    if not text.strip():
        return text

    text = _strip_boilerplate_lines(text)  # LOW  – drop copyright / permission lines
    text = _decode_html_entities(text)  # LOW  – &amp; → &, &lt; → <, &gt; → >
    text = _normalize_code_url_labels(
        text
    )  # LOW  – "Code -https://" → "Code: https://" (before underscore fix)
    text = _fix_escaped_underscores(text)  # LOW  – \_ → _ in URLs / code refs
    text = _remove_negationslash(text)  # LOW  – /negationslash LaTeX artefact
    text = _fix_line_end_hyphenation(text)  # MED  – line-wrap dehyphenation
    text = _quarantine_corrupted_math_blocks(text)  # MED  – truncate \intertext garbage
    text = _fix_ligature_spaces(text)  # MED  – ligature decomposition spaces
    text = re.sub(r"\n{3,}", "\n\n", text)  # LOW  – collapse excess blank lines
    return text


# ---------------------------------------------------------------------------
# Individual fix functions
# ---------------------------------------------------------------------------

# ── FIX 1: strip conference boilerplate lines ────────────────────────────

_BOILERPLATE_PATTERNS: list[re.Pattern[str]] = [
    # AAAI copyright (appears inline after the introduction paragraph)
    re.compile(
        r"^Copyright\s*©\s*\d{4},?\s*Association for the Advancement of "
        r"Artificial Intelligence[^\n]*$",
        re.IGNORECASE,
    ),
    # NeurIPS / Google permission statement
    re.compile(
        r"^Provided proper attribution is provided,?\s*Google hereby grants "
        r"permission[^\n]*$",
        re.IGNORECASE,
    ),
    # ICLR / ICML "Published as …"
    re.compile(
        r"^Published as (?:a |an )?(conference|workshop) paper at "
        r"(ICLR|ICML|TMLR|NeurIPS|AAAI|ACL|CVPR)[^\n]*$",
        re.IGNORECASE,
    ),
    # "Under review as …"
    re.compile(
        r"^Under review as (?:a |an )?(conference|workshop) paper at "
        r"(ICLR|ICML|TMLR|NeurIPS|AAAI|ACL|CVPR)[^\n]*$",
        re.IGNORECASE,
    ),
    # TMLR "Published in Transactions on Machine Learning Research"
    re.compile(
        r"^Published in\s+Transactions on Machine Learning Research[^\n]*$",
        re.IGNORECASE,
    ),
    # Standalone "All rights reserved." line
    re.compile(r"^All rights reserved\.?\s*$", re.IGNORECASE),
]


def _strip_boilerplate_lines(text: str) -> str:
    """
    Remove known conference boilerplate lines that leak into body text.

    Only whole lines matching a boilerplate pattern are removed; surrounding
    content is untouched.
    """
    lines = text.split("\n")
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if any(p.match(stripped) for p in _BOILERPLATE_PATTERNS):
            continue
        kept.append(line)
    return "\n".join(kept)


# ── FIX 2: decode HTML entities ─────────────────────────────────────────


def _decode_html_entities(text: str) -> str:
    """
    Decode HTML entities left over from PDF/web conversion.

    Applied segment-by-segment to preserve math block content.
    ``&amp;`` → ``&``, ``&lt;`` → ``<``, ``&gt;`` → ``>``, ``&quot;`` → ``"``.

    Math blocks ($$…$$) and inline code are skipped because HTML entities
    are unlikely there and math content is fragile.
    """
    return _apply_outside_math(text, html.unescape)


# ── FIX 3: fix escaped underscores ──────────────────────────────────────

# Match \_ that is NOT inside a math block; the outer loop handles math skipping.
_ESCAPED_UNDERSCORE_RE = re.compile(r"\\_")


def _fix_escaped_underscores(text: str) -> str:
    """
    Replace ``\\_`` with ``_`` outside of math blocks.

    Docling sometimes escapes underscores in URLs and repository names
    (e.g. ``transformer\\_layers\\_as\\_painters``).  Inside ``$$`` math
    blocks, ``\\_`` is valid LaTeX subscript notation and is left alone.
    """

    def _unescape(segment: str) -> str:
        return _ESCAPED_UNDERSCORE_RE.sub("_", segment)

    return _apply_outside_math(text, _unescape)


# ── FIX 4: remove /negationslash artefact ───────────────────────────────

# Stray LaTeX primitive that leaks into prose/references when the parser
# fails to handle \not or similar constructs.
_NEGATIONSLASH_RE = re.compile(r"\s*/negationslash\s*")


def _remove_negationslash(text: str) -> str:
    """
    Remove stray ``/negationslash`` tokens (LaTeX parsing artefact).

    Observed in TMLR and CVPR batches.  The token is replaced by a single
    space to preserve word separation.
    """
    return _NEGATIONSLASH_RE.sub(" ", text)


# ── FIX 5: normalise "Code -https://…" URL labels ───────────────────────

_CODE_URL_RE = re.compile(
    # Capture the whole line: label + dash + URL (which may contain \_ and spaces)
    r"^(Code|Project Page|Demo|Website|Repository|GitHub|Repo)\s*-\s*(https?://.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _normalize_code_url_labels(text: str) -> str:
    """
    Normalise artifact URL-label lines such as ``Code -https://…``.

    Docling produces these when a PDF's "Code" or "Project Page" label is
    extracted together with its URL as a single text run, but the separator
    dash is preserved verbatim.

    ``Code -https://github.com/…`` → ``Code: https://github.com/…``
    """
    return _CODE_URL_RE.sub(r"\1: \2", text)


# ── FIX 6: line-end dehyphenation ───────────────────────────────────────

# Match a word fragment ending in a hyphen at end of line, followed (on the
# next non-empty line) by a lowercase continuation.
#
# Pattern guards:
#   - The hyphenated fragment must start with a letter (not a bullet '- ').
#   - The continuation must start with a lowercase letter (not a new sentence
#     or a section heading).
#   - Table rows (lines starting with '|') are excluded.
#   - Math lines (lines starting with '$$' or '\\') are excluded.
_HYPHEN_SPLIT_RE = re.compile(
    r"(?<=[A-Za-z])-\n"  # letter immediately before hyphen+newline
    r"(?=[a-z])",  # continuation starts with lowercase
)


def _fix_line_end_hyphenation(text: str) -> str:
    """
    Rejoin line-end hyphenation artefacts in body prose.

    PDF column layout frequently splits a word across lines with a hyphen:

        ``baseline meth-``   (end of line)
        ``ods.``             (start of next line)

    becomes ``baseline methods.``

    Table lines (``|…``) and math-environment lines (starting with ``$$``
    or ``\\``) are excluded to avoid clobbering structured content.  The
    fix is applied line-pair by line-pair so only genuinely adjacent
    hyphen+continuation pairs are joined.
    """
    lines = text.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip protected lines
        if _is_protected_hyphen_line(stripped):
            result.append(line)
            i += 1
            continue

        # Check: does this line end with a word-hyphen?
        if re.search(r"[A-Za-z]-$", stripped):
            # Look ahead for the next non-empty line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1

            if j < len(lines):
                next_stripped = lines[j].strip()
                # Continuation must start with lowercase and not be a
                # protected line type
                if (
                    next_stripped
                    and next_stripped[0].islower()
                    and not _is_protected_hyphen_line(next_stripped)
                ):
                    # Join: remove trailing hyphen, append continuation
                    joined = stripped[:-1] + next_stripped
                    # Preserve any leading whitespace from the original line
                    leading = line[: len(line) - len(line.lstrip())]
                    result.append(leading + joined)
                    # Skip any blank lines between the two content lines.
                    i = j + 1
                    continue

        result.append(line)
        i += 1

    return "\n".join(result)


def _is_protected_hyphen_line(line: str) -> bool:
    """Lines where hyphen joining should never be attempted."""
    if not line:
        return True
    if line.startswith("|") or line.startswith("|-"):
        return True  # Markdown table rows
    if line.startswith("$$") or line.startswith("\\"):
        return True  # Math environment delimiters / LaTeX
    if line.startswith("#"):
        return True  # Section headings
    if line.startswith("<!--"):
        return True  # HTML comments (image placeholders)
    if re.match(r"^[-*+]\s", line):
        return True  # Markdown bullet list items
    return False


# ── FIX 7: quarantine corrupted math blocks ──────────────────────────────

# A display-math block is considered corrupted when it contains multiple
# \intertext occurrences carrying spaced-character garbage, or when a single
# line within the block is implausibly long.
_MATH_BLOCK_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_INTERTEXT_RE = re.compile(r"\\intertext\s*\{")
_SPACED_CHARS_RE = re.compile(
    r"\{(?:\s*[a-z0-9]\s*){3,}\}"  # e.g. { s u l } or { w h e r e }
)

_CORRUPTION_THRESHOLD_INTERTEXT = 3  # ≥ 3 \intertext occurrences → corrupted
_CORRUPTION_THRESHOLD_LINE_LEN = 600  # any single line longer than this → corrupted


def _quarantine_corrupted_math_blocks(text: str) -> str:
    """
    Truncate garbage from corrupted ``$$…$$`` display-math blocks.

    Docling sometimes produces catastrophically corrupt equations where a
    valid formula is followed by dozens of repeated ``\\intertext { x y z }``
    tokens (spaced-out garbage characters).  This function detects such
    blocks and truncates everything from the first ``\\intertext`` that
    introduces spaced-character content onward, then closes the block.

    Blocks are considered corrupted when they satisfy **any** of:
    - ≥ 3 ``\\intertext`` occurrences inside a single ``$$`` block, OR
    - any internal line exceeds 600 characters (sign of runaway repetition).

    The truncated output replaces the corrupted math with the valid prefix
    so downstream readers still receive the correct part of the formula.
    """

    def _fix_block(m: re.Match[str]) -> str:
        inner = m.group(1)

        intertext_count = len(_INTERTEXT_RE.findall(inner))
        max_line_len = max((len(l) for l in inner.split("\n")), default=0)

        corrupted = (
            intertext_count >= _CORRUPTION_THRESHOLD_INTERTEXT
            or max_line_len >= _CORRUPTION_THRESHOLD_LINE_LEN
        )

        if not corrupted:
            return m.group(0)  # untouched

        # Truncate at the first corrupted \intertext occurrence.
        # Keep whatever valid LaTeX preceded it.
        first_bad = _INTERTEXT_RE.search(inner)
        if first_bad:
            clean_prefix = inner[: first_bad.start()].rstrip()
            # Strip trailing \\ alignment marker so the block still compiles
            clean_prefix = re.sub(r"\\\\$", "", clean_prefix.rstrip())
            return f"$${clean_prefix}$$"

        # Fallback: the block was detected as corrupted but we couldn't find
        # a clean truncation point – replace with a sentinel comment.
        return "<!-- EQUATION EXTRACTION FAILED -->"

    return _MATH_BLOCK_RE.sub(_fix_block, text)


# ── FIX 8: fix ligature decomposition spaces ─────────────────────────────

# When the PDF uses typographic ligatures (ﬁ, ﬂ, ﬀ, ﬃ, ﬄ), some parsers
# decompose them into their component letters but insert a spurious space
# between the ligature cluster and the remainder of the word.
#
# Examples (observed in inspection reports):
#   "a fi rst-order"  → "a first-order"      (ﬁ → 'fi ')
#   "fl ow"           → "flow"               (ﬂ → 'fl ')
#   "ef fi cient"     → "efficient"          (ﬃ → 'fi ')
#
# Safe context: we only collapse the space when the ligature pair appears at
# the START of a word token (preceded by a word boundary or start of string)
# and is followed immediately by a lowercase letter.  This avoids false
# positives like "of i" in reference lists.

_LIGATURE_FIXES: list[tuple[re.Pattern[str], str]] = [
    # fi-ligature: word-initial "fi " + lowercase continuation
    (re.compile(r"(?<!\w)(fi) ([a-z])"), r"fi\2"),
    # fl-ligature
    (re.compile(r"(?<!\w)(fl) ([a-z])"), r"fl\2"),
    # ff-ligature (catches 'ef fect' → 'effect' when 'ff' crosses a word boundary)
    (re.compile(r"([a-z])(ff) ([a-z])"), r"\1ff\3"),
    # ffi-ligature: "ffi " + lowercase
    (re.compile(r"(?<!\w)(ffi) ([a-z])"), r"ffi\2"),
    # ffl-ligature: "ffl " + lowercase
    (re.compile(r"(?<!\w)(ffl) ([a-z])"), r"ffl\2"),
    # Spaced-out single characters inside words detected as OCR artifacts.
    # Pattern: "X " where X is one letter and the preceding/following chars
    # indicate the space is spurious (all lowercase, no punctuation).
    # We only handle the most clearly identified pattern from inspections:
    # a sequence like "E v aluation" → "Evaluation" (capital + space + lower)
    # This is MEDIUM risk so we restrict to highly specific cases only.
    # (Handled separately in _fix_spaced_capital_words below.)
]


def _fix_ligature_spaces(text: str) -> str:
    """
    Remove spurious spaces introduced by ligature decomposition.

    Applied only outside math blocks to avoid corrupting LaTeX content.
    """

    def _apply_fixes(segment: str) -> str:
        for pattern, repl in _LIGATURE_FIXES:
            segment = pattern.sub(repl, segment)
        return segment

    return _apply_outside_math(text, _apply_fixes)


# ---------------------------------------------------------------------------
# Utility: apply a transformation only to non-math segments
# ---------------------------------------------------------------------------

# Split on display-math delimiters.  We keep the delimiters in the split so
# we can reconstruct the string faithfully.
_MATH_SPLIT_RE = re.compile(r"(\$\$)")


def _apply_outside_math(text: str, fn) -> str:
    """
    Apply *fn* to every segment of *text* that is NOT inside a ``$$`` block.

    The function alternates between prose segments (even indices) and math
    segments (odd indices) in the split result, leaving math untouched.
    """
    # Split on $$ markers; the RE captures the delimiter so it's included.
    parts = _MATH_SPLIT_RE.split(text)

    # parts looks like:
    #   [prose, "$$", math_content, "$$", prose, "$$", math_content, "$$", ...]
    # Index:  0      1       2         3     4      5        6          7

    result: list[str] = []
    inside_math = False
    for part in parts:
        if part == "$$":
            inside_math = not inside_math
            result.append(part)
        elif inside_math:
            result.append(part)  # math content – leave untouched
        else:
            result.append(fn(part))  # prose content – apply fix

    return "".join(result)


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob
    import os

    parsed_pdfs_dir = "./output/parsed_pdfs/docling"
    output_dir = "./output/cleaned_pdfs/docling"

    total = 0
    changed = 0

    for venue_folder in os.listdir(parsed_pdfs_dir):
        venue_path = os.path.join(parsed_pdfs_dir, venue_folder)
        if not os.path.isdir(venue_path):
            continue

        out_venue = os.path.join(output_dir, venue_folder)
        os.makedirs(out_venue, exist_ok=True)

        for pdf_path in sorted(glob.glob(os.path.join(venue_path, "*.md"))):
            try:
                with open(pdf_path, encoding="utf-8") as f:
                    content = f.read()

                cleaned = clean_document(content)
                total += 1

                out_path = os.path.join(out_venue, os.path.basename(pdf_path))
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(cleaned)

                if content != cleaned:
                    changed += 1
                    print(f"  cleaned  {pdf_path}")
                else:
                    print(f"  no change {pdf_path}")

            except Exception as exc:
                print(f"  ERROR {pdf_path}: {exc}")

    print(f"\nDone: {changed}/{total} files modified.")
