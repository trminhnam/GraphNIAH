"""Command to run
python -m src.reference_parser.cvpr_reference_parser
"""

from __future__ import annotations

import re

from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener

from .antlr4_grammar.cvpr.cvprLexer import cvprLexer
from .antlr4_grammar.cvpr.cvprParser import cvprParser
from .reference_parser import Reference, ReferenceParser

_VENUE_STARTERS = re.compile(
    r"(?:"
    r"In\s|"
    r"arXiv[:\s]|"
    r"ArXiv\s|"
    r"Preprint[,\s]|"
    r"Proceedings\s|"
    r"Ph\.D\.\s|"
    r"PhD\s|"
    r"Master'?s\s|"
    r"Technical\s[Rr]eport|"
    r"(?:[A-Z][a-zA-Z&\s]*(?:Journal|Transactions|Letters|Review|Conference|Workshop|Proceedings|Advances|Annals|Bulletin|Reports|Research|Quarterly|Studies))|"
    r"(?:[A-Z][a-zA-Z]*(?:\s+[A-Za-z]+)*\s+[Pp]ress)|"
    r"Springer|Elsevier|Wiley|PMLR|IEEE|ACM|AAAI|SIAM|"
    r"Cambridge|Oxford|MIT|Princeton|"
    r"Curran\s|"
    r"CoRR[,\s]|"
    r"NeurIPS|ICML|ICLR|ECCV|ICCV|CVPR|EMNLP|NAACL|SIGIR|"
    r"Association\sfor"
    r")"
)

_JOURNAL_LIKE = re.compile(r"^(?:" r"[A-Z][a-zA-Z]*(?:\s+[A-Za-z&]+)*" r")\s*,")

_TRAILING_CITATIONS_RE = re.compile(r"\.\s+(\d{1,2}(?:\s*,\s*\d{1,2})*)\s*$")

_REF_NUMBER_RE = re.compile(r"^\[\d+\]\s*")

# Matches 2-letter name abbreviations like "Md." inside a comma-separated
# author list, e.g. ", Md. Abu" → ", Md Abu".  This prevents the ANTLR
# lexer from tokenising the period as a DOT (author-body separator).
_NAME_ABBREV_RE = re.compile(r"(?<=, )([A-Z][a-z])\.(?= [A-Z])")

# Common PDF-extraction / OCR artifacts from unresolved LaTeX accent commands:
#   ` (backtick / grave), ¨ (standalone diaeresis U+00A8),
#   ˜ (small tilde U+02DC), ˆ (modifier circumflex U+02C6).
# Each may be followed by whitespace before the base letter.
_ENCODING_ARTIFACT_RE = re.compile(r"[`\u00A8\u02DC\u02C6]\s*")


class _SilentErrorListener(ErrorListener):
    """Collects errors without printing to stderr."""

    def __init__(self) -> None:
        super().__init__()
        self.errors: list[str] = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.errors.append(f"line {line}:{column} {msg}")


def _extract_cvpr_author_name(ctx: cvprParser.AuthorNameContext) -> str:
    """Extract a human-readable author name from first-name-first format."""
    tokens = [nw.getText() for nw in ctx.nameWord()]
    if not tokens:
        return ctx.getText()

    parts = [tokens[0]]
    for prev, cur in zip(tokens, tokens[1:]):
        space = True
        if cur in ("'", "\u2019", "-", "\u2013", "\u2014"):
            space = False
        elif prev in ("'", "\u2018", "\u2019", "-", "\u2013", "\u2014"):
            space = False
        elif cur == "*" or prev == "*":
            space = False

        if space:
            parts.append(" ")
        parts.append(cur)

    name = "".join(parts)
    name = name.replace("*", " ")
    return re.sub(r"\s+", " ", name).strip()


def _reconstruct_body(ctx: cvprParser.BodyContext) -> str:
    """Reconstruct the body text from the parse tree,
    preserving the original spacing intent as faithfully as possible."""
    if ctx is None:
        return ""

    tokens: list[str] = []
    for i in range(ctx.getChildCount()):
        child = ctx.getChild(i)
        tok = child.getText()
        if tok:
            tokens.append(tok)

    if not tokens:
        return ""

    _ORDINAL_SUFFIXES = {"th", "st", "nd", "rd"}

    parts = [tokens[0]]
    for prev_tok, cur_tok in zip(tokens, tokens[1:]):
        space = True

        if cur_tok in (".", ",", ";", ":", ")", "?", "!", "'", "\u2019"):
            space = False
        elif prev_tok in ("(", "'", "\u2018", "\u2019"):
            space = False
        elif prev_tok == "-" or cur_tok == "-":
            space = False
        elif prev_tok in ("\u2013", "\u2014") or cur_tok in ("\u2013", "\u2014"):
            space = False
        elif prev_tok.endswith(".") and len(prev_tok) == 2 and cur_tok == "-":
            space = False
        elif prev_tok == ":" and cur_tok[0:1].isdigit():
            space = False
        elif prev_tok[-1:].isdigit() and cur_tok == ".":
            space = False
        elif prev_tok == "." and cur_tok[0:1].isdigit():
            space = False
        elif cur_tok == "(" and prev_tok[-1:].isdigit():
            space = False
        elif prev_tok[-1:].isdigit() and cur_tok.lower() in _ORDINAL_SUFFIXES:
            space = False
        elif (
            prev_tok == "."
            and len(cur_tok) == 2
            and cur_tok[0].isupper()
            and cur_tok[1] == "."
        ):
            space = False
        elif prev_tok == "/" or cur_tok == "/":
            space = False
        elif prev_tok == "*" or cur_tok == "*":
            space = False

        if space:
            parts.append(" ")
        parts.append(cur_tok)

    return "".join(parts)


def _strip_trailing_citations(text: str) -> str:
    """Strip trailing in-paper citation page numbers (e.g., '. 2, 3, 7')."""
    return _TRAILING_CITATIONS_RE.sub(".", text)


def _strip_reference_number(text: str) -> str:
    """Strip leading bracketed reference number like [1], [23], etc."""
    return _REF_NUMBER_RE.sub("", text)


def _normalize_name_abbreviations(text: str) -> str:
    """Remove the period from 2-letter name abbreviations (e.g. ``Md.``)
    that appear inside a comma-separated author list so that the ANTLR
    lexer does not mistake them for the author-body DOT separator."""
    return _NAME_ABBREV_RE.sub(r"\1", text)


def _clean_encoding_artifacts(text: str) -> str:
    """Strip common PDF-extraction artifacts from unresolved LaTeX accents
    (e.g. ``D'Inc` a`` → ``D'Inca``, ``M¨ uller`` → ``Muller``)."""
    return _ENCODING_ARTIFACT_RE.sub("", text)


def _extract_year(body: str) -> tuple[str, str]:
    """Extract the publication year from body text.

    CVPR abbrvnat places the year at the end of the reference, typically
    preceded by ', '.  Returns (year, cleaned_body_without_year).
    """
    cleaned = body.rstrip()
    if cleaned.endswith("."):
        cleaned = cleaned[:-1].rstrip()

    # Strategy 1: ", YYYY[a-z]?" at end — most common
    m = re.search(r",\s+(\d{4}[a-z]?)\s*$", cleaned)
    if m and 1900 <= int(m.group(1)[:4]) <= 2099:
        return m.group(1), cleaned[: m.start()].rstrip()

    # Strategy 2: ". YYYY[a-z]?" at end — misc/unpublished entries
    m = re.search(r"\.\s+(\d{4}[a-z]?)\s*$", cleaned)
    if m and 1900 <= int(m.group(1)[:4]) <= 2099:
        return m.group(1), cleaned[: m.start()].rstrip()

    # Strategy 3: ", YYYY" anywhere — year followed by post-year content (publisher)
    matches = list(re.finditer(r",\s+(\d{4}[a-z]?)\b", cleaned))
    for m in reversed(matches):
        yr = int(m.group(1)[:4])
        if 1900 <= yr <= 2099:
            before = cleaned[: m.start()].rstrip()
            after = cleaned[m.end() :].strip()
            if after.startswith("."):
                after = after[1:].strip()
            result = before
            if after:
                result += ". " + after
            return m.group(1), result

    # Strategy 4: any standalone YYYY near the end
    matches = list(re.finditer(r"\b(\d{4}[a-z]?)\b", cleaned))
    for m in reversed(matches):
        yr = int(m.group(1)[:4])
        if 1900 <= yr <= 2099:
            before = cleaned[: m.start()].rstrip(", ")
            after = cleaned[m.end() :].strip(". ")
            result = before
            if after:
                result += ". " + after
            return m.group(1), result

    return "", cleaned


def _split_title_venue(body: str) -> tuple[str, str]:
    """Split the body string into (title, venue).

    Heuristics (applied in order):
    1. Look for a venue-starter pattern (``In ...``, journal name, publisher, …)
       after a sentence-ending punctuation mark (``.``, ``?``, ``!``).
    2. If no venue-starter is found, the whole body is the title.
    """
    if not body:
        return ("", "")

    for m in re.finditer(r"[.?!]\s+", body):
        after = body[m.end() :]
        if _VENUE_STARTERS.match(after) or _JOURNAL_LIKE.match(after):
            end_char = body[m.start()]
            title = body[: m.start()].strip()
            if end_char in ("?", "!"):
                title += end_char
            venue = after.strip()
            if venue.endswith("."):
                venue = venue[:-1].rstrip()
            return (title, venue)

    title = body.rstrip()
    if title.endswith("."):
        title = title[:-1].rstrip()
    return (title, "")


class CvprReferenceParser(ReferenceParser):
    """Parser for references formatted in CVPR abbrvnat bibliography style.

    Uses the ANTLR4 grammar defined in ``cvpr.g4`` to structurally parse
    authors, then extracts the year from the end of the body text and
    applies heuristics to separate the title from venue information.

    Unlike AAAI/ACL where the year immediately follows the author list,
    CVPR abbrvnat places the year at the end of the reference body.
    Handles article, inproceedings, book, incollection, misc, thesis,
    techreport, and other entry types.

    Args:
        raise_error: When ``True``, raises :class:`ValueError` on any ANTLR4
            parse failure instead of falling back to regex extraction.
    """

    # Detects numbered reference markers like [1], [23], etc.
    _NUMBERED_MARKER_RE = re.compile(r"^\s*\[\s*\d+\s*\]", re.MULTILINE)

    def __init__(self, raise_error: bool = False) -> None:
        self.raise_error = raise_error

    # ── preprocessing: rejoin references broken by page/column breaks ──

    def parse(self, text: str) -> list[Reference]:
        """Override to rejoin references that were split across blank lines
        by PDF page or column breaks before the base-class chunking logic
        runs."""
        text = self._rejoin_numbered_references(text)
        return super().parse(text)

    @classmethod
    def _rejoin_numbered_references(cls, text: str) -> str:
        """Merge blank-line-separated paragraphs that are continuations of a
        ``[N]``-numbered reference (i.e. paragraphs that do *not* start with
        a reference marker or a markdown heading)."""
        if not cls._NUMBERED_MARKER_RE.search(text):
            return text
        paragraphs = re.split(r"\n\s*\n", text)
        merged: list[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if merged and not re.match(r"\s*(?:\[\s*\d+\s*\]|#+\s)", para):
                # Continuation of the preceding reference
                merged[-1] = merged[-1] + " " + para
            else:
                merged.append(para)
        return "\n\n".join(merged)

    def parse_one(self, text: str) -> Reference | None:
        if not text or not text.strip():
            return None

        text = text.strip()
        text = _strip_reference_number(text)
        text = _strip_trailing_citations(text)
        text = _clean_encoding_artifacts(text)
        text = _normalize_name_abbreviations(text)
        text = text.strip()

        if not text:
            return None

        input_stream = InputStream(text)

        lexer = cvprLexer(input_stream)
        lexer.removeErrorListeners()
        lex_errs = _SilentErrorListener()
        lexer.addErrorListener(lex_errs)

        token_stream = CommonTokenStream(lexer)

        parser = cvprParser(token_stream)
        parser.removeErrorListeners()
        parse_errs = _SilentErrorListener()
        parser.addErrorListener(parse_errs)

        tree = parser.reference()

        if parse_errs.errors:
            # Always attempt the regex fallback; only raise when it too
            # fails to extract any useful information (title or year).
            fallback = self._fallback_parse(text)
            if self.raise_error and not fallback.title and not fallback.year:
                raise ValueError(
                    f"ANTLR4 grammar (CVPR) failed to parse reference. "
                    f"Errors: {parse_errs.errors!r}. Text: {text!r}"
                )
            return fallback

        return self._extract_reference(tree, text)

    # ── tree → Reference ────────────────────────────────────────────

    def _extract_reference(
        self, tree: cvprParser.ReferenceContext, raw: str
    ) -> Reference:
        # Authors
        authors_ctx = tree.authors()
        author_names: list[str] = []
        if authors_ctx is not None:
            for author_ctx in authors_ctx.authorName():
                author_names.append(_extract_cvpr_author_name(author_ctx))
            if authors_ctx.ET_AL() is not None:
                author_names.append("et al.")

        # Body → extract year, then split title + venue
        body_text = _reconstruct_body(tree.body())
        year, body_no_year = _extract_year(body_text)
        title, venue = _split_title_venue(body_no_year)

        return Reference(
            authors=author_names,
            year=year,
            title=title,
            venue=venue,
            raw=raw,
        )

    # ── regex fallback for unusual formatting ────────────────────────

    @staticmethod
    def _fallback_parse(text: str) -> Reference:
        ref = Reference(raw=text)

        # Try to find a year near the end: ", YYYY." or ". YYYY."
        m = re.search(r"[,.\s]\s*(\d{4}[a-z]?)\s*\.\s*$", text)
        if not m:
            m = re.search(r",\s+(\d{4}[a-z]?)\b", text)
        if not m:
            return ref

        yr = int(m.group(1)[:4])
        if not (1900 <= yr <= 2099):
            return ref

        ref.year = m.group(1)

        # Find author-body boundary: first ". " followed by an uppercase letter
        # that isn't part of an initial (i.e., not "X. " where X is a single char)
        # or a short name abbreviation (e.g. "Md.") still inside the author list.
        author_body_split = None
        for ab_m in re.finditer(r"\.\s+", text):
            pos = ab_m.end()
            if pos >= m.start():
                break
            preceding = text[: ab_m.start()]
            last_word = preceding.split()[-1] if preceding.split() else ""
            # Skip if this period is part of an initial (single uppercase letter)
            if len(last_word) == 1 and last_word.isupper():
                continue
            # Skip short abbreviations (e.g. "Md.", "Jr.") when more
            # author-list patterns (", and" / "et al") follow before the year.
            if len(last_word) <= 3 and last_word[:1].isupper():
                remaining = text[pos : m.start()]
                if re.search(r",\s+(?:and|et\s+al)", remaining, re.IGNORECASE):
                    continue
            after_char = text[pos : pos + 1]
            if after_char and after_char.isupper():
                author_body_split = ab_m
                break

        if author_body_split is None:
            return ref

        author_part = text[: author_body_split.start()].strip()
        body_part = text[author_body_split.end() : m.start()].strip()

        # Parse authors: split on ", and " or " and ", with comma separators
        et_al_match = re.search(r",?\s*et\s+al\.?\s*$", author_part, re.IGNORECASE)
        if et_al_match:
            author_part = author_part[: et_al_match.start()]
            raw_authors = [a.strip() for a in author_part.split(",") if a.strip()]
            raw_authors.append("et al.")
        elif ", and " in author_part:
            last_and = author_part.rfind(", and ")
            before = author_part[:last_and]
            after = author_part[last_and + 6 :].strip()
            raw_authors = [a.strip() for a in before.split(",") if a.strip()]
            if after:
                raw_authors.append(after)
        elif " and " in author_part:
            parts = author_part.rsplit(" and ", 1)
            raw_authors = [p.strip() for p in parts if p.strip()]
        else:
            raw_authors = [author_part.strip()] if author_part.strip() else []

        ref.authors = raw_authors

        title, venue = _split_title_venue(body_part)
        ref.title = title
        ref.venue = venue
        return ref


if __name__ == "__main__":
    import glob
    import os

    target_conference_name = "cvpr"
    reference_document_dir = f"output/parsed_references/docling"
    parser = CvprReferenceParser(raise_error=True)
    for conference_name in os.listdir(reference_document_dir):
        if conference_name != target_conference_name:
            if not os.path.isdir(os.path.join(reference_document_dir, conference_name)):
                continue

            other_conference_reference_document_dir = os.path.join(
                reference_document_dir, conference_name
            )
            # In this part, we use the current parser to parse the references in the other conference
            # It must return the errors because we design the parser to be strict and only parse the references in the other conference
            print(f"=== Parsing references of conference `{conference_name}` ===")
            for reference_document_path in sorted(
                glob.glob(os.path.join(other_conference_reference_document_dir, "*.md"))
            ):
                print(f"=== {reference_document_path} ===")
                with open(reference_document_path, "r") as f:
                    reference_document = f.read()
                try:
                    ref = parser.parse(reference_document)
                    print(
                        f"WARNING: The reference style of conference {conference_name} is supported by the parser"
                    )
                except ValueError as e:
                    print(
                        f"The reference style of conference {conference_name} is not supported by the parser"
                    )
                print("--------------------------------")

            continue
        reference_document_dir = os.path.join(reference_document_dir, conference_name)
        for reference_document_path in sorted(
            glob.glob(os.path.join(reference_document_dir, "*.md"))
        ):
            print(f"=== {reference_document_path} ===")
            with open(reference_document_path, "r") as f:
                reference_document = f.read()
            try:
                ref = parser.parse(reference_document)
                print(len(ref))
            except ValueError as e:
                print(f"Error parsing {reference_document_path}: {e}")
            print("--------------------------------")
