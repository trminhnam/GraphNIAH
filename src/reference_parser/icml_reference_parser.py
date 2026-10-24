"""Command to run
python -m src.reference_parser.icml_reference_parser
"""

from __future__ import annotations

import re

from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener

from .antlr4_grammar.icml.icmlLexer import icmlLexer
from .antlr4_grammar.icml.icmlParser import icmlParser
from .reference_parser import Reference, ReferenceParser

_VENUE_STARTERS = re.compile(
    r"(?:"
    r"In\s|"
    r"arXiv\b|"
    r"ArXiv\b|"
    r"Preprint[,\s]|"
    r"Proceedings\s|"
    r"Ph\.?\s*D\.?\s|"
    r"PhD\s|"
    r"Master'?s\s|"
    r"Technical\s[Rr]eport|"
    r"(?:[A-Z][a-zA-Z&:\s]*(?:Journal|Transactions|Letters|Review|Conference|Workshop|Proceedings|Advances|Annals|Bulletin|Reports|Research|Quarterly|Studies|Systems|Society|Mathematics|Sciences|Computing|Applications|Intelligence|Surveys|Communications|Methods))|"
    r"(?:[A-Z][a-zA-Z]*(?:\s+[A-Za-z]+)*\s+[Pp]ress)|"
    r"Springer|Elsevier|Wiley|PMLR|IEEE|ACM|AAAI|SIAM|"
    r"Cambridge|Oxford|MIT|Princeton|"
    r"Curran\s|"
    r"CoRR[,\s]|"
    r"NeurIPS|ICML|ICLR|ECCV|ICCV|CVPR|EMNLP|NAACL|SIGIR|"
    r"Association\sfor|"
    r"OpenReview|"
    r"URL\s|"
    r"pp\.[\ \d]|"
    r"ISSN\s"
    r")"
)

_JOURNAL_LIKE = re.compile(r"^(?:[A-Z][a-zA-Z]*(?:(?:\s+|:\s*)[A-Za-z&]+)*)\s*,")

_EDITOR_TAG_RE = re.compile(r"^\s*\(eds?\.?\)\s*\.?\s*")

_TRAILING_CITATIONS_RE = re.compile(r"\.\s+(\d{1,2}(?:\s*,\s*\d{1,2})*)\s*$")

_REF_NUMBER_RE = re.compile(r"^\[\d+\]\s*")

_DAGGER_RE = re.compile(r"[\u2020\u2021]+")
_SPACED_INITIAL_RE = re.compile(r"\b([A-Z]) \.")
_PAGE_METADATA_RE = re.compile(r"^Pages?\s*:", re.IGNORECASE)

_MONTHS_PAT = (
    r"(?:January|February|March|April|May|June"
    r"|July|August|September|October|November|December)"
)


class _SilentErrorListener(ErrorListener):
    """Collects errors without printing to stderr."""

    def __init__(self) -> None:
        super().__init__()
        self.errors: list[str] = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.errors.append(f"line {line}:{column} {msg}")


def _extract_icml_author_name(ctx) -> str:
    """Extract a human-readable author name from an author context.

    Accepts both ``AuthorFormalContext`` and ``AuthorNameContext`` — both
    expose ``nameParts()`` and ``initials()`` with identical semantics.

    ICML format: ``LastName, F.`` or ``von LastName, F. M.``
    """
    name_parts_ctx = ctx.nameParts()
    if name_parts_ctx is None:
        return ctx.getText()

    last_name_tokens = [nw.getText() for nw in name_parts_ctx.nameWord()]

    last_name = ""
    for i, tok in enumerate(last_name_tokens):
        if i > 0 and tok not in ("'", "\u2019", "-", "\u2013", "\u2014"):
            prev = last_name_tokens[i - 1]
            if prev not in ("'", "\u2018", "\u2019", "-", "\u2013", "\u2014"):
                last_name += " "
        last_name += tok

    initials_ctx = ctx.initials()
    if initials_ctx is None:
        return last_name

    init_tokens = []
    for child_idx in range(initials_ctx.getChildCount()):
        init_tokens.append(initials_ctx.getChild(child_idx).getText())

    initials_str = ""
    for i, tok in enumerate(init_tokens):
        if i > 0 and tok != "-" and (i < 1 or init_tokens[i - 1] != "-"):
            initials_str += " "
        initials_str += tok

    return f"{last_name}, {initials_str}"


def _reconstruct_body(ctx: icmlParser.BodyContext) -> str:
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


def _extract_year(body: str) -> tuple[str, str]:
    """Extract the publication year from body text.

    ICML icml2025 places the year near the end of the reference,
    typically preceded by ', '.  Returns (year, cleaned_body_without_year).
    """
    cleaned = body.rstrip()
    if cleaned.endswith("."):
        cleaned = cleaned[:-1].rstrip()

    # ", YYYY[a-z]?" at end
    m = re.search(r",\s+(\d{4}[a-z]?)\s*$", cleaned)
    if m and 1900 <= int(m.group(1)[:4]) <= 2099:
        return m.group(1), cleaned[: m.start()].rstrip()

    # ". YYYY[a-z]?" at end (misc/unpublished entries)
    m = re.search(r"\.\s+(\d{4}[a-z]?)\s*$", cleaned)
    if m and 1900 <= int(m.group(1)[:4]) <= 2099:
        return m.group(1), cleaned[: m.start()].rstrip()

    # "[,.] Month YYYY" at end (e.g. ", July 1989" or ". March 2021")
    m = re.search(
        rf"[,.]\s+{_MONTHS_PAT}\s+(\d{{4}}[a-z]?)\s*$", cleaned, re.IGNORECASE
    )
    if m and 1900 <= int(m.group(1)[:4]) <= 2099:
        return m.group(1), cleaned[: m.start()].rstrip()

    # ", YYYY" anywhere — year followed by post-year content
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

    # Any standalone YYYY near the end (skip years inside URLs)
    matches = list(re.finditer(r"\b(\d{4}[a-z]?)\b", cleaned))
    for m in reversed(matches):
        yr = int(m.group(1)[:4])
        if 1900 <= yr <= 2099:
            pos = m.start()
            if pos > 0 and cleaned[pos - 1] == "/":
                continue
            if m.end() < len(cleaned) and cleaned[m.end()] == "/":
                continue
            before = cleaned[: m.start()].rstrip(", ")
            month_m = re.search(rf"[,.]?\s*{_MONTHS_PAT}\s*$", before, re.IGNORECASE)
            if month_m:
                before = before[: month_m.start()].rstrip(", ")
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


class IcmlReferenceParser(ReferenceParser):
    """Parser for references formatted in ICML icml2025 bibliography style.

    Uses the ANTLR4 grammar defined in ``icml.g4`` to structurally parse
    authors, then extracts the year from the end of the body text and
    applies heuristics to separate the title from venue information.

    ICML icml2025 uses last-name-first author ordering with abbreviated
    initials (``LastName, F.``), comma separators between authors, "and"
    before the last author.  The year appears within the body near the end,
    not immediately after the author list.

    Handles article, inproceedings, book, incollection, misc, thesis,
    techreport, proceedings, and other entry types.
    """

    def __init__(self, raise_error: bool = False) -> None:
        """
        Args:
            raise_error: When ``True``, raises :class:`ValueError` on any ANTLR4
                parse failure instead of falling back to regex extraction.
        """
        self.raise_error = raise_error

    def parse(self, text: str) -> list[Reference]:
        refs = super().parse(text)
        merged: list[Reference] = []
        for ref in refs:
            if not ref.authors and not ref.title and merged:
                prev = merged[-1]
                combined_raw = prev.raw + " " + ref.raw
                re_parsed = self.parse_one(combined_raw)
                if re_parsed is not None:
                    merged[-1] = re_parsed
                else:
                    prev.raw = combined_raw
            else:
                merged.append(ref)
        return merged

    def parse_one(self, text: str) -> Reference | None:
        if not text or not text.strip():
            return None

        text = text.strip()
        text = _strip_reference_number(text)
        text = _strip_trailing_citations(text)
        text = text.strip()

        if not text:
            return None

        if _PAGE_METADATA_RE.match(text):
            return None

        # Skip non-reference content that leaks from PDF extraction:
        # LaTeX math blocks, figure/table captions, very short fragments.
        if text.startswith("$$") or text.startswith("\\("):
            return None
        if len(text) < 10:
            return None
        if re.match(r"^\([a-z]\)\s", text):
            return None

        raw_text = text

        text = _DAGGER_RE.sub("", text)
        text = re.sub(r",\s*,", ",", text)
        text = _SPACED_INITIAL_RE.sub(r"\1.", text)
        text = text.strip()

        if not text:
            return None

        input_stream = InputStream(text)

        lexer = icmlLexer(input_stream)
        lexer.removeErrorListeners()
        lex_errs = _SilentErrorListener()
        lexer.addErrorListener(lex_errs)

        token_stream = CommonTokenStream(lexer)

        parser = icmlParser(token_stream)
        parser.removeErrorListeners()
        parse_errs = _SilentErrorListener()
        parser.addErrorListener(parse_errs)

        tree = parser.reference()

        if parse_errs.errors:
            if self.raise_error:
                raise ValueError(
                    f"ANTLR4 grammar (ICML) failed to parse reference. "
                    f"Errors: {parse_errs.errors!r}. Text: {text!r}"
                )
            return self._fallback_parse(text, raw_text)

        return self._extract_reference(tree, raw_text, text)

    # ── tree → Reference ────────────────────────────────────────────

    def _extract_reference(
        self, tree: icmlParser.ReferenceContext, raw: str, cleaned: str = ""
    ) -> Reference:
        # Authors — collect from both authorFormal and authorName contexts,
        # preserving their original order via getChildren().
        authors_ctx = tree.authors()
        author_names: list[str] = []
        if authors_ctx is not None:
            for child in authors_ctx.getChildren():
                if isinstance(
                    child,
                    (icmlParser.AuthorFormalContext, icmlParser.AuthorNameContext),
                ):
                    author_names.append(_extract_icml_author_name(child))
            if authors_ctx.ET_AL() is not None:
                author_names.append("et al.")

        # Body → strip editor tags, extract year, then split title + venue
        body_text = _reconstruct_body(tree.body())

        body_text = _EDITOR_TAG_RE.sub("", body_text)

        # If body starts with comma or "and", ANTLR captured too few
        # authors (e.g. a non-ICML format parsed as a single bare author);
        # fall back to the regex parser which can split more robustly.
        body_stripped = body_text.lstrip() if body_text else ""
        if body_stripped.startswith(",") or body_stripped.startswith("and "):
            return self._fallback_parse(cleaned or raw, raw)

        year, body_no_year = _extract_year(body_text)
        title, venue = _split_title_venue(body_no_year)

        # Strip leading punctuation from title (URL-only / misc references)
        title = re.sub(r"^[,.\s]+", "", title).strip()

        # Detect merged references in venue: ", YYYY." followed by a
        # second unrelated reference that was accidentally concatenated.
        if venue:
            venue_yr_m = re.search(r",\s+(\d{4}[a-z]?)\s*\.", venue)
            if venue_yr_m:
                yr_val = int(venue_yr_m.group(1)[:4])
                if 1900 <= yr_val <= 2099:
                    after_text = venue[venue_yr_m.end() :].strip()
                    if after_text and not re.match(
                        r"(?:URL|doi|DOI|ISSN|ISBN|pp\.|arXiv|http)", after_text
                    ):
                        year = venue_yr_m.group(1)
                        venue = venue[: venue_yr_m.start()].rstrip()

        return Reference(
            authors=author_names,
            year=year,
            title=title,
            venue=venue,
            raw=raw,
        )

    # ── regex fallback for unusual formatting ────────────────────────

    @staticmethod
    def _fallback_parse(text: str, raw: str | None = None) -> Reference:
        if raw is None:
            raw = text
        ref = Reference(raw=raw)

        m = re.search(r"[,.\s]\s*(\d{4}[a-z]?)\s*\.\s*$", text)
        if not m:
            m = re.search(r",\s+(\d{4}[a-z]?)\b", text)
        if not m:
            m = re.search(
                rf"[,.]\s+{_MONTHS_PAT}\s+(\d{{4}}[a-z]?)\b",
                text,
                re.IGNORECASE,
            )
        if not m:
            for candidate in reversed(list(re.finditer(r"\b(\d{4}[a-z]?)\b", text))):
                cyr = int(candidate.group(1)[:4])
                if 1900 <= cyr <= 2099:
                    p = candidate.start()
                    if p > 0 and text[p - 1] == "/":
                        continue
                    if candidate.end() < len(text) and text[candidate.end()] == "/":
                        continue
                    m = candidate
                    break
        if not m:
            return ref

        yr = int(m.group(1)[:4])
        if not (1900 <= yr <= 2099):
            return ref

        ref.year = m.group(1)

        # Find author-body boundary: first ". " not part of an initial
        author_body_split = None
        for ab_m in re.finditer(r"\.\s+", text):
            pos = ab_m.end()
            if pos >= m.start():
                break

            after_char = text[pos : pos + 1]
            if not (after_char and after_char.isupper()):
                continue

            preceding = text[: ab_m.start()]
            is_likely_initial = False
            if preceding:
                last_word = preceding.split()[-1] if preceding.split() else ""
                if len(last_word) == 1 and last_word.isupper():
                    is_likely_initial = True
                elif ab_m.start() > 0 and re.match(
                    r"^[A-Z]$", text[ab_m.start() - 1 : ab_m.start()]
                ):
                    is_likely_initial = True

            if is_likely_initial:
                remaining = text[pos:]
                next_word = remaining.split()[0] if remaining.split() else ""
                next_word_clean = next_word.rstrip(".,;:")
                if re.match(r"^[A-Z]\.?$", next_word_clean):
                    continue

            author_body_split = ab_m
            break

        if author_body_split is None:
            before_year = text[: m.start()].rstrip("., ")
            if before_year:
                ref.title = before_year
            after_year = text[m.end() :].strip(". ")
            if after_year:
                ref.venue = after_year
            return ref

        author_part = text[: author_body_split.start()].strip()
        body_part = text[author_body_split.end() : m.start()].strip()

        body_part = _EDITOR_TAG_RE.sub("", body_part)

        et_al_match = re.search(r",?\s*et\s+al\.?\s*$", author_part, re.IGNORECASE)
        if et_al_match:
            author_part = author_part[: et_al_match.start()]
            raw_authors = [a.strip() for a in _split_icml_authors(author_part)]
            raw_authors.append("et al.")
        elif ", and " in author_part:
            last_and = author_part.rfind(", and ")
            before = author_part[:last_and]
            after = author_part[last_and + 6 :].strip()
            raw_authors = _split_icml_authors(before)
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


def _split_icml_authors(text: str) -> list[str]:
    """Split a comma-separated ICML author string into individual names.

    ICML format uses ``LastName, F.`` so commas appear both inside names
    (between last name and initials) and between names.  A between-name
    comma is always preceded by an initial (``X.``) while an inside-name
    comma is followed by initials.
    """
    authors: list[str] = []
    parts = text.split(",")
    current: list[str] = []
    for part in parts:
        current.append(part)
        joined = ",".join(current).strip()
        if re.search(r"[A-Z]\.\s*$", joined):
            authors.append(joined)
            current = []

    if current:
        trailing = ",".join(current).strip()
        if trailing:
            if authors:
                authors[-1] += ", " + trailing
            else:
                authors.append(trailing)

    return [a.strip() for a in authors if a.strip()]


if __name__ == "__main__":
    import glob
    import os

    target_conference_name = "icml"
    reference_document_dir = f"output/parsed_references/docling"
    parser = IcmlReferenceParser(raise_error=True)
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
