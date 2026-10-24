"""Command to run
python -m src.reference_parser.ieee_reference_parser
"""

from __future__ import annotations

import re
import unicodedata

from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener

from .antlr4_grammar.ieee.ieeeLexer import ieeeLexer
from .antlr4_grammar.ieee.ieeeParser import ieeeParser
from .reference_parser import Reference, ReferenceParser

# Patterns that signal the start of venue information (used in title/venue split
# for the body text when there is no separate title from the grammar).
_VENUE_STARTERS = re.compile(
    r"(?:"
    r"[Ii]n\s|"
    r"arXiv[:\s]|"
    r"arXiv[-\s]*[Pp]reprint[:\s]|"
    r"[Pp]roc\.\s|"
    r"Proceedings\s|"
    r"Ph\.?\s*D\.?\s|"
    r"Master'?s\s|"
    r"Technical\s[Rr]eport|"
    r"(?:[A-Z][a-zA-Z&:\s]*(?:Journal|Transactions|Letters|Review|Conference|Workshop|Proceedings|Advances|Annals|Bulletin|Reports|Research|Quarterly|Studies|Systems|Society|Computing|Intelligence|Surveys|Communications))|"
    r"Springer|Elsevier|Wiley|PMLR|IEEE|ACM|AAAI|SIAM|"
    r"Cambridge|Oxford|MIT|Princeton|Artech|Prentice|Hoboken|"
    r"NeurIPS|ICML|ICLR|ECCV|ICCV|CVPR|EMNLP|NAACL|"
    r"URL[:\s]|"
    r"doi[:\s]|"
    r"pp\.[\ \d]|"
    r"vol\.|"
    r"no\."
    r")"
)

_JOURNAL_LIKE = re.compile(r"^(?:[A-Z][a-zA-Z]*(?:(?:\s+|:\s*)[A-Za-z&]+)*)\s*,")

_REF_NUMBER_RE = re.compile(r"^\[[\w\s+]+\]\s*")
_DAGGER_RE = re.compile(r"[\u2020\u2021]+")
_SPACED_INITIAL_RE = re.compile(r"\b([A-Z]) \.")
_PAGE_METADATA_RE = re.compile(r"^Pages?\s*:", re.IGNORECASE)
_TRAILING_CITATIONS_RE = re.compile(r"\.\s+(\d{1,2}(?:\s*,\s*\d{1,2})*)\s*$")

# ── Non-reference text detectors (author bios, table rows, fragments) ─────────
# Author biography: "Name (Member/Fellow, IEEE) ..." OR "Name received the [degree] ..."
_AUTHOR_BIO_RE = re.compile(
    r"^[A-Z][A-Za-z\s.,\u00C0-\u024F-]+"
    r"(?:"
    r"\((?:Senior\s+|Graduate\s+Student\s+|Student\s+|Life\s+)?"
    r"(?:Member|Fellow),\s*IEEE\)"
    r"|received\s+(?:the|his|her|their)\s+(?:B\.|M\.|Ph\.?D|bachelor|master|doctor|Laurea)"
    r"|is\s+(?:a\s+|an\s+|currently\s+|now\s+)"
    r"|photograph\s+and\s+biography"
    r"|was\s+a\s+recipient\s+of"
    r"|was\s+born\s+in\s+"
    r")"
)
# Biography with honorific prefix: "Dr Faenzi won ...", "Prof. Maci was ..."
_HONORIFIC_BIO_RE = re.compile(
    r"^(?:Dr|Prof)\.?\s+[A-Z][A-Za-z\u00C0-\u024F]+\s+"
    r"(?:won|served|was|received|is|has|had|co-founded|founded|joined|coordinated)",
    re.IGNORECASE,
)
# Biography continuation: sentences starting with pronouns + bio verbs.
_BIO_CONTINUATION_RE = re.compile(
    r"^(?:"
    # Pronoun-led bio sentences: "He has authored...", "Her research..."
    r"(?:He|She|They|His|Her|Their)\s+"
    r"(?:has|had|have|is|was|were|received|joined|current|also|served|co-founded|founded|coordinated|research)"
    # Temporal bio starters: "From 2008 to 2015, he was...", "Since 1997, he has..."
    r"|From\s+(?:\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\s+to\s+"
    r"|Since\s+\d{4}\b"
    # Activity bio starters: "During his doctoral studies..."
    r"|During\s+(?:his|her|their|the)\s+"
    # Institutional move: "In 2002, he was appointed..."
    r"|In\s+(?:\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December))\s*,\s*(?:he|she|they)\s+"
    r")",
    re.IGNORECASE,
)
# Non-reference text that commonly appears near reference sections.
_NON_REFERENCE_TEXT_RE = re.compile(
    r"^(?:"
    r"Open\s+Access\s+funding|"
    r"This\s+work\s+was\s+|"
    r"Acknowledg|"
    r"(?:\w+\s+)?Award\s+(?:at|in|from)\s+"
    r")",
    re.IGNORECASE,
)
# Markdown table row
_TABLE_ROW_RE = re.compile(r"^\|")
# Leading parenthesized date in electronic references: "(2025)." or "(n.d.)."
_LEADING_PAREN_DATE_RE = re.compile(
    r"^\(\s*(?:\d{4}(?:\s*,\s*\w+\.?)?|n\.?\s*d\.?)\s*\)\s*\.?\s*"
)
# Minimum length for a plausible reference string (author + title at minimum)
_MIN_REFERENCE_LEN = 20
# Corrupted author comma: "W., Santoso" → "W. Santoso"
_AUTHOR_COMMA_ARTIFACT_RE = re.compile(r"([A-Z]\.)\s*,\s+(?=[A-Z][a-z])")

_YEAR_TOKEN_PAT = r"(\d{4}(?:\s*[a-z])?)"
_MONTHS_PAT = (
    r"(?:January|February|March|April|May|June"
    r"|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
)

_ACCENT_ARTIFACT_RE = re.compile(r"([A-Za-z])[`´¨ˆˇ˙˚¸~]\s+([A-Za-z])")
_STANDALONE_ACCENT_RE = re.compile(r"[`´¨ˆˇ˙˚¸~]")

# ── Additional preprocessing patterns for OCR / extraction artifacts ──────────
# Fix lowercase initials after a hyphen: Y.-a. → Y.-A.
_LOWERCASE_HYPH_INITIAL_RE = re.compile(r"(?<=[A-Z]\.-)([a-z])(?=\.)")
# Fix missing period in hyphenated initials: M-J. → M.-J.
_MISSING_DOT_INITIAL_RE = re.compile(r"\b([A-Z])-([A-Z]\.)")
# Strip trailing bracketed citation numbers: ... [1]  or ... [1, 2]
_TRAILING_BRACKET_CITATION_RE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]\s*$")
# Fix concatenated 'and' before initials (OCR artefact): ,andK → , and K
_CONCAT_AND_RE = re.compile(r"(?<=,)and(?=[A-Z])")


def _clean_year_token(token: str) -> str:
    return re.sub(r"\s+", "", token)


def _normalize_ocr_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ı", "i").replace("İ", "I")
    text = _ACCENT_ARTIFACT_RE.sub(r"\1\2", text)
    text = _STANDALONE_ACCENT_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class _SilentErrorListener(ErrorListener):
    """Collects parse errors without printing to stderr."""

    def __init__(self) -> None:
        super().__init__()
        self.errors: list[str] = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.errors.append(f"line {line}:{column} {msg}")


# ── Author extraction ────────────────────────────────────────────────────────


def _reconstruct_initials(ctx: ieeeParser.InitialsContext) -> str:
    """Reconstruct the initials string (e.g. 'A. B.' or 'Z.-H.')."""
    parts: list[str] = []
    for i in range(ctx.getChildCount()):
        tok = ctx.getChild(i).getText()
        if (
            parts
            and tok not in ("-", "\u2013", "\u2014")
            and parts[-1] not in ("-", "\u2013", "\u2014")
        ):
            parts.append(" ")
        parts.append(tok)
    return "".join(parts)


def _reconstruct_name_words(ctx_list) -> str:
    """Reconstruct last-name (+ optional particles) from nameWord contexts."""
    tokens = [nw.getText() for nw in ctx_list]
    if not tokens:
        return ""
    parts = [tokens[0]]
    for prev, cur in zip(tokens, tokens[1:]):
        if cur in ("-", "\u2013", "\u2014") or prev in ("-", "\u2013", "\u2014"):
            space = False
        elif cur in ("'", "\u2019") or prev in ("'", "\u2018"):
            space = False
        else:
            space = True
        if space:
            parts.append(" ")
        parts.append(cur)
    return "".join(parts)


def _extract_ieee_author_name(ctx: ieeeParser.AuthorNameContext) -> str:
    """Extract a human-readable author name from an IEEE authorName context.

    IEEE format is initials-first: ``F. von LastName`` or ``F.-G. LastName``.
    """
    initials_ctx = ctx.initials()
    name_words = ctx.nameWord()

    initials_str = _reconstruct_initials(initials_ctx) if initials_ctx else ""
    name_str = _reconstruct_name_words(name_words) if name_words else ""

    if initials_str and name_str:
        return f"{initials_str} {name_str}"
    return initials_str or name_str


# ── Title / body reconstruction ──────────────────────────────────────────────

_ORDINAL_SUFFIXES = {"th", "st", "nd", "rd"}


def _add_spacing(parts: list[str], prev_tok: str, cur_tok: str) -> list[str]:
    """Append a space (or not) between two adjacent tokens based on punctuation rules."""
    space = True

    if cur_tok in (".", ",", ";", ":", ")", "]", "?", "!", "\u2019"):
        space = False
    elif prev_tok in ("(", "[", "'", "\u2018", "\u2019"):
        space = False
    elif prev_tok == "%" or cur_tok == "%":
        space = False
    elif prev_tok == "=" or cur_tok == "=":
        space = False
    elif prev_tok == "+" or cur_tok == "+":
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
    elif prev_tok == "@" or cur_tok == "@":
        space = False

    if space:
        parts.append(" ")
    parts.append(cur_tok)
    return parts


def _tokens_to_str(token_texts: list[str]) -> str:
    """Join a list of token strings into natural text."""
    if not token_texts:
        return ""
    parts = [token_texts[0]]
    for prev, cur in zip(token_texts, token_texts[1:]):
        _add_spacing(parts, prev, cur)
    return "".join(parts)


def _extract_title_text(ctx: ieeeParser.TitleContext) -> str:
    """Reconstruct the title string from the title context (without surrounding quotes).

    IEEE puts a trailing comma inside the closing quote ('Title,'); this is
    stripped to give a clean title string.
    """
    token_texts: list[str] = []

    if ctx.DQUOTE():
        # Alternative 1: DQUOTE titleTokenDq* DQUOTE
        for child_ctx in ctx.titleTokenDq():
            token_texts.append(child_ctx.getText())
    else:
        # Alternative 2: APOS titleTokenApos* APOS
        for child_ctx in ctx.titleTokenApos():
            token_texts.append(child_ctx.getText())

    raw = _tokens_to_str(token_texts)
    # Strip the trailing separator comma that IEEEtran.bst includes inside the quote.
    return raw.rstrip(",").strip()


def _reconstruct_body(ctx: ieeeParser.BodyContext) -> str:
    """Reconstruct the body text from the parse tree."""
    if ctx is None:
        return ""
    token_texts: list[str] = []
    for i in range(ctx.getChildCount()):
        tok = ctx.getChild(i).getText()
        if tok:
            token_texts.append(tok)
    return _tokens_to_str(token_texts)


def _extract_corporate_author(ctx: ieeeParser.CorporateAuthorContext) -> str:
    """Reconstruct the corporate/organizational author name from its context."""
    token_texts: list[str] = []
    for i in range(ctx.getChildCount()):
        tok = ctx.getChild(i).getText()
        if tok:
            token_texts.append(tok)
    return _tokens_to_str(token_texts)


def _split_title_from_body(body: str) -> tuple[str, str]:
    """Split an unquoted body into (title, remaining_body) for book/unquoted refs.

    IEEE book references place the title (originally italicized) before the
    publisher/location info.  After PDF extraction the italics are lost, so we
    use heuristic boundaries: known venue-starter patterns, or a sentence-ending
    period followed by an uppercase letter.
    """
    if not body:
        return "", ""

    # 1. Try known venue-starter patterns (e.g. "in ", "Proc.", "Springer", …)
    m = _VENUE_STARTERS.search(body)
    if m:
        title = body[: m.start()].rstrip(", ")
        return title, body[m.start():]

    # 2. Try ". " followed by an uppercase letter (sentence boundary)
    m = re.search(r"\.\s+(?=[A-Z])", body)
    if m:
        title = body[: m.start()].strip()
        return title, body[m.end() :].lstrip()

    # 3. No clear boundary — treat the whole body as venue info
    return "", body


# ── Year extraction ───────────────────────────────────────────────────────────


def _extract_year(body: str) -> tuple[str, str]:
    """Extract the publication year from an IEEE body string.

    IEEE format places the year at the end, typically as ', YYYY.' or
    ', Month YYYY.' or ', vol. X, no. Y, pp. Z, YYYY.'.

    Returns (year_str, body_without_year).
    """
    cleaned = body.rstrip()
    if cleaned.endswith("."):
        cleaned = cleaned[:-1].rstrip()

    # ", YYYY[a-z]?" at end — guard against page-range numbers (e.g. "1877-1901")
    m = re.search(rf",\s+{_YEAR_TOKEN_PAT}\s*$", cleaned)
    if m and 1900 <= int(m.group(1)[:4]) <= 2099:
        pos = m.start()
        if pos == 0 or cleaned[pos - 1] != "-":
            return _clean_year_token(m.group(1)), cleaned[: m.start()].rstrip()

    # ". YYYY[a-z]?" at end
    m = re.search(rf"\.\s+{_YEAR_TOKEN_PAT}\s*$", cleaned)
    if m and 1900 <= int(m.group(1)[:4]) <= 2099:
        return _clean_year_token(m.group(1)), cleaned[: m.start()].rstrip()

    # "[,.] Month YYYY" at end
    m = re.search(
        rf"[,.]\s+{_MONTHS_PAT}\.?\s+{_YEAR_TOKEN_PAT}\s*$",
        cleaned,
        re.IGNORECASE,
    )
    if m and 1900 <= int(m.group(1)[:4]) <= 2099:
        return _clean_year_token(m.group(1)), cleaned[: m.start()].rstrip()

    # Any standalone YYYY — scan in reverse, skip page-range numbers and URL paths.
    matches = list(re.finditer(rf"\b{_YEAR_TOKEN_PAT}\b", cleaned))
    for m in reversed(matches):
        yr = int(m.group(1)[:4])
        if 1900 <= yr <= 2099:
            pos = m.start()
            # Skip years embedded in numeric ranges (e.g. "1877-1901") or URL paths.
            if pos > 0 and cleaned[pos - 1] in ("/", "-"):
                continue
            if m.end() < len(cleaned) and cleaned[m.end()] in ("/", "-"):
                continue
            # Skip arXiv-style fragment IDs like "2005.14165"
            if re.match(r"^\.\d{4,5}\b", cleaned[m.end() :]):
                continue
            before = cleaned[:pos].rstrip(", ")
            # Strip the separator punctuation that preceded what follows the year
            after = re.sub(r"^[\s,;.]+", "", cleaned[m.end() :]).strip()
            result = before
            if after:
                result += ". " + after
            return _clean_year_token(m.group(1)), result

    return "", cleaned


# ── Main parser class ─────────────────────────────────────────────────────────


class IeeeReferenceParser(ReferenceParser):
    """Parser for references formatted in IEEE bibliography style (IEEEtran.bst).

    IEEE references use an initials-first author format (``F. LastName``),
    a comma-and-"and" separated author list, a double-quoted (or PDF-extracted
    single-quoted) title, and a body ending with the publication year.

    Example::

        K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image
        recognition," in Proc. CVPR, 2016, pp. 770-778.

    Args:
        raise_error: When ``True``, raises :class:`ValueError` on any ANTLR4
            parse failure instead of falling back to regex extraction.
    """

    def __init__(self, raise_error: bool = False) -> None:
        self.raise_error = raise_error

    def parse_one(self, text: str) -> Reference | None:
        if not text or not text.strip():
            return None

        text = text.strip()
        # Strip leading bracketed reference numbers: [1], [23], [4a], etc.
        text = _REF_NUMBER_RE.sub("", text).strip()

        if not text:
            return None
        if _PAGE_METADATA_RE.match(text):
            return None
        # Skip text that is too short to be a meaningful reference.
        if len(text) < _MIN_REFERENCE_LEN:
            return None
        # Skip markdown table rows.
        if _TABLE_ROW_RE.match(text):
            return None
        # Skip author biographies (commonly appended to IEEE papers).
        if _AUTHOR_BIO_RE.match(text):
            return None
        # Skip honorific-prefixed bio sentences ("Dr Faenzi won ...", "Prof. Maci was ...").
        if _HONORIFIC_BIO_RE.match(text):
            return None
        # Skip non-reference text (funding acknowledgements, award fragments, etc.).
        if _NON_REFERENCE_TEXT_RE.match(text):
            return None
        # Skip biography continuation sentences ("He has authored...", "His research...").
        if _BIO_CONTINUATION_RE.match(text):
            return None

        raw_text = text

        # Common OCR / extraction artifacts.
        text = _DAGGER_RE.sub("", text)
        text = re.sub(r",\s*,", ",", text)
        text = _SPACED_INITIAL_RE.sub(r"\1.", text)
        # Fix lowercase hyphenated initials: Y.-a. → Y.-A.
        text = _LOWERCASE_HYPH_INITIAL_RE.sub(
            lambda m: m.group(1).upper(), text
        )
        # Fix missing period in hyphenated initials: M-J. → M.-J.
        text = _MISSING_DOT_INITIAL_RE.sub(r"\1.-\2", text)
        # Strip trailing bracketed citation numbers: [1], [2, 3], etc.
        text = _TRAILING_BRACKET_CITATION_RE.sub("", text)
        # Fix concatenated 'and' (OCR): ,andK → , and K
        text = _CONCAT_AND_RE.sub(" and ", text)
        # Fix corrupted author comma: "W., Santoso" → "W. Santoso"
        text = _AUTHOR_COMMA_ARTIFACT_RE.sub(r"\1 ", text)
        # Strip leading parenthesized date for electronic references:
        # "(2025). Title..." → "Title..."
        text = _LEADING_PAREN_DATE_RE.sub("", text)
        text = text.strip()

        if not text:
            return None
        if len(text) < _MIN_REFERENCE_LEN:
            return None

        input_stream = InputStream(text)

        lexer = ieeeLexer(input_stream)
        lexer.removeErrorListeners()
        lex_errs = _SilentErrorListener()
        lexer.addErrorListener(lex_errs)

        token_stream = CommonTokenStream(lexer)

        parser = ieeeParser(token_stream)
        parser.removeErrorListeners()
        parse_errs = _SilentErrorListener()
        parser.addErrorListener(parse_errs)

        tree = parser.reference()

        if parse_errs.errors:
            if self.raise_error:
                raise ValueError(
                    f"ANTLR4 grammar (IEEE) failed to parse reference. "
                    f"Errors: {parse_errs.errors!r}. Text: {text!r}"
                )
            return self._fallback_parse(text, raw_text)

        return self._extract_reference(tree, raw_text)

    def _extract_reference(
        self, tree: ieeeParser.ReferenceContext, raw: str
    ) -> Reference:
        authors_ctx = tree.authors()
        corp_ctx = tree.corporateAuthor()
        title_ctx = tree.title()
        body_ctx = tree.body()

        # ── Authors ──
        author_names: list[str] = []
        if authors_ctx is not None:
            for author_ctx in authors_ctx.authorName():
                name = _extract_ieee_author_name(author_ctx)
                if name:
                    author_names.append(_normalize_ocr_text(name))
            if authors_ctx.ET_AL() is not None:
                author_names.append("et al.")
        elif corp_ctx is not None:
            corp_name = _normalize_ocr_text(_extract_corporate_author(corp_ctx))
            if corp_name:
                author_names.append(corp_name)

        # ── Title ──
        title = ""
        if title_ctx is not None:
            title = _extract_title_text(title_ctx)
            title = _normalize_ocr_text(title)
            title = re.sub(r"^[,.\s]+", "", title).strip()

        # ── Body / venue ──
        body_text = _reconstruct_body(body_ctx)

        # If no quoted title was found (book / unquoted alternative), attempt to
        # split the body into title + remaining venue information.
        if not title_ctx and body_text:
            split_title, body_text = _split_title_from_body(body_text)
            if split_title:
                title = _normalize_ocr_text(split_title)

        year, venue = _extract_year(body_text)

        venue = _normalize_ocr_text(venue)
        # Strip leading "in " from conference venue (IEEEtran adds "in Proc...")
        venue = re.sub(r"^[Ii]n\s+", "in ", venue)

        return Reference(
            authors=author_names,
            year=year,
            title=title,
            venue=venue,
            raw=raw,
        )

    @staticmethod
    def _fallback_parse(text: str, raw: str | None = None) -> Reference:
        """Regex-based fallback for references that fail ANTLR4 grammar parsing."""
        if raw is None:
            raw = text
        ref = Reference(raw=raw)

        # Extract year (at the end of IEEE references)
        m = None
        cleaned = text.rstrip()
        if cleaned.endswith("."):
            cleaned = cleaned[:-1].rstrip()

        for pattern in [
            rf",\s+{_YEAR_TOKEN_PAT}\s*$",
            rf"\.\s+{_YEAR_TOKEN_PAT}\s*$",
            rf"[,.]\s+{_MONTHS_PAT}\.?\s+{_YEAR_TOKEN_PAT}\s*$",
        ]:
            m = re.search(pattern, cleaned, re.IGNORECASE)
            if m:
                yr = int(m.group(1 if "MONTHS" not in pattern else 1)[:4])
                if 1900 <= yr <= 2099:
                    break
                m = None

        if not m:
            for candidate in reversed(
                list(re.finditer(rf"\b{_YEAR_TOKEN_PAT}\b", cleaned))
            ):
                yr = int(candidate.group(1)[:4])
                if 1900 <= yr <= 2099:
                    pos = candidate.start()
                    if pos > 0 and cleaned[pos - 1] == "/":
                        continue
                    if (
                        candidate.end() < len(cleaned)
                        and cleaned[candidate.end()] == "/"
                    ):
                        continue
                    if re.match(r"^\.\d{4,5}\b", cleaned[candidate.end() :]):
                        continue
                    m = candidate
                    break

        if m:
            yr_str = m.group(1) if "MONTHS" not in m.re.pattern else m.group(1)
            ref.year = _clean_year_token(yr_str)
            before_year = cleaned[: m.start()].rstrip(", ")
            after_year = cleaned[m.end() :].strip(". ")
            ref.venue = _normalize_ocr_text(after_year) if after_year else ""

            # Split authors and title: IEEE title is in quotes
            # Look for the first quoted block as the title
            quote_m = re.search(
                r'["\u201C\u2018\'](.*?)["\u201D\u2019]', before_year, re.DOTALL
            )
            if quote_m:
                author_part = before_year[: quote_m.start()].rstrip(", ")
                title_raw = quote_m.group(1).rstrip(",").strip()
                ref.title = _normalize_ocr_text(title_raw)
            else:
                # No quoted title found — try to split on ". " boundary
                dot_m = re.search(r"\.\s+", before_year)
                if dot_m:
                    author_part = before_year[: dot_m.start()].strip()
                    ref.title = _normalize_ocr_text(before_year[dot_m.end() :].strip())
                else:
                    author_part = before_year

            # Parse author list (comma-separated, "and" before last)
            author_part = author_part.rstrip(", ")
            et_al_m = re.search(r",?\s*et\s+al\.?\s*$", author_part, re.IGNORECASE)
            if et_al_m:
                author_part = author_part[: et_al_m.start()]
                raw_authors = [
                    a.strip()
                    for a in re.split(r",\s*(?:and\s+)?", author_part)
                    if a.strip()
                ]
                raw_authors.append("et al.")
            elif re.search(r",\s*and\s+", author_part):
                last_and = re.search(r",\s*and\s+", author_part)
                before_and = author_part[: last_and.start()]
                after_and = author_part[last_and.end() :].strip()
                raw_authors = [a.strip() for a in before_and.split(",") if a.strip()]
                if after_and:
                    raw_authors.append(after_and)
            elif " and " in author_part:
                parts = author_part.rsplit(" and ", 1)
                raw_authors = [p.strip() for p in parts if p.strip()]
            else:
                raw_authors = [author_part.strip()] if author_part.strip() else []

            ref.authors = [_normalize_ocr_text(a) for a in raw_authors if a]

        return ref


if __name__ == "__main__":
    import glob
    import os

    target_conference_name = "ieee"
    reference_document_dir = f"output/parsed_references/docling"
    parser = IeeeReferenceParser(raise_error=True)
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
