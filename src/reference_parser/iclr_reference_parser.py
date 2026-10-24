"""Command to run
python -m src.reference_parser.iclr_reference_parser
"""

from __future__ import annotations

import re
import unicodedata

from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener

from .antlr4_grammar.iclr.iclrLexer import iclrLexer
from .antlr4_grammar.iclr.iclrParser import iclrParser
from .reference_parser import Reference, ReferenceParser

_VENUE_STARTERS = re.compile(
    r"(?:"
    r"In\s|"
    r"arXiv[:\s]|"
    r"arXiv[-\s]*[Pp]reprint[:\s]|"
    r"ArXiv\s|"
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
    r"URL[:\s]|"
    r"doi[:\s]|"
    r"pp\.[\ \d]|"
    r"ISSN\s|"
    r"ISBN\s"
    r")"
)

_JOURNAL_LIKE = re.compile(r"^(?:[A-Z][a-zA-Z]*(?:(?:\s+|:\s*)[A-Za-z&]+)*)\s*,")

_TRAILING_CITATIONS_RE = re.compile(r"\.\s+(\d{1,2}(?:\s*,\s*\d{1,2})*)\s*$")
_REF_NUMBER_RE = re.compile(r"^\[[\w\s+]+\]\s*")
_DAGGER_RE = re.compile(r"[\u2020\u2021]+")
_SPACED_INITIAL_RE = re.compile(r"\b([A-Z]) \.")
_PAGE_METADATA_RE = re.compile(r"^Pages?\s*:", re.IGNORECASE)
_MARKDOWN_ARTIFACT_RE = re.compile(r"^[`~]{3,}\s*\w*\s*$")
_NAME_SUFFIX_COMMA_RE = re.compile(r",\s+(Jr|Sr)\.", re.IGNORECASE)
_EDITOR_TAG_RE = re.compile(r"^\s*\(eds?\.?\)\s*\.?\s*")
_INLINE_METADATA_START_RE = re.compile(
    r"\b(?:URL[:\s]|doi[:\s]|arXiv[-\s]*[Pp]reprint[:\s]|arXiv[:\s])",
    re.IGNORECASE,
)

_YEAR_TOKEN_PAT = r"(\d{4}(?:\s*[a-z])?)"
_SUFFIX_ARTIFACT_AFTER_YEAR_RE = re.compile(r"^[\s.,;:()]+([a-z])\b")

_ACCENT_ARTIFACT_RE = re.compile(r"([A-Za-z])[`´¨ˆˇ˙˚¸~]\s+([A-Za-z])")
_STANDALONE_ACCENT_RE = re.compile(r"[`´¨ˆˇ˙˚¸~]")
_SINGLE_TRAILING_LETTER_SPLIT_RE = re.compile(r"\b([A-Za-z]{2,})\s+([a-z])\b")
_LOWERCASE_NAME_PARTICLES = {
    "al",
    "bin",
    "da",
    "de",
    "del",
    "der",
    "di",
    "ibn",
    "la",
    "le",
    "van",
    "vanden",
    "vander",
    "von",
}
_GLUED_LEADING_LOWERCASE_RE = re.compile(r"\b([a-z])([A-Z][A-Za-z]+)\b")
_SINGLE_TOKEN_NAME_RE = re.compile(r"^[A-Z][A-Za-z'’.-]+$")
_HYPHEN_SPLIT_RE = re.compile(r"\b([A-Za-z]+-[A-Za-z]{2,})\s+([a-z]{2,})\b")

_MONTHS_PAT = (
    r"(?:January|February|March|April|May|June"
    r"|July|August|September|October|November|December)"
)


def _clean_year_token(token: str) -> str:
    """Normalize extracted year tokens such as '2025 a' -> '2025a'."""
    return re.sub(r"\s+", "", token)


def _normalize_ocr_text(text: str) -> str:
    """Normalize common OCR/extraction artifacts without heavy transliteration."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ı", "i").replace("İ", "I")
    text = _ACCENT_ARTIFACT_RE.sub(r"\1\2", text)
    text = _STANDALONE_ACCENT_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_author_name(name: str) -> str:
    """Fix common OCR artifacts in extracted author names."""
    name = _normalize_ocr_text(name)
    name = _GLUED_LEADING_LOWERCASE_RE.sub(r"\2", name)
    name = _SINGLE_TRAILING_LETTER_SPLIT_RE.sub(r"\1\2", name)
    name = _HYPHEN_SPLIT_RE.sub(r"\1\2", name)
    # Accent OCR often appears as apostrophe-in-middle (e.g., Guzm'an).
    name = re.sub(r"([A-Za-z])['’]([a-z])", r"\1\2", name)

    tokens = name.split()
    merged_tokens: list[str] = []
    i = 0
    while i < len(tokens):
        cur = tokens[i]
        if i + 1 < len(tokens):
            nxt = tokens[i + 1]
            nxt_l = nxt.lower()
            if (
                cur.isalpha()
                and nxt.isalpha()
                and nxt.islower()
                and nxt_l not in _LOWERCASE_NAME_PARTICLES
                and (
                    len(nxt) <= 3
                    or len(cur) <= 2
                    or (
                        cur[0].isupper()
                        and len(cur) <= 7
                        and len(nxt) <= 7
                        and (i + 2 >= len(tokens) or tokens[i + 2][:1].isupper())
                    )
                )
            ):
                merged_tokens.append(cur + nxt)
                i += 2
                continue
        merged_tokens.append(cur)
        i += 1

    name = " ".join(merged_tokens)
    return re.sub(r"\s+", " ", name).strip()


def _is_single_token_name_piece(name: str) -> bool:
    if " " in name:
        return False
    if name.endswith("."):
        return False
    return bool(_SINGLE_TOKEN_NAME_RE.match(name))


def _merge_split_single_token_authors(authors: list[str]) -> list[str]:
    """Merge adjacent one-token artifacts such as 'Dmitry', 'Lepikhin'."""
    merged: list[str] = []
    i = 0
    while i < len(authors):
        cur = authors[i].strip()
        if (
            i + 1 < len(authors)
            and _is_single_token_name_piece(cur)
            and _is_single_token_name_piece(authors[i + 1].strip())
        ):
            merged.append(f"{cur} {authors[i + 1].strip()}")
            i += 2
            continue
        merged.append(cur)
        i += 1
    return merged


class _SilentErrorListener(ErrorListener):
    """Collects errors without printing to stderr."""

    def __init__(self) -> None:
        super().__init__()
        self.errors: list[str] = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.errors.append(f"line {line}:{column} {msg}")


def _extract_iclr_author_name(ctx: iclrParser.AuthorNameContext) -> str:
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
    return _normalize_author_name(name)


def _reconstruct_body(ctx: iclrParser.BodyContext) -> str:
    """Reconstruct the body text from the parse tree, preserving spacing intent."""
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

    ordinal_suffixes = {"th", "st", "nd", "rd"}
    contraction_suffixes = {"t", "s", "re", "ve", "ll", "d", "m", "em"}

    prev_prev_tok = ""
    parts = [tokens[0]]
    for prev_tok, cur_tok in zip(tokens, tokens[1:]):
        space = True

        if cur_tok in (".", ",", ";", ":", ")", "?", "!", "'", "\u2019"):
            space = False
        elif prev_tok == "(":
            space = False
        elif prev_tok in ("'", "\u2018", "\u2019"):
            if prev_prev_tok and prev_prev_tok[-1:].isalpha():
                # After a word → closing quote, contraction, or name particle.
                if cur_tok.lower() in contraction_suffixes:
                    space = False  # contraction: don't, it's, we're
                elif len(prev_prev_tok) == 1 and prev_prev_tok.isalpha():
                    space = False  # name particle: O'Connor, D'souza
                else:
                    space = True   # closing quote → space before next word
            else:
                space = False  # opening quote → no space before quoted word
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
        elif prev_tok[-1:].isdigit() and cur_tok.lower() in ordinal_suffixes:
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
        prev_prev_tok = prev_tok

    return "".join(parts)


def _strip_trailing_citations(text: str) -> str:
    """Strip trailing in-paper citation page numbers (e.g., '. 2, 3, 7')."""
    return _TRAILING_CITATIONS_RE.sub(".", text)


def _strip_reference_number(text: str) -> str:
    """Strip leading bracketed reference number like [1], [23], etc."""
    return _REF_NUMBER_RE.sub("", text)


def _extract_year(body: str) -> tuple[str, str]:
    """Extract publication year from body text.

    ICLR natbib typically places year near the end, but it can appear before
    trailing metadata in some entry types (book/misc/techreport/thesis).
    """
    cleaned = body.rstrip()
    if cleaned.endswith("."):
        cleaned = cleaned[:-1].rstrip()

    # ", YYYY[a-z]?" at end
    m = re.search(rf",\s+{_YEAR_TOKEN_PAT}\s*$", cleaned)
    if m and 1900 <= int(m.group(1)[:4]) <= 2099:
        return _clean_year_token(m.group(1)), cleaned[: m.start()].rstrip()

    # ". YYYY[a-z]?" at end
    m = re.search(rf"\.\s+{_YEAR_TOKEN_PAT}\s*$", cleaned)
    if m and 1900 <= int(m.group(1)[:4]) <= 2099:
        return _clean_year_token(m.group(1)), cleaned[: m.start()].rstrip()

    # "[,.] Month YYYY" at end
    m = re.search(
        rf"[,.]\s+{_MONTHS_PAT}\s+{_YEAR_TOKEN_PAT}\s*$",
        cleaned,
        re.IGNORECASE,
    )
    if m and 1900 <= int(m.group(1)[:4]) <= 2099:
        return _clean_year_token(m.group(1)), cleaned[: m.start()].rstrip()

    # ", YYYY" anywhere near the end
    matches = list(re.finditer(rf",\s+{_YEAR_TOKEN_PAT}\b", cleaned))
    for m in reversed(matches):
        yr = int(m.group(1)[:4])
        if 1900 <= yr <= 2099:
            before = cleaned[: m.start()].rstrip()
            after = cleaned[m.end() :].strip()
            if after.startswith("."):
                after = after[1:].strip()
            suffix_match = _SUFFIX_ARTIFACT_AFTER_YEAR_RE.match(after)
            year_token = m.group(1)
            if suffix_match and not year_token.rstrip().endswith(suffix_match.group(1)):
                year_token = _clean_year_token(f"{year_token}{suffix_match.group(1)}")
                after = after[suffix_match.end() :].strip()
            else:
                year_token = _clean_year_token(year_token)
            result = before
            if after:
                result += ". " + after
            return year_token, result

    # Any standalone YYYY near the end (skip URL path years)
    matches = list(re.finditer(rf"\b{_YEAR_TOKEN_PAT}\b", cleaned))
    for m in reversed(matches):
        yr = int(m.group(1)[:4])
        if 1900 <= yr <= 2099:
            pos = m.start()
            if pos > 0 and cleaned[pos - 1] == "/":
                continue
            if m.end() < len(cleaned) and cleaned[m.end()] == "/":
                continue
            trailing = cleaned[m.end() :]
            if re.match(r"^\.\d{4,5}\b", trailing):
                # Skip arXiv id fragments such as 2005.14165.
                continue
            before = cleaned[: m.start()].rstrip(", ")
            month_m = re.search(
                rf"[,.]?\s*{_MONTHS_PAT}\s*$",
                before,
                re.IGNORECASE,
            )
            if month_m:
                before = before[: month_m.start()].rstrip(", ")
            after = trailing.strip(". ")
            suffix_match = _SUFFIX_ARTIFACT_AFTER_YEAR_RE.match(after)
            year_token = m.group(1)
            if suffix_match and not year_token.rstrip().endswith(suffix_match.group(1)):
                year_token = _clean_year_token(f"{year_token}{suffix_match.group(1)}")
                after = after[suffix_match.end() :].strip()
            else:
                year_token = _clean_year_token(year_token)
            result = before
            if after:
                result += ". " + after
            return year_token, result

    return "", cleaned


def _split_title_venue(body: str) -> tuple[str, str]:
    """Split body into (title, venue)."""
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

    # Fallback split for inline metadata markers without clear sentence boundary.
    inline_m = _INLINE_METADATA_START_RE.search(body)
    if inline_m:
        title = body[: inline_m.start()].rstrip(" ,.;:")
        venue = body[inline_m.start() :].strip()
        if venue.endswith("."):
            venue = venue[:-1].rstrip()
        return (title, venue)

    title = body.rstrip()
    if title.endswith("."):
        title = title[:-1].rstrip()
    return (title, "")


class IclrReferenceParser(ReferenceParser):
    """Parser for references formatted in ICLR natbib bibliography style.

    Uses the ANTLR4 grammar defined in ``iclr.g4`` to parse author names,
    then extracts the year from the body and splits title vs. venue using
    heuristics that support article/inproceedings/book/misc/thesis/etc.

    Args:
        raise_error: When ``True``, raises :class:`ValueError` on any ANTLR4
            parse failure instead of falling back to regex extraction.
    """

    def __init__(self, raise_error: bool = False) -> None:
        self.raise_error = raise_error

    def parse(self, text: str) -> list[Reference]:
        # Iterate chunks directly so that a single unparseable reference
        # (e.g. OCR corruption, truncated author list) falls back to regex
        # extraction instead of crashing the entire file.
        refs: list[Reference] = []
        for chunk in self._prepare_chunks(text):
            try:
                ref = self.parse_one(chunk)
            except ValueError:
                ref = self._fallback_parse(chunk)
            if ref is not None:
                refs.append(ref)

        merged: list[Reference] = []
        for ref in refs:
            # Merge broken lines/chunks that are continuations of prior reference.
            if not ref.authors and not ref.title and merged:
                prev = merged[-1]
                combined_raw = prev.raw + " " + ref.raw
                try:
                    re_parsed = self.parse_one(combined_raw)
                except ValueError:
                    re_parsed = self._fallback_parse(combined_raw)
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
        if _MARKDOWN_ARTIFACT_RE.match(text):
            return None

        raw_text = text

        # Common OCR / extraction artifacts.
        text = _DAGGER_RE.sub("", text)
        text = re.sub(r",\s*,", ",", text)
        text = _SPACED_INITIAL_RE.sub(r"\1.", text)
        # Normalize BibTeX junior-part ", Jr." / ", Sr." to " Jr." / " Sr." so
        # the SUFFIX lexer token is not split by a comma acting as author separator.
        text = _NAME_SUFFIX_COMMA_RE.sub(r" \1.", text)
        text = text.strip()

        if not text:
            return None

        input_stream = InputStream(text)

        lexer = iclrLexer(input_stream)
        lexer.removeErrorListeners()
        lex_errs = _SilentErrorListener()
        lexer.addErrorListener(lex_errs)

        token_stream = CommonTokenStream(lexer)

        parser = iclrParser(token_stream)
        parser.removeErrorListeners()
        parse_errs = _SilentErrorListener()
        parser.addErrorListener(parse_errs)

        tree = parser.reference()

        if parse_errs.errors:
            if self.raise_error:
                raise ValueError(
                    f"ANTLR4 grammar (ICLR) failed to parse reference. "
                    f"Errors: {parse_errs.errors!r}. Text: {text!r}"
                )
            return self._fallback_parse(text, raw_text)

        return self._extract_reference(tree, raw_text, text)

    def _extract_reference(
        self, tree: iclrParser.ReferenceContext, raw: str, cleaned: str = ""
    ) -> Reference:
        authors_ctx = tree.authors()
        author_names: list[str] = []
        if authors_ctx is not None:
            for author_ctx in authors_ctx.authorName():
                author_names.append(_extract_iclr_author_name(author_ctx))
            if authors_ctx.ET_AL() is not None:
                author_names.append("et al.")
        author_names = [
            a for a in (_normalize_author_name(a) for a in author_names) if a
        ]
        author_names = _merge_split_single_token_authors(author_names)

        body_text = _reconstruct_body(tree.body())
        body_text = _EDITOR_TAG_RE.sub("", body_text)

        # If body begins with comma, ANTLR likely split authors too early.
        if body_text and body_text.lstrip().startswith(","):
            return self._fallback_parse(cleaned or raw, raw)

        year, body_no_year = _extract_year(body_text)
        title, venue = _split_title_venue(body_no_year)

        title = _normalize_ocr_text(title)
        venue = _normalize_ocr_text(venue)
        title = re.sub(r"^[,.\s]+", "", title).strip()

        return Reference(
            authors=author_names,
            year=year,
            title=title,
            venue=venue,
            raw=raw,
        )

    @staticmethod
    def _fallback_parse(text: str, raw: str | None = None) -> Reference:
        if raw is None:
            raw = text
        ref = Reference(raw=raw)

        m = re.search(rf"[,.\s]\s*{_YEAR_TOKEN_PAT}\s*\.\s*$", text)
        if not m:
            m = re.search(rf",\s+{_YEAR_TOKEN_PAT}\b", text)
        if not m:
            m = re.search(
                rf"[,.]\s+{_MONTHS_PAT}\s+{_YEAR_TOKEN_PAT}\b",
                text,
                re.IGNORECASE,
            )
        if not m:
            for candidate in reversed(
                list(re.finditer(rf"\b{_YEAR_TOKEN_PAT}\b", text))
            ):
                cyr = int(candidate.group(1)[:4])
                if 1900 <= cyr <= 2099:
                    p = candidate.start()
                    if p > 0 and text[p - 1] == "/":
                        continue
                    if candidate.end() < len(text) and text[candidate.end()] == "/":
                        continue
                    if re.match(r"^\.\d{4,5}\b", text[candidate.end() :]):
                        continue
                    m = candidate
                    break
        if not m:
            return ref

        yr = int(m.group(1)[:4])
        if not (1900 <= yr <= 2099):
            return ref

        ref.year = _clean_year_token(m.group(1))

        # Find author/body boundary: first ". " before year, not initial chain.
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
                    r"^[A-Z]$",
                    text[ab_m.start() - 1 : ab_m.start()],
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

        ref.authors = [a for a in (_normalize_author_name(a) for a in raw_authors) if a]
        ref.authors = _merge_split_single_token_authors(ref.authors)

        title, venue = _split_title_venue(body_part)
        ref.title = _normalize_ocr_text(title)
        ref.venue = _normalize_ocr_text(venue)
        return ref


if __name__ == "__main__":
    import glob
    import os

    target_conference_name = "iclr"
    reference_document_dir = f"output/parsed_references/docling"
    parser = IclrReferenceParser(raise_error=True)
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
