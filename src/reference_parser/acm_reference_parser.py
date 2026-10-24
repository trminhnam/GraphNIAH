"""Command to run
python -m src.reference_parser.acm_reference_parser
"""

from __future__ import annotations

import re

from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener

from .antlr4_grammar.acm.acmLexer import acmLexer
from .antlr4_grammar.acm.acmParser import acmParser
from .reference_parser import Reference, ReferenceParser

# Patterns that signal the start of venue information after the title.
_VENUE_STARTERS = re.compile(
    r"(?:"
    r"In\s|"
    r"arXiv[:\s]|"
    r"ArXiv\s|"
    r"Preprint[,\s]|"
    r"Proceedings\s|"
    r"Ph\.?\s*D\.?\s|"
    r"Master'?s\s|"
    r"Technical\s[Rr]eport|"
    r"(?:[A-Z][a-zA-Z&\s]*(?:Journal|Transactions|Letters|Review|Conference|Workshop|"
    r"Proceedings|Advances|Annals|Bulletin|Reports|Research|Quarterly|Studies|"
    r"Systems|Society|Computing|Applications|Intelligence|Communications|Management))|"
    r"(?:[A-Z][a-zA-Z]*(?:\s+[A-Za-z]+)*\s+[Pp]ress)|"
    r"Springer|Elsevier|Wiley|PMLR|IEEE|ACM|AAAI|SIAM|"
    r"Cambridge|Oxford|MIT|Princeton|"
    r"Association\sfor|"
    r"International\s[A-Z]"
    r")"
)

_JOURNAL_LIKE = re.compile(r"^(?:[A-Z][a-zA-Z]*(?:(?:\s+|:\s*)[A-Za-z&]+)*)\s*,")

_REF_NUMBER_RE = re.compile(r"^\[\s*\d+\s*\]\s*")


class _SilentErrorListener(ErrorListener):
    """Collects errors without printing to stderr."""

    def __init__(self) -> None:
        super().__init__()
        self.errors: list[str] = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.errors.append(f"line {line}:{column} {msg}")


def _extract_author_name(ctx: acmParser.AuthorNameContext) -> str:
    """Extract a human-readable author name from an authorName context.

    ACM uses first-name-first ordering so all nameWords are joined in order.
    """
    tokens = [nw.getText() for nw in ctx.nameWord()]
    if not tokens:
        return ctx.getText()

    parts = [tokens[0]]
    for prev, cur in zip(tokens, tokens[1:]):
        space = True
        if cur in ("-", "\u2013", "\u2014"):
            space = False
        elif prev in ("-", "\u2013", "\u2014"):
            space = False
        elif cur in ("'", "\u2019"):
            space = False
        elif prev in ("'", "\u2018", "\u2019"):
            space = False
        elif cur == ")":
            space = False
        elif prev == "(":
            space = False
        # Initial followed by next token: "J." + "Smith" → "J. Smith" (space)
        # but initial within hyphenated name: "J." + "-" handled above

        if space:
            parts.append(" ")
        parts.append(cur)

    return "".join(parts)


def _reconstruct_body(ctx: acmParser.BodyContext) -> str:
    """Reconstruct the body text from the parse tree, preserving spacing intent."""
    if ctx is None:
        return ""

    tokens: list[str] = []
    for i in range(ctx.getChildCount()):
        tok = ctx.getChild(i).getText()
        if tok:
            tokens.append(tok)

    if not tokens:
        return ""

    _ORDINAL_SUFFIXES = {"th", "st", "nd", "rd"}

    parts = [tokens[0]]
    for prev_tok, cur_tok in zip(tokens, tokens[1:]):
        space = True

        if cur_tok in (".", ",", ";", ":", ")", "]", "?", "!", "'", "\u2019"):
            space = False
        elif prev_tok in ("(", "[", "'", "\u2018", "\u2019"):
            space = False
        elif prev_tok == "-" or cur_tok == "-":
            space = False
        elif prev_tok in ("\u2013", "\u2014") or cur_tok in ("\u2013", "\u2014"):
            space = False
        elif prev_tok.endswith(".") and len(prev_tok) == 2 and cur_tok == "-":
            space = False
        # arXiv-style: "arXiv:2403" or "vol 39(1):"
        elif prev_tok == ":" and cur_tok[:1].isdigit():
            space = False
        # digits.digits (e.g. DOI fragment "10.1145")
        elif prev_tok[-1:].isdigit() and cur_tok == ".":
            space = False
        elif prev_tok == "." and cur_tok[:1].isdigit():
            space = False
        # volume(issue): no space before "("
        elif cur_tok == "(" and prev_tok[-1:].isdigit():
            space = False
        # ordinal suffix: "11th", "1st"
        elif prev_tok[-1:].isdigit() and cur_tok.lower() in _ORDINAL_SUFFIXES:
            space = False
        elif prev_tok == "/" or cur_tok == "/":
            space = False

        if space:
            parts.append(" ")
        parts.append(cur_tok)

    return "".join(parts)


def _split_title_venue(body: str) -> tuple[str, str]:
    """Split the body string into (title, venue).

    Heuristics (applied in order):
    1. Look for a venue-starter pattern after a sentence-ending punctuation.
    2. Fall back to journal-like pattern (capitalised multi-word + comma).
    3. If nothing found, the whole body is the title.
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


class AcmReferenceParser(ReferenceParser):
    """Parser for references formatted in ACM bibliography style.

    Uses the ANTLR4 grammar defined in ``acm.g4`` to structurally parse
    authors and year, then applies heuristics to separate the title from
    venue information in the remaining body text.

    ACM format:
        [N] FirstName [Initials] LastName, ..., and FirstName LastName.
        YEAR. Title. Venue volume, issue (season year), pages. URL

    Args:
        raise_error: When ``True``, a :class:`ValueError` is raised instead of
            invoking the regex fallback whenever the ANTLR4 grammar fails to
            parse a reference.  Useful for strict grammar-detection probing.
            Defaults to ``False``.
    """

    def __init__(self, raise_error: bool = False) -> None:
        self.raise_error = raise_error

    def parse_one(self, text: str) -> Reference | None:
        if not text or not text.strip():
            return None

        text = text.strip()
        # Strip leading bracketed reference number e.g. "[1]", "[42]"
        text = _REF_NUMBER_RE.sub("", text).strip()
        if not text:
            return None

        # Strip leading dot left by a missing author name (PDF extraction
        # artifact, e.g. "[62]. 2022. Title..." → ". 2022. Title...").
        # This lets the "year DOT body" grammar alternative match.
        text = re.sub(r"^\.\s*", "", text).strip()
        if not text:
            return None

        raw_text = text

        input_stream = InputStream(text)

        lexer = acmLexer(input_stream)
        lexer.removeErrorListeners()
        lex_errs = _SilentErrorListener()
        lexer.addErrorListener(lex_errs)

        token_stream = CommonTokenStream(lexer)

        parser = acmParser(token_stream)
        parser.removeErrorListeners()
        parse_errs = _SilentErrorListener()
        parser.addErrorListener(parse_errs)

        tree = parser.reference()

        if parse_errs.errors:
            if self.raise_error:
                raise ValueError(
                    f"ANTLR4 grammar (ACM) failed to parse reference. "
                    f"Errors: {parse_errs.errors!r}. Text: {text!r}"
                )
            return self._fallback_parse(text, raw_text)

        return self._extract_reference(tree, raw_text)

    # ── tree → Reference ────────────────────────────────────────────

    def _extract_reference(
        self, tree: acmParser.ReferenceContext, raw: str
    ) -> Reference:
        # Authors
        authors_ctx = tree.authors()
        author_names: list[str] = []
        if authors_ctx is not None:
            for author_ctx in authors_ctx.authorName():
                author_names.append(_extract_author_name(author_ctx))
            if authors_ctx.ET_AL() is not None:
                author_names.append("et al.")
            elif authors_ctx.AND_OTHERS() is not None:
                author_names.append("et al.")

        # Year
        year_ctx = tree.year()
        year_str = ""
        if year_ctx is not None:
            if year_ctx.NUMBER() is not None:
                year_str = year_ctx.NUMBER().getText()
                suffix = year_ctx.WORD()
                if suffix is not None:
                    year_str += suffix.getText()
            elif year_ctx.ND() is not None:
                year_str = "n.d."
            elif year_ctx.FORTHCOMING() is not None:
                year_str = year_ctx.FORTHCOMING().getText()

        # Body → title + venue
        body_text = _reconstruct_body(tree.body())
        title, venue = _split_title_venue(body_text)

        return Reference(
            authors=author_names,
            year=year_str,
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

        # ACM year pattern: ". YYYY." right after the author block
        m = re.search(r"\.\s+(\d{4}[a-z]?)\.\s+", text)
        if not m:
            return ref

        ref.year = m.group(1)
        author_part = text[: m.start()].strip()
        rest = text[m.end() :]

        # Parse comma-separated "and"-terminated author list
        et_al_m = re.search(r",?\s*et\s+al\.?\s*$", author_part, re.IGNORECASE)
        if et_al_m:
            author_part = author_part[: et_al_m.start()]
            raw_authors = [
                a.strip() for a in re.split(r",\s*", author_part) if a.strip()
            ]
            raw_authors.append("et al.")
        elif ", and " in author_part:
            last_and = author_part.rfind(", and ")
            before = author_part[:last_and]
            after = author_part[last_and + 6 :].strip()
            raw_authors = [a.strip() for a in re.split(r",\s*", before) if a.strip()]
            if after:
                raw_authors.append(after)
        elif " and " in author_part:
            parts = author_part.rsplit(" and ", 1)
            raw_authors = [p.strip() for p in parts if p.strip()]
        else:
            raw_authors = [author_part.strip()] if author_part.strip() else []

        ref.authors = raw_authors

        title, venue = _split_title_venue(rest)
        ref.title = title
        ref.venue = venue
        return ref


if __name__ == "__main__":
    import glob
    import os

    target_conference_name = "acm"
    reference_document_dir = f"output/parsed_references/docling"
    parser = AcmReferenceParser(raise_error=True)
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
