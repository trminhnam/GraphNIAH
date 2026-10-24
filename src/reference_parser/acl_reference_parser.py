"""Command to run
python -m src.reference_parser.acl_reference_parser
"""

from __future__ import annotations

import re

from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener

from .antlr4_grammar.acl.aclLexer import aclLexer
from .antlr4_grammar.acl.aclParser import aclParser
from .reference_parser import Reference, ReferenceParser

_VENUE_STARTERS = re.compile(
    r"(?:"
    r"In\s|"
    r"arXiv[:\s]|"
    r"ArXiv\s|"
    r"Preprint[,\s]|"
    r"Proceedings\s|"
    r"Ph\.D\.\s|"
    r"Master'?s\s|"
    r"Technical\s[Rr]eport|"
    r"Journal\s|"
    r"Transactions\s|"
    r"Advances\s|"
    r"Conference\s|"
    r"Workshop\s|"
    r"Annals\s|"
    r"Bulletin\s|"
    r"Quarterly\s|"
    r"(?:[A-Z\u00C0-\u024F][a-zA-Z\u00C0-\u024F&\s-]*(?:Journal|Transactions|Letters|Review|Conference|Workshop|Proceedings|Advances|Annals|Bulletin|Reports|Research|Quarterly|Studies))|"
    r"(?:[A-Z\u00C0-\u024F][a-zA-Z\u00C0-\u024F]*(?:\s+[A-Za-z\u00C0-\u024F]+)*\s+[Pp]ress)|"
    r"Springer|Elsevier|Wiley|PMLR|IEEE|ACM|AAAI|SIAM|"
    r"Cambridge|Oxford|MIT|Princeton|"
    r"CoRR[,\s]|"
    r"OpenAI\s"
    r")"
)

_JOURNAL_LIKE = re.compile(
    r"^(?:#\s*)?(?:[A-Z\u00C0-\u024F][a-zA-Z\u00C0-\u024F]*(?:\s+[A-Za-z\u00C0-\u024F&]+)*)\s*,"
)


class _SilentErrorListener(ErrorListener):
    """Collects errors without printing to stderr."""

    def __init__(self) -> None:
        super().__init__()
        self.errors: list[str] = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.errors.append(f"line {line}:{column} {msg}")


def _extract_acl_author_name(ctx: aclParser.AuthorNameContext) -> str:
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
        elif cur == "@" or prev == "@":
            space = False

        if space:
            parts.append(" ")
        parts.append(cur)

    return "".join(parts)


def _reconstruct_body(ctx: aclParser.BodyContext) -> str:
    """Reconstruct the body text (title + venue) from the parse tree,
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
        # abbreviation sequences: Ph.D., U.S., e.g.
        elif (
            prev_tok == "."
            and len(cur_tok) == 2
            and cur_tok[0].isupper()
            and cur_tok[1] == "."
        ):
            space = False
        # slashes: URLs (https://...), paths, and/or
        elif prev_tok == "/" or cur_tok == "/":
            space = False

        if space:
            parts.append(" ")
        parts.append(cur_tok)

    return "".join(parts)


def _split_title_venue(body: str) -> tuple[str, str]:
    """Split the body string into (title, venue).

    Heuristics (applied in order):
    1. Look for a venue-starter pattern (``In ...``, ``arXiv:``, journal name, …)
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


class AclReferenceParser(ReferenceParser):
    """Parser for references formatted in ACL natbib bibliography style.

    Uses the ANTLR4 grammar defined in ``acl.g4`` to structurally parse
    authors and year, then applies heuristics to separate the title from
    venue information in the remaining body text.

    ACL natbib uses first-name-first author ordering with commas and
    "and" as separators, unlike AAAI which uses "LastName, Initials"
    with semicolons.

    Args:
        raise_error: When ``True``, raises :class:`ValueError` on any ANTLR4
            parse failure instead of falling back to regex extraction.
    """

    def __init__(self, raise_error: bool = False) -> None:
        self.raise_error = raise_error

    def parse(self, text: str) -> list[Reference]:
        chunks = self._prepare_chunks(text)
        merged: list[Reference] = []
        i = 0

        while i < len(chunks):
            chunk = chunks[i]

            # Parse without raising so we can decide whether to merge first.
            orig_raise = self.raise_error
            self.raise_error = False
            ref = self.parse_one(chunk)
            self.raise_error = orig_raise

            is_continuation = ref is None or (not ref.authors and not ref.year)

            if is_continuation:
                # Try appending to the previous reference (backward merge).
                if merged:
                    combined = merged[-1].raw.rstrip() + " " + chunk
                    orig_raise = self.raise_error
                    self.raise_error = False
                    new_ref = self.parse_one(combined)
                    self.raise_error = orig_raise
                    if new_ref is not None and (new_ref.authors or new_ref.year):
                        merged[-1] = new_ref
                        i += 1
                        continue

                # No previous to merge into — try forward merge with next chunk.
                if i + 1 < len(chunks):
                    combined = chunk + " " + chunks[i + 1]
                    orig_raise = self.raise_error
                    self.raise_error = False
                    new_ref = self.parse_one(combined)
                    self.raise_error = orig_raise
                    if new_ref is not None and (new_ref.authors or new_ref.year):
                        merged.append(new_ref)
                        i += 2
                        continue

                # Neither merge worked.
                if self.raise_error:
                    self.parse_one(
                        chunk
                    )  # re-run with raise_error=True for error message
                elif ref is not None:
                    merged.append(ref)
                # else: silently discard unparseable fragment
            else:
                merged.append(ref)

            i += 1

        return merged

    def parse_one(self, text: str) -> Reference | None:
        if not text or not text.strip():
            return None

        text = text.strip()
        # # Normalize PDF extraction artifacts before lexing:
        # # 1. Collapse "X ." → "X." for spaced initials (e.g. "V . Couto")
        # text = re.sub(r"\b([A-Z])\s+\.\s*", r"\1. ", text)
        # # 2. Remove stray spacing acute accent (U+00B4) that PDF extractors
        # #    sometimes inject between base letter and combining diacritic.
        # text = text.replace("\u00b4", "")
        # # 3. Collapse double commas (e.g. "Author, , and Other") from OCR artifacts.
        # text = re.sub(r",\s*,", ",", text)
        input_stream = InputStream(text)

        lexer = aclLexer(input_stream)
        lexer.removeErrorListeners()
        lex_errs = _SilentErrorListener()
        lexer.addErrorListener(lex_errs)

        token_stream = CommonTokenStream(lexer)

        parser = aclParser(token_stream)
        parser.removeErrorListeners()
        parse_errs = _SilentErrorListener()
        parser.addErrorListener(parse_errs)

        tree = parser.reference()

        if parse_errs.errors:
            if self.raise_error:
                raise ValueError(
                    f"ANTLR4 grammar (ACL) failed to parse reference. "
                    f"Errors: {parse_errs.errors!r}. Text: {text!r}"
                )
            return self._fallback_parse(text)

        return self._extract_reference(tree, text)

    # ── tree → Reference ────────────────────────────────────────────

    def _extract_reference(
        self, tree: aclParser.ReferenceContext, raw: str
    ) -> Reference:
        # Authors
        authors_ctx = tree.authors()
        author_names: list[str] = []
        if authors_ctx is not None:
            for author_ctx in authors_ctx.authorName():
                author_names.append(_extract_acl_author_name(author_ctx))
            if authors_ctx.ET_AL() is not None:
                author_names.append("et al.")
            elif authors_ctx.NUMBER() is not None:
                count = authors_ctx.NUMBER().getText()
                author_names.append(f"and {count} others")

        # Year
        year_ctx = tree.year()
        year_str = ""
        if year_ctx is not None:
            numbers = year_ctx.NUMBER()
            if isinstance(numbers, list):
                year_str = "".join(n.getText() for n in numbers) if numbers else ""
            else:
                year_str = numbers.getText() if numbers else ""
            suffix = year_ctx.WORD()
            if suffix is not None:
                year_str += suffix.getText()

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
    def _fallback_parse(text: str) -> Reference:
        ref = Reference(raw=text)

        m = re.search(r"\.\s+(\d{4}[a-z]?)\.\s*", text)
        if not m:
            return ref

        ref.year = m.group(1)
        author_part = text[: m.start()].rstrip(". ")
        rest = text[m.end() :]

        others_match = re.search(
            r",\s*and\s+(\d+)\s+others\s*$", author_part, re.IGNORECASE
        )
        et_al_match = re.search(
            r",?\s*(?:and\s+)?et\s+al\.?\s*$", author_part, re.IGNORECASE
        )
        if others_match:
            author_part = author_part[: others_match.start()]
            raw_authors = [a.strip() for a in author_part.split(",") if a.strip()]
            raw_authors.append(f"and {others_match.group(1)} others")
        elif et_al_match:
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

        title, venue = _split_title_venue(rest)
        ref.title = title
        ref.venue = venue
        return ref


if __name__ == "__main__":
    import glob
    import os

    target_conference_name = "acl"
    reference_document_dir = f"output/parsed_references/docling"
    parser = AclReferenceParser(raise_error=True)
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
