"""Command to run
python -m src.reference_parser.aaai_reference_parser
"""

from __future__ import annotations

import re

from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener

from .antlr4_grammar.aaai.aaaiLexer import aaaiLexer
from .antlr4_grammar.aaai.aaaiParser import aaaiParser
from .reference_parser import Reference, ReferenceParser
from ..utils.preprocessing.preprocess_references_section import (
    _normalize_reference_item,
)

# Patterns that signal the start of venue information after the title.
_VENUE_STARTERS = re.compile(
    r"(?:"
    r"In\s|"
    r"arXiv[:\s]|"
    r"ArXiv\s|"
    r"Proceedings\s|"
    r"Ph\.\s*D\.\s|"
    r"Master'?s\s|"
    r"Technical\sReport|"
    r"(?:[A-Z][a-zA-Z&\s]*(?:Journal|Transactions|Letters|Review|Conference|Workshop|Proceedings|Advances|Annals|Bulletin|Reports|Research|Quarterly|Studies))|"
    r"(?:[A-Z][a-zA-Z]*(?:\s+[A-Za-z]+)*\s+[Pp]ress)|"
    r"Springer|Elsevier|Wiley|PMLR|IEEE|ACM|AAAI|SIAM|"
    r"Cambridge|Oxford|MIT|Princeton"
    r")"
)

_JOURNAL_LIKE = re.compile(r"^(?:" r"[A-Z][a-zA-Z]*(?:\s+[A-Za-z&]+)*" r")\s*,")


class _SilentErrorListener(ErrorListener):
    """Collects errors without printing to stderr."""

    def __init__(self) -> None:
        super().__init__()
        self.errors: list[str] = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.errors.append(f"line {line}:{column} {msg}")


def _extract_author_name(ctx: aaaiParser.AuthorNameContext) -> str:
    """Extract a human-readable author name from an authorName context."""
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


def _reconstruct_body(ctx: aaaiParser.BodyContext) -> str:
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
        # arXiv-style IDs and colon-number patterns (arXiv:2403, vol 39(1): ...)
        elif prev_tok == ":" and cur_tok[0:1].isdigit():
            space = False
        # digits.digits (e.g. 2403.13187)
        elif prev_tok[-1:].isdigit() and cur_tok == ".":
            space = False
        elif prev_tok == "." and cur_tok[0:1].isdigit():
            space = False
        # volume(number) — no space between number/word and opening paren
        elif cur_tok == "(" and prev_tok[-1:].isdigit():
            space = False
        # ordinal suffixes: 59th, 11th, 1st, 2nd, 3rd
        elif prev_tok[-1:].isdigit() and cur_tok.lower() in _ORDINAL_SUFFIXES:
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

    # Strategy 1: find the *first* venue-starter after a sentence-end marker
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

    # Strategy 2: no venue detected — entire body is the title
    title = body.rstrip()
    if title.endswith("."):
        title = title[:-1].rstrip()
    return (title, "")


class AaaiReferenceParser(ReferenceParser):
    """Parser for references formatted in AAAI bibliography style.

    Uses the ANTLR4 grammar defined in ``aaai.g4`` to structurally parse
    authors and year, then applies heuristics to separate the title from
    venue information in the remaining body text.

    Args:
        raise_error: When ``True``, a :class:`ValueError` is raised instead of
            invoking the regex fallback whenever the ANTLR4 grammar fails to
            parse a reference.  Useful for strict grammar-detection probing in
            :class:`UnifiedReferenceParser`.  Defaults to ``False`` (production
            behaviour: fall back silently to regex extraction).
    """

    def __init__(self, raise_error: bool = False) -> None:
        self.raise_error = raise_error

    def parse_one(self, text: str) -> Reference | None:
        if not text or not text.strip():
            return None

        text = _normalize_reference_item(text.strip())
        input_stream = InputStream(text)

        lexer = aaaiLexer(input_stream)
        lexer.removeErrorListeners()
        lex_errs = _SilentErrorListener()
        lexer.addErrorListener(lex_errs)

        token_stream = CommonTokenStream(lexer)

        parser = aaaiParser(token_stream)
        parser.removeErrorListeners()
        parse_errs = _SilentErrorListener()
        parser.addErrorListener(parse_errs)

        tree = parser.reference()

        if parse_errs.errors:
            if self.raise_error:
                raise ValueError(
                    f"ANTLR4 grammar (AAAI) failed to parse reference. "
                    f"Errors: {parse_errs.errors!r}. Text: {text!r}"
                )
            return self._fallback_parse(text)

        return self._extract_reference(tree, text)

    # ── tree → Reference ────────────────────────────────────────────

    def _extract_reference(
        self, tree: aaaiParser.ReferenceContext, raw: str
    ) -> Reference:
        # Authors
        authors_ctx = tree.authors()
        author_names: list[str] = []
        if authors_ctx is not None:
            for author_ctx in authors_ctx.authorName():
                author_names.append(_extract_author_name(author_ctx))
            if authors_ctx.etAl() is not None:
                author_names.append("et al.")

        # Year
        year_ctx = tree.year()
        year_str = ""
        if year_ctx is not None:
            year_str = year_ctx.NUMBER().getText() if year_ctx.NUMBER() else ""
            # Suffix may be WORD ('a' before the DOT) or INITIAL ('a.' when the
            # lowercase-initial rule consumes single letter + dot as one token).
            suffix = year_ctx.WORD()
            if suffix is not None:
                year_str += suffix.getText()
            else:
                initial_suffix = year_ctx.INITIAL()
                if initial_suffix is not None:
                    year_str += initial_suffix.getText().rstrip(".")

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

        m = re.search(r"[.?!]\s*(\d{4}[a-z]?)\.\s*", text)
        if not m:
            return ref

        ref.year = m.group(1)
        author_part = text[: m.start() + 1].rstrip(". ")
        rest = text[m.end() :]

        raw_authors = re.split(r";\s*", author_part)
        for a in raw_authors:
            a = a.strip()
            if a.lower().startswith("and "):
                a = a[4:].strip()
            if a:
                ref.authors.append(a)

        title, venue = _split_title_venue(rest)
        ref.title = title
        ref.venue = venue
        return ref


if __name__ == "__main__":
    import glob
    import os

    target_conference_name = "aaai"
    reference_document_dir = f"output/parsed_references/docling"
    parser = AaaiReferenceParser(raise_error=True)
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
