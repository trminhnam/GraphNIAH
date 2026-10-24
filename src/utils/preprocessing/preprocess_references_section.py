"""
Preprocess a references section extracted from a parsed academic paper.

python -m src.utils.preprocessing.preprocess_references_section
"""

import re

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_INCOMPLETE_ENDING = re.compile(
    r"\b("
    # conjunctions / logical connectives
    r"and|or|nor|but|yet|&"
    r"|et"  # "et al." split across lines
    # articles
    r"|the|a|an"
    # prepositions (extended)
    r"|in|of|for|to|with|by|from|on|at|about|against|between"
    r"|via|as|into|over|under|through|within|without|among|per"
    # relative pronouns / subordinating conjunctions
    r"|which|that|who|whose|where|when|although|because|since|while"
    # determiners / possessives
    r"|their|its|our|this|these|his|her|any|each|all|both" r")\s*$",
    re.IGNORECASE,
)

_ARTIFACT_PREFIX = re.compile(r"^[-•]\s*|^\d{1,3}\.\s*")

_STRAY_LINE = re.compile(
    r"^\d+$"
    r"|^<!--.*-->$"
    r"|^\d+\s+https?://\S+$"
    r"|^Visit\s"
    r"|^!\[.*\]\(.*\)$"
    r"|^\*{3,}$|^-{3,}$|^_{3,}$"
    # Bare bracketed reference number without content (e.g. "[36]")
    r"|^\[\s*\d+\s*\]\s*$"
    # Bare "Retrieved from URL" fragments (optional [N] prefix)
    r"|^(?:\[\s*\d+\s*\]\s*)?Retrieved\s+"
    # Single-word running page/section headers from PDF extraction
    # (e.g. "Quantum", "References") — no spaces, digits, or punctuation
    r"|^[A-Z\u00C0-\u024F][a-zA-Z\u00C0-\u024F]+$"
    # Multi-word running headers / journal titles without digits or sentence
    # punctuation (e.g. "Computers & Security", "Neural Computing").
    # Each subsequent word must be "&" or 2+ letters to avoid matching
    # text like "Line A" which could be valid reference content.
    r"|^[A-Z][a-zA-Z\u00C0-\u024F]+(?:\s+(?:&|[a-zA-Z\u00C0-\u024F]{2,})){1,5}$"
    # Bare conjunctions left by PDF page breaks (e.g. "and" on its own line)
    r"|^(?:and|or|et)$"
    # Bare URL path fragments (optional [N] prefix)
    r"|^(?:\[\s*\d+\s*\]\s*)?[a-z][\w.-]*/[\w/.-]+$"
    # Running headers ending with article:page format without a 4-digit year
    # (e.g. "Paper Title 9:25")
    r"|^(?!.*\b\d{4}\b)(?!\[\s*\d).+\d+:\d+\s*$",
    re.IGNORECASE,
)
_RECEIVED_DATE_LINE = re.compile(
    r"^Received\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}\b",
    re.IGNORECASE,
)
# Content-based filters checked with re.search (can match anywhere in the line)
_STRAY_CONTENT = re.compile(
    # ACM journal metadata headers embedded between references
    # e.g. "Distributed Ledger Technologies: ... Publication date: March 2026."
    r"Publication\s+date\s*:"
    # Author biography lines from IEEE papers interleaved with references
    # e.g. "Lizong Zhang received the BEng degree from University of ..."
    # Also handles plural: "received the BEng and MEng degrees"
    r"|received\s+the\s+(?:\w+\s+){1,5}degrees?\b",
    re.IGNORECASE,
)

_LEADING_HYPHEN = re.compile(r"^\s*-\s*")
_SECTION_HEADER = re.compile(r"^\s*#{1,6}\s+")
_LEADING_YEAR_FRAGMENT = re.compile(r"^\d{4}\b")
_LEADING_VENUE_FRAGMENT = re.compile(r"^In\b")
_MONTH_ENDING = re.compile(
    r"\b("
    r"January|February|March|April|May|June|July|August|September|October|"
    r"November|December"
    r")\s*$",
    re.IGNORECASE,
)

# IEEE-style trailing reference numbers: "... pp. 200-212. [7]"
_TRAILING_REF_NUMBER = re.compile(r"\s*\[\s*\d+\s*\]\s*$")

# Abbreviation endings that look like sentence-ending dots but are actually
# incomplete (e.g. "pp." for pages, "vol." for volume).  Used to detect
# continuation across paragraph breaks after trailing [N] stripping.
_ABBREV_ENDING = re.compile(
    r"\b(pp|vol|no|ed|eds|pt|Proc|Trans|Conf|Int|Comput|Mach|Intell)\.\s*$",
    re.IGNORECASE,
)

# Markdown table rows (lines starting with "|") that leak into reference text
_TABLE_ROW = re.compile(r"^\|")


# ---------------------------------------------------------------------------
# Stage 0 – normalise
# ---------------------------------------------------------------------------


def _normalize_reference_item(text: str) -> str:
    """Normalize per-reference PDF/OCR artifacts before structural fixing."""
    text = text.strip()
    # Collapse "X ." -> "X." for spaced initials (e.g. "V . Couto").
    text = re.sub(r"\b([A-Z])\s+\.\s*", r"\1. ", text)
    # Remove stray spacing acute accent (U+00B4) injected by some extractors.
    text = text.replace("\u00b4", "")
    # Collapse double commas from OCR artifacts (e.g. "Author, , and Other").
    text = re.sub(r",\s*,", ",", text)
    # Normalize "et.al." → "et al." (some PDF extractors omit the space).
    text = re.sub(r"\bet\.\s*al\.", "et al.", text)
    # Normalize "; and et al." → "; et al." (redundant "and" before et al.).
    text = re.sub(r";\s*and\s+et al\.", "; et al.", text, flags=re.IGNORECASE)
    # Normalize garbled "et al." written as "e. a." (PDF extraction artifact).
    # Only apply when immediately preceding a 4-digit year to avoid false positives.
    text = re.sub(r",?\s*\be\.\s+a\.(?=\s+\d{4})", "; et al.", text)
    # Remove garbled OCR initials: runs of 4+ consecutive single-letter "X."
    # tokens (e.g. "w. c. d. t. r. . h. o. o. w. ...") are PDF extraction
    # artifacts.  Strip the garbled run and any trailing junk up to the year
    # boundary (". YYYY.") so the grammar sees a clean author name.
    text = re.sub(
        r",?\s*(?:[A-Za-z]\.\s*){4,}"  # optional comma + 4+ garbled initials
        r"[^;]*?"                        # trailing junk (non-greedy)
        r"(?=\.\s+\d{4}[a-z]?\.\s)",    # lookahead: year boundary
        "",
        text,
    )
    # Remove comma before company-name abbreviations that confuse the author parser
    # (e.g. "Gurobi Optimization, LLC." → "Gurobi Optimization LLC.").
    text = re.sub(r",\s*(LLC|Inc|Ltd|Corp|Co)\b", r" \1", text)
    # Strip "(Eds.)" / "(Ed.)" editor markers — the embedded dot breaks
    # grammar-based parsers that use "." as the author–year separator.
    text = re.sub(r"\s*\(Eds?\.\)", "", text)
    # When the last author name ends with a single-letter initial (e.g. "N. B.")
    # right before ". YEAR.", the INITIAL lexer token consumes the dot, leaving
    # no standalone DOT separator for the grammar.  Duplicate the dot so the
    # lexer gets INITIAL("B.") + DOT(".") + NUMBER("2022").
    # The negative lookbehind ensures we only match true single-letter initials,
    # not the last letter of a multi-letter word (e.g. "ACM." stays unchanged).
    # The negative lookbehind excludes letters (so multi-letter words like
    # "ACM." are unaffected) but deliberately allows digits so that names
    # containing digits (e.g. "CodeLlama-7B. 2025.") still get the fix.
    text = re.sub(
        r"(?<![A-Za-z\u00C0-\u024F])([A-Z\u00C0-\u00DE])\.\s+(\d{4}[a-z]?\.\s)",
        r"\1.. \2",
        text,
    )
    # Insert missing dot between author block and year when the ". YYYY."
    # separator is absent, e.g. "Author Name 2018. Title" → "Author Name. 2018. Title".
    # Only fires when no existing ". YYYY." boundary exists (avoids false positives).
    if not re.search(r"\.\s+\d{4}[a-z]?\.\s", text):
        text = re.sub(
            r"([a-zA-Z\u00C0-\u024F])\s+(\d{4}[a-z]?\.\s)",
            r"\1. \2",
            text,
            count=1,
        )
    # Remove dots from multi-letter abbreviations (2-4 chars) in the author
    # section that would be misinterpreted as the author–year separator dot.
    # E.g. "José Fco. Martínez" → "José Fco Martínez",
    #      "Md. Mahinur" → "Md Mahinur".
    # Only applies before the first ". YYYY." (author section boundary).
    # Single-letter initials (e.g. "A.") are unaffected (min 2 chars).
    m_year = re.search(r"\.\s+\d{4}[a-z]?\.\s+", text)
    if m_year:
        author_part = text[: m_year.start() + 1]
        rest = text[m_year.start() + 1 :]
        # Fix misplaced dot in reversed name format where the dot belongs
        # after the initial, not after the family name.  Common in Indian
        # naming conventions: "Madhini. M," → "Madhini M.,".
        author_part = re.sub(
            r"\b([A-Za-z\u00C0-\u024F]{2,})\.\s+([A-Z])\s*(?=,|\s+and\b)",
            r"\1 \2.",
            author_part,
        )
        author_part = re.sub(
            r"\b([A-Za-z\u00C0-\u024F]{2,4})\.\s+(?=[A-Z\u00C0-\u024F])",
            r"\1 ",
            author_part,
        )
        text = author_part + rest
    # Fix missing author-body separator dot in first-name-first references
    # (CVPR/ACL style) when a '?' or '!' in the title appears before any
    # non-initial sentence-ending dot — indicating the title was merged
    # with the author block during PDF extraction.
    # E.g. "..., and Michael Zeng Want to reduce labeling cost? ..."
    # → "..., and Michael Zeng. Want to reduce labeling cost? ..."
    first_qe = re.search(r"[?!]", text)
    if first_qe:
        prefix = text[: first_qe.start()]
        # No non-initial sentence-ending dot before the ?/!
        if not re.search(r"(?<![A-Z\u00C0-\u00DE])\.\s", prefix):
            m_and = re.search(
                r"(,\s+and\s+"
                r"(?:[A-Z\u00C0-\u024F][a-zA-Z\u00C0-\u024F.\x27-]+\s+){1,3}"
                r"[A-Z\u00C0-\u024F][a-zA-Z\u00C0-\u024F-]+)"
                r"(\s+[A-Z][a-z])",
                prefix,
            )
            if m_and:
                text = text[: m_and.end(1)] + "." + text[m_and.end(1) :]
    # Fix missing dot before "In" venue/editor marker when preceded by a
    # letter without sentence-ending punctuation and followed by an editor
    # list (", editor(s),") or proceedings.
    # E.g. "...can help In Marie-Francine Moens, ..., editors, ..."
    # → "...can help. In Marie-Francine Moens, ..., editors, ..."
    for m_in in re.finditer(
        r"([a-zA-Z\u00C0-\u024F])\s+(In\s+[A-Z])", text
    ):
        after_in = text[m_in.start(2) :]
        if re.search(r",\s+editors?\b", after_in, re.IGNORECASE) or re.match(
            r"In\s+Proceedings\b", after_in
        ):
            text = text[: m_in.end(1)] + "." + text[m_in.end(1) :]
            break
    return text


def normalize_references(references_text: str) -> str:
    """
    Normalise raw reference text before any structural fixing:

    * Drop blank lines (collapses \\n\\n gaps into nothing).
    * Drop section headers (e.g. ``## References``).
    * Strip leading whitespace + hyphen prefix, e.g. ``" - [2] Author"``
      → ``"[2] Author"``.
    * Strip numbered prefix, e.g. ``"15. Author"`` → ``"Author"``.
      This covers both standalone ``N.`` entries and stray page-number
      prefixes that end up at the start of a continuation line.

    Returns one reference candidate per line, each terminated by a
    single ``\\n\\n``.
    """
    normalized: list[str] = []

    for raw_line in references_text.split("\n"):
        if not raw_line.strip():
            continue
        if _SECTION_HEADER.match(raw_line):
            header_text = _SECTION_HEADER.sub("", raw_line).strip()
            # Preserve the "References" heading itself for downstream consumers.
            if re.match(r"(?i)^references\s*$", header_text):
                normalized.append(raw_line.strip() + "\n")
                continue
            # Drop all other section headers: numeric pseudo-headings from PDF
            # artifacts (e.g. "## 12:13") and appendix headers that leaked in
            # from PDF extraction (e.g. "## A. Proof of Theorem 2.3").
            continue

        # Strip leading hyphen/bullet first …
        if _LEADING_HYPHEN.match(raw_line):
            cleaned = _LEADING_HYPHEN.sub("", raw_line)
        else:
            cleaned = raw_line.strip()

        # … then strip any remaining N. prefix (covers "15. Author" style
        # entries and page-number artefacts like "18. Yoshua Bengio …").
        cleaned = _ARTIFACT_PREFIX.sub("", cleaned)

        # Strip IEEE-style trailing reference numbers "[N]" at end of line.
        # E.g. "... pp. 200-212. [7]" → "... pp. 200-212."
        cleaned = _TRAILING_REF_NUMBER.sub("", cleaned)

        # Skip markdown table rows that leaked into the reference section.
        if _TABLE_ROW.match(cleaned.strip()):
            continue

        if cleaned.strip():
            normalized.append(_normalize_reference_item(cleaned))

    return "\n\n".join(normalized) + "\n"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def preprocess_references_section(references_section: str) -> str:
    """
    Clean a parsed references section by removing stray non-reference
    content and rejoining references that were incorrectly split during
    PDF parsing (e.g. by docling).

    Four-stage pipeline:
    0. Normalise: strip headers, blank lines, leading hyphens, and N. prefixes.
    1. Remove stray lines (page numbers, HTML comments, footnote URLs).
    2. Rejoin paragraphs split by empty lines.
    3. Rejoin continuation lines within paragraphs.
    """
    text = normalize_references(references_section)
    text = _remove_stray_lines(text)
    text = _rejoin_paragraphs(text)
    text = _rejoin_lines_within_paragraphs(text)
    return text


# ---------------------------------------------------------------------------
# Stages 1-3
# ---------------------------------------------------------------------------


def _remove_stray_lines(text: str) -> str:
    lines = text.split("\n")
    return "\n".join(
        line
        for line in lines
        if not _STRAY_LINE.match(line.strip())
        and not _RECEIVED_DATE_LINE.match(line.strip())
        and not _STRAY_CONTENT.search(line)
    )


def _rejoin_paragraphs(text: str) -> str:
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if len(paragraphs) <= 1:
        return text

    merged = [paragraphs[0]]

    for para in paragraphs[1:]:
        prev = merged[-1]
        if _is_paragraph_continuation(prev, para):
            clean_para = _ARTIFACT_PREFIX.sub("", para)
            merged[-1] = _join_texts(prev, clean_para)
        else:
            merged.append(para)

    return "\n\n".join(merged)


def _rejoin_lines_within_paragraphs(text: str) -> str:
    paragraphs = text.split("\n\n")
    processed = []

    for para in paragraphs:
        if not para.strip():
            continue
        lines = para.split("\n")
        merged_lines = [lines[0]]

        for line in lines[1:]:
            if _is_line_continuation(merged_lines[-1], line):
                clean_line = _ARTIFACT_PREFIX.sub("", line.strip())
                merged_lines[-1] = _join_texts(merged_lines[-1], clean_line)
            else:
                merged_lines.append(line)

        processed.append("\n".join(merged_lines))

    return "\n\n".join(processed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _join_texts(prev: str, continuation: str) -> str:
    if prev.endswith("-") and continuation:
        if continuation[0].islower():
            return prev[:-1] + continuation
        if continuation[0].isdigit():
            return prev + continuation
    return prev.rstrip() + " " + continuation.lstrip()


def _is_paragraph_continuation(prev_text: str, next_text: str) -> bool:
    """Paragraph-level (across empty lines) continuation check."""
    if not next_text:
        return False

    # Never merge a section header with subsequent text.
    if _SECTION_HEADER.match(prev_text.strip()):
        return False

    clean_next = _ARTIFACT_PREFIX.sub("", next_text)
    if not clean_next:
        return False

    if clean_next[0].islower():
        return True

    # "(eds.)", "(pp.", "(Volume …" are always continuations
    if clean_next.startswith("(") and not prev_text.rstrip().endswith("."):
        return True

    # Bare parenthesized year fragment e.g. "(2021), 102122." is always
    # a continuation of the previous reference regardless of how it ended.
    if re.match(r"^\(\d{4}\b", clean_next):
        return True

    prev_stripped = prev_text.rstrip()

    if _INCOMPLETE_ENDING.search(prev_stripped):
        return True

    if prev_stripped[-1] in ",:;":
        return True

    if prev_stripped.endswith("-") and clean_next[0].isdigit():
        return True

    # Docling sometimes pushes the publication year onto the next paragraph
    # after a month name, e.g. "... January\n\n1989. ISSN ...".
    if _MONTH_ENDING.search(prev_stripped) and _LEADING_YEAR_FRAGMENT.match(clean_next):
        return True

    # Abbreviation endings like "pp.", "vol.", "Proc." — the dot is part of an
    # abbreviation, not a sentence boundary.  Digits or "In" following are
    # almost certainly the continuation (page numbers, venue names).
    if _ABBREV_ENDING.search(prev_stripped) and (
        clean_next[0].isdigit() or _LEADING_VENUE_FRAGMENT.match(clean_next)
    ):
        return True

    # Venue/publication details are often split into a new paragraph after the
    # title sentence, e.g. "... quaternion neural networks.\n\nIn ECCV 2020 ...".
    if prev_stripped.endswith(".") and _LEADING_VENUE_FRAGMENT.match(clean_next):
        return True

    # If the previous paragraph doesn't end with sentence-ending punctuation
    # and the next paragraph doesn't look like a new reference entry
    # (no bracketed ref number [N], no author–year ". YYYY." boundary),
    # it's likely a mid-sentence split from PDF extraction.
    # E.g. "...In Advances in Neural Information\n\nProcessing Systems 35: ..."
    if (
        prev_stripped
        and prev_stripped[-1] not in ".?!"
        and not re.match(r"^\[\s*\d+\s*\]", clean_next)
        and not re.search(r"\.\s+\d{4}[a-z]?\.\s", clean_next)
    ):
        return True

    return False


def _is_line_continuation(prev_line: str, next_line: str) -> bool:
    """Within-paragraph (single-newline) continuation check."""
    stripped = next_line.strip()
    if not stripped:
        return False

    clean_next = _ARTIFACT_PREFIX.sub("", stripped)
    if not clean_next:
        return False

    if clean_next[0].islower():
        return True

    # Bare parenthesized year fragment "(2021)," is always a continuation.
    if re.match(r"^\(\d{4}\b", clean_next):
        return True

    prev_end = prev_line.rstrip()

    if _INCOMPLETE_ENDING.search(prev_end):
        return True

    if prev_end and prev_end[-1] in ",:;":
        return True

    if prev_end.endswith("-") and clean_next[0].isdigit():
        return True

    if _MONTH_ENDING.search(prev_end) and _LEADING_YEAR_FRAGMENT.match(clean_next):
        return True

    # Abbreviation endings like "pp.", "vol." — digits or "In" following are
    # continuation (page numbers, venue names).
    if _ABBREV_ENDING.search(prev_end) and (
        clean_next[0].isdigit() or _LEADING_VENUE_FRAGMENT.match(clean_next)
    ):
        return True

    if prev_end.endswith(".") and _LEADING_VENUE_FRAGMENT.match(clean_next):
        return True

    # Mid-sentence split: previous line has no sentence-ending punctuation and
    # the next line doesn't look like a new reference entry.
    if (
        prev_end
        and prev_end[-1] not in ".?!"
        and not re.match(r"^\[\s*\d+\s*\]", clean_next)
        and not re.search(r"\.\s+\d{4}[a-z]?\.\s", clean_next)
    ):
        return True

    return False


if __name__ == "__main__":
    test_cases = [
        (
            "page-number split (case 1)",
            """- [17] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and

18. Yoshua Bengio. Generative adversarial nets. In NeurIPS , 2014.""",
        ),
        (
            "lowercase continuation (case 2)",
            """## References
            
- [48] Fei Wang, Wenxuan Zhou, James Y Huang, Nan Xu, Sheng Zhang, Hoifung Poon, and Muhao Chen. mdpo: Conditional

- preference optimization for multimodal large language models. arXiv preprint arXiv:2406.11839 , 2024. 6, 8
- [49] Jialu Wang, Yang Liu, and Xin Eric Wang. Are gender-neutral queries really gender-neutral? mitigating gender bias in image search. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021 , pages 1995-2008. Association for Computational Linguistics, 2021. 2""",
        ),
        (
            "original ICML test case",
            """## REFERENCES
            
Ge, R., Huang, F., Jin, C., and Yuan, Y. Escaping from saddle points-online stochastic gradient for tensor decomposition. In Conference on learning theory , pp. 797-842. PMLR, 2015.

Geyer, K., Kyrillidis, A., and Kalev, A. Low-rank regularization and solution uniqueness in over-parameterized

- matrix sensing. In International Conference on Artificial Intelligence and Statistics , pp. 930-940. PMLR, 2020.
- Ghorbani, B., Mei, S., Misiakiewicz, T., and Montanari, A. When do neural networks outperform kernel methods? Advances in Neural Information Processing Systems , 33: 14820-14830, 2020.""",
        ),
        (
            "Some random test case",
            """## REFERENCES
            
- [47] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 , 2023. 8
- [48] Fei Wang, Wenxuan Zhou, James Y Huang, Nan Xu, Sheng Zhang, Hoifung Poon, and Muhao Chen. mdpo: Conditional

- preference optimization for multimodal large language models. arXiv preprint arXiv:2406.11839 , 2024. 6, 8
- [49] Jialu Wang, Yang Liu, and Xin Eric Wang. Are gender-neutral queries really gender-neutral? mitigating gender bias in image search. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021 , pages 1995-2008. Association for Computational Linguistics, 2021. 2""",
        ),
        (
            "neurips6.md — numbered N. style with mixed hyphen entries",
            """## REFERENCES
            
15. M.-F. Balcan, A. Beygelzimer, and J. Langford. Agnostic active learning. Journal of Computer and System Sciences , 75(1):78-89, 2009.
16. M.-F. Balcan, S. Hanneke, and J. Wortman Vaughan. The true sample complexity of active learning. Machine Learning , 80(2-3):111-139, 2010.
17. M.-F. Balcan, A. Blum, S. Hanneke, and D. Sharma. Robustly-reliable learners under poisoning attacks. In Proceedings of the 35 th Conference on Learning Theory , 2022.
- P. Bartlett, M. I. Jordan, and J. McAuliffe. Convexity, classification, and risk bounds. Journal of the American Statistical Association , 101(473):138-156, 2006.
- P. L. Bartlett, O. Bousquet, and S. Mendelson. Local rademacher complexities. The Annals of Statistics , 33(4):1497-1537, 2005.""",
        ),
    ]

    for label, case in test_cases:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
        print("INPUT:")
        print(case)
        print("\nOUTPUT:")
        result = preprocess_references_section(case)
        print(result)
