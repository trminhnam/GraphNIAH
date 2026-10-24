from __future__ import annotations

import orjson
from typing import Any

REFERENCE_SHARED_RULES = (
    "## Extraction Rules\n\n"
    "### title\n"
    "- The full title of the referenced work, exactly as written.\n"
    '- Do not abbreviate or paraphrase. If no title can be identified, set to "".\n\n'
    "### authors\n"
    "- An ordered list of author name strings, preserving the sequence in which they appear.\n"
    "- Format each name as written (e.g. 'J. Smith' or 'John Smith').\n"
    "- Strip index markers or symbols attached to names (e.g. 'Smith¹' → 'Smith').\n"
    "- Do not include conjunctions such as 'and' at the start of an author name. "
    "For example, return 'Naor, A.' rather than 'and Naor, A.'.\n"
    "- If no authors are identifiable, return an empty list [].\n\n"
    "### year\n"
    "- The four-digit publication year (e.g. '2023').\n"
    "- Include a trailing letter if present in the reference (e.g. '2023a').\n"
    '- If the year cannot be determined, set to "".\n\n'
    "### venue\n"
    "- The publication venue: journal name, conference name, workshop, or 'arXiv preprint'.\n"
    "- Use the name as it appears in the reference; do not expand abbreviations unless "
    "the full name is also present.\n"
    '- If no venue can be identified, set to "".\n\n'
    "## Page Breaks and Page Number Lines\n"
    "PDF-to-text extraction often inserts artefacts mid-reference-section:\n"
    "  • Bare page numbers:  a lone integer on its own line (e.g. '12', '– 12 –', 'Page 12').\n"
    "  • Page headers/footers: running titles, journal names, author names repeated verbatim "
    "    from the paper header, or strings like 'REFERENCES (continued)'.\n"
    "Rules:\n"
    "1. IGNORE these artefact lines entirely — do not treat them as author names, years, "
    "   titles, or standalone reference entries.\n"
    "2. If a reference entry is SPLIT by such an artefact (i.e. the entry begins before the "
    "   artefact line and continues after it), MERGE the two fragments into a single entry.\n"
    "   Clue: the fragment after the artefact does not start with a reference-list marker "
    "   (e.g. a number in brackets '[N]', a bare number, or an author surname) and instead "
    "   continues with a lowercase word, a volume/pages indicator, a URL, a DOI, or "
    "   punctuation that makes no sense as the start of a new reference.\n"
    "3. Extract fields only from the merged bibliographic text; ignore the artefact "
    "   line itself.\n\n"
    "## Non-Reference Content Inside the References Section\n"
    "Academic PDFs sometimes embed non-bibliographic text inside the references section:\n"
    "  • Figure captions:  lines starting with 'Figure N', 'Fig. N', 'Fig N' (N = number).\n"
    "  • Table captions:   lines starting with 'Table N', 'Tab. N', 'Tab N'.\n"
    "  • Footnote text:    lines starting with a superscript digit or '* ', '† ', '‡ '.\n"
    "  • Section headings: e.g. 'A. Appendix', 'Supplementary Material'.\n"
    "Rules:\n"
    "1. SKIP these blocks entirely — do not parse them as references or use them "
    "   as part of an adjacent reference.\n"
    "2. If a caption or footnote appears between two otherwise-normal reference entries, "
    "   treat the entries before and after as separate, unrelated references.\n\n"
    "## Handling Missing or Ambiguous Data\n"
    "- If a field cannot be determined with confidence, use an empty string or empty list "
    "as appropriate. Never hallucinate or guess values.\n\n"
    "## Output Format\n"
    "- Respond with ONLY a valid JSON object — no preamble, no explanation, "
    "no markdown code fences.\n"
)


def build_batch_reference_system_prompt(schema: dict[str, Any]) -> str:
    """Return the batch reference-extraction prompt."""
    return (
        "You are a precise information-extraction assistant specialising in academic "
        "bibliography sections.\n\n"
        "## Your Task\n"
        "Parse every individual reference entry from the provided references section "
        "and return them as a structured JSON object containing a list named 'references'. "
        "Each element in the list corresponds to exactly one reference entry found in the text. "
        "Apply all page-break merging and non-reference filtering rules below before "
        "constructing your output.\n\n"
        + REFERENCE_SHARED_RULES
        + "- The JSON must strictly conform to this schema:\n"
        + orjson.dumps(schema, option=orjson.OPT_INDENT_2).decode("utf-8")
    )



def build_single_reference_system_prompt(schema: dict[str, Any]) -> str:
    """Return the single-reference extraction prompt."""
    return (
        "You are a precise information-extraction assistant specialising in academic "
        "bibliography sections.\n\n"
        "## Your Task\n"
        "Parse the single reference entry provided and return it as a structured JSON object. "
        "The input may contain page-break artefacts or non-reference text; apply all filtering "
        "rules below before extracting fields.\n\n"
        + REFERENCE_SHARED_RULES
        + "- The JSON must strictly conform to this schema:\n"
        + orjson.dumps(schema, option=orjson.OPT_INDENT_2).decode("utf-8")
    )
