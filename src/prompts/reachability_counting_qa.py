"""Prompt text for single-source scholarly reachability counting."""

from __future__ import annotations

from src.prompts.reachability_yesno_qa import (
    NEEDLE_SEMANTICS,
    PAPERS_ONLY_NEEDLE_SEMANTICS,
    SUPPORTED_NEEDLE_TYPES,
    SUPPORTED_TRAVERSAL_MODES,
    TRAVERSAL_MODE_MIXED,
    is_paper_only_traversal_mode,
)


QUESTION_TYPE = "reachability-counting"

PAPERS_ONLY_REACHABILITY_SEMANTICS = (
    "Use only citation relationships stated in the CONTEXT. Whenever Paper A "
    "cites Paper B, there is a `cites` relationship from A to B, and a "
    "`cited_by` relationship from B to A. A paper is reachable from the SOURCE "
    "paper if there exists a sequence of relationships that leads from the "
    "SOURCE paper to that paper, where each relationship is either a `cites` "
    "or `cited_by` relationship."
)

AUTHOR_AWARE_REACHABILITY_SEMANTICS = (
    "Use only citation and authorship relationships stated in the CONTEXT. "
    "Whenever Paper A cites Paper B, there is a `cites` relationship from A to "
    "B, and a `cited_by` relationship from B to A. Whenever Author A is listed "
    "as an author of Paper B in that paper's complete block, there is an "
    "`author_of` relationship from A to B, and an `authored_by` relationship "
    "from B to A. A paper or author is reachable from the SOURCE paper or "
    "author if there exists a sequence of relationships that leads from the "
    "SOURCE to that paper or author, where each relationship is either a "
    "`cites`, `cited_by`, `author_of`, or `authored_by` relationship."
)

PAPER_IDENTITY_RULE = (
    "Paper mentions with the same title and the same complete set of authors "
    "denote one paper node. Count that paper only once even if it appears in "
    "multiple blocks or reference lists. Paper mentions with the same title "
    "but different author sets denote different paper nodes. Ignore "
    "superficial differences in capitalization, punctuation, author order, "
    "and repeated author names when applying this identity rule."
)

PAPERS_ONLY_COUNTING_RULES = (
    "Starting from the SOURCE paper, count every distinct paper reachable "
    "through one or more `cites` or `cited_by` relationships. A paper "
    "appearing only in a References list is still a paper node and must be "
    "counted when reachable. Do not count the SOURCE paper. "
    f"{PAPER_IDENTITY_RULE} "
    "You must not include or exclude a paper based on how closely related it "
    "appears to be to the SOURCE paper, such as similarities in their topics, "
    "research fields, or titles. A paper on an unrelated topic must be counted "
    "if it is reachable, while a paper on a closely related topic must not be "
    "counted if it is unreachable."
)

AUTHOR_AWARE_COUNTING_RULES = (
    "Starting from the SOURCE paper or author, count every distinct paper and "
    "author reachable through one or more `cites`, `cited_by`, `author_of`, "
    "or `authored_by` relationships. Count papers and authors together as one "
    "total. A paper appearing only in a References list is still a paper node "
    "and must be counted when reachable. Author names inside a reference "
    "citation help identify the referenced paper but do not by themselves "
    "introduce author nodes. Do not count the SOURCE itself. "
    f"{PAPER_IDENTITY_RULE}"
)

SYSTEM_PROMPT_TEMPLATE = r"""You are a careful research assistant answering questions about academic papers using only the provided context.

You will receive a QUESTION and a CONTEXT containing academic paper documents.

{needle_semantics}

{reachability_semantics}

Read the QUESTION carefully to identify the SOURCE. Use only the information in the provided CONTEXT. The CONTEXT contains all information available for this task. Treat entities and relationships not stated in the CONTEXT as absent.

{counting_rules}

Your response may contain reasoning, but its final line must be exactly one answer block containing a non-negative base-10 integer, for example:
\answer{{17}}

Do not use commas or other text inside the answer block. Do not write more than one answer block or put anything after the final answer line.
"""

FORMAT_REMINDER = (
    "# Response Format\n\n"
    "You may give brief reasoning first. End with exactly one final line in the "
    "form \\answer{N}, where N is a non-negative base-10 integer. Do not count "
    "the SOURCE. Use only the provided context."
)


def build_system_prompt(
    needle_type: str,
    traversal_mode: str = TRAVERSAL_MODE_MIXED,
) -> str:
    """Return the counting prompt for one needle and traversal mode."""
    normalized_needle_type = str(needle_type or "").strip()
    if normalized_needle_type not in SUPPORTED_NEEDLE_TYPES:
        supported = ", ".join(sorted(SUPPORTED_NEEDLE_TYPES))
        raise ValueError(
            f"Unsupported reachability needle type: {needle_type!r}. "
            f"Expected one of: {supported}."
        )
    normalized_traversal_mode = str(traversal_mode or "").strip()
    if normalized_traversal_mode not in SUPPORTED_TRAVERSAL_MODES:
        supported = ", ".join(sorted(SUPPORTED_TRAVERSAL_MODES))
        raise ValueError(
            f"Unsupported reachability traversal mode: {traversal_mode!r}. "
            f"Expected one of: {supported}."
        )
    papers_only = is_paper_only_traversal_mode(normalized_traversal_mode)
    return SYSTEM_PROMPT_TEMPLATE.format(
        needle_semantics=(
            PAPERS_ONLY_NEEDLE_SEMANTICS[normalized_needle_type]
            if papers_only
            else NEEDLE_SEMANTICS[normalized_needle_type]
        ),
        reachability_semantics=(
            PAPERS_ONLY_REACHABILITY_SEMANTICS
            if papers_only
            else AUTHOR_AWARE_REACHABILITY_SEMANTICS
        ),
        counting_rules=(
            PAPERS_ONLY_COUNTING_RULES
            if papers_only
            else AUTHOR_AWARE_COUNTING_RULES
        ),
    )


def main() -> None:
    """Print every supported prompt configuration."""
    for needle_type in sorted(SUPPORTED_NEEDLE_TYPES):
        for traversal_mode in sorted(SUPPORTED_TRAVERSAL_MODES):
            print(
                f"=== needle_type={needle_type} "
                f"traversal_mode={traversal_mode} ==="
            )
            print(build_system_prompt(needle_type, traversal_mode), end="")


if __name__ == "__main__":
    main()
