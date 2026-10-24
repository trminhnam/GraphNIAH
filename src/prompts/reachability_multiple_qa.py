"""Prompt text for scholarly-graph reachability multiple-choice questions.

Print every supported system prompt configuration with:

    python -m src.prompts.reachability_multiple_qa
"""

from __future__ import annotations

from src.prompts.reachability_yesno_qa import (
    NEEDLE_SEMANTICS,
    PAPERS_ONLY_NEEDLE_SEMANTICS,
    SUPPORTED_NEEDLE_TYPES,
    SUPPORTED_TRAVERSAL_MODES,
    TRAVERSAL_MODE_MIXED,
    is_paper_only_traversal_mode,
)


QUESTION_TYPE = "reachability-multiple"

AUTHOR_AWARE_REACHABILITY_SEMANTICS = (
    "Use only citation and authorship relationships stated in the CONTEXT. "
    "Whenever Paper A cites Paper B, there is a "
    "`cites` relationship from A to B, and a `cited_by` relationship from B to A. "
    "Whenever Author A is an author of Paper B, there is an "
    "`author_of` relationship from A to B, and an `authored_by` relationship "
    "from B to A. For each option, starting from the SOURCE paper or author, "
    "determine whether there exists a sequence of relationships that leads to "
    "the paper or author named by that option, where each relationship is either "
    "a `cites`, `cited_by`, `author_of`, or `authored_by` relationship."
)

PAPERS_ONLY_REACHABILITY_SEMANTICS = (
    "Use only citation relationships stated in the CONTEXT. Whenever Paper A "
    "cites Paper B, there is a `cites` relationship from A to B, and a "
    "`cited_by` relationship from B to A. For each option, starting from the "
    "SOURCE paper, determine whether there exists a sequence of relationships "
    "that leads to the paper named by that option, where each relationship is "
    "either a `cites` or `cited_by` relationship."
)

PAPERS_ONLY_RELATEDNESS_RULE = (
    "You must not guess the answer based on how closely related the SOURCE paper "
    "appears to be to the papers listed in the options, such as similarities in "
    "their topics, research fields, or titles. Two papers on similar topics may "
    "have no connecting sequence of relationships, while two papers on very "
    "different topics may still be connected through several intermediate papers."
)

SYSTEM_PROMPT_TEMPLATE = r"""You are a careful research assistant answering questions about academic papers using only the provided context.

You will receive a multiple-choice QUESTION and a CONTEXT containing academic paper documents.

{needle_semantics}

{reachability_semantics}

Read the QUESTION carefully to identify the SOURCE and every option. Use only the information in the provided CONTEXT. The CONTEXT contains all information available for this task. Treat entities and relationships not stated in the CONTEXT as absent.{relatedness_rule}

Your response may contain reasoning, but its final line must be exactly one answer block containing the selected uppercase option letter, for example:
\answer{{C}}

Do not write more than one answer block or put anything after the final answer line.
"""

FORMAT_REMINDER = (
    "# Response Format\n\n"
    "You may give brief reasoning first. End with exactly one final line in the "
    "form \\answer{X}, where X is the selected uppercase option letter. Use only "
    "the provided context."
)


def build_system_prompt(
    needle_type: str,
    traversal_mode: str = TRAVERSAL_MODE_MIXED,
) -> str:
    """Return the multiple-choice prompt for one descriptive needle type."""
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
    reachability_semantics = (
        PAPERS_ONLY_REACHABILITY_SEMANTICS
        if is_paper_only_traversal_mode(normalized_traversal_mode)
        else AUTHOR_AWARE_REACHABILITY_SEMANTICS
    )
    needle_semantics = (
        PAPERS_ONLY_NEEDLE_SEMANTICS[normalized_needle_type]
        if is_paper_only_traversal_mode(normalized_traversal_mode)
        else NEEDLE_SEMANTICS[normalized_needle_type]
    )
    relatedness_rule = (
        f" {PAPERS_ONLY_RELATEDNESS_RULE}"
        if is_paper_only_traversal_mode(normalized_traversal_mode)
        else ""
    )
    return SYSTEM_PROMPT_TEMPLATE.format(
        needle_semantics=needle_semantics,
        reachability_semantics=reachability_semantics,
        relatedness_rule=relatedness_rule,
    )


def main() -> None:
    """Print every supported needle-type and traversal-mode prompt."""
    cases = [
        (needle_type, traversal_mode)
        for needle_type in sorted(SUPPORTED_NEEDLE_TYPES)
        for traversal_mode in sorted(SUPPORTED_TRAVERSAL_MODES)
    ]
    for case_index, (needle_type, traversal_mode) in enumerate(cases):
        if case_index:
            print()
        print(
            f"=== needle_type={needle_type} "
            f"traversal_mode={traversal_mode} ==="
        )
        print(build_system_prompt(needle_type, traversal_mode), end="")


if __name__ == "__main__":
    main()
