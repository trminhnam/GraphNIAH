"""Prompt text for scholarly-graph reachability questions.

Print every supported system prompt configuration with:

    python -m src.prompts.reachability_yesno_qa
"""

from __future__ import annotations


QUESTION_TYPE = "reachability-yesno"
SUPPORTED_NEEDLE_TYPES = {
    "apa-describe",
    "exact-describe",
    "general-describe",
}
TRAVERSAL_MODE_MIXED = "mixed"
TRAVERSAL_MODE_ALTERNATING = "alternating"
TRAVERSAL_MODE_PAPERS_ONLY = "papers-only"
TRAVERSAL_MODE_PAPERS_FORWARD = "papers-forward"
TRAVERSAL_MODE_PAPERS_BACKWARD = "papers-backward"
PAPER_ONLY_TRAVERSAL_MODES = frozenset(
    {
        TRAVERSAL_MODE_PAPERS_ONLY,
        TRAVERSAL_MODE_PAPERS_FORWARD,
        TRAVERSAL_MODE_PAPERS_BACKWARD,
    }
)
SUPPORTED_TRAVERSAL_MODES = {
    TRAVERSAL_MODE_MIXED,
    TRAVERSAL_MODE_ALTERNATING,
    *PAPER_ONLY_TRAVERSAL_MODES,
}


def is_paper_only_traversal_mode(traversal_mode: str) -> bool:
    """Return whether traversal contains paper nodes and citation edges only."""
    return str(traversal_mode or "").strip() in PAPER_ONLY_TRAVERSAL_MODES

NEEDLE_SEMANTICS = {
    "apa-describe": (
        "The CONTEXT is organized into structured blocks about academic papers and "
        "their references. A block beginning with Title describes one paper. Its "
        "Authors field identifies that paper's authors. Each item in References "
        "identifies a paper cited by the paper described in that block."
    ),
    "general-describe": (
        "The CONTEXT is organized into paper-description blocks containing author "
        "and reference information. Each block describes one paper. Its author "
        "list identifies that paper's authors. Each item in its reference list "
        "identifies a paper cited by the paper described in that block."
    ),
    "exact-describe": (
        "The CONTEXT is organized into blocks of explicit statements about papers, "
        "authors, and their relationships. A statement that one paper cites or "
        "references another expresses `cites`; a statement that one paper is cited "
        "by another expresses `cited_by`. A statement that an author wrote a paper "
        "or an Authors field expresses `author_of` from the author to the paper and "
        "`authored_by` from the paper to the author."
    ),
}

PAPERS_ONLY_NEEDLE_SEMANTICS = {
    "apa-describe": (
        "The CONTEXT is organized into structured blocks about academic papers and "
        "their references. A block beginning with Title describes one paper. Each "
        "item in References identifies a paper cited by the paper described in that "
        "block. The Authors field is publication metadata only; do not follow it "
        "as a relationship."
    ),
    "general-describe": (
        "The CONTEXT is organized into paper-description blocks containing "
        "reference information. Each block describes one paper. Each item in its "
        "reference list identifies a paper cited by the paper described in that "
        "block. Author names are publication metadata only; do not follow them "
        "as relationships."
    ),
    "exact-describe": (
        "The CONTEXT is organized into blocks of explicit statements about papers "
        "and their citation relationships. A statement that one paper cites or "
        "references another expresses `cites`; a statement that one paper is cited "
        "by another expresses `cited_by`. Author names are publication metadata "
        "only; do not follow them as relationships."
    ),
}

AUTHOR_AWARE_GRAPH_SEMANTICS = (
    "Use only citation and authorship relationships stated in the CONTEXT. Whenever Paper A "
    "cites Paper B, there is a `cites` relationship from A to B, and a `cited_by` relationship from B to A. "
    "Whenever Author A is an author of Paper B, there is an `author_of` relationship from A to B, and an `authored_by` relationship from B to A. "
    "Starting from the SOURCE paper or author, determine whether there exists a sequence of relationships that leads to the TARGET paper or author, where each relationship is either a `cites`, `cited_by`, `author_of`, or `authored_by` relationship."
)

PAPERS_ONLY_REACHABILITY_SEMANTICS = (
    "Use only citation relationships stated in the CONTEXT. Whenever Paper A "
    "cites Paper B, there is a `cites` relationship from A to B, and a `cited_by` relationship from B to A. "
    "Starting from the SOURCE paper, determine whether there exists a sequence of relationships that leads to the TARGET paper, where each relationship is either a `cites` or `cited_by` relationship."
)

PAPERS_ONLY_RELATEDNESS_RULE = (
    "You must not guess the answer based on how closely related the SOURCE and "
    "TARGET papers appear to be, such as their topics, research fields, or title "
    "wording. Two papers on similar topics may have no connecting sequence of "
    "relationships, while two papers on very different topics may still be "
    "connected through several intermediate papers."
)


# Retained for the separate reachability-multiple prompt, which still uses the
# explicit graph formulation.
PAPERS_ONLY_GRAPH_SEMANTICS = (
    "Given a set of academic papers, imagine a graph in which each node is a paper. "
    "Every occurrence of the same uniquely titled paper denotes the same paper "
    "node, including occurrences that appear only in reference lists. "
    "Whenever Paper A cites Paper B, there is a directed edge from vertex A "
    "to vertex B, together with a paired directed `cited_by` edge from vertex B back "
    "to vertex A. Within this graph, a paper Y is reachable from a "
    "paper X if there exists a directed path from X to Y — that is, a sequence of "
    "directed edges that leads from X to Y. Your task is to determine whether the "
    "TARGET paper is reachable from the SOURCE paper."
)

SYSTEM_PROMPT_TEMPLATE = r"""You are a careful research assistant answering questions about academic papers using only the provided context.

You will receive a QUESTION and a CONTEXT containing academic paper documents.

{needle_semantics}

{reachability_semantics}

Read the QUESTION carefully to identify the SOURCE and TARGET. Use only the information in the provided CONTEXT. The CONTEXT contains all information available for this task. Treat entities and relationships not stated in the CONTEXT as absent.{relatedness_rule}

Your response may contain reasoning, but its final line must be exactly one of:
\answer{{YES}}
\answer{{NO}}

Do not write more than one answer block or put anything after the final answer line.
"""

FORMAT_REMINDER = (
    "# Response Format\n\n"
    "You may give brief reasoning first. End with exactly one final line: "
    "\\answer{YES} or \\answer{NO}. Use only the provided context."
)


def build_system_prompt(
    needle_type: str,
    traversal_mode: str = TRAVERSAL_MODE_MIXED,
) -> str:
    """Return the reachability prompt for one descriptive needle type."""
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
        else AUTHOR_AWARE_GRAPH_SEMANTICS
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
