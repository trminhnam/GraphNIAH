from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.setup.pipeline_parameters import (
    NUM_HOPS_PARAMETER_FIELD,
    PARAMETERS_FIELD,
    get_parameters_from_payload,
)


COUNT_AUTHOR_PAPERS = "COUNT_AUTHOR_PAPERS"
COUNT_PAPER_CITATIONS = "COUNT_PAPER_CITATIONS"
NEEDLE_REFERENCE_STATISTICS_FIELD = "needle_reference_statistics"
NEEDLE_REFERENCE_STATISTICS_VERSION = 1
DEPTH_ZERO_REFERENCE_STATISTICS_FIELD = "depth_zero_reference_statistics"
PAPER_DISTRACTOR_REFERENCE_CAPACITY_POLICY_FIELD = (
    "paper_distractor_reference_capacity_filter"
)
PAPER_DISTRACTOR_REFERENCE_CAPACITY_POLICY = (
    "outgoing-references-at-least-ceil-mean-plus-3-std-v1"
)


def calculate_needle_reference_statistics(
    paper_node_ids: Sequence[str],
    references_by_paper: dict[str, Sequence[str]],
) -> dict[str, Any]:
    """Return population statistics for references rendered per needle paper."""
    citation_counts = [
        len(references_by_paper.get(str(paper_node_id), ()))
        for paper_node_id in paper_node_ids
    ]
    paper_count = len(citation_counts)
    if paper_count == 0:
        mean = 0.0
        std = 0.0
    else:
        mean = math.fsum(citation_counts) / paper_count
        variance = math.fsum(
            (citation_count - mean) ** 2
            for citation_count in citation_counts
        ) / paper_count
        std = math.sqrt(variance)
    return {
        "version": NEEDLE_REFERENCE_STATISTICS_VERSION,
        "paper_count": paper_count,
        "citation_counts": citation_counts,
        "mean": mean,
        "std": std,
        "std_type": "population",
    }


def needle_reference_distribution(
    question: Mapping[str, Any],
) -> tuple[float, float] | None:
    """Return the saved sample-specific reference distribution when valid."""
    statistics = question.get(DEPTH_ZERO_REFERENCE_STATISTICS_FIELD)
    if not isinstance(statistics, Mapping):
        statistics = question.get(NEEDLE_REFERENCE_STATISTICS_FIELD)
    if not isinstance(statistics, Mapping):
        return None
    mean = statistics.get("mean")
    std = statistics.get("std")
    if (
        not isinstance(mean, (int, float))
        or isinstance(mean, bool)
        or not isinstance(std, (int, float))
        or isinstance(std, bool)
    ):
        return None
    resolved_mean = float(mean)
    resolved_std = float(std)
    if (
        not math.isfinite(resolved_mean)
        or not math.isfinite(resolved_std)
        or resolved_mean < 0
        or resolved_std < 0
    ):
        return None
    return resolved_mean, resolved_std


def minimum_paper_distractor_reference_count(
    question: Mapping[str, Any],
) -> int | None:
    """Return ``ceil(mean + 3 * std)`` for distractor eligibility."""
    distribution = needle_reference_distribution(question)
    if distribution is None:
        return None
    mean, std = distribution
    minimum = mean + 3.0 * std
    if not math.isfinite(minimum):
        return None
    return math.ceil(minimum)


@dataclass(frozen=True)
class QuestionSpec:
    """Question-generation plan for one target node / answer mode."""

    answer_type: str
    answer_id: str
    answer_value: str | None = None


def add_common_cli_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the common question-generation CLI flags to *parser*."""
    parser.add_argument(
        "--paths",
        required=True,
        metavar="JSON",
        help="Path to the sampled-paths JSON file (output of sample_paths.py).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Write JSON output to this file (default: stdout).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all valid question types per path (not just one random).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        metavar="INT",
        help="Integer seed for reproducible answer-type selection.",
    )
    parser.add_argument(
        "--graph",
        default=None,
        nargs="+",
        metavar="DB_OR_DIR",
        help=(
            "Path to the citation graph DB, a graphs directory, or an output root "
            "(enables ambiguity checking). "
            "When omitted, the graph path is read from the walk file's "
            "metadata.graph_db_path."
        ),
    )
    parser.add_argument(
        "--db-path",
        default=None,
        metavar="DB",
        help=(
            "Explicit SQLite database path for ambiguity checking. Optional — "
            "auto-resolved from the graph via src.graph.resolve_db_path()."
        ),
    )
    parser.add_argument(
        "--skip-ambiguity",
        action="store_true",
        help="Disable the forward-edge ambiguity check even if a graph is available.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        metavar="N",
        help="JSON indentation width (default: 2).",
    )
    return parser


def resolve_ambiguity_backend(
    args: argparse.Namespace,
    walk_data: dict[str, Any],
    *,
    checker_factory: Callable[[Any, Any], Any],
) -> tuple[Any | None, Path | None, Path | None, str]:
    """Resolve the graph + DB handles used for ambiguity checking."""
    if getattr(args, "skip_ambiguity", False):
        return None, None, None, "skipped"

    meta = walk_data.get("metadata", {}) or {}
    graph_hint = (
        getattr(args, "graph", None)
        or meta.get("graph_db_path")
        or meta.get("db_path")
    )
    db_hint = getattr(args, "db_path", None)

    if graph_hint is None:
        warnings.warn(
            "No --graph provided and walk metadata lacks 'graph_db_path'/'db_path'; "
            "ambiguity check will be skipped.",
            UserWarning,
            stacklevel=2,
        )
        return None, None, None, "no_graph_provided"

    from src.graph.readers import CitationGraphLoader, DBReader
    from src.graph.resolution import resolve_graph_db_paths

    try:
        graph_path, db_path, db_resolution = resolve_graph_db_paths(
            graph_hint,
            db_hint,
        )
    except FileNotFoundError as exc:
        warnings.warn(
            f"Could not resolve graph/SQLite DB for ambiguity check: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return None, None, None, "graph_or_db_missing"

    graph_loader = CitationGraphLoader(graph_path)
    db_reader = DBReader(db_path)
    return checker_factory(graph_loader, db_reader), graph_path, db_path, db_resolution


def anchor_has_identity(node: dict[str, Any]) -> bool:
    """Return True when *node* has the metadata needed to name the anchor."""
    node_type = node.get("node_type")
    if node_type == "paper":
        return bool(node.get("title"))
    if node_type == "author":
        return bool(node.get("name"))
    return False


def prepare_question_path(
    path: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any] | None]] | None:
    """Extract nodes + edges from a sampled path, or return None if unusable."""
    if len(path) < 1:
        return None

    nodes = [step["node"] for step in path]
    if not anchor_has_identity(nodes[0]):
        return None

    return nodes, [step["edge_taken"] for step in path]


def get_answer_choices(
    target: dict[str, Any],
    *,
    max_author_index: int | None = None,
) -> list[tuple[str, str]]:
    """Return valid ``(answer_type, answer_value)`` pairs for *target*."""
    cap = None if max_author_index is None or max_author_index <= 0 else max_author_index

    choices: list[tuple[str, str]] = []
    if target["node_type"] == "paper":
        title = target.get("title", "")
        if title:
            choices.append(("TITLE", title))
        for index, author in enumerate(target.get("authors") or [], start=1):
            if cap is not None and index > cap:
                break
            if author:
                choices.append((f"AUTHOR_{index}", author))
    elif target["node_type"] == "author":
        name = target.get("name", "")
        if name:
            choices.append(("AUTHOR_NAME", name))

    return choices


def get_question_specs(
    target: dict[str, Any],
    *,
    max_author_index: int | None = None,
) -> list[QuestionSpec]:
    """Return every supported question spec for *target*."""
    answer_id = target["node_id"]
    specs = [
        QuestionSpec(answer_type=answer_type, answer_id=answer_id, answer_value=value)
        for answer_type, value in get_answer_choices(
            target,
            max_author_index=max_author_index,
        )
    ]

    if target["node_type"] == "paper":
        specs.append(QuestionSpec(answer_type=COUNT_PAPER_CITATIONS, answer_id=answer_id))
    elif target["node_type"] == "author":
        specs.append(QuestionSpec(answer_type=COUNT_AUTHOR_PAPERS, answer_id=answer_id))

    return specs


def get_zero_hop_question_specs(target: dict[str, Any]) -> list[QuestionSpec]:
    """Return the allowed question specs for a single-node path."""
    if target.get("node_type") != "paper":
        return []

    answer_id = target["node_id"]
    title = str(target.get("title") or "").strip()
    authors = [str(author) for author in target.get("authors") or [] if author]
    specs: list[QuestionSpec] = []

    if title:
        specs.extend(
            QuestionSpec(
                answer_type=f"AUTHOR_{index}",
                answer_id=answer_id,
                answer_value=author,
            )
            for index, author in enumerate(authors, start=1)
        )

    if len(authors) >= 2:
        specs.append(
            QuestionSpec(
                answer_type="TITLE",
                answer_id=answer_id,
                answer_value=title or None,
            )
        )

    specs.append(QuestionSpec(answer_type=COUNT_PAPER_CITATIONS, answer_id=answer_id))

    return specs


def get_path_question_specs(nodes: list[dict[str, Any]]) -> list[QuestionSpec]:
    """Return the supported question specs for one prepared path."""
    if not nodes:
        return []
    if len(nodes) == 1:
        return get_zero_hop_question_specs(nodes[0])
    return get_question_specs(nodes[-1])


def build_question_record(
    nodes: list[dict[str, Any]],
    *,
    question: str,
    answer_type: str,
    answer_id: str,
    footprint: dict[str, Any],
) -> dict[str, Any]:
    """Package a built question with the shared metadata fields."""
    return {
        "question": question,
        "answer": None,
        "answer_type": answer_type,
        "answer_id": answer_id,
        "node_ids": [node["node_id"] for node in nodes],
        "footprint": footprint,
    }


def build_record_for_spec(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any] | None],
    *,
    spec: QuestionSpec,
    checker: Any,
    builder: Callable[..., tuple[str, dict[str, Any]] | None],
    builder_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build one question record for one question spec."""
    kwargs = builder_kwargs or {}
    built = builder(
        nodes,
        edges,
        spec.answer_type,
        spec.answer_value,
        checker=checker,
        **kwargs,
    )
    if built is None:
        return None

    question, footprint = built
    return build_question_record(
        nodes,
        question=question,
        answer_type=spec.answer_type,
        answer_id=spec.answer_id,
        footprint=footprint,
    )


def build_records_for_specs(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any] | None],
    *,
    specs: list[QuestionSpec],
    checker: Any,
    builder: Callable[..., tuple[str, dict[str, Any]] | None],
    builder_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build one question record for every question spec."""
    records: list[dict[str, Any]] = []
    for spec in specs:
        record = build_record_for_spec(
            nodes,
            edges,
            spec=spec,
            checker=checker,
            builder=builder,
            builder_kwargs=builder_kwargs,
        )
        if record is not None:
            records.append(record)
    return records


def collect_questions_from_walk_data(
    walk_data: dict[str, Any],
    *,
    all_mode: bool,
    create_question_fn: Callable[..., dict[str, Any] | None],
    create_all_questions_fn: Callable[..., list[dict[str, Any]]],
    checker: Any,
    rng: Any,
    annotate_with_ambiguity: Callable[[dict[str, Any], list[dict[str, Any]], Any], None]
    | None,
    create_question_kwargs: dict[str, Any] | None = None,
    create_all_questions_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run question generation for every stored path in *walk_data*."""
    question_kwargs = create_question_kwargs or {}
    all_question_kwargs = create_all_questions_kwargs or {}
    all_questions: list[dict[str, Any]] = []

    for path_entry in walk_data.get("paths", []):
        steps = path_entry.get("path", [])
        path_index = path_entry.get("path_index")

        if all_mode:
            questions = create_all_questions_fn(
                steps,
                checker=checker,
                **all_question_kwargs,
            )
        else:
            question = create_question_fn(
                steps,
                rng,
                checker=checker,
                **question_kwargs,
            )
            questions = [] if question is None else [question]

        for question in questions:
            question["path_index"] = path_index
            if checker is not None and annotate_with_ambiguity is not None:
                annotate_with_ambiguity(question, steps, checker)
            all_questions.append(question)

    return all_questions


def build_generation_result(
    *,
    paths_file: Path,
    walk_data: dict[str, Any],
    questions: list[dict[str, Any]],
    graph_path: Path | None,
    db_path: Path | None,
    db_resolution: str,
    mode: str,
    random_seed: int | None,
    include_ambiguity_totals: bool,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the standard output payload for a generation run."""
    parameters = get_parameters_from_payload(walk_data)
    walk_metadata = walk_data.get("metadata", {}) or {}
    if (
        parameters.get(NUM_HOPS_PARAMETER_FIELD) is None
        and isinstance(walk_metadata, dict)
        and isinstance(walk_metadata.get("path_length"), int)
    ):
        parameters[NUM_HOPS_PARAMETER_FIELD] = max(
            int(walk_metadata["path_length"]) - 1,
            0,
        )

    result = {
        "source_paths": str(paths_file),
        PARAMETERS_FIELD: parameters,
        "total_paths": len(walk_data.get("paths", [])),
        "total_questions": len(questions),
        "mode": mode,
        "questions": questions,
    }
    if include_ambiguity_totals:
        result["total_ambiguous"] = sum(
            1 for question in questions if question.get("is_ambiguous")
        )
    if random_seed is not None:
        result["random_seed"] = random_seed
    if extra_fields:
        result.update(extra_fields)
    return result
