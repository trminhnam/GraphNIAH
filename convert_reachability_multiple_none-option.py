"""Add a None choice to final reachability multiple-choice prompts.

``insert-only`` preserves every question and its reachable target. ``balanced``
keeps an exactly balanced subset: every displayed answer label occurs equally
often, and None is correct for exactly one displayed-label share of the data.

For experiment 3 JSONL, use ``--pool pool.jsonl`` and mode ``append``. This
appends None after the existing choices, preserving every original label and
answer. Only multiple-choice examples are exported to the new experiment.
``--predictions model.json`` restricts conversion to examples with successful
responses from that model (including wrong answers). New content qids prevent
reuse of responses to the original prompts; ``config.source_qid`` links pairs.
Converted records are appended to the shared data pool; originals are retained.
Original token measurements are retained as ``source_prompt_measurements``;
converted token counts are not computed, and the context is never repacked.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import networkx as nx
import orjson

from src.graph.readers import DBReader
from src.prompts.reachability_multiple_qa import FORMAT_REMINDER
from src.questions.reachability_multiple import (
    ANSWER_TYPE,
    NONE_OPTION_TEXT,
    QUESTION_TYPE,
    answer_variants,
    build_question_text,
    option_labels,
)
from src.questions.text import quoted_title
from src.setup.constants import nid_db_id

CLI_NAME = "convert_reachability_multiple_option"
CONVERSION = "add-none-option"
CONVERSION_FORMAT_VERSION = 1
INSERT_ONLY_MODE = "insert-only"
BALANCED_MODE = "balanced"
CONVERSION_MODES = (INSERT_ONLY_MODE, BALANCED_MODE)
APPEND_MODE = "append"
QUESTION_HEADING = "# Question\n\n"
CONTEXT_HEADING = "# Context\n\n"
QUESTION_SEPARATOR = "\n\n# Question\n\n"
CONTEXT_SEPARATOR = "\n\n# Context\n\n"
RESPONSE_SEPARATOR = "\n\n# Response Format\n\n"
QUESTION_PREFIX = (
    "Using only the relationships described in the context, which of the following is "
    "reachable from "
)
QUESTION_SUFFIX = " by following one or more `cites` or `cited_by` relationships?"
NONE_INSTRUCTION = (
    ' If no listed target is reachable, select the "None of these targets is '
    'reachable from the source" option.'
)
OPTION_BOUNDARY_PATTERN = re.compile(r"\n([A-Z])\. ")


class InputNoneOptionError(ValueError):
    """Indicate that the source already enables a None option."""


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Insert a None choice into final reachability multiple-choice prompts."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--none-option-conversion-mode",
        choices=(*CONVERSION_MODES, APPEND_MODE),
        required=True,
    )
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument(
        "--pool",
        type=Path,
        help="Experiment JSONL mode: read original examples and append converted ones here.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Select only multiple-choice examples successfully requested in this model output.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check an existing output against a fresh deterministic conversion.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output that differs from the expected conversion.",
    )
    return parser


def load_payload(path: Path) -> dict[str, Any]:
    """Load one JSON object from ``path``."""
    try:
        payload = orjson.loads(path.read_bytes())
    except orjson.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return payload


def stable_rank(random_seed: int, namespace: str, value: str) -> bytes:
    """Return a deterministic pseudorandom ranking key."""
    encoded = f"{random_seed}\0{namespace}\0{value}".encode()
    return hashlib.sha256(encoded).digest()


def parse_prompt_sections(user_message: str) -> tuple[str, str, bool]:
    """Return ``(question, exact context, question_last)`` from a full prompt."""
    body, separator, response_tail = user_message.rpartition(RESPONSE_SEPARATOR)
    if not separator:
        raise ValueError("The user message has no final # Response Format section.")
    response_format = "# Response Format\n\n" + response_tail
    if response_format.strip() != FORMAT_REMINDER.strip():
        raise ValueError("The user message has an unexpected response format.")

    if body.startswith(QUESTION_HEADING):
        question, section_separator, context = body[len(QUESTION_HEADING) :].partition(
            CONTEXT_SEPARATOR
        )
        question_last = False
    elif body.startswith(CONTEXT_HEADING):
        context, section_separator, question = body[len(CONTEXT_HEADING) :].rpartition(
            QUESTION_SEPARATOR
        )
        question_last = True
    else:
        raise ValueError("The user message must begin with # Question or # Context.")

    if not section_separator:
        raise ValueError("The user message is missing a Question/Context boundary.")
    if not question.strip() or not context:
        raise ValueError("The user message contains an empty question or context.")
    return question.strip(), context, question_last


def render_user_message(
    question: str,
    context: str,
    *,
    question_last: bool,
) -> str:
    """Render a converted user message without changing ``context``."""
    if question_last:
        body = f"{CONTEXT_HEADING}{context}{QUESTION_SEPARATOR}{question}"
    else:
        body = f"{QUESTION_HEADING}{question}{CONTEXT_SEPARATOR}{context}"
    return f"{body}\n\n{FORMAT_REMINDER}"


def parse_source_label(question: str) -> str:
    """Return the source-paper label from a papers-only question."""
    question_head = question.split("\n\nA. ", 1)[0]
    suffix_index = question_head.find(QUESTION_SUFFIX)
    if not question_head.startswith(QUESTION_PREFIX) or suffix_index < 0:
        raise ValueError("The multiple-choice question has an unexpected format.")
    trailing_text = question_head[suffix_index + len(QUESTION_SUFFIX) :]
    if trailing_text not in {"", NONE_INSTRUCTION}:
        raise ValueError("The multiple-choice question has an unexpected instruction.")
    source_label = question_head[len(QUESTION_PREFIX) : suffix_index]
    if not source_label.startswith("Paper "):
        raise ValueError("Only papers-only reachability prompts can be converted.")
    return source_label


def parse_question_options(question: str) -> list[tuple[str, str]]:
    """Return labeled options parsed from a rendered question."""
    matches = list(OPTION_BOUNDARY_PATTERN.finditer(question))
    options: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(question)
        options.append((match.group(1), question[start:end].strip()))
    return options


def validate_prompt_payload(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate and return prompt parameters and question records."""
    parameters = payload.get("parameters")
    questions = payload.get("questions")
    metadata = payload.get("metadata")
    if not isinstance(parameters, dict):
        raise TypeError("The input payload has no parameters object.")
    if not isinstance(metadata, dict):
        raise TypeError("The input payload has no metadata object.")
    if not isinstance(questions, list) or not all(
        isinstance(question, dict) for question in questions
    ):
        raise TypeError("The input payload has no valid questions list.")
    none_option = parameters.get("none_option")
    if none_option != 0 or isinstance(none_option, bool):
        raise InputNoneOptionError(
            "Input parameters.none_option must be 0 before adding a None option; "
            f"received {none_option!r}."
        )
    if parameters.get("question_type") != QUESTION_TYPE:
        raise ValueError(f"Input parameters.question_type must be {QUESTION_TYPE!r}.")
    if parameters.get("traversal_mode") != "papers-only":
        raise ValueError("Only papers-only reachability prompts can be converted.")
    num_options = parameters.get("num_options")
    if not isinstance(num_options, int) or isinstance(num_options, bool):
        raise TypeError("Input parameters.num_options must be an integer.")
    option_labels(num_options + 1)
    return parameters, questions


def validate_source_questions_payload(
    payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return structured source records keyed by qid."""
    parameters = payload.get("parameters")
    questions = payload.get("questions")
    if not isinstance(parameters, dict) or not isinstance(questions, list):
        raise TypeError("The structured source payload is invalid.")
    none_option = parameters.get("none_option")
    if none_option != 0 or isinstance(none_option, bool):
        raise InputNoneOptionError(
            "Structured source parameters.none_option must be 0; "
            f"received {none_option!r}."
        )
    records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(questions):
        if not isinstance(record, dict):
            raise TypeError(f"Structured source question {index} is invalid.")
        qid = str(record.get("qid") or "").strip()
        if not qid or qid in records:
            raise ValueError(f"Invalid or duplicate structured source qid: {qid!r}.")
        records[qid] = record
    return records


def structured_options(
    record: dict[str, Any],
    *,
    expected_num_options: int,
) -> tuple[list[dict[str, Any]], int]:
    """Return validated source options and the correct option index."""
    multiple_choice = record.get("multiple_choice")
    if not isinstance(multiple_choice, dict):
        raise TypeError(
            f"Question {record.get('qid')!r} has no multiple_choice object."
        )
    options = multiple_choice.get("options")
    if not isinstance(options, list) or len(options) != expected_num_options:
        raise ValueError(
            f"Question {record.get('qid')!r} has an unexpected option count."
        )
    labels = option_labels(expected_num_options)
    normalized: list[dict[str, Any]] = []
    for index, (label, option) in enumerate(zip(labels, options)):
        if not isinstance(option, dict):
            raise TypeError(
                f"Question {record.get('qid')!r} option {index} is invalid."
            )
        if option.get("label") != label or not str(option.get("text") or "").strip():
            raise ValueError(
                f"Question {record.get('qid')!r} option {index} is malformed."
            )
        normalized.append(copy.deepcopy(option))
    correct_label = str(multiple_choice.get("correct_option") or "").strip().upper()
    if correct_label not in labels:
        raise ValueError(f"Question {record.get('qid')!r} has no valid correct option.")
    correct_index = labels.index(correct_label)
    positive_options = [
        index
        for index, option in enumerate(normalized)
        if option.get("search_kind") == "yes" or option.get("tree_role") == "positive"
    ]
    if positive_options != [correct_index]:
        raise ValueError(
            f"Question {record.get('qid')!r} must have one positive option."
        )
    return normalized, correct_index


def candidate_endpoint_ids(record: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Return negative-tree endpoint IDs with tree and endpoint roles."""
    trees = record.get("paper_negative_trees")
    if not isinstance(trees, list):
        return []
    candidates: list[tuple[str, str, str]] = []
    for tree in trees:
        if not isinstance(tree, dict):
            continue
        tree_id = str(tree.get("tree_id") or "")
        for endpoint in ("source", "target"):
            node_id = str(tree.get(f"{endpoint}_node_id") or "")
            if node_id:
                candidates.append((node_id, tree_id, endpoint))
    return candidates


def choose_replacement(
    record: dict[str, Any],
    context: str,
    paper_titles_by_id: Mapping[str, str],
    *,
    random_seed: int,
) -> dict[str, str] | None:
    """Choose one packed negative-tree endpoint as a replacement option."""
    qid = str(record["qid"])
    multiple_choice = record.get("multiple_choice")
    options = (
        multiple_choice.get("options") if isinstance(multiple_choice, dict) else []
    )
    existing_node_ids = {
        str(option.get("node_id") or "")
        for option in options
        if isinstance(option, dict)
    }
    candidates = sorted(
        candidate_endpoint_ids(record),
        key=lambda item: stable_rank(
            random_seed,
            f"replacement:{qid}",
            "\0".join(item),
        ),
    )
    for node_id, tree_id, endpoint in candidates:
        title = str(paper_titles_by_id.get(node_id) or "").strip()
        if not title or node_id in existing_node_ids:
            continue
        if f"Title: {title}" not in context and title not in context:
            continue
        option_text = f"Paper {quoted_title(title)}"
        if any(
            isinstance(option, dict) and option.get("text") == option_text
            for option in options
        ):
            continue
        return {
            "node_id": node_id,
            "text": option_text,
            "tree_id": tree_id,
            "endpoint": endpoint,
        }
    return None


def insert_only_assignments(
    questions: Sequence[dict[str, Any]],
    structured_by_qid: Mapping[str, dict[str, Any]],
    *,
    num_target_options: int,
    random_seed: int,
) -> dict[str, dict[str, Any]]:
    """Assign one deterministic random None insertion to every question."""
    assignments: dict[str, dict[str, Any]] = {}
    displayed_count = num_target_options + 1
    for index, question in enumerate(questions):
        qid = str(question.get("qid") or "").strip()
        source_record = structured_by_qid.get(qid)
        if source_record is None:
            raise ValueError(
                f"Prompt question {index} has no structured source record."
            )
        _options, correct_index = structured_options(
            source_record,
            expected_num_options=num_target_options,
        )
        insertion_index = (
            int.from_bytes(
                stable_rank(random_seed, "insert-only-position", qid)[:8],
                "big",
            )
            % displayed_count
        )
        correct_final_index = correct_index + int(insertion_index <= correct_index)
        assignments[qid] = {
            "correct_index": correct_final_index,
            "insertion_index": insertion_index,
            "none_is_correct": False,
            "replacement": None,
        }
    return assignments


def balanced_assignments(
    questions: Sequence[dict[str, Any]],
    structured_by_qid: Mapping[str, dict[str, Any]],
    paper_titles_by_id: Mapping[str, str],
    *,
    num_target_options: int,
    random_seed: int,
) -> dict[str, dict[str, Any]]:
    """Return an exactly label-balanced and None-balanced assignment subset."""
    displayed_count = num_target_options + 1
    retained_total = len(questions) // displayed_count * displayed_count
    if retained_total == 0:
        raise ValueError(
            f"Balanced mode needs at least {displayed_count} source questions."
        )
    per_label_quota = retained_total // displayed_count
    target_total = retained_total - per_label_quota
    sample_data: dict[str, dict[str, Any]] = {}
    ordered_qids: list[str] = []
    for index, question in enumerate(questions):
        qid = str(question.get("qid") or "").strip()
        if not qid or qid in sample_data:
            raise ValueError(
                f"Invalid or duplicate prompt qid at index {index}: {qid!r}."
            )
        source_record = structured_by_qid.get(qid)
        if source_record is None:
            raise ValueError(
                f"Prompt question {index} has no structured source record."
            )
        _options, correct_index = structured_options(
            source_record,
            expected_num_options=num_target_options,
        )
        user_message = question.get("user_message")
        if not isinstance(user_message, str):
            raise TypeError(f"Prompt question {index} has no user_message string.")
        _question_text, context, _question_last = parse_prompt_sections(user_message)
        replacement = choose_replacement(
            source_record,
            context,
            paper_titles_by_id,
            random_seed=random_seed,
        )
        sample_data[qid] = {
            "correct_index": correct_index,
            "replacement": replacement,
        }
        ordered_qids.append(qid)

    ranked_qids = sorted(
        ordered_qids,
        key=lambda qid: stable_rank(random_seed, "balanced-sample", qid),
    )
    sample_rank = {qid: index for index, qid in enumerate(ranked_qids)}
    source_node = "source"
    target_gate_node = "target-gate"
    sink_node = "sink"
    flow_graph = nx.DiGraph()
    flow_graph.add_edge(
        source_node,
        target_gate_node,
        capacity=target_total,
        weight=0,
    )
    eligibility_penalty = (len(questions) + 1) ** 3
    for qid in ranked_qids:
        correct_index = int(sample_data[qid]["correct_index"])
        replacement = sample_data[qid]["replacement"]
        selection_cost = sample_rank[qid] * (displayed_count + 1)
        if replacement is not None:
            selection_cost += eligibility_penalty
        sample_node = f"sample:{qid}"
        flow_graph.add_edge(
            target_gate_node,
            sample_node,
            capacity=1,
            weight=selection_cost,
        )
        for final_index in (correct_index, correct_index + 1):
            assignment_cost = stable_rank(
                random_seed,
                f"balanced-target-label:{qid}",
                str(final_index),
            )[0]
            flow_graph.add_edge(
                sample_node,
                f"label:{final_index}",
                capacity=1,
                weight=assignment_cost,
            )
    for label_index in range(displayed_count):
        flow_graph.add_edge(
            f"label:{label_index}",
            sink_node,
            capacity=per_label_quota,
            weight=0,
        )

    flow = nx.max_flow_min_cost(flow_graph, source_node, sink_node)
    actual_target_total = sum(
        int(flow[target_gate_node][f"sample:{qid}"]) for qid in ranked_qids
    )
    if actual_target_total != target_total:
        raise ValueError("Could not allocate an exactly balanced target-answer subset.")

    assignments: dict[str, dict[str, Any]] = {}
    target_counts = [0] * displayed_count
    for qid in ranked_qids:
        sample_node = f"sample:{qid}"
        if not flow[target_gate_node][sample_node]:
            continue
        final_indexes = [
            int(node.split(":", 1)[1])
            for node, amount in flow[sample_node].items()
            if amount
        ]
        if len(final_indexes) != 1:
            raise ValueError(f"Question {qid!r} has an invalid target assignment.")
        final_index = final_indexes[0]
        correct_index = int(sample_data[qid]["correct_index"])
        if final_index == correct_index:
            allowed_insertions = range(correct_index + 1, displayed_count)
        else:
            allowed_insertions = range(correct_index + 1)
        insertion_choices = list(allowed_insertions)
        insertion_index = insertion_choices[
            int.from_bytes(
                stable_rank(random_seed, "balanced-insertion", qid)[:8],
                "big",
            )
            % len(insertion_choices)
        ]
        assignments[qid] = {
            "correct_index": final_index,
            "insertion_index": insertion_index,
            "none_is_correct": False,
            "replacement": None,
        }
        target_counts[final_index] += 1

    none_label_slots = [
        label_index
        for label_index, target_count in enumerate(target_counts)
        for _unused in range(per_label_quota - target_count)
    ]
    none_candidates = [
        qid
        for qid in ranked_qids
        if qid not in assignments and sample_data[qid]["replacement"] is not None
    ]
    if len(none_candidates) < per_label_quota:
        raise ValueError(
            "Balanced mode cannot find enough packed negative-tree endpoints: "
            f"needed {per_label_quota}, found {len(none_candidates)}."
        )
    none_candidates.sort(
        key=lambda qid: stable_rank(random_seed, "balanced-none-sample", qid)
    )
    none_label_slots.sort(
        key=lambda label_index: stable_rank(
            random_seed,
            "balanced-none-label",
            str(label_index),
        )
    )
    for qid, label_index in zip(
        none_candidates[:per_label_quota],
        none_label_slots,
    ):
        assignments[qid] = {
            "correct_index": label_index,
            "insertion_index": label_index,
            "none_is_correct": True,
            "replacement": sample_data[qid]["replacement"],
        }
    return assignments


def convert_record(
    prompt_record: dict[str, Any],
    source_record: dict[str, Any],
    assignment: dict[str, Any],
    *,
    num_target_options: int,
    conversion_mode: str,
) -> dict[str, Any]:
    """Convert one final multiple-choice prompt record."""
    qid = str(prompt_record["qid"])
    user_message = prompt_record.get("user_message")
    if not isinstance(user_message, str):
        raise TypeError(f"Question {qid!r} has no user_message string.")
    original_question, context, question_last = parse_prompt_sections(user_message)
    if source_record.get("question") != original_question:
        raise ValueError(f"Question {qid!r} differs from its structured source.")
    source_label = parse_source_label(original_question)
    options, original_correct_index = structured_options(
        source_record,
        expected_num_options=num_target_options,
    )
    replacement = assignment["replacement"]
    if assignment["none_is_correct"]:
        if not isinstance(replacement, dict):
            raise ValueError(f"Question {qid!r} has no replacement endpoint.")
        options[original_correct_index] = {
            "node_id": replacement["node_id"],
            "text": replacement["text"],
            "search_kind": "no",
            "tree_role": "packed-distractor-negative-tree",
        }
    options.insert(
        int(assignment["insertion_index"]),
        {
            "node_id": None,
            "text": NONE_OPTION_TEXT,
            "search_kind": "none",
            "tree_role": "none-option",
        },
    )
    labels = option_labels(num_target_options + 1)
    rendered_options: list[tuple[str, str]] = []
    for label, option in zip(labels, options):
        option["label"] = label
        rendered_options.append((label, str(option["text"])))

    converted_question = build_question_text(
        source_label=source_label,
        options=rendered_options,
        include_authorship=False,
    )
    converted_message = render_user_message(
        converted_question,
        context,
        question_last=question_last,
    )
    correct_index = int(assignment["correct_index"])
    correct_label = labels[correct_index]
    correct_text = str(options[correct_index]["text"])
    converted = copy.deepcopy(prompt_record)
    converted.update(
        {
            "answer": answer_variants(correct_label, correct_text, [correct_text]),
            "answer_type": ANSWER_TYPE,
            "question_type": QUESTION_TYPE,
            "user_message": converted_message,
            "none_option_conversion": {
                "mode": conversion_mode,
                "source_correct_option": option_labels(num_target_options)[
                    original_correct_index
                ],
                "correct_option": correct_label,
                "none_option": labels[int(assignment["insertion_index"])],
                "none_is_correct": bool(assignment["none_is_correct"]),
                **(
                    {
                        "replacement_node_id": replacement["node_id"],
                        "replacement_tree_id": replacement["tree_id"],
                        "replacement_endpoint": replacement["endpoint"],
                    }
                    if isinstance(replacement, dict)
                    else {}
                ),
            },
        }
    )
    converted.pop("prompt_without_distractors_tokens", None)
    return converted


def update_prompt_fill(
    metadata: dict[str, Any],
    questions: Sequence[dict[str, Any]],
) -> None:
    """Filter prompt-fill rows and refresh changed character counts."""
    prompt_fill = metadata.get("prompt_fill")
    if not isinstance(prompt_fill, dict):
        return
    prompt_chars_by_qid = {
        str(question["qid"]): len(str(question["user_message"]))
        for question in questions
    }
    per_question = prompt_fill.get("per_question")
    if isinstance(per_question, list):
        filtered = [
            entry
            for entry in per_question
            if isinstance(entry, dict)
            and str(entry.get("qid") or "") in prompt_chars_by_qid
        ]
        for entry in filtered:
            qid = str(entry["qid"])
            entry["prompt_chars"] = prompt_chars_by_qid[qid]
        prompt_fill["per_question"] = filtered
    chars = sorted(prompt_chars_by_qid.values())
    prompt_fill["total_prompts"] = len(chars)
    if chars:
        median_chars = chars[len(chars) // 2]
        prompt_fill.update(
            {
                "median_prompt_chars": median_chars,
                "min_prompt_chars": chars[0],
                "max_prompt_chars": chars[-1],
                "min_fill_vs_median": (
                    round(chars[0] / median_chars, 4) if median_chars else None
                ),
            }
        )


def validate_conversion_invariants(
    source_payload: dict[str, Any],
    source_questions_payload: dict[str, Any],
    converted: dict[str, Any],
    *,
    conversion_mode: str,
) -> None:
    """Validate contexts, options, answers, and balancing guarantees."""
    source_parameters, source_questions = validate_prompt_payload(source_payload)
    structured_by_qid = validate_source_questions_payload(source_questions_payload)
    converted_questions = converted.get("questions")
    if not isinstance(converted_questions, list):
        raise TypeError("Converted payload has no questions list.")
    source_by_qid = {str(record["qid"]): record for record in source_questions}
    num_target_options = int(source_parameters["num_options"])
    labels = option_labels(num_target_options + 1)
    label_counts = {label: 0 for label in labels}
    none_correct_count = 0
    seen_qids: set[str] = set()
    for index, converted_record in enumerate(converted_questions):
        qid = str(converted_record.get("qid") or "")
        if qid in seen_qids or qid not in source_by_qid or qid not in structured_by_qid:
            raise ValueError(f"Converted question {index} has an invalid qid.")
        seen_qids.add(qid)
        source_record = source_by_qid[qid]
        source_question, source_context, source_question_last = parse_prompt_sections(
            str(source_record.get("user_message") or "")
        )
        question, context, question_last = parse_prompt_sections(
            str(converted_record.get("user_message") or "")
        )
        if context != source_context or question_last != source_question_last:
            raise ValueError(f"Question {qid!r} changed its context layout.")
        if parse_source_label(question) != parse_source_label(source_question):
            raise ValueError(f"Question {qid!r} changed its source paper.")
        options = parse_question_options(question)
        if [label for label, _text in options] != list(labels):
            raise ValueError(f"Question {qid!r} has invalid displayed labels.")
        none_options = [label for label, text in options if text == NONE_OPTION_TEXT]
        if len(none_options) != 1:
            raise ValueError(f"Question {qid!r} must display exactly one None option.")
        answers = converted_record.get("answer")
        if not isinstance(answers, list) or not answers:
            raise ValueError(f"Question {qid!r} has no answer variants.")
        correct_label = str(answers[0]).strip().upper()
        if correct_label not in labels:
            raise ValueError(f"Question {qid!r} has an invalid correct label.")
        correct_text = dict(options)[correct_label]
        if answers != answer_variants(correct_label, correct_text, [correct_text]):
            raise ValueError(f"Question {qid!r} has incorrect answer variants.")
        conversion = converted_record.get("none_option_conversion")
        if (
            not isinstance(conversion, dict)
            or conversion.get("mode") != conversion_mode
        ):
            raise ValueError(f"Question {qid!r} has invalid conversion metadata.")
        none_is_correct = conversion.get("none_is_correct") is True
        if none_is_correct != (correct_text == NONE_OPTION_TEXT):
            raise ValueError(f"Question {qid!r} has inconsistent None ground truth.")
        source_options, source_correct_index = structured_options(
            structured_by_qid[qid],
            expected_num_options=num_target_options,
        )
        positive_text = str(source_options[source_correct_index]["text"])
        displayed_texts = [text for _label, text in options]
        if none_is_correct and positive_text in displayed_texts:
            raise ValueError(f"Question {qid!r} retained its reachable target.")
        if not none_is_correct and positive_text not in displayed_texts:
            raise ValueError(f"Question {qid!r} lost its reachable target.")
        label_counts[correct_label] += 1
        none_correct_count += int(none_is_correct)

    if conversion_mode == INSERT_ONLY_MODE:
        if len(converted_questions) != len(source_questions) or none_correct_count:
            raise ValueError("Insert-only mode changed samples or made None correct.")
    else:
        if len(set(label_counts.values())) != 1:
            raise ValueError("Balanced mode did not balance correct answer labels.")
        expected_none = len(converted_questions) // len(labels)
        if none_correct_count != expected_none:
            raise ValueError("Balanced mode did not balance None-correct samples.")


def convert_payload(
    payload: dict[str, Any],
    source_questions_payload: dict[str, Any],
    *,
    conversion_mode: str,
    random_seed: int = 1,
    paper_titles_by_id: Mapping[str, str] | None = None,
    source_path: str | None = None,
    source_questions_path: str | None = None,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a fully converted and validated prompt payload."""
    if conversion_mode not in CONVERSION_MODES:
        raise ValueError(f"Unsupported conversion mode: {conversion_mode!r}.")
    source_parameters, source_questions = validate_prompt_payload(payload)
    structured_by_qid = validate_source_questions_payload(source_questions_payload)
    num_target_options = int(source_parameters["num_options"])
    if conversion_mode == INSERT_ONLY_MODE:
        assignments = insert_only_assignments(
            source_questions,
            structured_by_qid,
            num_target_options=num_target_options,
            random_seed=random_seed,
        )
    else:
        assignments = balanced_assignments(
            source_questions,
            structured_by_qid,
            paper_titles_by_id or {},
            num_target_options=num_target_options,
            random_seed=random_seed,
        )

    converted_questions = []
    for prompt_record in source_questions:
        qid = str(prompt_record["qid"])
        assignment = assignments.get(qid)
        if assignment is None:
            continue
        converted_questions.append(
            convert_record(
                prompt_record,
                structured_by_qid[qid],
                assignment,
                num_target_options=num_target_options,
                conversion_mode=conversion_mode,
            )
        )

    converted = copy.deepcopy(payload)
    converted["questions"] = converted_questions
    parameters = converted["parameters"]
    parameters.update(
        {
            "none_option": 1,
            "conversion": CONVERSION,
            "conversion_format_version": CONVERSION_FORMAT_VERSION,
            "none_option_conversion_mode": conversion_mode,
            "source_none_option": 0,
            "conversion_random_seed": random_seed,
        }
    )
    metadata = converted["metadata"]
    full_prompts = metadata.get("full_prompts")
    if not isinstance(full_prompts, dict):
        raise TypeError("The input metadata has no full_prompts object.")
    full_prompts["total_questions"] = len(converted_questions)
    labels = option_labels(num_target_options + 1)
    correct_option_counts = {label: 0 for label in labels}
    none_correct_count = 0
    for record in converted_questions:
        conversion = record["none_option_conversion"]
        correct_option_counts[str(conversion["correct_option"])] += 1
        none_correct_count += int(conversion["none_is_correct"] is True)
    conversion_metadata: dict[str, Any] = {
        "conversion": CONVERSION,
        "conversion_format_version": CONVERSION_FORMAT_VERSION,
        "none_option_conversion_mode": conversion_mode,
        "random_seed": random_seed,
        "source_none_option": 0,
        "target_none_option": 1,
        "num_target_options": num_target_options,
        "num_displayed_options": num_target_options + 1,
        "source_questions": len(source_questions),
        "retained_questions": len(converted_questions),
        "discarded_questions": len(source_questions) - len(converted_questions),
        "correct_option_counts": correct_option_counts,
        "none_correct_count": none_correct_count,
        "context_preservation": "byte-exact",
        "selection_policy": (
            "seeded-random-insertion"
            if conversion_mode == INSERT_ONLY_MODE
            else "seeded-exact-label-and-none-balance"
        ),
    }
    if source_path is not None:
        conversion_metadata["source_prompt_path"] = source_path
    if source_questions_path is not None:
        conversion_metadata["source_questions_path"] = source_questions_path
    if source_sha256 is not None:
        conversion_metadata["source_payload_sha256"] = source_sha256
    metadata[CLI_NAME] = conversion_metadata
    update_prompt_fill(metadata, converted_questions)
    validate_conversion_invariants(
        payload,
        source_questions_payload,
        converted,
        conversion_mode=conversion_mode,
    )
    return converted


def resolve_source_questions_path(
    payload: dict[str, Any],
    input_path: Path,
) -> Path:
    """Resolve the structured source path recorded by prompt construction."""
    metadata = payload.get("metadata")
    source_value = (
        metadata.get("source_questions") if isinstance(metadata, dict) else None
    )
    if not isinstance(source_value, str) or not source_value.strip():
        raise ValueError("Input metadata.source_questions is required.")
    source_path = Path(source_value)
    candidates = [source_path]
    if not source_path.is_absolute():
        candidates.append(input_path.parent / source_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Structured source file does not exist: {source_value}")


def load_paper_titles(
    source_questions_payload: dict[str, Any],
    prompt_questions: Sequence[dict[str, Any]],
) -> dict[str, str]:
    """Load negative-tree endpoint titles needed by balanced conversion."""
    source_by_qid = validate_source_questions_payload(source_questions_payload)
    node_ids: set[str] = set()
    for prompt_record in prompt_questions:
        source_record = source_by_qid.get(str(prompt_record.get("qid") or ""))
        if source_record is not None:
            node_ids.update(
                node_id
                for node_id, _tree, _endpoint in candidate_endpoint_ids(source_record)
            )
    if not node_ids:
        raise ValueError(
            "Balanced mode requires paper-negative-tree distractors with saved endpoints."
        )
    metadata = source_questions_payload.get("metadata")
    distracting_content = (
        metadata.get("distracting_content") if isinstance(metadata, dict) else None
    )
    db_value = (
        distracting_content.get("graph_db_path")
        if isinstance(distracting_content, dict)
        else None
    )
    if not isinstance(db_value, str) or not db_value.strip():
        raise ValueError("Structured source metadata has no graph_db_path.")
    db_path = Path(db_value)
    if not db_path.is_file():
        raise FileNotFoundError(f"Graph database does not exist: {db_path}")
    reader = DBReader(db_path)
    try:
        db_ids = [nid_db_id(node_id) for node_id in sorted(node_ids)]
        titles_by_db_id = reader.get_paper_titles(db_ids)
    finally:
        reader.close()
    return {
        node_id: titles_by_db_id.get(nid_db_id(node_id), "") for node_id in node_ids
    }


def payload_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_none_to_pool_record(record: dict[str, Any]) -> dict[str, Any]:
    """Append None to a pool example, preserving the context and every answer label."""
    from src.questions.dataset_pool import digest

    config = record["config"]
    if (
        record["question_type"] != QUESTION_TYPE
        or config["question_format"] != QUESTION_TYPE
    ):
        raise ValueError("Only multiple-choice pool examples can be converted.")
    if config.get("none_option") != 0:
        raise InputNoneOptionError(f"Example {record['qid']} already enables None.")
    question, context, question_last = parse_prompt_sections(record["user_message"])
    if question != record["question"]:
        raise ValueError(f"Example {record['qid']} has inconsistent question text.")
    parse_source_label(question)
    options = parse_question_options(question)
    labels = option_labels(config["num_options"] + 1)
    if [label for label, _text in options] != list(labels[:-1]):
        raise ValueError(f"Example {record['qid']} has unexpected option labels.")
    if (
        record["answer"] != [config["answer_condition"]]
        or record["answer"][0] not in labels[:-1]
    ):
        raise ValueError(f"Example {record['qid']} has an inconsistent correct label.")
    if any(text == NONE_OPTION_TEXT for _label, text in options):
        raise InputNoneOptionError(f"Example {record['qid']} already displays None.")

    converted = copy.deepcopy(record)
    converted["source_qid"] = record["qid"]
    conversion = {"conversion": CONVERSION, "mode": APPEND_MODE, "version": 1}
    converted["compatibility"] = {**record["compatibility"], "conversion": conversion}
    converted["config"].update(none_option=1, source_qid=record["qid"])
    converted["planned_config"] = {
        **record.get("planned_config", config),
        "none_option": 1,
        "source_qid": record["qid"],
    }
    converted["question"] = f"{question}\n{labels[-1]}. {NONE_OPTION_TEXT}"
    converted["user_message"] = render_user_message(
        converted["question"], context, question_last=question_last
    )
    converted["reachability"]["options"].append(
        {"label": labels[-1], "node_id": None, "text": NONE_OPTION_TEXT}
    )
    converted["none_option_conversion"] = {
        **conversion,
        "source_qid": record["qid"],
        "correct_option": record["answer"][0],
        "none_option": labels[-1],
        "none_is_correct": False,
        "context_preservation": "byte-exact",
    }
    # Adding an option changes prompt length. Keep the original measurements
    # explicitly as source data, rather than presenting them as converted counts.
    converted["source_prompt_measurements"] = {
        key: converted.pop(key)
        for key in ("token_counts", "packing", "prompt_without_distractors_tokens")
        if key in converted
    }
    converted["qid"] = digest(
        [
            converted["compatibility"],
            converted["planned_config"],
            converted["system_message"],
            converted["user_message"],
            converted["answer"],
        ]
    )
    return converted


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON payload atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = orjson.dumps(
        payload,
        option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE,
    )
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(data)
    temporary_path.replace(path)


def run() -> int:
    """Run conversion or check an existing deterministic conversion."""
    args = build_parser().parse_args()
    input_path = args.input
    output_path = args.output
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--input and --output must be different files.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if args.pool is not None:
        if args.none_option_conversion_mode != APPEND_MODE:
            raise ValueError("Experiment pool input requires conversion mode append.")
        from src.questions.convert_pool_experiment import convert_pool_experiment

        count = convert_pool_experiment(
            input_path,
            args.pool,
            output_path,
            append_none_to_pool_record,
            predictions_path=args.predictions,
            check_only=args.check_only,
            overwrite=args.overwrite,
        )
        print(
            f"{'Checked' if args.check_only else 'Converted'} {count} multiple-choice examples: {output_path}"
        )
        return 0
    if args.predictions is not None or args.none_option_conversion_mode == APPEND_MODE:
        raise ValueError(
            "--predictions and append mode require --pool and an experiment JSONL --input."
        )
    source = load_payload(input_path)
    try:
        _parameters, prompt_questions = validate_prompt_payload(source)
    except InputNoneOptionError as exc:
        print(f"[skip] {input_path}: {exc}", file=sys.stderr)
        return 1
    source_questions_path = resolve_source_questions_path(source, input_path)
    source_questions_payload = load_payload(source_questions_path)
    try:
        validate_source_questions_payload(source_questions_payload)
    except InputNoneOptionError as exc:
        print(f"[skip] {source_questions_path}: {exc}", file=sys.stderr)
        return 1
    paper_titles_by_id = (
        load_paper_titles(source_questions_payload, prompt_questions)
        if args.none_option_conversion_mode == BALANCED_MODE
        else {}
    )
    converted = convert_payload(
        source,
        source_questions_payload,
        conversion_mode=args.none_option_conversion_mode,
        random_seed=args.random_seed,
        paper_titles_by_id=paper_titles_by_id,
        source_path=str(input_path),
        source_questions_path=str(source_questions_path),
        source_sha256=payload_sha256(input_path),
    )
    if args.check_only:
        if not output_path.is_file():
            raise FileNotFoundError(f"Converted file does not exist: {output_path}")
        actual = load_payload(output_path)
        if actual != converted:
            raise ValueError(
                "Converted output differs from the expected deterministic payload: "
                f"{output_path}"
            )
        print(f"Conversion check passed: {output_path}")
        return 0
    if output_path.exists() and not args.overwrite:
        actual = load_payload(output_path)
        if actual == converted:
            print(f"Converted output is already up to date: {output_path}")
            return 0
        raise FileExistsError(
            f"Output already exists and differs: {output_path}. Use --overwrite."
        )
    write_payload(output_path, converted)
    print(
        f"Converted {len(converted['questions'])} question(s) in "
        f"{args.none_option_conversion_mode} mode: {input_path} -> {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
