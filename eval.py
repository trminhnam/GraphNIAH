"""
eval.py — Score inference.py predictions from the model's ``\\answer{...}`` box.

Reads:
  * --groundtruth-path : JSON file (bare list or {"questions": [...]}) with an
                         'answer' field per item — same shape accepted by
                         inference.py.
  * --prediction-path  : JSON produced by inference.py, each element a record
                         with either {"qid", "raw_answer", ...} or the legacy
                         {"index", "model_output", ...} fields.

For each prediction we:
  1. Check whether the raw model output contains any ``\\answer{...}`` box.
     If not, ``no_answer_box += 1`` and we use the last characters of the
     raw output for the next comparison stages instead:
       * ``COUNT_PAPER_CITATIONS`` / ``COUNT_AUTHOR_PAPERS`` → last 30 chars
       * every other ``answer_type`` → last 100 chars
  2. If a box is present, extract the last ``\\answer{...}`` block from the
     model output.
  3. Run BOTH sides through ``normalize_answer`` (NFKC, markdown/LaTeX
     unwrap, quote peel, prefix strip, trailing-punct strip, etc.).
  4. Then run BOTH sides through ``canonical_form`` — strip to
     ``[a-zA-Z0-9]`` and lowercase.
  5. First try exact equality on the canonical strings. This is the
     priority path and does not increment any error counter.
  6. If exact equality fails but the canonical ground truth is a substring
     of the canonical extracted answer, score the item correct. If the
     answer came from an ``\\answer{...}`` box, also increment
     ``wrong_format_in_answer_box``.
  7. Otherwise, increment ``wrong_answer``.

The normalization stages are symmetric, so they can only forgive false
negatives. The soft substring fallback is intentionally one-sided because it
is meant to catch answer boxes or raw-output tails that still contain the
right answer plus extra formatting or stray text.

For ``REACHABILITY_MULTIPLE``, evaluation extracts an option letter from an
answer block and still recovers common legacy forms such as ``C``, ``C.``,
``C)``, and ``C. <text>``. Comparison is case-insensitive, while a separate
``format_mismatch`` counter records every response whose final non-empty line
is not exactly one ``\\answer{X}`` block containing an uppercase option letter.

Writes a JSON report to --output containing overall accuracy, counts, and a
per-item breakdown so downstream analysis can slice by ``answer_type`` etc.
Reachability reports also contain task-specific per-label or per-option metrics.
"""

import argparse
import re
import unicodedata
from pathlib import Path

import orjson

from src.llm.accounting import aggregate_accounting, aggregate_retry_accounting

# ``\answer{...}`` — the inner group rejects braces so a greedy match cannot
# swallow across multiple blocks. Author names / titles never contain braces,
# so the simple non-nested pattern is sufficient for this dataset.
ANSWER_PATTERN = re.compile(r"\\answer\{([^{}]*)\}")
REACHABILITY_COUNTING_ANSWER_TYPE = "REACHABILITY_COUNTING"
REACHABILITY_COUNTING_QUESTION_TYPE = "reachability-counting"
COUNTING_ANSWER_TYPES = {
    "COUNT_PAPER_CITATIONS",
    "COUNT_AUTHOR_PAPERS",
    REACHABILITY_COUNTING_ANSWER_TYPE,
}
REACHABILITY_ANSWER_TYPE = "REACHABILITY_YESNO"
REACHABILITY_LABELS = {"yes", "no", "unknown"}
REACHABILITY_QUESTION_TYPE = "reachability-yesno"
REACHABILITY_MULTIPLE_ANSWER_TYPE = "REACHABILITY_MULTIPLE"
REACHABILITY_MULTIPLE_QUESTION_TYPE = "reachability-multiple"
DEFAULT_RAW_TAIL_FALLBACK_CHARS = 100
COUNTING_RAW_TAIL_FALLBACK_CHARS = 30
MULTIPLE_CHOICE_STRICT_PATTERN = re.compile(r"^\\answer\{[A-Z]\}$")
MULTIPLE_CHOICE_LEADING_PATTERN = re.compile(
    r"^\s*([A-Za-z])(?:\s*[.):](?:\s+.*)?|\s*)$"
)
MULTIPLE_CHOICE_PREFIX_PATTERN = re.compile(
    r"^\s*(?:option|answer)\s*[:\-]?\s*([A-Za-z])"
    r"(?:\s*[.):](?:\s+.*)?|\s*)$",
    flags=re.IGNORECASE,
)
NON_NEGATIVE_INTEGER_PATTERN = re.compile(r"^[0-9]+$")


# ─────────────────────────────────────────────────────────────────────────────
# Strange-case cleanup rules
# Each pattern targets a specific failure mode we have observed or expect:
# sentence-wrapped answers, markdown/LaTeX emphasis, quoting, trailing
# annotations, and so on. Every rule is applied symmetrically to both sides
# so it can only ever forgive cosmetic drift, never invent a match.
# ─────────────────────────────────────────────────────────────────────────────

# Conversational prefixes like "The answer is", "The seventh author is", "It is".
# Applied iteratively so stacked prefixes all come off.
_PREFIX_PATTERN = re.compile(
    r"^(?:"
    r"the\s+(?:answer|value|result|affiliation)\s+(?:to\s+(?:the\s+)?(?:question\s+)?)?is\s*[:\-]?\s*"
    r"|the\s+\w+\s+(?:author|paper|title|name)\s+is\s*[:\-]?\s*"
    r"|(?:it|this|that)\s+is\s*[:\-]?\s*"
    r"|answer\s*[:\-]\s*"
    r")",
    flags=re.IGNORECASE,
)

# Quote characters we peel off if they wrap the whole answer.
_QUOTE_CHARS = set("\"'`\u201c\u201d\u2018\u2019")  # includes curly quotes

# Trailing punctuation that is safe to strip — only on the right side so
# internal punctuation inside titles stays intact.
_TRAILING_PUNCT = ".,;:!?"

# Leading delimiters left behind after prefix stripping: ":", "-", em/en dashes.
_LEADING_DELIM = re.compile(r"^[\s:\-\u2013\u2014]+")

# Markdown-style emphasis wrapping the whole answer: **X**, *X*, __X__, _X_, `X`.
# Anchored to full-string so we don't eat internal asterisks.
_MARKDOWN_WRAPPERS = [
    re.compile(r"^\*\*(.+?)\*\*$"),
    re.compile(r"^\*(.+?)\*$"),
    re.compile(r"^__(.+?)__$"),
    re.compile(r"^_(.+?)_$"),
    re.compile(r"^`(.+?)`$"),
]

# LaTeX-style emphasis wrapping the whole answer: \textbf{X}, \textit{X},
# \emph{X}, \mathbf{X}. Models sometimes copy these straight out of source.
_LATEX_WRAPPER = re.compile(
    r"^\\(?:textbf|textit|emph|mathbf|mathrm|mathit)\{(.+)\}$"
)

# Trailing parenthetical annotation: "Wei Xiong (the seventh author)".
_TRAILING_PAREN = re.compile(r"\s*\([^()]*\)\s*$")

# Runs of whitespace inside the answer (tabs, newlines, multiple spaces).
_INNER_WHITESPACE = re.compile(r"\s+")

# Final-stage alphanumeric filter — anything that isn't an ASCII letter or
# digit is discarded before the equality check. This collapses spacing,
# hyphenation, and punctuation differences that survive ``normalize_answer``.
_ALNUM_ONLY = re.compile(r"[^a-zA-Z0-9]")


def extract_answer(text: str | None) -> str | None:
    """Return the string inside the *last* ``\\answer{...}`` block, stripped.

    We take the last match because models frequently echo the format spec
    ("I will put it in \\answer{}") during reasoning before producing the
    real final answer. The redesigned system prompt also encourages chain-
    of-thought, which makes this "last match" policy load-bearing.

    Returns ``None`` both when no block is found and when the block is
    present but empty/whitespace-only (``\\answer{}`` or ``\\answer{   }``)
    — both cases mean "no extractable answer" downstream.
    """
    if not text:
        return None
    matches = ANSWER_PATTERN.findall(text)
    if not matches:
        return None
    captured = matches[-1].strip()
    if not captured:
        return None
    return captured


def has_answer_box(text: str | None) -> bool:
    """Return ``True`` when the raw model output contains ``\\answer{...}``."""
    if not text:
        return False
    return ANSWER_PATTERN.search(text) is not None


def get_tail_fallback_answer(
    text: str | None,
    max_chars: int,
) -> str | None:
    """Return the last ``max_chars`` of raw output, stripped, or ``None``."""
    if not text:
        return None
    tail = text[-max_chars:].strip()
    if not tail:
        return None
    return tail


def normalize_answer(text: str | None) -> str:
    """Canonicalise an answer string so cosmetic drift doesn't score 0.

    Applied symmetrically to extracted prediction and ground truth, so any
    rule here can only forgive a false negative — never invent a false
    positive.

    Pipeline (order matters — peel the outside layers before the inside):
      1. Unicode NFKC (``ﬁ`` → ``fi``, full-width → ASCII, etc.).
      2. Strip leading/trailing whitespace.
      3. Unwrap markdown emphasis (``**X**``, ``*X*``, etc.) if it wraps
         the whole string.
      4. Unwrap LaTeX emphasis (``\\textbf{X}``, ``\\textit{X}``, etc.).
      5. Peel matching surrounding quotes.
      6. Iteratively strip conversational prefixes ("The answer is ...").
      7. Strip leading leftover delimiters (``:``, ``-``, em/en dash).
      8. Strip trailing parenthetical annotation (" (…)").
      9. Strip trailing punctuation (``.,;:!?``).
     10. Collapse runs of internal whitespace to a single space.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text).strip()

    # (3) Markdown emphasis — apply each wrapper at most once; loop so that,
    # e.g. "***X***" (nested) comes off in passes.
    changed = True
    while changed:
        changed = False
        for pat in _MARKDOWN_WRAPPERS:
            m = pat.match(text)
            if m:
                text = m.group(1).strip()
                changed = True
                break

    # (4) LaTeX emphasis (usually wraps once, no iteration needed).
    m = _LATEX_WRAPPER.match(text)
    if m:
        text = m.group(1).strip()

    # (5) Surrounding quotes — loop to catch doubly-quoted cases like ``"'X'"``.
    while len(text) >= 2 and text[0] in _QUOTE_CHARS and text[-1] in _QUOTE_CHARS:
        text = text[1:-1].strip()

    # (6) Conversational prefixes, iteratively.
    while True:
        new = _PREFIX_PATTERN.sub("", text, count=1)
        if new == text:
            break
        text = new.lstrip()

    # (7) Leading leftover delimiters from prefix stripping.
    text = _LEADING_DELIM.sub("", text)

    # (8) Trailing parenthetical annotation — loop for repeated trailing
    # parens like "X (alias) (affil)".
    while True:
        new = _TRAILING_PAREN.sub("", text)
        if new == text:
            break
        text = new

    # (9) Trailing punctuation (right-only so internal punctuation stays).
    text = text.rstrip(_TRAILING_PUNCT + " \t\n")

    # (10) Collapse internal whitespace.
    text = _INNER_WHITESPACE.sub(" ", text).strip()

    return text


def canonical_form(text: str | None) -> str:
    """Final comparison key: strip to ``[a-zA-Z0-9]`` and lowercase.

    Applied AFTER ``normalize_answer``. The exact-match check runs on this
    form, so differences in spacing, hyphenation, capitalisation, and
    punctuation are all discarded. E.g. ``"Wei Xiong"``, ``"Wei-Xiong"``,
    and ``"wei xiong"`` all collapse to ``"weixiong"``.

    NOTE: the filter is strict ASCII, so diacritics are removed (``í`` is
    not in ``[a-zA-Z0-9]``). If you later want "Martín" == "Martin",
    prepend ``unicodedata.normalize("NFD", text)`` so the accent detaches
    as a combining mark and drops out with the rest of the non-letters.
    """
    if text is None:
        return ""
    return _ALNUM_ONLY.sub("", text).lower()


def normalize_reachability_label(text: str | None) -> str:
    """Return a strict lowercase reachability label or an empty string."""
    label = normalize_answer(text).casefold()
    return label if label in REACHABILITY_LABELS else ""


def normalize_counting_integer(text: str | None) -> str:
    """Return one canonical non-negative integer or an empty string."""
    normalized = normalize_answer(text)
    if NON_NEGATIVE_INTEGER_PATTERN.fullmatch(normalized) is None:
        return ""
    return str(int(normalized))


def extract_multiple_choice_option(text: str | None) -> str | None:
    """Extract one option letter from supported response renderings.

    The complete response is tried first, then a legacy ``\answer{...}``
    payload, then the final non-empty line. This recovers common formatting
    drift without searching arbitrary letters inside prose.
    """
    if not text:
        return None
    normalized = unicodedata.normalize("NFKC", str(text)).strip()
    candidates = [normalized]
    boxed = extract_answer(normalized)
    if boxed is not None and boxed not in candidates:
        candidates.append(boxed)
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if lines and lines[-1] not in candidates:
        candidates.append(lines[-1])
    for candidate in candidates:
        for pattern in (
            MULTIPLE_CHOICE_LEADING_PATTERN,
            MULTIPLE_CHOICE_PREFIX_PATTERN,
        ):
            match = pattern.fullmatch(candidate)
            if match is not None:
                return match.group(1)
    return None


def normalize_multiple_choice_option(text: str | None) -> str:
    """Return a lowercase extracted option letter or an empty string."""
    option = extract_multiple_choice_option(text)
    return option.casefold() if option is not None else ""


def multiple_choice_format_valid(text: str | None) -> bool:
    """Return whether the final line is one boxed uppercase option letter."""
    if text is None:
        return False
    normalized = unicodedata.normalize("NFKC", str(text)).strip()
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not lines:
        return False
    return (
        len(ANSWER_PATTERN.findall(normalized)) == 1
        and MULTIPLE_CHOICE_STRICT_PATTERN.fullmatch(lines[-1]) is not None
    )


def get_raw_tail_fallback_chars(answer_type: str | None) -> int:
    """Return the no-answer-box fallback window for one answer type."""
    if answer_type in COUNTING_ANSWER_TYPES:
        return COUNTING_RAW_TAIL_FALLBACK_CHARS
    return DEFAULT_RAW_TAIL_FALLBACK_CHARS


def normalize_ground_truth_answers(answer: object) -> list[str]:
    """Return the acceptable ground-truth answers as normalized strings."""
    if answer is None:
        return []
    if isinstance(answer, list):
        values = answer
    elif isinstance(answer, tuple):
        values = list(answer)
    else:
        values = [answer]
    return [str(value) for value in values if value is not None]


def load_ground_truth(path: Path) -> list[dict]:
    """Load ground-truth items — accepts the same two shapes as inference.py."""
    data = orjson.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    raise ValueError(
        f"Unrecognised ground-truth structure at {path}: "
        "expected a list or an object with a 'questions' key."
    )


def load_predictions(path: Path) -> tuple[list[dict], dict]:
    """Read the prediction JSON written by inference.py.

    Accepts two shapes:
      * Current: ``{"metadata": {...}, "records": [...]}`` — the metadata
        block carries run-level token/cost aggregates we want to forward.
      * Legacy:  a bare list of records (no metadata available).
    Returns ``(records, metadata)`` in either case.
    """
    data = orjson.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data, {}
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return data["records"], data.get("metadata") or {}
    raise ValueError(
        f"Expected a JSON list of records or {{'metadata', 'records'}} "
        f"at {path}, got {type(data).__name__}."
    )


def resolve_prediction_ground_truth(
    record: dict,
    ground_truth_by_qid: dict[str, dict],
    ground_truth_by_index: dict[int, dict],
) -> tuple[dict | None, str | None, int | None]:
    """Resolve one prediction record to its ground-truth item."""
    raw_qid = record.get("qid")
    qid = str(raw_qid).strip() if raw_qid is not None else None

    index = record.get("index")
    if qid:
        gt_item = ground_truth_by_qid.get(qid)
        if gt_item is not None:
            return gt_item, qid, None

    if isinstance(index, int):
        return ground_truth_by_index.get(index), qid, index

    return None, qid, None


def get_prediction_text(record: dict) -> str | None:
    """Return the raw model text from either the new or legacy field name."""
    raw_answer = record.get("raw_answer")
    if raw_answer is not None:
        return raw_answer
    return record.get("model_output")


def score_prediction(
    raw_prediction: str | None,
    ground_truth: object,
    answer_type: str,
    question_type: str | None = None,
) -> dict:
    """Score one model output against its ground truth.

    Multiple-choice outputs are read as an option letter. Other outputs use the
    last ``\\answer{...}`` box, or the raw tail when there is none. Both sides
    go through ``normalize_answer`` and then a task-specific canonical form;
    free-form answers also accept a soft canonical substring match.
    """
    multiple_choice_item = (
        question_type == REACHABILITY_MULTIPLE_QUESTION_TYPE
        or answer_type == REACHABILITY_MULTIPLE_ANSWER_TYPE
    )
    answer_box_present = has_answer_box(raw_prediction)
    if multiple_choice_item:
        extracted = extract_multiple_choice_option(raw_prediction)
        comparison_source = "multiple_choice_response"
    elif answer_box_present:
        extracted = extract_answer(raw_prediction)
        comparison_source = "answer_box"
    else:
        fallback_chars = get_raw_tail_fallback_chars(answer_type)
        extracted = get_tail_fallback_answer(raw_prediction, fallback_chars)
        comparison_source = f"raw_tail_{fallback_chars}"
    parsed = extracted is not None

    # Two-stage match. First, ``normalize_answer`` peels off cosmetic
    # wrappers (quotes, markdown, LaTeX, prefixes, trailing parens).
    # Then ``canonical_form`` strips to lowercase alphanumerics only —
    # any remaining differences in spacing, hyphenation, or
    # capitalisation vanish. Both stages run symmetrically on the
    # comparison candidate and the ground truth, so they can only
    # forgive false negatives.
    ground_truth_values = normalize_ground_truth_answers(ground_truth)
    extracted_norm = normalize_answer(extracted) if parsed else ""
    ground_truth_norm = [normalize_answer(value) for value in ground_truth_values]
    if (
        question_type == REACHABILITY_COUNTING_QUESTION_TYPE
        or answer_type == REACHABILITY_COUNTING_ANSWER_TYPE
    ):
        normalize = normalize_counting_integer
    elif answer_type == REACHABILITY_MULTIPLE_ANSWER_TYPE:
        normalize = normalize_multiple_choice_option
    elif answer_type == REACHABILITY_ANSWER_TYPE:
        normalize = normalize_reachability_label
    else:
        normalize = None
    if normalize is not None:
        extracted_canonical = normalize(extracted) if parsed else ""
        ground_truth_canonical = [normalize(value) for value in ground_truth_values]
        exact_match = (
            parsed
            and bool(extracted_canonical)
            and extracted_canonical in ground_truth_canonical
        )
        soft_match = False
    else:
        extracted_canonical = canonical_form(extracted_norm) if parsed else ""
        ground_truth_canonical = [canonical_form(value) for value in ground_truth_norm]
        exact_match = parsed and extracted_canonical in ground_truth_canonical
        soft_match = (
            not exact_match
            and parsed
            and any(
                bool(canonical) and canonical in extracted_canonical
                for canonical in ground_truth_canonical
            )
        )

    format_valid: bool | None = None
    if multiple_choice_item:
        format_valid = multiple_choice_format_valid(raw_prediction)
    if exact_match:
        match_type = "exact_match"
    elif soft_match:
        match_type = "soft_match"
    else:
        match_type = "wrong_answer"
    return {
        "multiple_choice_item": multiple_choice_item,
        "ground_truth_normalized": ground_truth_norm,
        "ground_truth_canonical": ground_truth_canonical,
        "extracted": extracted,
        "extracted_normalized": extracted_norm if parsed else None,
        "extracted_canonical": extracted_canonical if parsed else None,
        "extracted_option": (
            extracted_canonical if multiple_choice_item and parsed else None
        ),
        "answer_box_present": answer_box_present,
        "comparison_source": comparison_source,
        "parsed": parsed,
        "correct": exact_match or soft_match,
        "match_type": match_type,
        "format_valid": format_valid,
        "format_mismatch": multiple_choice_item and not format_valid,
    }


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score inference.py predictions from the final \\answer{...} box "
            "using exact canonical match plus a soft canonical substring "
            "fallback."
        ),
    )
    parser.add_argument(
        "--groundtruth-path",
        dest="groundtruth_path",
        type=str,
        required=True,
        help="Ground-truth JSON (list or {'questions': [...]}).",
    )
    parser.add_argument(
        "--prediction-path",
        dest="prediction_path",
        type=str,
        required=True,
        help="Prediction JSON (list of records) produced by inference.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the evaluation report JSON.",
    )
    parser.add_argument(
        "--question-type",
        type=str,
        default=None,
        help=(
            "Optional question type. Reachability types add task-specific "
            "metrics to the evaluation report."
        ),
    )
    return parser


def main():
    args = arg_parser().parse_args()

    gt_path = Path(args.groundtruth_path)
    pred_path = Path(args.prediction_path)
    out_path = Path(args.output)
    question_type = getattr(args, "question_type", None)

    gt_items = load_ground_truth(gt_path)
    pred_records, inference_metadata = load_predictions(pred_path)

    gt_by_index = {i: item for i, item in enumerate(gt_items)}
    gt_by_qid = {
        str(item.get("qid")).strip(): item
        for item in gt_items
        if str(item.get("qid") or "").strip()
    }

    details: list[dict] = []
    correct = 0
    unparseable = 0
    missing_gt = 0
    no_answer_box = 0
    wrong_format_in_answer_box = 0
    format_mismatch = 0
    wrong_answer = 0
    reachability_counts = {
        "yes": {"total": 0, "correct": 0},
        "no": {"total": 0, "correct": 0},
    }
    multiple_choice_counts: dict[str, dict[str, int]] = {}

    for rec in pred_records:
        turn_index = rec.get("turn_index", 1)
        turn_fields = {
            "turn_index": turn_index,
            "inference_id": rec.get("inference_id")
            or f"{rec.get('qid', rec.get('index'))}::turn-{turn_index}",
            **{
                field: rec.get(field)
                for field in ("usage", "provider_usage", "cost", "service_tier")
            },
            "failed_attempts": rec.get("failed_attempts", []),
        }
        gt_item, qid, idx = resolve_prediction_ground_truth(
            rec,
            gt_by_qid,
            gt_by_index,
        )

        if gt_item is None:
            missing_gt += 1
            details.append(
                {
                    **turn_fields,
                    "qid": qid,
                    "index": idx,
                    "ground_truth": None,
                    "extracted": None,
                    "parsed": False,
                    "correct": False,
                    "note": "No matching ground-truth item for this qid/index.",
                }
            )
            continue

        ground_truth = gt_item.get("answer")
        answer_type = str(gt_item.get("answer_type") or "").strip()
        raw_prediction = get_prediction_text(rec)
        score = score_prediction(raw_prediction, ground_truth, answer_type, question_type)
        multiple_choice_item = score["multiple_choice_item"]
        answer_box_present = score["answer_box_present"]
        ground_truth_canonical = score["ground_truth_canonical"]
        is_correct = score["correct"]

        if not score["parsed"]:
            unparseable += 1
        if not answer_box_present and not multiple_choice_item:
            no_answer_box += 1
        if score["format_mismatch"]:
            format_mismatch += 1
        if is_correct:
            correct += 1
            if score["match_type"] == "soft_match" and answer_box_present:
                wrong_format_in_answer_box += 1
        else:
            wrong_answer += 1

        if (
            question_type == REACHABILITY_QUESTION_TYPE
            and answer_type == REACHABILITY_ANSWER_TYPE
        ):
            for label in ("yes", "no"):
                if label not in ground_truth_canonical:
                    continue
                reachability_counts[label]["total"] += 1
                if is_correct:
                    reachability_counts[label]["correct"] += 1
                break

        if multiple_choice_item:
            correct_option = next(
                (
                    value
                    for value in ground_truth_canonical
                    if len(value) == 1 and value.isalpha()
                ),
                "",
            )
            if correct_option:
                counts = multiple_choice_counts.setdefault(
                    correct_option,
                    {"total": 0, "correct": 0},
                )
                counts["total"] += 1
                if is_correct:
                    counts["correct"] += 1

        details.append(
            {
                **turn_fields,
                "qid": qid or gt_item.get("qid"),
                "index": idx,
                "question": gt_item.get("question"),
                "answer_type": gt_item.get("answer_type"),
                "ground_truth": ground_truth,
                "ground_truth_normalized": score["ground_truth_normalized"],
                "ground_truth_canonical": ground_truth_canonical,
                "raw_prediction": raw_prediction,
                "extracted": score["extracted"],
                "extracted_normalized": score["extracted_normalized"],
                "extracted_canonical": score["extracted_canonical"],
                "extracted_option": score["extracted_option"],
                "answer_box_present": answer_box_present,
                "comparison_source": score["comparison_source"],
                "parsed": score["parsed"],
                "correct": is_correct,
                "match_type": score["match_type"],
                "format_valid": score["format_valid"],
                "format_mismatch": score["format_mismatch"],
            }
        )

    total = len(details)
    accuracy = (correct / total) if total else 0.0

    cost_and_tokens = aggregate_accounting(details)
    retry_accounting = aggregate_retry_accounting(details)
    turn_metrics = {}
    for detail in details:
        counts = turn_metrics.setdefault(
            str(detail["turn_index"]), {"total": 0, "correct": 0}
        )
        counts["total"] += 1
        counts["correct"] += int(detail["correct"])
    for counts in turn_metrics.values():
        counts["accuracy"] = counts["correct"] / counts["total"]

    answer_type_metrics = None
    if question_type == REACHABILITY_QUESTION_TYPE:
        answer_type_metrics = {}
        for label in ("yes", "no"):
            label_total = reachability_counts[label]["total"]
            label_correct = reachability_counts[label]["correct"]
            answer_type_metrics[label] = {
                "total": label_total,
                "correct": label_correct,
                "accuracy": label_correct / label_total if label_total else 0.0,
            }

    option_metrics = None
    if question_type == REACHABILITY_MULTIPLE_QUESTION_TYPE:
        option_metrics = {}
        for option, counts in sorted(multiple_choice_counts.items()):
            option_total = counts["total"]
            option_correct = counts["correct"]
            option_metrics[option] = {
                "total": option_total,
                "correct": option_correct,
                "accuracy": (
                    option_correct / option_total if option_total else 0.0
                ),
            }

    report = {
        "metric": "exact_then_soft_match_accuracy",
        "total": total,
        "total_examples": len(
            {detail.get("qid") or detail.get("index") for detail in details}
        ),
        "turn_metrics": turn_metrics,
        "retry_accounting": retry_accounting,
        "correct": correct,
        "unparseable": unparseable,
        "missing_ground_truth": missing_gt,
        "no_answer_box": no_answer_box,
        "wrong_format_in_answer_box": wrong_format_in_answer_box,
        "format_mismatch": format_mismatch,
        "format_mismatch_rate": format_mismatch / total if total else 0.0,
        "wrong_answer": wrong_answer,
        "accuracy": accuracy,
        "groundtruth_path": str(gt_path),
        "prediction_path": str(pred_path),
        # Pass-through: the full inference run's metadata block (model name,
        # temperature, max_workers, aggregate cost across the whole run).
        "inference_metadata": inference_metadata,
        # Fresh aggregation over only the scored items — matches the rest of
        # the report scope, so accuracy and cost refer to the same subset.
        "cost_and_tokens": cost_and_tokens,
        "details": details,
    }
    if answer_type_metrics is not None:
        report["answer_type_metrics"] = answer_type_metrics
    if option_metrics is not None:
        report["option_metrics"] = option_metrics

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        orjson.dumps(report, option=orjson.OPT_INDENT_2).decode("utf-8"),
        encoding="utf-8",
    )

    print(f"Accuracy: {accuracy:.4f}  ({correct}/{total})")
    print(f"No answer box: {no_answer_box}")
    print(f"Wrong format in answer box: {wrong_format_in_answer_box}")
    if question_type == REACHABILITY_MULTIPLE_QUESTION_TYPE:
        print(f"Format mismatch: {format_mismatch}")
    print(f"Wrong answer: {wrong_answer}")
    print(f"Unparseable: {unparseable}")
    if answer_type_metrics is not None:
        for label in ("yes", "no"):
            metrics = answer_type_metrics[label]
            print(
                f"{label.upper()} accuracy: {metrics['accuracy']:.4f}  "
                f"({metrics['correct']}/{metrics['total']})"
            )
    if option_metrics is not None:
        for option, metrics in option_metrics.items():
            print(
                f"Option {option.upper()} accuracy: {metrics['accuracy']:.4f}  "
                f"({metrics['correct']}/{metrics['total']})"
            )
    if missing_gt:
        print(f"Missing ground truth: {missing_gt}")
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
