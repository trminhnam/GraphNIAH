from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest

import eval as eval_module


def run_eval(
    monkeypatch,
    groundtruth_path: Path,
    prediction_path: Path,
    output_path: Path,
    question_type: str | None = None,
) -> dict:
    monkeypatch.setattr(
        eval_module,
        "arg_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(
                groundtruth_path=str(groundtruth_path),
                prediction_path=str(prediction_path),
                output=str(output_path),
                question_type=question_type,
            )
        ),
    )

    eval_module.main()

    return orjson.loads(output_path.read_text(encoding="utf-8"))


def test_eval_uses_strict_reachability_labels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    groundtruth_path = tmp_path / "groundtruth.json"
    prediction_path = tmp_path / "predictions.json"
    output_path = tmp_path / "report.json"
    groundtruth_path.write_text(
        orjson.dumps({
                "questions": [
                    {
                        "qid": "yes-question",
                        "answer": ["YES"],
                        "answer_type": "REACHABILITY_YESNO",
                    },
                    {
                        "qid": "no-question",
                        "answer": ["NO"],
                        "answer_type": "REACHABILITY_YESNO",
                    },
                ]
            }).decode("utf-8"),
        encoding="utf-8",
    )
    prediction_path.write_text(
        orjson.dumps({
                "records": [
                    {"qid": "yes-question", "raw_answer": "\\answer{yEs}"},
                    {
                        "qid": "no-question",
                        "raw_answer": "\\answer{No explanation}",
                    },
                ]
            }).decode("utf-8"),
        encoding="utf-8",
    )

    report = run_eval(
        monkeypatch,
        groundtruth_path=groundtruth_path,
        prediction_path=prediction_path,
        output_path=output_path,
        question_type="reachability-yesno",
    )

    assert report["correct"] == 1
    assert report["wrong_answer"] == 1
    assert report["answer_type_metrics"] == {
        "yes": {"total": 1, "correct": 1, "accuracy": 1.0},
        "no": {"total": 1, "correct": 0, "accuracy": 0.0},
    }
    assert report["details"][0]["match_type"] == "exact_match"
    assert report["details"][1]["match_type"] == "wrong_answer"


def test_eval_recovers_multiple_choice_variants_and_counts_format_mismatches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    groundtruth_path = tmp_path / "groundtruth.json"
    prediction_path = tmp_path / "predictions.json"
    output_path = tmp_path / "report.json"
    answers = [
        "C",
        "C.",
        "C)",
        'C. Paper "Target"',
        'C) Paper "Target"',
    ]
    responses = [
        "C",
        "c",
        "C.",
        'C) Paper "Target"',
        "Option C",
        "\\answer{C}",
        "Brief reasoning.\n\\answer{C}",
        "\\answer{D}",
    ]
    groundtruth_path.write_text(
        orjson.dumps({
                "questions": [
                    {
                        "qid": f"question-{index}",
                        "answer": answers,
                        "answer_type": "REACHABILITY_MULTIPLE",
                    }
                    for index in range(len(responses))
                ]
            }).decode("utf-8"),
        encoding="utf-8",
    )
    prediction_path.write_text(
        orjson.dumps({
                "records": [
                    {
                        "qid": f"question-{index}",
                        "raw_answer": response,
                    }
                    for index, response in enumerate(responses)
                ]
            }).decode("utf-8"),
        encoding="utf-8",
    )

    report = run_eval(
        monkeypatch,
        groundtruth_path=groundtruth_path,
        prediction_path=prediction_path,
        output_path=output_path,
        question_type="reachability-multiple",
    )

    assert report["correct"] == 7
    assert report["wrong_answer"] == 1
    assert report["unparseable"] == 0
    assert report["no_answer_box"] == 0
    assert report["format_mismatch"] == 5
    assert report["format_mismatch_rate"] == 0.625
    assert report["option_metrics"] == {
        "c": {"total": 8, "correct": 7, "accuracy": 0.875}
    }
    assert report["details"][0]["format_valid"] is False
    assert report["details"][1]["raw_prediction"] == "c"
    assert report["details"][1]["extracted_option"] == "c"
    assert report["details"][1]["format_mismatch"] is True
    assert report["details"][-1]["correct"] is False
    assert report["details"][-1]["format_valid"] is True


def test_eval_uses_strict_reachability_counting_integers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    groundtruth_path = tmp_path / "groundtruth.json"
    prediction_path = tmp_path / "predictions.json"
    output_path = tmp_path / "report.json"
    groundtruth_path.write_text(
        orjson.dumps({
                "questions": [
                    {
                        "qid": "exact-count",
                        "question_type": "reachability-counting",
                        "answer": ["12"],
                        "answer_type": "REACHABILITY_COUNTING",
                    },
                    {
                        "qid": "substring-count",
                        "question_type": "reachability-counting",
                        "answer": ["12"],
                        "answer_type": "REACHABILITY_COUNTING",
                    },
                    {
                        "qid": "text-count",
                        "question_type": "reachability-counting",
                        "answer": ["12"],
                        "answer_type": "REACHABILITY_COUNTING",
                    },
                ]
            }).decode("utf-8"),
        encoding="utf-8",
    )
    prediction_path.write_text(
        orjson.dumps({
                "records": [
                    {"qid": "exact-count", "raw_answer": "\\answer{0012}"},
                    {"qid": "substring-count", "raw_answer": "\\answer{312}"},
                    {"qid": "text-count", "raw_answer": "\\answer{12 papers}"},
                ]
            }).decode("utf-8"),
        encoding="utf-8",
    )

    report = run_eval(
        monkeypatch,
        groundtruth_path=groundtruth_path,
        prediction_path=prediction_path,
        output_path=output_path,
        question_type="reachability-counting",
    )

    assert report["correct"] == 1
    assert report["wrong_answer"] == 2
    assert report["details"][0]["ground_truth_canonical"] == ["12"]
    assert report["details"][1]["match_type"] == "wrong_answer"
    assert report["details"][2]["match_type"] == "wrong_answer"


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("\\answer{C}", True),
        ("Brief reasoning.\n\\answer{C}", True),
        ("\\answer{c}", False),
        ("C", False),
        ("\\answer{C}.", False),
        ("\\answer{C}\nTrailing text", False),
        ("\\answer{B}\n\\answer{C}", False),
    ],
)
def test_multiple_choice_format_requires_final_boxed_uppercase_option(
    response: str,
    expected: bool,
) -> None:
    assert eval_module.multiple_choice_format_valid(response) is expected


def test_eval_matches_qid_predictions_with_raw_answer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    groundtruth_path = tmp_path / "groundtruth.json"
    prediction_path = tmp_path / "predictions.json"
    output_path = tmp_path / "report.json"

    groundtruth_path.write_text(
        orjson.dumps({
                "questions": [
                    {
                        "qid": "question-1",
                        "question": "Who wrote the paper?",
                        "answer": ["Alice", "A. Smith"],
                        "answer_type": "AUTHOR_NAME",
                    }
                ]
            }).decode("utf-8"),
        encoding="utf-8",
    )
    prediction_path.write_text(
        orjson.dumps({
                "records": [
                    {
                        "qid": "question-1",
                        "raw_answer": "reasoning\n\\answer{A. Smith}",
                    }
                ]
            }).decode("utf-8"),
        encoding="utf-8",
    )

    report = run_eval(
        monkeypatch,
        groundtruth_path=groundtruth_path,
        prediction_path=prediction_path,
        output_path=output_path,
    )

    assert report["accuracy"] == 1.0
    assert report["correct"] == 1
    assert report["no_answer_box"] == 0
    assert report["wrong_format_in_answer_box"] == 0
    assert report["wrong_answer"] == 0
    assert "answer_type_metrics" not in report
    assert report["details"][0]["qid"] == "question-1"
    assert report["details"][0]["match_type"] == "exact_match"


def test_eval_accepts_scalar_ground_truth_answers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    groundtruth_path = tmp_path / "groundtruth.json"
    prediction_path = tmp_path / "predictions.json"
    output_path = tmp_path / "report.json"

    groundtruth_path.write_text(
        orjson.dumps({
                "questions": [
                    {
                        "qid": "question-1",
                        "question": "Who wrote the paper?",
                        "answer": "Alice",
                        "answer_type": "AUTHOR_NAME",
                    }
                ]
            }).decode("utf-8"),
        encoding="utf-8",
    )
    prediction_path.write_text(
        orjson.dumps({
                "records": [
                    {
                        "qid": "question-1",
                        "raw_answer": "reasoning\n\\answer{Alice}",
                    }
                ]
            }).decode("utf-8"),
        encoding="utf-8",
    )

    report = run_eval(
        monkeypatch,
        groundtruth_path=groundtruth_path,
        prediction_path=prediction_path,
        output_path=output_path,
    )

    assert report["accuracy"] == 1.0
    assert report["correct"] == 1
    assert report["wrong_answer"] == 0
    assert report["details"][0]["ground_truth"] == "Alice"
    assert report["details"][0]["match_type"] == "exact_match"


def test_eval_uses_raw_tail_fallback_when_answer_box_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    groundtruth_path = tmp_path / "groundtruth.json"
    prediction_path = tmp_path / "predictions.json"
    output_path = tmp_path / "report.json"

    groundtruth_path.write_text(
        orjson.dumps({
                "questions": [
                    {
                        "qid": "question-1",
                        "question": "Exact match question",
                        "answer": ["Alice"],
                        "answer_type": "AUTHOR_NAME",
                    },
                    {
                        "qid": "question-2",
                        "question": "Soft match question",
                        "answer": ["Graph Theory"],
                        "answer_type": "TITLE",
                    },
                    {
                        "qid": "question-3",
                        "question": "No answer box exact match question",
                        "answer": ["Charlie"],
                        "answer_type": "AUTHOR_NAME",
                    },
                    {
                        "qid": "question-4",
                        "question": "No answer box soft match question",
                        "answer": ["Delta"],
                        "answer_type": "TITLE",
                    },
                    {
                        "qid": "question-5",
                        "question": "No answer box wrong answer question",
                        "answer": ["Echo"],
                        "answer_type": "AUTHOR_NAME",
                    },
                    {
                        "qid": "question-6",
                        "question": "Empty answer box question",
                        "answer": ["Foxtrot"],
                        "answer_type": "TITLE",
                    },
                    {
                        "qid": "question-7",
                        "question": "No answer box counting exact match question",
                        "answer": [12],
                        "answer_type": "COUNT_PAPER_CITATIONS",
                    },
                    {
                        "qid": "question-8",
                        "question": "No answer box counting wrong answer question",
                        "answer": [34],
                        "answer_type": "COUNT_AUTHOR_PAPERS",
                    },
                ]
            }).decode("utf-8"),
        encoding="utf-8",
    )
    prediction_path.write_text(
        orjson.dumps({
                "records": [
                    {
                        "qid": "question-1",
                        "raw_answer": "reasoning\n\\answer{Alice}",
                    },
                    {
                        "qid": "question-2",
                        "raw_answer": "\\answer{Graph Theory extra text}",
                    },
                    {
                        "qid": "question-3",
                        "raw_answer": (
                            "ignored prefix "
                            + "." * 93
                            + "Charlie"
                        ),
                    },
                    {
                        "qid": "question-4",
                        "raw_answer": (
                            "ignored prefix "
                            + "." * 84
                            + "Delta extra text"
                        ),
                    },
                    {
                        "qid": "question-5",
                        "raw_answer": "ignored prefix " + "." * 93 + "Hotel",
                    },
                    {
                        "qid": "question-6",
                        "raw_answer": "\\answer{   }",
                    },
                    {
                        "qid": "question-7",
                        "raw_answer": "ignored prefix " + "." * 27 + "12",
                    },
                    {
                        "qid": "question-8",
                        "raw_answer": "ignored prefix 34" + "." * 31,
                    },
                ]
            }).decode("utf-8"),
        encoding="utf-8",
    )

    report = run_eval(
        monkeypatch,
        groundtruth_path=groundtruth_path,
        prediction_path=prediction_path,
        output_path=output_path,
    )

    assert report["accuracy"] == pytest.approx(5 / 8)
    assert report["correct"] == 5
    assert report["no_answer_box"] == 5
    assert report["wrong_format_in_answer_box"] == 1
    assert report["wrong_answer"] == 3
    assert report["unparseable"] == 1

    details_by_qid = {detail["qid"]: detail for detail in report["details"]}

    assert details_by_qid["question-1"]["match_type"] == "exact_match"
    assert details_by_qid["question-2"]["match_type"] == "soft_match"
    assert details_by_qid["question-2"]["correct"] is True
    assert details_by_qid["question-3"]["match_type"] == "exact_match"
    assert details_by_qid["question-3"]["answer_box_present"] is False
    assert details_by_qid["question-3"]["comparison_source"] == "raw_tail_100"
    assert details_by_qid["question-3"]["correct"] is True
    assert details_by_qid["question-4"]["match_type"] == "soft_match"
    assert details_by_qid["question-4"]["answer_box_present"] is False
    assert details_by_qid["question-4"]["comparison_source"] == "raw_tail_100"
    assert details_by_qid["question-4"]["correct"] is True
    assert details_by_qid["question-5"]["match_type"] == "wrong_answer"
    assert details_by_qid["question-5"]["answer_box_present"] is False
    assert details_by_qid["question-5"]["comparison_source"] == "raw_tail_100"
    assert details_by_qid["question-6"]["match_type"] == "wrong_answer"
    assert details_by_qid["question-6"]["answer_box_present"] is True
    assert details_by_qid["question-6"]["parsed"] is False
    assert details_by_qid["question-7"]["match_type"] == "exact_match"
    assert details_by_qid["question-7"]["comparison_source"] == "raw_tail_30"
    assert details_by_qid["question-7"]["correct"] is True
    assert details_by_qid["question-8"]["match_type"] == "wrong_answer"
    assert details_by_qid["question-8"]["comparison_source"] == "raw_tail_30"


def test_eval_preserves_turns_and_reports_partial_costs(tmp_path: Path, monkeypatch):
    groundtruth_path = tmp_path / "groundtruth.json"
    prediction_path = tmp_path / "predictions.json"
    output_path = tmp_path / "report.json"
    groundtruth_path.write_bytes(orjson.dumps([{"qid": "q1", "answer": "Alice"}]))
    prediction_path.write_bytes(
        orjson.dumps(
            {
                "records": [
                    {
                        "qid": "q1",
                        "turn_index": 1,
                        "inference_id": "q1::turn-1",
                        "raw_answer": "\\answer{Alice}",
                        "cost": {"total_cost": 0.1},
                    },
                    {
                        "qid": "q1",
                        "turn_index": 2,
                        "inference_id": "q1::turn-2",
                        "raw_answer": "\\answer{Bob}",
                        "cost": {"total_cost": None},
                        "failed_attempts": [{"cost": {"total_cost": 0.02}}],
                    },
                    {
                        "qid": "missing",
                        "turn_index": 2,
                        "inference_id": "missing::turn-2",
                        "raw_answer": "answer",
                    },
                ]
            }
        )
    )
    report = run_eval(monkeypatch, groundtruth_path, prediction_path, output_path)
    assert report["total"] == 3
    assert report["total_examples"] == 2
    assert report["accuracy"] == pytest.approx(1 / 3)
    assert report["turn_metrics"] == {
        "1": {"total": 1, "correct": 1, "accuracy": 1.0},
        "2": {"total": 2, "correct": 0, "accuracy": 0.0},
    }
    assert [d["inference_id"] for d in report["details"]] == [
        "q1::turn-1",
        "q1::turn-2",
        "missing::turn-2",
    ]
    assert report["cost_and_tokens"]["avg_total_cost"] == 0.1
    assert report["cost_and_tokens"]["records_without_cost"] == 2
    assert report["retry_accounting"]["total_total_cost"] == 0.02
