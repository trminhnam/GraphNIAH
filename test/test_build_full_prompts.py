from __future__ import annotations

from collections import UserDict
from collections.abc import Callable
import orjson
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

import build_full_prompts
from src.prompts import FORMAT_REMINDER
from src.graph.storage import DatabaseManager


class FakeTokenizer:
    def __init__(self, *, prompt_offset: int = 0) -> None:
        self.prompt_offset = prompt_offset

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return list(text)

    def decode(
        self,
        token_ids: list[str],
        skip_special_tokens: bool = True,
    ) -> str:
        del skip_special_tokens
        return "".join(token_ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
    ) -> list[str]:
        assert tokenize is True
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        return list(rendered) + (["~"] * self.prompt_offset)


class CountingFragmentTokenizer(FakeTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.chat_template_calls = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> list[str]:
        self.chat_template_calls += 1
        token_ids = super().apply_chat_template(messages, tokenize=tokenize)
        if truncation and max_length is not None:
            return token_ids[:max_length]
        return token_ids


class RecordingMessageTokenizer(FakeTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.message_calls: list[list[dict[str, str]]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
    ) -> list[str]:
        self.message_calls.append(messages)
        return super().apply_chat_template(messages, tokenize=tokenize)


class WordCountingTokenizer:
    def __init__(self) -> None:
        self.count_calls = 0

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        self.count_calls += 1
        return sum(len(message["content"].split()) for message in messages)


class TruncatingTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[bool, int | None]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> list[str]:
        assert tokenize is True
        self.calls.append((truncation, max_length))
        rendered = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        token_ids = list(rendered)
        if truncation and max_length is not None:
            return token_ids[:max_length]
        return token_ids


class TemplateLessTokenizer(FakeTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
    ) -> list[str]:
        del messages, tokenize
        raise ValueError("tokenizer.chat_template is not set")


class ComparableTokenizer(FakeTokenizer):
    def __init__(
        self,
        *,
        vocab: dict | None = None,
        tokens_by_id: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab or {}
        self.vocab_size = len(self.vocab)
        self.tokens_by_id = tokens_by_id or []

    def get_vocab(self) -> dict:
        return self.vocab

    def convert_ids_to_tokens(self, index: int) -> str:
        return self.tokens_by_id[index]


class FakeOpenAIEncoding:
    name = "fake-openai"

    def encode(self, text: str) -> list[str]:
        return list(text)


class FakeTiktoken:
    def __init__(self) -> None:
        self.model_names: list[str] = []
        self.encoding_names: list[str] = []

    def encoding_for_model(self, model_name: str) -> FakeOpenAIEncoding:
        self.model_names.append(model_name)
        return FakeOpenAIEncoding()

    def get_encoding(self, encoding_name: str) -> FakeOpenAIEncoding:
        self.encoding_names.append(encoding_name)
        return FakeOpenAIEncoding()


def build_input_file(
    tmp_path: Path,
    questions: list[dict],
    *,
    metadata: dict | None = None,
    parameters: dict | None = None,
) -> Path:
    input_path = tmp_path / "questions.json"
    input_path.write_text(
        orjson.dumps({
                "metadata": metadata or {"source_questions": "source.json"},
                "parameters": parameters or {},
                "questions": questions,
            }).decode("utf-8"),
        encoding="utf-8",
    )
    return input_path


def test_build_full_prompts_builds_question_first_user_message(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Who wrote the target paper?",
                "needles": ["needle 1", "needle 2"],
                "answer": ["Alice", " A. Smith ", None, ""],
                "answer_type": "AUTHOR_NAME",
            }
        ],
    )

    result = build_full_prompts.build_full_prompts(input_path)

    assert result["metadata"]["full_prompts"]["question_last"] is False
    assert result["metadata"]["full_prompts"]["synthesized_qids"] == 0
    assert result["parameters"] == {
        "num_hops": None,
        "needle_type": None,
        "distracting_content_type": None,
        "needle_order": None,
        "target_context_length": None,
    }
    built = [
        {key: value for key, value in question.items() if key != "source_fingerprint"}
        for question in result["questions"]
    ]
    assert built == [
        {
            "qid": "question-1",
            "user_message": (
                "# Question\n\n"
                "Who wrote the target paper?\n\n"
                "# Context\n\n"
                "needle 1\n\n"
                "###\n\n"
                "needle 2\n\n"
                f"{FORMAT_REMINDER}"
            ),
            "answer": ["Alice", "A. Smith"],
            "answer_type": "AUTHOR_NAME",
            "needle_items": 2,
            "distractor_items": 0,
        }
    ]
    assert result["questions"][0]["source_fingerprint"]


def test_build_full_prompts_builds_question_last_user_message(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "question": "How many citations are there?",
                "needles": ["needle 1"],
                "answer": 3,
                "answer_type": "COUNT_PAPER_CITATIONS",
                "distracting_content_items": [],
            }
        ],
    )

    result = build_full_prompts.build_full_prompts(
        input_path,
        question_last=True,
    )

    output_question = result["questions"][0]
    assert result["metadata"]["full_prompts"]["question_last"] is True
    assert result["metadata"]["full_prompts"]["synthesized_qids"] == 1
    assert result["parameters"]["needle_order"] is None
    assert result["parameters"]["target_context_length"] is None
    assert output_question["qid"]
    assert output_question["answer"] == "3"
    assert output_question["answer_type"] == "COUNT_PAPER_CITATIONS"
    assert output_question["user_message"] == (
        "# Context\n\n"
        "needle 1\n\n"
        "# Question\n\n"
        "How many citations are there?\n\n"
        f"{FORMAT_REMINDER}"
    )


def test_build_full_prompts_includes_path_distractor_paragraphs(
    tmp_path: Path,
) -> None:
    distractor_path = tmp_path / "distractors.txt"
    distractor_path.write_text(
        "distractor 1\n\ndistractor 2",
        encoding="utf-8",
    )
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Who wrote the target paper?",
                "needles": ["needle 1", "needle 2"],
                "answer": "Alice",
                "distracting_content_items": [f"path:{distractor_path}"],
            }
        ],
    )

    result = build_full_prompts.build_full_prompts(
        input_path,
        random_seed=7,
        needle_order="forward",
        context_item_prefix="",
        context_item_separator="\n\n",
    )

    user_message = result["questions"][0]["user_message"]
    assert "# Context\n\n" in user_message
    assert "needle 1" in user_message
    assert "needle 2" in user_message
    assert "distractor 1" in user_message
    assert "distractor 2" in user_message
    assert user_message.endswith(FORMAT_REMINDER)
    assert user_message.index("needle 1") < user_message.index("needle 2")


def test_build_full_prompts_includes_paper_index_distractors(
    tmp_path: Path,
) -> None:
    db = DatabaseManager(tmp_path / "graph.db")
    try:
        paper_id = db.insert_paper(
            title="Distractor Paper",
            published="2024-02-03",
            text_path="",
        )
        author_id = db.insert_author(name="Dana Author")
        db.insert_authorship(paper_id=paper_id, author_id=author_id, position=1)
    finally:
        db.close()

    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Who wrote the target paper?",
                "needles": ["needle 1"],
                "answer": "Alice",
                "distracting_content_items": [f"paper_index:{paper_id}"],
            }
        ],
        metadata={
            "source_questions": "source.json",
            "distracting_content": {"db_path": str(tmp_path / "graph.db")},
        },
        parameters={"distracting_content_type": "paper-random"},
    )

    result = build_full_prompts.build_full_prompts(
        input_path,
        random_seed=7,
        needle_order="forward",
        context_item_prefix="",
        context_item_separator="\n\n",
    )

    user_message = result["questions"][0]["user_message"]
    assert (
        'Paper "Distractor Paper", written by Dana Author, was published on ' "February 3, 2024."
    ) in user_message


def test_build_full_prompts_filters_paper_header_lines_from_path_distractors(
    tmp_path: Path,
) -> None:
    distractor_path = tmp_path / "paper.txt"
    distractor_path.write_text(
        "# Title\nabstract paragraph\n\n## Introduction\nbody paragraph\n\n## References\n\nRef1.\n\nRef2.",
        encoding="utf-8",
    )
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Who wrote the target paper?",
                "needles": ["needle 1"],
                "answer": "Alice",
                "distracting_content_items": [f"path:{distractor_path}"],
            }
        ],
        parameters={"distracting_content_type": "paper-hard-sparse"},
    )

    result = build_full_prompts.build_full_prompts(
        input_path,
        random_seed=7,
        needle_order="forward",
        context_item_prefix="",
        context_item_separator="\n\n",
    )

    user_message = result["questions"][0]["user_message"]
    assert "abstract paragraph" in user_message
    assert "body paragraph" in user_message
    assert "# Title" not in user_message
    assert "## Introduction" not in user_message
    assert "## References" not in user_message
    assert "Ref1." not in user_message
    assert "Ref2." not in user_message


def test_build_full_prompts_keeps_non_paper_headers_in_path_distractors(
    tmp_path: Path,
) -> None:
    distractor_path = tmp_path / "notes.txt"
    distractor_path.write_text(
        "# keep me\nregular paragraph",
        encoding="utf-8",
    )
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Who wrote the target paper?",
                "needles": ["needle 1"],
                "answer": "Alice",
                "distracting_content_items": [f"path:{distractor_path}"],
            }
        ],
        parameters={"distracting_content_type": "books"},
    )

    result = build_full_prompts.build_full_prompts(
        input_path,
        random_seed=7,
        needle_order="forward",
        context_item_prefix="",
        context_item_separator="\n\n",
    )

    user_message = result["questions"][0]["user_message"]
    assert "# keep me" in user_message
    assert "regular paragraph" in user_message


def test_load_path_distracting_content_paragraphs_uses_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distractor_path = tmp_path / "paper.txt"
    distractor_path.write_text(
        "# Title\nabstract paragraph\n\n## Body\nbody paragraph\n\n## References\n\nRef1.",
        encoding="utf-8",
    )

    original_read_text = Path.read_text
    read_count = 0

    def counting_read_text(path: Path, *args, **kwargs) -> str:
        nonlocal read_count
        if path == distractor_path:
            read_count += 1
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)

    path_paragraph_cache: dict[tuple[str, bool], tuple[str, ...]] = {}
    paragraphs = build_full_prompts.load_path_distracting_content_paragraphs(
        distractor_path,
        distracting_content_type="paper-related",
        path_paragraph_cache=path_paragraph_cache,
    )
    cached_paragraphs = build_full_prompts.load_path_distracting_content_paragraphs(
        distractor_path,
        distracting_content_type="paper-related",
        path_paragraph_cache=path_paragraph_cache,
    )

    assert paragraphs == ("abstract paragraph", "body paragraph")
    assert cached_paragraphs == paragraphs
    assert read_count == 1


@pytest.mark.parametrize(
    ("paragraph_count", "max_count", "largest_fitting_count"),
    [(256, 50000, 37), (4, 100, 20)],
    ids=["bounded-search", "expands-past-one-cycle"],
)
def test_find_max_fitting_distractor_count_uses_bounded_search(
    paragraph_count: int,
    max_count: int,
    largest_fitting_count: int,
) -> None:
    checked_counts: list[int] = []

    def fits_distractor_count(distractor_count: int) -> bool:
        checked_counts.append(distractor_count)
        return distractor_count <= largest_fitting_count

    found_count = build_full_prompts.find_max_fitting_distractor_count(
        ["paragraph"] * paragraph_count,
        max_count=max_count,
        fits_distractor_count=fits_distractor_count,
    )

    assert found_count == largest_fitting_count
    assert len(checked_counts) <= 12


def test_measure_tokenized_chat_length_caps_chat_template_tokenization() -> None:
    tokenizer = TruncatingTokenizer()
    messages = [{"role": "user", "content": "x" * 100}]

    token_count = build_full_prompts.measure_tokenized_chat_length(
        tokenizer,
        messages,
        max_tokens=12,
    )

    assert token_count == 12
    assert tokenizer.calls == [(True, 12)]


class BatchEncodingTokenizer(FakeTokenizer):
    """Mimic transformers 5, whose chat template returns a dict-like encoding."""

    def apply_chat_template(self, messages, tokenize=True):
        token_ids = super().apply_chat_template(messages, tokenize=tokenize)
        return UserDict(input_ids=token_ids, attention_mask=[1] * len(token_ids))


def test_measure_tokenized_chat_length_reads_input_ids_from_batch_encoding() -> None:
    messages = [{"role": "user", "content": "x" * 100}]

    token_count = build_full_prompts.measure_tokenized_chat_length(
        BatchEncodingTokenizer(), messages
    )

    assert token_count == len(FakeTokenizer().apply_chat_template(messages)) == 106


def test_build_full_prompts_packs_prompt_to_shared_token_budget(
    tmp_path: Path,
) -> None:
    distractor_path = tmp_path / "distractors.txt"
    distractor_path.write_text(
        "abcdefghij\n\nklmnopqrst\n\nuvwxyzABCD",
        encoding="utf-8",
    )
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Q?",
                "needles": ["needle"],
                "answer": "A",
                "distracting_content_items": [f"path:{distractor_path}"],
            }
        ],
    )

    tokenizers = [
        ("ref", FakeTokenizer(prompt_offset=0)),
        ("wide", FakeTokenizer(prompt_offset=12)),
    ]
    target_context_length = 75
    system_prompt = "SYS"
    format_reminder = "FMT"

    result = build_full_prompts.build_full_prompts(
        input_path,
        target_context_length=target_context_length,
        system_prompt=system_prompt,
        format_reminder=format_reminder,
        random_seed=5,
        needle_order="forward",
        context_item_prefix="",
        context_item_separator="\n\n",
        context_tail_margin=5,
        tokenizers=tokenizers,
    )

    user_message = result["questions"][0]["user_message"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    assert "needle" in user_message
    assert user_message.endswith(format_reminder)
    assert (
        sum(paragraph in user_message for paragraph in ["abcdefghij", "klmnopqrst", "uvwxyzABCD"])
        < 3
    )
    assert (
        len(tokenizers[0][1].apply_chat_template(messages, tokenize=True)) <= target_context_length
    )
    assert (
        len(tokenizers[1][1].apply_chat_template(messages, tokenize=True)) <= target_context_length
    )


def test_build_full_prompts_packs_no_system_mode_as_one_user_message(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Q?",
                "needles": ["needle"],
                "answer": "A",
            }
        ],
    )
    tokenizer = RecordingMessageTokenizer()

    result = build_full_prompts.build_full_prompts(
        input_path,
        no_system_message=True,
        target_context_length=200,
        system_prompt="SYS",
        format_reminder="FMT",
        context_item_prefix="",
        tokenizers=[("recording", tokenizer)],
    )

    record = result["questions"][0]
    assert record["user_message"].startswith(
        "# Instruction\nSYS\n\n# Question\n\nQ?\n\n# Context\n\nneedle"
    )
    assert record["no_system_message"] is True
    assert "system_prompt" not in record
    assert tokenizer.message_calls
    assert all(
        [message["role"] for message in messages] == ["user"]
        for messages in tokenizer.message_calls
    )


def test_build_full_prompts_uses_fragment_estimates_for_packing(
    tmp_path: Path,
) -> None:
    distractor_path = tmp_path / "distractors.txt"
    distractor_path.write_text(
        "\n\n".join(
            f"distractor paragraph {index} with enough text to matter"
            for index in range(200)
        ),
        encoding="utf-8",
    )
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Q?",
                "needles": ["needle"],
                "answer": "A",
                "distracting_content_items": [f"path:{distractor_path}"],
            }
        ],
    )
    tokenizer = CountingFragmentTokenizer()

    result = build_full_prompts.build_full_prompts(
        input_path,
        target_context_length=500,
        system_prompt="SYS",
        format_reminder="FMT",
        random_seed=5,
        needle_order="forward",
        context_item_prefix="",
        context_item_separator="\n\n",
        context_tail_margin=0,
        tokenizers=[("counting", tokenizer)],
    )

    assert "needle" in result["questions"][0]["user_message"]
    assert tokenizer.chat_template_calls <= 5


def test_build_full_prompts_skips_when_needles_exceed_target_context_length(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Q?",
                "needles": ["x" * 40],
                "answer": "A",
            }
        ],
    )

    result = build_full_prompts.build_full_prompts(
        input_path,
        target_context_length=20,
        system_prompt="SYS",
        format_reminder="FMT",
        tokenizers=[("ref", FakeTokenizer())],
        context_item_prefix="",
        context_item_separator="\n\n",
    )

    assert result["questions"] == []
    assert result["metadata"]["full_prompts"]["total_questions"] == 0
    assert result["metadata"]["full_prompts"]["skipped_qids"] == ["question-1"]


def test_build_full_prompts_short_circuits_tokenizer_checks_after_failure(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Q?",
                "needles": ["needle"],
                "answer": "A",
            }
        ],
    )

    class TooLongTokenizer:
        def __init__(self) -> None:
            self.count_calls = 0

        def count_messages(self, messages: list[dict[str, str]]) -> int:
            del messages
            self.count_calls += 1
            return 100

    too_long_tokenizer = TooLongTokenizer()
    later_tokenizer = TooLongTokenizer()
    result = build_full_prompts.build_full_prompts(
        input_path,
        target_context_length=5,
        system_prompt="",
        format_reminder="",
        tokenizers=[
            ("too-long", too_long_tokenizer),
            ("later", later_tokenizer),
        ],
        context_item_prefix="",
        context_item_separator="\n",
    )

    assert result["questions"] == []
    # The fit check stops at the first tokenizer over the target; the extra call
    # to each is the one measurement taken to explain the dropped question.
    assert too_long_tokenizer.count_calls == 2
    assert later_tokenizer.count_calls == 1


def test_build_full_prompts_skips_missing_answer_and_logs_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Question 1?",
                "needles": ["needle 1"],
                "answer": "Answer 1",
            },
            {
                "qid": "question-2",
                "question": "Question 2?",
                "needles": ["needle 2"],
                "answer": "  ",
            },
            {
                "qid": "question-3",
                "question": "Question 3?",
                "needles": ["needle 3"],
                "answer": "Answer 3",
            },
        ],
    )

    result = build_full_prompts.build_full_prompts(input_path)
    stdout = capsys.readouterr().out

    assert [question["qid"] for question in result["questions"]] == [
        "question-1",
        "question-3",
    ]
    assert result["metadata"]["full_prompts"]["skipped_qids"] == ["question-2"]
    assert "Question entry at index 1 must contain a non-empty 'answer' value." in stdout
    assert "SKIP index=1 qid=question-2 reason=invalid-answer" in stdout


def test_build_full_prompts_repeats_repeat_distractors_to_fill_budget(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "What is the title?",
                "needles": ["needle 1"],
                "answer": "A title",
                "distracting_content_items": ["repeat:distractor sentence"],
            }
        ],
    )

    result = build_full_prompts.build_full_prompts(
        input_path,
        target_context_length=140,
        system_prompt="SYS",
        format_reminder="FMT",
        tokenizers=[("ref", FakeTokenizer())],
        context_item_prefix="",
        context_item_separator="\n\n",
        context_tail_margin=0,
    )

    user_message = result["questions"][0]["user_message"]
    assert user_message.count("distractor sentence") >= 2
    assert "needle 1" in user_message
    assert user_message.endswith("FMT")


def test_build_full_prompts_needles_only_accepts_underfilled_prompt(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "What is the answer?",
                "needles": ["needle A", "needle B"],
                "answer": "An answer",
                "distracting_content_items": [],
            }
        ],
        parameters={"distracting_content_type": "needles-only"},
    )

    result = build_full_prompts.build_full_prompts(
        input_path,
        target_context_length=300,
        system_prompt="SYS",
        format_reminder="FMT",
        needle_order="random",
        random_seed=7,
        tokenizers=[("ref", FakeTokenizer())],
        context_item_prefix="",
        context_tail_margin=0,
    )

    user_message = result["questions"][0]["user_message"]
    assert user_message.count("needle A") == 1
    assert user_message.count("needle B") == 1
    assert "distractor" not in user_message.lower()
    assert len(user_message) < 300


def test_build_full_prompts_repeat_needles_packs_complete_shuffled_rounds(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "What is the answer?",
                "needles": ["needle A", "needle B", "needle C"],
                "answer": "An answer",
                "distracting_content_items": [],
            }
        ],
        parameters={"distracting_content_type": "repeat-needles"},
    )
    build_kwargs = {
        "target_context_length": 240,
        "system_prompt": "SYS",
        "format_reminder": "FMT",
        "needle_order": "random",
        "random_seed": 7,
        "tokenizers": [("ref", FakeTokenizer())],
        "context_item_prefix": "",
        "context_tail_margin": 0,
    }

    first = build_full_prompts.build_full_prompts(input_path, **build_kwargs)
    second = build_full_prompts.build_full_prompts(input_path, **build_kwargs)

    user_message = first["questions"][0]["user_message"]
    counts = [user_message.count(needle) for needle in ("needle A", "needle B", "needle C")]
    assert counts[0] >= 2
    assert counts[0] == counts[1] == counts[2]
    assert first["questions"] == second["questions"]
    context_text = user_message.split("# Context\n\n", 1)[1].rsplit("\n\nFMT", 1)[0]
    context_items = context_text.split("\n\n")
    assert context_items != ["needle A", "needle B", "needle C"] * counts[0]
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": user_message},
    ]
    assert len(FakeTokenizer().apply_chat_template(messages)) <= 240


def test_build_full_prompts_repeat_needles_requires_target_context_length(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [{"question": "Question?", "needles": ["needle"], "answer": "answer"}],
        parameters={"distracting_content_type": "repeat-needles"},
    )

    with pytest.raises(ValueError, match="requires --target-context-length"):
        build_full_prompts.build_full_prompts(
            input_path,
            needle_order="random",
        )


def test_build_full_prompts_repeat_needles_requires_random_order(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [{"question": "Question?", "needles": ["needle"], "answer": "answer"}],
        parameters={"distracting_content_type": "repeat-needles"},
    )

    with pytest.raises(ValueError, match="requires --needle-order=random"):
        build_full_prompts.build_full_prompts(
            input_path,
            target_context_length=200,
            system_prompt="SYS",
            format_reminder="FMT",
            needle_order="forward",
            tokenizers=[("ref", FakeTokenizer())],
        )


def test_build_full_prompts_paper_related_prioritizes_related_before_remaining(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Q",
                "needles": ["NEEDLE_A", "NEEDLE_B"],
                "answer": "A",
                "distracting_content_items": [
                    "paper_related_index_1:1",
                    "paper_related_index_1:2",
                    "paper_index:3",
                    "paper_index:4",
                ],
            }
        ],
        parameters={"distracting_content_type": "paper-related"},
    )

    def load_paper_distractor(item, index, **kwargs):
        del index, kwargs
        return (f"PARAGRAPH_{item}",)

    result = build_full_prompts.build_full_prompts(
        input_path,
        target_context_length=9,
        system_prompt="",
        format_reminder="",
        random_seed=3,
        needle_order="forward",
        context_item_prefix="",
        context_item_separator="\n",
        context_tail_margin=0,
        tokenizers=[("words", WordCountingTokenizer())],
        paper_index_distractor_loader=load_paper_distractor,
    )

    user_message = result["questions"][0]["user_message"]
    assert "NEEDLE_A" in user_message
    assert "NEEDLE_B" in user_message
    assert "PARAGRAPH_paper_related_index_1:1" in user_message
    assert "PARAGRAPH_paper_related_index_1:2" in user_message
    assert "PARAGRAPH_paper_index:3" not in user_message
    assert "PARAGRAPH_paper_index:4" not in user_message


@pytest.mark.parametrize(
    "distracting_content_type",
    [
        "paper-negative-tree",
        "paper-negative-tree-v4",
        "paper-negative-tree-v5",
        "paper-negative-tree-v6",
        "paper-negative-tree-v7",
        "paper-negative-tree-v8",
        "paper-negative-tree-v10",
        "paper-negative-tree-v11",
        "paper-negative-tree-v9",
    ],
)
def test_paper_negative_tree_packing_keeps_each_tree_atomic_when_disabled(
    tmp_path: Path,
    distracting_content_type: str,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Q",
                "question_type": "reachability-yesno",
                "answer_type": "REACHABILITY_YESNO",
                "needles": ["NEEDLE"],
                "answer": "YES",
                "distracting_content_items": [
                    "paper_negative_tree:0",
                    "paper_negative_tree:1",
                ],
            }
        ],
        parameters={"distracting_content_type": distracting_content_type},
    )

    def load_negative_tree(item, index, **kwargs):
        del index, kwargs
        tree_index = item.rsplit(":", 1)[1]
        return (f"TREE_{tree_index}_A", f"TREE_{tree_index}_B")

    result = build_full_prompts.build_full_prompts(
        input_path,
        question_type="reachability-yesno",
        traversal_mode="papers-only",
        target_context_length=9,
        system_prompt="",
        format_reminder="",
        random_seed=3,
        needle_order="random",
        context_item_prefix="",
        context_item_separator="\n",
        context_tail_margin=0,
        partial_negative_tree_packing=False,
        tokenizers=[("words", WordCountingTokenizer())],
        paper_index_distractor_loader=load_negative_tree,
    )

    user_message = result["questions"][0]["user_message"]
    included_groups = 0
    for tree_index in (0, 1):
        first_present = f"TREE_{tree_index}_A" in user_message
        second_present = f"TREE_{tree_index}_B" in user_message
        assert first_present == second_present
        included_groups += int(first_present)
    assert included_groups == 1
    assert result["metadata"]["full_prompts"][
        "partial_negative_tree_packing"
    ] is False


@pytest.mark.parametrize(
    "distracting_content_type",
    [
        "paper-negative-tree",
        "paper-negative-tree-v4",
        "paper-negative-tree-v5",
        "paper-negative-tree-v6",
        "paper-negative-tree-v7",
        "paper-negative-tree-v8",
        "paper-negative-tree-v10",
        "paper-negative-tree-v11",
        "paper-negative-tree-v9",
    ],
)
def test_paper_negative_tree_packing_uses_partial_next_tree_by_default(
    tmp_path: Path,
    distracting_content_type: str,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Q",
                "question_type": "reachability-yesno",
                "answer_type": "REACHABILITY_YESNO",
                "needles": ["NEEDLE"],
                "answer": "YES",
                "distracting_content_items": [
                    "paper_negative_tree:0",
                    "paper_negative_tree:1",
                ],
            }
        ],
        parameters={"distracting_content_type": distracting_content_type},
    )

    def load_negative_tree(item, index, **kwargs):
        del index, kwargs
        tree_index = item.rsplit(":", 1)[1]
        return (f"TREE_{tree_index}_A", f"TREE_{tree_index}_B")

    result = build_full_prompts.build_full_prompts(
        input_path,
        question_type="reachability-yesno",
        traversal_mode="papers-only",
        target_context_length=9,
        system_prompt="",
        format_reminder="",
        random_seed=3,
        needle_order="random",
        context_item_prefix="",
        context_item_separator="\n",
        context_tail_margin=0,
        tokenizers=[("words", WordCountingTokenizer())],
        paper_index_distractor_loader=load_negative_tree,
    )

    output_question = result["questions"][0]
    user_message = output_question["user_message"]
    included_block_counts = [
        sum(f"TREE_{tree_index}_{suffix}" in user_message for suffix in ("A", "B"))
        for tree_index in (0, 1)
    ]
    assert sorted(included_block_counts) == [1, 2]
    assert output_question["needle_items"] == 1
    assert output_question["distractor_items"] == 3
    assert result["metadata"]["full_prompts"][
        "partial_negative_tree_packing"
    ] is True


def test_partial_negative_tree_packing_uses_logarithmic_fit_checks(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Q",
                "question_type": "reachability-yesno",
                "answer_type": "REACHABILITY_YESNO",
                "needles": ["NEEDLE"],
                "answer": "YES",
                "distracting_content_items": ["paper_negative_tree:0"],
            }
        ],
        parameters={"distracting_content_type": "paper-negative-tree-v5"},
    )
    loader_calls: list[str] = []
    block_count = 1_024

    def load_negative_tree(item, index, **kwargs):
        del index, kwargs
        loader_calls.append(item)
        return tuple(f"BLOCK_{block_index}" for block_index in range(block_count))

    tokenizer = WordCountingTokenizer()
    result = build_full_prompts.build_full_prompts(
        input_path,
        question_type="reachability-yesno",
        traversal_mode="papers-only",
        target_context_length=70,
        system_prompt="",
        format_reminder="",
        random_seed=3,
        needle_order="random",
        context_item_prefix="",
        context_item_separator="\n",
        context_tail_margin=0,
        tokenizers=[("words", tokenizer)],
        paper_index_distractor_loader=load_negative_tree,
    )

    output_question = result["questions"][0]
    assert output_question["needle_items"] == 1
    assert output_question["distractor_items"] == 64
    assert loader_calls == ["paper_negative_tree:0"]
    assert tokenizer.count_calls <= 20


def test_paper_distractor_loader_receives_current_question_after_skip(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "skipped-question",
                "question": "Skipped",
                "needles": ["needle"],
                "answer": None,
                "distracting_content_items": ["paper_index:1"],
            },
            {
                "qid": "rendered-question",
                "question": "Rendered",
                "needles": ["needle"],
                "answer": "answer",
                "distracting_content_items": ["paper_index:2"],
            },
        ],
        parameters={"distracting_content_type": "paper-related"},
    )
    loaded: list[tuple[int, str, str]] = []

    def load_paper_distractor(item, index, **kwargs):
        question = kwargs["question"]
        loaded.append((index, question["qid"], item))
        return (f"PARAGRAPH_{item}",)

    result = build_full_prompts.build_full_prompts(
        input_path,
        paper_index_distractor_loader=load_paper_distractor,
    )

    assert loaded == [(1, "rendered-question", "paper_index:2")]
    assert [record["qid"] for record in result["questions"]] == [
        "rendered-question"
    ]


def test_build_full_prompts_paper_related_drops_related_before_needles(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Q",
                "needles": ["NEEDLE_A", "NEEDLE_B"],
                "answer": "A",
                "distracting_content_items": [
                    "paper_related_index_1:1",
                    "paper_related_index_2:2",
                    "paper_index:3",
                ],
            }
        ],
        parameters={"distracting_content_type": "paper-related"},
    )

    def load_paper_distractor(item, index, **kwargs):
        del index, kwargs
        return (f"PARAGRAPH_{item}",)

    result = build_full_prompts.build_full_prompts(
        input_path,
        target_context_length=8,
        system_prompt="",
        format_reminder="",
        random_seed=3,
        needle_order="forward",
        context_item_prefix="",
        context_item_separator="\n",
        context_tail_margin=0,
        tokenizers=[("words", WordCountingTokenizer())],
        paper_index_distractor_loader=load_paper_distractor,
    )

    user_message = result["questions"][0]["user_message"]
    assert "NEEDLE_A" in user_message
    assert "NEEDLE_B" in user_message
    assert "PARAGRAPH_paper_related_index_1:1" in user_message
    assert "PARAGRAPH_paper_related_index_2:2" not in user_message
    assert "PARAGRAPH_paper_index:3" not in user_message


def test_build_full_prompts_paper_related_respects_backward_needle_order(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Q",
                "needles": ["NEEDLE_A", "NEEDLE_B"],
                "answer": "A",
                "distracting_content_items": ["paper_related_index_1:1"],
            }
        ],
        parameters={"distracting_content_type": "paper-related"},
    )

    def load_paper_distractor(item, index, **kwargs):
        del index, kwargs
        return (f"PARAGRAPH_{item}",)

    result = build_full_prompts.build_full_prompts(
        input_path,
        random_seed=3,
        needle_order="backward",
        context_item_prefix="",
        context_item_separator="\n",
        paper_index_distractor_loader=load_paper_distractor,
    )

    user_message = result["questions"][0]["user_message"]
    assert "PARAGRAPH_paper_related_index_1:1" in user_message
    assert user_message.index("NEEDLE_B") < user_message.index("NEEDLE_A")


def test_build_full_prompts_paper_related_adds_remaining_with_bounded_search(
    tmp_path: Path,
) -> None:
    remaining_items = [f"paper_index:{index}" for index in range(128)]
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Q",
                "needles": ["NEEDLE"],
                "answer": "A",
                "distracting_content_items": [
                    "paper_related_index_1:1",
                    *remaining_items,
                ],
            }
        ],
        parameters={"distracting_content_type": "paper-related"},
    )

    def load_paper_distractor(item, index, **kwargs):
        del index, kwargs
        if item.startswith("paper_index:"):
            loaded_remaining_items.append(item)
        return (f"PARAGRAPH_{item}",)

    loaded_remaining_items: list[str] = []
    tokenizer = WordCountingTokenizer()
    result = build_full_prompts.build_full_prompts(
        input_path,
        target_context_length=60,
        system_prompt="",
        format_reminder="",
        random_seed=3,
        needle_order="forward",
        context_item_prefix="",
        context_item_separator="\n",
        context_tail_margin=0,
        tokenizers=[("words", tokenizer)],
        paper_index_distractor_loader=load_paper_distractor,
    )

    user_message = result["questions"][0]["user_message"]
    included_remaining_count = sum(f"PARAGRAPH_{item}" in user_message for item in remaining_items)
    assert "NEEDLE" in user_message
    assert "PARAGRAPH_paper_related_index_1:1" in user_message
    assert 0 < included_remaining_count < len(remaining_items)
    assert len(loaded_remaining_items) < len(remaining_items)
    assert tokenizer.count_calls < 40


def test_reachability_paper_related_reuses_safe_pool_to_fill_budget(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "reachability-question",
                "question_type": "reachability-yesno",
                "question": "Is T reachable from S?",
                "needles": ["NEEDLE"],
                "answer": ["NO"],
                "answer_type": "REACHABILITY_YESNO",
                "distracting_content_items": ["paper_index:1"],
            }
        ],
        parameters={"distracting_content_type": "paper-related"},
    )

    result = build_full_prompts.build_full_prompts(
        input_path,
        question_type="reachability-yesno",
        target_context_length=40,
        system_prompt="",
        format_reminder="",
        context_item_prefix="",
        context_item_separator="\n",
        context_tail_margin=0,
        tokenizers=[("words", WordCountingTokenizer())],
        paper_index_distractor_loader=lambda *_args, **_kwargs: (
            "SAFE DISTRACTOR",
        ),
    )

    user_message = result["questions"][0]["user_message"]
    assert user_message.count("SAFE DISTRACTOR") > 1
    assert "NEEDLE" in user_message


def test_build_full_prompts_paper_related_shuffles_final_packed_entries(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Q",
                "needles": ["NEEDLE"],
                "answer": "A",
                "distracting_content_items": [
                    "paper_related_index_1:1",
                    "paper_index:2",
                    "paper_index:3",
                ],
            }
        ],
        parameters={"distracting_content_type": "paper-related"},
    )

    def load_paper_distractor(item, index, **kwargs):
        del index, kwargs
        return (f"PARAGRAPH_{item}",)

    result = build_full_prompts.build_full_prompts(
        input_path,
        target_context_length=20,
        system_prompt="",
        format_reminder="",
        random_seed=3,
        needle_order="forward",
        context_item_prefix="",
        context_item_separator="\n",
        context_tail_margin=0,
        tokenizers=[("words", WordCountingTokenizer())],
        paper_index_distractor_loader=load_paper_distractor,
    )

    user_message = result["questions"][0]["user_message"]
    assert "NEEDLE" in user_message
    assert "PARAGRAPH_paper_related_index_1:1" in user_message
    assert "PARAGRAPH_paper_index:2" in user_message
    assert "PARAGRAPH_paper_index:3" in user_message
    assert user_message.index("PARAGRAPH_paper_index:2") < user_message.index(
        "NEEDLE"
    )


def test_build_full_prompts_paper_related_trims_related_with_bounded_search(
    tmp_path: Path,
) -> None:
    related_items = [f"paper_related_index_1:{index}" for index in range(128)]
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Q",
                "needles": ["NEEDLE"],
                "answer": "A",
                "distracting_content_items": related_items,
            }
        ],
        parameters={"distracting_content_type": "paper-related"},
    )

    def load_paper_distractor(item, index, **kwargs):
        del index, kwargs
        return (f"PARAGRAPH_{item}",)

    tokenizer = WordCountingTokenizer()
    result = build_full_prompts.build_full_prompts(
        input_path,
        target_context_length=20,
        system_prompt="",
        format_reminder="",
        random_seed=3,
        needle_order="forward",
        context_item_prefix="",
        context_item_separator="\n",
        context_tail_margin=0,
        tokenizers=[("words", tokenizer)],
        paper_index_distractor_loader=load_paper_distractor,
    )

    user_message = result["questions"][0]["user_message"]
    included_related_count = sum(
        f"PARAGRAPH_{item}" in user_message for item in related_items
    )
    assert "NEEDLE" in user_message
    assert 0 < included_related_count < len(related_items)
    assert tokenizer.count_calls < 35


def test_build_full_prompts_sets_prompt_parameters_for_distractor_inputs(
    tmp_path: Path,
) -> None:
    distractor_path = tmp_path / "distractors.txt"
    distractor_path.write_text("distractor", encoding="utf-8")
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Who wrote the target paper?",
                "needles": ["needle 1"],
                "answer": "Alice",
                "distracting_content_items": [f"path:{distractor_path}"],
            }
        ],
        metadata={
            "source_questions": "source.json",
            "distracting_content": {"source_dir": "wiki"},
        },
        parameters={
            "num_hops": 1,
            "needle_type": "exact-describe",
            "distracting_content_type": "paper-random",
        },
    )

    result = build_full_prompts.build_full_prompts(
        input_path,
        target_context_length=5000,
        needle_order="backward",
        tokenizers=[("ref", FakeTokenizer())],
    )

    assert result["parameters"] == {
        "num_hops": 1,
        "needle_type": "exact-describe",
        "distracting_content_type": "paper-random",
        "needle_order": "backward",
        "target_context_length": 5000,
    }


def test_build_full_prompts_max_samples_uses_next_questions_after_skips(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Question 1?",
                "needles": ["needle 1"],
                "answer": "  ",
            },
            {
                "qid": "question-2",
                "question": "Question 2?",
                "needles": ["needle 2"],
                "answer": "Answer 2",
            },
            {
                "qid": "question-3",
                "question": "Question 3?",
                "needles": ["needle 3"],
                "answer": "Answer 3",
            },
            {
                "qid": "question-4",
                "question": "Question 4?",
                "needles": ["needle 4"],
                "answer": "Answer 4",
            },
        ],
    )

    result = build_full_prompts.build_full_prompts(
        input_path,
        max_samples=2,
    )

    assert [question["qid"] for question in result["questions"]] == [
        "question-2",
        "question-3",
    ]
    assert result["metadata"]["full_prompts"]["skipped_qids"] == ["question-1"]
    assert result["metadata"]["full_prompts"]["max_samples"] == 2
    assert result["metadata"]["full_prompts"]["total_questions"] == 2


def test_stateful_prompt_workers_refill_the_first_available_slot() -> None:
    first_jobs_started = threading.Barrier(2)
    third_job_started = threading.Event()
    records_lock = threading.Lock()
    owner_threads: dict[int, int] = {}
    contexts_by_position: dict[int, int] = {}
    closed_contexts: list[int] = []
    completed_positions: list[int] = []

    def worker_factory(
        effective_workers: int,
    ) -> tuple[
        build_full_prompts.PromptPlanWorker,
        Callable[[], None],
    ]:
        assert effective_workers == 2
        with records_lock:
            context_id = len(owner_threads)
            owner_threads[context_id] = threading.get_ident()
        owner_thread = threading.get_ident()

        def process_plan(
            plan: dict,
        ) -> build_full_prompts.PromptQuestionResult:
            assert threading.get_ident() == owner_thread
            position = int(plan["index"])
            with records_lock:
                contexts_by_position[position] = context_id
            if position in {0, 1}:
                first_jobs_started.wait(timeout=5)
            if position == 1:
                assert third_job_started.wait(timeout=5)
            elif position == 2:
                third_job_started.set()
            return {"index": position}, None, {}

        def close_worker() -> None:
            assert threading.get_ident() == owner_thread
            with records_lock:
                closed_contexts.append(context_id)

        return process_plan, close_worker

    def collect_result(
        position: int,
        result: build_full_prompts.PromptQuestionResult,
    ) -> bool:
        del result
        completed_positions.append(position)
        return True

    build_full_prompts.run_stateful_prompt_workers(
        [{"index": position} for position in range(4)],
        worker_factory,
        num_workers=2,
        on_result=collect_result,
    )

    assert third_job_started.is_set()
    assert len(owner_threads) == 2
    assert contexts_by_position[2] == contexts_by_position[0]
    assert sorted(completed_positions) == [0, 1, 2, 3]
    assert sorted(closed_contexts) == sorted(owner_threads)


def test_parallel_full_prompts_use_worker_owned_tokenizers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": f"question-{index}",
                "question": f"Question {index}?",
                "needles": [f"needle {index}"],
                "answer": f"Answer {index}",
            }
            for index in range(2)
        ],
    )
    workers_started = threading.Barrier(2)
    tokenizer_ids: set[int] = set()
    records_lock = threading.Lock()

    def record_worker_tokenizer(question, index, **kwargs):
        worker_tokenizers = kwargs["tokenizers"]
        with records_lock:
            tokenizer_ids.add(id(worker_tokenizers[0][1]))
        workers_started.wait(timeout=5)
        return (
            {
                "qid": question["qid"],
                "user_message": f"prompt {index}",
                "answer": question["answer"],
            },
            False,
        )

    monkeypatch.setattr(
        build_full_prompts,
        "process_question",
        record_worker_tokenizer,
    )

    build_full_prompts.build_full_prompts(
        input_path,
        tokenizers=[("ref", FakeTokenizer())],
        num_workers=2,
    )

    assert len(tokenizer_ids) == 2


def test_build_full_prompts_parallel_max_samples_keeps_input_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": f"question-{index}",
                "question": f"Question {index}?",
                "needles": [f"needle {index}"],
                "answer": f"Answer {index}",
            }
            for index in range(4)
        ],
    )
    fourth_job_started = threading.Event()
    processed_indexes: list[int] = []
    records_lock = threading.Lock()

    def reordered_process_question(question, index, **kwargs):
        del kwargs
        with records_lock:
            processed_indexes.append(index)
        if index == 0:
            assert fourth_job_started.wait(timeout=5)
        elif index == 1:
            raise ValueError(
                build_full_prompts.INVALID_ANSWER_ERROR.format(index=index)
            )
        elif index == 3:
            fourth_job_started.set()
        return (
            {
                "qid": question["qid"],
                "user_message": f"prompt {index}",
                "answer": question["answer"],
            },
            False,
        )

    monkeypatch.setattr(
        build_full_prompts,
        "process_question",
        reordered_process_question,
    )

    result = build_full_prompts.build_full_prompts(
        input_path,
        max_samples=2,
        num_workers=3,
    )

    assert sorted(processed_indexes) == [0, 1, 2, 3]
    assert [question["qid"] for question in result["questions"]] == [
        "question-0",
        "question-2",
    ]
    assert result["metadata"]["full_prompts"]["skipped_qids"] == [
        "question-1"
    ]


def test_parallel_full_prompts_match_sequential_output(tmp_path: Path) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": f"question-{index}",
                "question": f"Question {index}?",
                "needles": [f"needle {index}-a", f"needle {index}-b"],
                "answer": f"Answer {index}",
            }
            for index in range(6)
        ],
    )

    sequential = build_full_prompts.build_full_prompts(
        input_path,
        random_seed=7,
        num_workers=1,
    )
    parallel = build_full_prompts.build_full_prompts(
        input_path,
        random_seed=7,
        num_workers=3,
    )

    assert parallel["questions"] == sequential["questions"]
    assert parallel["parameters"] == sequential["parameters"]


def test_build_full_prompts_resumes_from_checkpoint_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Question 1?",
                "needles": ["needle 1"],
                "answer": "Answer 1",
                "answer_type": "TITLE",
            },
            {
                "qid": "question-2",
                "question": "Question 2?",
                "needles": ["needle 2"],
                "answer": "Answer 2",
                "answer_type": "TITLE",
            },
            {
                "qid": "question-3",
                "question": "Question 3?",
                "needles": ["needle 3"],
                "answer": "Answer 3",
                "answer_type": "TITLE",
            },
        ],
    )
    out_path = tmp_path / "full_prompts.json"

    def flaky_process_question(question, index, **kwargs):
        del kwargs
        if index == 1:
            raise RuntimeError("stop after checkpoint")
        return (
            {
                "qid": question["qid"],
                "user_message": f"prompt {index}",
                "answer": question["answer"],
                "answer_type": question["answer_type"],
            },
            False,
        )

    monkeypatch.setattr(
        build_full_prompts,
        "process_question",
        flaky_process_question,
    )
    with pytest.raises(RuntimeError, match="stop after checkpoint"):
        build_full_prompts.build_full_prompts(
            input_path,
            checkpoint_path=out_path,
            max_samples=3,
        )

    partial_payload = orjson.loads(out_path.read_text(encoding="utf-8"))
    assert [question["qid"] for question in partial_payload["questions"]] == ["question-1"]
    assert (
        partial_payload["metadata"]["full_prompts"].pop("question_type")
        == "factoid"
    )
    out_path.write_text(orjson.dumps(partial_payload).decode("utf-8"), encoding="utf-8")

    capsys.readouterr()
    resumed_indexes: list[int] = []

    def resumed_process_question(question, index, **kwargs):
        del kwargs
        resumed_indexes.append(index)
        return (
            {
                "qid": question["qid"],
                "user_message": f"prompt {index}",
                "answer": question["answer"],
                "answer_type": question["answer_type"],
            },
            False,
        )

    monkeypatch.setattr(
        build_full_prompts,
        "process_question",
        resumed_process_question,
    )
    result = build_full_prompts.build_full_prompts(
        input_path,
        checkpoint_path=out_path,
        max_samples=3,
        question_type="factoid",
    )
    stdout = capsys.readouterr().out

    assert resumed_indexes == [1, 2]
    assert [question["qid"] for question in result["questions"]] == [
        "question-1",
        "question-2",
        "question-3",
    ]
    assert "Checkpoint found" in stdout
    assert "Resume: 1/3 question(s) already built." in stdout


@pytest.mark.parametrize(
    "resumed_max_samples",
    [None, 3],
    ids=["max-samples-removed", "max-samples-increased"],
)
def test_build_full_prompts_resumes_when_max_samples_grows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    resumed_max_samples: int | None,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Question 1?",
                "needles": ["needle 1"],
                "answer": "Answer 1",
            },
            {
                "qid": "question-2",
                "question": "Question 2?",
                "needles": ["needle 2"],
                "answer": "Answer 2",
            },
            {
                "qid": "question-3",
                "question": "Question 3?",
                "needles": ["needle 3"],
                "answer": "Answer 3",
            },
        ],
    )
    out_path = tmp_path / "full_prompts.json"

    build_full_prompts.build_full_prompts(
        input_path,
        checkpoint_path=out_path,
        max_samples=1,
    )
    initial_payload = orjson.loads(out_path.read_text(encoding="utf-8"))
    assert [question["qid"] for question in initial_payload["questions"]] == [
        "question-1"
    ]

    capsys.readouterr()
    resumed_indexes: list[int] = []

    def resumed_process_question(question, index, **kwargs):
        del kwargs
        resumed_indexes.append(index)
        return (
            {
                "qid": question["qid"],
                "user_message": f"prompt {index}",
                "answer": question["answer"],
                "answer_type": question.get("answer_type"),
            },
            False,
        )

    monkeypatch.setattr(
        build_full_prompts,
        "process_question",
        resumed_process_question,
    )
    result = build_full_prompts.build_full_prompts(
        input_path,
        checkpoint_path=out_path,
        max_samples=resumed_max_samples,
    )
    stdout = capsys.readouterr().out

    assert resumed_indexes == [1, 2]
    assert [question["qid"] for question in result["questions"]] == [
        "question-1",
        "question-2",
        "question-3",
    ]
    assert result["metadata"]["full_prompts"]["max_samples"] == resumed_max_samples
    assert "Checkpoint found" in stdout
    assert "Resume: 1/3 question(s) already built." in stdout


def test_build_full_prompts_resumes_when_tokenizer_list_changes(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Question 1?",
                "needles": ["needle 1"],
                "answer": "Answer 1",
            }
        ],
    )
    out_path = tmp_path / "full_prompts.json"
    common_kwargs = {
        "checkpoint_path": out_path,
        "target_context_length": 10_000,
        "system_prompt": "SYS",
        "format_reminder": "FMT",
    }

    build_full_prompts.build_full_prompts(
        input_path,
        tokenizers=[("old-tokenizer", FakeTokenizer())],
        **common_kwargs,
    )
    result = build_full_prompts.build_full_prompts(
        input_path,
        tokenizers=[
            ("new-tokenizer", FakeTokenizer()),
            ("another-tokenizer", FakeTokenizer()),
        ],
        **common_kwargs,
    )

    assert result["metadata"]["full_prompts"]["tokenizer_model_slugs"] == [
        "new-tokenizer",
        "another-tokenizer",
    ]


def test_build_full_prompts_resumes_interleaved_completed_qids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Question 1?",
                "needles": ["needle 1"],
                "answer": "Answer 1",
            },
            {
                "qid": "question-2",
                "question": "Question 2?",
                "needles": ["needle 2"],
                "answer": "Answer 2",
            },
        ],
    )
    out_path = tmp_path / "full_prompts.json"
    build_full_prompts.build_full_prompts(input_path, checkpoint_path=out_path)

    input_path.write_text(
        orjson.dumps({
                "metadata": {"source_questions": "source.json"},
                "parameters": {},
                "questions": [
                    {
                        "qid": "question-1",
                        "question": "Question 1?",
                        "needles": ["needle 1"],
                        "answer": "Answer 1",
                    },
                    {
                        "qid": "question-new",
                        "question": "New question?",
                        "needles": ["needle new"],
                        "answer": "Answer new",
                    },
                    {
                        "qid": "question-2",
                        "question": "Question 2?",
                        "needles": ["needle 2"],
                        "answer": "Answer 2",
                    },
                    {
                        "qid": "question-3",
                        "question": "Question 3?",
                        "needles": ["needle 3"],
                        "answer": "Answer 3",
                    },
                ],
            }).decode("utf-8"),
        encoding="utf-8",
    )

    processed_qids: list[str] = []

    def resumed_process_question(question, index, **kwargs):
        del index, kwargs
        processed_qids.append(question["qid"])
        return (
            {
                "qid": question["qid"],
                "user_message": f"prompt {question['qid']}",
                "answer": question["answer"],
                "answer_type": question.get("answer_type"),
            },
            False,
        )

    monkeypatch.setattr(
        build_full_prompts,
        "process_question",
        resumed_process_question,
    )
    result = build_full_prompts.build_full_prompts(input_path, checkpoint_path=out_path)

    assert processed_qids == ["question-new", "question-3"]
    assert [question["qid"] for question in result["questions"]] == [
        "question-1",
        "question-new",
        "question-2",
        "question-3",
    ]


def test_build_full_prompts_resumes_with_skipped_checkpoint_question(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Question 1?",
                "needles": ["needle 1"],
                "answer": "Answer 1",
            },
            {
                "qid": "question-2",
                "question": "Question 2?",
                "needles": ["x" * 40],
                "answer": "Answer 2",
            },
            {
                "qid": "question-3",
                "question": "Question 3?",
                "needles": ["needle 3"],
                "answer": "Answer 3",
            },
        ],
    )
    out_path = tmp_path / "full_prompts.json"

    result = build_full_prompts.build_full_prompts(
        input_path,
        checkpoint_path=out_path,
        target_context_length=80,
        system_prompt="SYS",
        format_reminder="FMT",
        tokenizers=[("ref", FakeTokenizer())],
        context_item_prefix="",
        context_item_separator="\n\n",
    )

    assert [question["qid"] for question in result["questions"]] == [
        "question-1",
        "question-3",
    ]
    assert result["metadata"]["full_prompts"]["skipped_qids"] == ["question-2"]

    payload = orjson.loads(out_path.read_text(encoding="utf-8"))
    assert [question["qid"] for question in payload["questions"]] == [
        "question-1",
        "question-3",
    ]
    assert payload["metadata"]["full_prompts"]["skipped_qids"] == ["question-2"]

    capsys.readouterr()

    def fail_process_question(*args, **kwargs):
        del args, kwargs
        raise AssertionError("resume should not rebuild completed or skipped questions")

    monkeypatch.setattr(
        build_full_prompts,
        "process_question",
        fail_process_question,
    )
    resumed = build_full_prompts.build_full_prompts(
        input_path,
        checkpoint_path=out_path,
        target_context_length=80,
        system_prompt="SYS",
        format_reminder="FMT",
        tokenizers=[("ref", FakeTokenizer())],
        context_item_prefix="",
        context_item_separator="\n\n",
    )
    stdout = capsys.readouterr().out

    assert [question["qid"] for question in resumed["questions"]] == [
        "question-1",
        "question-3",
    ]
    assert resumed["metadata"]["full_prompts"]["skipped_qids"] == ["question-2"]
    assert "Checkpoint found" in stdout
    assert "Resume: 3/3 question(s) already built." in stdout


def test_build_full_prompts_rebuilds_checkpoint_missing_prompt_token_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Question?",
                "needles": ["needle"],
                "answer": "Answer",
            }
        ],
    )
    out_path = tmp_path / "full_prompts.json"
    run_kwargs = {
        "checkpoint_path": out_path,
        "target_context_length": 100,
        "system_prompt": "SYS",
        "format_reminder": "FMT",
        "tokenizers": [("ref", FakeTokenizer())],
    }
    build_full_prompts.build_full_prompts(input_path, **run_kwargs)
    checkpoint = orjson.loads(out_path.read_bytes())
    checkpoint["questions"][0].pop("prompt_without_distractors_tokens")
    out_path.write_bytes(orjson.dumps(checkpoint))

    original_process_question = build_full_prompts.process_question
    processed_indexes: list[int] = []

    def recording_process_question(question, index, **kwargs):
        processed_indexes.append(index)
        return original_process_question(question, index, **kwargs)

    monkeypatch.setattr(
        build_full_prompts,
        "process_question",
        recording_process_question,
    )

    result = build_full_prompts.build_full_prompts(input_path, **run_kwargs)

    assert processed_indexes == [0]
    assert isinstance(
        result["questions"][0]["prompt_without_distractors_tokens"],
        int,
    )


def test_build_full_prompts_rejects_incompatible_checkpoint(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Question 1?",
                "needles": ["needle 1"],
                "answer": "Answer 1",
            }
        ],
    )
    out_path = tmp_path / "full_prompts.json"

    build_full_prompts.build_full_prompts(
        input_path,
        checkpoint_path=out_path,
        max_samples=1,
    )

    with pytest.raises(ValueError, match="different full-prompt settings") as error:
        build_full_prompts.build_full_prompts(
            input_path,
            checkpoint_path=out_path,
            max_samples=1,
            question_last=True,
        )

    assert "question_last: saved=false; current=true" in str(error.value)


def test_build_full_prompts_rejects_checkpoint_without_overlapping_qids(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Question 1?",
                "needles": ["needle 1"],
                "answer": "Answer 1",
            },
            {
                "qid": "question-2",
                "question": "Question 2?",
                "needles": ["needle 2"],
                "answer": "Answer 2",
            },
        ],
    )
    out_path = tmp_path / "full_prompts.json"
    run_settings = build_full_prompts.build_full_prompts_run_settings(
        question_last=False,
        target_context_length=None,
        tokenizers=None,
        needle_order=build_full_prompts.DEFAULT_NEEDLE_ORDER,
        random_seed=None,
        context_item_prefix=build_full_prompts.DEFAULT_CONTEXT_ITEM_PREFIX,
        context_item_separator=build_full_prompts.DEFAULT_CONTEXT_ITEM_SEPARATOR,
        context_tail_margin=build_full_prompts.DEFAULT_CONTEXT_TAIL_MARGIN,
        max_samples=1,
    )
    payload = {
        "metadata": build_full_prompts.build_base_full_prompts_metadata(
            input_path,
            run_settings=run_settings,
            total_questions=1,
            synthesized_qids=0,
        ),
        "parameters": {},
        "questions": [
            {
                "qid": "different-question",
                "user_message": "prompt",
                "answer": "Answer",
            }
        ],
    }
    out_path.write_text(orjson.dumps(payload).decode("utf-8"), encoding="utf-8")

    with pytest.raises(ValueError, match="different full-prompt settings") as error:
        build_full_prompts.build_full_prompts(
            input_path,
            checkpoint_path=out_path,
        )

    assert (
        'checkpoint_qids_not_in_current_input: saved=["different-question"]; '
        "current=[]" in str(error.value)
    )


def test_load_prompt_packing_tokenizers_skips_models_below_target_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingAutoTokenizer:
        def __init__(self) -> None:
            self.loaded_model_ids: list[tuple[str, bool]] = []

        def from_pretrained(
            self,
            model_id: str,
            trust_remote_code: bool = False,
        ) -> FakeTokenizer:
            self.loaded_model_ids.append((model_id, trust_remote_code))
            return FakeTokenizer()

    import src.llm.model_info as model_info

    recording_tokenizer = RecordingAutoTokenizer()
    monkeypatch.setattr(build_full_prompts, "AutoTokenizer", recording_tokenizer)
    monkeypatch.setattr(
        model_info,
        "DEFAULT_LLM_MANAGER",
        SimpleNamespace(
            models={
                "short": SimpleNamespace(
                    hf_model_id="short-hf-id",
                    native_context_length=32000,
                ),
                "wide": SimpleNamespace(
                    hf_model_id="wide-hf-id",
                    native_context_length=128000,
                ),
                "missing": SimpleNamespace(
                    hf_model_id="",
                    native_context_length=256000,
                ),
            }
        ),
    )

    tokenizers = build_full_prompts.load_prompt_packing_tokenizers(target_context_length=50000)

    assert [slug for slug, _tokenizer in tokenizers] == ["wide"]
    assert recording_tokenizer.loaded_model_ids == [("wide-hf-id", True)]


def test_load_prompt_packing_tokenizers_uses_long_context_hf_fallback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class MappingAutoTokenizer:
        def __init__(self, tokenizers_by_model_id: dict[str, ComparableTokenizer]) -> None:
            self.loaded_model_ids: list[tuple[str, bool]] = []
            self.tokenizers_by_model_id = tokenizers_by_model_id

        def from_pretrained(
            self,
            model_id: str,
            trust_remote_code: bool = False,
        ) -> ComparableTokenizer:
            self.loaded_model_ids.append((model_id, trust_remote_code))
            return self.tokenizers_by_model_id[model_id]

    import src.llm.model_info as model_info

    mapping_tokenizer = MappingAutoTokenizer(
        {
            "qwen-hf-id": ComparableTokenizer(
                vocab={"qwen": 0},
                tokens_by_id=["qwen"],
            ),
            "gemma-hf-id": ComparableTokenizer(
                vocab={"gemma": 0},
                tokens_by_id=["gemma"],
            ),
        }
    )
    fake_tiktoken = FakeTiktoken()
    monkeypatch.setattr(build_full_prompts, "AutoTokenizer", mapping_tokenizer)
    monkeypatch.setattr(build_full_prompts, "tiktoken", fake_tiktoken)
    monkeypatch.setattr(
        model_info,
        "DEFAULT_LLM_MANAGER",
        SimpleNamespace(
            models={
                "qwen": SimpleNamespace(
                    hf_model_id="qwen-hf-id",
                    native_context_length=262144,
                ),
                "gemma": SimpleNamespace(
                    hf_model_id="gemma-hf-id",
                    native_context_length=262144,
                ),
                "gpt-openai": SimpleNamespace(
                    api_model_name="openai/gpt-5.4",
                    hf_model_id="",
                    native_context_length=1050000,
                    provider="OpenAI",
                ),
            }
        ),
    )

    tokenizers = build_full_prompts.load_prompt_packing_tokenizers(
        target_context_length=500000,
    )
    stdout = capsys.readouterr().out

    assert [slug for slug, _tokenizer in tokenizers] == [
        "qwen",
        "gemma",
        "gpt-openai",
    ]
    assert mapping_tokenizer.loaded_model_ids == [
        ("qwen-hf-id", True),
        ("gemma-hf-id", True),
    ]
    assert fake_tiktoken.model_names == ["gpt-5.4"]
    assert "using local Hugging Face tokenizers below the target" in stdout


def test_load_prompt_packing_tokenizers_skips_unloadable_models(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class SelectiveAutoTokenizer:
        def __init__(self) -> None:
            self.loaded_model_ids: list[str] = []

        def from_pretrained(
            self,
            model_id: str,
            trust_remote_code: bool = False,
        ) -> FakeTokenizer:
            del trust_remote_code
            self.loaded_model_ids.append(model_id)
            if model_id == "broken-hf-id":
                raise AttributeError("broken tokenizer metadata")
            return FakeTokenizer()

    import src.llm.model_info as model_info

    selective_tokenizer = SelectiveAutoTokenizer()
    monkeypatch.setattr(build_full_prompts, "AutoTokenizer", selective_tokenizer)
    monkeypatch.setattr(
        model_info,
        "DEFAULT_LLM_MANAGER",
        SimpleNamespace(
            models={
                "broken": SimpleNamespace(
                    hf_model_id="broken-hf-id",
                    native_context_length=128000,
                ),
                "wide": SimpleNamespace(
                    hf_model_id="wide-hf-id",
                    native_context_length=128000,
                ),
            }
        ),
    )

    tokenizers = build_full_prompts.load_prompt_packing_tokenizers(target_context_length=50000)
    stdout = capsys.readouterr().out

    assert [slug for slug, _tokenizer in tokenizers] == ["wide"]
    assert selective_tokenizer.loaded_model_ids == ["broken-hf-id", "wide-hf-id"]
    assert "Skipping Hugging Face tokenizer for prompt packing" in stdout
    assert "broken-hf-id" in stdout


def test_load_prompt_packing_tokenizers_deduplicates_identical_tokenizers(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class MappingAutoTokenizer:
        def __init__(self, tokenizers_by_model_id: dict[str, ComparableTokenizer]) -> None:
            self.tokenizers_by_model_id = tokenizers_by_model_id

        def from_pretrained(
            self,
            model_id: str,
            trust_remote_code: bool = False,
        ) -> ComparableTokenizer:
            del trust_remote_code
            return self.tokenizers_by_model_id[model_id]

    import src.llm.model_info as model_info

    monkeypatch.setattr(
        build_full_prompts,
        "AutoTokenizer",
        MappingAutoTokenizer(
            {
                "first-hf-id": ComparableTokenizer(
                    vocab={"hello": 0, "world": 1},
                    tokens_by_id=["hello", "world"],
                ),
                "duplicate-hf-id": ComparableTokenizer(
                    vocab={"hello": 0, "world": 1},
                    tokens_by_id=["hello", "world"],
                ),
                "distinct-hf-id": ComparableTokenizer(
                    vocab={"hello": 0, "world": 1},
                    tokens_by_id=["hello", "planet"],
                ),
            }
        ),
    )
    monkeypatch.setattr(
        model_info,
        "DEFAULT_LLM_MANAGER",
        SimpleNamespace(
            models={
                "first": SimpleNamespace(
                    hf_model_id="first-hf-id",
                    native_context_length=128000,
                ),
                "duplicate": SimpleNamespace(
                    hf_model_id="duplicate-hf-id",
                    native_context_length=128000,
                ),
                "distinct": SimpleNamespace(
                    hf_model_id="distinct-hf-id",
                    native_context_length=128000,
                ),
            }
        ),
    )

    tokenizers = build_full_prompts.load_prompt_packing_tokenizers(target_context_length=50000)
    stdout = capsys.readouterr().out

    assert [slug for slug, _tokenizer in tokenizers] == ["first", "distinct"]
    assert (
        "Prompt-packing tokenizers: first, distinct (1 duplicate(s) collapsed)" in stdout
    )


def test_load_prompt_packing_tokenizers_loads_only_the_named_slugs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_ids: list[str] = []

    class RecordingAutoTokenizer:
        def from_pretrained(
            self,
            model_id: str,
            trust_remote_code: bool = False,
        ) -> ComparableTokenizer:
            del trust_remote_code
            loaded_ids.append(model_id)
            return ComparableTokenizer(
                vocab={"hello": 0, "world": 1},
                tokens_by_id=["hello", model_id],
            )

    import src.llm.model_info as model_info

    monkeypatch.setattr(build_full_prompts, "AutoTokenizer", RecordingAutoTokenizer())
    monkeypatch.setattr(
        model_info,
        "DEFAULT_LLM_MANAGER",
        SimpleNamespace(
            models={
                slug: SimpleNamespace(
                    hf_model_id=f"{slug}-hf-id",
                    native_context_length=128000,
                )
                for slug in ("first", "duplicate", "distinct")
            }
        ),
    )

    tokenizers = build_full_prompts.load_prompt_packing_tokenizers(
        target_context_length=50000,
        only_slugs=["first", "distinct"],
    )

    assert [slug for slug, _tokenizer in tokenizers] == ["first", "distinct"]
    assert loaded_ids == ["first-hf-id", "distinct-hf-id"]


def test_normalize_openai_input_token_counter_model_name_strips_provider_prefix() -> None:
    assert (
        build_full_prompts.normalize_openai_input_token_counter_model_name("openai/gpt-5.4")
        == "gpt-5.4"
    )
    assert (
        build_full_prompts.normalize_openai_input_token_counter_model_name("gpt-5.4") == "gpt-5.4"
    )


def test_openai_input_token_counter_counts_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_calls: list[dict[str, object]] = []

    class FakeInputTokensEndpoint:
        def count(self, *, model: str, input: list[dict[str, str]]) -> SimpleNamespace:
            captured_calls.append({"model": model, "input": input})
            return SimpleNamespace(input_tokens=123)

    class FakeResponsesApi:
        def __init__(self) -> None:
            self.input_tokens = FakeInputTokensEndpoint()

    class FakeOpenAIClient:
        def __init__(self) -> None:
            self.responses = FakeResponsesApi()

    monkeypatch.setattr(build_full_prompts, "OpenAI", FakeOpenAIClient)

    counter = build_full_prompts.OpenAIInputTokenCounter("gpt-5.4")
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]

    assert counter.count_messages(messages) == 123
    assert captured_calls == [{"model": "gpt-5.4", "input": messages}]


def test_fake_openai_tokenizer_falls_back_to_character_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenTiktoken:
        def encoding_for_model(self, model_name: str) -> None:
            del model_name
            raise ConnectionError("offline")

    monkeypatch.setattr(build_full_prompts, "tiktoken", BrokenTiktoken())

    tokenizer = build_full_prompts.FakeOpenAITokenizer("gpt-5.4")

    assert tokenizer.prompt_packing_identity == (
        "fake_openai_tokenizer",
        "character_fallback",
    )
    assert len(tokenizer.encode("abcd")) == 4


def test_fake_openai_tokenizer_treats_reserved_markers_as_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []

    class RecordingEncoding:
        name = "recording"

        def encode(
            self,
            text: str,
            *,
            disallowed_special: tuple[str, ...],
        ) -> list[str]:
            calls.append((text, disallowed_special))
            return list(text)

    class RecordingTiktoken:
        def encoding_for_model(self, model_name: str) -> RecordingEncoding:
            del model_name
            return RecordingEncoding()

    monkeypatch.setattr(build_full_prompts, "tiktoken", RecordingTiktoken())
    tokenizer = build_full_prompts.FakeOpenAITokenizer("gpt-5.4")

    text = "ordinary <|endoftext|> and <|endofprompt|> text"
    assert len(tokenizer.encode(text)) == len(text)
    assert calls == [(text, ())]


def test_measure_tokenized_chat_length_uses_message_counter() -> None:
    class CountingTokenizer:
        def __init__(self) -> None:
            self.calls: list[list[dict[str, str]]] = []

        def count_messages(self, messages: list[dict[str, str]]) -> int:
            self.calls.append(messages)
            return 77

    tokenizer = CountingTokenizer()
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]

    measured_length = build_full_prompts.measure_tokenized_chat_length(
        tokenizer,
        messages,
    )

    assert measured_length == 77
    assert tokenizer.calls == [messages]


@pytest.mark.parametrize(
    "target_context_length",
    [500_001, 100_000],
    ids=["at-threshold", "below-old-threshold"],
)
def test_load_prompt_packing_tokenizers_includes_fake_openai_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    target_context_length: int,
) -> None:
    class RecordingAutoTokenizer:
        def __init__(self) -> None:
            self.loaded_model_ids: list[tuple[str, bool]] = []

        def from_pretrained(
            self,
            model_id: str,
            trust_remote_code: bool = False,
        ) -> FakeTokenizer:
            self.loaded_model_ids.append((model_id, trust_remote_code))
            return FakeTokenizer()

    import src.llm.model_info as model_info

    recording_tokenizer = RecordingAutoTokenizer()
    fake_tiktoken = FakeTiktoken()
    monkeypatch.setattr(build_full_prompts, "AutoTokenizer", recording_tokenizer)
    monkeypatch.setattr(build_full_prompts, "tiktoken", fake_tiktoken)
    monkeypatch.setattr(
        model_info,
        "DEFAULT_LLM_MANAGER",
        SimpleNamespace(
            models={
                "hf-wide": SimpleNamespace(
                    hf_model_id="wide-hf-id",
                    native_context_length=10_000_000,
                ),
                "gpt-openai": SimpleNamespace(
                    api_model_name="openai/gpt-5.4",
                    hf_model_id="",
                    native_context_length=10_000_000,
                    provider="OpenAI",
                ),
            }
        ),
    )

    tokenizers = build_full_prompts.load_prompt_packing_tokenizers(
        target_context_length=target_context_length
    )

    assert [slug for slug, _tokenizer in tokenizers] == ["hf-wide", "gpt-openai"]
    assert recording_tokenizer.loaded_model_ids == [("wide-hf-id", True)]
    assert isinstance(tokenizers[1][1], build_full_prompts.FakeOpenAITokenizer)
    assert tokenizers[1][1].model_name == "gpt-5.4"
    assert fake_tiktoken.model_names == ["gpt-5.4"]


def test_same_tokenizer_uses_vocab_and_token_ids() -> None:
    tokenizer1 = ComparableTokenizer(
        vocab={"hello": 0, "world": 1},
        tokens_by_id=["hello", "world"],
    )
    tokenizer2 = ComparableTokenizer(
        vocab={"hello": 0, "world": 1},
        tokens_by_id=["hello", "world"],
    )
    tokenizer3 = ComparableTokenizer(
        vocab={"hello": 0, "world": 1},
        tokens_by_id=["hello", "planet"],
    )

    assert build_full_prompts.same_tokenizer(tokenizer1, tokenizer2) is True
    assert build_full_prompts.same_tokenizer(tokenizer1, tokenizer3) is False


def test_measure_tokenized_chat_length_falls_back_when_chat_template_is_missing() -> None:
    tokenizer = TemplateLessTokenizer(prompt_offset=4)
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]

    measured_length = build_full_prompts.measure_tokenized_chat_length(
        tokenizer,
        messages,
    )

    assert measured_length == len(
        tokenizer.encode(
            build_full_prompts.render_messages_for_token_count(messages),
            add_special_tokens=True,
        )
    )
    assert tokenizer.prompt_packing_mode == "plain_encode"


def test_build_full_prompts_records_needle_and_distractor_token_stats(
    tmp_path: Path,
) -> None:
    required_needles = ["needle one", "needle two"]
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "What is the title?",
                "needles": required_needles,
                "answer": "A title",
                "distracting_content_items": ["repeat:distractor sentence"],
            }
        ],
    )

    result = build_full_prompts.build_full_prompts(
        input_path,
        target_context_length=200,
        system_prompt="SYS",
        format_reminder="FMT",
        tokenizers=[
            ("ref", FakeTokenizer()),
            ("longer", FakeTokenizer(prompt_offset=7)),
        ],
        context_item_prefix="",
        context_item_separator="\n\n",
        context_tail_margin=0,
    )

    output_question = result["questions"][0]
    user_message = output_question["user_message"]
    context_text = user_message.split("# Context\n\n", 1)[1].rsplit("\n\nFMT", 1)[0]
    context_items = context_text.split("\n\n")
    needle_items = [item for item in context_items if item.startswith("needle ")]
    distractor_items = [item for item in context_items if item not in needle_items]

    # FakeTokenizer encodes one token per character, and the last distractor may
    # be truncated to fill the budget, so compare against the packed items.
    assert len(needle_items) == 2
    assert len(distractor_items) >= 2
    assert output_question["needle_items"] == len(needle_items)
    assert output_question["needle_tokens"] == sum(len(item) for item in needle_items)
    assert output_question["distractor_items"] == len(distractor_items)
    assert output_question["distractor_tokens"] == sum(
        len(item) for item in distractor_items
    )
    prompt_without_distractors = build_full_prompts.append_format_reminder(
        build_full_prompts.build_user_message(
            "What is the title?",
            build_full_prompts.build_context_text(
                required_needles,
                context_item_prefix="",
                context_item_separator="\n\n",
            ),
            False,
        ),
        "FMT",
    )
    assert (
        output_question["prompt_without_distractors_tokens"]
        == len(f"<system>SYS<user>{prompt_without_distractors}") + 7
    )


def test_build_full_prompts_omits_token_stats_without_tokenizers(
    tmp_path: Path,
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "What is the title?",
                "needles": ["needle one"],
                "answer": "A title",
                "distracting_content_items": [],
            }
        ],
    )

    result = build_full_prompts.build_full_prompts(input_path)

    output_question = result["questions"][0]
    assert output_question["needle_items"] == 1
    assert output_question["distractor_items"] == 0
    assert "needle_tokens" not in output_question
    assert "distractor_tokens" not in output_question
    assert "prompt_without_distractors_tokens" not in output_question


def test_print_summary_reports_needle_and_distractor_usage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata = {
        "full_prompts": {
            "question_last": False,
            "total_questions": 2,
            "synthesized_qids": 0,
            "skipped_qids": [],
        },
        "source_questions": "source.json",
    }
    questions = [
        {
            "needle_items": 4,
            "needle_tokens": 400,
            "distractor_items": 10,
            "distractor_tokens": 500,
        },
        {
            "needle_items": 6,
            "needle_tokens": 800,
            "distractor_items": 20,
            "distractor_tokens": 1_000,
        },
    ]

    build_full_prompts.print_summary(
        Path("prompts.json"),
        metadata,
        {"needle_order": "random", "target_context_length": 100},
        questions,
    )

    stdout = capsys.readouterr().out
    assert "  Measured       : 2 prompt(s)" in stdout
    assert (
        "  Needles        : 5.0 +/- 1.0 items/prompt (min=4, max=6)"
        " | 120.0 tokens/needle" in stdout
    )
    assert (
        "  Distractors    : 15.0 +/- 5.0 items/prompt (min=10, max=20)"
        " | 50.0 tokens/distractor" in stdout
    )


def test_print_summary_renders_caller_rows_inside_its_block(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A caller's rows join this block instead of trailing prefixed lines."""
    metadata = {
        "full_prompts": {
            "question_last": False,
            "total_questions": 1,
            "synthesized_qids": 0,
            "skipped_qids": [],
        },
        "source_questions": "source.json",
    }

    build_full_prompts.print_summary(
        Path("prompts.json"),
        metadata,
        {"needle_order": "random", "target_context_length": 100},
        None,
        cli_name="combined_stage",
        extra_rows=[("Token budget", "10 used | 90 free")],
    )

    stdout = capsys.readouterr().out
    assert "[combined_stage]" in stdout
    assert "[build_full_prompts]" not in stdout
    assert "  Token budget   : 10 used | 90 free" in stdout


def test_print_summary_reports_measured_prompts_per_item_kind(
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata = {
        "full_prompts": {
            "question_last": False,
            "total_questions": 2,
            "synthesized_qids": 0,
            "skipped_qids": [],
        },
        "source_questions": "source.json",
    }
    questions = [
        {"needle_items": 4, "distractor_items": 10},
        {"distractor_items": 20},
    ]

    build_full_prompts.print_summary(
        Path("prompts.json"),
        metadata,
        {"needle_order": "random", "target_context_length": 100},
        questions,
    )

    stdout = capsys.readouterr().out
    assert "  Measured       : needles=1, distractors=2 prompt(s)" in stdout


def test_build_full_prompts_skip_line_reports_needle_size(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = build_input_file(
        tmp_path,
        [
            {
                "qid": "question-1",
                "question": "Q?",
                "needles": ["needle one", "needle two", "needle three"],
                "answer": "A",
            }
        ],
    )

    result = build_full_prompts.build_full_prompts(
        input_path,
        target_context_length=10,
        system_prompt="SYS",
        format_reminder="FMT",
        tokenizers=[("ref", FakeTokenizer())],
        context_item_prefix="",
        context_item_separator="\n\n",
    )

    stdout = capsys.readouterr().out
    assert result["questions"] == []
    # FakeTokenizer encodes one token per character, so the rendered prompt is
    # longer than the needle text it carries.
    needle_chars = len("needle one") + len("needle two") + len("needle three")
    skip_line = next(line for line in stdout.splitlines() if line.startswith("SKIP "))
    assert "SKIP index=0 qid=question-1 reason=needles-exceed-target" in skip_line
    assert "needles=3" in skip_line
    assert "target_context_length=10" in skip_line
    prompt_tokens = int(
        skip_line.split("prompt_tokens=")[1].split()[0].replace(",", "")
    )
    assert prompt_tokens > needle_chars
