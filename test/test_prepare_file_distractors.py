from pathlib import Path

from prepare_file_distractors import copy_code, write_text_chunks


def test_text_chunks_preserve_order_and_discard_incomplete_tail(tmp_path: Path):
    rows = [
        {"text": "\n".join(["", *[f"first {i}" for i in range(1500)]])},
        {"text": "\n".join(f"second {i}" for i in range(501))},
    ]
    assert write_text_chunks(rows, tmp_path, "essays", 5000) == 1
    files = list(tmp_path.iterdir())
    assert [path.name for path in files] == ["paulgraham_0_1_2000.txt"]
    lines = files[0].read_text().splitlines()
    assert len(lines) == 2000
    assert lines[0] == "first 0" and lines[-1] == "second 499"


def test_code_filters_by_size_and_removes_blank_lines(tmp_path: Path):
    source, output = tmp_path / "linux", tmp_path / "code"
    (source / "drivers").mkdir(parents=True)
    output.mkdir()
    (source / "drivers" / "example.c").write_text("int x;\n\n" * 1000)
    (source / "short.h").write_text("int x;\n" * 999)
    (source / "long.cpp").write_text("int x;\n" * 2001)
    (source / "ignored.txt").write_text("text\n" * 1500)
    assert copy_code(source, output) == 1
    assert (output / "drivers-example.c").read_text() == "int x;\n" * 1000
