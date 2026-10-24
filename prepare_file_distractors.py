#!/usr/bin/env python3
"""Prepare the code, essay, and book distractors used by dataset v1.

The preprocessing follows the original preparation notebooks: code files
have 1,000–2,000 nonblank lines; essays and books form 2,000-line chunks.
The final incomplete text chunk is discarded. Existing output files are
never mixed with a new build. Downloaded corpora remain subject to their
source licenses.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path


def text_lines(rows: Iterable[dict]) -> Iterator[tuple[int, str]]:
    for index, row in enumerate(rows):
        for line in row["text"].splitlines():
            if line.strip():
                yield index, line


def write_text_chunks(rows: Iterable[dict], output: Path, kind: str, count: int) -> int:
    lines = text_lines(rows)
    written = 0
    for index in range(count):
        chunk = list(islice(lines, 2000))
        if len(chunk) < 2000:
            break
        start, end = index * 2000 + 1, (index + 1) * 2000
        if kind == "essays":
            name = f"paulgraham_{chunk[0][0]}_{start}_{end}.txt"
        else:
            name = f"pg19_train_index_{start}_{end}.txt"
        (output / name).write_text(
            "\n".join(line for _, line in chunk) + "\n", encoding="utf-8"
        )
        written += 1
    return written


def copy_code(source: Path, output: Path) -> int:
    if not source.is_dir():
        raise ValueError(f"Code source directory does not exist: {source}")
    written = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if ".git" in relative.parts or path.suffix.lower() not in {
            ".c",
            ".h",
            ".cpp",
            ".hpp",
        }:
            continue
        if not path.is_file():
            continue
        data = path.read_bytes()
        nonblank = sum(
            bool(line.strip())
            for line in data.decode("utf-8", errors="ignore").splitlines()
        )
        if 1000 <= nonblank <= 2000:
            (output / "-".join(relative.parts)).write_bytes(
                b"".join(
                    line for line in data.splitlines(keepends=True) if line.strip()
                )
            )
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["code", "essays", "books"])
    parser.add_argument("--code-source", type=Path, help="Local Linux source checkout.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-files", type=int, default=5000)
    parser.add_argument(
        "--revision", help="Hugging Face dataset revision, if available."
    )
    args = parser.parse_args()
    if args.num_files < 1:
        parser.error("--num-files must be positive")
    if args.kind == "code" and args.code_source is None:
        parser.error("code requires --code-source")
    output = args.output or Path("resources/distracting_content") / args.kind
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.kind == "code":
        written = copy_code(args.code_source, output)
    else:
        from datasets import load_dataset

        name = "sgoel9/paul_graham_essays" if args.kind == "essays" else "emozilla/pg19"
        rows = load_dataset(
            name, split="train", streaming=args.kind == "books", revision=args.revision
        )
        if args.kind == "books":
            rows = rows.shuffle(seed=42, buffer_size=10000)
        written = write_text_chunks(rows, output, args.kind, args.num_files)
    if not written:
        raise ValueError("No complete distractor files were produced")
    print(f"Wrote {written} files to {output}")


if __name__ == "__main__":
    main()
