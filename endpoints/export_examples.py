#!/usr/bin/env python3
"""Export the examples of every experiment under an output root to one JSONL file.

Each experiment.jsonl (written by create_dataset.vN.py or the none-option
conversion) lists only qids; the prompts live in the shared pool named by its
.settings.json. This script finds every experiment.jsonl below --root, skips the
growth-parameter search (any folder named growth_search and any components-only
example), looks each qid up in its pool, and writes one line per unique example.
An example used by several experiments is written once; its "experiments" field
lists all of them. Neither the pools nor the experiments are modified.

Example (activate the project environment first)::

    python endpoints/export_examples.py --root output/experiments.v1 \
        --output output/experiments.v1/examples.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.questions.dataset_pool import (
    REPO_ROOT,
    ExamplePool,
    portable_path,
    read_experiment_qids,
    settings_path_for,
)

EXCLUDED_FOLDER = "growth_search"
REACHABILITY_TASKS = {
    "reachability-yesno": "yes-no",
    "reachability-multiple": "multiple-choice",
    "reachability-counting": "counting",
}


def find_experiments(root: Path) -> list[Path]:
    """Every experiment.jsonl below root, outside the growth-parameter search."""
    return sorted(
        path
        for path in root.rglob("experiment.jsonl")
        if EXCLUDED_FOLDER not in path.relative_to(root).parts
    )


def experiment_name(root: Path, experiment_path: Path) -> str:
    """The vary=<name> folder, plus the none-option levels below it."""
    parts = experiment_path.parent.relative_to(root).parts
    for index, part in enumerate(parts):
        if part.startswith("vary="):
            return "/".join([part.removeprefix("vary="), *parts[index + 1 :]])
    return "/".join(parts)


def export_record(record: dict[str, Any]) -> dict[str, Any]:
    """The fields kept for one pool example."""
    exported = {
        "qid": record["qid"],
        "reachability_task": REACHABILITY_TASKS[record["question_type"]],
        "system_prompt": record["system_message"],
        "user_prompt": record["user_message"],
        "question": record["question"],
        "answer": record["answer"][0],
        "question_type": record["question_type"],
        "answer_type": record["answer_type"],
        "config": record["config"],
    }
    options = record.get("reachability", {}).get("options")
    if options is not None:
        exported["options"] = [
            {"label": option["label"], "text": option["text"]} for option in options
        ]
    # Converted none-option examples keep the source counts under another name.
    token_counts = record.get("token_counts") or record.get(
        "source_prompt_measurements", {}
    ).get("token_counts")
    exported["token_counts"] = token_counts
    exported["generator_version"] = record.get("generator_version")
    return exported


def write_jsonl_atomically(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            for record in records:
                handle.write(orjson.dumps(record) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def export_examples(root: Path, output: Path) -> list[dict[str, Any]]:
    """Write the examples of every experiment below root; return them."""
    experiments = find_experiments(root)
    if not experiments:
        raise ValueError(f"No experiment.jsonl found below {root}.")
    # Experiments in the same pool share one scan of it.
    by_pool: dict[Path, list[tuple[Path, dict[str, Any]]]] = {}
    for experiment_path in experiments:
        settings_path = settings_path_for(experiment_path)
        if not settings_path.exists():
            raise ValueError(f"{settings_path} is missing.")
        identity = orjson.loads(settings_path.read_bytes())["identity"]
        pool_path = REPO_ROOT / identity["pool"]
        if not pool_path.exists():
            raise ValueError(f"{experiment_path} uses pool {pool_path}, which is missing.")
        by_pool.setdefault(pool_path, []).append((experiment_path, identity))

    exported: dict[str, dict[str, Any]] = {}
    for pool_path, members in by_pool.items():
        print(f"Indexing {portable_path(pool_path)} ...", flush=True)
        with pool_path.open("rb") as handle:
            pool = ExamplePool(handle)
            pool.scan(repair=False)
            for experiment_path, identity in members:
                name = experiment_name(root, experiment_path)
                qids = read_experiment_qids(experiment_path, repair=False)
                missing = [qid for qid in qids if qid not in pool.offsets]
                if missing:
                    raise ValueError(
                        f"{experiment_path}: {len(missing)} qids are missing from the pool."
                    )
                skipped = 0
                for slot, qid in enumerate(tqdm(qids, desc=name, unit="ex")):
                    if qid not in exported:
                        record = pool.get(qid)
                        if record["config"].get("components_only"):
                            skipped += 1
                            continue
                        exported[qid] = {**export_record(record), "experiments": []}
                    exported[qid]["experiments"].append(
                        {
                            "name": name,
                            "context_length": identity["context_length"],
                            "random_seed": identity["random_seed"],
                            "slot": slot,
                            "path": portable_path(experiment_path),
                        }
                    )
                if skipped:
                    print(f"  skipped {skipped} components-only examples in {name}")
    records = list(exported.values())
    write_jsonl_atomically(output, records)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/experiments.v1"),
        help="Output root of one experiment version (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL file to write (default: <root>/examples.jsonl).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or args.root / "examples.jsonl"
    records = export_examples(args.root, output)
    counts: dict[str, int] = {}
    for record in records:
        for experiment in record["experiments"]:
            key = f"{experiment['name']} @ {experiment['context_length']:,}"
            counts[key] = counts.get(key, 0) + 1
    for key, count in sorted(counts.items()):
        print(f"  {key}: {count} examples")
    print(f"Wrote {len(records)} unique examples to {output}")


if __name__ == "__main__":
    main()
