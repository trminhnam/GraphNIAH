"""Save paper-degree distributions of a create_dataset growth-parameter search.

Reads every candidate folder that a growth_search.1_create_data.sh script made
under --search-root: theta=<theta>_discount=<discount> for create_dataset.v2.py
(--parameters theta-discount) or alpha=<alpha> for create_dataset.v1.py
(--parameters alpha). Each folder has its own pool.jsonl and
vary=<factor>/experiment.jsonl. The script measures paper degrees in each
example's own components and writes one file per question format and
candidate, in the format the degree search reads:

    <output root>/question_format=<format>/graph_statistics/
        sample_paper_degree_distribution.<name>=<value>[_<name>=<value>].json

In-, out-, and total degree are all saved, because the search checks all three;
the in-degree search endpoints score in-degree only.

Run from the repository root (activate the niah environment first)::

    python endpoints/calculate_sample_paper_degree_distribution_v2.py \
        --search-root output/experiments.v2/growth_search/cutoff_date=2026-04-01/hops=10-22-46-94/context_length=200k/seed=1
    python endpoints/calculate_sample_paper_degree_distribution_v2.py --parameters alpha \
        --search-root output/experiments.v1/growth_search/cutoff_date=2026-04-01/hops=10-22-46-94/context_length=200k/seed=1
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.graph.alpha_degree_search import V2_EXPERIMENTS
from src.graph.sample_degree_distribution import (
    aggregate_sample_degrees,
    calculate_graph_degrees,
)
from src.questions.dataset_pool import ExamplePool, read_experiment_qids

# The growth settings one search candidate fixes, by --parameters choice.
PARAMETER_SETS = {"theta-discount": ("theta", "discount"), "alpha": ("alpha",)}
FOLDER_BY_FORMAT = {
    folder.split("=", 1)[1]: folder for folder in V2_EXPERIMENTS.values()
}


def candidate_label(names: Sequence[str], values: Sequence[float]) -> str:
    return "_".join(f"{name}={value:g}" for name, value in zip(names, values))


def candidate_folders(
    search_root: Path,
    names: Sequence[str],
    grids: Sequence[Sequence[float] | None],
) -> list[tuple[tuple[float, ...], Path]]:
    """(values, folder) for every requested candidate, sorted.

    A grid of None means every value found for that name.
    """
    pattern = re.compile(
        "^" + "_".join(f"{name}=(?P<p{i}>[^_]+)" for i, name in enumerate(names)) + "$"
    )
    found: dict[tuple[float, ...], Path] = {}
    for path in sorted(search_root.glob("_".join(f"{name}=*" for name in names))):
        match = pattern.match(path.name)
        if not match or not path.is_dir():
            continue
        key = tuple(float(match[f"p{i}"]) for i in range(len(names)))
        if key in found:
            raise ValueError(f"Two folders for the same candidate: {found[key]} and {path}")
        found[key] = path
    if all(grid is None for grid in grids):
        wanted = sorted(found)
    else:
        wanted = list(
            itertools.product(
                *(
                    grid if grid is not None else sorted({key[i] for key in found})
                    for i, grid in enumerate(grids)
                )
            )
        )
    missing = [key for key in wanted if key not in found]
    if missing:
        raise FileNotFoundError(
            f"Missing candidate folders in {search_root}: "
            + ", ".join(candidate_label(names, key) for key in missing)
        )
    if not wanted:
        raise FileNotFoundError(
            f"No {'_'.join(f'{name}=*' for name in names)} folders in {search_root}"
        )
    return [(key, found[key]) for key in wanted]


def load_records(folder: Path, vary: str) -> list[dict[str, Any]]:
    """The experiment's examples, read from the folder's own pool (never modified)."""
    experiment = folder / f"vary={vary}" / "experiment.jsonl"
    qids = read_experiment_qids(experiment, repair=False)
    if not qids:
        raise ValueError(f"No examples in {experiment}")
    with (folder / "pool.jsonl").open("rb") as handle:
        pool = ExamplePool(handle)
        pool.scan(repair=False)
        missing = [qid for qid in qids if qid not in pool.offsets]
        if missing:
            raise ValueError(f"{len(missing)} qids of {experiment} are not in its pool.")
        return [pool.get(qid) for qid in qids]


def save_candidate(
    names: Sequence[str],
    values: Sequence[float],
    folder: Path,
    vary: str,
    output_root: Path,
) -> list[Path]:
    """Write one degree file per question format for one candidate."""
    label = candidate_label(names, values)
    by_format: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in load_records(folder, vary):
        config = record["config"]
        built = tuple(config[name] for name in names)
        if built != tuple(values):
            raise ValueError(
                f"{folder} holds an example built with {candidate_label(names, built)}."
            )
        by_format[record["question_type"]].append(record)
    outputs = []
    for question_format, format_folder in FOLDER_BY_FORMAT.items():
        records = by_format.get(question_format)
        if not records:
            print(f"WARNING: no {question_format} examples in {folder}", flush=True)
            continue
        config = records[0]["config"]
        source_id = f"{folder.name}/vary={vary}/experiment.jsonl#{question_format}"
        samples = []
        for record in records:
            degrees = calculate_graph_degrees(record["components"])
            degrees.pop("paper_degrees")
            samples.append(
                {
                    "source_id": source_id,
                    "qid": record["qid"],
                    # Each example is its own context, so the bootstrap resamples examples.
                    "context_group_id": record["qid"],
                    "num_components": len(record["components"]),
                    **degrees,
                }
            )
        aggregate = aggregate_sample_degrees(samples)
        settings = {
            "cutoff_date": config["cutoff_date"],
            "question_format": question_format,
            "vary": vary,
            "context_length": str(config["context_length"]),
            **{name: f"{value:g}" for name, value in zip(names, values)},
        }
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiment_root": str(folder.parent.resolve()),
            **dict(zip(names, values)),
            "edge_scope": "each example's own grown components (no distractors)",
            "prompt_packing_applied": False,
            "edge_counting": "distinct directed citing/cited paper ID pairs per sample",
            "node_counting": "distinct expanded papers and citation endpoints per sample",
            "self_edges": "one in-degree, one out-degree, two total degree",
            "aggregation_unit": "paper_per_sample",
            "shared_contexts": "each example counted separately",
            "sample_identity": ["source_id", "qid"],
            "bin_width": 1,
            "sources": [
                {
                    "source_id": source_id,
                    "path": str((folder / f"vary={vary}" / "experiment.jsonl").resolve()),
                    "settings": settings,
                    "aggregate": aggregate,
                }
            ],
            "samples": samples,
            "aggregate": aggregate,
        }
        output = (
            output_root
            / format_folder
            / "graph_statistics"
            / f"sample_paper_degree_distribution.{label}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2) + b"\n")
        outputs.append(output)
        print(
            f"{label.replace('_', ' ')} {question_format}: {len(samples)} examples, "
            f"{aggregate['num_paper_observations']} papers; saved {output}",
            flush=True,
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        required=True,
        help="Folder that holds the candidate folders.",
    )
    parser.add_argument(
        "--parameters",
        choices=list(PARAMETER_SETS),
        default="theta-discount",
        help="theta-discount for create_dataset.v2.py, alpha for create_dataset.v1.py.",
    )
    parser.add_argument("--vary", default="question-format")
    for name in ("theta", "discount", "alpha"):
        parser.add_argument(
            f"--{name}", nargs="+", type=float, help="Default: every folder found."
        )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Where the question_format=<format> folders go (default: --search-root).",
    )
    args = parser.parse_args()
    names = PARAMETER_SETS[args.parameters]
    for values, folder in candidate_folders(
        args.search_root, names, [getattr(args, name) for name in names]
    ):
        save_candidate(names, values, folder, args.vary, args.output_root or args.search_root)


if __name__ == "__main__":
    main()
