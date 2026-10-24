#!/usr/bin/env python3
r"""Build the create_dataset.v1.py growth-parameter search data in one run.

For every alpha of the grid, build the components-only examples of one
experiment (default --vary question-format) into the same files that
``create_dataset.v1.py --components-only`` writes for that alpha:

    <search root>/alpha=<alpha>/pool.jsonl
    <search root>/alpha=<alpha>/vary=<factor>/experiment.jsonl

Pools, settings, and qids are identical, so runs made either way resume each
other, and growth_search.2_degrees.sh reads them unchanged. Unlike one
create_dataset.v1.py run per alpha, the main process and the workers start
once for the whole grid (each start cost 1.5-2.5 minutes per candidate on
for the slowest slot of the current one. Folder names use the values as typed
(alpha=1 and alpha=1.0 are different folders).

Arguments:
    --paths PATH [PATH ...]   Required. One papers-only sampled-path file per
                              question depth (paths of 10, 22, 46, 94 papers),
                              each with its disjoint_path_groups.json.
    --graph-db PATH           Required. Graph SQLite database.
    --search-root DIR         Required. Folder for the alpha folders.
    --alpha X [X ...]         Required. Zero-child selection weights, each > 0.
    --examples-per-value N    Required. Examples per studied value, per alpha.
                              Raising it appends to every alpha.
    --vary FACTOR             Studied factor (default question-format).
    --context-length N        Token budget per prompt (default 200000); the
                              needles of every example must fit it.
    --random-seed N           Seed for the plan and constructions (default 1).
    --cutoff-date YYYY-MM-DD  Earliest publication date of every paper in a
                              component (default 2026-04-01).
    --max-attempts N          Construction attempts per slot (default 500).
    --workers N               Processes that build examples (default 1),
                              shared by every alpha.

Example (activate the niah environment first)::

    python create_growth_search.v1.py \
        --paths output/data.2026-04-01/hops={10,22,46,94}/papers-only/sampled_paths.json \
        --graph-db resources/semantic_scholar_graph.2026-04-01/semantic_scholar_paths.db \
        --search-root output/experiments.v1/growth_search/cutoff_date=2026-04-01/hops=10-22-46-94/context_length=200k/seed=1 \
        --alpha 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
        --examples-per-value 2000 --workers 96
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "create_dataset_v1", REPO_ROOT / "create_dataset.v1.py"
)
dataset = importlib.util.module_from_spec(SPEC)
# Registered before it runs: spawned workers re-run this file and look the
# construction function up by this module name.
sys.modules[SPEC.name] = dataset
SPEC.loader.exec_module(dataset)

from src.questions import growth_search  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    growth_search.add_run_arguments(parser, dataset)
    parser.add_argument(
        "--alpha",
        nargs="+",
        type=growth_search.number_text,
        required=True,
        help="Selection weights for papers with no accepted optional children; > 0.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    candidates = growth_search.grid_candidates({"alpha": args.alpha})
    growth_search.run_growth_search(dataset, args, candidates)


if __name__ == "__main__":
    main()
