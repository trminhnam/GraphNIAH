"""One run for every candidate of a growth-parameter search.

Shared by create_growth_search.v1.py (alpha) and create_growth_search.v2.py
(theta, discount). Each candidate gets exactly the files that
``create_dataset.vN.py --components-only`` writes for it, with the same pool
key, settings, and qids, so runs made either way resume each other:

    <search root>/<candidate>/pool.jsonl
    <search root>/<candidate>/vary=<factor>/experiment.jsonl

The difference is speed: the main process and the workers start once for the
whole grid, not once per candidate, and a free worker takes the next
candidate's slots instead of waiting for the slowest slot of the current one.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from src.logging.console import log_line
from src.questions.dataset_pool import ExperimentSpec, run_experiments, settings_path_for
from src.setup import bootstrap_runtime


@dataclass(frozen=True)
class Candidate:
    """One growth setting: its folder name and its experiment_configs keywords."""

    name: str
    growth: dict[str, float]


def number_text(text: str) -> str:
    """Check a grid value but keep it as typed, because it names a folder."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if "_" in text or not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"not a plain finite number: {text!r}")
    return text


def grid_candidates(grid: dict[str, list[str]]) -> list[Candidate]:
    """Every combination of the grid, the last parameter changing fastest.

    Folder names use the values as typed, as the bash scripts always did
    (``theta=10`` and ``theta=10.0`` are different folders), so earlier runs
    are found again. Two spellings of the same values are refused.
    """
    candidates: list[Candidate] = []
    seen: dict[tuple[float, ...], str] = {}
    for texts in itertools.product(*grid.values()):
        name = "_".join(f"{key}={text}" for key, text in zip(grid, texts))
        values = tuple(float(text) for text in texts)
        if values in seen:
            raise ValueError(f"{name} repeats {seen[values]}.")
        seen[values] = name
        candidates.append(Candidate(name, dict(zip(grid, values))))
    return candidates


def add_run_arguments(parser: argparse.ArgumentParser, dataset: ModuleType) -> None:
    """The arguments of every growth search; each script adds its own grid."""
    parser.add_argument(
        "--paths",
        type=Path,
        nargs="+",
        required=True,
        help="One papers-only sampled-path file per question depth.",
    )
    parser.add_argument("--graph-db", type=Path, required=True)
    parser.add_argument(
        "--search-root",
        type=Path,
        required=True,
        help="Folder that gets one <candidate> folder per growth setting.",
    )
    parser.add_argument(
        "--examples-per-value",
        type=int,
        required=True,
        help="Examples per studied value, per candidate. Raising it appends.",
    )
    parser.add_argument(
        "--vary", choices=list(dataset.DIMENSIONS), default="question-format"
    )
    parser.add_argument(
        "--context-length", type=int, default=dataset.CONTEXT_LENGTH_VALUES[0]
    )
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument(
        "--cutoff-date",
        default=dataset.FIXED_CONFIG["cutoff_date"],
        help="Earliest publication date (YYYY-MM-DD) of every paper in a component.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=500,
        help="Construction attempts per slot, each with a new seed and the same config.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Processes that build examples, shared by every candidate.",
    )


def run_growth_search(
    dataset: ModuleType, args: argparse.Namespace, candidates: list[Candidate]
) -> list[dict[str, Any]]:
    """Build every candidate's components-only examples; one stats dict each."""
    if min(args.max_attempts, args.workers, args.examples_per_value) <= 0:
        raise ValueError("Attempts, workers, and examples per value must be positive.")
    configs = [
        dataset.experiment_configs(
            args.vary,
            args.examples_per_value,
            args.context_length,
            args.random_seed,
            cutoff_date=args.cutoff_date,
            components_only=True,
            **candidate.growth,
        )
        for candidate in candidates
    ]
    # What create_dataset's open_construction and experiment_settings read.
    construction_args = argparse.Namespace(
        paths=args.paths,
        vary=args.vary,
        random_seed=args.random_seed,
        context_length=args.context_length,
        graph_db=args.graph_db,
        cutoff_date=args.cutoff_date,
        components_only=True,
        max_distractor_items=dataset.build_parser().get_default("max_distractor_items"),
    )
    bootstrap_runtime(setup_litellm_runtime=False)
    log_line("Opening the input paths, graph database, and tokenizers...")
    construction = dataset.WORKER_CONSTRUCTION["current"] = dataset.open_construction(
        construction_args
    )
    try:
        compatibility = dataset.pool_compatibility(construction_args, construction)
        settings = dataset.experiment_settings(construction_args)
        experiments = [
            ExperimentSpec(
                args.search_root / candidate.name / "pool.jsonl",
                args.search_root / candidate.name / f"vary={args.vary}" / "experiment.jsonl",
                candidate_configs,
                settings,
                compatibility,
                label=candidate.name,
            )
            for candidate, candidate_configs in zip(candidates, configs)
        ]
        dataset.check_destinations(
            construction_args,
            construction,
            [
                path
                for experiment in experiments
                for path in (
                    experiment.pool_path,
                    experiment.output_path,
                    settings_path_for(experiment.output_path),
                )
            ],
        )
        dataset.log_path_groups(construction)
        log_line(
            f"Growth search: {len(candidates)} candidates, each {len(configs[0]):,} "
            f"slots (vary {args.vary}, {args.examples_per_value} examples per "
            f"value), {args.context_length:,}-token context, seed {args.random_seed}, "
            f"cutoff {args.cutoff_date}, {dataset.GENERATOR_VERSION}, components only."
        )
        if args.workers > 1:
            # Each worker opens its own; free this process's copy.
            dataset.WORKER_CONSTRUCTION.clear()
            construction.close()
        started = time.monotonic()
        stats = run_experiments(
            experiments,
            functools.partial(
                dataset.construct_in_worker, construction_args, compatibility["tokenizers"]
            ),
            seed=args.random_seed,
            max_attempts=args.max_attempts,
            workers=args.workers,
        )
        for candidate, result in zip(candidates, stats):
            retries = ", ".join(
                f"{reason} x{count}" for reason, count in sorted(result["retries"].items())
            )
            log_line(
                f"{candidate.name}: {result['created']:,} new, {result['reused']:,} "
                f"reused from the pool, {result['resumed']:,} already there. "
                f"Retries: {retries or 'none'}."
            )
        log_line(
            f"Done: {len(candidates)} candidates in {args.search_root} "
            f"({time.monotonic() - started:,.0f} s after startup)."
        )
        return stats
    finally:
        if dataset.WORKER_CONSTRUCTION.pop("current", None) is not None:
            construction.close()
