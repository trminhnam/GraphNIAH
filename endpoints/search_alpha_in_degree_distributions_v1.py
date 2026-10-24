"""Select the create_dataset.v1 growth alpha from in-degree only.

Same search as search_alpha_in_degree_distributions.v8.py (KL and EMD against
the real citation graph, degree cutoffs K, bootstrap selection,
alpha_selection_summary, and all_alphas_in_degree.pdf), but the
candidates come from the v1 growth-parameter search: one folder per question
format under --experiment-root, written by
endpoints/calculate_sample_paper_degree_distribution_v2.py --parameters alpha.
Only the in-degree distribution is scored; in-degree is the number of papers in
the same example's graph that cite a paper, so with K=1 the search compares
papers cited by at least two papers. Exact score ties go to the smaller alpha.

Run after scripts/experiments/v1/growth_search.2_degrees.sh (and
scripts/db_utils/calculate_bulk_paper_degree_distribution.sh for the global
reference). From the repository root, with the default grid:
    python endpoints/search_alpha_in_degree_distributions_v1.py
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.graph.alpha_degree_search import (
    ALPHA_V1_IN_DEGREE,
    BULK_DISTRIBUTION_DIR,
    METHODS,
    add_search_arguments,
    map_in_processes,
    run_searches,
    save_all_pairs_total_degree,
)

# Same data snapshot as scripts/experiments/v1/config.sh.
CUTOFF_DATE = "2026-04-01"
DEFAULT_SEARCH_ROOT = (
    REPO_ROOT
    / f"output/experiments.v1/growth_search/cutoff_date={CUTOFF_DATE}"
    / "hops=10-22-46-94/context_length=200k/seed=1"
)
DEFAULT_GLOBAL_DISTRIBUTIONS = [
    BULK_DISTRIBUTION_DIR / f"{category}.{CUTOFF_DATE}.strict-reference-count-{strict}.json"
    for category in ("cs", "all")
    for strict in (0, 1)
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_search_arguments(
        parser, REPO_ROOT / "output/graph_statistics/alpha_in_degree_search.v1"
    )
    parser.set_defaults(
        experiment_root=DEFAULT_SEARCH_ROOT,
        global_distribution=DEFAULT_GLOBAL_DISTRIBUTIONS,
    )
    # Same default grid as scripts/experiments/v1/config.sh.
    parser.add_argument(
        "--alpha", nargs="+", type=float, default=[i / 10 for i in range(1, 11)]
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes for the searches and PDFs (default 1: no parallelism).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=METHODS,
        help="Distance methods to calculate (default: kl emd).",
    )
    parser.add_argument(
        "--power-law-fit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw the power law fitted to the generated examples on the log-log panel "
        "of selected_<metric>_no_head and the summary figure (default: off).",
    )
    parser.add_argument(
        "--power-law-repeats",
        type=int,
        default=1000,
        help="Bootstrap repeats of the power-law analysis that writes power_law.json "
        "and power_law.md in each K folder (default 1000; 0 skips the analysis).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run_searches(
        global_paths=args.global_distribution,
        experiment_root=args.experiment_root,
        output_dir=args.output_dir,
        alphas=args.alpha,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
        epsilon=args.kl_smoothing,
        unit=args.bootstrap_unit,
        validation_root=args.validation_root,
        command=shlex.join([sys.executable, *sys.argv]),
        cutoffs=args.degree_cutoffs,
        methods=args.methods,
        workers=args.workers,
        power_law_fit=args.power_law_fit,
        power_law_repeats=args.power_law_repeats,
        parameters=ALPHA_V1_IN_DEGREE,
    )
    # One all-alphas PDF per K folder; the head panel reads K=0/distributions.json.
    jobs = []
    for name, methods in results.items():
        for method, rows in methods.items():
            method_dir = args.output_dir / name / method
            cutoffs = [row["K"] for row in rows]
            if 0 in cutoffs:
                jobs.extend((method_dir, cutoff) for cutoff in cutoffs)
            else:
                print(
                    f"WARNING: {method_dir} has no K=0 run, so "
                    f"{ALPHA_V1_IN_DEGREE.all_candidates_pdf} is skipped.",
                    flush=True,
                )
    # With several workers, PDFs are reported in the order they finish.
    for index, (path,) in enumerate(
        map_in_processes(
            save_all_pairs_total_degree,
            [
                (method_dir, [cutoff], ALPHA_V1_IN_DEGREE)
                for method_dir, cutoff in jobs
            ],
            args.workers,
        ),
        start=1,
    ):
        print(f"[PDF {index}/{len(jobs)}] saved {path}", flush=True)


if __name__ == "__main__":
    main()
