"""Compare generated degree distributions with a fixed global reference.

A search candidate fixes the generation parameters: alpha for v8 (ALPHA) or a
(theta, discount) pair for v10 (THETA_DISCOUNT).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import itertools
import multiprocessing
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import orjson
import pandas as pd
from pypdf import PdfWriter
from scipy.special import rel_entr, zeta

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
from matplotlib.transforms import Bbox

from src.graph.power_law import (
    SCAN_MIN_PAPERS,
    SCAN_MIN_SHARE,
    SIGNIFICANCE,
    analyze_power_law,
    fit_power_law,
    ks_distance,
)

# Times New Roman text and Times-like STIX math. Where Times is not installed
# (e.g. a Linux cluster), STIXGeneral, a Times-like font that ships with
# matplotlib, is used instead.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 12,
    }
)

METRICS = ("in_degree", "out_degree", "total_degree")
# Axis label and short adjective for each metric in figures and reports.
METRIC_LABELS = {
    "in_degree": ("In-degree", "in-degree"),
    "out_degree": ("Out-degree", "out-degree"),
    "total_degree": ("Total degree", "total degree"),
}
# v8 and v10 share these question types; only the experiment folders differ.
EXPERIMENTS = {
    "yesno": "experiment_7_reachability_yesno.v8",
    "multiple": "experiment_8_reachability_multiple.v8",
    "counting": "experiment_9_reachability_counting.v8",
}
V10_EXPERIMENTS = {
    "yesno": "experiment_7_reachability_yesno.v10",
    "multiple": "experiment_8_reachability_multiple.v10",
    "counting": "experiment_9_reachability_counting.v10",
}

V11_EXPERIMENTS = {
    "yesno": "experiment_7_reachability_yesno.v11",
    "multiple": "experiment_8_reachability_multiple.v11",
    "counting": "experiment_9_reachability_counting.v11",
}
# create_dataset.v2 mixes question formats in one experiment, so its degree
# files are split into one folder per format under the growth-search root.
V2_EXPERIMENTS = {
    "yesno": "question_format=reachability-yesno",
    "multiple": "question_format=reachability-multiple",
    "counting": "question_format=reachability-counting",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
BULK_DISTRIBUTION_DIR = (
    REPO_ROOT / "output/graph_statistics/bulk_paper_degree_distribution"
)
DEFAULT_GLOBAL_DISTRIBUTIONS = [
    BULK_DISTRIBUTION_DIR
    / f"{category}.2025-09-01.strict-reference-count-{strict}.json"
    for category in ("cs", "all")
    for strict in (0, 1)
]


def add_search_arguments(parser: argparse.ArgumentParser, output_dir: Path) -> None:
    """Options shared by the alpha and theta/discount search endpoints."""
    parser.add_argument(
        "--global-distribution",
        type=Path,
        nargs="+",
        default=DEFAULT_GLOBAL_DISTRIBUTIONS,
        help="One or more global histogram files; each gets its own output subfolder. "
        "Default: the cs and all files for strict-reference-count 0 and 1.",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path(
            os.environ.get("EXPERIMENT_OUTPUT_ROOT", REPO_ROOT / "output/experiments")
        ),
        help="Parent folder of all three experiment folders; honors EXPERIMENT_OUTPUT_ROOT.",
    )
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--kl-smoothing", type=float, default=1e-8)
    parser.add_argument(
        "--bootstrap-unit",
        choices=("context_group", "source_file"),
        default="context_group",
    )
    parser.add_argument(
        "--validation-root",
        type=Path,
        help="Independent generation outputs, with the same experiment layout.",
    )
    parser.add_argument(
        "--degree-cutoffs",
        nargs="+",
        type=int,
        default=list(range(11)),
        help="Cutoffs K; each K selects a candidate using degree > K only. "
        "Default: 0-10.",
    )


@dataclass(frozen=True)
class SearchParameters:
    """The generation settings one candidate fixes, and how outputs name them.

    A candidate key is a float for one parameter (alpha) and a tuple of floats
    for two (theta, discount), so alpha outputs keep their original layout.
    """

    names: tuple[str, ...]
    experiments: dict[str, str]
    create_script: str
    noun: str
    plural: str
    plural_key: str
    tie_break: str
    latex_names: tuple[str, ...]
    # LaTeX symbol of each name, for figure text such as $\theta=10$.
    symbols: tuple[str, ...]
    rule: str
    check: Callable[[tuple[float, ...]], bool]
    # Degree metrics the score averages over; inputs are still validated for all three.
    metrics: tuple[str, ...] = METRICS

    def __post_init__(self) -> None:
        if len(self.names) not in (1, 2) or set(self.experiments) != set(EXPERIMENTS):
            raise ValueError("Use one or two parameters and the three question types.")
        if (
            not self.metrics
            or len(set(self.metrics)) != len(self.metrics)
            or not set(self.metrics) <= set(METRICS)
        ):
            raise ValueError(f"Metrics must be one or more distinct values from {METRICS}.")

    @property
    def featured(self) -> str:
        """Metric of the selected_<metric> and all_pairs_<metric> figures.

        Total degree whenever it is scored, so those outputs keep their names.
        """
        return "total_degree" if "total_degree" in self.metrics else self.metrics[0]

    @property
    def all_candidates_pdf(self) -> str:
        """save_all_pairs_total_degree's file in each K folder, such as all_alphas_in_degree.pdf."""
        candidates = "pairs" if len(self.names) == 2 else self.plural_key
        return f"all_{candidates}_{self.featured}.pdf"

    @property
    def title(self) -> str:
        return "/".join(self.names)

    @property
    def stem(self) -> str:
        return "_".join(self.names)

    @property
    def index(self) -> str | list[str]:
        """pandas key that groups or indexes rows by candidate."""
        return self.names[0] if len(self.names) == 1 else list(self.names)

    @property
    def file_pattern(self) -> str:
        return (
            "sample_paper_degree_distribution."
            + "_".join(f"{name}=*" for name in self.names)
            + ".json"
        )

    def key(self, values: Any) -> Any:
        values = tuple(
            map(float, values if isinstance(values, (tuple, list)) else (values,))
        )
        return values[0] if len(values) == 1 else values

    def values(self, key: Any) -> tuple[float, ...]:
        return key if isinstance(key, tuple) else (key,)

    def columns(self, key: Any, prefix: str = "") -> dict[str, float]:
        return {f"{prefix}{n}": v for n, v in zip(self.names, self.values(key))}

    def label(self, key: Any, separator: str = ", ") -> str:
        return separator.join(
            f"{n}={v:g}" for n, v in zip(self.names, self.values(key))
        )

    @property
    def math_names(self) -> tuple[str, ...]:
        return tuple(f"${symbol}$" for symbol in self.symbols)

    @property
    def math_title(self) -> str:
        return "/".join(self.math_names)

    def math_label(self, key: Any, separator: str = ", ") -> str:
        """label() for figures, such as $\\theta=10$, $\\delta=0.8$."""
        return separator.join(
            f"${s}={v:g}$" for s, v in zip(self.symbols, self.values(key))
        )

    def text(self, key: Any) -> str:
        """The bare value for one parameter; name=value pairs for two."""
        return f"{key:g}" if len(self.names) == 1 else self.label(key)

    def column(self, key: Any) -> Any:
        """bootstrap_scores.csv header: the alpha value itself, or a pair label."""
        return key if len(self.names) == 1 else self.label(key, "_")

    def file_name(self, key: Any) -> str:
        return f"sample_paper_degree_distribution.{self.label(key, '_')}.json"


# Named functions, not lambdas, so worker processes can receive the parameters.
def alpha_is_valid(values: tuple[float, ...]) -> bool:
    return values[0] > 0


def theta_discount_is_valid(values: tuple[float, ...]) -> bool:
    return values[0] > 0 and 0 <= values[1] < 1


ALPHA = SearchParameters(
    names=("alpha",),
    experiments=EXPERIMENTS,
    create_script="scripts/experiments.v8/calculate_alpha_degree_distributions.sh",
    noun="alpha",
    plural="alphas",
    plural_key="alphas",
    tie_break="smaller alpha",
    latex_names=(r"$\alpha$",),
    symbols=(r"\alpha",),
    rule="Alpha values must be distinct, positive, and finite.",
    check=alpha_is_valid,
)
THETA_DISCOUNT = SearchParameters(
    names=("theta", "discount"),
    experiments=V10_EXPERIMENTS,
    create_script=(
        "scripts/experiments.v10/calculate_theta_discount_degree_distributions.sh"
    ),
    noun="theta/discount pair",
    plural="theta/discount pairs",
    plural_key="theta_discount_pairs",
    tie_break="smaller theta, then smaller discount",
    latex_names=(r"$\theta$", "Discount"),
    symbols=(r"\theta", r"\delta"),
    rule="Theta/discount pairs must be distinct and finite, "
    "with theta > 0 and 0 <= discount < 1.",
    check=theta_discount_is_valid,
)

THETA_DISCOUNT_V11 = replace(
    THETA_DISCOUNT,
    experiments=V11_EXPERIMENTS,
    create_script="scripts/experiments.v11/calculate_theta_discount_degree_distributions.sh",
)

# Score only the total-degree distribution (with K=1: papers with total degree >= 2).
ALPHA_TOTAL_DEGREE = replace(ALPHA, metrics=("total_degree",))
THETA_DISCOUNT_TOTAL_DEGREE = replace(THETA_DISCOUNT, metrics=("total_degree",))
THETA_DISCOUNT_V11_TOTAL_DEGREE = replace(
    THETA_DISCOUNT_V11, metrics=("total_degree",)
)
# Score only the in-degree distribution: the number of papers that cite each paper.
ALPHA_IN_DEGREE = replace(ALPHA, metrics=("in_degree",))
THETA_DISCOUNT_IN_DEGREE = replace(THETA_DISCOUNT, metrics=("in_degree",))
THETA_DISCOUNT_V11_IN_DEGREE = replace(THETA_DISCOUNT_V11, metrics=("in_degree",))
THETA_DISCOUNT_V2_IN_DEGREE = replace(
    THETA_DISCOUNT,
    experiments=V2_EXPERIMENTS,
    create_script="scripts/experiments/v2/growth_search.2_degrees.sh",
    metrics=("in_degree",),
)
# create_dataset.v1.py (v8 alpha growth) writes the same per-format layout.
ALPHA_V1_IN_DEGREE = replace(
    ALPHA_IN_DEGREE,
    experiments=V2_EXPERIMENTS,
    create_script="scripts/experiments/v1/growth_search.2_degrees.sh",
)


def read_json(path: Path) -> tuple[dict, dict]:
    raw = path.read_bytes()
    return orjson.loads(raw), {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def histogram(record: dict, metric: str, *, cutoff: int | None = 0) -> Counter[int]:
    """Read counts with degree > cutoff for this metric; None keeps every degree.

    One QA example may have no paper above the cutoff, so the result can be empty.
    """
    counts: Counter[int] = Counter()
    for key, count in record[f"{metric}_histogram"].items():
        degree = int(key)
        if degree < 0 or type(count) is not int or count < 0:
            raise ValueError("Degrees and counts must be nonnegative integers.")
        if cutoff is None or degree > cutoff:
            counts[degree] += count
    return counts


def pooled_counts(
    records: list[dict], metric: str, *, cutoff: int | None = 0
) -> Counter[int]:
    """Add counts across records; a compared distribution needs at least one paper."""
    counts: Counter[int] = Counter()
    for record in records:
        counts.update(histogram(record, metric, cutoff=cutoff))
    if not sum(counts.values()):
        scope = "" if cutoff is None else f"degree > {cutoff} "
        raise ValueError(f"Cannot compare an empty {scope}distribution for {metric}.")
    return counts


def skipped_cutoffs(
    global_data: dict,
    datasets: dict,
    cutoffs: list[int],
    parameters: SearchParameters = ALPHA,
) -> dict[int, str]:
    """Map each K that the global file or a generated distribution cannot support.

    P(degree | degree > K) is undefined when no paper has degree > K, which
    happens for every K at or above that distribution's highest degree. The
    reason names the first failing distribution, checking the global file first.
    """
    records_by_name = {
        "global": [global_data],
        **{
            f"{question_type}, {parameters.label(alpha)}": entry["examples"]
            for (question_type, alpha), entry in sorted(datasets.items())
        },
    }
    limits = [
        (
            max(
                (
                    int(k)
                    for record in records
                    for k, v in record[f"{metric}_histogram"].items()
                    if v > 0
                ),
                default=0,
            ),
            name,
            metric,
        )
        for name, records in records_by_name.items()
        for metric in parameters.metrics
    ]
    skipped = {}
    for cutoff in cutoffs:
        for highest, name, metric in limits:
            if cutoff >= highest:
                skipped[cutoff] = (
                    f"{name} {metric} has no paper with degree > {cutoff} "
                    f"(highest degree {highest})"
                )
                break
    return skipped


def experiments_cache_key(
    root: Path, alphas: list[Any], cutoff_date: str, parameters: SearchParameters
) -> tuple:
    """Identify one set of loaded experiment files, including their folders."""
    return (
        "experiments",
        str(root.resolve()),
        parameters.names,
        tuple(sorted(parameters.experiments.items())),
        tuple(alphas),
        cutoff_date,
    )


def cached_experiments(
    cache: dict,
    root: Path,
    alphas: list[Any],
    cutoff_date: str,
    parameters: SearchParameters,
) -> dict:
    """Load experiment files once per root, parameters, candidates, and cutoff date."""
    key = experiments_cache_key(root, alphas, cutoff_date, parameters)
    if key not in cache:
        cache[key] = load_experiments(root, alphas, cutoff_date, parameters)
    return cache[key]


def validate_degrees(
    record: dict, num_papers: int, in_edges: int, out_edges: int | None = None
) -> None:
    """Check paper and weighted-degree totals; out_edges defaults to in_edges."""
    out_edges = in_edges if out_edges is None else out_edges
    totals = {
        "in_degree": in_edges,
        "out_degree": out_edges,
        "total_degree": in_edges + out_edges,
    }
    for metric in METRICS:
        counts = pooled_counts([record], metric, cutoff=None)
        expected = totals[metric]
        if (
            sum(counts.values()) != num_papers
            or sum(k * v for k, v in counts.items()) != expected
        ):
            raise ValueError(f"Inconsistent paper or edge count for {metric}.")


def load_experiments(
    root: Path,
    alphas: list[Any],
    cutoff: str,
    parameters: SearchParameters = ALPHA,
) -> dict:
    """Load one file per question type and candidate key (alpha or theta/discount)."""
    datasets = {}
    missing_lines = []
    for question_type, folder in parameters.experiments.items():
        reference_settings = None
        statistics_dir = root / folder / "graph_statistics"
        for path in sorted(statistics_dir.glob(parameters.file_pattern)):
            data, provenance = read_json(path)
            alpha = parameters.key(tuple(data[name] for name in parameters.names))
            if alpha not in alphas:
                continue
            key = (question_type, alpha)
            if key in datasets:
                raise ValueError(f"Duplicate {parameters.title} file: {path}")
            if (
                data.get("schema_version") != 1
                or data.get("aggregation_unit") != "paper_per_sample"
                or data.get("prompt_packing_applied") is not False
            ):
                raise ValueError(f"Unsupported schema or graph scope: {path}")
            sources = {s["source_id"]: s for s in data["sources"]}
            settings = sorted(
                tuple(
                    sorted(
                        (k, v)
                        for k, v in s["settings"].items()
                        if k not in parameters.names
                    )
                )
                for s in sources.values()
            )
            if not settings or (
                reference_settings is not None and settings != reference_settings
            ):
                raise ValueError(
                    f"Different conditions across {parameters.plural}: {path}"
                )
            reference_settings = settings
            for source in sources.values():
                if source["settings"].get("cutoff_date") != cutoff:
                    raise ValueError(f"Global/generated cutoff mismatch: {path}")
            examples = data["samples"]
            if not examples:
                raise ValueError(f"No examples in {path}")
            seen = set()
            for example in examples:
                example.pop("paper_degrees", None)
                identity = (example["source_id"], example["qid"])
                if identity in seen or identity[0] not in sources:
                    raise ValueError(
                        f"Duplicate example or unknown source in {path}: {identity}"
                    )
                seen.add(identity)
                validate_degrees(
                    example, example["num_papers"], example["num_citation_edges"]
                )
            for metric in METRICS:
                if pooled_counts(examples, metric, cutoff=None) != histogram(
                    data["aggregate"], metric, cutoff=None
                ):
                    raise ValueError(f"Saved aggregate does not match examples: {path}")
            datasets[key] = {
                "examples": examples,
                "settings": settings,
                "provenance": {
                    **provenance,
                    "question_type": question_type,
                    **parameters.columns(alpha),
                    "sources": data["sources"],
                },
            }
        missing = set(alphas) - {a for t, a in datasets if t == question_type}
        if missing:
            note = "" if statistics_dir.is_dir() else " (folder does not exist)"
            missing_lines.append(
                f"  {question_type}: {len(missing)} of {len(alphas)} "
                f"{parameters.title}(s) missing in {statistics_dir}{note}"
            )
            missing_lines.extend(
                f"    {parameters.file_name(alpha)}" for alpha in sorted(missing)
            )
    # Report every missing file at once instead of stopping at the first type.
    if missing_lines:
        raise FileNotFoundError(
            f"Missing {parameters.title} degree distribution files:\n"
            + "\n".join(missing_lines)
            + f"\nCreate them with {parameters.create_script}."
        )
    return datasets


def align_counts(counts: Counter[int], support: np.ndarray) -> np.ndarray:
    unknown = set(counts) - set(support)
    if unknown:
        raise ValueError(
            f"Degrees outside fixed calibration support: {sorted(unknown)}"
        )
    return np.array([counts[k] for k in support], dtype=float)


def probability_distance(
    p: np.ndarray, q: np.ndarray, support: np.ndarray, method: str, epsilon: float
) -> float:
    """Normalize counts on sorted support; return KL in nats or EMD in degrees."""
    p, q = p / p.sum(), q / q.sum()
    if method == "kl":
        p = (1 - epsilon) * p + epsilon / len(support)
        q = (1 - epsilon) * q + epsilon / len(support)
        return float(rel_entr(p, q).sum())
    if method == "emd":
        # Integral of the absolute CDF difference; avoids sorting each resample.
        return float(np.sum(np.abs(np.cumsum(p - q)[:-1]) * np.diff(support)))
    raise ValueError("Method must be kl or emd.")


def group_examples(examples: list[dict], unit: str) -> list[list[dict]]:
    groups = defaultdict(list)
    for example in examples:
        group = (
            (example.get("context_group_id") or example["qid"])
            if unit == "context_group"
            else "whole_file"
        )
        groups[(example["source_id"], group)].append(example)
    return list(groups.values())


def measure(
    datasets: dict,
    references: dict,
    method: str,
    epsilon: float,
    cutoff: int,
    parameters: SearchParameters = ALPHA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows, example_rows = [], []
    for (question_type, alpha), entry in sorted(datasets.items()):
        examples = entry["examples"]
        for metric, reference in references.items():
            support, global_counts, global_mean = reference
            scale = global_mean if method == "emd" else 1
            for example in [None, *examples]:
                counts = (
                    pooled_counts(examples, metric, cutoff=cutoff)
                    if example is None
                    else histogram(example, metric, cutoff=cutoff)
                )
                if not sum(counts.values()):
                    continue  # This example has no paper above the cutoff.
                weights = align_counts(counts, support)
                value = probability_distance(
                    weights, global_counts, support, method, epsilon
                )
                row = {
                    "question_type": question_type,
                    **parameters.columns(alpha),
                    "metric": metric,
                    "distance": value,
                    "score_component": value / scale,
                }
                if method == "kl":
                    row.update(
                        unsmoothed_kl=probability_distance(
                            weights, global_counts, support, method, 0
                        ),
                        global_zero_mass=float(
                            weights[global_counts == 0].sum() / weights.sum()
                        ),
                    )
                if example is None:
                    row.update(
                        num_examples=len(examples),
                        num_paper_observations=sum(counts.values()),
                        mean_num_papers=float(
                            np.mean([e["num_papers"] for e in examples])
                        ),
                        generated_mean_degree=float(
                            np.average(support, weights=weights)
                        ),
                    )
                    pooled_rows.append(row)
                else:
                    row.update(
                        source_id=example["source_id"],
                        qid=example["qid"],
                        context_group_id=example.get("context_group_id"),
                    )
                    example_rows.append(row)
    return pd.DataFrame(pooled_rows), pd.DataFrame(example_rows)


def bootstrap_scores(
    datasets: dict,
    references: dict,
    method: str,
    epsilon: float,
    alphas: list[float],
    repeats: int,
    seed: int,
    unit: str,
    cutoff: int,
    parameters: SearchParameters = ALPHA,
) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    scores = np.zeros((repeats, len(alphas)))
    group_rows = []
    for (question_type, alpha), entry in sorted(datasets.items()):
        groups = group_examples(entry["examples"], unit)
        n = len(groups)
        if n < 2:
            raise ValueError(
                "At least two independent groups needed: "
                f"{question_type}, {parameters.label(alpha)}"
            )
        group_rows.append(
            {
                "question_type": question_type,
                **parameters.columns(alpha),
                "num_groups": n,
            }
        )
        draws = rng.multinomial(n, np.full(n, 1 / n), size=repeats)
        for metric, (support, global_counts, global_mean) in references.items():
            # A group with no paper above the cutoff adds a row of zeros.
            matrix = np.array(
                [
                    align_counts(
                        sum(
                            (histogram(e, metric, cutoff=cutoff) for e in g), Counter()
                        ),
                        support,
                    )
                    for g in groups
                ]
            )
            scale = global_mean if method == "emd" else 1
            # Limit temporary allocations for large supports and many resamples.
            for start in range(0, repeats, 64):
                counts = draws[start : start + 64] @ matrix
                values = [
                    probability_distance(c, global_counts, support, method, epsilon)
                    for c in counts
                ]
                scores[start : start + len(values), alphas.index(alpha)] += (
                    np.array(values)
                    / scale
                    / (len(EXPERIMENTS) * len(parameters.metrics))
                )
        print(
            f"{method}: bootstrapped {question_type}, {parameters.label(alpha)}, "
            f"groups={n}, K={cutoff}",
            flush=True,
        )
    return scores, group_rows


def rank_alphas(
    pooled: pd.DataFrame,
    scores: np.ndarray,
    alphas: list[Any],
    parameters: SearchParameters = ALPHA,
) -> pd.DataFrame:
    """Sort candidates by score; exact ties go to smaller parameter values in order."""
    index = parameters.index
    ranking = pooled.groupby(index).score_component.mean().rename("score").to_frame()
    raw = pooled.pivot(
        index=index, columns=["question_type", "metric"], values="distance"
    )
    raw.columns = [f"{t}_{m}_distance" for t, m in raw.columns]
    low, high = np.quantile(scores, [0.025, 0.975], axis=0)
    ties = np.isclose(scores, scores.min(axis=1, keepdims=True), rtol=1e-10, atol=1e-12)
    uncertainty = (
        pd.DataFrame([parameters.columns(alpha) for alpha in alphas])
        .assign(
            score_ci_low=low,
            score_ci_high=high,
            bootstrap_selection_frequency=(
                ties / ties.sum(axis=1, keepdims=True)
            ).mean(axis=0),
        )
        .set_index(index)
    )
    return (
        ranking.join(raw)
        .join(uncertainty)
        .sort_values(["score", *parameters.names])
    )


def viridis_colors(count: int) -> list[Any]:
    """count viridis colors, dark to light, for lines and markers.

    Viridis 0.25-0.8 skips the dark purple end (kept for the global line) and the
    pale yellow end (hard to see on white).
    """
    return list(plt.get_cmap("viridis")(np.linspace(0.25, 0.8, count)))


# Global vs generated distributions: dark purple (viridis low end) and the green
# just below viridis's yellow (5-class viridis, class 4); the yellow itself is too
# pale on white.
GLOBAL_COLOR, GENERATED_COLOR = "#440154", "#5dc863"
# Power-law fit to the generated examples: a darker viridis green than their points.
FIT_COLOR = plt.get_cmap("viridis")(0.6)
# The K=0 reference (dashed) in the head panel, and the alpha score line.
REFERENCE_COLOR, SCORE_COLOR = plt.get_cmap("viridis")([0.3, 0.65])
# One fixed color per question type, the same in every figure.
QUESTION_TYPE_COLORS = dict(zip(EXPERIMENTS, viridis_colors(len(EXPERIMENTS))))


def save_figure(
    figure: Any, output: Path, name: str, panels: dict[str, list[Any]] | None = None
) -> None:
    """Save PNG and PDF; ``panels`` maps a file name to axes that are also saved alone.

    A panel file is cropped from the laid-out figure around its axes (with titles,
    labels, legends, and colorbars), so it looks exactly as it does in the figure.
    """
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    for panel, axes in (panels or {}).items():
        box = Bbox.union([ax.get_tightbbox() for ax in axes])
        box = box.transformed(figure.dpi_scale_trans.inverted()).padded(0.05)
        for extension in ("png", "pdf"):
            figure.savefig(output / f"{panel}.{extension}", dpi=180, bbox_inches=box)
    plt.close(figure)


def plot_alpha_scores(
    ax: Any,
    ranking: pd.DataFrame,
    method: str,
    selected: float,
    cutoff: int,
    summary_style: bool = False,
) -> None:
    """``summary_style`` gives the Title Case title of the summary figure, without
    the cutoff, and scores with five decimals, like the theta/discount heat map."""
    ordered = ranking.sort_index()
    ax.plot(ordered.index, ordered.score, marker="o", color=SCORE_COLOR)
    ax.fill_between(
        ordered.index,
        ordered.score_ci_low,
        ordered.score_ci_high,
        color=SCORE_COLOR,
        alpha=0.2,
        label="95% bootstrap interval",
    )
    ax.axvline(
        selected,
        color="black",
        linestyle="--",
        label=f"Selected {ALPHA.math_label(selected)}",
    )
    ax.set(
        xlabel=ALPHA.math_title,
        ylabel="Mean KL (nats)" if method == "kl" else "Mean relative EMD",
        title=f"{method.upper()} {ALPHA.math_title} selection, degree > {cutoff}",
    )
    if summary_style:
        ax.set_title("Mean KL Divergence" if method == "kl" else "Mean Relative EMD")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    ax.set_xticks(ordered.index, labels=[f"{alpha:g}" for alpha in ordered.index])
    ax.legend()


def plot_alpha_frequencies(ax: Any, ranking: pd.DataFrame, selected: float) -> None:
    """One-parameter frequency heat map: one bar per alpha, the selected one outlined.

    Bars are evenly spaced whatever the alpha values; each nonzero share is printed
    above its bar.
    """
    ordered = ranking.sort_index()
    shares = ordered.bootstrap_selection_frequency
    bars = ax.bar(np.arange(len(ordered)), shares, width=0.7, color=SCORE_COLOR)
    bars[ordered.index.get_loc(selected)].set(edgecolor="black", linewidth=2.5)
    for bar, share in zip(bars, shares):
        if share > 0:
            # A few points above the bar, clear of the selected bar's outline.
            ax.annotate(
                format_share(share),
                (bar.get_x() + bar.get_width() / 2, share),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    ax.set_xticks(
        np.arange(len(ordered)), labels=[f"{alpha:g}" for alpha in ordered.index]
    )
    # Headroom for the label above a 100% bar.
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(
        title="Bootstrap Selection Frequency",
        xlabel=ALPHA.math_title,
        ylabel="Share of bootstrap resamples",
    )


def format_share(value: float) -> str:
    return "<1%" if 0 < value < 0.005 else f"{value:.0%}"


def plot_heat_map(
    ax: Any,
    values: pd.Series,
    selected: tuple[float, float],
    parameters: SearchParameters,
    *,
    colormap: str,
    label: str,
    text: Callable[[float], str],
    limits: tuple[float, float] | None = None,
) -> Any:
    """Rows are the first parameter and columns the second; outline the selected cell.

    colormap is "viridis" (lighter cells hold larger values) or "viridis_r" (lighter
    cells hold smaller values). Returns the colorbar.
    """
    grid = values.unstack(parameters.names[1]).sort_index().sort_index(axis=1)
    data = grid.to_numpy(dtype=float)
    mesh = ax.pcolormesh(
        np.ma.masked_invalid(data),
        cmap=colormap,
        edgecolors="white",
        linewidth=2,
        **({} if limits is None else {"vmin": limits[0], "vmax": limits[1]}),
    )
    for (row, column), value in np.ndenumerate(data):
        if np.isnan(value):
            continue
        red, green, blue, _ = mesh.cmap(mesh.norm(value))
        dark = 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5
        ax.text(
            column + 0.5,
            row + 0.5,
            text(value),
            ha="center",
            va="center",
            fontsize=10,
            color="white" if dark else "#0b0b0b",
        )
    ax.add_patch(
        Rectangle(
            (grid.columns.get_loc(selected[1]), grid.index.get_loc(selected[0])),
            1,
            1,
            fill=False,
            edgecolor="black",
            linewidth=2.5,
            clip_on=False,
        )
    )
    ax.set_xticks(
        np.arange(len(grid.columns)) + 0.5, labels=[f"{v:g}" for v in grid.columns]
    )
    ax.set_yticks(
        np.arange(len(grid.index)) + 0.5, labels=[f"{v:g}" for v in grid.index]
    )
    ax.invert_yaxis()
    ax.set(
        xlabel=parameters.math_names[1],
        ylabel=parameters.math_names[0],
    )
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    return ax.figure.colorbar(mesh, ax=ax, label=label)


def plot_score_heat_maps(
    score_ax: Any,
    frequency_ax: Any,
    ranking: pd.DataFrame,
    method: str,
    selected: tuple[float, float],
    cutoff: int,
    parameters: SearchParameters,
    summary_style: bool = False,
) -> None:
    """Two-parameter plot_alpha_scores: score and bootstrap selection frequency.

    In both maps lighter cells are better: a smaller score, a larger frequency.
    ``summary_style`` gives the Title Case titles of <stem>_scores and the summary
    figure, without the cutoff and the selected pair.
    """
    score_colorbar = plot_heat_map(
        score_ax,
        ranking.score,
        selected,
        parameters,
        colormap="viridis_r",
        label="Mean KL" if method == "kl" else "Mean relative EMD",
        text=lambda value: f"{value:.5f}",
    )
    # Every score in the panel, in the cells and on the colorbar, has five decimals.
    score_colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    if summary_style:
        score_ax.set_title("Mean KL Divergence" if method == "kl" else "Mean Relative EMD")
    else:
        score_ax.set_title(
            f"{method.upper()} {parameters.math_title} selection, degree > {cutoff}\n"
            f"outlined: selected {parameters.math_label(selected)}"
        )
    colorbar = plot_heat_map(
        frequency_ax,
        ranking.bootstrap_selection_frequency,
        selected,
        parameters,
        colormap="viridis",
        label="Share of bootstrap resamples",
        text=format_share,
        limits=(0, 1),
    )
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    frequency_ax.set_title(
        "Bootstrap Selection Frequency"
        if summary_style
        else f"Bootstrap selection frequency, degree > {cutoff}"
    )


def total_degree_counts(
    distributions: dict,
    references: dict,
    selected: float,
    cutoff: int,
    metric: str = "total_degree",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return support, global counts, and generated counts pooled over question types."""
    support, global_counts, _ = references[metric]
    generated_counts = np.sum(
        [
            align_counts(
                pooled_counts(
                    distributions[(question_type, selected)]["examples"],
                    metric,
                    cutoff=cutoff,
                ),
                support,
            )
            for question_type in EXPERIMENTS
        ],
        axis=0,
    )
    return support, global_counts, generated_counts


def plot_head(
    ax: Any,
    counts: dict[str, Counter[int]],
    selected: Any,
    reference: Any | None,
    cutoff: int,
    max_degree: int = 20,
    parameters: SearchParameters = ALPHA,
    fitted: bool = True,
) -> dict[str, np.ndarray]:
    """Plot degrees 1..max_degree on linear axes; shade degrees <= K left out of the fit.

    counts holds positive-degree counts for "global", "selected", and optionally
    "reference" (the candidate selected at K=0). Each probability divides by all
    positive-degree papers, not only the plotted degrees. Returns the probabilities.
    ``fitted=False`` labels a candidate that the fit did not select.
    """
    degrees = np.arange(1, max_degree + 1)
    values = {
        name: np.array([c[d] for d in degrees]) / sum(c.values())
        for name, c in counts.items()
    }
    style = {"marker": "o", "markersize": 3}
    if cutoff > 0:
        ax.axvspan(
            0.5,
            cutoff + 0.5,
            color="gray",
            alpha=0.15,
            label=f"degree <= {cutoff}: not used in the fit",
        )
    ax.plot(degrees, values["global"], color=GLOBAL_COLOR, label="Global", **style)
    ax.plot(
        degrees,
        values["selected"],
        color=GENERATED_COLOR,
        label=parameters.math_label(selected)
        + (f", fit on degree > {cutoff}" if fitted else ""),
        **style,
    )
    if reference is not None and reference != selected:
        ax.plot(
            degrees,
            values["reference"],
            color=REFERENCE_COLOR,
            linestyle="--",
            label=f"{parameters.math_label(reference)}, fit on degree > 0",
            **style,
        )
    ax.set(ylabel="P(degree | degree > 0)")
    ax.set_xticks(degrees)
    return values


def plot_total_degree(
    linear: Any,
    log_log: Any,
    head: Any | None,
    counts: tuple[np.ndarray, np.ndarray, np.ndarray],
    head_counts: dict[str, Counter[int]],
    selected: Any,
    reference: Any | None,
    cutoff: int,
    parameters: SearchParameters = ALPHA,
    fitted: bool = True,
    summary_style: bool = False,
    power_law_fit: bool = False,
) -> None:
    """The featured metric (usually total degree) > K on linear and log-log axes,
    then the full-distribution head (skipped when ``head`` is None).

    ``fitted=False`` draws a candidate that the fit did not select.
    ``summary_style`` gives selected_<metric>_no_head and the summary figure their
    Title Case titles, such as "In-Degree Distribution (Linear Axes)", a legend on
    both panels, the legend names "Semantic Scholar citation graph" and
    "Generated examples (...)", and on the log-log panel the power law fitted to the
    generated examples (fit_power_law with minimum degree K + 1) when
    ``power_law_fit`` is True (off by default).
    """
    xlabel = METRIC_LABELS[parameters.featured][0]
    distribution = f"{xlabel.title()} Distribution"
    support, global_counts, generated_counts = counts
    series = {
        ("Semantic Scholar citation graph" if summary_style else "Global"): (
            global_counts / global_counts.sum(),
            GLOBAL_COLOR,
        ),
        (
            f"Generated examples ({parameters.math_label(selected)})"
            if summary_style
            else f"Generated (all question types, {parameters.math_label(selected)})"
        ): (
            generated_counts / generated_counts.sum(),
            GENERATED_COLOR,
        ),
    }
    # The linear panel stops once 99.9% of global papers are covered; the
    # log-log panel keeps the whole tail above the cutoff.
    limit = support[
        np.searchsorted(np.cumsum(global_counts) / global_counts.sum(), 0.999)
    ]
    for label, (probabilities, color) in series.items():
        style = {"marker": "o", "markersize": 3, "color": color}
        shown = support <= limit
        linear.plot(
            support[shown],
            probabilities[shown],
            linewidth=0.8,
            label=label,
            **style,
        )
        # Points only: lines would bridge tail degrees that have no papers.
        positive = (support > 0) & (probabilities > 0)
        log_log.plot(
            support[positive],
            probabilities[positive],
            linestyle="none",
            label=label,
            **style,
        )
    if summary_style and power_law_fit:
        # A power law is a straight line on log-log axes; draw it across the
        # generated degrees, normalized over degree > K like the points.
        minimum = cutoff + 1
        gamma = fit_power_law(support, generated_counts, minimum)
        observed = support[(support >= minimum) & (generated_counts > 0)]
        ends = np.array([observed.min(), observed.max()], dtype=float)
        log_log.plot(
            ends,
            ends**-gamma / zeta(gamma, minimum),
            color=FIT_COLOR,
            linewidth=1.5,
            label=rf"Power-law fit, $P(a) \propto a^{{-{gamma:.2f}}}$",
        )
    linear.set(
        title=f"{distribution} (Linear Axes)"
        if summary_style
        else f"Linear axes, degree > {cutoff}",
        xlabel=xlabel,
        ylabel="Probability",
    )
    log_log.set(
        title=f"{distribution} (Log-Log Axes)"
        if summary_style
        else f"Log-log axes, degree > {cutoff}",
        xlabel=xlabel,
        ylabel="Probability",
        xscale="log",
        yscale="log",
    )
    linear.legend()
    if summary_style:
        log_log.legend()
    if head is None:
        return
    plot_head(
        head,
        head_counts,
        selected,
        reference,
        cutoff,
        parameters=parameters,
        fitted=fitted,
    )
    head.set(
        title="Head (degree > 0)"
        + (f", {parameters.math_title} from degree > {cutoff} fit" if fitted else ""),
        xlabel=xlabel,
    )
    head.legend()


def save_figures(
    output: Path,
    ranking: pd.DataFrame,
    pooled: pd.DataFrame,
    distributions: dict,
    references: dict,
    method: str,
    selected: Any,
    cutoff: int,
    head_counts: dict[str, Counter[int]],
    reference: Any | None,
    parameters: SearchParameters = ALPHA,
    power_law_fit: bool = False,
) -> None:
    """``power_law_fit`` adds the power-law line to the summary panels (off by default)."""
    distance_label = "KL (nats)" if method == "kl" else "EMD (degree units)"
    metrics = parameters.metrics
    if len(parameters.names) == 1:
        ordered = ranking.sort_index()
        figure, ax = plt.subplots(figsize=(8, 4))
        plot_alpha_scores(ax, ranking, method, selected, cutoff)
        save_figure(figure, output, "alpha_scores")
        figure, axes = plt.subplots(
            1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False
        )
        axes = axes[0]
        for ax, metric in zip(axes, metrics):
            for question_type in EXPERIMENTS:
                rows = pooled[
                    (pooled.metric == metric) & (pooled.question_type == question_type)
                ].sort_values("alpha")
                ax.plot(
                    rows.alpha,
                    rows.distance,
                    marker="o",
                    color=QUESTION_TYPE_COLORS[question_type],
                    label=question_type,
                )
            ax.set(
                title=f"{metric.replace('_', ' ')}, degree > {cutoff}",
                xlabel=parameters.math_title,
                ylabel=distance_label,
            )
            ax.set_xticks(
                ordered.index, labels=[f"{alpha:g}" for alpha in ordered.index]
            )
        axes[0].legend()
    else:
        figure, axes = plt.subplots(1, 2, figsize=(14, 4.6))
        plot_score_heat_maps(
            *axes, ranking, method, selected, cutoff, parameters, summary_style=True
        )
        save_figure(figure, output, f"{parameters.stem}_scores")
        # One heat map per question type (row) and degree metric (column).
        figure, axes = plt.subplots(
            len(EXPERIMENTS),
            len(metrics),
            figsize=(6 * len(metrics), 10.5),
            squeeze=False,
        )
        for row_axes, question_type in zip(axes, EXPERIMENTS):
            for ax, metric in zip(row_axes, metrics):
                rows = pooled[
                    (pooled.metric == metric) & (pooled.question_type == question_type)
                ]
                plot_heat_map(
                    ax,
                    rows.set_index(list(parameters.names)).distance,
                    selected,
                    parameters,
                    colormap="viridis_r",
                    label=distance_label,
                    text=lambda value: f"{value:.3g}",
                )
                ax.set_title(
                    f"{question_type}, {metric.replace('_', ' ')}, degree > {cutoff}"
                )
    save_figure(figure, output, "distance_by_metric")
    for cumulative in (False, True):
        figure, axes = plt.subplots(
            1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False
        )
        axes = axes[0]
        for ax, metric in zip(axes, metrics):
            support, global_counts, _ = references[metric]
            lines = {"Global": global_counts / global_counts.sum()}
            for question_type in EXPERIMENTS:
                counts = pooled_counts(
                    distributions[(question_type, selected)]["examples"],
                    metric,
                    cutoff=cutoff,
                )
                weights = align_counts(counts, support)
                lines[question_type] = weights / weights.sum()
            for label, probabilities in lines.items():
                values = probabilities.cumsum() if cumulative else probabilities
                style = {
                    "label": label,
                    "color": QUESTION_TYPE_COLORS.get(label, GLOBAL_COLOR),
                }
                if cumulative:
                    ax.step(support, values, where="post", **style)
                else:
                    # Log-log view of the tail; zero probabilities cannot be shown.
                    shown = values > 0
                    ax.plot(
                        support[shown],
                        values[shown],
                        linestyle="none",
                        marker=".",
                        **style,
                    )
            ax.set_xscale("symlog", linthresh=1)
            if not cumulative:
                ax.set_yscale("log")
            ax.set(
                title=f"{metric.replace('_', ' ')}, degree > {cutoff}",
                xlabel="Degree",
                ylabel="Cumulative probability" if cumulative else "Probability",
            )
        axes[0].legend()
        save_figure(figure, output, "selected_cdf" if cumulative else "selected_pmf")
    featured_counts = total_degree_counts(
        distributions, references, selected, cutoff, parameters.featured
    )
    # The _no_head copy keeps only the linear and log-log panels.
    for name, panels in (("", 3), ("_no_head", 2)):
        figure, axes = plt.subplots(1, panels, figsize=(6 * panels, 4.5))
        plot_total_degree(
            axes[0],
            axes[1],
            axes[2] if panels == 3 else None,
            featured_counts,
            head_counts,
            selected,
            reference,
            cutoff,
            parameters,
            summary_style=panels == 2,
            power_law_fit=power_law_fit,
        )
        save_figure(figure, output, f"selected_{parameters.featured}{name}")
    save_selection_summary(
        output,
        ranking,
        featured_counts,
        method,
        selected,
        cutoff,
        parameters,
        power_law_fit=power_law_fit,
    )


def save_selection_summary(
    output: Path,
    ranking: pd.DataFrame,
    featured_counts: tuple[np.ndarray, np.ndarray, np.ndarray],
    method: str,
    selected: Any,
    cutoff: int,
    parameters: SearchParameters,
    power_law_fit: bool = False,
) -> None:
    """selected_<metric>_no_head (first row) above the candidate scores (second row).

    The second row is the score and the bootstrap selection frequency: the two
    theta_discount_scores heat maps, or for alpha the alpha_scores line and one
    bar per alpha. The panels are drawn by the same functions as those figures,
    so they match them.
    """
    single = len(parameters.names) == 1
    # TrueType fonts instead of Type 3 (some PDF checkers reject Type 3); the
    # figure looks the same.
    with plt.rc_context({"pdf.fonttype": 42}):
        figure, ((linear, log_log), (score, frequency)) = plt.subplots(
            2, 2, figsize=(14, 9.1)
        )
        plot_total_degree(
            linear,
            log_log,
            None,
            featured_counts,
            {},
            selected,
            None,
            cutoff,
            parameters,
            summary_style=True,
            power_law_fit=power_law_fit,
        )
        if single:
            plot_alpha_scores(
                score, ranking, method, selected, cutoff, summary_style=True
            )
            plot_alpha_frequencies(frequency, ranking, selected)
        else:
            plot_score_heat_maps(
                score,
                frequency,
                ranking,
                method,
                selected,
                cutoff,
                parameters,
                summary_style=True,
            )
        figure.tight_layout()
        if not single:
            # Each colorbar takes width from its heat map, so a heat-map title would
            # be centered left of the column; center it over the column, like the
            # first row.
            for top, bottom in ((linear, score), (log_log, frequency)):
                top_box, bottom_box = top.get_position(), bottom.get_position()
                bottom.title.set_x(
                    (top_box.x0 + top_box.width / 2 - bottom_box.x0) / bottom_box.width
                )
        # save_figure runs tight_layout again, with the centered titles in place, and
        # also saves each panel alone (the heat maps with their colorbars).
        metric = parameters.featured
        save_figure(
            figure,
            output,
            f"{parameters.stem}_selection_summary",
            panels={
                f"1_{metric}_linear": [linear],
                f"2_{metric}_log_log": [log_log],
                f"3_mean_{method}": [score]
                if single
                else [score, score.collections[0].colorbar.ax],
                "4_bootstrap_selection_frequency": [frequency]
                if single
                else [frequency, frequency.collections[0].colorbar.ax],
            },
        )


def load_selection_outputs(
    folder: Path, parameters: SearchParameters
) -> tuple[pd.DataFrame, tuple[np.ndarray, np.ndarray, np.ndarray], str, Any, int]:
    """Read the save_selection_summary inputs back from a finished K=<K> folder.

    Returns (ranking, featured counts, method, selected candidate, K), so figures
    can be redrawn without rerunning the search. distributions.json keeps the
    counts above K of every candidate, so the support is the one the search used.
    """
    selection = orjson.loads((folder / "selection.json").read_bytes())
    if selection["degree_metrics"] != list(parameters.metrics):
        raise ValueError(
            f"{folder} scores {selection['degree_metrics']}, but the parameters score "
            f"{list(parameters.metrics)}; pass the parameters of the search that wrote it."
        )
    distributions = orjson.loads((folder / "distributions.json").read_bytes())
    global_counter = Counter(
        {
            int(k): v
            for k, v in distributions["global"][parameters.featured]["counts"].items()
        }
    )
    generated = total_degree_by_candidate(distributions, parameters)
    support = np.array(sorted(set(global_counter).union(*generated.values())))
    selected = parameters.key(
        tuple(selection[f"selected_{name}"] for name in parameters.names)
    )
    ranking = pd.read_csv(folder / "ranking.csv").set_index(parameters.index)
    featured_counts = (
        support,
        align_counts(global_counter, support),
        align_counts(generated[selected], support),
    )
    return (
        ranking,
        featured_counts,
        selection["method"],
        selected,
        selection["degree_cutoff"],
    )


def save_cutoff_grid(
    output: Path, method: str, rows: list[dict], parameters: SearchParameters = ALPHA
) -> None:
    """One row per K: candidate scores, then the three selected_<metric> panels.

    Scores take one panel (alpha line plot) or two (theta/discount heat maps).
    """
    panels = 1 if len(parameters.names) == 1 else 2
    figure, axes = plt.subplots(
        len(rows),
        panels + 3,
        figsize=(26 if panels == 1 else 36, 4.5 * len(rows)),
        squeeze=False,
    )
    for row_axes, row in zip(axes, rows):
        linear, log_log, head = row_axes[panels:]
        scores = (row["ranking"], method, row["selected"], row["cutoff"])
        if panels == 1:
            plot_alpha_scores(row_axes[0], *scores)
        else:
            plot_score_heat_maps(*row_axes[:2], *scores, parameters)
        plot_total_degree(
            linear,
            log_log,
            head,
            row["featured_counts"],
            row["head_counts"],
            row["selected"],
            row["reference"],
            row["cutoff"],
            parameters,
        )
    save_figure(figure, output, f"{parameters.stem}_selection_by_K")


def save_head_check(
    method_dir: Path, cutoffs: list[int] | None = None, max_degree: int = 20
) -> None:
    """Show the head of each full distribution at the alpha selected from degree > K.

    Reads a finished search folder: summary_by_K.csv and K=0/distributions.json.
    Probabilities are normalized over all positive degrees, so degrees <= K, which
    the degree > K fit ignored, are compared on the same scale as the rest. One row
    per K (default: every K in summary_by_K.csv), one column per degree metric;
    question types are pooled. The alpha selected at K=0 is drawn as a dashed
    reference when it differs.
    """
    summary = pd.read_csv(method_dir / "summary_by_K.csv").set_index("K")
    cutoffs = sorted(summary.index.tolist()) if cutoffs is None else cutoffs
    missing = sorted({0, *cutoffs} - set(summary.index))
    if missing:
        raise ValueError(
            f"No search results for K={missing} in {method_dir}; rerun the search "
            "with these values in --degree-cutoffs."
        )
    distributions = orjson.loads(
        (method_dir / "K=0" / "distributions.json").read_bytes()
    )
    # A total-degree-only search saves only that metric.
    metrics = [m for m in METRICS if m in distributions["global"]]
    global_counts = {
        m: Counter({int(k): v for k, v in distributions["global"][m]["counts"].items()})
        for m in metrics
    }
    generated: dict[tuple[float, str], Counter[int]] = {}
    for record in distributions["generated"]:
        generated.setdefault((record["alpha"], record["metric"]), Counter()).update(
            {int(k): v for k, v in record["counts"].items()}
        )
    degrees = np.arange(1, max_degree + 1)
    reference = float(summary.loc[0, "selected_alpha"])
    rows = []
    figure, axes = plt.subplots(
        len(cutoffs),
        len(metrics),
        figsize=(6 * len(metrics), 4.5 * len(cutoffs)),
        squeeze=False,
    )
    for row_axes, cutoff in zip(axes, cutoffs):
        selected = float(summary.loc[cutoff, "selected_alpha"])
        for ax, metric in zip(row_axes, metrics):
            values = plot_head(
                ax,
                {
                    "global": global_counts[metric],
                    "selected": generated[(selected, metric)],
                    "reference": generated[(reference, metric)],
                },
                selected,
                reference,
                cutoff,
                max_degree,
            )
            ax.set(
                title=f"{metric.replace('_', ' ')}, {ALPHA.math_title} from degree > "
                f"{cutoff} fit",
                xlabel="Degree",
            )
            rows.extend(
                {
                    "K": cutoff,
                    "metric": metric,
                    "degree": int(degree),
                    "global_probability": values["global"][i],
                    "selected_alpha": selected,
                    "selected_probability": values["selected"][i],
                    "reference_alpha": reference,
                    "reference_probability": values["reference"][i],
                }
                for i, degree in enumerate(degrees)
            )
        row_axes[0].legend()
    save_figure(figure, method_dir, "head_check_by_K")
    pd.DataFrame(rows).to_csv(method_dir / "head_check_by_K.csv", index=False)


def total_degree_by_candidate(
    distributions: dict, parameters: SearchParameters
) -> dict[Any, Counter[int]]:
    """Featured-metric counts from a saved distributions.json, pooled over question types."""
    pooled: dict[Any, Counter[int]] = {}
    for record in distributions["generated"]:
        if record["metric"] == parameters.featured:
            key = parameters.key(tuple(record[name] for name in parameters.names))
            pooled.setdefault(key, Counter()).update(
                {int(k): v for k, v in record["counts"].items()}
            )
    return pooled


# Ordered lines in chapters 2-3: dark-to-light viridis colors plus a marker per
# value, so close neighbors stay apart when there are six or more values.
GROUP_MARKERS = ("o", "s", "^", "D", "v", "P", "X")


def plot_total_degree_group(
    linear: Any,
    log_log: Any,
    head: Any,
    support: np.ndarray,
    global_counts: np.ndarray,
    head_global: Counter[int],
    lines: list[tuple[str, np.ndarray, Counter[int], Any, str]],
    cutoff: int,
    max_degree: int = 20,
    xlabel: str = "Total degree",
) -> None:
    """Draw several candidates on the three selected_<metric> panels.

    lines holds (label, counts on support, positive-degree counts, color, marker).
    The global graph is drawn in dark purple on top of the candidates.
    """
    limit = support[
        np.searchsorted(np.cumsum(global_counts) / global_counts.sum(), 0.999)
    ]
    shown = support <= limit
    degrees = np.arange(1, max_degree + 1)
    if cutoff > 0:
        head.axvspan(
            0.5,
            cutoff + 0.5,
            color="gray",
            alpha=0.15,
            label=f"degree <= {cutoff}: not used in the fit",
        )
    for label, counts, head_counts, color, marker in [
        ("Global", global_counts, head_global, GLOBAL_COLOR, "o"),
        *lines,
    ]:
        probabilities = counts / counts.sum()
        style = {
            "color": color,
            "marker": marker,
            "markersize": 3.5,
            "zorder": 3 if label == "Global" else 2,
        }
        linear.plot(
            support[shown], probabilities[shown], linewidth=1, label=label, **style
        )
        # Points only: lines would bridge tail degrees that have no papers.
        positive = (support > 0) & (probabilities > 0)
        log_log.plot(
            support[positive], probabilities[positive], linestyle="none", **style
        )
        head.plot(
            degrees,
            np.array([head_counts[d] for d in degrees]) / sum(head_counts.values()),
            linewidth=1,
            **style,
        )
    linear.set(
        title=f"Linear axes, degree > {cutoff}",
        xlabel=xlabel,
        ylabel="Probability",
    )
    log_log.set(
        title=f"Log-log axes, degree > {cutoff}",
        xlabel=xlabel,
        ylabel="Probability",
        xscale="log",
        yscale="log",
    )
    head.set(
        title="Head (degree > 0)",
        xlabel=xlabel,
        ylabel="P(degree | degree > 0)",
    )
    head.set_xticks(degrees)
    linear.legend(fontsize=8)
    if cutoff > 0:
        head.legend(fontsize=8)


def add_pdf_outline(path: Path, outline: list[tuple[int, str, int]]) -> None:
    """Add bookmarks to a PDF: (level 0 for a chapter or 1 inside it, title, page)."""
    writer = PdfWriter(clone_from=io.BytesIO(path.read_bytes()))
    chapter = None
    for level, title, page in outline:
        item = writer.add_outline_item(title, page, parent=chapter if level else None)
        if level == 0:
            chapter = item
    writer.page_mode = "/UseOutlines"
    buffer = io.BytesIO()
    writer.write(buffer)
    path.write_bytes(buffer.getvalue())


def save_all_pairs_total_degree(
    method_dir: Path,
    cutoffs: list[int] | None = None,
    parameters: SearchParameters = THETA_DISCOUNT,
    max_degree: int = 20,
) -> list[Path]:
    """Write K=<K>/<parameters.all_candidates_pdf>, with chapters and bookmarks.

    <metric> is parameters.featured, so a theta/discount search that scores total
    degree writes all_pairs_total_degree.pdf. Chapter 1 has one page per candidate,
    best rank first, with the three selected_<metric> panels. For (theta, discount),
    chapter 2 has one page per discount value, with every theta drawn on the same
    panels (sorted by theta), and chapter 3 has one page per theta value, with every
    discount (sorted by discount). For alpha (all_alphas_<metric>.pdf), chapter 2
    is one page with every alpha (sorted by alpha). Every page uses the same axis
    ranges. Reads only the saved search output: ranking.csv and distributions.json
    of each K, plus K=0/distributions.json for the head panel (default K values:
    every K in summary_by_K.csv).
    """
    single = len(parameters.names) == 1
    summary = pd.read_csv(method_dir / "summary_by_K.csv").set_index("K")
    cutoffs = sorted(summary.index.tolist()) if cutoffs is None else cutoffs
    missing = sorted({0, *cutoffs} - set(summary.index))
    if missing:
        raise ValueError(
            f"No search results for K={missing} in {method_dir}; rerun the search "
            "with these values in --degree-cutoffs."
        )
    metric = parameters.featured
    head_data = orjson.loads((method_dir / "K=0" / "distributions.json").read_bytes())
    head_global = Counter(
        {int(k): v for k, v in head_data["global"][metric]["counts"].items()}
    )
    head_generated = total_degree_by_candidate(head_data, parameters)
    degrees = np.arange(1, max_degree + 1)
    outputs = []
    for cutoff in cutoffs:
        folder = method_dir / f"K={cutoff}"
        distributions = orjson.loads((folder / "distributions.json").read_bytes())
        method = orjson.loads((folder / "selection.json").read_bytes())["method"]
        score_label = "mean KL (nats)" if method == "kl" else "mean relative EMD"
        ranking = pd.read_csv(folder / "ranking.csv")
        global_counter = Counter(
            {int(k): v for k, v in distributions["global"][metric]["counts"].items()}
        )
        generated = total_degree_by_candidate(distributions, parameters)
        support = np.array(sorted(set(global_counter).union(*generated.values())))
        global_counts = align_counts(global_counter, support)
        candidates = [
            parameters.key(tuple(row[name] for name in parameters.names))
            for _, row in ranking.iterrows()
        ]
        rows = dict(zip(candidates, (row for _, row in ranking.iterrows())))
        ranks = {key: rank for rank, key in enumerate(candidates, start=1)}
        counts = {key: align_counts(generated[key], support) for key in candidates}
        # Shared axis ranges: the same limits on every page of this K.
        limit = support[
            np.searchsorted(np.cumsum(global_counts) / global_counts.sum(), 0.999)
        ]
        probabilities = [c / c.sum() for c in (global_counts, *counts.values())]
        linear_top = max(p[support <= limit].max() for p in probabilities)
        positive = [p[(support > 0) & (p > 0)] for p in probabilities]
        log_bottom = min(p.min() for p in positive)
        log_top = max(p.max() for p in positive)
        head_top = max(
            max(c[d] for d in degrees) / sum(c.values())
            for c in (head_global, *(head_generated[key] for key in candidates))
        )
        path = folder / parameters.all_candidates_pdf
        outline: list[tuple[int, str, int]] = []
        with PdfPages(path) as pdf:

            def save_page(figure: Any, title: str, axes: Any = None) -> None:
                if axes is not None:
                    linear, log_log, head = axes
                    linear.set_ylim(0, linear_top * 1.05)
                    log_log.set_xlim(
                        support[support > 0].min() / 1.3, support.max() * 1.3
                    )
                    log_log.set_ylim(log_bottom / 2, log_top * 2)
                    head.set_ylim(0, head_top * 1.05)
                figure.suptitle(title, fontsize=13)
                figure.tight_layout(rect=(0, 0, 1, 0.95))
                pdf.savefig(figure)
                plt.close(figure)

            def chapter_page(title: str, lines: list[str]) -> None:
                outline.append((0, title, pdf.get_pagecount()))
                figure = plt.figure(figsize=(18, 5))
                figure.text(0.5, 0.62, title, ha="center", va="center",
                            fontsize=24, weight="bold")
                figure.text(0.5, 0.38, "\n".join(lines), ha="center", va="center",
                            fontsize=12, linespacing=1.8)
                pdf.savefig(figure)
                plt.close(figure)

            shared = (
                f"K = {cutoff}: the first two panels use degrees > {cutoff}; the head "
                f"panel shows degrees 1-{max_degree} of the full distribution. "
                "Dark purple: global graph. All pages share the same axis ranges."
            )
            chapter_page(
                "Chapter 1: sorted by rank",
                [
                    f"One page per {parameters.noun if single else '(theta, discount) pair'}, "
                    f"best first, with the three selected_{metric} panels.",
                    shared,
                ],
            )
            for key in candidates:
                rank, row = ranks[key], rows[key]
                outline.append((1, f"Rank {rank}: {parameters.label(key)}", pdf.get_pagecount()))
                figure, axes = plt.subplots(1, 3, figsize=(18, 5))
                plot_total_degree(
                    *axes,
                    (support, global_counts, counts[key]),
                    {"global": head_global, "selected": head_generated[key]},
                    key,
                    None,
                    cutoff,
                    parameters,
                    fitted=rank == 1,
                )
                save_page(
                    figure,
                    f"Chapter 1 · Rank {rank}/{len(candidates)}: "
                    f"{parameters.math_label(key)}"
                    + (" (selected)" if rank == 1 else "")
                    + f"  |  {score_label} {row['score']:.4g}  |  selected in "
                    f"{row['bootstrap_selection_frequency']:.1%} of bootstrap resamples",
                    axes,
                )
            # Chapter 2 groups by the second parameter, chapter 3 by the first; with
            # one parameter, chapter 2 is one page with every candidate.
            groupings = ((2, 0, None),) if single else ((2, 0, 1), (3, 1, 0))
            for chapter, line_index, group_index in groupings:
                line_name = parameters.names[line_index]
                # Bookmarks and chapter pages use names, plot pages use symbols.
                line_symbol = parameters.symbols[line_index]
                line_values = sorted({parameters.values(k)[line_index] for k in candidates})
                # The same value keeps its color and marker on every page.
                colors = dict(zip(line_values, viridis_colors(len(line_values))))
                markers = dict(zip(line_values, itertools.cycle(GROUP_MARKERS)))
                legend_note = (
                    f"Each colored line is one {line_name} value (darker = smaller "
                    f"{line_name}); the legend gives each {'alpha' if single else 'pair'}'s "
                    "rank and score."
                )
                # (bookmark, page heading, candidates on the page)
                if group_index is None:
                    chapter_page(
                        f"Chapter {chapter}: every {line_name} on one page",
                        [legend_note, shared],
                    )
                    groups = [(f"every {line_name}", f"Every ${line_symbol}$", candidates)]
                else:
                    group_name = parameters.names[group_index]
                    group_symbol = parameters.symbols[group_index]
                    chapter_page(
                        f"Chapter {chapter}: sorted by {line_name}, grouped by {group_name}",
                        [f"One page per {group_name} value. {legend_note}", shared],
                    )
                    groups = [
                        (
                            f"{group_name} = {group_value:g}",
                            f"${group_symbol}={group_value:g}$",
                            [
                                k
                                for k in candidates
                                if parameters.values(k)[group_index] == group_value
                            ],
                        )
                        for group_value in sorted(
                            {parameters.values(k)[group_index] for k in candidates}
                        )
                    ]
                for bookmark, heading, members in groups:
                    members = sorted(
                        members, key=lambda k: parameters.values(k)[line_index]
                    )
                    outline.append((1, bookmark, pdf.get_pagecount()))
                    lines = []
                    for key in members:
                        value = parameters.values(key)[line_index]
                        lines.append(
                            (
                                f"${line_symbol}={value:g}$: rank {ranks[key]}, "
                                f"{score_label.split(' (')[0]} {rows[key]['score']:.3g}"
                                + (" (selected)" if ranks[key] == 1 else ""),
                                counts[key],
                                head_generated[key],
                                colors[value],
                                markers[value],
                            )
                        )
                    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
                    plot_total_degree_group(
                        *axes,
                        support,
                        global_counts,
                        head_global,
                        lines,
                        cutoff,
                        max_degree,
                        METRIC_LABELS[metric][0],
                    )
                    best = min(members, key=ranks.get)
                    save_page(
                        figure,
                        f"Chapter {chapter} · {heading}, sorted by "
                        f"${line_symbol}$  |  best in this group: "
                        f"{parameters.math_label(best)} "
                        f"(rank {ranks[best]}/{len(candidates)})",
                        axes,
                    )
        add_pdf_outline(path, outline)
        outputs.append(path)
    return outputs


def save_power_law_analysis(
    output: Path,
    datasets: dict,
    global_data: dict,
    selected: Any,
    cutoff: int,
    parameters: SearchParameters,
    unit: str,
    repeats: int,
    seed: int,
) -> dict:
    """Write power_law.json and power_law.md for one K=<K> folder.

    Analyzes the featured metric above K (minimum degree K + 1, the degrees that
    the selection compares) for the generated examples at the selected candidate,
    pooled over the three tasks, and for the global graph; see power_law.py. The
    interval of gamma resamples context groups within each task for the generated
    examples, as the KL bootstrap does, and papers for the global graph.
    """
    metric, minimum = parameters.featured, cutoff + 1
    rng = np.random.default_rng(seed)
    by_task = {
        task: pooled_counts(datasets[(task, selected)]["examples"], metric, cutoff=cutoff)
        for task in EXPERIMENTS
    }
    global_counter = pooled_counts([global_data], metric, cutoff=cutoff)
    support = np.array(sorted(set(global_counter).union(*by_task.values())))
    generated = np.sum([align_counts(c, support) for c in by_task.values()], axis=0)
    global_counts = align_counts(global_counter, support)
    generated_replicates = np.zeros((repeats, len(support)))
    for task in EXPERIMENTS:
        groups = group_examples(datasets[(task, selected)]["examples"], unit)
        matrix = np.array(
            [
                align_counts(
                    sum((histogram(e, metric, cutoff=cutoff) for e in g), Counter()),
                    support,
                )
                for g in groups
            ]
        )
        weights = rng.multinomial(
            len(groups), np.full(len(groups), 1 / len(groups)), size=repeats
        )
        generated_replicates += weights @ matrix
    global_replicates = rng.multinomial(
        int(global_counts.sum()), global_counts / global_counts.sum(), size=repeats
    )
    task_fits = {}
    for task, counter in by_task.items():
        counts = align_counts(counter, support)
        kept = (support >= minimum) & (counts > 0)
        gamma = fit_power_law(support[kept], counts[kept], minimum)
        task_fits[task] = {
            "papers": int(counts[kept].sum()),
            "gamma": gamma,
            "ks_distance": ks_distance(support[kept], counts[kept], gamma, minimum),
        }
    results = {
        "metric": metric,
        "degree_cutoff": cutoff,
        "minimum_degree": minimum,
        "selected": parameters.columns(selected),
        "settings": {
            "model": "P(a) = a^-gamma / zeta(gamma, minimum_degree), a >= minimum_degree",
            "estimator": "maximum likelihood (Clauset, Shalizi, and Newman 2009)",
            "bootstrap_repeats": repeats,
            "seed": seed,
            "generated_interval_resampling": f"{unit.replace('_', ' ')}s within each task",
            "global_interval_resampling": "papers",
            "goodness_of_fit": "KS distance; parametric bootstrap with papers drawn "
            "independently from the fitted law and the minimum degree fixed",
            "significance": SIGNIFICANCE,
        },
        "generated": analyze_power_law(
            support, generated, minimum, repeats, rng, generated_replicates
        ),
        "global": analyze_power_law(
            support, global_counts, minimum, repeats, rng, global_replicates
        ),
        "generated_by_task": task_fits,
    }
    (output / "power_law.json").write_bytes(
        orjson.dumps(results, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY)
        + b"\n"
    )
    (output / "power_law.md").write_text(
        power_law_report(results, parameters.label(selected))
    )
    return results


def format_p_value(value: float | None, repeats: int | None = None) -> str:
    """A bootstrap p-value of 0 means below 1/repeats."""
    if value is None:
        return "not tested"
    if value == 0:
        return f"< {1 / repeats:.2g}" if repeats else "< 1e-300"
    return f"{value:.3f}" if value >= 0.001 else f"{value:.1e}"


def p_text(value: float | None, repeats: int | None = None) -> str:
    """'p = 0.087', 'p < 0.001', or 'no test'."""
    if value is None:
        return "no test"
    formatted = format_p_value(value, repeats)
    return f"p {formatted}" if formatted.startswith("<") else f"p = {formatted}"


def power_law_report(results: dict, generated_label: str) -> str:
    """power_law.md: the tables of power_law.json and how to read them."""
    metric_text = METRIC_LABELS[results["metric"]][1]
    minimum, repeats = results["minimum_degree"], results["settings"]["bootstrap_repeats"]
    names = {
        "generated": f"Generated examples ({generated_label})",
        "global": "Filtered Semantic Scholar citation graph",
    }

    def fit_table(key: str) -> list[str]:
        fits = {name: results[name].get(key) for name in names}
        lines = ["| | " + " | ".join(names.values()) + " |", "|---|---|---|"]
        rows = [
            ("Minimum degree $a_{min}$", lambda f: str(f["minimum_degree"])),
            (f"Papers with {metric_text} >= $a_{{min}}$", lambda f: f"{f['papers']:,}"),
            (
                "Distinct degrees (largest)",
                lambda f: f"{f['distinct_degrees']} ({f['largest_degree']})",
            ),
            ("gamma", lambda f: f"{f['gamma']:.4f}"),
            ("Standard error (Fisher information)", lambda f: f"{f['gamma_standard_error']:.4f}"),
            (
                "95% bootstrap interval of gamma",
                lambda f: "[{:.4f}, {:.4f}]".format(*f["gamma_95_interval"]),
            ),
            ("KS distance D", lambda f: f"{f['goodness_of_fit']['ks_distance']:.5f}"),
            (
                "Synthetic D: median, 95th percentile",
                lambda f: f"{f['goodness_of_fit']['synthetic_ks_median']:.5f}, "
                f"{f['goodness_of_fit']['synthetic_ks_95th_percentile']:.5f}",
            ),
            (
                "Goodness-of-fit p-value",
                lambda f: format_p_value(f["goodness_of_fit"]["p_value"], repeats),
            ),
            (
                f"Power law plausible (p > {SIGNIFICANCE:g})",
                lambda f: "yes" if f["goodness_of_fit"]["power_law_plausible"] else "no",
            ),
        ]
        for index, (label, cell) in enumerate(rows):
            values = []
            for name, fit in fits.items():
                if fit is None:
                    # No separate KS-optimal fit: it is the main fit, or the scan is empty.
                    same = results[name].get("ks_optimal_minimum") == minimum
                    values.append("same as above" if same else "-")
                elif "skipped" in fit:
                    values.append(f"skipped: {fit['skipped']}" if index == 0 else "-")
                else:
                    values.append(cell(fit))
            lines.append(f"| {label} | " + " | ".join(values) + " |")
        return lines

    def alternative_table(key: str) -> list[str]:
        lines = [
            "| Alternative | " + " | ".join(names.values()) + " |",
            "|---|---|---|",
        ]
        for alternative, title in (
            ("log_normal", "Log-normal"),
            ("exponential", "Exponential"),
            ("power_law_with_cutoff", "Power law with exponential cutoff"),
        ):
            values = []
            for name in names:
                fit = results[name].get(key)
                if fit is None or "skipped" in fit:
                    same = fit is None and results[name].get("ks_optimal_minimum") == minimum
                    values.append("same as above" if same else "-")
                    continue
                test = fit["alternatives"][alternative]
                values.append(
                    f"R = {test['log_likelihood_ratio']:.2f}, "
                    f"{p_text(test['p_value'])}: {test['favored']}"
                )
            lines.append(f"| {title} | " + " | ".join(values) + " |")
        return lines

    def conclusion(name: str, key: str = "at_minimum") -> str:
        fit = results[name][key]
        if "skipped" in fit:
            return f"- {names[name]}: not analyzed ({fit['skipped']})."
        test = fit["goodness_of_fit"]
        verdict = "is plausible" if test["power_law_plausible"] else "is rejected"
        low, high = fit["gamma_95_interval"]
        better = [
            t["favored"]
            for t in fit["alternatives"].values()
            if t["favored"] != "power law" and not t["favored"].startswith("neither")
        ]
        comparison = (
            "an alternative fits significantly better: " + ", ".join(better)
            if better
            else "no alternative fits significantly better"
        )
        where = "" if key == "at_minimum" else ", the minimum with the smallest KS distance"
        return (
            f"- {names[name]}: a power law {verdict} for {metric_text} >= "
            f"{fit['minimum_degree']}{where} (gamma = {fit['gamma']:.3f}, 95% interval "
            f"[{low:.3f}, {high:.3f}], {p_text(test['p_value'], repeats)}); {comparison}."
        )

    scans = {name: {row["minimum_degree"]: row for row in results[name]["minimum_scan"]} for name in names}
    best = {name: results[name].get("ks_optimal_minimum") for name in names}
    scan_lines = [
        "| $a_{min}$ | " + " | ".join(
            f"{label}: papers, gamma, D" for label in ("Generated", "Semantic Scholar")
        ) + " |",
        "|---|---|---|",
    ]
    for value in sorted(set(scans["generated"]) | set(scans["global"])):
        cells = []
        for name in names:
            row = scans[name].get(value)
            cells.append(
                "-" if row is None else f"{row['papers']:,}, {row['gamma']:.4f}, "
                f"{row['ks_distance']:.5f}" + (" (smallest D)" if best[name] == value else "")
            )
        scan_lines.append(f"| {value} | " + " | ".join(cells) + " |")
    if len(scan_lines) == 2:
        scan_lines = ["No minimum degree leaves enough papers for the scan."]
    task_lines = ["| Task | Papers | gamma | KS distance D |", "|---|---|---|---|"] + [
        f"| {task} | {fit['papers']:,} | {fit['gamma']:.4f} | {fit['ks_distance']:.5f} |"
        for task, fit in results["generated_by_task"].items()
    ]
    settings = results["settings"]
    return "\n".join(
        [
            f"# Power-law analysis of the {metric_text} distributions, K = {results['degree_cutoff']}",
            "",
            f"Model: the discrete power law $P(a) = a^{{-\\gamma}} / \\zeta(\\gamma, a_{{min}})$ for "
            f"{metric_text} $a \\geq a_{{min}}$, fitted by maximum likelihood (Clauset, Shalizi, "
            "and Newman 2009, *Power-law distributions in empirical data*, SIAM Review). "
            f"The main fit uses $a_{{min}} = K + 1 = {minimum}$, the degrees that the selection compares. "
            "The generated examples are pooled over the three tasks.",
            "",
            f"Settings: {repeats} bootstrap repeats, seed {settings['seed']}. The interval of "
            f"gamma resamples {settings['generated_interval_resampling']} for the generated "
            "examples and papers for the Semantic Scholar graph. Goodness of fit: "
            f"{settings['goodness_of_fit']}. Significance level {SIGNIFICANCE:g}.",
            "",
            "## Conclusion",
            "",
            conclusion("generated"),
            conclusion("global"),
            *(
                conclusion(name, "at_ks_optimal_minimum")
                for name in names
                if "at_ks_optimal_minimum" in results[name]
            ),
            "",
            f"## Fit at $a_{{min}} = {minimum}$",
            "",
            *fit_table("at_minimum"),
            "",
            "## Comparison with other distributions",
            "",
            "R is the log-likelihood ratio, power law minus alternative; R > 0 favors the "
            "power law. Log-normal and exponential use the Vuong test (two-sided). The power "
            "law with cutoff contains the power law, so it uses the nested test (2 x gain "
            "against chi-squared with one degree of freedom); a small p means the cutoff "
            f"fits significantly better. A test decides only when p < {SIGNIFICANCE:g}. "
            "\"The fit converges to the power law\" means the best fit of that family is the "
            "power law itself, so the two cannot be told apart.",
            "",
            *alternative_table("at_minimum"),
            "",
            "## Minimum degree with the smallest KS distance",
            "",
            "Clauset et al. choose the minimum degree that makes the fit closest to the data. "
            "Its p-value keeps this minimum fixed, so it is optimistic.",
            "",
            *fit_table("at_ks_optimal_minimum"),
            "",
            *alternative_table("at_ks_optimal_minimum"),
            "",
            "## Minimum-degree scan",
            "",
            f"Each row fits only {metric_text} >= $a_{{min}}$. Rows stop when fewer than "
            f"{SCAN_MIN_PAPERS} papers, or less than {SCAN_MIN_SHARE:.0%} of the papers at "
            f"$a_{{min}} = {minimum}$, remain; the smallest KS distance is chosen among these rows.",
            "",
            *scan_lines,
            "",
            f"## Generated examples by task ($a_{{min}} = {minimum}$)",
            "",
            *task_lines,
            "",
            "## How to read these results",
            "",
            f"- A goodness-of-fit p-value above {SIGNIFICANCE:g} means a power law is a plausible "
            "model; it does not prove that the data follow one.",
            "- With many papers, the test detects even small deviations, so a rejection can come "
            "from a small but systematic gap (compare D with the synthetic D).",
            "- The goodness-of-fit bootstrap draws papers independently. Papers in one example are "
            "not independent, so the test is, if anything, too strict for the generated examples.",
            "- The interval of gamma for the generated examples resamples whole context groups, "
            "so it includes the variation between examples.",
            "",
        ]
    )


def save_report(
    output: Path,
    summary: dict,
    ranking: pd.DataFrame,
    parameters: SearchParameters = ALPHA,
) -> None:
    p = parameters
    selected = p.key(tuple(summary[f"selected_{name}"] for name in p.names))
    row = ranking.loc[selected]
    method = summary["method"]
    cutoff = summary["degree_cutoff"]
    second_row = (
        f"the {p.stem}_scores line and one bar per {p.noun} for the bootstrap selection"
        "\n  frequency (second row) in one 2 x 2 figure."
        if len(p.names) == 1
        else f"{p.stem}_scores (second row) in one 2 x 2 figure, with the same panels."
    )
    heat_map_note = (
        (
            ""
            if len(p.names) == 1
            else f"\n  {p.stem}_scores has two heat maps (score and bootstrap selection "
            "frequency);\n  distance_by_metric has one per question type and degree metric."
        )
        + f"\n- {p.stem}_selection_summary: selected_{p.featured}_no_head (first row) above"
        f"\n  {second_row}"
        f"\n  Each panel is also saved alone: 1_{p.featured}_linear, 2_{p.featured}_log_log,"
        f"\n  3_mean_{method}, and 4_bootstrap_selection_frequency (PNG and PDF)."
    )
    description = (
        f"KL(generated || global), with natural logarithms (nats). Both probability distributions are mixed with a uniform distribution using epsilon={summary['kl_epsilon']:g}. The fixed support is the union of degrees in the global reference and all calibration candidates. KL is averaged directly, without degree scaling."
        if method == "kl"
        else "Wasserstein-1 (Earth Mover's Distance) on the original degree values, using normalized histogram counts. Each distance is divided by its fixed global mean degree before aggregation. Raw distances remain in degree units. No smoothing, clipping, or logarithmic transformation is applied to the calculation."
    )
    if p.metrics == METRICS:
        metric_phrase = "three degree metrics"
        related_note = " The three degree metrics are related, not independent evidence."
        draws_note = "The three degree metrics share the same draws. "
    else:
        metric_phrase = (
            "the "
            + " and ".join(METRIC_LABELS[metric][1] for metric in p.metrics)
            + (" metrics" if len(p.metrics) > 1 else " metric")
        )
        related_note = draws_note = ""
    # "total-degree distributions", "in-degree distributions", ...
    featured_text = METRIC_LABELS[p.featured][1].replace(" ", "-")
    distance_count = {3: "three", 6: "six", 9: "nine"}[
        len(EXPERIMENTS) * len(p.metrics)
    ]
    text = f"""# {method.upper()} {p.title} search, degree > {cutoff}: methods and results

Degree cutoff: **K = {cutoff}** (compare degrees greater than {cutoff} only).
Selected {p.noun}: **{p.text(selected)}**. Calibration score: **{row["score"]:.8g}**.
Pointwise 95% percentile bootstrap interval: [{row["score_ci_low"]:.8g}, {row["score_ci_high"]:.8g}].
Bootstrap selection frequency: {row["bootstrap_selection_frequency"]:.3%}.
Status: **{summary["selection_status"]}**.

## Method

{description}

Candidate {p.plural}: {summary[f"candidate_{p.plural_key}"]}.
Keep only degrees greater than K = {cutoff} in both global and generated histograms,
separately for each metric, then normalize the remaining counts to compare
P(degree | degree > {cutoff}). This also applies to validation, bootstrap resamples,
smoothing support, the global mean degree used by EMD, and figures. A requested K is skipped
when a pooled global or generated distribution has no paper above K, because its conditional
distribution is undefined; skipped values are listed in ../skipped_K.csv.
A single QA example with no paper above K has no per-example distance, but still takes
part in pooled and bootstrap histograms (where it adds nothing). Original graph counts
are validated before filtering; mean_num_papers still describes the original graphs.
This K is one of the cutoffs in this run; ../summary_by_K.csv and the one-row-per-K
figure ../{p.stem}_selection_by_K.png compare all of them.
Every K reuses the same bootstrap seed and groups, so its draws match the other K values.
Within each question type, add positive-degree counts across QA examples, then normalize.
Average the resulting scores equally across the three question types and {metric_phrase}.
Larger examples contribute more paper observations within a type. Each QA example counts
separately, including examples sharing a context. Repeated papers in different examples
count again.{related_note}
Choose the smallest score; exact point-estimate ties favor the {p.tie_break}.
Per-example mean and median distances are supporting diagnostics, not the selection rule.

## Uncertainty and validation

Bootstrap repeats: {summary["bootstrap"]["repeats"]}; seed: {summary["bootstrap"]["seed"]};
resampling unit: {summary["bootstrap"]["unit"]}. Resample groups within each question type
and {p.noun}. Keep all examples in a selected group together and reconstruct pooled histograms.
{draws_note}The global reference is held fixed.
Groups are identified within input files; missing context IDs fall back to qid.
Groups, question types, and {p.plural} are assumed independent; no cross-{p.title} pairing is inferred.
If larger generation units are shared, use whole independent source files or independent runs.
Selection frequencies split credit between tied bootstrap minima. They are stability estimates,
not posterior probabilities or significance tests. Intervals are not simultaneous across {p.plural}.
Score-difference intervals compare each candidate to the calibration-selected {p.noun} and are
descriptive, not selection-adjusted hypothesis tests.

Validation score for this method's fixed selected {p.noun}: {summary["validation_score"]}.
If no independent validation was provided, this is a calibration choice only. New independent
generation runs are needed to assess generalization; different paths alone do not prove independence.

## Scope and interpretation

The generated graphs include needle graphs and all stored distractor trees before prompt trimming.
The input global graph's cutoff/category and all source settings are in selection.json.
Cutoff agreement is checked; category eligibility must be verified in the generation configuration.
Search graphs can be structurally constrained, so the result is the closest tested {p.noun} under this
criterion, not evidence of a perfect global match or general graph realism.
KL remains the primary criterion for the overall study; EMD provides a separate sensitivity analysis.
Do not switch the primary criterion, or the primary cutoff K, after observing the winners;
report the other cutoffs as a sensitivity analysis.

## Artifacts

- ranking.csv: scores, all {distance_count} distances, intervals, and selection frequencies.
- ranking.tex: compact LaTeX table of scores, intervals, and selection frequencies.
- pooled_distances.csv: distances, graph sizes, and means for each type/{p.title}/degree metric.
- per_example_distances.csv / per_example_summary.csv: individual and summarized diagnostics;
  examples with no paper above K have no row, so the summary count can be below num_examples.
- bootstrap_scores.csv / score_differences.csv: resampling results and differences from the selected {p.noun}.
- distributions.json: global and pooled counts and probabilities for reproducing plots/comparisons.
- selection.json: method, K, input SHA-256 hashes, source settings, command, software versions, and results.
- {p.stem}_scores, distance_by_metric, selected_pmf (log-log axes), selected_cdf: PNG and PDF figures.{heat_map_note}
- selected_{p.featured}: PNG and PDF figure of the global and generated {featured_text} distributions
  at the selected {p.noun}, with the three question types pooled. The linear and log-log panels
  show probabilities conditional on {METRIC_LABELS[p.featured][1]} > {cutoff}. The third panel checks the head:
  degrees 1-20 of the full distributions, normalized over all positive degrees, with degrees
  <= {cutoff} (not used in the fit) shaded and the {p.noun} selected at K=0 dashed for reference.
- selected_{p.featured}_no_head: the same figure with only the linear and log-log panels;
  with --power-law-fit, the log-log panel also shows the power law fitted to the generated
  examples by maximum likelihood for degrees above K (Clauset, Shalizi, and Newman 2009).
- validation_distances.csv: held-out results, only when validation was requested.
- smoothing_sensitivity.csv: KL-only sensitivity to epsilon; unsmoothed KL infinities are preserved as inf in CSV.

Figures show unsmoothed distributions; logarithmic degree axes are display choices only.
The KL smoothing check does not change the configured primary epsilon.

References: [SciPy KL](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html),
[SciPy EMD](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html),
[selection bias](https://www.jmlr.org/papers/v11/cawley10a.html).
"""
    (output / "report.md").write_text(text)
    columns = [
        "score",
        "score_ci_low",
        "score_ci_high",
        "bootstrap_selection_frequency",
    ]
    latex = [
        r"\begin{tabular}{" + "r" * (len(p.names) + 4) + "}",
        r"\hline",
        " & ".join(p.latex_names)
        + r" & Score & CI lower & CI upper & Selection frequency \\",
        r"\hline",
    ]
    for alpha, values in ranking.iterrows():
        latex.append(
            " & ".join(
                [
                    *[f"{value:g}" for value in p.values(alpha)],
                    *[f"{values[c]:.6g}" for c in columns],
                ]
            )
            + r" \\"
        )
    latex.extend([r"\hline", r"\end{tabular}"])
    (output / "ranking.tex").write_text("\n".join(latex) + "\n")


METHODS = ("kl", "emd")
# Each worker process uses one math-library thread, so N workers use about N cores.
SINGLE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
# Experiment data and validation cache kept by one worker process for all its tasks.
SEARCH_WORKER_STATE: dict[str, Any] = {}


@dataclass
class SearchSetup:
    """Validated inputs that every cutoff K of one global file shares."""

    global_path: Path
    global_data: dict
    global_provenance: dict
    datasets_key: tuple
    output_dir: Path
    alphas: list[Any]
    cutoffs: list[int]
    skipped: dict[int, str]
    head_global: Counter[int]
    head_generated: dict[Any, Counter[int]]
    parameters: SearchParameters
    repeats: int
    seed: int
    epsilon: float
    unit: str
    validation_root: Path | None
    command: str
    label: str
    power_law_fit: bool = False
    power_law_repeats: int = 0


def prepare_search(
    global_path: Path,
    experiment_root: Path,
    output_dir: Path,
    alphas: list[Any],
    repeats: int = 500,
    seed: int = 2026,
    epsilon: float = 1e-8,
    unit: str = "context_group",
    validation_root: Path | None = None,
    command: str = "",
    cutoffs: list[int] | None = None,
    cache: dict | None = None,
    label: str = "",
    parameters: SearchParameters = ALPHA,
    power_law_fit: bool = False,
    power_law_repeats: int = 0,
) -> tuple[SearchSetup, dict]:
    """Validate inputs, load the experiments, and drop each K that cannot be compared.

    Returns the shared setup and the loaded experiment datasets.
    """
    cutoffs = [0] if cutoffs is None else sorted(cutoffs)
    alphas = [parameters.key(alpha) for alpha in alphas]
    if (
        not alphas
        or len(set(alphas)) != len(alphas)
        or any(
            not np.isfinite(parameters.values(a)).all()
            or not parameters.check(parameters.values(a))
            for a in alphas
        )
    ):
        raise ValueError(parameters.rule)
    if (
        not cutoffs
        or len(set(cutoffs)) != len(cutoffs)
        or any(type(k) is not int or k < 0 for k in cutoffs)
    ):
        raise ValueError("Degree cutoffs K must be distinct nonnegative integers.")
    if repeats < 2 or not np.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("Use at least two bootstrap repeats and 0 < epsilon < 1.")
    if unit not in {"context_group", "source_file"}:
        raise ValueError("Unknown bootstrap unit.")
    if (
        validation_root is not None
        and validation_root.resolve() == experiment_root.resolve()
    ):
        raise ValueError("Validation and calibration roots must differ.")
    alphas = sorted(alphas)
    global_data, global_provenance = read_json(global_path)
    global_histograms = {m: histogram(global_data, m, cutoff=None) for m in METRICS}
    n = sum(global_histograms["total_degree"].values())
    out_edges = sum(k * v for k, v in global_histograms["out_degree"].items())
    # Strict reference-count files list only some papers, so their in- and
    # out-edge totals differ; other files must have equal totals.
    in_edges = (
        sum(k * v for k, v in global_histograms["in_degree"].items())
        if "num_in_edges" in global_data
        else out_edges
    )
    validate_degrees(global_data, n, in_edges, out_edges)
    cache = {} if cache is None else cache
    datasets = cached_experiments(
        cache, experiment_root, alphas, global_data["cutoff_date"], parameters
    )
    prefix = f"{label} " if label else ""
    # Skip each K that the global file or a generated distribution cannot support,
    # before any bootstrap work.
    skipped = skipped_cutoffs(global_data, datasets, cutoffs, parameters)
    for cutoff, reason in skipped.items():
        print(f"WARNING: {prefix}K={cutoff} skipped: {reason}", flush=True)
    cutoffs = [cutoff for cutoff in cutoffs if cutoff not in skipped]
    if not cutoffs:
        raise ValueError(
            f"No requested cutoff K can be compared for {global_path.name}; "
            "every K was skipped."
        )
    # Head panels compare full positive-degree distributions of the featured
    # metric (usually total degree) at every K.
    head_global = pooled_counts([global_data], parameters.featured)
    head_generated = {
        alpha: sum(
            (
                pooled_counts(datasets[(t, alpha)]["examples"], parameters.featured)
                for t in EXPERIMENTS
            ),
            Counter(),
        )
        for alpha in alphas
    }
    setup = SearchSetup(
        global_path=global_path,
        global_data=global_data,
        global_provenance=global_provenance,
        datasets_key=experiments_cache_key(
            experiment_root, alphas, global_data["cutoff_date"], parameters
        ),
        output_dir=output_dir,
        alphas=alphas,
        cutoffs=cutoffs,
        skipped=skipped,
        head_global=head_global,
        head_generated=head_generated,
        parameters=parameters,
        repeats=repeats,
        seed=seed,
        epsilon=epsilon,
        unit=unit,
        validation_root=validation_root,
        command=command,
        label=label,
        power_law_fit=power_law_fit,
        power_law_repeats=power_law_repeats,
    )
    return setup, datasets


def search_cutoff_inputs(
    setup: SearchSetup, datasets: dict, cutoff: int
) -> tuple[dict, dict]:
    """Fixed degree supports and pooled distributions that both methods use at K."""
    global_histograms = {
        m: pooled_counts([setup.global_data], m, cutoff=cutoff)
        for m in setup.parameters.metrics
    }
    references = {}
    for metric in setup.parameters.metrics:
        support = set(global_histograms[metric])
        for entry in datasets.values():
            support.update(pooled_counts(entry["examples"], metric, cutoff=cutoff))
        support = np.array(sorted(support))
        counts = align_counts(global_histograms[metric], support)
        mean = float(np.average(support, weights=counts))
        if mean <= 0:
            raise ValueError("Relative EMD requires positive global mean degrees.")
        references[metric] = (support, counts, mean)
    distributions = {
        "global": {
            m: {
                "counts": {str(k): v for k, v in h.items()},
                "probabilities": {
                    str(k): v / sum(h.values()) for k, v in h.items()
                },
            }
            for m, h in global_histograms.items()
        },
        "generated": [],
    }
    for (question_type, alpha), entry in sorted(datasets.items()):
        for metric in setup.parameters.metrics:
            counts = pooled_counts(entry["examples"], metric, cutoff=cutoff)
            distributions["generated"].append(
                {
                    "question_type": question_type,
                    **setup.parameters.columns(alpha),
                    "metric": metric,
                    "counts": {str(k): v for k, v in counts.items()},
                    "probabilities": {
                        str(k): v / sum(counts.values()) for k, v in counts.items()
                    },
                }
            )
    return references, distributions


def search_method(
    setup: SearchSetup,
    datasets: dict,
    cutoff: int,
    method: str,
    references: dict,
    distributions: dict,
    reference: Any | None,
    cache: dict,
) -> dict:
    """Select a candidate for one K and method, and save its K=<K>/ folder.

    ``reference`` is the candidate selected at K=0 (drawn for comparison), or None.
    Returns the row for summary_by_K.csv and the panels for the per-K figure.
    """
    parameters, alphas, epsilon = setup.parameters, setup.alphas, setup.epsilon
    global_data = setup.global_data
    pooled, examples = measure(
        datasets, references, method, epsilon, cutoff, parameters
    )
    distance_units = "nats" if method == "kl" else "degree units"
    score_label = "mean KL (nats)" if method == "kl" else "mean relative EMD"
    for alpha, rows in pooled.groupby(parameters.index, sort=True):
        print(
            f"{method.upper()}: {parameters.label(parameters.key(alpha))}; "
            f"selection_score={rows.score_component.mean():.8g} ({score_label})",
            flush=True,
        )
        for question_type, type_rows in rows.groupby("question_type", sort=True):
            distances = type_rows.set_index("metric")["distance"]
            values = ", ".join(
                f"{metric}={distances[metric]:.8g}" for metric in parameters.metrics
            )
            print(
                f"  {question_type}: {values} ({distance_units})",
                flush=True,
            )
    scores, groups = bootstrap_scores(
        datasets,
        references,
        method,
        epsilon,
        alphas,
        setup.repeats,
        setup.seed,
        setup.unit,
        cutoff,
        parameters,
    )
    ranking = rank_alphas(pooled, scores, alphas, parameters)
    selected = parameters.key(ranking.index[0])
    validation_score, validation_inputs = None, []
    validation_table = None
    if setup.validation_root is not None:
        # Different K values and global files can select the same
        # candidate; load its held-out files once.
        validation_key = (
            "validation",
            str(setup.validation_root.resolve()),
            parameters.names,
            tuple(sorted(parameters.experiments.items())),
            selected,
            global_data["cutoff_date"],
        )
        if validation_key not in cache:
            cache[validation_key] = load_experiments(
                setup.validation_root,
                [selected],
                global_data["cutoff_date"],
                parameters,
            )
        validation = cache[validation_key]
        for key, entry in validation.items():
            if entry["settings"] != datasets[key]["settings"]:
                raise ValueError(f"Validation settings differ: {key}")
        validation_table, _ = measure(
            validation, references, method, epsilon, cutoff, parameters
        )
        validation_score = float(validation_table.score_component.mean())
        validation_inputs = [entry["provenance"] for entry in validation.values()]
    summary = {
        "schema_version": 1,
        "method": method,
        "degree_cutoff": cutoff,
        "degree_filter": f"degree > {cutoff}, separately for each metric "
        "and both distributions",
        "degree_metrics": list(parameters.metrics),
        "normalization": f"conditional on degree > {cutoff}, "
        "before KL smoothing",
        **parameters.columns(selected, "selected_"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        f"candidate_{parameters.plural_key}": alphas,
        "candidate_cutoffs": setup.cutoffs,
        **(
            {"skipped_cutoffs": {str(k): r for k, r in setup.skipped.items()}}
            if setup.skipped
            else {}
        ),
        "selection_status": "independently_evaluated"
        if validation_score is not None
        else "calibration_only",
        "selection_rule": "equal mean across question types and degree metrics of "
        + (
            "KL(generated || global) in nats"
            if method == "kl"
            else "EMD divided by fixed global mean degree"
        ),
        "exact_tie_break": parameters.tie_break,
        "kl_epsilon": epsilon if method == "kl" else None,
        "kl_direction": "generated_to_global" if method == "kl" else None,
        "support": {m: support.tolist() for m, (support, _, _) in references.items()},
        "global_mean_degrees": {m: mean for m, (_, _, mean) in references.items()},
        "global_input": {
            **setup.global_provenance,
            "metadata": {
                k: v for k, v in global_data.items() if not k.endswith("_histogram")
            },
        },
        "inputs": [entry["provenance"] for entry in datasets.values()],
        "bootstrap": {
            "repeats": setup.repeats,
            "seed": setup.seed,
            "unit": setup.unit,
            "groups": groups,
            f"paired_across_{parameters.plural_key}": False,
        },
        "command": setup.command,
        "working_directory": str(Path.cwd()),
        "software_versions": {
            p: version(p) for p in ("numpy", "scipy", "pandas", "matplotlib", "orjson")
        },
        "python_version": sys.version,
        "analysis_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "kl_smoothing_rule": "(1-epsilon)*probability + epsilon/number_of_support_values"
        if method == "kl"
        else None,
        "validation_inputs": validation_inputs,
        "validation_score": validation_score,
        "ranking": ranking.reset_index().to_dict(orient="records"),
    }
    output = setup.output_dir / method / f"K={cutoff}"
    output.mkdir(parents=True, exist_ok=True)
    ranking.reset_index().to_csv(output / "ranking.csv", index=False)
    pooled.to_csv(output / "pooled_distances.csv", index=False)
    examples.to_csv(output / "per_example_distances.csv", index=False)
    examples.groupby(["question_type", *parameters.names, "metric"])[
        ["distance", "score_component"]
    ].agg(["mean", "median", "count"]).to_csv(output / "per_example_summary.csv")
    pd.DataFrame(scores, columns=[parameters.column(a) for a in alphas]).to_csv(
        output / "bootstrap_scores.csv", index=False
    )
    differences = scores - scores[:, [alphas.index(selected)]]
    low, high = np.quantile(differences, [0.025, 0.975], axis=0)
    pd.DataFrame(
        [
            {
                **parameters.columns(a),
                **parameters.columns(selected, "reference_"),
            }
            for a in alphas
        ]
    ).assign(
        point_difference=[
            ranking.loc[a, "score"] - ranking.loc[selected, "score"] for a in alphas
        ],
        difference_ci_low=low,
        difference_ci_high=high,
    ).to_csv(output / "score_differences.csv", index=False)
    if validation_table is not None:
        validation_table.to_csv(output / "validation_distances.csv", index=False)
    if method == "kl":
        sensitivity = []
        for eps in sorted({1e-10, epsilon, 1e-6}):
            for alpha in alphas:
                values = []
                for t in EXPERIMENTS:
                    for m, (support, q, _) in references.items():
                        p = align_counts(
                            pooled_counts(
                                datasets[(t, alpha)]["examples"],
                                m,
                                cutoff=cutoff,
                            ),
                            support,
                        )
                        values.append(probability_distance(p, q, support, "kl", eps))
                sensitivity.append(
                    {
                        "epsilon": eps,
                        **parameters.columns(alpha),
                        "score": float(np.mean(values)),
                    }
                )
        pd.DataFrame(sensitivity).to_csv(
            output / "smoothing_sensitivity.csv", index=False
        )
    for name, payload in (
        ("selection", summary),
        ("distributions", distributions),
    ):
        (output / f"{name}.json").write_bytes(
            orjson.dumps(payload, option=orjson.OPT_INDENT_2) + b"\n"
        )
    head_counts = {
        "global": setup.head_global,
        "selected": setup.head_generated[selected],
    }
    if reference is not None:
        head_counts["reference"] = setup.head_generated[reference]
    save_figures(
        output,
        ranking,
        pooled,
        datasets,
        references,
        method,
        selected,
        cutoff,
        head_counts,
        reference,
        parameters,
        power_law_fit=setup.power_law_fit,
    )
    save_report(output, summary, ranking, parameters)
    if setup.power_law_repeats:
        save_power_law_analysis(
            output,
            datasets,
            global_data,
            selected,
            cutoff,
            parameters,
            setup.unit,
            setup.power_law_repeats,
            setup.seed,
        )
    best = ranking.loc[selected]
    print(
        f"{method.upper()}: selected {parameters.label(selected)}; "
        f"selection_score={best['score']:.8g} ({score_label}); "
        f"artifacts saved to {output}",
        flush=True,
    )
    return {
        "grid_row": {
            "cutoff": cutoff,
            "ranking": ranking,
            "selected": selected,
            "featured_counts": total_degree_counts(
                datasets, references, selected, cutoff, parameters.featured
            ),
            "head_counts": head_counts,
            "reference": reference,
        },
        "result_row": {
            "K": cutoff,
            **parameters.columns(selected, "selected_"),
            "score": float(best["score"]),
            "score_ci_low": float(best["score_ci_low"]),
            "score_ci_high": float(best["score_ci_high"]),
            "bootstrap_selection_frequency": float(
                best["bootstrap_selection_frequency"]
            ),
            "validation_score": validation_score,
        },
    }


def reference_candidate(outcomes: dict[int, dict[str, dict]], method: str) -> Any | None:
    """The candidate a method selected at K=0, if K=0 has run."""
    return outcomes[0][method]["grid_row"]["selected"] if 0 in outcomes else None


def finish_search(
    setup: SearchSetup,
    outcomes: dict[int, dict[str, dict]],
    methods: Sequence[str] = METHODS,
) -> dict[str, list[dict]]:
    """Write summary_by_K.csv, the per-K figure, and skipped_K.csv for each method."""
    parameters, prefix = setup.parameters, f"{setup.label} " if setup.label else ""
    cutoffs = sorted(outcomes)
    results = {
        method: [outcomes[k][method]["result_row"] for k in cutoffs]
        for method in methods
    }
    # One row per K: how the selected candidate moves as low degrees are removed.
    for method, rows in results.items():
        method_dir = setup.output_dir / method
        grid_rows = [outcomes[k][method]["grid_row"] for k in cutoffs]
        pd.DataFrame(rows).to_csv(method_dir / "summary_by_K.csv", index=False)
        save_cutoff_grid(method_dir, method, grid_rows, parameters)
        skipped_path = method_dir / "skipped_K.csv"
        if setup.skipped:
            pd.DataFrame(
                {"K": list(setup.skipped), "reason": list(setup.skipped.values())}
            ).to_csv(skipped_path, index=False)
        else:
            # A rerun with no skipped K must not keep an earlier run's list.
            skipped_path.unlink(missing_ok=True)
        selections = ", ".join(
            f"K={r['cutoff']}: {parameters.text(r['selected'])}" for r in grid_rows
        )
        print(
            f"{prefix}{method.upper()} selected {parameters.title} by K: {selections}",
            flush=True,
        )
        if setup.skipped:
            print(
                f"WARNING: {prefix}{method.upper()} skipped K="
                f"{', '.join(map(str, setup.skipped))}; see {skipped_path}",
                flush=True,
            )
    return results


def run_search(
    global_path: Path,
    experiment_root: Path,
    output_dir: Path,
    alphas: list[Any],
    repeats: int = 500,
    seed: int = 2026,
    epsilon: float = 1e-8,
    unit: str = "context_group",
    validation_root: Path | None = None,
    command: str = "",
    cutoffs: list[int] | None = None,
    cache: dict | None = None,
    label: str = "",
    run_offset: int = 0,
    total_runs: int | None = None,
    parameters: SearchParameters = ALPHA,
    methods: Sequence[str] = METHODS,
    power_law_fit: bool = False,
    power_law_repeats: int = 0,
) -> dict:
    """Select a candidate separately for each cutoff K, comparing only degree > K.

    ``alphas`` holds the candidates: alpha values for ALPHA, or (theta, discount)
    pairs for THETA_DISCOUNT. ``cache`` shares loaded experiment files between
    calls. Its keys include the experiment or validation root, the parameters,
    the candidates, and the cutoff date, so calls with different inputs never
    share an entry. ``label``, ``run_offset``, and ``total_runs`` only change
    the progress lines.
    """
    cache = {} if cache is None else cache
    setup, datasets = prepare_search(
        global_path,
        experiment_root,
        output_dir,
        alphas,
        repeats,
        seed,
        epsilon,
        unit,
        validation_root,
        command,
        cutoffs,
        cache,
        label,
        parameters,
        power_law_fit=power_law_fit,
        power_law_repeats=power_law_repeats,
    )
    prefix = f"{label} " if label else ""
    outcomes: dict[int, dict[str, dict]] = {}
    for run, cutoff in enumerate(setup.cutoffs, start=run_offset + 1):
        print(
            f"[RUN {run}/{total_runs or len(setup.cutoffs)}] {prefix}K={cutoff}: "
            f"compare degree > {cutoff}",
            flush=True,
        )
        references, distributions = search_cutoff_inputs(setup, datasets, cutoff)
        # Cutoffs are sorted, so a K=0 run comes first and gives the reference.
        outcomes[cutoff] = {
            method: search_method(
                setup,
                datasets,
                cutoff,
                method,
                references,
                distributions,
                reference_candidate(outcomes, method),
                cache,
            )
            for method in methods
        }
    return finish_search(setup, outcomes, methods)


@contextlib.contextmanager
def process_pool(
    workers: int, initializer: Callable[..., None] | None = None, initargs: tuple = ()
) -> Iterator[ProcessPoolExecutor]:
    """Spawned worker processes that use one math-library thread each.

    Thread variables the caller has not set are 1 while the pool runs. On an
    error, queued tasks are cancelled.
    """
    added = [name for name in SINGLE_THREAD_VARIABLES if name not in os.environ]
    os.environ.update({name: "1" for name in added})
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initializer,
        initargs=initargs,
    )
    try:
        yield pool
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    finally:
        for name in added:
            os.environ.pop(name, None)


def map_in_processes(
    function: Callable[..., Any], arguments: Sequence[tuple], workers: int = 1
) -> Iterator[Any]:
    """Yield function(*args) per tuple: in order for one worker, else as each finishes."""
    if workers <= 1:
        for args in arguments:
            yield function(*args)
        return
    if not arguments:
        return
    with process_pool(min(workers, len(arguments))) as pool:
        futures = [pool.submit(function, *args) for args in arguments]
        for future in as_completed(futures):
            yield future.result()


def initialize_search_worker(datasets_by_key: dict) -> None:
    """Keep the experiment data in this worker process for all of its tasks."""
    SEARCH_WORKER_STATE.update(datasets=datasets_by_key, cache={})


def run_search_task(
    setup: SearchSetup, cutoff: int, method: str, reference: Any | None
) -> tuple[dict, str, float]:
    """Run one (global file, K, method) search in a worker.

    Returns the outcome, the printed log (shown as one block), and the seconds taken.
    """
    start = time.perf_counter()
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        datasets = SEARCH_WORKER_STATE["datasets"][setup.datasets_key]
        references, distributions = search_cutoff_inputs(setup, datasets, cutoff)
        outcome = search_method(
            setup,
            datasets,
            cutoff,
            method,
            references,
            distributions,
            reference,
            SEARCH_WORKER_STATE["cache"],
        )
    return outcome, log.getvalue(), time.perf_counter() - start


def run_searches(
    global_paths: list[Path],
    output_dir: Path,
    cutoffs: list[int] | None = None,
    workers: int = 1,
    methods: Sequence[str] = METHODS,
    **options: Any,
) -> dict[str, dict]:
    """Run run_search for each global file into <output_dir>/<file name without .json>/.

    Experiment and validation files are loaded once and shared across files.
    With ``workers`` > 1, each (global file, K, method) search runs in its own
    worker process: K=0 first (its selection is drawn in the other K figures),
    then everything else. The saved results are the same as with one worker.
    """
    names = [path.stem for path in global_paths]
    if not names or len(set(names)) != len(names):
        raise ValueError("Global distribution files need distinct file names.")
    # Check every global file first: each K it cannot support is reported before
    # any bootstrap work, and progress lines count only the K values that run.
    # The experiment files loaded here are cached for the searches.
    parameters = options.get("parameters", ALPHA)
    alphas = sorted(parameters.key(alpha) for alpha in options["alphas"])
    requested = [0] if cutoffs is None else list(cutoffs)
    cache: dict = {}
    runs = []
    for path, name in zip(global_paths, names):
        global_data = read_json(path)[0]
        datasets = cached_experiments(
            cache,
            options["experiment_root"],
            alphas,
            global_data["cutoff_date"],
            parameters,
        )
        skipped = skipped_cutoffs(global_data, datasets, requested, parameters)
        for cutoff, reason in skipped.items():
            print(f"WARNING: {name} K={cutoff} will be skipped: {reason}", flush=True)
        runs.append(len(set(requested)) - len(skipped))
        if requested and not runs[-1]:
            raise ValueError(
                f"No requested cutoff K can be compared for {path.name}; "
                "every K was skipped."
            )
    if workers <= 1:
        return {
            name: run_search(
                global_path=path,
                output_dir=output_dir / name,
                cutoffs=cutoffs,
                cache=cache,
                label=name,
                run_offset=sum(runs[:index]),
                total_runs=sum(runs),
                methods=methods,
                **options,
            )
            for index, (path, name) in enumerate(zip(global_paths, names))
        }
    setups, datasets_by_key = [], {}
    for path, name in zip(global_paths, names):
        setup, datasets = prepare_search(
            path,
            output_dir=output_dir / name,
            cutoffs=cutoffs,
            cache=cache,
            label=name,
            **options,
        )
        setups.append(setup)
        datasets_by_key[setup.datasets_key] = datasets
    total = sum(len(setup.cutoffs) for setup in setups) * len(methods)
    print(
        f"Running {total} (global file, K, method) searches with "
        f"{min(workers, total)} worker processes.",
        flush=True,
    )
    outcomes: list[dict[int, dict[str, dict]]] = [{} for _ in setups]
    finished = 0
    with process_pool(
        min(workers, total), initialize_search_worker, (datasets_by_key,)
    ) as pool:
        for first_phase in (True, False):
            futures = {
                pool.submit(
                    run_search_task,
                    setup,
                    cutoff,
                    method,
                    reference_candidate(outcomes[index], method),
                ): (index, cutoff, method)
                for index, setup in enumerate(setups)
                for cutoff in setup.cutoffs
                if (cutoff == 0) == first_phase
                for method in methods
            }
            for future in as_completed(futures):
                index, cutoff, method = futures[future]
                outcome, log, seconds = future.result()
                outcomes[index].setdefault(cutoff, {})[method] = outcome
                finished += 1
                print(
                    f"[RUN {finished}/{total}] {setups[index].label} K={cutoff} "
                    f"{method.upper()}: done in {seconds:.0f}s",
                    flush=True,
                )
                print(log, end="", flush=True)
    return {
        setup.label: finish_search(setup, outcomes[index], methods)
        for index, setup in enumerate(setups)
    }
