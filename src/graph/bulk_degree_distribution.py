"""Exact degree histograms for a date-filtered Semantic Scholar bulk graph."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

BROADCAST_MAX_ELIGIBLE_IDS = 5_000_000
BROADCAST_MAX_MIB = 256
# Allow room for the long-key hash table, collected rows, and broadcast buffers.
# This is a conservative estimate, not a measurement of Spark's peak allocation.
BROADCAST_ESTIMATED_BYTES_PER_ID = 128
FIGURE_METRICS = (
    ("total_degree", "Total degree"),
    ("in_degree", "In-degree"),
    ("out_degree", "Out-degree"),
)
# The linear panels stop at the degree covering this share of papers.
LINEAR_PANEL_COVERAGE = 0.999


def spark_memory_bytes(value: str) -> int:
    """Parse Spark's integer byte-size settings, such as 512g or 8589934592b."""
    match = re.fullmatch(r"(\d+)([kmgtp]?)(?:b)?", value.strip().lower())
    if match is None:
        raise ValueError(f"Invalid Spark memory size: {value!r}")
    return int(match[1]) * 1024 ** (" kmgtp".index(match[2]) if match[2] else 0)


def broadcast_memory_budget(spark: Any, max_mib: int) -> int:
    """Keep broadcasts below the user budget, heap, and Spark size limits."""
    conf = spark.sparkContext.getConf()
    heap = spark_memory_bytes(conf.get("spark.driver.memory", "1g"))
    result_limit = spark_memory_bytes(conf.get("spark.driver.maxResultSize", "1g"))
    # Before Spark 4.1 this limit is fixed at 8 GiB; newer versions expose it.
    table_limit = spark_memory_bytes(
        spark.conf.get("spark.sql.maxBroadcastTableSize", "8g")
    )
    limits = [max_mib * 1024**2, heap // 8, table_limit // 2]
    if result_limit:
        limits.append(result_limit // 2)
    return min(limits)


def calculate_degree_histograms(
    papers: Any,
    citations: Any,
    cutoff_date: date,
    broadcast_max_eligible_ids: int = BROADCAST_MAX_ELIGIBLE_IDS,
    *,
    category: str = "cs",
    broadcast_max_mib: int = BROADCAST_MAX_MIB,
    strict_reference_counts: Sequence[int] = (0,),
) -> tuple[dict[int, dict[str, dict[int, int]]], int]:
    """Count distinct directed edges whose two papers pass an inclusive cutoff.

    Dates must be complete YYYY-MM-DD values; missing/invalid dates are excluded.
    Eligible isolated papers count as zero. A self-edge contributes one to each
    directional degree and two to total degree. Only histograms reach Python.

    ``strict_reference_counts`` selects the modes to compute from one edge scan.
    Mode 0 lists every eligible paper. Mode 1 lists only eligible papers whose
    ``referenceCount`` is positive and equals their number of citation rows.
    That count uses every row, including rows with a null ``citedCorpusId``, and
    applies no filter to the cited papers. In both modes, neighbors need only be
    eligible.

    The eligible ids are materialised before the edge scan so they are read once
    and so a small enough set can be broadcast, which filters the citation table
    map-side instead of shuffling all of it.

    Returns the histograms for each requested mode and the number of eligible
    papers.
    """
    from pyspark import StorageLevel
    from pyspark.sql import functions as F

    modes = sorted(set(strict_reference_counts))
    if not modes or not set(modes) <= {0, 1}:
        raise ValueError("strict_reference_counts must contain only 0 and 1")
    strict = 1 in modes
    if category not in {"cs", "all"}:
        raise ValueError("category must be 'cs' or 'all'")
    if broadcast_max_eligible_ids < 0 or broadcast_max_mib < 0:
        raise ValueError("broadcast limits must be non-negative")
    if strict and "referenceCount" not in papers.columns:
        raise ValueError(
            "The papers index needs referenceCount for the strict reference count."
        )
    if category == "cs":
        if "fieldsOfStudy" not in papers.columns:
            raise ValueError("The papers index needs fieldsOfStudy for category 'cs'.")
        papers = papers.filter(F.array_contains("fieldsOfStudy", "Computer Science"))

    metrics = ("total_degree", "in_degree", "out_degree")
    dated = (
        papers.filter(F.col("corpusId").isNotNull())
        .filter(F.trim("publicationDate").rlike(r"^\d{4}-\d{2}-\d{2}$"))
        .filter(F.expr("try_cast(trim(publicationDate) AS DATE)") >= F.lit(cutoff_date))
    )
    if strict:
        # Duplicate metadata rows with different counts can never match below.
        eligible = dated.groupBy("corpusId").agg(
            F.min("referenceCount").alias("minReferenceCount"),
            F.max("referenceCount").alias("maxReferenceCount"),
        )
    else:
        eligible = dated.select("corpusId").distinct()
    eligible = eligible.persist(StorageLevel.MEMORY_AND_DISK)
    cached = [eligible]
    try:
        num_eligible = eligible.count()
        histograms: dict[int, dict[str, dict[int, int]]] = {
            mode: {metric: {} for metric in metrics} for mode in modes
        }
        if num_eligible == 0:
            return histograms, 0
        estimated_bytes = num_eligible * BROADCAST_ESTIMATED_BYTES_PER_ID
        budget = broadcast_memory_budget(papers.sparkSession, broadcast_max_mib)
        broadcastable = (
            num_eligible <= broadcast_max_eligible_ids and estimated_bytes <= budget
        )
        strategy = "broadcast" if broadcastable else "shuffle"
        print(
            f"Eligible papers: {num_eligible:,}; estimated broadcast memory: "
            f"{estimated_bytes / 1024**2:.1f} MiB; budget: {budget / 1024**2:.1f} MiB; "
            f"filtering citations by {strategy} join.",
            flush=True,
        )

        def join_hint(frame: Any) -> Any:
            return F.broadcast(frame) if broadcastable else frame.hint("merge")

        def id_filter(ids: Any, column_name: str) -> Any:
            return join_hint(ids.select(F.col("corpusId").alias(column_name)))

        citing_rows = citations.select("citingCorpusId", "citedCorpusId").join(
            id_filter(eligible, "citingCorpusId"), "citingCorpusId", "left_semi"
        )
        num_listed = {0: num_eligible}
        if strict:
            # The reference check and the edge scan share one citation-table read.
            citing_rows = citing_rows.persist(StorageLevel.MEMORY_AND_DISK)
            cached.append(citing_rows)
            reference_rows = (
                citing_rows.groupBy("citingCorpusId")
                .count()
                .withColumnRenamed("citingCorpusId", "corpusId")
            )
            listed = (
                eligible.filter(
                    (F.col("minReferenceCount") == F.col("maxReferenceCount"))
                    & (F.col("minReferenceCount") > 0)
                )
                .join(reference_rows, "corpusId")
                .filter(F.col("count") == F.col("minReferenceCount"))
                .select("corpusId")
                .persist(StorageLevel.MEMORY_AND_DISK)
            )
            cached.append(listed)
            num_listed[1] = listed.count()
            print(
                f"Papers passing the reference-count check: {num_listed[1]:,} of "
                f"{num_eligible:,}.",
                flush=True,
            )

        edges = citing_rows.join(
            id_filter(eligible, "citedCorpusId"), "citedCorpusId", "left_semi"
        ).distinct()
        # Aggregate both directions from the same filtered edge scan.
        endpoints = edges.select(
            F.explode(
                F.array(
                    F.struct(
                        F.col("citingCorpusId").alias("corpusId"),
                        F.lit(0).alias("in_degree"),
                        F.lit(1).alias("out_degree"),
                    ),
                    F.struct(
                        F.col("citedCorpusId").alias("corpusId"),
                        F.lit(1).alias("in_degree"),
                        F.lit(0).alias("out_degree"),
                    ),
                )
            ).alias("endpoint")
        ).select("endpoint.*")
        if modes == [1]:
            # Neighbors stay in the edges; unlisted papers' own rows are not needed.
            endpoints = endpoints.join(
                id_filter(listed, "corpusId"), "corpusId", "left_semi"
            )
        degrees = endpoints.groupBy("corpusId").agg(
            F.sum("in_degree").alias("in_degree"),
            F.sum("out_degree").alias("out_degree"),
        )
        degrees = degrees.withColumn(
            "total_degree", F.col("in_degree") + F.col("out_degree")
        )
        if modes == [0, 1]:
            # Mark listed papers so one aggregation serves both modes.
            flags = join_hint(listed.withColumn("listed", F.lit(True)))
            degrees = degrees.join(flags, "corpusId", "left")
        else:
            degrees = degrees.withColumn("listed", F.lit(modes == [1]))
        counts = (
            degrees.select(
                "listed",
                F.explode(
                    F.array(
                        *[
                            F.struct(
                                F.lit(metric).alias("metric"),
                                F.col(metric).alias("degree"),
                            )
                            for metric in metrics
                        ]
                    )
                ).alias("value"),
            )
            .select("listed", "value.*")
            .groupBy("listed", "metric", "degree")
            .count()
        )
        for row in counts.collect():
            for mode in modes:
                if mode == 0 or row.listed:
                    histogram = histograms[mode][row.metric]
                    degree = int(row.degree)
                    histogram[degree] = histogram.get(degree, 0) + int(row["count"])
        # Every paper here touches a retained edge, so total degree is positive.
        # Add isolated papers without joining all eligible IDs back to degrees.
        for mode, mode_histograms in histograms.items():
            isolated = num_listed[mode] - sum(mode_histograms["total_degree"].values())
            if isolated:
                for histogram in mode_histograms.values():
                    histogram[0] = histogram.get(0, 0) + isolated
        return {
            mode: {
                metric: dict(sorted(histogram.items()))
                for metric, histogram in mode_histograms.items()
            }
            for mode, mode_histograms in histograms.items()
        }, num_eligible
    finally:
        for frame in cached:
            frame.unpersist()



def save_degree_figure(payload: dict[str, Any], path: Path) -> None:
    """Save one row per degree metric: linear histogram left, log-log right.

    Both columns show the fraction of listed papers. The linear panel stops at
    the degree covering 99.9% of papers; the log-log panel keeps the whole tail
    but cannot show degree 0, so its share is stated in the panel title.
    """
    import numpy as np
    from matplotlib.figure import Figure

    num_papers = payload["num_papers"]
    if payload["strict_reference_count"]:
        summary = (
            f"{num_papers:,} of {payload['num_papers_after_cutoff']:,} papers pass the "
            f"reference check; {payload['num_in_edges']:,} in-edges, "
            f"{payload['num_out_edges']:,} out-edges"
        )
    else:
        summary = (
            f"{num_papers:,} papers; {payload['num_citation_edges']:,} citation edges"
        )
    figure = Figure(figsize=(12, 11), layout="constrained")
    figure.suptitle(f"{path.stem}\n{summary}")
    axes = figure.subplots(len(FIGURE_METRICS), 2)
    for (metric, label), (linear, log_log) in zip(FIGURE_METRICS, axes):
        counts = {int(k): v for k, v in payload[f"{metric}_histogram"].items()}
        degrees = np.array(sorted(counts))
        fractions = np.array([counts[d] for d in degrees]) / num_papers
        limit = degrees[np.searchsorted(np.cumsum(fractions), LINEAR_PANEL_COVERAGE)]
        shown = degrees <= limit
        linear.bar(degrees[shown], fractions[shown], width=0.8, color="C0")
        linear.set(
            title=f"{label}: degree <= {limit:,} (99.9% of papers)",
            xlabel=label,
            ylabel="Fraction of papers",
        )
        positive = degrees > 0
        zero_share = fractions[~positive].sum()
        log_log.set(
            title=f"{label}, log-log: degree 0 ({zero_share:.1%} of papers) not shown",
            xlabel=label,
            ylabel="Fraction of papers",
        )
        if positive.any():
            log_log.loglog(
                degrees[positive],
                fractions[positive],
                marker="o",
                markersize=3,
                linestyle="none",
                color="C0",
            )
        else:
            log_log.text(
                0.5,
                0.5,
                "No paper has degree > 0",
                ha="center",
                transform=log_log.transAxes,
            )
        for ax in (linear, log_log):
            ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
