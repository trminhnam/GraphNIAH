"""Save exact degree histograms from the local Semantic Scholar bulk index.

Each JSON file gets a figure with the same name and a .png extension: one row
per degree metric (total, in, out), linear histogram left, log-log right.

Example:
    python endpoints/calculate_bulk_paper_degree_distribution.py \
        --cutoff-date 2025-09-01 --strict-reference-count 0 1
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import orjson

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "output/graph_statistics/bulk_paper_degree_distribution"
)

from src.clients.semantic_scholar_bulk_client import (
    BULK_INDEX_VERSION,
    DEFAULT_INDEX_ROOT,
    INDEX_METADATA_FILE_NAME,
    import_pyspark_sql,
)
from src.graph.bulk_degree_distribution import (
    BROADCAST_MAX_ELIGIBLE_IDS,
    BROADCAST_MAX_MIB,
    calculate_degree_histograms,
    save_degree_figure,
)


def parse_cutoff_date(value: str) -> date:
    """Require a complete ISO calendar date for the inclusive cutoff."""
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() == value:
            return parsed
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("cutoff date must be a valid YYYY-MM-DD date")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--cutoff-date", type=parse_cutoff_date, default="2025-09-01")
    parser.add_argument(
        "--category",
        choices=("cs", "all"),
        default="cs",
        help="Filter both endpoints: cs = Computer Science; all = any category.",
    )
    parser.add_argument(
        "--strict-reference-count",
        type=int,
        nargs="+",
        choices=(0, 1),
        default=[0],
        help="Modes to save from one Spark run. 1 = list only papers whose "
        "referenceCount is positive and equals their citation rows in the index; "
        "neighbors are not checked.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder for <category>.<cutoff>.strict-reference-count-<0|1>.json files "
        "and a .png degree figure next to each.",
    )
    parser.add_argument("--spark-master", default="local[*]")
    parser.add_argument("--driver-memory", default="4g")
    parser.add_argument(
        "--driver-max-result-size",
        default="8g",
        help="Spark action result limit; separate from the broadcast table limit.",
    )
    parser.add_argument(
        "--broadcast-max-eligible-ids",
        type=int,
        default=BROADCAST_MAX_ELIGIBLE_IDS,
        help="Maximum eligible IDs for broadcasting; 0 forces shuffle joins.",
    )
    parser.add_argument(
        "--broadcast-max-mib",
        type=int,
        default=BROADCAST_MAX_MIB,
        help="Estimated broadcast memory budget in MiB; 0 forces shuffle joins.",
    )
    parser.add_argument("--shuffle-partitions", type=int, default=200)
    parser.add_argument("--input-max-partition-bytes", default="256m")
    parser.add_argument("--local-dir", type=Path, help="Spark spill directory.")
    return parser


def output_file_name(
    category: str, cutoff_date: date, strict_reference_count: int
) -> str:
    return (
        f"{category}.{cutoff_date.isoformat()}"
        f".strict-reference-count-{strict_reference_count}.json"
    )


def save_degree_distribution(
    spark: Any,
    index_root: Path,
    cutoff_date: date,
    output_dir: Path,
    *,
    category: str = "cs",
    broadcast_max_eligible_ids: int = BROADCAST_MAX_ELIGIBLE_IDS,
    broadcast_max_mib: int = BROADCAST_MAX_MIB,
    strict_reference_counts: Sequence[int] = (0,),
) -> dict[int, dict[str, Any]]:
    """Read completed index tables and save one histogram file per strict mode."""
    metadata = orjson.loads((index_root / INDEX_METADATA_FILE_NAME).read_bytes())
    if metadata.get("indexVersion") != BULK_INDEX_VERSION:
        raise ValueError(
            f"Expected Semantic Scholar bulk index version {BULK_INDEX_VERSION}"
        )
    for table in ("papers", "citations_by_citing"):
        if not (index_root / table / "_SUCCESS").is_file():
            raise FileNotFoundError(
                f"Missing or incomplete bulk index table: {index_root / table}"
            )

    modes = sorted(set(strict_reference_counts))
    print(
        f"Calculating degrees for category={category}, papers and neighbors "
        f"published >= {cutoff_date}, strict-reference-count="
        f"{','.join(map(str, modes))}.",
        flush=True,
    )
    histograms_by_mode, num_papers_after_cutoff = calculate_degree_histograms(
        spark.read.parquet(str(index_root / "papers")),
        spark.read.parquet(str(index_root / "citations_by_citing")),
        cutoff_date,
        category=category,
        broadcast_max_eligible_ids=broadcast_max_eligible_ids,
        broadcast_max_mib=broadcast_max_mib,
        strict_reference_counts=modes,
    )
    payloads = {}
    for mode, histograms in histograms_by_mode.items():
        num_out_edges = sum(
            degree * count for degree, count in histograms["out_degree"].items()
        )
        if mode:
            # Neighbors need not pass the check, so the two directions differ.
            edge_totals = {
                "num_out_edges": num_out_edges,
                "num_in_edges": sum(
                    degree * count for degree, count in histograms["in_degree"].items()
                ),
            }
        else:
            edge_totals = {"num_citation_edges": num_out_edges}
        payload = {
            "index_root": str(index_root.resolve()),
            "index_metadata": metadata,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cutoff_date": cutoff_date.isoformat(),
            "cutoff_inclusive": True,
            "cutoff_applies_to": "both endpoints",
            "category": category,
            "category_applies_to": "both endpoints",
            "category_filter": (
                "fieldsOfStudy contains Computer Science"
                if category == "cs"
                else "none"
            ),
            "strict_reference_count": mode,
            "reference_count_filter": (
                "listed papers only: referenceCount > 0 and equal to all their "
                "citations_by_citing rows, including null citedCorpusId, with no "
                "filter on cited papers; neighbors are not checked"
                if mode
                else "none"
            ),
            "broadcast_max_eligible_ids": broadcast_max_eligible_ids,
            "broadcast_max_mib": broadcast_max_mib,
            "missing_or_invalid_publication_dates": "excluded; no year fallback",
            "edge_counting": "distinct directed citingCorpusId/citedCorpusId pairs",
            "self_edges": "one in-degree, one out-degree, two total degree",
            "bin_width": 1,
            "num_papers_after_cutoff": num_papers_after_cutoff,
            "num_papers": sum(histograms["total_degree"].values()),
            **edge_totals,
            **{
                f"{metric}_histogram": {
                    str(degree): count for degree, count in histogram.items()
                }
                for metric, histogram in histograms.items()
            },
        }
        output_path = output_dir / output_file_name(category, cutoff_date, mode)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(
            orjson.dumps(payload, option=orjson.OPT_INDENT_2) + b"\n"
        )
        edge_summary = "; ".join(
            f"{key}: {value:,}" for key, value in edge_totals.items()
        )
        print(
            f"strict-reference-count={mode}: listed papers: {payload['num_papers']:,} "
            f"of {num_papers_after_cutoff:,} after the cutoff; {edge_summary}"
        )
        print(f"Saved exact histograms to {output_path}")
        if payload["num_papers"]:
            figure_path = output_path.with_suffix(".png")
            save_degree_figure(payload, figure_path)
            print(f"Saved degree figure to {figure_path}")
        payloads[mode] = payload
    return payloads


def main() -> None:
    args = build_parser().parse_args()
    builder = (
        import_pyspark_sql()[0]
        .builder.appName("bulk_paper_degree_distribution")
        .master(args.spark_master)
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.driver.maxResultSize", args.driver_max_result_size)
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .config("spark.sql.files.maxPartitionBytes", args.input_max_partition_bytes)
    )
    if args.local_dir:
        args.local_dir.mkdir(parents=True, exist_ok=True)
        builder = builder.config("spark.local.dir", str(args.local_dir))
    spark = builder.getOrCreate()
    try:
        save_degree_distribution(
            spark,
            args.index_root,
            args.cutoff_date,
            args.output_dir,
            category=args.category,
            broadcast_max_eligible_ids=args.broadcast_max_eligible_ids,
            broadcast_max_mib=args.broadcast_max_mib,
            strict_reference_counts=args.strict_reference_count,
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
