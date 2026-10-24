"""Save the fields of study of the papers that the v1 benchmark uses.

Two category sets, one JSON file each in --output-dir:

    sampled_paths.<cutoff>.json
        Papers on the sampled paths (--paths), split by path length (hops).
    generated_examples.<cutoff>.json
        Papers in the growth search's examples: every alpha=<alpha> folder
        under --search-root, combined so that every alpha is compared with the
        same set. Split by alpha and by question format.

The papers of an example are the papers whose degrees
calculate_sample_paper_degree_distribution_v2.py measures: the expanded papers
and citation endpoints of the example's own components.

Counting: each distinct paper counts once in each of its Semantic Scholar
fieldsOfStudy (read from the bulk index), so the shares can add up to more
than 1. A paper without a field counts as "none". A paper that the bulk index
cannot resolve is counted in num_unresolved and left out of the shares.

Lookup: path papers by their saved corpus_ids; example papers through their
semantic_scholar_id in --graph-db and the index's paper_ids table. Spark reads
two columns of the papers and paper_ids tables and never the citation table.

Run from the repository root after growth_search.1_create_data.sh:
    bash scripts/experiments/v1/paper_categories.sh
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/graph_statistics/paper_category_sets"

from endpoints.calculate_sample_paper_degree_distribution_v2 import (
    candidate_folders,
    load_records,
)
from src.clients.semantic_scholar_bulk_client import (
    BULK_INDEX_VERSION,
    DEFAULT_INDEX_ROOT,
    HEX_SHA_RE,
    INDEX_METADATA_FILE_NAME,
    import_pyspark_sql,
)
from src.graph.paper_categories import (
    category_counts,
    lookup_corpus_ids,
    read_fields_of_study,
)
from src.graph.readers import (
    GRAPH_DB_IMMUTABLE_ENV,
    connect_immutable_database,
    env_flag_enabled,
)
from src.graph.sample_degree_distribution import calculate_graph_degrees
from src.setup.constants import nid_db_id

COUNTING = {
    "unit": "distinct papers",
    "fields": "Semantic Scholar fieldsOfStudy in the bulk index papers table "
    "(union over a paper's rows)",
    "multiple_fields": "a paper counts once in each of its fields, so shares "
    "can add up to more than 1",
    "no_field": "papers without a field count as 'none'",
    "share_denominator": "num_resolved: papers found in the bulk index",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--paths",
        type=Path,
        nargs="+",
        required=True,
        help="sampled_paths.json files from construct_paths_from_bulk_graph.py.",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        required=True,
        help="Growth search folder that holds the alpha=<alpha> folders.",
    )
    parser.add_argument(
        "--alpha", nargs="+", type=float, help="Default: every alpha folder found."
    )
    parser.add_argument("--vary", default="question-format")
    parser.add_argument("--graph-db", type=Path, required=True)
    parser.add_argument(
        "--cutoff-date",
        required=True,
        help="Must equal the paths' start_date and the examples' cutoff_date.",
    )
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--spark-master", default="local[*]")
    parser.add_argument("--driver-memory", default="16g")
    parser.add_argument("--local-dir", type=Path, help="Spark spill directory.")
    return parser


def read_index_metadata(index_root: Path) -> dict[str, Any]:
    metadata = orjson.loads((index_root / INDEX_METADATA_FILE_NAME).read_bytes())
    if metadata.get("indexVersion") != BULK_INDEX_VERSION:
        raise ValueError(
            f"Expected Semantic Scholar bulk index version {BULK_INDEX_VERSION}"
        )
    for table in ("papers", "paper_ids"):
        if not (index_root / table / "_SUCCESS").is_file():
            raise FileNotFoundError(
                f"Missing or incomplete bulk index table: {index_root / table}"
            )
    return metadata


def path_papers(
    paths_files: Sequence[Path], cutoff: str
) -> tuple[dict[str, set[int]], list[dict[str, Any]]]:
    """Corpus ids on the paths of each file, keyed by path length."""
    by_hops: dict[str, set[int]] = defaultdict(set)
    sources = []
    for path in paths_files:
        payload = orjson.loads(path.read_bytes())
        metadata = payload.get("metadata") or {}
        if metadata.get("start_date") != cutoff:
            raise ValueError(
                f"{path}: start_date {metadata.get('start_date')!r} is not the "
                f"cutoff date {cutoff!r}."
            )
        hops = str(metadata["path_length"])
        entries = payload.get("paths") or []
        for index, entry in enumerate(entries):
            corpus_ids = entry.get("corpus_ids")
            if not isinstance(corpus_ids, list):
                raise ValueError(
                    f"{path}: paths[{index}] has no corpus_ids; sample the paths "
                    "with construct_paths_from_bulk_graph.py."
                )
            by_hops[hops].update(int(corpus_id) for corpus_id in corpus_ids)
        sources.append({"path": str(path.resolve()), "hops": hops, "num_paths": len(entries)})
    return dict(by_hops), sources


def example_papers(
    search_root: Path,
    alphas: Sequence[float] | None,
    vary: str,
    cutoff: str,
) -> tuple[list[str], dict[str, dict[str, set[int]]], dict[str, int]]:
    """Paper node ids of every example, by alpha and by question format.

    Returns the distinct node ids, the groups as positions in that list (so a
    paper's id is stored once however many groups hold it), and the number of
    examples per alpha.
    """
    positions: dict[str, int] = {}
    groups: dict[str, dict[str, set[int]]] = {
        "alpha": defaultdict(set),
        "question_format": defaultdict(set),
    }
    num_examples: dict[str, int] = {}
    for (alpha,), folder in candidate_folders(search_root, ("alpha",), [alphas]):
        records = load_records(folder, vary)
        for record in records:
            if record["config"]["cutoff_date"] != cutoff:
                raise ValueError(
                    f"{folder}: example {record['qid']} has cutoff_date "
                    f"{record['config']['cutoff_date']!r}, not {cutoff!r}."
                )
            papers = {
                positions.setdefault(paper["paper_id"], len(positions))
                for paper in calculate_graph_degrees(record["components"])["paper_degrees"]
            }
            groups["alpha"][f"{alpha:g}"] |= papers
            groups["question_format"][record["question_type"]] |= papers
        num_examples[f"{alpha:g}"] = len(records)
        print(f"alpha={alpha:g}: {len(records):,} examples from {folder}", flush=True)
    return list(positions), {key: dict(value) for key, value in groups.items()}, num_examples


def read_semantic_scholar_ids(graph_db: Path, db_ids: Sequence[str]) -> dict[str, str]:
    """Map graph DB paper ids to lowercase paperIds; other ids are left out."""
    if env_flag_enabled(GRAPH_DB_IMMUTABLE_ENV):
        connection = connect_immutable_database(graph_db)
    else:
        connection = sqlite3.connect(f"{graph_db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        result = {}
        for start in range(0, len(db_ids), 500):
            chunk = db_ids[start : start + 500]
            rows = connection.execute(
                "SELECT id, semantic_scholar_id FROM papers WHERE id IN "
                f"({','.join('?' for _ in chunk)})",
                chunk,
            ).fetchall()
            for db_id, paper_id in rows:
                paper_id = str(paper_id or "").strip().lower()
                if HEX_SHA_RE.fullmatch(paper_id):
                    result[str(db_id)] = paper_id
        return result
    finally:
        connection.close()


def category_set_payload(
    name: str,
    groups: Mapping[str, Mapping[str, Iterable[Any]]],
    fields: Mapping[Any, Sequence[str]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """The report of one category set: overall counts and every breakdown."""
    breakdowns = {
        key: {label: category_counts(papers, fields) for label, papers in by_label.items()}
        for key, by_label in groups.items()
    }
    # Every breakdown covers the same papers, so any one of them gives the union.
    overall = category_counts(
        set().union(*next(iter(groups.values())).values()), fields
    )
    return {
        "name": name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **settings,
        "counting": COUNTING,
        "categories": list(overall["categories"]),
        "overall": overall,
        "breakdowns": breakdowns,
    }


def print_report(payload: dict[str, Any]) -> None:
    """Print the share of each category (in %) overall and per breakdown group."""
    overall = payload["overall"]
    print(
        f"\n{payload['name']}: {overall['num_papers']:,} papers, "
        f"{overall['num_unresolved']:,} unresolved, "
        f"{len(payload['categories'])} categories"
    )
    for key, by_label in payload["breakdowns"].items():
        columns = {"overall": overall, **by_label}
        table = pd.DataFrame(
            {
                column: {
                    name: 100 * value["share"]
                    for name, value in counts["categories"].items()
                }
                for column, counts in columns.items()
            },
            index=payload["categories"],
        ).fillna(0)
        print(f"\nShare of resolved papers (%), by {key}:")
        print(table.to_string(float_format=lambda value: f"{value:.2f}"))


def save_category_sets(spark: Any, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    cutoff = args.cutoff_date
    metadata = read_index_metadata(args.index_root)
    paths_by_hops, path_sources = path_papers(args.paths, cutoff)
    node_ids, example_groups, num_examples = example_papers(
        args.search_root, args.alpha, args.vary, cutoff
    )

    print(f"Resolving {len(node_ids):,} example papers through {args.graph_db}", flush=True)
    db_ids = [nid_db_id(node_id) for node_id in node_ids]
    paper_ids = read_semantic_scholar_ids(args.graph_db, db_ids)
    corpus_by_paper_id = lookup_corpus_ids(spark, args.index_root, paper_ids.values())
    example_corpus_ids = {
        position: corpus_by_paper_id[paper_ids[db_id]]
        for position, db_id in enumerate(db_ids)
        if paper_ids.get(db_id) in corpus_by_paper_id
    }
    path_corpus_ids = set().union(*paths_by_hops.values())
    print(
        f"Reading fields of study for {len(path_corpus_ids):,} path papers and "
        f"{len(example_corpus_ids):,} resolved example papers",
        flush=True,
    )
    fields = read_fields_of_study(
        spark, args.index_root, path_corpus_ids | set(example_corpus_ids.values())
    )
    example_fields = {
        position: fields[corpus_id]
        for position, corpus_id in example_corpus_ids.items()
        if corpus_id in fields
    }

    index_settings = {
        "cutoff_date": cutoff,
        "index_root": str(args.index_root.resolve()),
        "index_created_at": metadata.get("createdAt"),
    }
    payloads = {
        "sampled_paths": category_set_payload(
            "sampled_paths",
            {"hops": paths_by_hops},
            fields,
            {**index_settings, "sources": path_sources},
        ),
        "generated_examples": category_set_payload(
            "generated_examples",
            example_groups,
            example_fields,
            {
                **index_settings,
                "graph_db": str(args.graph_db.resolve()),
                "search_root": str(args.search_root.resolve()),
                "vary": args.vary,
                "num_examples_by_alpha": num_examples,
            },
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        print_report(payload)
        output = args.output_dir / f"{name}.{cutoff}.json"
        output.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2) + b"\n")
        print(f"Saved {output}", flush=True)
    return payloads


def main() -> None:
    args = build_parser().parse_args()
    builder = (
        import_pyspark_sql()[0]
        .builder.appName("paper_category_sets")
        .master(args.spark_master)
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.driver.maxResultSize", "8g")
    )
    if args.local_dir:
        args.local_dir.mkdir(parents=True, exist_ok=True)
        builder = builder.config("spark.local.dir", str(args.local_dir))
    spark = builder.getOrCreate()
    try:
        save_category_sets(spark, args)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
