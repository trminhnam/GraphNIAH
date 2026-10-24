"""Read and count the Semantic Scholar fields of study of a set of papers.

Used by endpoints/calculate_paper_category_sets.py. A paper's fields are the
union of ``fieldsOfStudy`` over its rows in the bulk index ``papers`` table,
the same column that the category filters of src/graph/bulk_window_graph.py
and src/graph/bulk_degree_distribution.py read.

Spark work: the wanted ids are written to a text file and semi-joined with the
index table, so any number of ids costs one scan of two columns. The citation
table is never read.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# Name under which papers with no field of study are counted.
NO_FIELD = "none"


def category_counts(
    papers: Iterable[Hashable],
    fields: Mapping[Hashable, Sequence[str]],
) -> dict[str, Any]:
    """Count each distinct paper once in every one of its fields.

    A paper missing from ``fields`` is unresolved and left out of the shares.
    A paper with an empty field list counts under NO_FIELD. A paper with
    several fields counts in each one, so the shares can add up to more than 1.
    Categories are sorted by count, largest first, then by name.
    """
    counts: Counter[str] = Counter()
    resolved = unresolved = 0
    for paper in set(papers):
        paper_fields = fields.get(paper)
        if paper_fields is None:
            unresolved += 1
            continue
        resolved += 1
        counts.update(set(paper_fields) or {NO_FIELD})
    return {
        "num_papers": resolved + unresolved,
        "num_resolved": resolved,
        "num_unresolved": unresolved,
        "categories": {
            name: {"count": count, "share": count / resolved}
            for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        },
    }


def semi_join_ids(
    spark: Any,
    table: Any,
    values: Iterable[Any],
    column: str,
    folder: Path,
) -> Any:
    """Keep the rows of ``table`` whose ``column`` is one of ``values``.

    The ids reach Spark through a text file, not a literal list or
    ``createDataFrame``, so their number is not limited and no Python worker
    starts. The file must exist until the caller's Spark action finishes.
    """
    from pyspark.sql import functions as F

    path = folder / f"{column}.txt"
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
    ids = spark.read.text(str(path)).select(
        F.col("value").cast(table.schema[column].dataType).alias(column)
    )
    return table.join(ids, column, "left_semi")


def lookup_corpus_ids(
    spark: Any,
    index_root: Path,
    paper_ids: Iterable[str],
) -> dict[str, int]:
    """Map lowercase Semantic Scholar paperIds to corpusIds.

    As in SemanticScholarBulkClient.lookup_paper_ids, the smallest corpusId
    wins when a paperId has several.
    """
    from pyspark.sql import functions as F

    wanted = sorted(set(paper_ids))
    if not wanted:
        return {}
    table = spark.read.parquet(str(index_root / "paper_ids")).select(
        "paperId", "corpusId"
    )
    with tempfile.TemporaryDirectory() as folder:
        rows = (
            semi_join_ids(spark, table, wanted, "paperId", Path(folder))
            .groupBy("paperId")
            .agg(F.min("corpusId").alias("corpusId"))
            .collect()
        )
    return {row["paperId"]: int(row["corpusId"]) for row in rows}


def read_fields_of_study(
    spark: Any,
    index_root: Path,
    corpus_ids: Iterable[int],
) -> dict[int, tuple[str, ...]]:
    """Return the fields of study of each corpusId found in the papers table.

    A paper with several rows gets the union of their fields, because the
    category filters keep a paper when any of its rows passes.
    """
    from pyspark.sql import functions as F

    wanted = sorted(set(corpus_ids))
    if not wanted:
        return {}
    table = spark.read.parquet(str(index_root / "papers")).select(
        "corpusId", "fieldsOfStudy"
    )
    no_fields = F.array().cast("array<string>")
    with tempfile.TemporaryDirectory() as folder:
        rows = (
            semi_join_ids(spark, table, wanted, "corpusId", Path(folder))
            .groupBy("corpusId")
            .agg(
                F.array_distinct(
                    F.flatten(F.collect_list(F.coalesce("fieldsOfStudy", no_fields)))
                ).alias("fieldsOfStudy")
            )
            .collect()
        )
    return {int(row["corpusId"]): tuple(sorted(row["fieldsOfStudy"])) for row in rows}
