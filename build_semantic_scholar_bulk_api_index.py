"""Build Parquet lookup indexes for the local Semantic Scholar bulk client.

Run this after downloading/decompressing the ``papers`` and ``paper-ids``
datasets. ``paper-ids`` supplies the canonical primary SHA returned as
``paperId`` and permits alternate SHA lookups:

```bash
python build_semantic_scholar_bulk_api_index.py
```

The index is optimized for API-like lookup methods in
``SemanticScholarBulkClient``. It creates:

- ``papers``: full paper metadata, partitioned by ``corpusIdBucket``
- ``paper_ids``: paperId/sha -> corpusId, partitioned by ``paperIdBucket``
- ``external_ids``: DOI/arXiv/etc. -> corpusId, partitioned by type and bucket
- ``authors``: author metadata, partitioned by ``authorIdBucket``
- ``author_papers``: authorId -> corpusId, partitioned by ``authorIdBucket``
- ``citations_by_citing`` and ``citations_by_cited``: citation/reference edges
- optional side tables for abstracts, TLDRs, publication venues, and embeddings
- ``s2orc_text``: body text for local snippet search
- ``title_index``: normalized exact title -> corpusId
- ``title_tokens``: title token -> corpusId for bounded title search
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import orjson
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any

from src.clients.semantic_scholar_bulk_client import (
    AUTHOR_COLUMN_RENAMES,
    BULK_INDEX_VERSION,
    CITATION_COLUMN_RENAMES,
    DEFAULT_DATA_ROOT,
    DEFAULT_INDEX_BUCKETS,
    DEFAULT_INDEX_ROOT,
    EXTERNAL_ID_KEYS,
    HEX_SHA_RE,
    INDEX_METADATA_FILE_NAME,
    PAPER_COLUMN_RENAMES,
    TITLE_TOKEN_STOPWORDS,
    add_canonical_paper_ids,
    dataset_json_paths,
    import_pyspark_sql,
)


INDEX_VERSION = BULK_INDEX_VERSION

INDEX_TABLE_NAMES = [
    "papers",
    "paper_ids",
    "external_ids",
    "authors",
    "author_name_tokens",
    "author_papers",
    "citations_by_citing",
    "citations_by_cited",
    "abstracts",
    "tldrs",
    "publication_venues",
    "embeddings",
    "s2orc_text",
    "title_index",
    "title_tokens",
]

SKIPPABLE_TABLE_GROUPS = {
    "external_ids": ["external_ids"],
    "authors": ["authors", "author_name_tokens"],
    "author_papers": ["author_papers"],
    "citations": ["citations_by_citing", "citations_by_cited"],
    "abstracts": ["abstracts"],
    "tldrs": ["tldrs"],
    "publication_venues": ["publication_venues"],
    "embeddings": ["embeddings"],
    "s2orc_text": ["s2orc_text"],
    "titles": ["title_index", "title_tokens"],
}

PAPER_DATASET_NAME = "papers"
PAPER_IDS_DATASET_NAME = "paper-ids"


def dataset_schema(dataset_name: str) -> Any | None:
    """Return explicit schemas for known Semantic Scholar JSONL datasets.

    Spark JSON schema inference can be a full extra pass over very large files.
    These schemas cover the fields this client indexes and returns.
    """
    from pyspark.sql import types as T

    string_map = T.MapType(T.StringType(), T.StringType(), True)
    open_access_info = T.StructType(
        [
            T.StructField("url", T.StringType(), True),
            T.StructField("status", T.StringType(), True),
            T.StructField("license", T.StringType(), True),
            T.StructField("disclaimer", T.StringType(), True),
            T.StructField("externalids", string_map, True),
        ]
    )

    schemas = {
        PAPER_DATASET_NAME: T.StructType(
            [
                T.StructField("corpusid", T.LongType(), True),
                T.StructField("externalids", string_map, True),
                T.StructField("url", T.StringType(), True),
                T.StructField("title", T.StringType(), True),
                T.StructField(
                    "authors",
                    T.ArrayType(
                        T.StructType(
                            [
                                T.StructField("authorId", T.StringType(), True),
                                T.StructField("name", T.StringType(), True),
                            ]
                        )
                    ),
                    True,
                ),
                T.StructField("venue", T.StringType(), True),
                T.StructField("publicationvenueid", T.StringType(), True),
                T.StructField("year", T.IntegerType(), True),
                T.StructField("referencecount", T.LongType(), True),
                T.StructField("citationcount", T.LongType(), True),
                T.StructField("influentialcitationcount", T.LongType(), True),
                T.StructField("isopenaccess", T.BooleanType(), True),
                T.StructField(
                    "s2fieldsofstudy",
                    T.ArrayType(
                        T.StructType(
                            [
                                T.StructField("category", T.StringType(), True),
                                T.StructField("source", T.StringType(), True),
                            ]
                        )
                    ),
                    True,
                ),
                T.StructField("publicationtypes", T.ArrayType(T.StringType()), True),
                T.StructField("publicationdate", T.StringType(), True),
                T.StructField(
                    "journal",
                    T.StructType(
                        [
                            T.StructField("name", T.StringType(), True),
                            T.StructField("pages", T.StringType(), True),
                            T.StructField("volume", T.StringType(), True),
                        ]
                    ),
                    True,
                ),
            ]
        ),
        PAPER_IDS_DATASET_NAME: T.StructType(
            [
                T.StructField("sha", T.StringType(), True),
                T.StructField("corpusid", T.LongType(), True),
                T.StructField("primary", T.BooleanType(), True),
            ]
        ),
        "authors": T.StructType(
            [
                T.StructField("authorid", T.StringType(), True),
                T.StructField("externalids", string_map, True),
                T.StructField("url", T.StringType(), True),
                T.StructField("name", T.StringType(), True),
                T.StructField("aliases", T.ArrayType(T.StringType()), True),
                T.StructField("affiliations", T.ArrayType(T.StringType()), True),
                T.StructField("homepage", T.StringType(), True),
                T.StructField("papercount", T.LongType(), True),
                T.StructField("citationcount", T.LongType(), True),
                T.StructField("hindex", T.LongType(), True),
            ]
        ),
        "citations": T.StructType(
            [
                T.StructField("citationid", T.LongType(), True),
                T.StructField("citingcorpusid", T.LongType(), True),
                T.StructField("citedcorpusid", T.LongType(), True),
                T.StructField("isinfluential", T.BooleanType(), True),
                T.StructField("contexts", T.ArrayType(T.StringType()), True),
                T.StructField("intents", T.ArrayType(T.ArrayType(T.StringType())), True),
            ]
        ),
        "abstracts": T.StructType(
            [
                T.StructField("corpusid", T.LongType(), True),
                T.StructField("openaccessinfo", open_access_info, True),
                T.StructField("abstract", T.StringType(), True),
            ]
        ),
        "tldrs": T.StructType(
            [
                T.StructField("corpusid", T.LongType(), True),
                T.StructField("model", T.StringType(), True),
                T.StructField("text", T.StringType(), True),
            ]
        ),
        "publication-venues": T.StructType(
            [
                T.StructField("id", T.StringType(), True),
                T.StructField("alternate_issns", T.ArrayType(T.StringType()), True),
                T.StructField("alternate_names", T.ArrayType(T.StringType()), True),
                T.StructField("alternate_urls", T.ArrayType(T.StringType()), True),
                T.StructField("issn", T.StringType(), True),
                T.StructField("name", T.StringType(), True),
                T.StructField("type", T.StringType(), True),
                T.StructField("url", T.StringType(), True),
            ]
        ),
        "embeddings-specter_v2": T.StructType(
            [
                T.StructField("corpusid", T.LongType(), True),
                T.StructField("model", T.StringType(), True),
                T.StructField("vector", T.StringType(), True),
            ]
        ),
        "s2orc_v2": T.StructType(
            [
                T.StructField("corpusid", T.LongType(), True),
                T.StructField("openaccessinfo", open_access_info, True),
                T.StructField("title", T.StringType(), True),
                T.StructField("authors", T.ArrayType(T.StringType()), True),
                T.StructField(
                    "body",
                    T.StructType([T.StructField("text", T.StringType(), True)]),
                    True,
                ),
            ]
        ),
        "s2orc": T.StructType(
            [
                T.StructField("corpusid", T.LongType(), True),
                T.StructField("externalids", string_map, True),
                T.StructField(
                    "content",
                    T.StructType(
                        [
                            T.StructField("text", T.StringType(), True),
                            T.StructField(
                                "source",
                                T.StructType(
                                    [
                                        T.StructField(
                                            "oainfo",
                                            T.StructType(
                                                [
                                                    T.StructField("license", T.StringType(), True),
                                                    T.StructField("status", T.StringType(), True),
                                                ]
                                            ),
                                            True,
                                        )
                                    ]
                                ),
                                True,
                            ),
                        ]
                    ),
                    True,
                ),
            ]
        ),
    }
    return schemas.get(dataset_name)


def bucket_column(functions: Any, column: Any, num_buckets: int) -> Any:
    return functions.pmod(
        functions.crc32(column.cast("string")),
        functions.lit(num_buckets),
    ).cast("int")


def normalized_title_column(functions: Any, column: Any) -> Any:
    lower_title = functions.lower(functions.coalesce(column, functions.lit("")))
    alnum_title = functions.regexp_replace(lower_title, r"[^A-Za-z0-9]+", " ")
    return functions.trim(functions.regexp_replace(alnum_title, r"\s+", " "))


def struct_field_names(dataframe: Any, column_name: str) -> set[str]:
    if column_name not in dataframe.columns:
        return set()
    data_type = dataframe.schema[column_name].dataType
    if not hasattr(data_type, "fields"):
        return set()
    return {field.name for field in data_type.fields}


def nested_value_column(functions: Any, dataframe: Any, column_name: str, field_name: str) -> Any | None:
    if column_name not in dataframe.columns:
        return None
    data_type = dataframe.schema[column_name].dataType
    if hasattr(data_type, "fields"):
        field_names = {field.name for field in data_type.fields}
        if field_name in field_names:
            return functions.col(f"{column_name}.{field_name}")
        return None
    if data_type.typeName() == "map":
        return functions.col(column_name).getItem(field_name)
    return None


def rename_dataframe_columns(dataframe: Any, renames: dict[str, str]) -> Any:
    for old_name, new_name in renames.items():
        if old_name in dataframe.columns and new_name not in dataframe.columns:
            dataframe = dataframe.withColumnRenamed(old_name, new_name)
    return dataframe


def read_json_dataset(
    spark_session: Any,
    paths: list[Path],
    dataset_name: str,
    *,
    infer_schema: bool,
) -> Any:
    reader = spark_session.read.option("mode", "PERMISSIVE")
    if not infer_schema:
        schema = dataset_schema(dataset_name)
        if schema is not None:
            reader = reader.schema(schema)
    return reader.json([str(path) for path in paths])


def load_raw_papers(spark_session: Any, raw_root: Path, *, infer_schema: bool = False) -> Any:
    functions = import_pyspark_sql()[2]
    paper_paths = dataset_json_paths(raw_root / "papers")
    if not paper_paths:
        raise FileNotFoundError(f"No paper JSON shards found in {raw_root / 'papers'}")

    papers = read_json_dataset(
        spark_session,
        paper_paths,
        PAPER_DATASET_NAME,
        infer_schema=infer_schema,
    )
    papers = rename_dataframe_columns(papers, PAPER_COLUMN_RENAMES)

    if "s2FieldsOfStudy" in papers.columns and "fieldsOfStudy" not in papers.columns:
        papers = papers.withColumn(
            "fieldsOfStudy",
            functions.expr("array_distinct(transform(s2FieldsOfStudy, item -> item.category))"),
        )

    return papers


def load_raw_paper_ids(spark_session: Any, raw_root: Path, *, infer_schema: bool = False) -> Any:
    paper_id_paths = dataset_json_paths(raw_root / "paper-ids")
    if not paper_id_paths:
        raise FileNotFoundError(
            "The paper-ids dataset is required for API-compatible paperId values: "
            f"{raw_root / 'paper-ids'}"
        )

    paper_ids = read_json_dataset(
        spark_session,
        paper_id_paths,
        PAPER_IDS_DATASET_NAME,
        infer_schema=infer_schema,
    )
    paper_ids = rename_dataframe_columns(paper_ids, {"corpusid": "corpusId"})
    required_columns = {"corpusId", "sha", "primary"}
    missing_columns = required_columns - set(paper_ids.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"The paper-ids dataset is missing required columns: {missing}")
    return paper_ids


def add_primary_paper_ids(papers: Any, paper_ids: Any) -> Any:
    """Attach the release's canonical primary SHA as ``paperId``."""
    functions = import_pyspark_sql()[2]
    return add_canonical_paper_ids(papers, paper_ids, functions)


def load_optional_dataset(
    spark_session: Any,
    raw_root: Path,
    dataset_name: str,
    renames: dict[str, str] | None = None,
    *,
    infer_schema: bool = False,
) -> Any | None:
    paths = dataset_json_paths(raw_root / dataset_name)
    if not paths:
        return None
    dataframe = read_json_dataset(
        spark_session,
        paths,
        dataset_name,
        infer_schema=infer_schema,
    )
    if renames:
        dataframe = rename_dataframe_columns(dataframe, renames)
    return dataframe


def write_table(dataframe: Any, output_path: Path, partition_columns: list[str], overwrite: bool) -> None:
    mode = "overwrite" if overwrite else "errorifexists"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if partition_columns:
        dataframe = dataframe.repartition(
            *[dataframe[column_name] for column_name in partition_columns]
        )
    writer = dataframe.write.mode(mode)
    if partition_columns:
        writer = writer.partitionBy(*partition_columns)
    writer.parquet(str(output_path))


def validate_skip_table_groups(skip_table_groups: set[str]) -> None:
    unknown_groups = skip_table_groups - set(SKIPPABLE_TABLE_GROUPS)
    if unknown_groups:
        valid_groups = ", ".join(sorted(SKIPPABLE_TABLE_GROUPS))
        unknown = ", ".join(sorted(unknown_groups))
        raise ValueError(f"Unknown skip table group(s): {unknown}. Valid groups: {valid_groups}")


def validate_num_buckets(num_buckets: int) -> None:
    if num_buckets < 1:
        raise ValueError("num_buckets must be at least 1")


def validate_resume_metadata(index_root: Path, raw_root: Path, num_buckets: int) -> None:
    metadata_path = index_root / INDEX_METADATA_FILE_NAME
    if not metadata_path.exists():
        return

    metadata = orjson.loads(metadata_path.read_bytes())
    index_version = metadata.get("indexVersion")
    if index_version != INDEX_VERSION:
        raise ValueError(
            "Cannot resume index built with version "
            f"{index_version!r}; expected {INDEX_VERSION}. Use --overwrite to rebuild."
        )

    existing_num_buckets = metadata.get("numBuckets")
    if existing_num_buckets != num_buckets:
        raise ValueError(
            "Cannot resume index with a different bucket count: "
            f"existing={existing_num_buckets!r}, requested={num_buckets}. "
            "Use --overwrite to rebuild."
        )

    existing_raw_root = metadata.get("rawRoot")
    if (
        existing_raw_root is not None
        and Path(existing_raw_root).resolve() != raw_root.resolve()
    ):
        raise ValueError(
            "Cannot resume index from a different raw dataset root: "
            f"existing={existing_raw_root!r}, requested={str(raw_root)!r}. "
            "Use --overwrite to rebuild."
        )


def is_skipped(skip_table_groups: set[str], table_group: str) -> bool:
    return table_group in skip_table_groups


def print_skip(table_group: str) -> None:
    table_names = ", ".join(SKIPPABLE_TABLE_GROUPS[table_group])
    print(f"Skipping {table_group} index group by request ({table_names}).")


def table_is_complete(index_root: Path, table_name: str) -> bool:
    table_path = index_root / table_name
    return table_path.exists() and (table_path / "_SUCCESS").exists()


def table_group_exists(index_root: Path, table_group: str) -> bool:
    return all(table_is_complete(index_root, table_name) for table_name in SKIPPABLE_TABLE_GROUPS[table_group])


def print_reuse(table_group: str) -> None:
    table_names = ", ".join(SKIPPABLE_TABLE_GROUPS[table_group])
    print(f"Reusing existing {table_group} index group ({table_names}).")


def run_step(step_name: str, function: Any, *args: Any, **kwargs: Any) -> Any:
    start = perf_counter()
    print(f"[semantic-scholar-index] START {step_name}", flush=True)
    try:
        return function(*args, **kwargs)
    finally:
        elapsed = perf_counter() - start
        print(f"[semantic-scholar-index] END {step_name} ({elapsed:.1f}s)", flush=True)


def build_papers_table(
    papers: Any,
    index_root: Path,
    num_buckets: int,
    overwrite: bool,
    *,
    reuse_written_table: bool = False,
) -> Any:
    functions = import_pyspark_sql()[2]
    indexed_papers = papers.withColumn(
        "corpusIdBucket",
        bucket_column(functions, functions.col("corpusId"), num_buckets),
    )
    write_table(
        indexed_papers,
        index_root / "papers",
        ["corpusIdBucket"],
        overwrite,
    )
    if reuse_written_table:
        return indexed_papers.sparkSession.read.parquet(str(index_root / "papers"))
    return indexed_papers


def build_paper_ids_table(
    papers: Any,
    paper_ids: Any,
    index_root: Path,
    num_buckets: int,
    overwrite: bool,
) -> None:
    functions = import_pyspark_sql()[2]
    raw_ids = paper_ids.select(
        "corpusId",
        functions.lower(functions.trim("sha")).alias("paperId"),
    )
    canonical_ids = papers.select(
        "corpusId",
        functions.lower(functions.trim("paperId")).alias("paperId"),
    )
    combined = raw_ids.unionByName(canonical_ids)

    combined = combined.filter(
        (functions.col("corpusId") > 0)
        & functions.col("paperId").rlike(HEX_SHA_RE.pattern)
    ).dropDuplicates(["corpusId", "paperId"])

    combined = combined.withColumn(
        "paperIdBucket",
        bucket_column(functions, functions.col("paperId"), num_buckets),
    )
    write_table(combined, index_root / "paper_ids", ["paperIdBucket"], overwrite)


def build_external_ids_table(papers: Any, index_root: Path, num_buckets: int, overwrite: bool) -> None:
    functions = import_pyspark_sql()[2]
    entries = []
    for external_id_type in sorted(set(EXTERNAL_ID_KEYS.values())):
        external_id_value = nested_value_column(functions, papers, "externalIds", external_id_type)
        if external_id_value is None:
            continue
        entries.append(
            functions.struct(
                functions.lit(external_id_type).alias("externalIdType"),
                external_id_value.cast("string").alias("externalIdValue"),
            )
        )
    if not entries:
        print("Skipping external_ids index; no externalIds struct fields found.")
        return

    external_ids = (
        papers.select("corpusId", functions.explode(functions.array(*entries)).alias("externalId"))
        .select(
            "corpusId",
            functions.col("externalId.externalIdType").alias("externalIdType"),
            functions.col("externalId.externalIdValue").alias("externalIdValue"),
        )
        .filter(
            functions.col("externalIdValue").isNotNull()
            & (functions.length(functions.col("externalIdValue")) > 0)
        )
        .withColumn(
            "externalIdBucket",
            bucket_column(functions, functions.col("externalIdValue"), num_buckets),
        )
    )
    write_table(
        external_ids,
        index_root / "external_ids",
        ["externalIdType", "externalIdBucket"],
        overwrite,
    )


def build_author_papers_table(papers: Any, index_root: Path, num_buckets: int, overwrite: bool) -> None:
    functions = import_pyspark_sql()[2]
    if "authors" not in papers.columns:
        print("Skipping author_papers index; no authors column found.")
        return

    author_papers = (
        papers.select(
            "corpusId",
            "paperId",
            "title",
            "year",
            "publicationDate",
            functions.explode("authors").alias("author"),
        )
        .select(
            functions.col("author.authorId").cast("string").alias("authorId"),
            "corpusId",
            "paperId",
            "title",
            "year",
            "publicationDate",
        )
        .filter(functions.col("authorId").isNotNull() & (functions.length("authorId") > 0))
        .withColumn(
            "authorIdBucket",
            bucket_column(functions, functions.col("authorId"), num_buckets),
        )
    )
    write_table(
        author_papers,
        index_root / "author_papers",
        ["authorIdBucket"],
        overwrite,
    )


def build_authors_tables(
    authors: Any | None,
    index_root: Path,
    num_buckets: int,
    overwrite: bool,
    *,
    reuse_written_table: bool = False,
) -> None:
    functions = import_pyspark_sql()[2]
    if authors is None or "authorId" not in authors.columns:
        print("Skipping authors index; no authors dataset found.")
        return

    indexed_authors = authors.withColumn(
        "authorIdBucket",
        bucket_column(functions, functions.col("authorId"), num_buckets),
    )
    write_table(indexed_authors, index_root / "authors", ["authorIdBucket"], overwrite)
    if reuse_written_table:
        indexed_authors = indexed_authors.sparkSession.read.parquet(str(index_root / "authors"))

    if "name" not in indexed_authors.columns:
        return
    named = (
        indexed_authors.select("authorId", "name")
        .withColumn("normalizedName", normalized_title_column(functions, functions.col("name")))
        .filter(functions.length("normalizedName") > 0)
    )
    author_tokens = (
        named.withColumn("authorNameTokens", title_tokens_column(functions, functions.col("normalizedName")))
        .select("authorId", functions.explode("authorNameTokens").alias("authorNameToken"))
        .withColumn(
            "authorNameTokenBucket",
            bucket_column(functions, functions.col("authorNameToken"), num_buckets),
        )
    )
    write_table(
        author_tokens,
        index_root / "author_name_tokens",
        ["authorNameTokenBucket"],
        overwrite,
    )


def build_citation_tables(citations: Any | None, index_root: Path, num_buckets: int, overwrite: bool) -> None:
    functions = import_pyspark_sql()[2]
    if citations is None:
        print("Skipping citations indexes; no citations dataset found.")
        return
    required = {"citationId", "citingCorpusId", "citedCorpusId"}
    if not required.issubset(set(citations.columns)):
        print("Skipping citations indexes; citations dataset is missing required columns.")
        return

    base = citations.select(
        "citationId",
        "citingCorpusId",
        "citedCorpusId",
        "contexts",
        "intents",
        "isInfluential",
    )
    by_citing = base.withColumn(
        "citingCorpusIdBucket",
        bucket_column(functions, functions.col("citingCorpusId"), num_buckets),
    )
    write_table(
        by_citing,
        index_root / "citations_by_citing",
        ["citingCorpusIdBucket"],
        overwrite,
    )

    by_cited_source = by_citing.sparkSession.read.parquet(str(index_root / "citations_by_citing"))
    if "citingCorpusIdBucket" in by_cited_source.columns:
        by_cited_source = by_cited_source.drop("citingCorpusIdBucket")

    by_cited = by_cited_source.withColumn(
        "citedCorpusIdBucket",
        bucket_column(functions, functions.col("citedCorpusId"), num_buckets),
    )
    write_table(
        by_cited,
        index_root / "citations_by_cited",
        ["citedCorpusIdBucket"],
        overwrite,
    )


def build_abstracts_table(abstracts: Any | None, index_root: Path, num_buckets: int, overwrite: bool) -> None:
    functions = import_pyspark_sql()[2]
    if abstracts is None or "corpusId" not in abstracts.columns:
        print("Skipping abstracts table; no abstracts dataset found.")
        return
    columns = ["corpusId", "abstract"]
    dataframe = abstracts
    if "openaccessinfo" in abstracts.columns:
        dataframe = dataframe.withColumn(
            "openAccessPdf",
            functions.struct(
                functions.col("openaccessinfo.url").alias("url"),
                functions.col("openaccessinfo.status").alias("status"),
                functions.col("openaccessinfo.license").alias("license"),
                functions.col("openaccessinfo.disclaimer").alias("disclaimer"),
            ),
        )
        columns.append("openAccessPdf")
    indexed = (
        dataframe.select(*columns)
        .withColumn("corpusIdBucket", bucket_column(functions, functions.col("corpusId"), num_buckets))
    )
    write_table(indexed, index_root / "abstracts", ["corpusIdBucket"], overwrite)


def build_tldrs_table(tldrs: Any | None, index_root: Path, num_buckets: int, overwrite: bool) -> None:
    functions = import_pyspark_sql()[2]
    if tldrs is None or "corpusId" not in tldrs.columns:
        print("Skipping TLDR table; no tldrs dataset found.")
        return
    indexed = (
        tldrs.select("corpusId", "model", "text")
        .withColumn("corpusIdBucket", bucket_column(functions, functions.col("corpusId"), num_buckets))
    )
    write_table(indexed, index_root / "tldrs", ["corpusIdBucket"], overwrite)


def build_publication_venues_table(venues: Any | None, index_root: Path, num_buckets: int, overwrite: bool) -> None:
    functions = import_pyspark_sql()[2]
    if venues is None or "id" not in venues.columns:
        print("Skipping publication venues table; no publication-venues dataset found.")
        return
    indexed = venues.withColumn(
        "publicationVenueIdBucket",
        bucket_column(functions, functions.col("id"), num_buckets),
    )
    write_table(indexed, index_root / "publication_venues", ["publicationVenueIdBucket"], overwrite)


def build_embeddings_table(embeddings: Any | None, index_root: Path, num_buckets: int, overwrite: bool) -> None:
    functions = import_pyspark_sql()[2]
    if embeddings is None or "corpusId" not in embeddings.columns:
        print("Skipping embeddings table; no embeddings dataset found.")
        return
    dataframe = embeddings.select("corpusId", "model", "vector")
    if dict(dataframe.dtypes).get("vector") == "string":
        dataframe = dataframe.withColumn(
            "vector",
            functions.from_json(functions.col("vector"), "array<double>"),
        )
    indexed = dataframe.withColumn(
        "corpusIdBucket",
        bucket_column(functions, functions.col("corpusId"), num_buckets),
    )
    write_table(indexed, index_root / "embeddings", ["corpusIdBucket"], overwrite)


def build_s2orc_text_table(s2orc_text: Any | None, index_root: Path, num_buckets: int, overwrite: bool) -> None:
    functions = import_pyspark_sql()[2]
    if s2orc_text is None or "corpusId" not in s2orc_text.columns:
        print("Skipping S2ORC text table; no s2orc_v2/s2orc dataset found.")
        return

    dataframe = s2orc_text
    if "body" in dataframe.columns:
        dataframe = dataframe.withColumn("bodyText", functions.col("body.text"))
    elif "content" in dataframe.columns:
        dataframe = dataframe.withColumn("bodyText", functions.col("content.text"))
    else:
        dataframe = dataframe.withColumn("bodyText", functions.lit(None).cast("string"))

    if "openaccessinfo" in dataframe.columns:
        dataframe = dataframe.withColumnRenamed("openaccessinfo", "openAccessInfo")
    elif "content" in dataframe.columns:
        dataframe = dataframe.withColumn(
            "openAccessInfo",
            functions.struct(
                functions.col("content.source.oainfo.license").alias("license"),
                functions.col("content.source.oainfo.status").alias("status"),
                functions.lit(None).cast("string").alias("disclaimer"),
            ),
        )
    else:
        dataframe = dataframe.withColumn(
            "openAccessInfo",
            functions.struct(
                functions.lit(None).cast("string").alias("license"),
                functions.lit(None).cast("string").alias("status"),
                functions.lit(None).cast("string").alias("disclaimer"),
            ),
        )

    if "title" not in dataframe.columns:
        dataframe = dataframe.withColumn("title", functions.lit(None).cast("string"))
    if "authors" not in dataframe.columns:
        dataframe = dataframe.withColumn("authors", functions.expr("array()").cast("array<string>"))

    indexed = (
        dataframe.select("corpusId", "title", "authors", "openAccessInfo", "bodyText")
        .withColumn("corpusIdBucket", bucket_column(functions, functions.col("corpusId"), num_buckets))
    )
    write_table(indexed, index_root / "s2orc_text", ["corpusIdBucket"], overwrite)


def title_tokens_column(functions: Any, normalized_title: Any) -> Any:
    stopwords = functions.array(*[functions.lit(word) for word in sorted(TITLE_TOKEN_STOPWORDS)])
    tokens = functions.array_distinct(functions.split(normalized_title, " "))
    return functions.filter(
        tokens,
        lambda token: (functions.length(token) > 1) & (~functions.array_contains(stopwords, token)),
    )


def build_title_tables(
    papers: Any,
    index_root: Path,
    num_buckets: int,
    overwrite: bool,
    *,
    reuse_written_table: bool = False,
) -> None:
    functions = import_pyspark_sql()[2]
    titled = (
        papers.select("corpusId", "title")
        .withColumn("normalizedTitle", normalized_title_column(functions, functions.col("title")))
        .filter(functions.length("normalizedTitle") > 0)
    )

    title_index = titled.withColumn(
        "normalizedTitleBucket",
        bucket_column(functions, functions.col("normalizedTitle"), num_buckets),
    )
    write_table(
        title_index.select("normalizedTitle", "normalizedTitleBucket", "corpusId"),
        index_root / "title_index",
        ["normalizedTitleBucket"],
        overwrite,
    )

    if reuse_written_table:
        titled = title_index.sparkSession.read.parquet(str(index_root / "title_index"))
    title_tokens = (
        titled.withColumn("titleTokens", title_tokens_column(functions, functions.col("normalizedTitle")))
        .select("corpusId", functions.explode("titleTokens").alias("titleToken"))
        .withColumn(
            "titleTokenBucket",
            bucket_column(functions, functions.col("titleToken"), num_buckets),
        )
    )
    write_table(
        title_tokens,
        index_root / "title_tokens",
        ["titleTokenBucket"],
        overwrite,
    )


def write_metadata(
    index_root: Path,
    raw_root: Path,
    num_buckets: int,
    skipped_table_groups: set[str] | None = None,
) -> None:
    built_tables = [table_name for table_name in INDEX_TABLE_NAMES if table_is_complete(index_root, table_name)]
    metadata = {
        "indexVersion": INDEX_VERSION,
        "numBuckets": num_buckets,
        "rawRoot": str(raw_root.resolve()),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "tables": built_tables,
        "skippedTableGroups": sorted(skipped_table_groups or []),
    }
    metadata_path = index_root / INDEX_METADATA_FILE_NAME
    metadata_path.write_text(orjson.dumps(metadata, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS).decode("utf-8"), encoding="utf-8")


def build_semantic_scholar_bulk_index(
    *,
    raw_root: Path,
    index_root: Path,
    num_buckets: int,
    overwrite: bool,
    spark_session: Any | None = None,
    skip_table_groups: set[str] | list[str] | tuple[str, ...] | None = None,
    infer_schema: bool = False,
    resume_existing: bool = False,
    reuse_written_sources: bool = False,
) -> None:
    skip_table_groups = set(skip_table_groups or [])
    validate_skip_table_groups(skip_table_groups)
    validate_num_buckets(num_buckets)

    if overwrite and resume_existing:
        raise ValueError("overwrite and resume_existing cannot both be true")
    if resume_existing:
        validate_resume_metadata(index_root, raw_root, num_buckets)

    SparkSession = import_pyspark_sql()[0]
    spark = spark_session or SparkSession.builder.appName("build_semantic_scholar_bulk_api_index").getOrCreate()

    if overwrite and index_root.exists():
        shutil.rmtree(index_root)
    index_root.mkdir(parents=True, exist_ok=True)
    table_overwrite = overwrite or resume_existing

    paper_ids = None
    if resume_existing and table_is_complete(index_root, "papers"):
        print("Reusing existing papers table.")
        indexed_papers = spark.read.parquet(str(index_root / "papers"))
    else:
        papers = run_step("load raw papers", load_raw_papers, spark, raw_root, infer_schema=infer_schema)
        paper_ids = run_step(
            "load raw paper-ids",
            load_raw_paper_ids,
            spark,
            raw_root,
            infer_schema=infer_schema,
        )
        papers = add_primary_paper_ids(papers, paper_ids)
        reuse_written_papers = True
        indexed_papers = run_step(
            "build papers",
            build_papers_table,
            papers,
            index_root,
            num_buckets,
            table_overwrite,
            reuse_written_table=reuse_written_papers,
        )

    if resume_existing and table_is_complete(index_root, "paper_ids"):
        print("Reusing existing paper_ids table.")
    else:
        if paper_ids is None:
            paper_ids = run_step(
                "load raw paper-ids",
                load_raw_paper_ids,
                spark,
                raw_root,
                infer_schema=infer_schema,
            )
        run_step("build paper_ids", build_paper_ids_table, indexed_papers, paper_ids, index_root, num_buckets, table_overwrite)

    if is_skipped(skip_table_groups, "external_ids"):
        print_skip("external_ids")
    elif resume_existing and table_group_exists(index_root, "external_ids"):
        print_reuse("external_ids")
    else:
        run_step("build external_ids", build_external_ids_table, indexed_papers, index_root, num_buckets, table_overwrite)

    if is_skipped(skip_table_groups, "authors"):
        print_skip("authors")
    elif resume_existing and table_group_exists(index_root, "authors"):
        print_reuse("authors")
    else:
        authors = run_step(
            "load raw authors",
            load_optional_dataset,
            spark,
            raw_root,
            "authors",
            AUTHOR_COLUMN_RENAMES,
            infer_schema=infer_schema,
        )
        run_step(
            "build authors",
            build_authors_tables,
            authors,
            index_root,
            num_buckets,
            table_overwrite,
            reuse_written_table=reuse_written_sources,
        )

    if is_skipped(skip_table_groups, "author_papers"):
        print_skip("author_papers")
    elif resume_existing and table_group_exists(index_root, "author_papers"):
        print_reuse("author_papers")
    else:
        run_step("build author_papers", build_author_papers_table, indexed_papers, index_root, num_buckets, table_overwrite)

    if is_skipped(skip_table_groups, "citations"):
        print_skip("citations")
    elif resume_existing and table_group_exists(index_root, "citations"):
        print_reuse("citations")
    else:
        citations = run_step(
            "load raw citations",
            load_optional_dataset,
            spark,
            raw_root,
            "citations",
            CITATION_COLUMN_RENAMES,
            infer_schema=infer_schema,
        )
        run_step("build citations", build_citation_tables, citations, index_root, num_buckets, table_overwrite)

    if is_skipped(skip_table_groups, "abstracts"):
        print_skip("abstracts")
    elif resume_existing and table_group_exists(index_root, "abstracts"):
        print_reuse("abstracts")
    else:
        abstracts = run_step(
            "load raw abstracts",
            load_optional_dataset,
            spark,
            raw_root,
            "abstracts",
            {"corpusid": "corpusId"},
            infer_schema=infer_schema,
        )
        run_step("build abstracts", build_abstracts_table, abstracts, index_root, num_buckets, table_overwrite)

    if is_skipped(skip_table_groups, "tldrs"):
        print_skip("tldrs")
    elif resume_existing and table_group_exists(index_root, "tldrs"):
        print_reuse("tldrs")
    else:
        tldrs = run_step(
            "load raw tldrs",
            load_optional_dataset,
            spark,
            raw_root,
            "tldrs",
            {"corpusid": "corpusId"},
            infer_schema=infer_schema,
        )
        run_step("build tldrs", build_tldrs_table, tldrs, index_root, num_buckets, table_overwrite)

    if is_skipped(skip_table_groups, "publication_venues"):
        print_skip("publication_venues")
    elif resume_existing and table_group_exists(index_root, "publication_venues"):
        print_reuse("publication_venues")
    else:
        venues = run_step(
            "load raw publication venues",
            load_optional_dataset,
            spark,
            raw_root,
            "publication-venues",
            infer_schema=infer_schema,
        )
        run_step("build publication_venues", build_publication_venues_table, venues, index_root, num_buckets, table_overwrite)

    if is_skipped(skip_table_groups, "embeddings"):
        print_skip("embeddings")
    elif resume_existing and table_group_exists(index_root, "embeddings"):
        print_reuse("embeddings")
    else:
        embeddings = run_step(
            "load raw embeddings",
            load_optional_dataset,
            spark,
            raw_root,
            "embeddings-specter_v2",
            {"corpusid": "corpusId"},
            infer_schema=infer_schema,
        )
        run_step("build embeddings", build_embeddings_table, embeddings, index_root, num_buckets, table_overwrite)

    if is_skipped(skip_table_groups, "s2orc_text"):
        print_skip("s2orc_text")
    elif resume_existing and table_group_exists(index_root, "s2orc_text"):
        print_reuse("s2orc_text")
    else:
        s2orc_text = run_step(
            "load raw s2orc_v2",
            load_optional_dataset,
            spark,
            raw_root,
            "s2orc_v2",
            {"corpusid": "corpusId"},
            infer_schema=infer_schema,
        )
        if s2orc_text is None:
            s2orc_text = run_step(
                "load raw s2orc",
                load_optional_dataset,
                spark,
                raw_root,
                "s2orc",
                {"corpusid": "corpusId"},
                infer_schema=infer_schema,
            )
        run_step("build s2orc_text", build_s2orc_text_table, s2orc_text, index_root, num_buckets, table_overwrite)

    if is_skipped(skip_table_groups, "titles"):
        print_skip("titles")
    elif resume_existing and table_group_exists(index_root, "titles"):
        print_reuse("titles")
    else:
        run_step(
            "build titles",
            build_title_tables,
            indexed_papers,
            index_root,
            num_buckets,
            table_overwrite,
            reuse_written_table=reuse_written_sources,
        )

    write_metadata(index_root, raw_root, num_buckets, skip_table_groups)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Parquet indexes for Semantic Scholar bulk API lookups.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root containing raw_datasets/papers and raw_datasets/paper-ids.",
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=DEFAULT_INDEX_ROOT,
        help="Directory where Parquet index tables will be written.",
    )
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=DEFAULT_INDEX_BUCKETS,
        help="Hash bucket count for partitioned lookup tables.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild an existing index.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Reuse completed tables in an existing index and build missing groups. "
            "Incomplete Spark output directories are overwritten."
        ),
    )
    parser.add_argument(
        "--spark-master",
        default=None,
        help="Optional Spark master URL, e.g. local[*] or spark://host:7077.",
    )
    parser.add_argument(
        "--shuffle-partitions",
        type=int,
        default=None,
        help="Optional spark.sql.shuffle.partitions value.",
    )
    parser.add_argument(
        "--driver-memory",
        default=None,
        help="Optional Spark driver heap size, e.g. 64g.",
    )
    parser.add_argument(
        "--executor-memory",
        default=None,
        help="Optional Spark executor heap size, e.g. 64g.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Optional Spark spill/local directory. Prefer a large scratch path.",
    )
    parser.add_argument(
        "--skip-table",
        action="append",
        choices=sorted(SKIPPABLE_TABLE_GROUPS),
        default=[],
        metavar="GROUP",
        help=(
            "Repeatable index group to skip. Useful for building core lookup tables first; "
            "large groups include citations, embeddings, and s2orc_text."
        ),
    )
    parser.add_argument(
        "--infer-schema",
        action="store_true",
        help=(
            "Ask Spark to infer JSON schemas instead of using the built-in Semantic Scholar schemas. "
            "This is slower, but useful if the bulk format changes."
        ),
    )
    parser.add_argument(
        "--reuse-written-sources",
        action="store_true",
        help=(
            "Build derived indexes from already-written Parquet sources instead of the active Spark lineage. "
            "This can reduce JSON rereads, but may be slower on filesystems with expensive partition listing."
        ),
    )
    parser.add_argument(
        "--input-max-partition-bytes",
        default=None,
        help="Optional spark.sql.files.maxPartitionBytes value for reading raw JSON, e.g. 512m.",
    )
    parser.add_argument(
        "--max-records-per-file",
        type=int,
        default=None,
        help="Optional spark.sql.files.maxRecordsPerFile value for output Parquet files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.overwrite and args.resume_existing:
        raise ValueError("--overwrite and --resume-existing cannot be used together")
    SparkSession = import_pyspark_sql()[0]
    builder = SparkSession.builder.appName("build_semantic_scholar_bulk_api_index")
    if args.spark_master:
        builder = builder.master(args.spark_master)
    if args.shuffle_partitions is not None:
        builder = builder.config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
    if args.driver_memory:
        builder = builder.config("spark.driver.memory", args.driver_memory)
    if args.executor_memory:
        builder = builder.config("spark.executor.memory", args.executor_memory)
    if args.local_dir:
        args.local_dir.mkdir(parents=True, exist_ok=True)
        builder = builder.config("spark.local.dir", str(args.local_dir))
    if args.input_max_partition_bytes:
        builder = builder.config("spark.sql.files.maxPartitionBytes", str(args.input_max_partition_bytes))
    if args.max_records_per_file is not None:
        builder = builder.config("spark.sql.files.maxRecordsPerFile", str(args.max_records_per_file))
    spark = builder.getOrCreate()

    build_semantic_scholar_bulk_index(
        raw_root=args.raw_root,
        index_root=args.index_root,
        num_buckets=args.num_buckets,
        overwrite=args.overwrite,
        spark_session=spark,
        skip_table_groups=args.skip_table,
        infer_schema=args.infer_schema,
        resume_existing=args.resume_existing,
        reuse_written_sources=args.reuse_written_sources,
    )
    print(f"Semantic Scholar bulk index written to: {args.index_root}")


if __name__ == "__main__":
    main()
