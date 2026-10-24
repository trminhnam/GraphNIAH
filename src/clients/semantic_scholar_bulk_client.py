"""Local Semantic Scholar bulk-dataset client backed by PySpark.

The class mirrors the public methods of ``SemanticScholarClient`` where the
downloaded bulk datasets make that possible. Exact Semantic Scholar relevance
ranking and proprietary snippet retrieval are approximated with deterministic
local title/name/text matching.

Run examples:

```bash
python -m src.clients.semantic_scholar_bulk_client
```
"""

from __future__ import annotations

import argparse
import re
import zlib
from collections import OrderedDict
from collections.abc import Iterator, KeysView
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import orjson
import requests

from src.clients.semantic_scholar_client import (
    _DEFAULT_AUTHOR_FIELDS,
    _DEFAULT_CITE_FIELDS,
    _DEFAULT_PAPER_FIELDS,
    _DEFAULT_REF_FIELDS,
    DEFAULT_MAX_RESULTS,
    normalize_paper_id,
)

DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "semantic_scholar_bulk"
    / "raw_datasets"
)
DEFAULT_INDEX_ROOT = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "semantic_scholar_bulk"
    / "index"
)
DEFAULT_BULK_PAGE_SIZE = 1000
DEFAULT_INDEX_BUCKETS = 8192
DEFAULT_SEARCH_CANDIDATE_LIMIT = 50000
DEFAULT_PAPER_CACHE_BYTES = 64 * 1024 * 1024
DEFAULT_SPARK_MASTER = "local[64]"
DEFAULT_SPARK_DRIVER_MEMORY = "128g"
DEFAULT_SPARK_SHUFFLE_PARTITIONS = 192
DEFAULT_SPARK_INPUT_MAX_PARTITION_BYTES = "64m"
INDEX_METADATA_FILE_NAME = "metadata.json"
BULK_INDEX_VERSION = 4
# Read on every title-search sub-batch (exact_title_matches/fuzzy_title_matches);
# small enough (tens of GB) to persist for a client's lifetime with cache_title_index=True,
# unlike the much larger raw "papers" table.
TITLE_SEARCH_INDEX_TABLES = ("title_index", "title_tokens")
HEX_SHA_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
TITLE_TOKEN_STOPWORDS = {
    "a",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "with",
    "you",
}

PAPER_COLUMN_RENAMES = {
    "corpusid": "corpusId",
    "externalids": "externalIds",
    "publicationvenueid": "publicationVenueId",
    "referencecount": "referenceCount",
    "citationcount": "citationCount",
    "influentialcitationcount": "influentialCitationCount",
    "isopenaccess": "isOpenAccess",
    "s2fieldsofstudy": "s2FieldsOfStudy",
    "publicationtypes": "publicationTypes",
    "publicationdate": "publicationDate",
}

AUTHOR_COLUMN_RENAMES = {
    "authorid": "authorId",
    "externalids": "externalIds",
    "papercount": "paperCount",
    "citationcount": "citationCount",
    "hindex": "hIndex",
}

CITATION_COLUMN_RENAMES = {
    "citationid": "citationId",
    "citingcorpusid": "citingCorpusId",
    "citedcorpusid": "citedCorpusId",
    "isinfluential": "isInfluential",
}

EXTERNAL_ID_KEYS = {
    "ARXIV": "ArXiv",
    "ACL": "ACL",
    "CorpusId": "CorpusId",
    "DBLP": "DBLP",
    "DOI": "DOI",
    "MAG": "MAG",
    "PMCID": "PubMedCentral",
    "PMID": "PubMed",
    "PubMed": "PubMed",
    "PubMedCentral": "PubMedCentral",
}

# Expansion side -> (edge index table, source column, target column,
# neighbor wrapper field, default neighbor fields).
CITATION_SIDES = {
    "citations": (
        "citations_by_cited",
        "citedCorpusId",
        "citingCorpusId",
        "citingPaper",
        _DEFAULT_CITE_FIELDS,
    ),
    "references": (
        "citations_by_citing",
        "citingCorpusId",
        "citedCorpusId",
        "citedPaper",
        _DEFAULT_REF_FIELDS,
    ),
}


def import_pyspark_sql() -> tuple[Any, Any, Any]:
    try:
        from pyspark.sql import SparkSession, Window
        from pyspark.sql import functions as spark_functions
    except ImportError as exc:
        raise ImportError(
            "pyspark is required for SemanticScholarBulkClient. "
            "Install it with `pip install pyspark` or run on a Spark server."
        ) from exc
    return SparkSession, Window, spark_functions


def dataset_json_paths(dataset_dir: Path) -> list[Path]:
    """Return usable local JSON shard paths, preferring decompressed duplicates."""
    if not dataset_dir.exists():
        return []

    files = [
        path
        for path in dataset_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
        and path.name != "description.txt"
        and not path.name.endswith(".part")
    ]
    file_names = {path.name for path in files}
    selected: list[Path] = []
    for path in sorted(files):
        if path.suffix == ".gz" and path.name.removesuffix(".gz") in file_names:
            continue
        selected.append(path)
    return selected


def normalize_requested_fields(
    fields: Optional[Iterable[str] | str],
    default_fields: Iterable[str],
) -> list[str]:
    if fields is None:
        return list(default_fields)
    if isinstance(fields, str):
        return [field.strip() for field in fields.split(",") if field.strip()]
    return [str(field).strip() for field in fields if str(field).strip()]


def normalize_string_values(values: Optional[Iterable[str] | str]) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [value.strip() for value in values.split(",") if value.strip()]
    return [str(value).strip() for value in values if str(value).strip()]


def canonical_paper_record(record: dict[str, Any]) -> dict[str, Any]:
    paper = dict(record)
    for old_name, new_name in PAPER_COLUMN_RENAMES.items():
        if old_name in paper and new_name not in paper:
            paper[new_name] = paper.pop(old_name)

    fields_of_study = paper.get("fieldsOfStudy")
    s2_fields = paper.get("s2FieldsOfStudy")
    if fields_of_study is None and isinstance(s2_fields, list):
        categories = [
            item.get("category")
            for item in s2_fields
            if isinstance(item, dict) and item.get("category")
        ]
        paper["fieldsOfStudy"] = list(dict.fromkeys(categories)) or None

    return paper


def canonical_author_record(record: dict[str, Any]) -> dict[str, Any]:
    author = dict(record)
    for old_name, new_name in AUTHOR_COLUMN_RENAMES.items():
        if old_name in author and new_name not in author:
            author[new_name] = author.pop(old_name)

    affiliations = author.get("affiliations")
    if isinstance(affiliations, str):
        author["affiliations"] = [affiliations] if affiliations else []
    return author


def canonical_citation_record(record: dict[str, Any]) -> dict[str, Any]:
    citation = dict(record)
    for old_name, new_name in CITATION_COLUMN_RENAMES.items():
        if old_name in citation and new_name not in citation:
            citation[new_name] = citation.pop(old_name)
    return citation


def add_canonical_paper_ids(papers: Any, paper_ids: Any, functions: Any) -> Any:
    """Attach the release's canonical SHA as each paper's ``paperId``.

    Semantic Scholar documents one primary SHA per corpus ID, but some release
    snapshots mark zero or multiple SHAs as primary. If none is marked, use a
    valid SHA from the paper's own ``url``. If multiple are marked, use the URL
    SHA only when it is one of those primary mappings. Records with a null,
    zero, or negative corpus ID are invalid and are discarded.
    """
    required_columns = {"corpusId", "sha", "primary"}
    missing_columns = required_columns - set(paper_ids.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"The paper-ids dataset is missing required columns: {missing}")

    valid_primary_ids = (
        paper_ids.filter(functions.col("primary").cast("boolean"))
        .select(
            "corpusId",
            functions.lower(functions.trim("sha")).alias("primaryPaperId"),
        )
        .filter(
            (functions.col("corpusId") > 0)
            & functions.col("primaryPaperId").rlike(HEX_SHA_RE.pattern)
        )
        .dropDuplicates(["corpusId", "primaryPaperId"])
    )
    primary_ids = (
        valid_primary_ids
        .groupBy("corpusId")
        .agg(
            functions.min("primaryPaperId").alias("singlePrimaryPaperId"),
            functions.countDistinct("primaryPaperId").alias("primaryIdCount"),
        )
    )

    if "url" in papers.columns:
        extracted_url_paper_id = functions.lower(
            functions.regexp_extract(
                functions.coalesce(functions.col("url").cast("string"), functions.lit("")),
                r"(?:^|/)([0-9a-fA-F]{40})(?:[/?#]|$)",
                1,
            )
        )
        url_paper_id = functions.when(
            extracted_url_paper_id.rlike(HEX_SHA_RE.pattern),
            extracted_url_paper_id,
        )
    else:
        url_paper_id = functions.lit(None).cast("string")

    valid_papers = papers.filter(functions.col("corpusId") > 0)
    papers_without_id = (
        valid_papers.drop("paperId")
        if "paperId" in valid_papers.columns
        else valid_papers
    )
    joined = (
        papers_without_id.withColumn("urlPaperId", url_paper_id)
        .join(primary_ids, on="corpusId", how="left")
        .withColumn(
            "canonicalPaperId",
            functions.when(
                functions.col("primaryIdCount") == 1,
                functions.col("singlePrimaryPaperId"),
            ).otherwise(
                functions.col("urlPaperId"),
            ),
        )
    )

    conflicting_candidates = (
        joined.filter(functions.col("primaryIdCount") > 1)
        .select("corpusId", "canonicalPaperId")
        .dropDuplicates()
    )
    valid_conflicting_candidates = valid_primary_ids.select(
        "corpusId",
        functions.col("primaryPaperId").alias("canonicalPaperId"),
    )
    conflict = (
        conflicting_candidates.join(
            valid_conflicting_candidates,
            on=["corpusId", "canonicalPaperId"],
            how="left_anti",
        )
        .select("corpusId")
        .limit(1)
        .collect()
    )
    if conflict:
        raise ValueError(
            "The paper-ids dataset has multiple primary SHAs for corpusId "
            f"{conflict[0]['corpusId']} and papers.url does not identify one of them."
        )

    missing = (
        joined.filter(
            functions.col("corpusId").isNotNull()
            & functions.col("canonicalPaperId").isNull()
        )
        .select("corpusId")
        .limit(1)
        .collect()
    )
    if missing:
        raise ValueError(
            "The paper-ids dataset has no valid primary SHA for corpusId "
            f"{missing[0]['corpusId']}."
        )
    return (
        joined.drop("urlPaperId", "singlePrimaryPaperId", "primaryIdCount")
        .withColumnRenamed("canonicalPaperId", "paperId")
    )


def select_paper_fields(
    paper: dict[str, Any],
    fields: Optional[Iterable[str] | str],
    default_fields: Iterable[str],
) -> dict[str, Any]:
    requested = normalize_requested_fields(fields, default_fields)
    top_level_fields = {field.split(".", 1)[0] for field in requested}
    top_level_fields.add("paperId")

    selected = {
        field: paper[field]
        for field in top_level_fields
        if field in paper
    }
    return selected


def select_author_fields(
    author: dict[str, Any],
    fields: Optional[Iterable[str] | str],
) -> dict[str, Any]:
    requested = normalize_requested_fields(fields, _DEFAULT_AUTHOR_FIELDS)
    requested_fields = {field.split(".", 1)[0] for field in requested}
    requested_fields.update({"authorId", "name"})
    return {
        field: author[field]
        for field in requested_fields
        if field in author
    }


def requested_top_level_fields(
    fields: Optional[Iterable[str] | str],
    default_fields: Iterable[str],
) -> set[str]:
    requested = normalize_requested_fields(fields, default_fields)
    return {field.split(".", 1)[0] for field in requested}


def edge_paper_fields(
    fields: Optional[Iterable[str] | str],
    wrapper_field: str,
    default_fields: Iterable[str],
) -> list[str]:
    requested = normalize_requested_fields(fields, default_fields)
    prefix = f"{wrapper_field}."
    paper_fields = [
        field.removeprefix(prefix)
        for field in requested
        if field == wrapper_field or field.startswith(prefix)
    ]
    if not paper_fields:
        paper_fields = [
            field.split(".", 1)[1]
            for field in requested
            if "." in field and field.split(".", 1)[0] in {"citedPaper", "citingPaper"}
        ]
    if not paper_fields:
        paper_fields = [
            field
            for field in requested
            if field not in {"contexts", "intents", "contextsWithIntent", "isInfluential"}
        ]
    return paper_fields or ["paperId", "title"]


def author_requested_paper_fields(fields: Optional[Iterable[str] | str]) -> list[str]:
    requested = normalize_requested_fields(fields, _DEFAULT_AUTHOR_FIELDS)
    paper_fields = [
        field.split(".", 1)[1]
        for field in requested
        if field.startswith("papers.") and "." in field
    ]
    return paper_fields or ["paperId", "title"]


def flatten_intents(intents: Any) -> list[str]:
    flattened: list[str] = []
    if not isinstance(intents, list):
        return flattened
    for item in intents:
        if isinstance(item, list):
            flattened.extend(str(value) for value in item if value)
        elif item:
            flattened.append(str(item))
    return list(dict.fromkeys(flattened))


def contexts_with_intent(contexts: Any, intents: Any) -> list[dict[str, Any]]:
    if not isinstance(contexts, list):
        return []
    result: list[dict[str, Any]] = []
    intent_groups = intents if isinstance(intents, list) else []
    for index, context in enumerate(contexts):
        item_intents = intent_groups[index] if index < len(intent_groups) else []
        if not isinstance(item_intents, list):
            item_intents = [item_intents] if item_intents else []
        result.append({"context": context, "intents": item_intents})
    return result


def response_not_found(message: str) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = 404
    response.reason = message
    return requests.HTTPError(message, response=response)


def query_terms(query: str) -> list[str]:
    return [term.lower() for term in TOKEN_RE.findall(query)]


def query_terms_for_index(query: str) -> list[str]:
    terms = [
        term
        for term in query_terms(query)
        if len(term) > 1 and term not in TITLE_TOKEN_STOPWORDS
    ]
    return terms or [term for term in query_terms(query) if len(term) > 1]


def normalize_title_text(title: str | None) -> str:
    return " ".join(query_terms(title or ""))


def stable_bucket(value: Any, num_buckets: int) -> int:
    return zlib.crc32(str(value).encode("utf-8")) % num_buckets


def year_bounds(year: str) -> tuple[int | None, int | None]:
    year = year.strip()
    if not year:
        return None, None
    if ":" in year:
        start, end = year.split(":", 1)
    elif "-" in year:
        start, end = year.split("-", 1)
    else:
        start = end = year

    start_year = int(start) if start else None
    end_year = int(end) if end else None
    return start_year, end_year


def date_prefix_lower(value: str) -> str:
    parts = value.split("-")
    if len(parts) == 1:
        return f"{parts[0]}-01-01"
    if len(parts) == 2:
        return f"{parts[0]}-{parts[1]}-01"
    return value


def date_prefix_upper(value: str) -> str:
    parts = value.split("-")
    if len(parts) == 1:
        return f"{parts[0]}-12-31"
    if len(parts) == 2:
        return f"{parts[0]}-{parts[1]}-31"
    return value


def author_year(authors: list[dict[str, Any]] | None, year: Any) -> str:
    if not authors:
        return str(year or "")
    first_author = authors[0].get("name", "") if isinstance(authors[0], dict) else ""
    suffix = " et al." if len(authors) > 1 else ""
    if year:
        return f"{first_author}{suffix}, {year}"
    return f"{first_author}{suffix}"


def sort_components(sort: str | None) -> tuple[str, bool] | None:
    if not sort:
        return None
    field, _, direction = sort.partition(":")
    field = field.strip()
    if not field:
        return None
    return field, direction.strip().lower() != "asc"


@dataclass
class ExpansionSubgraphBatch:
    """Unique metadata and ordered neighbor IDs for a bounded group of sources."""

    source_ids: dict[str, int | None]
    papers: dict[int, dict[str, Any]]
    citations: dict[int, list[int | None]]
    references: dict[int, list[int | None]]

    def keys(self) -> KeysView[str]:
        return self.source_ids.keys()

    def get(self, paper_id: str, default: Any = None) -> Any:
        """Adapt just one source to the existing atomic writer's input format."""
        if paper_id not in self.source_ids:
            return default
        corpus_id = self.source_ids[paper_id]
        paper = self.papers.get(corpus_id)
        if paper is None:
            return None, [], []
        return (
            paper,
            [
                {"citingPaper": self.papers.get(target)}
                for target in self.citations.get(corpus_id, [])
            ],
            [
                {"citedPaper": self.papers.get(target)}
                for target in self.references.get(corpus_id, [])
            ],
        )


def expansion_source_groups(
    source_ids: dict[str, int | None],
    counts: dict[int, int],
    edge_budget: int,
) -> Iterator[dict[str, int | None]]:
    """Preserve input order; an indivisible hub may exceed the edge budget."""
    group: dict[str, int | None] = {}
    seen: set[int | None] = set()
    size = 0
    for paper_id, corpus_id in source_ids.items():
        cost = 0 if corpus_id in seen else counts.get(corpus_id, 0)
        if group and size + cost > edge_budget:
            yield group
            group, seen, size = {}, set(), 0
            cost = counts.get(corpus_id, 0)
        group[paper_id] = corpus_id
        seen.add(corpus_id)
        size += cost
    if group:
        yield group


class SemanticScholarBulkClient:
    """Offline Semantic Scholar client using local bulk JSON shards and PySpark."""

    def __init__(
        self,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        *,
        spark_session: Any = None,
        app_name: str = "semantic_scholar_bulk_client",
        cache: bool = False,
        cache_title_index: bool = False,
        bulk_page_size: int = DEFAULT_BULK_PAGE_SIZE,
        index_root: str | Path | None = DEFAULT_INDEX_ROOT,
        use_index: bool = True,
        search_candidate_limit: int = DEFAULT_SEARCH_CANDIDATE_LIMIT,
        spark_master: str = DEFAULT_SPARK_MASTER,
        spark_driver_memory: str = DEFAULT_SPARK_DRIVER_MEMORY,
        spark_shuffle_partitions: int = DEFAULT_SPARK_SHUFFLE_PARTITIONS,
        spark_input_max_partition_bytes: str = (
            DEFAULT_SPARK_INPUT_MAX_PARTITION_BYTES
        ),
        paper_cache_max_bytes: int = 0,
    ) -> None:
        if paper_cache_max_bytes < 0:
            raise ValueError("paper_cache_max_bytes must be non-negative")
        SparkSession, Window, spark_functions = import_pyspark_sql()
        self.spark_functions = spark_functions
        self.window = Window
        self.created_spark_session = spark_session is None
        if self.created_spark_session:
            spark_session = (
                SparkSession.builder.appName(app_name)
                .master(spark_master)
                .config("spark.driver.memory", spark_driver_memory)
                .config(
                    "spark.sql.shuffle.partitions",
                    str(spark_shuffle_partitions),
                )
                .config(
                    "spark.sql.files.maxPartitionBytes",
                    spark_input_max_partition_bytes,
                )
                .getOrCreate()
            )
        self.spark_session = spark_session
        self.data_root = Path(data_root)
        self.index_root = Path(index_root) if index_root is not None else None
        self.bulk_page_size = bulk_page_size
        self.search_candidate_limit = search_candidate_limit
        self.index_metadata = self.load_index_metadata() if use_index else None
        self.use_index = self.index_metadata is not None
        self.cache_title_index = cache_title_index
        self.persisted_index_tables: set[str] = set()
        self.index_dataframes: dict[str, Any] = {}
        self.raw_dataframes: dict[str, Any] = {}
        self.paper_ids_dataframe: Any | None = None
        # Cache immutable response bytes, scoped to this client's dataset release.
        # The limit bounds serialized payloads; dictionary keys add some overhead.
        self.paper_cache_max_bytes = paper_cache_max_bytes
        self.paper_cache_size_bytes = 0
        self.paper_cache: OrderedDict[tuple[tuple[str, ...], int], bytes] = OrderedDict()
        self.num_buckets = int(
            (self.index_metadata or {}).get("numBuckets", DEFAULT_INDEX_BUCKETS)
        )

        if self.use_index:
            self.papers_dataframe = self.read_index_table("papers")
        else:
            self.papers_dataframe = self.load_raw_papers_dataframe()
        if cache:
            self.papers_dataframe = self.papers_dataframe.cache()

    def close(self, *, stop_spark: bool = False) -> None:
        for table_name in self.persisted_index_tables:
            self.index_dataframes[table_name].unpersist()
        self.persisted_index_tables.clear()
        if stop_spark and self.created_spark_session:
            self.spark_session.stop()

    def load_index_metadata(self) -> dict[str, Any] | None:
        if self.index_root is None:
            return None
        metadata_path = self.index_root / INDEX_METADATA_FILE_NAME
        if not metadata_path.exists():
            return None
        metadata = orjson.loads(metadata_path.read_bytes())
        index_version = metadata.get("indexVersion")
        if index_version != BULK_INDEX_VERSION:
            raise ValueError(
                "Unsupported Semantic Scholar bulk index version "
                f"{index_version!r}; expected {BULK_INDEX_VERSION}. Rebuild the index."
            )
        required_tables = {"papers", "paper_ids"}
        indexed_tables = set(metadata.get("tables") or [])
        missing_tables = required_tables - indexed_tables
        if missing_tables:
            missing = ", ".join(sorted(missing_tables))
            raise ValueError(
                "Semantic Scholar bulk index is missing required table(s): "
                f"{missing}. Rebuild the index."
            )
        return metadata

    def index_table_path(self, table_name: str) -> Path:
        if self.index_root is None:
            raise FileNotFoundError("No Semantic Scholar index root configured")
        return self.index_root / table_name

    def index_table_exists(self, table_name: str) -> bool:
        return self.index_root is not None and self.index_table_path(table_name).exists()

    def read_index_table(self, table_name: str) -> Any:
        if table_name not in self.index_dataframes:
            table_path = self.index_table_path(table_name)
            if not table_path.exists():
                raise FileNotFoundError(f"Missing Semantic Scholar index table: {table_path}")
            dataframe = self.spark_session.read.parquet(str(table_path))
            if self.cache_title_index and table_name in TITLE_SEARCH_INDEX_TABLES:
                from pyspark import StorageLevel

                dataframe = dataframe.persist(StorageLevel.MEMORY_AND_DISK)
                self.persisted_index_tables.add(table_name)
            self.index_dataframes[table_name] = dataframe
        return self.index_dataframes[table_name]

    def load_raw_papers_dataframe(self) -> Any:
        F = self.spark_functions
        papers_path = self.data_root / "papers"
        paper_paths = dataset_json_paths(papers_path)
        if not paper_paths:
            raise FileNotFoundError(f"No paper JSON shards found in {papers_path}")

        papers = self.spark_session.read.json([str(path) for path in paper_paths])
        papers = self.rename_dataframe_columns(papers, PAPER_COLUMN_RENAMES)

        if "s2FieldsOfStudy" in papers.columns and "fieldsOfStudy" not in papers.columns:
            papers = papers.withColumn(
                "fieldsOfStudy",
                F.expr("array_distinct(transform(s2FieldsOfStudy, item -> item.category))"),
            )

        self.paper_ids_dataframe = self.load_paper_ids_dataframe()
        papers = add_canonical_paper_ids(papers, self.paper_ids_dataframe, F)
        canonical_ids = papers.select(
            "corpusId",
            F.col("paperId").alias("sha"),
            F.lit(True).alias("primary"),
        )
        self.paper_ids_dataframe = (
            self.paper_ids_dataframe.unionByName(canonical_ids)
            .dropDuplicates(["corpusId", "sha"])
        )
        return papers

    def load_paper_ids_dataframe(self) -> Any:
        F = self.spark_functions
        paper_ids_path = self.data_root / "paper-ids"
        paper_id_paths = dataset_json_paths(paper_ids_path)
        if not paper_id_paths:
            raise FileNotFoundError(
                "The paper-ids dataset is required for API-compatible paperId "
                f"values: {paper_ids_path}"
            )

        paper_ids = self.spark_session.read.json([str(path) for path in paper_id_paths])
        paper_ids = self.rename_dataframe_columns(paper_ids, {"corpusid": "corpusId"})
        required_columns = {"corpusId", "sha", "primary"}
        missing_columns = required_columns - set(paper_ids.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"The paper-ids dataset is missing required columns: {missing}")
        return (
            paper_ids.select(
                "corpusId",
                F.lower(F.trim("sha")).alias("sha"),
                F.col("primary").cast("boolean").alias("primary"),
            )
            .filter(
                (F.col("corpusId") > 0)
                & F.col("sha").rlike(HEX_SHA_RE.pattern)
            )
            .dropDuplicates(["corpusId", "sha"])
        )

    def read_raw_dataset(self, dataset_name: str, renames: dict[str, str] | None = None) -> Any | None:
        if dataset_name in self.raw_dataframes:
            return self.raw_dataframes[dataset_name]
        dataset_path = self.data_root / dataset_name
        paths = dataset_json_paths(dataset_path)
        if not paths:
            return None
        dataframe = self.spark_session.read.json([str(path) for path in paths])
        if renames:
            dataframe = self.rename_dataframe_columns(dataframe, renames)
        self.raw_dataframes[dataset_name] = dataframe
        return dataframe

    def read_authors_dataframe(self) -> Any | None:
        if self.use_index and self.index_table_exists("authors"):
            return self.read_index_table("authors")
        return self.read_raw_dataset("authors", AUTHOR_COLUMN_RENAMES)

    def read_citations_dataframe(self) -> Any | None:
        return self.read_raw_dataset("citations", CITATION_COLUMN_RENAMES)

    def read_s2orc_text_dataframe(self) -> Any | None:
        if self.use_index and self.index_table_exists("s2orc_text"):
            return self.read_index_table("s2orc_text")
        dataframe = self.read_raw_dataset("s2orc_v2", {"corpusid": "corpusId"})
        if dataframe is not None:
            return dataframe
        return self.read_raw_dataset("s2orc", {"corpusid": "corpusId"})

    def rename_dataframe_columns(self, dataframe: Any, renames: dict[str, str]) -> Any:
        for old_name, new_name in renames.items():
            if old_name in dataframe.columns and new_name not in dataframe.columns:
                dataframe = dataframe.withColumnRenamed(old_name, new_name)
        return dataframe

    def get_paper_by_id(
        self,
        paper_id: str,
        *,
        fields: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        if self.use_index:
            paper = self.get_indexed_papers_batch([paper_id], fields=fields)[0]
            if paper is None:
                raise response_not_found(f"Paper not found: {paper_id}")
            return paper

        normalized_id = normalize_paper_id(paper_id)
        row = self.filter_by_paper_id(self.papers_dataframe, normalized_id).limit(1).collect()
        if not row:
            raise response_not_found(f"Paper not found: {paper_id}")
        paper = canonical_paper_record(row[0].asDict(recursive=True))
        papers = {int(paper["corpusId"]): paper}
        self.enrich_paper_records(papers, fields)
        return select_paper_fields(paper, fields, _DEFAULT_PAPER_FIELDS)

    def get_papers_batch(
        self,
        paper_ids: list[str],
        *,
        fields: Optional[list[str]] = None,
    ) -> list[dict[str, Any] | None]:
        if self.use_index:
            return self.get_indexed_papers_batch(paper_ids, fields=fields)

        papers: list[dict[str, Any] | None] = []
        for paper_id in paper_ids:
            try:
                papers.append(self.get_paper_by_id(paper_id, fields=fields))
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    papers.append(None)
                else:
                    raise
        return papers

    def get_indexed_papers_batch(
        self,
        paper_ids: list[str],
        *,
        fields: Optional[list[str]] = None,
    ) -> list[dict[str, Any] | None]:
        resolved_corpus_ids = self.resolve_indexed_paper_ids(paper_ids)
        corpus_ids = [
            corpus_id
            for corpus_id in resolved_corpus_ids
            if corpus_id is not None
        ]
        papers_by_corpus_id = self.collect_indexed_papers_by_corpus_ids(
            corpus_ids,
            fields=fields,
        )
        return [
            papers_by_corpus_id.get(corpus_id) if corpus_id is not None else None
            for corpus_id in resolved_corpus_ids
        ]

    def resolve_indexed_paper_id(self, paper_id: str) -> int | None:
        normalized_id = normalize_paper_id(paper_id)
        upper_id = normalized_id.upper()

        if upper_id.startswith("CORPUSID:"):
            return int(normalized_id.split(":", 1)[1])
        if HEX_SHA_RE.fullmatch(normalized_id):
            return self.lookup_paper_id(normalized_id)
        if normalized_id.isdigit():
            return int(normalized_id)
        if upper_id.startswith("ARXIV:"):
            return self.lookup_external_id("ArXiv", normalized_id.split(":", 1)[1])

        prefix, separator, value = normalized_id.partition(":")
        external_id_key = EXTERNAL_ID_KEYS.get(prefix) or EXTERNAL_ID_KEYS.get(prefix.upper())
        if separator and external_id_key:
            return self.lookup_external_id(external_id_key, value)
        if normalized_id.startswith("10."):
            return self.lookup_external_id("DOI", normalized_id)
        return self.lookup_paper_id(normalized_id)

    def resolve_indexed_paper_ids(self, paper_ids: list[str]) -> list[int | None]:
        """Batch version of ``resolve_indexed_paper_id``, in input order.

        Plain paperIds (40-hex SHAs) are resolved together in one Spark job;
        any other ID form goes through ``resolve_indexed_paper_id``.
        """
        normalized_ids = [normalize_paper_id(paper_id) for paper_id in paper_ids]
        sha_ids = [
            normalized_id.lower()
            for normalized_id in normalized_ids
            if HEX_SHA_RE.fullmatch(normalized_id)
        ]
        corpus_ids_by_sha = self.lookup_paper_ids(sha_ids)
        return [
            corpus_ids_by_sha.get(normalized_id.lower())
            if HEX_SHA_RE.fullmatch(normalized_id)
            else self.resolve_indexed_paper_id(paper_id)
            for paper_id, normalized_id in zip(paper_ids, normalized_ids)
        ]

    def lookup_paper_ids(self, paper_ids: list[str]) -> dict[str, int]:
        """Map lowercase paperIds to corpusIds with one Spark job.

        When a paperId maps to several corpusIds, the smallest one wins.
        """
        if not paper_ids or not self.index_table_exists("paper_ids"):
            return {}
        rows = (
            self.bucketed_id_filter(
                self.read_index_table("paper_ids"),
                id_column="paperId",
                bucket_column="paperIdBucket",
                ids=paper_ids,
            )
            .select("paperId", "corpusId")
            .collect()
        )
        corpus_ids: dict[str, int] = {}
        for row in rows:
            corpus_id = int(row["corpusId"])
            corpus_ids[row["paperId"]] = min(corpus_ids.get(row["paperId"], corpus_id), corpus_id)
        return corpus_ids

    def lookup_paper_id(self, paper_id: str) -> int | None:
        if not self.index_table_exists("paper_ids"):
            return None
        F = self.spark_functions
        normalized = paper_id.lower()
        row = (
            self.read_index_table("paper_ids")
            .filter(
                (F.col("paperIdBucket") == stable_bucket(normalized, self.num_buckets))
                & (F.col("paperId") == normalized)
            )
            .select("corpusId")
            .limit(1)
            .collect()
        )
        return int(row[0]["corpusId"]) if row else None

    def lookup_external_id(self, external_id_type: str, external_id_value: str) -> int | None:
        if not self.index_table_exists("external_ids"):
            return None
        F = self.spark_functions
        row = (
            self.read_index_table("external_ids")
            .filter(
                (F.col("externalIdType") == external_id_type)
                & (
                    F.col("externalIdBucket")
                    == stable_bucket(external_id_value, self.num_buckets)
                )
                & (F.col("externalIdValue") == external_id_value)
            )
            .select("corpusId")
            .limit(1)
            .collect()
        )
        return int(row[0]["corpusId"]) if row else None

    def indexed_papers_dataframe_by_corpus_ids(self, corpus_ids: list[int]) -> Any:
        return self.corpus_bucket_filter(self.papers_dataframe, corpus_ids)

    def bucketed_id_filter(
        self,
        dataframe: Any,
        *,
        id_column: str,
        bucket_column: str,
        ids: Iterable[Any],
    ) -> Any:
        """Keep rows whose ``id_column`` is one of ``ids`` (IDs must not contain commas).

        When the table has ``bucket_column``, only the index partitions that can
        hold those IDs are read. The IDs are matched with a broadcast semi-join,
        not ``isin``: ``isin`` makes one Py4J call per ID and Spark evaluates a
        huge IN list slowly (~30 s for 150k IDs). Here all IDs reach the JVM as
        one string (see ``strings_dataframe``).
        """
        F = self.spark_functions
        unique_ids = sorted({str(value) for value in ids})
        if not unique_ids:
            return dataframe.limit(0)
        if bucket_column in dataframe.columns:
            buckets = sorted({stable_bucket(value, self.num_buckets) for value in unique_ids})
            # A SQL string builds the same partition filter with one Py4J call.
            dataframe = dataframe.filter(
                f"{bucket_column} IN ({', '.join(map(str, buckets))})"
            )
        id_type = dataframe.schema[id_column].dataType
        ids_dataframe = self.strings_dataframe(unique_ids, id_column).select(
            F.col(id_column).cast(id_type).alias(id_column)
        )
        return dataframe.join(F.broadcast(ids_dataframe), on=id_column, how="left_semi")

    def strings_dataframe(self, values: list[str], column: str) -> Any:
        """One-column DataFrame of ``values`` (which must not contain commas).

        Built in the JVM from a single joined string: one Py4J call, and no
        ``createDataFrame``, which would need Python workers.
        """
        F = self.spark_functions
        return self.spark_session.range(1).select(
            F.explode(F.split(F.lit(",".join(values)), ",")).alias(column)
        )

    def collect_indexed_papers_by_corpus_ids(
        self,
        corpus_ids: list[int],
        *,
        fields: list[str] | str | None = None,
        join_abstracts: bool = False,
    ) -> dict[int, dict[str, Any]]:
        """Read unique papers, reusing bounded, field-specific cached metadata."""
        unique_ids = list(dict.fromkeys(int(corpus_id) for corpus_id in corpus_ids))
        if not unique_ids:
            return {}
        read_options = {"join_abstracts": True} if join_abstracts else {}
        if not self.paper_cache_max_bytes:
            return self.read_indexed_papers_by_corpus_ids(
                unique_ids, fields=fields, **read_options
            )

        field_key = tuple(
            sorted(requested_top_level_fields(fields, _DEFAULT_PAPER_FIELDS))
        )
        papers: dict[int, dict[str, Any]] = {}
        missing_ids = []
        for corpus_id in unique_ids:
            key = (field_key, corpus_id)
            cached = self.paper_cache.get(key)
            if cached is None:
                missing_ids.append(corpus_id)
            else:
                self.paper_cache.move_to_end(key)
                papers[corpus_id] = orjson.loads(cached)

        if missing_ids:
            fetched = self.read_indexed_papers_by_corpus_ids(
                missing_ids, fields=fields, **read_options
            )
            papers.update(fetched)
            for corpus_id, paper in fetched.items():
                payload = orjson.dumps(paper)
                if len(payload) > self.paper_cache_max_bytes:
                    continue
                while (
                    self.paper_cache_size_bytes + len(payload)
                    > self.paper_cache_max_bytes
                ):
                    _, evicted = self.paper_cache.popitem(last=False)
                    self.paper_cache_size_bytes -= len(evicted)
                self.paper_cache[(field_key, corpus_id)] = payload
                self.paper_cache_size_bytes += len(payload)
        return papers

    def read_indexed_papers_by_corpus_ids(
        self,
        corpus_ids: list[int],
        *,
        fields: list[str] | str | None = None,
        join_abstracts: bool = False,
    ) -> dict[int, dict[str, Any]]:
        dataframe = self.indexed_papers_dataframe_by_corpus_ids(corpus_ids)
        requested_fields = requested_top_level_fields(fields, _DEFAULT_PAPER_FIELDS)
        required_columns = {"corpusId", "paperId", *requested_fields}
        if "publicationVenue" in requested_fields:
            required_columns.add("publicationVenueId")
        dataframe = dataframe.select(
            *[
                column_name
                for column_name in dataframe.columns
                if column_name in required_columns
            ]
        )
        joined_abstracts = (
            join_abstracts
            and bool(requested_fields & {"abstract", "openAccessPdf"})
            and "abstracts" in (self.index_metadata or {}).get("tables", [])
            and (self.index_table_path("abstracts") / "_SUCCESS").is_file()
        )
        if joined_abstracts:
            F = self.spark_functions
            abstracts = self.corpus_bucket_filter(
                self.read_index_table("abstracts"), corpus_ids
            )
            columns = [
                column
                for column in abstracts.columns
                if column in {"abstract", "openAccessPdf", "openaccessinfo"}
                and (column != "abstract" or "abstract" in requested_fields)
            ]
            abstracts = abstracts.select(
                "corpusId", F.struct(*columns).alias("abstractRecord")
            )
            dataframe = dataframe.join(abstracts, "corpusId", "left")
        rows = dataframe.collect()
        papers: dict[int, dict[str, Any]] = {}
        for row in rows:
            data = row.asDict(recursive=True)
            paper = canonical_paper_record(data)
            if joined_abstracts and data.get("abstractRecord"):
                self.merge_abstract_record(paper, data["abstractRecord"])
            papers[int(paper["corpusId"])] = paper
        enrichment_fields = (
            list(requested_fields - {"abstract", "openAccessPdf"})
            if join_abstracts
            else fields
        )
        # The joined path uses available completed indexes only. Do not fall
        # back to scanning raw abstracts when optional enrichment is absent.
        self.enrich_paper_records(papers, enrichment_fields)
        return {
            corpus_id: select_paper_fields(paper, fields, _DEFAULT_PAPER_FIELDS)
            for corpus_id, paper in papers.items()
        }

    def collect_papers_by_corpus_ids(
        self,
        corpus_ids: list[int],
        *,
        fields: Optional[list[str] | str] = None,
    ) -> dict[int, dict[str, Any]]:
        if self.use_index:
            return self.collect_indexed_papers_by_corpus_ids(corpus_ids, fields=fields)

        F = self.spark_functions
        unique_ids = sorted({int(corpus_id) for corpus_id in corpus_ids})
        if not unique_ids:
            return {}
        rows = self.papers_dataframe.filter(F.col("corpusId").isin(unique_ids)).collect()
        papers = {
            int(row["corpusId"]): canonical_paper_record(row.asDict(recursive=True))
            for row in rows
        }
        self.enrich_paper_records(papers, fields)
        return {
            corpus_id: select_paper_fields(paper, fields, _DEFAULT_PAPER_FIELDS)
            for corpus_id, paper in papers.items()
        }

    def enrich_paper_records(
        self,
        papers: dict[int, dict[str, Any]],
        fields: Optional[list[str] | str],
    ) -> None:
        if not papers:
            return
        requested = requested_top_level_fields(fields, _DEFAULT_PAPER_FIELDS)
        corpus_ids = list(papers)
        if requested & {"abstract", "openAccessPdf"}:
            self.add_abstracts(papers, corpus_ids, include_abstract="abstract" in requested)
        if "tldr" in requested:
            self.add_tldrs(papers, corpus_ids)
        if "embedding" in requested:
            self.add_embeddings(papers, corpus_ids)
        if "publicationVenue" in requested:
            self.add_publication_venues(papers)

    def corpus_bucket_filter(self, dataframe: Any, corpus_ids: list[int]) -> Any:
        return self.bucketed_id_filter(
            dataframe,
            id_column="corpusId",
            bucket_column="corpusIdBucket",
            ids=(int(corpus_id) for corpus_id in corpus_ids),
        )

    def add_abstracts(
        self,
        papers: dict[int, dict[str, Any]],
        corpus_ids: list[int],
        *,
        include_abstract: bool = True,
    ) -> None:
        dataframe = None
        if self.use_index and self.index_table_exists("abstracts"):
            dataframe = self.read_index_table("abstracts")
        else:
            dataframe = self.read_raw_dataset("abstracts", {"corpusid": "corpusId"})
        if dataframe is None:
            return
        # The abstract text is by far the widest column; read it only when requested.
        columns = {"corpusId", "openAccessPdf", "openaccessinfo"}
        if include_abstract:
            columns.add("abstract")
        dataframe = self.corpus_bucket_filter(dataframe, corpus_ids)
        dataframe = dataframe.select(*[column for column in dataframe.columns if column in columns])
        for row in dataframe.collect():
            data = row.asDict(recursive=True)
            paper = papers.get(int(data["corpusId"]))
            if paper is None:
                continue
            self.merge_abstract_record(paper, data)

    def merge_abstract_record(
        self, paper: dict[str, Any], data: dict[str, Any]
    ) -> None:
        """Apply the same abstract/PDF precedence for separate reads and joins."""
        if data.get("abstract") is not None:
            paper["abstract"] = data.get("abstract")
        if data.get("openAccessPdf") is not None and paper.get("openAccessPdf") is None:
            paper["openAccessPdf"] = data.get("openAccessPdf")
        if (
            data.get("openaccessinfo") is not None
            and paper.get("openAccessPdf") is None
        ):
            info = data["openaccessinfo"]
            paper["openAccessPdf"] = {
                "url": info.get("url"),
                "status": info.get("status"),
                "license": info.get("license"),
                "disclaimer": info.get("disclaimer"),
            }

    def add_tldrs(self, papers: dict[int, dict[str, Any]], corpus_ids: list[int]) -> None:
        dataframe = None
        if self.use_index and self.index_table_exists("tldrs"):
            dataframe = self.read_index_table("tldrs")
        else:
            dataframe = self.read_raw_dataset("tldrs", {"corpusid": "corpusId"})
        if dataframe is None:
            return
        for row in self.corpus_bucket_filter(dataframe, corpus_ids).collect():
            data = row.asDict(recursive=True)
            paper = papers.get(int(data["corpusId"]))
            if paper is not None:
                paper["tldr"] = {"model": data.get("model"), "text": data.get("text")}

    def add_embeddings(self, papers: dict[int, dict[str, Any]], corpus_ids: list[int]) -> None:
        dataframe = None
        if self.use_index and self.index_table_exists("embeddings"):
            dataframe = self.read_index_table("embeddings")
        else:
            dataframe = self.read_raw_dataset("embeddings-specter_v2", {"corpusid": "corpusId"})
        if dataframe is None:
            return
        for row in self.corpus_bucket_filter(dataframe, corpus_ids).collect():
            data = row.asDict(recursive=True)
            paper = papers.get(int(data["corpusId"]))
            if paper is not None:
                vector = data.get("vector")
                if isinstance(vector, str):
                    try:
                        vector = orjson.loads(vector)
                    except orjson.JSONDecodeError:
                        pass
                paper["embedding"] = {"model": data.get("model"), "vector": vector}

    def add_publication_venues(self, papers: dict[int, dict[str, Any]]) -> None:
        venue_ids = sorted({
            str(paper.get("publicationVenueId"))
            for paper in papers.values()
            if paper.get("publicationVenueId")
        })
        if not venue_ids:
            return
        dataframe = None
        if self.use_index and self.index_table_exists("publication_venues"):
            dataframe = self.read_index_table("publication_venues")
        else:
            dataframe = self.read_raw_dataset("publication-venues")
        if dataframe is None:
            return
        F = self.spark_functions
        if "publicationVenueIdBucket" in dataframe.columns:
            buckets = sorted({stable_bucket(venue_id, self.num_buckets) for venue_id in venue_ids})
            dataframe = dataframe.filter(F.col("publicationVenueIdBucket").isin(buckets))
        rows = dataframe.filter(F.col("id").isin(venue_ids)).collect()
        venues = {row["id"]: row.asDict(recursive=True) for row in rows}
        for paper in papers.values():
            venue_id = paper.get("publicationVenueId")
            if venue_id in venues:
                venue = dict(venues[venue_id])
                venue.pop("publicationVenueIdBucket", None)
                paper["publicationVenue"] = venue

    def author_bucket_filter(self, dataframe: Any, author_ids: list[str]) -> Any:
        F = self.spark_functions
        ids = sorted({str(author_id) for author_id in author_ids})
        if not ids:
            return dataframe.limit(0)
        if "authorIdBucket" in dataframe.columns:
            buckets = sorted({stable_bucket(author_id, self.num_buckets) for author_id in ids})
            dataframe = dataframe.filter(F.col("authorIdBucket").isin(buckets))
        return dataframe.filter(F.col("authorId").isin(ids))

    def collect_authors_by_ids(
        self,
        author_ids: list[str],
        *,
        fields: Optional[list[str] | str] = None,
    ) -> dict[str, dict[str, Any]]:
        dataframe = self.read_authors_dataframe()
        if dataframe is None:
            return {}
        rows = self.author_bucket_filter(dataframe, author_ids).collect()
        authors = {
            str(row["authorId"]): canonical_author_record(row.asDict(recursive=True))
            for row in rows
        }
        self.enrich_author_records(authors, fields)
        return {
            author_id: select_author_fields(author, fields)
            for author_id, author in authors.items()
        }

    def enrich_author_records(
        self,
        authors: dict[str, dict[str, Any]],
        fields: Optional[list[str] | str],
    ) -> None:
        requested = requested_top_level_fields(fields, _DEFAULT_AUTHOR_FIELDS)
        if "papers" not in requested:
            return
        paper_fields = author_requested_paper_fields(fields)
        for author_id, author in authors.items():
            author["papers"] = self.get_all_papers_of_author(author_id, fields=paper_fields)

    def get_authors_batch(
        self,
        author_ids: list[str],
        *,
        fields: Optional[list[str] | str] = None,
    ) -> list[dict[str, Any] | None]:
        authors_by_id = self.collect_authors_by_ids(author_ids, fields=fields)
        return [authors_by_id.get(str(author_id)) for author_id in author_ids]

    def get_author(
        self,
        author_id: str,
        *,
        fields: Optional[list[str] | str] = None,
    ) -> dict[str, Any]:
        author = self.get_authors_batch([author_id], fields=fields)[0]
        if author is None:
            raise response_not_found(f"Author not found: {author_id}")
        return author

    def search_authors(
        self,
        query: str,
        *,
        fields: Optional[list[str] | str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        dataframe = self.author_search_dataframe(query)
        total = dataframe.count()
        rows = (
            dataframe.orderBy(self.spark_functions.desc("paperCount"), self.spark_functions.asc("name"))
            .limit(offset + limit)
            .collect()[offset:]
        )
        authors_by_id = {
            str(row["authorId"]): canonical_author_record(row.asDict(recursive=True))
            for row in rows
        }
        self.enrich_author_records(authors_by_id, fields)
        data = [select_author_fields(author, fields) for author in authors_by_id.values()]
        result: dict[str, Any] = {"total": total, "offset": offset, "data": data}
        if offset + limit < total:
            result["next"] = offset + limit
        return result

    def author_search_dataframe(self, query: str) -> Any:
        F = self.spark_functions
        terms = query_terms_for_index(query)
        authors = self.read_authors_dataframe()
        if authors is None:
            return self.spark_session.createDataFrame([], "authorId string, name string")
        if self.use_index and self.index_table_exists("author_name_tokens") and terms:
            unique_terms = sorted(set(terms))
            buckets = sorted({stable_bucket(term, self.num_buckets) for term in unique_terms})
            token_matches = (
                self.read_index_table("author_name_tokens")
                .filter(
                    F.col("authorNameTokenBucket").isin(buckets)
                    & F.col("authorNameToken").isin(unique_terms)
                )
            )
            grouped = token_matches.groupBy("authorId").agg(
                F.countDistinct("authorNameToken").alias("matchedNameTokens")
            )
            if len(unique_terms) > 1:
                grouped = grouped.filter(F.col("matchedNameTokens") == len(unique_terms))
            author_ids = [row["authorId"] for row in grouped.select("authorId").limit(self.search_candidate_limit).collect()]
            return self.author_bucket_filter(authors, author_ids)

        name = F.lower(F.coalesce(F.col("name"), F.lit("")))
        for term in query_terms(query):
            authors = authors.filter(name.contains(term))
        return authors

    def autocomplete_paper(self, query: str) -> dict[str, list[dict[str, Any]]]:
        dataframe = self.filter_by_query(self.papers_dataframe, query)
        rows = self.order_for_search(dataframe, query).limit(10).collect()
        matches = []
        for row in rows:
            paper = canonical_paper_record(row.asDict(recursive=True))
            matches.append(
                {
                    "id": paper.get("paperId"),
                    "title": paper.get("title"),
                    "authorsYear": author_year(paper.get("authors"), paper.get("year")),
                }
            )
        return {"matches": matches}

    def search_papers(
        self,
        query: str,
        *,
        fields: Optional[list[str] | str] = None,
        publication_types: Optional[list[str] | str] = None,
        open_access_pdf: bool = False,
        min_citation_count: Optional[int | str] = None,
        publication_date_or_year: Optional[str] = None,
        year: Optional[str] = None,
        venue: Optional[list[str] | str] = None,
        fields_of_study: Optional[list[str] | str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        dataframe = self.filtered_search_dataframe(
            query=query,
            publication_types=publication_types,
            open_access_pdf=open_access_pdf,
            min_citation_count=min_citation_count,
            publication_date_or_year=publication_date_or_year,
            year=year,
            venue=venue,
            fields_of_study=fields_of_study,
        )
        total = dataframe.count()
        rows = self.order_for_search(dataframe, query).limit(offset + limit).collect()[offset:]
        papers = {
            int(row["corpusId"]): canonical_paper_record(row.asDict(recursive=True))
            for row in rows
        }
        self.enrich_paper_records(papers, fields)
        data = [
            select_paper_fields(papers[int(row["corpusId"])], fields, ["paperId", "title"])
            for row in rows
        ]
        result: dict[str, Any] = {"total": total, "offset": offset, "data": data}
        if offset + limit < total:
            result["next"] = offset + limit
        return result

    def search_papers_bulk(
        self,
        query: str,
        *,
        token: Optional[str] = None,
        fields: Optional[list[str] | str] = None,
        sort: Optional[str] = None,
        publication_types: Optional[list[str] | str] = None,
        open_access_pdf: bool = False,
        min_citation_count: Optional[int | str] = None,
        publication_date_or_year: Optional[str] = None,
        year: Optional[str] = None,
        venue: Optional[list[str] | str] = None,
        fields_of_study: Optional[list[str] | str] = None,
    ) -> dict[str, Any]:
        offset = int(token) if token else 0
        dataframe = self.filtered_search_dataframe(
            query=query,
            publication_types=publication_types,
            open_access_pdf=open_access_pdf,
            min_citation_count=min_citation_count,
            publication_date_or_year=publication_date_or_year,
            year=year,
            venue=venue,
            fields_of_study=fields_of_study,
        )
        total = dataframe.count()
        ordered = self.order_for_bulk(dataframe, query, sort)
        rows = ordered.limit(offset + self.bulk_page_size).collect()[offset:]
        papers = {
            int(row["corpusId"]): canonical_paper_record(row.asDict(recursive=True))
            for row in rows
        }
        self.enrich_paper_records(papers, fields)
        data = [
            select_paper_fields(papers[int(row["corpusId"])], fields, ["paperId", "title"])
            for row in rows
        ]
        result: dict[str, Any] = {"total": total, "data": data}
        next_offset = offset + self.bulk_page_size
        if next_offset < total:
            result["token"] = str(next_offset)
        return result

    def match_paper_title(
        self,
        query: str,
        *,
        fields: Optional[list[str] | str] = None,
        publication_types: Optional[list[str] | str] = None,
        open_access_pdf: bool = False,
        min_citation_count: Optional[int | str] = None,
        publication_date_or_year: Optional[str] = None,
        year: Optional[str] = None,
        venue: Optional[list[str] | str] = None,
        fields_of_study: Optional[list[str] | str] = None,
        return_none_on_404: bool = False,
    ) -> Optional[dict[str, list[dict[str, Any]]]]:
        if self.use_index:
            exact_title_dataframe = self.lookup_normalized_title_dataframe(query)
            if exact_title_dataframe is not None:
                exact_title_dataframe = self.apply_publication_types_filter(
                    exact_title_dataframe,
                    publication_types,
                )
                exact_title_dataframe = self.apply_open_access_filter(
                    exact_title_dataframe,
                    open_access_pdf,
                )
                exact_title_dataframe = self.apply_min_citation_filter(
                    exact_title_dataframe,
                    min_citation_count,
                )
                exact_title_dataframe = self.apply_publication_date_filter(
                    exact_title_dataframe,
                    publication_date_or_year,
                )
                exact_title_dataframe = self.apply_year_filter(exact_title_dataframe, year)
                exact_title_dataframe = self.apply_venue_filter(exact_title_dataframe, venue)
                exact_title_dataframe = self.apply_fields_of_study_filter(
                    exact_title_dataframe,
                    fields_of_study,
                )
                rows = exact_title_dataframe.orderBy(
                    self.spark_functions.desc("citationCount"),
                    self.spark_functions.asc("corpusId"),
                ).limit(1).collect()
                if rows:
                    paper = canonical_paper_record(rows[0].asDict(recursive=True))
                    papers = {int(paper["corpusId"]): paper}
                    self.enrich_paper_records(papers, fields)
                    selected = select_paper_fields(paper, fields, _DEFAULT_PAPER_FIELDS)
                    selected["matchScore"] = 100.0
                    return {"data": [selected]}

        dataframe = self.filtered_search_dataframe(
            query=query,
            publication_types=publication_types,
            open_access_pdf=open_access_pdf,
            min_citation_count=min_citation_count,
            publication_date_or_year=publication_date_or_year,
            year=year,
            venue=venue,
            fields_of_study=fields_of_study,
        )
        scored = self.add_title_match_score(dataframe, query)
        rows = scored.orderBy(
            self.spark_functions.desc("matchScore"),
            self.spark_functions.desc("citationCount"),
            self.spark_functions.asc("corpusId"),
        ).limit(1).collect()
        if not rows:
            if return_none_on_404:
                return None
            raise response_not_found(f"No title match found: {query}")

        paper = canonical_paper_record(rows[0].asDict(recursive=True))
        papers = {int(paper["corpusId"]): paper}
        self.enrich_paper_records(papers, fields)
        selected = select_paper_fields(paper, fields, _DEFAULT_PAPER_FIELDS)
        selected["matchScore"] = paper.get("matchScore")
        return {"data": [selected]}

    def lookup_normalized_title_dataframe(self, query: str) -> Any | None:
        if not self.index_table_exists("title_index"):
            return None
        F = self.spark_functions
        normalized_title = normalize_title_text(query)
        if not normalized_title:
            return None
        rows = (
            self.read_index_table("title_index")
            .filter(
                (F.col("normalizedTitleBucket") == stable_bucket(normalized_title, self.num_buckets))
                & (F.col("normalizedTitle") == normalized_title)
            )
            .select("corpusId")
            .orderBy(F.asc("corpusId"))
            .limit(self.search_candidate_limit)
            .collect()
        )
        corpus_ids = [int(row["corpusId"]) for row in rows]
        if not corpus_ids:
            return None
        return self.indexed_papers_dataframe_by_corpus_ids(corpus_ids)

    def search_by_title(
        self,
        title: str,
        *,
        fields: Optional[list[str]] = None,
        publication_types: Optional[list[str]] = None,
        open_access_pdf: bool = False,
        min_citation_count: Optional[int] = None,
        publication_date_or_year: Optional[str] = None,
        year: Optional[str] = None,
        venue: Optional[list[str]] = None,
        fields_of_study: Optional[list[str]] = None,
    ) -> Optional[dict[str, Any]]:
        data = self.match_paper_title(
            title,
            fields=fields or _DEFAULT_PAPER_FIELDS,
            publication_types=publication_types,
            open_access_pdf=open_access_pdf,
            min_citation_count=min_citation_count,
            publication_date_or_year=publication_date_or_year,
            year=year,
            venue=venue,
            fields_of_study=fields_of_study,
            return_none_on_404=True,
        )
        if data is None or not data.get("data"):
            return None
        return data["data"][0]

    def search_by_titles_batch(
        self,
        titles: list[str],
        *,
        fields: Optional[list[str]] = None,
    ) -> list[Optional[dict[str, Any]]]:
        """Batch version of ``search_by_title`` (without filters), in input order.

        Matches like ``match_paper_title``: an exact normalized-title match
        wins (most-cited first); otherwise the title-token search keeps the
        papers that hold every query token and picks the best Levenshtein
        score (most-cited first). Remaining ties go to the smallest corpusId.
        On the index this is a fixed number of Spark jobs for all titles,
        instead of several jobs -- often a full papers scan -- per title.
        """
        if not self.use_index:
            return [self.search_by_title(title, fields=fields) for title in titles]

        matches = self.exact_title_matches(titles)
        matches.update(
            self.fuzzy_title_matches(
                {index: title for index, title in enumerate(titles) if index not in matches}
            )
        )
        papers_by_corpus_id = self.collect_indexed_papers_by_corpus_ids(
            [corpus_id for corpus_id, _ in matches.values()],
            fields=fields or _DEFAULT_PAPER_FIELDS,
        )
        results: list[Optional[dict[str, Any]]] = []
        for index in range(len(titles)):
            corpus_id, match_score = matches.get(index, (None, None))
            paper = papers_by_corpus_id.get(corpus_id) if corpus_id is not None else None
            results.append({**paper, "matchScore": match_score} if paper is not None else None)
        return results

    def exact_title_matches(self, titles: list[str]) -> dict[int, tuple[int, float]]:
        """Return ``{title index: (corpusId, 100.0)}`` for exact normalized-title matches."""
        normalized_titles = [normalize_title_text(title) for title in titles]
        wanted_titles = {title for title in normalized_titles if title}
        if not wanted_titles or not self.index_table_exists("title_index"):
            return {}
        exact_order = self.window.partitionBy("normalizedTitle").orderBy(
            self.spark_functions.asc("corpusId")
        )
        rows = (
            self.bucketed_id_filter(
                self.read_index_table("title_index"),
                id_column="normalizedTitle",
                bucket_column="normalizedTitleBucket",
                ids=wanted_titles,
            )
            .select("normalizedTitle", "corpusId")
            .withColumn("candidateRank", self.spark_functions.row_number().over(exact_order))
            .filter(self.spark_functions.col("candidateRank") <= self.search_candidate_limit)
            .select("normalizedTitle", "corpusId")
            .collect()
        )
        corpus_ids_by_title: dict[str, list[int]] = {}
        for row in rows:
            corpus_ids_by_title.setdefault(row["normalizedTitle"], []).append(int(row["corpusId"]))
        citation_counts = {
            corpus_id: paper.get("citationCount")
            for corpus_id, paper in self.collect_indexed_papers_by_corpus_ids(
                [corpus_id for corpus_ids in corpus_ids_by_title.values() for corpus_id in corpus_ids],
                fields=["citationCount"],
            ).items()
        }

        def citation_rank(corpus_id: int) -> tuple[bool, int, int]:
            # Spark's desc() puts nulls last; smallest corpusId breaks ties.
            count = citation_counts[corpus_id]
            return count is not None, count or 0, -corpus_id

        matches: dict[int, tuple[int, float]] = {}
        for index, normalized_title in enumerate(normalized_titles):
            candidates = [
                corpus_id
                for corpus_id in corpus_ids_by_title.get(normalized_title, [])
                if corpus_id in citation_counts
            ]
            if candidates:
                matches[index] = (max(candidates, key=citation_rank), 100.0)
        return matches

    def fuzzy_title_matches(self, titles_by_index: dict[int, str]) -> dict[int, tuple[int, float]]:
        """Return ``{title index: (corpusId, matchScore)}`` from the title-token search.

        Mirrors ``indexed_search_corpus_ids`` + ``add_title_match_score`` for
        every title at once: the token scan, the per-title candidate limit and
        the scoring each run once for the whole batch.
        """
        queries = {
            index: (sorted(set(query_terms_for_index(title))), " ".join(query_terms(title)))
            for index, title in titles_by_index.items()
        }
        queries = {index: query for index, query in queries.items() if query[0]}
        if not queries or not self.index_table_exists("title_tokens"):
            return {}

        F = self.spark_functions
        # Tokens and normalized titles are [a-z0-9] words joined by spaces, so
        # ":" and "," are safe separators.
        query_terms_dataframe = self.strings_dataframe(
            [f"{index}:{term}" for index, (terms, _) in queries.items() for term in terms],
            "queryTerm",
        ).select(
            F.split("queryTerm", ":")[0].cast("int").alias("queryIndex"),
            F.split("queryTerm", ":")[1].alias("titleToken"),
        )
        query_dataframe = self.strings_dataframe(
            [f"{index}:{len(terms)}:{text}" for index, (terms, text) in queries.items()],
            "query",
        ).select(
            F.split("query", ":")[0].cast("int").alias("queryIndex"),
            F.split("query", ":")[1].cast("int").alias("termCount"),
            F.split("query", ":")[2].alias("normalizedQuery"),
        )

        # A candidate must hold every query token; keep the first
        # search_candidate_limit per title by corpusId, like the per-title search.
        candidate_order = self.window.partitionBy("queryIndex").orderBy(F.asc("corpusId"))
        candidates = (
            self.bucketed_id_filter(
                self.read_index_table("title_tokens"),
                id_column="titleToken",
                bucket_column="titleTokenBucket",
                ids=[term for terms, _ in queries.values() for term in terms],
            )
            .join(F.broadcast(query_terms_dataframe), on="titleToken")
            .groupBy("queryIndex", "corpusId")
            .agg(F.countDistinct("titleToken").alias("matchedTitleTokens"))
            .join(F.broadcast(query_dataframe.select("queryIndex", "termCount")), on="queryIndex")
            .filter(F.col("matchedTitleTokens") == F.col("termCount"))
            .withColumn("candidateRank", F.row_number().over(candidate_order))
            .filter(F.col("candidateRank") <= self.search_candidate_limit)
            .select("queryIndex", "corpusId")
        )

        title = F.lower(F.coalesce(F.col("title"), F.lit("")))
        normalized_query = F.col("normalizedQuery")
        max_length = F.greatest(F.length(title), F.length(normalized_query))
        distance = F.levenshtein(title, normalized_query)
        score = F.round(
            F.greatest(F.lit(0.0), F.lit(100.0) - (distance * 100.0 / max_length)),
            2,
        )
        best_order = self.window.partitionBy("queryIndex").orderBy(
            F.desc("matchScore"),
            F.desc("citationCount"),
            F.asc("corpusId"),
        )
        # Reuse the token intersection for bucket discovery and scoring. Only
        # bucket IDs (at most num_buckets) cross into Python, not all candidates.
        candidates = candidates.persist()
        try:
            papers = self.papers_dataframe
            if "corpusIdBucket" in papers.columns:
                buckets = [
                    int(row["bucket"])
                    for row in candidates.select(
                        F.pmod(
                            F.crc32(F.col("corpusId").cast("string")),
                            F.lit(self.num_buckets),
                        ).alias("bucket")
                    )
                    .distinct()
                    .collect()
                ]
                if not buckets:
                    return {}
                papers = papers.filter(
                    f"corpusIdBucket IN ({', '.join(map(str, sorted(buckets)))})"
                )
            rows = (
                papers.select("corpusId", "title", "citationCount")
                # Candidate sets can be large; let Spark choose the join strategy.
                .join(candidates, on="corpusId")
                .join(
                    F.broadcast(
                        query_dataframe.select("queryIndex", "normalizedQuery")
                    ),
                    on="queryIndex",
                )
                .withColumn(
                    "matchScore",
                    F.when(title == normalized_query, F.lit(100.0)).otherwise(score),
                )
                .withColumn("bestRank", F.row_number().over(best_order))
                .filter(F.col("bestRank") == 1)
                .select("queryIndex", "corpusId", "matchScore")
                .collect()
            )
            return {
                int(row["queryIndex"]): (int(row["corpusId"]), float(row["matchScore"]))
                for row in rows
            }
        finally:
            candidates.unpersist()

    def get_paper_authors(
        self,
        paper_id: str,
        *,
        fields: Optional[list[str]] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> dict[str, Any]:
        paper = self.get_paper_by_id(paper_id, fields=["paperId", "authors"])
        authors = paper.get("authors") or []
        page_authors = [
            dict(author)
            for author in authors[offset : offset + limit]
            if isinstance(author, dict)
        ]
        requested = requested_top_level_fields(fields, _DEFAULT_AUTHOR_FIELDS)
        if requested - {"authorId", "name"}:
            author_ids = [
                str(author["authorId"])
                for author in page_authors
                if author.get("authorId")
            ]
            enriched = self.collect_authors_by_ids(author_ids, fields=fields)
            for author in page_authors:
                if author.get("authorId") in enriched:
                    author.update(enriched[author["authorId"]])
        page = [select_author_fields(author, fields) for author in page_authors]
        result: dict[str, Any] = {"offset": offset, "data": page}
        if offset + limit < len(authors):
            result["next"] = offset + limit
        return result

    def get_all_authors_of_paper(
        self,
        paper_id: str,
        *,
        fields: Optional[list[str]] = None,
        max_results: int = 1000,
    ) -> list[dict[str, Any]]:
        authors: list[dict[str, Any]] = []
        offset = 0
        page_size = min(100, max_results)
        while len(authors) < max_results:
            page = self.get_paper_authors(
                paper_id,
                fields=fields,
                limit=page_size,
                offset=offset,
            )
            batch = page.get("data", [])
            authors.extend(batch)
            next_offset = page.get("next")
            if not batch or next_offset is None:
                break
            offset = next_offset
        return authors[:max_results]

    def get_author_papers(
        self,
        author_id: str,
        *,
        fields: Optional[list[str] | str] = None,
        publication_date_or_year: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        if self.use_index and self.index_table_exists("author_papers"):
            return self.get_indexed_author_papers(
                author_id,
                fields=fields,
                publication_date_or_year=publication_date_or_year,
                offset=offset,
                limit=limit,
            )

        F = self.spark_functions
        escaped_author_id = author_id.replace("'", "\\'")
        dataframe = self.papers_dataframe.filter(
            F.expr(f"exists(authors, item -> item.authorId = '{escaped_author_id}')")
        )
        dataframe = self.apply_publication_date_filter(dataframe, publication_date_or_year)
        total = dataframe.count()
        rows = dataframe.orderBy(F.desc("year"), F.asc("title")).limit(offset + limit).collect()[offset:]
        data = [
            select_paper_fields(
                canonical_paper_record(row.asDict(recursive=True)),
                fields,
                _DEFAULT_PAPER_FIELDS,
            )
            for row in rows
        ]
        result: dict[str, Any] = {"offset": offset, "data": data}
        if offset + limit < total:
            result["next"] = offset + limit
        return result

    def get_indexed_author_papers(
        self,
        author_id: str,
        *,
        fields: Optional[list[str] | str] = None,
        publication_date_or_year: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        F = self.spark_functions
        dataframe = self.read_index_table("author_papers").filter(
            (F.col("authorIdBucket") == stable_bucket(author_id, self.num_buckets))
            & (F.col("authorId") == author_id)
        )
        dataframe = self.apply_publication_date_filter(dataframe, publication_date_or_year)
        total = dataframe.count()
        rows = (
            dataframe.orderBy(F.desc("year"), F.asc("title"))
            .limit(offset + limit)
            .collect()[offset:]
        )
        ordered_corpus_ids = [int(row["corpusId"]) for row in rows]
        papers_by_corpus_id = self.collect_indexed_papers_by_corpus_ids(
            ordered_corpus_ids,
            fields=fields,
        )
        result: dict[str, Any] = {
            "offset": offset,
            "data": [
                papers_by_corpus_id[corpus_id]
                for corpus_id in ordered_corpus_ids
                if corpus_id in papers_by_corpus_id
            ],
        }
        if offset + limit < total:
            result["next"] = offset + limit
        return result

    def get_all_papers_of_author(
        self,
        author_id: str,
        *,
        fields: Optional[list[str] | str] = None,
        publication_date_or_year: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        papers: list[dict[str, Any]] = []
        while True:
            batch = self.get_author_papers(
                author_id,
                fields=fields,
                publication_date_or_year=publication_date_or_year,
                offset=offset,
                limit=limit,
            )
            data = batch.get("data", [])
            papers.extend(data)
            next_offset = batch.get("next")
            if not data or next_offset is None:
                break
            offset = next_offset
        return papers

    def get_references(
        self,
        paper_id: str,
        *,
        fields: Optional[list[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        return self.get_citation_edges(
            paper_id,
            table_name="citations_by_citing",
            source_column="citingCorpusId",
            target_column="citedCorpusId",
            wrapper_field="citedPaper",
            fields=fields,
            default_fields=_DEFAULT_REF_FIELDS,
            limit=limit,
            offset=offset,
        )

    def get_all_references(
        self,
        paper_id: str,
        *,
        fields: Optional[list[str]] = None,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        offset = 0
        page_size = max_results
        while len(refs) < max_results:
            page = self.get_references(paper_id, fields=fields, limit=page_size, offset=offset)
            data = page.get("data", [])
            refs.extend(data)
            next_offset = page.get("next")
            if not data or next_offset is None:
                break
            offset = next_offset
        return refs[:max_results]

    def get_citations(
        self,
        paper_id: str,
        *,
        fields: Optional[list[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        return self.get_citation_edges(
            paper_id,
            table_name="citations_by_cited",
            source_column="citedCorpusId",
            target_column="citingCorpusId",
            wrapper_field="citingPaper",
            fields=fields,
            default_fields=_DEFAULT_CITE_FIELDS,
            limit=limit,
            offset=offset,
        )

    def get_all_citations(
        self,
        paper_id: str,
        *,
        fields: Optional[list[str]] = None,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> list[dict[str, Any]]:
        cites: list[dict[str, Any]] = []
        offset = 0
        page_size = max_results
        while len(cites) < max_results:
            page = self.get_citations(paper_id, fields=fields, limit=page_size, offset=offset)
            data = page.get("data", [])
            cites.extend(data)
            next_offset = page.get("next")
            if not data or next_offset is None:
                break
            offset = next_offset
        return cites[:max_results]

    def get_paper_with_citations_and_references(
        self,
        paper_id: str,
        *,
        fields: Optional[list[str]] = None,
        max_results: int = DEFAULT_MAX_RESULTS,
        include_citations: bool = True,
        include_references: bool = True,
        min_publication_date: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        """Return one paper and its selected local citation neighbors."""
        result = self.get_papers_with_citations_and_references_batch(
            [paper_id],
            fields=fields,
            max_results=max_results,
            include_citations=include_citations,
            include_references=include_references,
            min_publication_date=min_publication_date,
        )[paper_id]
        paper, citations, references = result
        if paper is None:
            raise response_not_found(f"Paper not found: {paper_id}")
        return paper, citations, references

    def get_papers_with_citations_and_references_batch(
        self,
        paper_ids: list[str],
        *,
        fields: Optional[list[str]] = None,
        max_results: int = DEFAULT_MAX_RESULTS,
        include_citations: bool = True,
        include_references: bool = True,
        min_publication_date: str | None = None,
    ) -> dict[
        str,
        tuple[
            dict[str, Any] | None,
            list[dict[str, Any]],
            list[dict[str, Any]],
        ],
    ]:
        """Return local paper expansions keyed by each original input ID.

        The whole batch costs a fixed number of Spark jobs, however many IDs
        it holds: resolve the IDs, read the edges of both requested sides in
        one pass, then fetch the papers and all their neighbors together.
        Each side keeps up to ``max_results`` neighbors per paper, ordered by
        ``citationId`` like ``get_all_citations``/``get_all_references``. Like
        the online client's batch method, each neighbor entry holds only
        ``citingPaper``/``citedPaper``, never the edge-level
        contexts/intents/isInfluential. ``min_publication_date`` drops
        neighbors older than that date on both sides before the
        ``max_results`` cap is applied -- see ``citation_edges``.
        """
        requested_fields = list(fields or _DEFAULT_PAPER_FIELDS)
        if include_citations and "citationCount" not in requested_fields:
            requested_fields.append("citationCount")
        if include_references and "referenceCount" not in requested_fields:
            requested_fields.append("referenceCount")
        sides = [
            side
            for side, included in (("citations", include_citations), ("references", include_references))
            if included
        ]
        neighbor_fields = {
            side: edge_paper_fields(fields, CITATION_SIDES[side][3], CITATION_SIDES[side][4])
            for side in sides
        }

        corpus_ids = self.resolve_paper_ids_to_corpus_ids(paper_ids)
        resolved_ids = sorted({corpus_id for corpus_id in corpus_ids if corpus_id is not None})
        targets_by_side = self.collect_citation_edges(
            resolved_ids,
            sides,
            max_results=max(0, max_results),
            min_publication_date=min_publication_date,
        )

        # One fetch serves the papers and every neighbor: it asks for the union
        # of both roles' fields, and each role is then trimmed to its own fields.
        fetch_fields = list(
            dict.fromkeys(
                [*requested_fields, *(field for side in sides for field in neighbor_fields[side])]
            )
        )
        fetched_papers = self.collect_papers_by_corpus_ids(
            [
                *resolved_ids,
                *(
                    target_id
                    for targets_by_source in targets_by_side.values()
                    for target_ids in (targets_by_source or {}).values()
                    for target_id in target_ids
                    if target_id is not None
                ),
            ],
            fields=fetch_fields,
        )

        def papers_with_fields(corpus_ids_in_role: Iterable[int], role_fields: list[str]) -> dict[int, dict[str, Any]]:
            trim = requested_top_level_fields(role_fields, _DEFAULT_PAPER_FIELDS) != (
                requested_top_level_fields(fetch_fields, _DEFAULT_PAPER_FIELDS)
            )
            return {
                corpus_id: (
                    select_paper_fields(fetched_papers[corpus_id], role_fields, _DEFAULT_PAPER_FIELDS)
                    if trim
                    else fetched_papers[corpus_id]
                )
                for corpus_id in corpus_ids_in_role
                if corpus_id in fetched_papers
            }

        papers_by_corpus_id = papers_with_fields(resolved_ids, requested_fields)
        missing_sides = [side for side in sides if targets_by_side[side] is None]
        if papers_by_corpus_id and missing_sides:
            raise FileNotFoundError("Semantic Scholar citations index/table is not available.")

        neighbors_by_side: dict[str, dict[int, list[dict[str, Any]]]] = {}
        for side in sides:
            wrapper_field = CITATION_SIDES[side][3]
            targets_by_source = targets_by_side[side] or {}
            neighbor_papers = papers_with_fields(
                {
                    target_id
                    for target_ids in targets_by_source.values()
                    for target_id in target_ids
                    if target_id is not None
                },
                neighbor_fields[side],
            )
            neighbors_by_side[side] = {
                source_id: [
                    {wrapper_field: neighbor_papers.get(target_id) if target_id is not None else None}
                    for target_id in target_ids
                ]
                for source_id, target_ids in targets_by_source.items()
            }

        result = {}
        for original_id, corpus_id in zip(paper_ids, corpus_ids):
            paper = papers_by_corpus_id.get(corpus_id) if corpus_id is not None else None
            if paper is None:
                result[original_id] = (None, [], [])
                continue
            result[original_id] = (
                paper,
                neighbors_by_side.get("citations", {}).get(corpus_id, []),
                neighbors_by_side.get("references", {}).get(corpus_id, []),
            )
        return result

    def iter_expansion_subgraph_batches(
        self,
        paper_ids: list[str],
        *,
        fields: list[str] | None = None,
        max_results: int = DEFAULT_MAX_RESULTS,
        edge_budget: int = 100_000,
        include_citations: bool = True,
        include_references: bool = True,
        min_publication_date: str | None = None,
    ) -> Iterator[ExpansionSubgraphBatch]:
        """Prepare selected edge sides once, then yield groups by actual edge volume.

        Ranking and edge counts run in Spark. Candidate edges are temporarily
        persisted for cutoff filtering, then only capped edges remain cached.
        Each group holds at most edge_budget edges, except that a single
        source stays intact and can have up to 2 * max_results edges.
        Metadata uses the existing bounded cache. Call close() on an abandoned
        iterator to release persisted edges immediately.
        Optional abstract/PDF enrichment uses completed indexes only.
        """
        if edge_budget < 1:
            raise ValueError("edge_budget must be positive")
        if not self.use_index:
            raise ValueError("Subgraph expansion requires the bulk index")
        source_ids = dict(zip(paper_ids, self.resolve_indexed_paper_ids(paper_ids)))
        resolved_ids = sorted(
            {value for value in source_ids.values() if value is not None}
        )
        if not resolved_ids:
            if source_ids:
                yield ExpansionSubgraphBatch(source_ids, {}, {}, {})
            return
        F = self.spark_functions
        sides = [
            side
            for side, included in (
                ("citations", include_citations),
                ("references", include_references),
            )
            if included
        ]
        ranked_edges = None
        try:
            with self.citation_edges(
                resolved_ids, sides, min_publication_date=min_publication_date
            ) as (edges, missing):
                if missing:
                    raise FileNotFoundError(
                        "Semantic Scholar citations index/table is not available."
                    )
                counts = {}
                if edges is not None:
                    order = self.window.partitionBy("side", "sourceCorpusId").orderBy(
                        F.asc("citationId")
                    )
                    ranked_edges = (
                        edges.withColumn("edgeRank", F.row_number().over(order))
                        .filter(F.col("edgeRank") <= max(0, max_results))
                        .persist()
                    )
                    counts = {
                        int(row["sourceCorpusId"]): int(row["count"])
                        for row in ranked_edges.groupBy("sourceCorpusId")
                        .count()
                        .collect()
                    }
            # Counting materializes capped edges before candidate edges are released.
            fetch_fields = list(
                dict.fromkeys(
                    [
                        *(fields or _DEFAULT_PAPER_FIELDS),
                        *(["referenceCount"] if include_references else []),
                        *(["citationCount"] if include_citations else []),
                    ]
                )
            )
            for group in expansion_source_groups(source_ids, counts, edge_budget):
                ids = sorted({value for value in group.values() if value is not None})
                rows = (
                    ranked_edges.filter(F.col("sourceCorpusId").isin(ids))
                    .select("side", "sourceCorpusId", "targetCorpusId", "edgeRank")
                    .collect()
                    if ranked_edges is not None
                    else []
                )
                rows.sort(
                    key=lambda row: (
                        row["side"],
                        row["sourceCorpusId"],
                        row["edgeRank"],
                    )
                )
                neighbors: dict[str, dict[int, list[int | None]]] = {
                    "citations": {},
                    "references": {},
                }
                neighbor_ids: set[int] = set()
                for row in rows:
                    target = row["targetCorpusId"]
                    if target is not None:
                        neighbor_ids.add(int(target))
                    neighbors[row["side"]].setdefault(
                        int(row["sourceCorpusId"]), []
                    ).append(int(target) if target is not None else None)
                del rows
                # The bounded edge chunk already contains the IDs. Reusing them
                # avoids another Spark action just to collect distinct IDs.
                papers = self.collect_indexed_papers_by_corpus_ids(
                    [*ids, *sorted(neighbor_ids)],
                    fields=fetch_fields,
                    join_abstracts=True,
                )
                yield ExpansionSubgraphBatch(
                    group, papers, neighbors["citations"], neighbors["references"]
                )
        finally:
            if ranked_edges is not None:
                ranked_edges.unpersist()

    def get_reference_validation_counts(
        self,
        paper_ids: list[str],
        *,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> dict[str, tuple[int | None, int]]:
        """Return (reported count, visible edge count) without loading neighbors.

        Resolve all SHA IDs together, then scan the papers and citing-citation
        indexes once each. The edge count has the same cap as the batch API.
        """
        if not paper_ids:
            return {}
        if not self.use_index:
            raise ValueError("Reference-count validation requires the bulk index.")
        citation_table = "citations_by_citing"
        if (
            citation_table not in (self.index_metadata.get("tables") or [])
            or not (self.index_table_path(citation_table) / "_SUCCESS").is_file()
        ):
            raise FileNotFoundError(
                "Incomplete Semantic Scholar bulk index table: "
                f"{self.index_table_path(citation_table)}"
            )

        resolved_ids = self.resolve_indexed_paper_ids(paper_ids)
        corpus_ids = sorted({corpus_id for corpus_id in resolved_ids if corpus_id is not None})
        if not corpus_ids:
            return {paper_id: (None, 0) for paper_id in paper_ids}

        paper_rows = (
            self.corpus_bucket_filter(self.papers_dataframe, corpus_ids)
            .select("corpusId", "referenceCount")
            .collect()
        )
        reported_counts = {
            int(row["corpusId"]): (
                int(row["referenceCount"]) if row["referenceCount"] is not None else None
            )
            for row in paper_rows
        }
        visible_counts: dict[int, int] = {}
        effective_max_results = max(0, max_results)
        countable_ids = [
            corpus_id for corpus_id, reported_count in reported_counts.items()
            if reported_count is not None
        ]
        if countable_ids and effective_max_results:
            citation_rows = (
                self.bucketed_id_filter(
                    self.read_index_table(citation_table),
                    id_column="citingCorpusId",
                    bucket_column="citingCorpusIdBucket",
                    ids=countable_ids,
                )
                .select("citingCorpusId")
                .groupBy("citingCorpusId")
                .count()
                .collect()
            )
            visible_counts = {
                int(row["citingCorpusId"]): min(int(row["count"]), effective_max_results)
                for row in citation_rows
            }
        return {
            paper_id: (
                reported_counts.get(corpus_id),
                visible_counts.get(corpus_id, 0),
            )
            for paper_id, corpus_id in zip(paper_ids, resolved_ids)
        }

    def citation_edge_frames(
        self,
        corpus_ids: list[int],
        sides: list[str],
    ) -> dict[str, Any]:
        """Read only the edge columns needed for the requested sources and sides."""
        F = self.spark_functions
        side_edges = {}
        for side in sides:
            table_name, source_column, target_column, _, _ = CITATION_SIDES[side]
            dataframe = self.citation_dataframe_for_lookup(table_name)
            if dataframe is None:
                side_edges[side] = None
                continue
            side_edges[side] = self.bucketed_id_filter(
                dataframe,
                id_column=source_column,
                bucket_column=f"{source_column}Bucket",
                ids=corpus_ids,
            ).select(
                F.lit(side).alias("side"),
                F.col(source_column).alias("sourceCorpusId"),
                F.col(target_column).alias("targetCorpusId"),
                "citationId",
            )
        return side_edges

    @contextmanager
    def citation_edges(
        self,
        corpus_ids: list[int],
        sides: list[str],
        *,
        min_publication_date: str | None = None,
    ) -> Iterator[tuple[Any, list[str]]]:
        """Yield combined edges and missing sides, releasing temporary caches.

        With a cutoff, persist candidate edges while discovering neighbor
        buckets and joining dates. Only bucket IDs reach Python; the exact
        neighbor join stays in Spark. Dates use publicationDate, falling back
        to January 1 of year. Unknown or undated targets are dropped. The
        ISO cutoff is inclusive. Callers must rank/cap edges after this filter.
        """
        frames = self.citation_edge_frames(corpus_ids, sides)
        missing = [side for side, frame in frames.items() if frame is None]
        available = [frame for frame in frames.values() if frame is not None]
        if not available:
            yield None, missing
            return
        edges = available[0]
        for frame in available[1:]:
            edges = edges.unionByName(frame)
        if not min_publication_date:
            yield edges, missing
            return

        F = self.spark_functions
        candidates = edges.persist()
        try:
            papers = self.papers_dataframe
            if "corpusIdBucket" in papers.columns:
                buckets = [
                    int(row["bucket"])
                    for row in candidates.filter(F.col("targetCorpusId").isNotNull())
                    .select(
                        F.pmod(
                            F.crc32(F.col("targetCorpusId").cast("string")),
                            F.lit(self.num_buckets),
                        ).alias("bucket")
                    )
                    .distinct()
                    .collect()
                ]
                if not buckets:
                    yield candidates.limit(0), missing
                    return
                papers = papers.filter(
                    f"corpusIdBucket IN ({', '.join(map(str, sorted(buckets)))})"
                )
            target_dates = papers.select(
                F.col("corpusId").alias("targetCorpusId"),
                F.coalesce(
                    F.col("publicationDate"),
                    F.concat(F.col("year").cast("string"), F.lit("-01-01")),
                ).alias("targetEffectiveDate"),
            ).filter(F.col("targetEffectiveDate") >= min_publication_date)
            yield (
                candidates.join(target_dates, on="targetCorpusId").select(
                    "side", "sourceCorpusId", "targetCorpusId", "citationId"
                ),
                missing,
            )
        finally:
            candidates.unpersist()

    def collect_citation_edges(
        self,
        corpus_ids: list[int],
        sides: list[str],
        *,
        max_results: int,
        min_publication_date: str | None = None,
    ) -> dict[str, dict[int, list[int | None]] | None]:
        """Return ``{side: {source corpusId: [target corpusId or None, ...]}}``.

        A combined query reads the edges of every source paper for every side
        in ``sides`` (keys of ``CITATION_SIDES``), keeping the first
        ``max_results`` per side and source by ``citationId``. Only the ID
        columns of the edge tables are read. A side whose edge table is
        missing maps to ``None``. ``min_publication_date`` is forwarded to
        ``citation_edges`` to drop older edges before the ``max_results`` cap.
        Cutoff filtering adds a bucket-discovery action for indexed papers.
        """
        targets_by_side: dict[str, dict[int, list[int | None]] | None] = {
            side: {} for side in sides
        }
        if not corpus_ids or max_results <= 0:
            return targets_by_side

        F = self.spark_functions
        with self.citation_edges(
            corpus_ids, sides, min_publication_date=min_publication_date
        ) as (edges, missing):
            for side in missing:
                targets_by_side[side] = None
            if edges is None:
                return targets_by_side
            edge_order = self.window.partitionBy("side", "sourceCorpusId").orderBy(
                F.asc("citationId")
            )
            rows = (
                edges.withColumn("edgeRank", F.row_number().over(edge_order))
                .filter(F.col("edgeRank") <= max_results)
                .collect()
            )
        rows.sort(key=lambda row: (row["side"], row["sourceCorpusId"], row["edgeRank"]))
        for row in rows:
            target_id = row["targetCorpusId"]
            targets_by_side[row["side"]].setdefault(
                int(row["sourceCorpusId"]), []
            ).append(int(target_id) if target_id is not None else None)
        return targets_by_side

    def enrich_with_references(
        self,
        papers: list[Any],
        *,
        fields: Optional[list[str]] = None,
        max_refs_per_paper: int = 500,
    ) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for paper in papers:
            if isinstance(paper, dict):
                meta = paper.copy()
                arxiv_id = meta.get("arxiv_id", "")
            else:
                meta = paper.to_dict() if hasattr(paper, "to_dict") else vars(paper)
                arxiv_id = getattr(paper, "arxiv_id", "")

            if not arxiv_id:
                meta["references"] = []
                enriched.append(meta)
                continue

            try:
                refs = self.get_all_references(
                    normalize_paper_id(arxiv_id),
                    fields=fields,
                    max_results=max_refs_per_paper,
                )
                meta["references"] = [ref["citedPaper"] for ref in refs if ref.get("citedPaper")]
            except requests.HTTPError:
                meta["references"] = []
            enriched.append(meta)
        return enriched

    def get_citation_edges(
        self,
        paper_id: str,
        *,
        table_name: str,
        source_column: str,
        target_column: str,
        wrapper_field: str,
        fields: Optional[list[str]],
        default_fields: Iterable[str],
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        corpus_id = self.resolve_paper_id_to_corpus_id(paper_id)
        if corpus_id is None:
            raise response_not_found(f"Paper not found: {paper_id}")

        dataframe = self.citation_dataframe_for_lookup(table_name)
        if dataframe is None:
            raise FileNotFoundError("Semantic Scholar citations index/table is not available.")

        F = self.spark_functions
        if f"{source_column}Bucket" in dataframe.columns:
            dataframe = dataframe.filter(
                F.col(f"{source_column}Bucket") == stable_bucket(corpus_id, self.num_buckets)
            )
        dataframe = dataframe.filter(F.col(source_column) == corpus_id)
        total = dataframe.count()
        rows = dataframe.orderBy(F.asc("citationId")).limit(offset + limit).collect()[offset:]
        edge_records = [canonical_citation_record(row.asDict(recursive=True)) for row in rows]
        target_ids = [int(record[target_column]) for record in edge_records if record.get(target_column) is not None]
        paper_fields = edge_paper_fields(fields, wrapper_field, default_fields)
        papers_by_id = self.collect_papers_by_corpus_ids(target_ids, fields=paper_fields)

        data = []
        for record in edge_records:
            target_id = record.get(target_column)
            item = {
                "contexts": record.get("contexts"),
                "intents": flatten_intents(record.get("intents")),
                "contextsWithIntent": contexts_with_intent(record.get("contexts"), record.get("intents")),
                "isInfluential": record.get("isInfluential", False),
                wrapper_field: papers_by_id.get(int(target_id)) if target_id is not None else None,
            }
            data.append(item)

        result: dict[str, Any] = {"offset": offset, "data": data}
        if offset + limit < total:
            result["next"] = offset + limit
        return result

    def citation_dataframe_for_lookup(self, table_name: str) -> Any | None:
        if self.use_index and self.index_table_exists(table_name):
            return self.read_index_table(table_name)
        return self.read_citations_dataframe()

    def resolve_paper_id_to_corpus_id(self, paper_id: str) -> int | None:
        if self.use_index:
            return self.resolve_indexed_paper_id(paper_id)
        normalized_id = normalize_paper_id(paper_id)
        upper_id = normalized_id.upper()
        if upper_id.startswith("CORPUSID:"):
            return int(normalized_id.split(":", 1)[1])
        if normalized_id.isdigit() and not HEX_SHA_RE.fullmatch(normalized_id):
            return int(normalized_id)
        try:
            paper = self.get_paper_by_id(normalized_id, fields=["corpusId"])
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise
        return int(paper["corpusId"]) if paper.get("corpusId") is not None else None

    def resolve_paper_ids_to_corpus_ids(self, paper_ids: list[str]) -> list[int | None]:
        if self.use_index:
            return self.resolve_indexed_paper_ids(paper_ids)
        return [self.resolve_paper_id_to_corpus_id(paper_id) for paper_id in paper_ids]

    def search_snippets(
        self,
        query: str,
        *,
        fields: Optional[list[str] | str] = None,
        paper_ids: Optional[list[str] | str] = None,
        authors: Optional[list[str] | str] = None,
        min_citation_count: Optional[int | str] = None,
        inserted_before: Optional[str] = None,
        publication_date_or_year: Optional[str] = None,
        year: Optional[str] = None,
        venue: Optional[list[str] | str] = None,
        fields_of_study: Optional[list[str] | str] = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        del fields, inserted_before
        dataframe = self.read_s2orc_text_dataframe()
        if dataframe is None:
            raise FileNotFoundError("Semantic Scholar S2ORC text index/table is not available.")

        F = self.spark_functions
        dataframe = self.normalize_s2orc_text_dataframe(dataframe)
        terms = query_terms(query)
        searchable = F.lower(F.concat_ws(" ", F.coalesce(F.col("title"), F.lit("")), F.coalesce(F.col("bodyText"), F.lit(""))))
        for term in terms:
            dataframe = dataframe.filter(searchable.contains(term))

        requested_paper_ids = normalize_string_values(paper_ids)
        if requested_paper_ids:
            corpus_ids = [
                corpus_id
                for corpus_id in (self.resolve_paper_id_to_corpus_id(paper_id) for paper_id in requested_paper_ids)
                if corpus_id is not None
            ]
            dataframe = self.corpus_bucket_filter(dataframe, corpus_ids)

        author_values = [value.lower() for value in normalize_string_values(authors)]
        if author_values:
            author_text = F.lower(F.concat_ws(" ", F.col("authors")))
            for author_value in author_values:
                dataframe = dataframe.filter(author_text.contains(author_value))

        if any([min_citation_count, publication_date_or_year, year, venue, fields_of_study]):
            corpus_rows = dataframe.select("corpusId").limit(self.search_candidate_limit).collect()
            candidate_ids = [int(row["corpusId"]) for row in corpus_rows]
            paper_frame = self.indexed_papers_dataframe_by_corpus_ids(candidate_ids) if self.use_index else self.papers_dataframe.filter(F.col("corpusId").isin(candidate_ids))
            paper_frame = self.apply_min_citation_filter(paper_frame, min_citation_count)
            paper_frame = self.apply_publication_date_filter(paper_frame, publication_date_or_year)
            paper_frame = self.apply_year_filter(paper_frame, year)
            paper_frame = self.apply_venue_filter(paper_frame, venue)
            paper_frame = self.apply_fields_of_study_filter(paper_frame, fields_of_study)
            filtered_ids = [int(row["corpusId"]) for row in paper_frame.select("corpusId").collect()]
            dataframe = self.corpus_bucket_filter(dataframe, filtered_ids)

        rows = dataframe.limit(limit).collect()
        data = [self.snippet_match_from_row(row.asDict(recursive=True), query) for row in rows]
        return {"data": data, "retrievalVersion": "local-bulk-title-body-v1"}

    def normalize_s2orc_text_dataframe(self, dataframe: Any) -> Any:
        F = self.spark_functions
        if "bodyText" in dataframe.columns:
            return dataframe
        if "body" in dataframe.columns:
            dataframe = dataframe.withColumn("bodyText", F.col("body.text"))
        elif "content" in dataframe.columns:
            dataframe = dataframe.withColumn("bodyText", F.col("content.text"))
        else:
            dataframe = dataframe.withColumn("bodyText", F.lit(None).cast("string"))
        if "openAccessInfo" not in dataframe.columns and "openaccessinfo" in dataframe.columns:
            dataframe = dataframe.withColumnRenamed("openaccessinfo", "openAccessInfo")
        if "openAccessInfo" not in dataframe.columns and "content" in dataframe.columns:
            dataframe = dataframe.withColumn(
                "openAccessInfo",
                F.struct(
                    F.col("content.source.oainfo.license").alias("license"),
                    F.col("content.source.oainfo.status").alias("status"),
                    F.lit(None).cast("string").alias("disclaimer"),
                ),
            )
        if "title" not in dataframe.columns:
            dataframe = dataframe.withColumn("title", F.lit(None).cast("string"))
        if "authors" not in dataframe.columns:
            dataframe = dataframe.withColumn("authors", F.expr("array()").cast("array<string>"))
        return dataframe

    def snippet_match_from_row(self, row: dict[str, Any], query: str) -> dict[str, Any]:
        body_text = row.get("bodyText") or ""
        title = row.get("title")
        snippet_text = self.extract_snippet_text(body_text, query) or title or ""
        terms = query_terms(query)
        lower_text = body_text.lower()
        matched_terms = sum(1 for term in set(terms) if term in lower_text)
        score = float(matched_terms / max(len(set(terms)), 1))
        open_access = row.get("openAccessInfo") or {}
        return {
            "snippet": {
                "text": snippet_text,
                "snippetKind": "body" if body_text else "title",
            },
            "score": score,
            "paper": {
                "corpusId": str(row.get("corpusId")),
                "title": title,
                "authors": row.get("authors") or [],
                "openAccessInfo": {
                    "license": open_access.get("license"),
                    "status": open_access.get("status"),
                    "disclaimer": open_access.get("disclaimer"),
                },
            },
        }

    def extract_snippet_text(self, text: str, query: str, radius: int = 220) -> str:
        if not text:
            return ""
        lower_text = text.lower()
        positions = [lower_text.find(term) for term in query_terms(query)]
        positions = [position for position in positions if position >= 0]
        center = min(positions) if positions else 0
        start = max(0, center - radius)
        end = min(len(text), center + radius)
        return text[start:end].strip()

    def filter_by_paper_id(self, dataframe: Any, paper_id: str) -> Any:
        F = self.spark_functions
        normalized_id = paper_id.strip()
        upper_id = normalized_id.upper()

        if upper_id.startswith("CORPUSID:"):
            return dataframe.filter(F.col("corpusId") == int(normalized_id.split(":", 1)[1]))
        if normalized_id.isdigit() and not HEX_SHA_RE.fullmatch(normalized_id):
            return dataframe.filter(F.col("corpusId") == int(normalized_id))
        if upper_id.startswith("ARXIV:"):
            return dataframe.filter(F.col("externalIds.ArXiv") == normalized_id.split(":", 1)[1])

        prefix, separator, value = normalized_id.partition(":")
        external_id_key = EXTERNAL_ID_KEYS.get(prefix) or EXTERNAL_ID_KEYS.get(prefix.upper())
        if separator and external_id_key:
            return dataframe.filter(F.col(f"externalIds.{external_id_key}") == value)
        if HEX_SHA_RE.fullmatch(normalized_id):
            if not self.use_index and self.paper_ids_dataframe is not None:
                matching_corpus_ids = (
                    self.paper_ids_dataframe.filter(
                        F.col("sha") == normalized_id.lower()
                    )
                    .select("corpusId")
                    .distinct()
                )
                return dataframe.join(matching_corpus_ids, on="corpusId", how="inner")
            return dataframe.filter(F.lower(F.col("paperId")) == normalized_id.lower())
        if normalized_id.startswith("10."):
            return dataframe.filter(F.col("externalIds.DOI") == normalized_id)
        return dataframe.filter(F.col("paperId") == normalized_id)

    def filtered_search_dataframe(
        self,
        *,
        query: str,
        publication_types: Optional[list[str] | str] = None,
        open_access_pdf: bool = False,
        min_citation_count: Optional[int | str] = None,
        publication_date_or_year: Optional[str] = None,
        year: Optional[str] = None,
        venue: Optional[list[str] | str] = None,
        fields_of_study: Optional[list[str] | str] = None,
    ) -> Any:
        if self.use_index:
            dataframe = self.indexed_search_dataframe(query)
        else:
            dataframe = self.filter_by_query(self.papers_dataframe, query)
        dataframe = self.apply_publication_types_filter(dataframe, publication_types)
        dataframe = self.apply_open_access_filter(dataframe, open_access_pdf)
        dataframe = self.apply_min_citation_filter(dataframe, min_citation_count)
        dataframe = self.apply_publication_date_filter(dataframe, publication_date_or_year)
        dataframe = self.apply_year_filter(dataframe, year)
        dataframe = self.apply_venue_filter(dataframe, venue)
        dataframe = self.apply_fields_of_study_filter(dataframe, fields_of_study)
        return dataframe

    def indexed_search_dataframe(self, query: str) -> Any:
        corpus_ids = self.indexed_search_corpus_ids(query)
        return self.indexed_papers_dataframe_by_corpus_ids(corpus_ids)

    def indexed_search_corpus_ids(self, query: str) -> list[int]:
        if not self.index_table_exists("title_tokens"):
            return []
        terms = query_terms_for_index(query)
        if not terms:
            return []
        unique_terms = sorted(set(terms))
        buckets = sorted({
            stable_bucket(term, self.num_buckets)
            for term in unique_terms
        })
        F = self.spark_functions
        token_matches = (
            self.read_index_table("title_tokens")
            .filter(
                F.col("titleTokenBucket").isin(buckets)
                & F.col("titleToken").isin(unique_terms)
            )
        )
        grouped = token_matches.groupBy("corpusId").agg(
            F.countDistinct("titleToken").alias("matchedTitleTokens")
        )
        if len(unique_terms) > 1:
            grouped = grouped.filter(F.col("matchedTitleTokens") == len(unique_terms))
        rows = (
            grouped.orderBy(F.desc("matchedTitleTokens"), F.asc("corpusId"))
            .limit(self.search_candidate_limit)
            .collect()
        )
        return [int(row["corpusId"]) for row in rows]

    def filter_by_query(self, dataframe: Any, query: str) -> Any:
        F = self.spark_functions
        terms = query_terms(query)
        if not terms:
            return dataframe
        title = F.lower(F.coalesce(F.col("title"), F.lit("")))
        for term in terms:
            dataframe = dataframe.filter(title.contains(term))
        return dataframe

    def apply_publication_types_filter(
        self,
        dataframe: Any,
        publication_types: Optional[list[str] | str],
    ) -> Any:
        values = normalize_string_values(publication_types)
        if not values:
            return dataframe
        F = self.spark_functions
        return dataframe.filter(F.arrays_overlap(F.col("publicationTypes"), F.array(*[F.lit(value) for value in values])))

    def apply_open_access_filter(self, dataframe: Any, open_access_pdf: bool) -> Any:
        if not open_access_pdf:
            return dataframe
        return dataframe.filter(self.spark_functions.col("isOpenAccess") == True)

    def apply_min_citation_filter(
        self,
        dataframe: Any,
        min_citation_count: Optional[int | str],
    ) -> Any:
        if min_citation_count is None:
            return dataframe
        return dataframe.filter(self.spark_functions.col("citationCount") >= int(min_citation_count))

    def apply_year_filter(self, dataframe: Any, year: Optional[str]) -> Any:
        if not year:
            return dataframe
        F = self.spark_functions
        start_year, end_year = year_bounds(year)
        if start_year is not None:
            dataframe = dataframe.filter(F.col("year") >= start_year)
        if end_year is not None:
            dataframe = dataframe.filter(F.col("year") <= end_year)
        return dataframe

    def apply_publication_date_filter(
        self,
        dataframe: Any,
        publication_date_or_year: Optional[str],
    ) -> Any:
        if not publication_date_or_year:
            return dataframe
        F = self.spark_functions
        effective_date = F.coalesce(
            F.col("publicationDate"),
            F.concat(F.col("year").cast("string"), F.lit("-01-01")),
        )
        value = publication_date_or_year.strip()
        if ":" not in value:
            if re.fullmatch(r"\d{4}", value):
                return dataframe.filter(F.col("year") == int(value))
            return dataframe.filter(effective_date.startswith(value))

        start, end = value.split(":", 1)
        if start:
            dataframe = dataframe.filter(effective_date >= date_prefix_lower(start))
        if end:
            dataframe = dataframe.filter(effective_date <= date_prefix_upper(end))
        return dataframe

    def apply_venue_filter(
        self,
        dataframe: Any,
        venue: Optional[list[str] | str],
    ) -> Any:
        values = [value.lower() for value in normalize_string_values(venue)]
        if not values:
            return dataframe
        F = self.spark_functions
        return dataframe.filter(F.lower(F.col("venue")).isin(values))

    def apply_fields_of_study_filter(
        self,
        dataframe: Any,
        fields_of_study: Optional[list[str] | str],
    ) -> Any:
        values = normalize_string_values(fields_of_study)
        if not values:
            return dataframe
        F = self.spark_functions
        return dataframe.filter(F.arrays_overlap(F.col("fieldsOfStudy"), F.array(*[F.lit(value) for value in values])))

    def order_for_search(self, dataframe: Any, query: str) -> Any:
        F = self.spark_functions
        scored = self.add_title_match_score(dataframe, query)
        return scored.orderBy(
            F.desc("matchScore"),
            F.desc("citationCount"),
            F.asc("title"),
        )

    def order_for_bulk(self, dataframe: Any, query: str, sort: str | None) -> Any:
        F = self.spark_functions
        sort_spec = sort_components(sort)
        if sort_spec is not None:
            field, descending = sort_spec
            if field in dataframe.columns:
                order_column = F.desc(field) if descending else F.asc(field)
                return dataframe.orderBy(order_column, F.asc("corpusId"))
        return self.order_for_search(dataframe, query)

    def add_title_match_score(self, dataframe: Any, query: str) -> Any:
        F = self.spark_functions
        normalized_query = " ".join(query_terms(query))
        if not normalized_query:
            return dataframe.withColumn("matchScore", F.lit(0.0))

        title = F.lower(F.coalesce(F.col("title"), F.lit("")))
        max_length = F.greatest(F.length(title), F.lit(len(normalized_query)))
        distance = F.levenshtein(title, F.lit(normalized_query))
        score = F.round(F.greatest(F.lit(0.0), F.lit(100.0) - (distance * 100.0 / max_length)), 2)
        exact_score = F.when(title == normalized_query, F.lit(100.0)).otherwise(score)
        return dataframe.withColumn("matchScore", exact_score)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local Semantic Scholar bulk-client examples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root containing papers/ and paper-ids/ bulk dataset folders.",
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=DEFAULT_INDEX_ROOT,
        help="Root containing the built Parquet lookup index.",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Read raw JSON shards even if a built index exists.",
    )
    parser.add_argument(
        "--search-candidate-limit",
        type=int,
        default=DEFAULT_SEARCH_CANDIDATE_LIMIT,
        help="Maximum title-token index candidates to score per search.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Title query for search examples. Defaults to words from the first local paper title.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache the loaded papers DataFrame in Spark.",
    )
    return parser


def example_query_from_title(title: str | None) -> str:
    terms = query_terms(title or "")
    return " ".join(terms[:4]) if terms else "paper"


def print_example(label: str, callback: Any) -> Any:
    print(f"{label}:")
    try:
        result = callback()
    except (FileNotFoundError, requests.HTTPError) as exc:
        print(f"  Skipped: {exc}")
        return None
    print(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode("utf-8"))
    return result


def main() -> None:
    args = build_parser().parse_args()
    client = SemanticScholarBulkClient(
        data_root=args.data_root,
        index_root=args.index_root,
        use_index=not args.no_index,
        cache=args.cache,
        search_candidate_limit=args.search_candidate_limit,
    )

    print(f"Using indexed data: {client.use_index}")

    print("Papers schema:")
    client.papers_dataframe.printSchema()

    print("First papers:")
    client.papers_dataframe.select(
        "paperId",
        "corpusId",
        "title",
        "year",
        "citationCount",
    ).show(5, truncate=80)

    first = client.papers_dataframe.select("paperId", "title", "corpusId").limit(1).collect()
    if not first:
        print("No papers found.")
        return

    first_paper = first[0].asDict(recursive=True)
    query = args.query or example_query_from_title(first_paper.get("title"))
    paper_id = first_paper.get("paperId") or f"CorpusId:{first_paper['corpusId']}"

    title = first_paper.get("title") or query

    print(f"Search query: {query!r}")
    print_example(
        "Search papers",
        lambda: client.search_papers(query, fields=["title", "year", "citationCount"], limit=3),
    )
    print_example(
        "Search by exact paper title",
        lambda: client.search_by_title(title, fields=["title", "year", "citationCount"]),
    )
    print_example(
        f"Lookup by paper id: {paper_id}",
        lambda: client.get_paper_by_id(
            paper_id,
            fields=["title", "authors", "year", "citationCount", "abstract", "tldr"],
        ),
    )
    print_example("Autocomplete", lambda: client.autocomplete_paper(query))
    print_example(
        "Title match",
        lambda: client.match_paper_title(query, fields=["title", "year", "citationCount"]),
    )
    print_example(
        "Bulk search page",
        lambda: client.search_papers_bulk(query, fields=["title", "year"], sort="citationCount:desc"),
    )
    authors_page = print_example(
        "Paper authors",
        lambda: client.get_paper_authors(paper_id, fields=["name", "hIndex"], limit=5),
    )

    first_author_id = None
    if authors_page and authors_page.get("data"):
        first_author_id = authors_page["data"][0].get("authorId")
    if first_author_id:
        print_example(
            f"Author lookup: {first_author_id}",
            lambda: client.get_author(first_author_id, fields=["name", "paperCount", "citationCount"]),
        )
        print_example(
            f"Author papers: {first_author_id}",
            lambda: client.get_author_papers(first_author_id, fields=["title", "year"], limit=3),
        )

    print_example(
        "References",
        lambda: client.get_references(paper_id, fields=["citedPaper.title", "citedPaper.year"], limit=3),
    )
    print_example(
        "Citations",
        lambda: client.get_citations(paper_id, fields=["citingPaper.title", "citingPaper.year"], limit=3),
    )
    print_example("Snippet search", lambda: client.search_snippets(query, limit=3))


if __name__ == "__main__":
    main()
