"""Tests for build_semantic_scholar_bulk_api_index.py."""

from __future__ import annotations

import orjson
from pathlib import Path

import pytest

import build_semantic_scholar_bulk_api_index as module
from src.clients.semantic_scholar_bulk_client import SemanticScholarBulkClient


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(orjson.dumps(record).decode("utf-8") + "\n")


@pytest.fixture(scope="module")
def spark_session():
    pyspark_sql = pytest.importorskip("pyspark.sql")
    try:
        spark = (
            pyspark_sql.SparkSession.builder.master("local[1]")
            .appName("semantic-scholar-index-builder-tests")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
    except Exception as exc:
        pytest.skip(f"Could not start local Spark session: {exc}")
    yield spark
    spark.stop()


@pytest.fixture()
def raw_data_root(tmp_path: Path) -> Path:
    raw_root = tmp_path / "raw_datasets"
    write_jsonl(
        raw_root / "papers" / "papers.json",
        [
            {
                "corpusid": 1,
                "externalids": {
                    "CorpusId": "1",
                    "ArXiv": "1706.03762",
                    "DOI": "10.5555/attention",
                },
                "url": "https://www.semanticscholar.org/paper/dddddddddddddddddddddddddddddddddddddddd",
                "title": "Attention Is All You Need",
                "authors": [{"authorId": "a1", "name": "Ashish Vaswani"}],
                "venue": "NeurIPS",
                "publicationvenueid": "venue-neurips",
                "year": 2017,
                "citationcount": 1000,
                "isopenaccess": True,
                "s2fieldsofstudy": [
                    {"category": "Computer Science", "source": "s2-fos-model"}
                ],
                "publicationtypes": ["Conference"],
                "publicationdate": "2017-06-12",
            },
            {
                "corpusid": 2,
                "externalids": {"CorpusId": "2"},
                "url": "https://www.semanticscholar.org/paper/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "title": "Biology of Neural Attention",
                "authors": [{"authorId": "a2", "name": "B. Researcher"}],
                "venue": "Nature",
                "year": 2020,
                "citationcount": 25,
                "isopenaccess": False,
                "s2fieldsofstudy": [{"category": "Biology", "source": "s2-fos-model"}],
                "publicationdate": "2020-01-02",
            },
        ],
    )
    write_jsonl(
        raw_root / "paper-ids" / "paper-ids.json",
        [
            {
                "sha": "dddddddddddddddddddddddddddddddddddddddd",
                "corpusid": 1,
                "primary": False,
            },
            {
                "sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "corpusid": 1,
                "primary": True,
            },
            {
                "sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "corpusid": 2,
                "primary": True,
            },
        ],
    )
    write_jsonl(
        raw_root / "authors" / "authors.json",
        [{"authorid": "a1", "url": "https://example.test/a1", "name": "Ashish Vaswani", "papercount": 10}],
    )
    write_jsonl(
        raw_root / "citations" / "citations.json",
        [
            {
                "citationid": 10,
                "citingcorpusid": 1,
                "citedcorpusid": 2,
                "contexts": ["Attention cites biology."],
                "intents": [["background"]],
                "isinfluential": False,
            }
        ],
    )
    write_jsonl(
        raw_root / "abstracts" / "abstracts.json",
        [
            {
                "corpusid": 1,
                "abstract": "A transformer architecture.",
                "openaccessinfo": {"url": "https://example.test/attention.pdf"},
            }
        ],
    )
    write_jsonl(
        raw_root / "publication-venues" / "venues.json",
        [{"id": "venue-neurips", "name": "NeurIPS", "type": "conference", "alternate_names": [], "alternate_urls": []}],
    )
    return raw_root


def test_build_index_supports_indexed_client_lookup(spark_session, raw_data_root: Path, tmp_path: Path) -> None:
    index_root = tmp_path / "index"

    module.build_semantic_scholar_bulk_index(
        raw_root=raw_data_root,
        index_root=index_root,
        num_buckets=8,
        overwrite=True,
        spark_session=spark_session,
    )
    client = SemanticScholarBulkClient(
        data_root=raw_data_root,
        index_root=index_root,
        spark_session=spark_session,
    )

    assert client.use_index is True
    assert (index_root / "metadata.json").exists()
    paper = client.get_paper_by_id("CorpusId:1", fields=["title"])
    assert paper["paperId"] == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert paper["title"] == "Attention Is All You Need"
    assert client.get_paper_by_id(
        "dddddddddddddddddddddddddddddddddddddddd",
        fields=["title"],
    )["paperId"] == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert client.read_index_table("paper_ids").filter("corpusId = 1").count() == 2
    assert client.get_paper_by_id("1706.03762", fields=["title"])["title"] == "Attention Is All You Need"
    assert client.get_paper_by_id("CorpusId:1", fields=["abstract"])["abstract"] == "A transformer architecture."
    assert client.get_paper_by_id("CorpusId:1", fields=["openAccessPdf"])[
        "openAccessPdf"
    ]["url"] == "https://example.test/attention.pdf"
    assert client.get_author("a1", fields=["name"])["name"] == "Ashish Vaswani"
    assert client.get_references("CorpusId:1", fields=["citedPaper.title"])["data"][0]["citedPaper"]["title"] == "Biology of Neural Attention"
    assert client.search_papers("attention", fields=["title"], limit=1)["data"][0]["title"] == "Attention Is All You Need"


def test_validate_num_buckets_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        module.validate_num_buckets(0)


def test_load_raw_paper_ids_requires_dataset(spark_session, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="paper-ids dataset is required"):
        module.load_raw_paper_ids(spark_session, tmp_path)


def test_primary_paper_ids_reject_missing_mapping(spark_session) -> None:
    papers = spark_session.createDataFrame([{"corpusId": 1, "title": "Paper"}])
    paper_ids = spark_session.createDataFrame(
        [
            {
                "corpusId": 1,
                "primary": False,
                "sha": "dddddddddddddddddddddddddddddddddddddddd",
            }
        ]
    )

    with pytest.raises(ValueError, match="no valid primary SHA for corpusId 1"):
        module.add_primary_paper_ids(papers, paper_ids)


def test_primary_paper_ids_reject_conflicting_mapping(spark_session) -> None:
    papers = spark_session.createDataFrame([{"corpusId": 1, "title": "Paper"}])
    paper_ids = spark_session.createDataFrame(
        [
            {
                "corpusId": 1,
                "primary": True,
                "sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            },
            {
                "corpusId": 1,
                "primary": True,
                "sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            },
        ]
    )

    with pytest.raises(ValueError, match="multiple primary SHAs for corpusId 1"):
        module.add_primary_paper_ids(papers, paper_ids)


def test_primary_paper_ids_resolve_conflict_from_paper_url(spark_session) -> None:
    canonical_id = "14ebf26edcfba252e2325e698ce16441089ff386"
    conflicting_id = "af9195a849060d1543dab9bbe202ed40c12591b8"
    papers = spark_session.createDataFrame(
        [
            {
                "corpusId": 206478129,
                "title": "Paper",
                "url": f"https://www.semanticscholar.org/paper/{canonical_id}",
            }
        ]
    )
    paper_ids = spark_session.createDataFrame(
        [
            {
                "corpusId": 206478129,
                "primary": True,
                "sha": canonical_id,
            },
            {
                "corpusId": 206478129,
                "primary": True,
                "sha": conflicting_id,
            },
        ]
    )

    result = module.add_primary_paper_ids(papers, paper_ids).collect()

    assert result[0]["paperId"] == canonical_id


def test_primary_paper_ids_resolve_missing_primary_from_paper_url(
    spark_session,
) -> None:
    canonical_id = "00c248fd2cb23c18a2385c59f091416790f9c7c8"
    alternate_id = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    papers = spark_session.createDataFrame(
        [
            {
                "corpusId": 18083,
                "title": "Paper",
                "url": f"https://www.semanticscholar.org/paper/{canonical_id}",
            }
        ]
    )
    paper_ids = spark_session.createDataFrame(
        [
            {
                "corpusId": 18083,
                "primary": False,
                "sha": alternate_id,
            }
        ]
    )

    result = module.add_primary_paper_ids(papers, paper_ids).collect()

    assert result[0]["paperId"] == canonical_id


def test_primary_paper_ids_discard_non_positive_corpus_ids(spark_session) -> None:
    canonical_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    papers = spark_session.createDataFrame(
        [
            {"corpusId": 0, "title": "Invalid paper"},
            {"corpusId": 1, "title": "Valid paper"},
        ]
    )
    paper_ids = spark_session.createDataFrame(
        [
            {
                "corpusId": 1,
                "primary": True,
                "sha": canonical_id,
            }
        ]
    )

    result = module.add_primary_paper_ids(papers, paper_ids).collect()

    assert [(row["corpusId"], row["paperId"]) for row in result] == [
        (1, canonical_id)
    ]


def test_build_paper_ids_includes_url_fallback_canonical_id(
    spark_session,
    tmp_path: Path,
) -> None:
    canonical_id = "00c248fd2cb23c18a2385c59f091416790f9c7c8"
    alternate_id = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    papers = spark_session.createDataFrame(
        [{"corpusId": 18083, "paperId": canonical_id}]
    )
    paper_ids = spark_session.createDataFrame(
        [
            {"corpusId": 18083, "sha": alternate_id},
            {
                "corpusId": 0,
                "sha": "cccccccccccccccccccccccccccccccccccccccc",
            },
        ]
    )

    module.build_paper_ids_table(
        papers,
        paper_ids,
        tmp_path,
        num_buckets=8,
        overwrite=True,
    )
    rows = spark_session.read.parquet(str(tmp_path / "paper_ids")).collect()

    assert {(row["corpusId"], row["paperId"]) for row in rows} == {
        (18083, canonical_id),
        (18083, alternate_id),
    }


def test_write_table_groups_rows_before_partitioned_write(
    spark_session,
    tmp_path: Path,
) -> None:
    functions = module.import_pyspark_sql()[2]
    dataframe = spark_session.range(64, numPartitions=8).withColumn(
        "bucket",
        functions.pmod(functions.col("id"), functions.lit(2)),
    )
    output_path = tmp_path / "partitioned"

    module.write_table(dataframe, output_path, ["bucket"], overwrite=True)

    assert len(list(output_path.rglob("*.parquet"))) == 2


def test_validate_resume_metadata_rejects_incompatible_settings(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    index_root = tmp_path / "index"
    index_root.mkdir()
    metadata = {
        "indexVersion": module.INDEX_VERSION,
        "numBuckets": 8,
        "rawRoot": str(raw_root),
    }
    (index_root / "metadata.json").write_bytes(orjson.dumps(metadata))

    with pytest.raises(ValueError, match="different bucket count"):
        module.validate_resume_metadata(index_root, raw_root, 16)
