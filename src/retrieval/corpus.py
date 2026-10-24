"""On-disk article store for the DPR Wikipedia (psgs_w100) corpus.

Layout:

    paragraphs_dir/
        wiki_<title>.txt      # all TSV paragraphs for one title, concatenated
    metadata.jsonl            # JSONL sidecar with id, source_id, title, path

The TSV `text` column is double-quote-escaped CSV-style (e.g. embedded `""`),
so we parse with `csv.reader` rather than `str.split('\\t')`.
"""

from __future__ import annotations

import csv
import hashlib
import orjson
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Passages can be long enough that the default CSV field limit (~131k chars)
# is sometimes hit on pathological rows; raise it preemptively.
csv.field_size_limit(sys.maxsize)


@dataclass(frozen=True)
class ParagraphMetadata:
    id: int
    title: str | None
    path: Path
    source_id: int | None = None


@dataclass(frozen=True)
class Paragraph(ParagraphMetadata):
    text: str = ""


class ParagraphCorpus:
    """Owns the `paragraphs/` directory and a metadata JSONL sidecar.

    Three layers of access:

    - `path_for_id(pid)` — on-disk path for an indexed article.
    - `iter_metadata()` / `iter_paragraphs()` — stream written article files.
    - `read_paragraph(pid)` / `title_for_id(pid)` — random-access lookups
      (used by retrievers to populate `RetrievedItem.content` / `.title`).
    """

    def __init__(
        self,
        paragraphs_dir: str | Path,
        titles_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        max_articles: int | None = None,
    ):
        self.paragraphs_dir = Path(paragraphs_dir)
        if metadata_path is None:
            metadata_path = titles_path
        if metadata_path is None:
            metadata_path = self.paragraphs_dir.parent / "metadata.jsonl"
        self.metadata_path = Path(metadata_path)
        self.titles_path = self.metadata_path
        if max_articles is not None and max_articles < 0:
            raise ValueError("max_articles must be greater than or equal to 0")
        self.max_articles = max_articles
        self.titles_cache: dict[int, str | None] | None = None
        self.metadata_cache: dict[int, ParagraphMetadata] | None = None

    def path_for_id(self, paragraph_id: int) -> Path:
        metadata = self.metadata_for_id(paragraph_id)
        if metadata is not None:
            return metadata.path
        return self.fallback_path_for_id(paragraph_id)

    def fallback_path_for_id(self, paragraph_id: int) -> Path:
        return self.paragraphs_dir / f"wiki_{paragraph_id}.txt"

    def filename_for_title(self, title: str) -> str:
        safe_title = "".join(
            "_" if char in {"/", "\\", "\0"} or ord(char) < 32 else char
            for char in title
        ).strip()
        if not safe_title:
            safe_title = "untitled"
        if len(safe_title) > 180:
            digest = hashlib.sha1(safe_title.encode("utf-8")).hexdigest()[:10]
            safe_title = f"{safe_title[:160]}_{digest}"
        return safe_title

    def path_for_title(self, title: str) -> Path:
        return self.paragraphs_dir / f"wiki_{self.filename_for_title(title)}.txt"

    def unique_path_for_title(
        self,
        title: str,
        article_id: int,
        used_paths: set[Path],
    ) -> Path:
        path = self.path_for_title(title)
        if path not in used_paths:
            return path
        return path.with_name(f"{path.stem}_{article_id}{path.suffix}")

    def write_from_tsv(
        self,
        tsv_path: str | Path,
        progress_every: int = 100_000,
        max_rows: int | None = None,
        max_articles: int | None = None,
        skip_existing: bool = False,
    ) -> int:
        """Stream a DPR-style psgs_w100.tsv and write one wiki_<title>.txt.

        Header expected: `id\\ttext\\ttitle`. TSV rows are grouped by title, and
        each article file contains that title's paragraphs joined by blank
        lines. Returns the number of article metadata rows written. If
        `skip_existing=True`, article files whose .txt already exists are not
        re-written.
        """
        from tqdm.auto import tqdm

        article_limit = max_articles if max_articles is not None else self.max_articles
        if article_limit is not None and article_limit < 0:
            raise ValueError("max_articles must be greater than or equal to 0")

        self.paragraphs_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        article_records: dict[str, ParagraphMetadata] = {}
        current_source_id: int | None = None
        current_texts: list[str] = []
        current_title: str | None = None
        skipped_existing_paths: set[Path] = set()
        touched_paths: set[Path] = set()
        used_paths: set[Path] = set()
        n_articles = 0
        n_skipped_existing = 0
        n_malformed = 0
        n_paragraphs = 0
        reached_max_articles = False

        with (
            open(tsv_path, "r", encoding="utf-8", newline="") as src,
            open(self.metadata_path, "w", encoding="utf-8") as metadata_out,
        ):
            reader = csv.reader(src, delimiter="\t", quotechar='"')
            header = next(reader)
            if header != ["id", "text", "title"]:
                raise ValueError(
                    f"expected header [id, text, title], got {header}"
                )
            if article_limit == 0:
                print(
                    f"done: 0 articles from 0 paragraphs into {self.paragraphs_dir} "
                    "(skipped-existing=0, malformed=0)"
                )
                return 0

            def flush_article(article_progress) -> None:
                nonlocal n_articles
                nonlocal n_skipped_existing

                if current_title is None:
                    return

                record = article_records.get(current_title)
                if record is None:
                    article_id = len(article_records) + 1
                    out_path = self.unique_path_for_title(
                        current_title,
                        article_id,
                        used_paths,
                    )
                    record = ParagraphMetadata(
                        id=article_id,
                        title=current_title,
                        path=out_path,
                        source_id=current_source_id,
                    )
                    article_records[current_title] = record
                    used_paths.add(out_path)
                    metadata_out.write(
                        orjson.dumps({
                                "id": record.id,
                                "source_id": record.source_id,
                                "title": record.title,
                                "path": str(record.path.resolve()),
                            }).decode("utf-8")
                        + "\n"
                    )
                    n_articles += 1

                if (
                    skip_existing
                    and record.path.exists()
                    and record.path not in touched_paths
                ):
                    if record.path not in skipped_existing_paths:
                        skipped_existing_paths.add(record.path)
                        n_skipped_existing += 1
                    article_progress.update(1)
                    article_progress.set_postfix(
                        articles=n_articles,
                        skipped_existing=n_skipped_existing,
                    )
                    return

                article_text = "\n\n".join(current_texts)
                if record.path in touched_paths and record.path.exists():
                    with open(record.path, "a", encoding="utf-8") as out:
                        out.write("\n\n")
                        out.write(article_text)
                else:
                    record.path.write_text(article_text, encoding="utf-8")
                touched_paths.add(record.path)
                article_progress.update(1)
                article_progress.set_postfix(
                    articles=n_articles,
                    skipped_existing=n_skipped_existing,
                )

            with (
                tqdm(
                    total=max_rows,
                    desc="Reading TSV paragraphs",
                    unit="paragraph",
                ) as row_progress,
                tqdm(
                    desc="Saving article txt files",
                    unit="article",
                    total=article_limit,
                ) as article_progress,
            ):
                for row_index, row in enumerate(reader, start=1):
                    if len(row) != 3:
                        n_malformed += 1
                        continue
                    source_id_str, text, title = row
                    try:
                        source_id = int(source_id_str)
                    except ValueError:
                        n_malformed += 1
                        continue

                    if current_title is None:
                        current_title = title
                        current_source_id = source_id
                    elif title != current_title:
                        flush_article(article_progress)
                        if article_limit is not None and n_articles >= article_limit:
                            reached_max_articles = True
                            break
                        current_title = title
                        current_source_id = source_id
                        current_texts = []

                    current_texts.append(text)
                    n_paragraphs += 1
                    row_progress.update(1)

                    if max_rows is not None and n_paragraphs >= max_rows:
                        break
                    if progress_every and n_paragraphs % progress_every == 0:
                        row_progress.set_postfix(
                            articles=n_articles,
                            skipped_existing=n_skipped_existing,
                            malformed=n_malformed,
                        )
                if not reached_max_articles:
                    flush_article(article_progress)

        self.metadata_cache = None
        self.titles_cache = None
        print(
            f"done: {n_articles:,} articles from {n_paragraphs:,} paragraphs "
            f"into {self.paragraphs_dir} "
            f"(skipped-existing={n_skipped_existing}, malformed={n_malformed})"
        )
        return n_articles

    def iter_from_tsv(self, tsv_path: str | Path) -> Iterator[Paragraph]:
        """Stream paragraphs straight from the TSV without writing files."""
        from tqdm.auto import tqdm

        with open(tsv_path, "r", encoding="utf-8", newline="") as src:
            reader = csv.reader(src, delimiter="\t", quotechar='"')
            header = next(reader)
            if header != ["id", "text", "title"]:
                raise ValueError(
                    f"expected header [id, text, title], got {header}"
                )

            for row_index, row in enumerate(
                tqdm(
                    reader,
                    desc="Reading TSV paragraphs",
                    unit="paragraph",
                ),
                start=1,
            ):
                if len(row) != 3:
                    continue
                source_id_str, text, title = row
                try:
                    source_id = int(source_id_str)
                except ValueError:
                    continue
                yield Paragraph(
                    id=row_index,
                    title=title,
                    text=text,
                    path=self.path_for_id(row_index),
                    source_id=source_id,
                )

    def iter_metadata(self) -> Iterator[ParagraphMetadata]:
        """Stream metadata for written article files without reading text."""
        from tqdm.auto import tqdm

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                n_yielded = 0
                for line in tqdm(
                    f,
                    desc="Reading metadata",
                    unit="record",
                    leave=False,
                ):
                    if (
                        self.max_articles is not None
                        and n_yielded >= self.max_articles
                    ):
                        break
                    obj = orjson.loads(line)
                    yield self.metadata_from_obj(obj)
                    n_yielded += 1
            return

        paths = sorted(self.paragraphs_dir.glob("wiki_*.txt"))
        if self.max_articles is not None:
            paths = paths[: self.max_articles]
        for fallback_id, path in enumerate(
            tqdm(
                paths,
                total=len(paths),
                desc="Discovering wiki txt files",
                unit="file",
                leave=False,
            ),
            start=1,
        ):
            pid = self.paragraph_id_from_path(path) or fallback_id
            title = path.stem.removeprefix("wiki_")
            yield ParagraphMetadata(id=pid, title=title, path=path)

    def metadata_from_obj(self, obj: dict) -> ParagraphMetadata:
        pid = int(obj["id"])
        fallback_path = self.fallback_path_for_id(pid)
        path = Path(obj.get("path") or fallback_path)
        if not path.exists() and fallback_path.exists():
            path = fallback_path
        return ParagraphMetadata(
            id=pid,
            title=obj.get("title"),
            path=path,
            source_id=obj.get("source_id"),
        )

    def load_metadata(self) -> dict[int, ParagraphMetadata]:
        from tqdm.auto import tqdm

        cache: dict[int, ParagraphMetadata] = {}
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                n_loaded = 0
                for line in tqdm(
                    f,
                    desc="Loading metadata",
                    unit="record",
                ):
                    if (
                        self.max_articles is not None
                        and n_loaded >= self.max_articles
                    ):
                        break
                    metadata = self.metadata_from_obj(orjson.loads(line))
                    cache[metadata.id] = metadata
                    n_loaded += 1
        else:
            for metadata in tqdm(
                self.iter_metadata(),
                desc="Loading metadata from article paths",
                unit="record",
            ):
                cache[metadata.id] = metadata
        self.metadata_cache = cache
        return cache

    def metadata_for_id(self, paragraph_id: int) -> ParagraphMetadata | None:
        if self.metadata_cache is None:
            self.load_metadata()
        assert self.metadata_cache is not None
        return self.metadata_cache.get(paragraph_id)

    def iter_existing_metadata(self) -> Iterator[ParagraphMetadata]:
        from tqdm.auto import tqdm

        for metadata in tqdm(
            self.iter_metadata(),
            desc="Filtering existing metadata",
            unit="record",
            leave=False,
        ):
            if metadata.path.exists():
                yield metadata

    def iter_paragraphs(self) -> Iterator[Paragraph]:
        """Stream written articles from their .txt files."""
        from tqdm.auto import tqdm

        for metadata in tqdm(
            self.iter_existing_metadata(),
            desc="Reading article txt files",
            unit="article",
            leave=False,
        ):
            yield Paragraph(
                id=metadata.id,
                title=metadata.title,
                text=metadata.path.read_text(encoding="utf-8"),
                path=metadata.path,
                source_id=metadata.source_id,
            )

    def paragraph_id_from_path(self, path: str | Path) -> int | None:
        stem = Path(path).stem
        if not stem.startswith("wiki_"):
            return None
        try:
            return int(stem.removeprefix("wiki_"))
        except ValueError:
            return None

    def load_titles(self) -> dict[int, str | None]:
        """Load id → title mapping into memory."""
        from tqdm.auto import tqdm

        cache: dict[int, str | None] = {}
        if not self.metadata_path.exists():
            for paragraph_id, metadata in self.load_metadata().items():
                cache[paragraph_id] = metadata.title
            self.titles_cache = cache
            return cache
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            n_loaded = 0
            for line in tqdm(
                f,
                desc="Loading title metadata",
                unit="record",
            ):
                if (
                    self.max_articles is not None
                    and n_loaded >= self.max_articles
                ):
                    break
                obj = orjson.loads(line)
                cache[int(obj["id"])] = obj.get("title")
                n_loaded += 1
        self.titles_cache = cache
        return cache

    def title_for_id(self, paragraph_id: int) -> str | None:
        if self.titles_cache is None:
            self.load_titles()
        assert self.titles_cache is not None
        return self.titles_cache.get(paragraph_id)

    def read_paragraph(self, paragraph_id: int) -> str:
        return self.path_for_id(paragraph_id).read_text(encoding="utf-8")
