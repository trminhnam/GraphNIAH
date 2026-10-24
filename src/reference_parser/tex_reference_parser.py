"""
python -m src.reference_parser.tex_reference_parser <path_to_tex_dir>

tex_reference_parser.py

`TexReferenceParser.parse(src)` – single entry-point that accepts:
  • A directory  – walks recursively for .tex and .bib files, resolves only
                   entries that are actually \\cited in the manuscript.
  • A compressed archive  – extracts a ZIP/TAR source bundle to a temporary
                   directory, then parses it exactly like a source directory.
  • .tex file  – reads that file only; looks for .bib files in its parent
                   directory; resolves cited keys.
  • .bib/.bbl files  – compiled bibliography (``\\begin{thebibliography}`` block
                   produced by BibTeX/BibLaTeX); returns ALL entries directly,
                   no cite-key filtering needed.

Bibliography entries are resolved from TWO possible sources:
    a) .bib files parsed with bibtexparser
    b) \\begin{thebibliography} ... \\end{thebibliography} blocks, which covers
       both hand-written inline bibitems and pre-compiled .bbl files.

Each `Reference` maps to the shared base dataclass:
    title    – article / book title
    authors  – list of author strings
    venue    – journal, booktitle, conference, etc.
    year     – publication year
    raw      – original BibTeX / \\bibitem key

Two \\bibitem body formats are recognised automatically:
    • ``\\newblock``-separated (AAAI / author-year BST style, typical in .bbl)
    • Free-form prose (numbered styles, inline thebibliography in .tex)

Dependencies:
    pip install bibtexparser

Usage:
    from src.reference_parser import TexReferenceParser
    parser = TexReferenceParser()
    refs = parser.parse("/path/to/manuscript/")          # directory
    refs = parser.parse("/path/to/manuscript.tar.gz")    # archive
    refs = parser.parse("/path/to/manuscript/main.tex")  # single .tex
    refs = parser.parse("/path/to/manuscript/paper.bbl") # compiled .bbl
    for r in refs:
        print(r)
"""

from __future__ import annotations

import re
import tarfile
import tempfile
import zipfile
from pathlib import Path

try:
    import bibtexparser
    from bibtexparser.bparser import BibTexParser
    from bibtexparser.customization import convert_to_unicode
except ImportError:
    raise ImportError("Please install bibtexparser:  pip install bibtexparser")

from .reference_parser import Reference, ReferenceParser

# ── module-level helpers ────────────────────────────────────────────────────


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def is_archive(path: Path) -> bool:
    if not path.is_file():
        return False
    if zipfile.is_zipfile(path):
        return True
    if tarfile.is_tarfile(path):
        return True
    return False


def assert_safe_archive_members(member_names: list[str]) -> None:
    for member_name in member_names:
        member_path = Path(member_name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe path in archive: {member_name!r}")


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zip_file:
            assert_safe_archive_members(zip_file.namelist())
            zip_file.extractall(dest_dir)
        return

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tar_file:
            assert_safe_archive_members(
                [member.name for member in tar_file.getmembers()]
            )
            try:
                tar_file.extractall(dest_dir, filter="data")
            except TypeError:
                tar_file.extractall(dest_dir)
        return

    raise ValueError(
        f"{archive_path.name!r} is not a recognised ZIP or TAR archive."
    )


def _strip_comments(tex: str) -> str:
    """Remove LaTeX line comments (% …)."""
    return re.sub(r"(?m)(?<!\\)%.*$", "", tex)


def _clean(text: str) -> str:
    """Strip LaTeX braces and collapse whitespace."""
    text = re.sub(r"[{}]", "", text)
    return re.sub(r"\s+", " ", text).strip()


_CITE_RE = re.compile(
    r"\\[Cc]ite[a-zA-Z*]*" r"(?:\[[^\]]*\])*" r"\{([^}]+)\}",
    re.MULTILINE,
)

_VENUE_FIELDS = (
    "journal",
    "booktitle",
    "publisher",
    "school",
    "institution",
    "howpublished",
    "series",
)

_THEBIB_RE = re.compile(
    r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
    re.DOTALL,
)
_BIBITEM_RE = re.compile(
    r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}"
    r"(.*?)"
    r"(?=\\bibitem|\\end\{thebibliography\})",
    re.DOTALL,
)

_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_TITLE_QUOTED_RE = re.compile(r"[``\"](.*?)(?:''|\")", re.DOTALL)


class UnresolvedCitationKeyError(ValueError):
    """Raised when a cited LaTeX key cannot be resolved to a reference."""

    def __init__(self, missing_keys: list[str]) -> None:
        self.missing_keys = tuple(missing_keys)
        key_list = ", ".join(self.missing_keys)
        super().__init__(
            f"Could not resolve {len(self.missing_keys)} cited key(s): {key_list}"
        )


# ── parser class ────────────────────────────────────────────────────────────


class TexReferenceParser(ReferenceParser):
    """Parse cited references from a LaTeX manuscript source.

    Overrides ``parse`` to accept a source directory, compressed source
    archive, or single .tex/.bbl file instead of a plain text string.
    ``parse_one`` is intentionally unsupported: TeX reference parsing
    requires the complete manuscript source (to resolve \\cite keys against
    .bib files or \\begin{thebibliography} blocks) and cannot operate on a
    single isolated reference string.
    """

    # ── public API ────────────────────────────────────────────────────────

    def parse_one(self, text: str) -> Reference | None:  # type: ignore[override]
        raise NotImplementedError(
            "TexReferenceParser does not support parse_one(): TeX reference "
            "parsing requires the full manuscript source. Use parse(src) with "
            "a directory or a .tex file path instead."
        )

    def parse(self, src: str | Path) -> list[Reference]:  # type: ignore[override]
        """Parse cited references from a LaTeX source.

        Parameters
        ----------
        src : str | Path
            One of:

            * A **directory** – searched recursively for ``.tex`` and ``.bib``
              files; only entries actually cited (``\\cite{…}``) are returned.
            * A **compressed archive** – a ZIP/TAR LaTeX source bundle, which
              is extracted to a temporary directory and then parsed like a
              directory input.
            * A **``.tex`` file** – reads that file only; ``.bib`` files in
              its parent directory are also loaded; only cited entries returned.
            * A **``.bbl`` file** – compiled bibliography produced by
              BibTeX/BibLaTeX; *all* entries in the file are returned directly
              without cite-key filtering.

        Returns
        -------
        list[Reference]
            One ``Reference`` object per resolved entry.

        Raises
        ------
        ValueError
            If *src* is not a directory, a ZIP/TAR archive, a ``.tex`` file,
            or a ``.bbl`` file.
        NotImplementedError
            If ``parse_one`` is called directly.
        """
        src = Path(src).resolve()

        if src.is_dir():
            return self._parse_directory(src)

        if src.is_file() and is_archive(src):
            with tempfile.TemporaryDirectory(
                prefix="tex_reference_parser_"
            ) as temp_dir:
                extract_archive(src, Path(temp_dir))
                return self._parse_directory(Path(temp_dir))

        if src.is_file() and src.suffix == ".tex":
            return self._parse_single_file(src)

        if src.is_file() and src.suffix == ".bbl":
            return self._parse_bbl_file(src)

        raise ValueError(
            "src must be a directory, a ZIP/TAR archive, a .tex file, "
            f"or a .bbl file, got: {src}"
        )

    # ── private: per-mode entry points ────────────────────────────────────

    def _parse_directory(self, src_dir: Path) -> list[Reference]:
        tex_files = sorted(src_dir.rglob("*.tex"))
        if not tex_files:
            raise ValueError(f"No .tex files found under {src_dir}")

        combined_tex = "\n".join(_strip_comments(_read(f)) for f in tex_files)

        # .bbl files contain the compiled \begin{thebibliography} block;
        # include them so _parse_inline_bibitems can find the entries.
        bbl_files = sorted(src_dir.rglob("*.bbl"))
        bbl_entries: dict[str, Reference] = {}
        for bbl_file in bbl_files:
            bbl_entries.update(
                self._parse_inline_bibitems(_strip_comments(_read(bbl_file)))
            )

        # .bib entries take priority over .bbl entries on key collision.
        bib_entries = {**bbl_entries, **self._load_bib_files(src_dir)}
        return self._resolve(combined_tex, bib_entries)

    def _parse_single_file(self, tex_file: Path) -> list[Reference]:
        combined_tex = _strip_comments(_read(tex_file))
        bib_entries = self._load_bib_files(tex_file.parent)
        return self._resolve(combined_tex, bib_entries)

    def _parse_bbl_file(self, bbl_file: Path) -> list[Reference]:
        """Parse every entry from a compiled .bbl file.

        A .bbl file is a pre-formatted ``\\begin{thebibliography}`` block
        written by BibTeX.  It already contains all references for the
        manuscript, so no cite-key filtering is applied — every entry is
        returned as a ``Reference``.
        """
        content = _strip_comments(_read(bbl_file))
        entries = self._parse_inline_bibitems(content)
        print(
            f"[TexReferenceParser] {len(entries)} reference(s) parsed "
            f"from {bbl_file.name}."
        )
        return list(entries.values())

    # ── private: shared resolution logic ──────────────────────────────────

    def _resolve(self, combined_tex: str, bib_entries: dict) -> list[Reference]:
        cited_keys = self._extract_cited_keys(combined_tex)
        print(f"[TexReferenceParser] {len(cited_keys)} unique cited key(s) found.")

        bib_lookup: dict[str, Reference] = {}
        bib_lookup.update(self._parse_inline_bibitems(combined_tex))
        bib_lookup.update(bib_entries)  # .bib files win on key collision

        # Case-insensitive fallback: last writer wins when multiple bib keys
        # collapse to the same lowercase form.
        lower_bib_lookup: dict[str, Reference] = {
            k.lower(): v for k, v in bib_lookup.items()
        }

        matched = sum(1 for k in cited_keys if k in bib_lookup)
        matched_ci = sum(
            1
            for k in cited_keys
            if k not in bib_lookup and k.lower() in lower_bib_lookup
        )
        print(
            f"[TexReferenceParser] {len(bib_lookup)} total bib entries loaded; "
            f"{matched} match cited keys"
            + (f" ({matched_ci} via case-insensitive fallback)." if matched_ci else ".")
        )

        references: list[Reference] = []
        for k in cited_keys:
            if k in bib_lookup:
                references.append(bib_lookup[k])
            elif k.lower() in lower_bib_lookup:
                references.append(lower_bib_lookup[k.lower()])

        missing = sorted(
            k
            for k in cited_keys
            if k not in bib_lookup and k.lower() not in lower_bib_lookup
        )
        if missing:
            print(
                "[TexReferenceParser] Warning: "
                f"{len(missing)} cited key(s) could not be resolved:"
            )
            for k in missing:
                print(f"    - {k}")
            raise UnresolvedCitationKeyError(missing)

        return references

    # ── private: cite-key extraction ──────────────────────────────────────

    @staticmethod
    def _extract_cited_keys(combined_tex: str) -> set[str]:
        keys: set[str] = set()
        for m in _CITE_RE.finditer(combined_tex):
            for k in m.group(1).split(","):
                k = k.strip()
                if k:
                    keys.add(k)
        return keys

    # ── private: .bib file parsing ────────────────────────────────────────

    @staticmethod
    def _load_bib_files(search_dir: Path) -> dict[str, Reference]:
        """Return {key: Reference} for all entries found in .bib files under search_dir."""
        parser = BibTexParser(common_strings=True)
        parser.ignore_nonstandard_types = False
        parser.customization = convert_to_unicode
        entries: dict[str, Reference] = {}
        for bib_file in sorted(search_dir.rglob("*.bib")):
            with open(bib_file, encoding="utf-8", errors="replace") as fh:
                db = bibtexparser.load(fh, parser=parser)
            for e in db.entries:
                key = e.get("ID", "").strip()
                if not key:
                    continue
                raw_authors = e.get("author", "")
                authors = [
                    _clean(a) for a in re.split(r"\s+and\s+", raw_authors) if a.strip()
                ]
                entries[key] = Reference(
                    title=_clean(e.get("title", "")),
                    authors=authors,
                    venue=TexReferenceParser._venue_from_entry(e),
                    year=_clean(e.get("year", "")),
                    raw=key,
                )
        return entries

    @staticmethod
    def _venue_from_entry(entry: dict) -> str:
        for f in _VENUE_FIELDS:
            v = _clean(entry.get(f, ""))
            if v:
                return v
        if entry.get("archiveprefix"):
            prefix = _clean(entry["archiveprefix"])
            eprint = _clean(entry.get("eprint", ""))
            return f"{prefix}:{eprint}" if eprint else prefix
        return ""

    # ── private: inline \bibitem parsing ──────────────────────────────────

    @staticmethod
    def _parse_inline_bibitems(combined_tex: str) -> dict[str, Reference]:
        """Return {key: Reference} for \\bibitem entries embedded in .tex files."""
        entries: dict[str, Reference] = {}
        for bib_block in _THEBIB_RE.finditer(combined_tex):
            block_text = bib_block.group(0)
            for item in _BIBITEM_RE.finditer(block_text):
                key = item.group(1).strip()
                body = item.group(2)
                if key:
                    entries[key] = TexReferenceParser._heuristic_parse_bibitem(
                        key, body
                    )
        return entries

    @staticmethod
    def _heuristic_parse_bibitem(key: str, body: str) -> Reference:
        """Best-effort parse of a \\bibitem body.

        Delegates to ``_parse_newblock_bibitem`` when the body uses
        ``\\newblock`` separators (AAAI / author-year BST style, typical in
        ``.bbl`` files), otherwise falls back to sentence-level heuristics
        (numbered / prose styles).
        """
        if r"\newblock" in body:
            return TexReferenceParser._parse_newblock_bibitem(key, body)

        body = re.sub(r"\\(?:emph|textbf|textit|textrm|url)\{([^}]*)\}", r"\1", body)
        body = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^}]*)\}", r"\1", body)
        body = re.sub(r"\\[a-zA-Z]+\*?", " ", body)
        body = re.sub(r"\s+", " ", body).strip()

        year_match = _YEAR_RE.search(body)
        year = year_match.group(1) if year_match else ""

        title = ""
        tq = _TITLE_QUOTED_RE.search(body)
        if tq:
            title = _clean(tq.group(1))

        remainder = body
        if tq:
            remainder = body[: tq.start()] + " " + body[tq.end() :]
        remainder = re.sub(r"\s+", " ", remainder).strip()

        sentences = [s.strip() for s in re.split(r"\.\s+|\n", remainder) if s.strip()]

        authors_raw = ""
        for s in sentences:
            if year and year in s:
                authors_raw = s[: s.index(year)].strip().rstrip(",")
                break
            if s:
                authors_raw = s
                break

        author_parts = re.split(r"\s+and\s+|;\s*|\s*&\s*", authors_raw)
        authors = [_clean(a) for a in author_parts if _clean(a)]

        venue = ""
        if not title:
            if len(sentences) >= 3:
                title = _clean(sentences[1])
                venue = _clean(sentences[2])
            elif len(sentences) == 2:
                title = _clean(sentences[1])
        else:
            for s in sentences[1:]:
                candidate = _clean(s)
                if (
                    candidate
                    and candidate != title
                    and (not year or year not in candidate)
                ):
                    venue = candidate
                    break

        return Reference(
            title=title,
            authors=authors,
            venue=venue,
            year=year,
            raw=key,
        )

    @staticmethod
    def _parse_newblock_bibitem(key: str, body: str) -> Reference:
        r"""Parse a \bibitem body that uses \newblock field separators.

        This format is produced by BibTeX with author-year style files (e.g.
        AAAI, ACL, named.bst) and is the standard layout in compiled .bbl
        files::

            Author, F.; Co-Author, G.; and Third, H.
            \newblock 2024.
            \newblock Title of the paper.
            \newblock In {\em Conference Name}.

        Field order is positional:
            segment 0  – author list (semicolon / "and" separated)
            segment 1  – year  (four digits, optionally followed by a letter)
            segment 2  – title
            segment 3+ – venue  (``In {\em …}`` or ``{\em journal} vol:pages``)

        IMPORTANT: ``\newblock`` is split out BEFORE any macro expansion so
        that the bare-macro removal step cannot swallow the delimiters.
        """

        def _clean_seg(s: str) -> str:
            """Expand LaTeX macros and strip braces from a single segment."""
            s = re.sub(r"\\(?:emph|textbf|textit|textrm|url)\{([^}]*)\}", r"\1", s)
            s = re.sub(r"\{\\em\s+([^}]*)\}", r"\1", s)
            s = re.sub(r"\\(?:protect|bgroup|egroup)\b", "", s)
            s = re.sub(r"\\mbox\{([^}]*)\}", r"\1", s)
            s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^}]*)\}", r"\1", s)
            s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
            s = s.replace("~", " ")
            return _clean(s)

        # Split on \newblock FIRST to preserve the delimiters during cleanup
        raw_segments = re.split(r"\\newblock\s*", body)
        segments = [_clean_seg(s).rstrip(".") for s in raw_segments]
        segments = [s for s in segments if s]

        # ── segment 0: authors ─────────────────────────────────────────────
        authors_raw = segments[0] if segments else ""
        author_parts = re.split(r"\s+and\s+|;\s*", authors_raw)
        authors = [a.strip() for a in author_parts if a.strip()]

        year = ""
        title = ""
        venue = ""

        # ── segments 1+: year → title → venue (positional) ────────────────
        non_year_count = 0
        for seg in segments[1:]:
            if not seg:
                continue

            # Year segment: four digits optionally followed by a single letter
            if re.match(r"^(1[0-9]{3}|20[0-9]{2})[a-z]?\s*$", seg):
                if not year:
                    year = re.match(r"(1[0-9]{3}|20[0-9]{2})", seg).group(1)
                continue

            # Bare page-range segment (e.g. "2787--2795"); skip it
            if re.match(r"^\d+[-–]+\d+$", seg):
                continue

            non_year_count += 1
            if non_year_count == 1:
                title = seg
            elif non_year_count == 2:
                # Strip leading "In " then isolate core venue from editor noise
                venue = re.sub(r"^[Ii]n\s+", "", seg)
                if "eds" in venue:
                    after_eds = re.search(r"eds\.\s*,?\s*(.*)", venue)
                    if after_eds:
                        venue = after_eds.group(1).strip()
                break

        # Fallback: extract year from full body if not found positionally
        if not year:
            ym = _YEAR_RE.search(body)
            if ym:
                year = ym.group(1)

        return Reference(
            title=title,
            authors=authors,
            venue=venue,
            year=year,
            raw=key,
        )


# ── CLI convenience ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # import argparse
    # import orjson
    # import sys

    # ap = argparse.ArgumentParser(
    #     description="Extract cited references from a LaTeX source directory or file."
    # )
    # ap.add_argument("src", help="Path to the LaTeX source directory or .tex file")
    # ap.add_argument("--json", action="store_true", help="Dump results as JSON")
    # args = ap.parse_args()

    # parser = TexReferenceParser()
    # refs = parser.parse(args.src)

    # if args.json:
    #     print(orjson.dumps([r.to_dict() for r in refs], option=orjson.OPT_INDENT_2).decode("utf-8"))
    # else:
    #     for i, r in enumerate(refs, 1):
    #         print(f"\n[{i}] {r.raw}")
    #         print(f"  Title  : {r.title}")
    #         print(f"  Authors: {'; '.join(r.authors)}")
    #         print(f"  Venue  : {r.venue}")
    #         print(f"  Year   : {r.year}")

    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "src",
        type=str,
        help="Path to the directory containing different directories of the manuscripts",
    )
    args = parser.parse_args()

    parser = TexReferenceParser()

    for folder in os.listdir(args.src):
        if not os.path.isdir(os.path.join(args.src, folder)):
            continue

        refs = parser.parse(os.path.join(args.src, folder))
        print(f"Parsed {len(refs)} references from {folder}")
        for r in refs:
            print(r)

    sys.exit(0)
