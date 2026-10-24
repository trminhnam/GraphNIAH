"""
latex_to_pdf.py
===============
Compile a LaTeX project (directory or archive) to PDF.

Example commands:
```bash
python -m src.utils.preprocessing.build_latex_to_pdf ./resources/tex_source/arXiv-1412.6544v6.tar.gz  compiled.pdf

python -m src.utils.preprocessing.build_latex_to_pdf ./resources/tex_source/arXiv-2505.07843v2.tar.gz  compiled.pdf

python -m src.utils.preprocessing.build_latex_to_pdf ./resources/tex_source/arXiv-2507.05568v1.tar.gz  compiled.pdf
```



Behaviour overview
------------------
- Accepts a **source directory** or a **compressed archive**
  (.zip, .tar, .tar.gz, .tar.bz2, .tar.xz, .tgz, .tbz2).
  Archives are extracted to a temporary directory and cleaned up automatically.

- Discovers **every root .tex file** (any file containing \\documentclass),
  so projects with multiple root files (e.g. main paper + supplement) are all
  compiled.

- Each candidate is compiled in an **isolated temporary copy** of the source
  tree — the originals are never modified.

- Runs the LaTeX engine up to `max_runs` times per candidate and inserts a
  bibtex/biber pass after the first run when a bibliography is detected.

- Selects the PDF with the **most pages** as the winner and copies it to
  `output_path`.

Usage (library)
---------------
    from latex_to_pdf import build_latex_to_pdf

    build_latex_to_pdf("paper/", "out/paper.pdf")
    build_latex_to_pdf("submission.zip", "out/paper.pdf", engine="xelatex")

Usage (CLI)
-----------
    python latex_to_pdf.py paper/ out/paper.pdf
    python latex_to_pdf.py submission.zip out/paper.pdf --engine xelatex --log-level DEBUG
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

from loguru import logger as log

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LATEX_ENGINES = ["pdflatex", "xelatex", "lualatex"]

# .tex filenames that are almost always the project root
PRIORITY_TEX_NAMES = {"main.tex", "paper.tex", "index.tex"}

# ============================================================================
# Data classes
# ============================================================================


@dataclass(order=True)
class CompiledPDF:
    """Holds the result of compiling one root .tex file."""

    pages: int
    path: Path = field(compare=False)
    source_tex: Path = field(compare=False)

    def __str__(self) -> str:
        return f"{self.source_tex.name} -> {self.pages} page(s)"


# ============================================================================
# Archive utilities
# ============================================================================


def is_archive(path: Path) -> bool:
    """
    Return True if *path* is a supported archive, detected by magic bytes.

    Magic-byte sniffing is used instead of extension matching so that
    arXiv-style filenames like ``paper_tar.gz`` (underscore, not dot) are
    handled correctly alongside the standard ``.tar.gz`` convention.
    """
    if not path.is_file():
        return False
    if zipfile.is_zipfile(path):
        return True
    if tarfile.is_tarfile(path):
        return True
    return False


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """
    Extract *archive_path* into *dest_dir*.

    Format is detected by magic bytes, not file extension, so non-standard
    names (e.g. arXiv's ``paper_tar.gz``) are handled transparently.

    Raises
    ------
    ValueError
        If the archive contains unsafe paths (absolute or path-traversal) or
        if the format is not recognised.
    """
    if zipfile.is_zipfile(archive_path):
        log.debug("Extracting ZIP archive: {}", archive_path.name)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)

    elif tarfile.is_tarfile(archive_path):
        log.debug("Extracting TAR archive: {}", archive_path.name)
        with tarfile.open(archive_path) as tf:
            _assert_safe_tar_members(tf)
            try:
                tf.extractall(dest_dir, filter="data")
            except TypeError:
                # Python < 3.12 does not support the `filter` keyword
                tf.extractall(dest_dir)
    else:
        raise ValueError(
            f"{archive_path.name!r} is not a recognised ZIP or TAR archive."
        )


def _assert_safe_tar_members(tf: tarfile.TarFile) -> None:
    """Raise ValueError if any TAR member has an unsafe path."""
    for member in tf.getmembers():
        member_path = Path(member.name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe path in archive: {member.name!r}")


def unwrap_single_subdir(directory: Path) -> Path:
    """
    If *directory* contains exactly one child directory, return that child.

    Many archives wrap their content in a top-level folder
    (e.g. ``paper-v2/paper-v2/main.tex``). This unwraps that layer.
    """
    entries = list(directory.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        log.debug("Unwrapping single subdirectory: {}", entries[0].name)
        return entries[0]
    return directory


@contextmanager
def extracted_source(source: Path) -> Generator[Path, None, None]:
    """
    Context manager that yields the resolved LaTeX source directory.

    If *source* is a directory, yields it directly (no cleanup needed).
    If *source* is a recognised archive, extracts it to a temp dir and cleans
    up on exit.  Raises ValueError for any other file type.
    """
    if source.is_dir():
        yield source
        return

    if not source.is_file():
        raise ValueError(f"Source path does not exist or is not a file/dir: {source}")

    if not is_archive(source):
        raise ValueError(
            f"{source.name!r} is not a directory or a recognised archive "
            f"(ZIP / TAR). Check the path and file format."
        )

    tmp = Path(tempfile.mkdtemp(prefix="latex_archive_"))
    log.info("Extracting archive {} -> {}", source.name, tmp)
    try:
        extract_archive(source, tmp)
        yield unwrap_single_subdir(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        log.debug("Removed archive temp dir: {}", tmp)


# ============================================================================
# Root .tex file discovery
# ============================================================================


def find_root_tex_files(directory: Path) -> list[Path]:
    """
    Return every .tex file in *directory* (recursively) that contains
    a ``\\documentclass`` declaration.

    Ordering heuristic (most-likely root first):
    1. Files whose names are in ``PRIORITY_TEX_NAMES`` (main.tex, paper.tex, ...).
    2. All remaining matches, sorted by descending line count.
    """
    priority: list[Path] = []
    rest: list[tuple[int, Path]] = []  # (line_count, path)

    for tex_path in sorted(directory.rglob("*.tex")):
        text = _try_read_text(tex_path)
        if text is None:
            continue

        if not re.search(r"\\documentclass", text):
            log.debug("Skipping (no \\documentclass): {}", tex_path.name)
            continue

        if tex_path.name in PRIORITY_TEX_NAMES:
            log.debug("Priority root file: {}", tex_path.relative_to(directory))
            priority.append(tex_path)
        else:
            line_count = text.count("\n")
            log.debug(
                "Root file: {}  ({} lines)",
                tex_path.relative_to(directory),
                line_count,
            )
            rest.append((line_count, tex_path))

    rest.sort(key=lambda pair: pair[0], reverse=True)
    return priority + [path for _, path in rest]


def _try_read_text(path: Path) -> Optional[str]:
    """Read *path* as UTF-8 text; return None and log a warning on failure."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.warning("Cannot read {}: {}", path, exc)
        return None


# ============================================================================
# PDF utilities
# ============================================================================


def count_pdf_pages(pdf_path: Path) -> int:
    """
    Return the number of pages in *pdf_path*.

    Tries three strategies in order:
    1. ``pdfinfo`` (poppler-utils) -- fastest.
    2. ``pypdf`` -- pure-Python fallback.
    3. File size -- rough proxy of last resort (returns bytes, not pages).
    """
    count = _page_count_via_pdfinfo(pdf_path)
    if count is not None:
        return count

    count = _page_count_via_pypdf(pdf_path)
    if count is not None:
        return count

    log.warning(
        "Could not determine page count for {}; falling back to file size.",
        pdf_path.name,
    )
    return pdf_path.stat().st_size  # rough proxy


def _page_count_via_pdfinfo(pdf_path: Path) -> Optional[int]:
    try:
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        for line in result.stdout.splitlines():
            if line.lower().startswith("pages:"):
                return int(line.split(":")[1].strip())
    except Exception as exc:
        log.debug("pdfinfo unavailable for {}: {}", pdf_path.name, exc)
    return None


def _page_count_via_pypdf(pdf_path: Path) -> Optional[int]:
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(pdf_path)).pages)
    except Exception as exc:
        log.debug("pypdf unavailable for {}: {}", pdf_path.name, exc)
    return None


# ============================================================================
# LaTeX engine helpers
# ============================================================================


def find_latex_engine() -> Optional[str]:
    """Return the first available LaTeX engine from ``LATEX_ENGINES``, or None."""
    for engine in LATEX_ENGINES:
        if shutil.which(engine):
            log.debug("LaTeX engine found: {}", engine)
            return engine
    return None


def run_bibliography_if_needed(work_dir: Path, jobname: str) -> None:
    """
    Run bibtex or biber only when a real .bib database is present.

    Skips the run — and preserves any existing .bbl — when the project ships
    a pre-compiled .bbl without a matching .bib (the arXiv norm).
    Running bibtex without a .bib would overwrite the good .bbl with an empty
    one, replacing every citation with a "?" placeholder.

    bibtex is tried first; biber is the fallback.
    """
    aux_path = work_dir / f"{jobname}.aux"
    if not aux_path.exists():
        return

    aux_text = _try_read_text(aux_path) or ""
    if "\\bibdata" not in aux_text:
        return

    # Extract every database name declared via \bibdata{name1,name2,...}
    bib_names = re.findall(r"\\bibdata\{([^}]+)\}", aux_text)
    bib_files: list[str] = []
    for entry in bib_names:
        bib_files.extend(name.strip() for name in entry.split(","))

    # Resolve each name to an actual .bib file on disk
    def _find_bib(name: str) -> bool:
        stem = name if name.endswith(".bib") else f"{name}.bib"
        return (work_dir / stem).exists()

    found_bib_files = [name for name in bib_files if _find_bib(name)]

    if not found_bib_files:
        bbl_path = work_dir / f"{jobname}.bbl"
        if bbl_path.exists():
            bbl_lines = bbl_path.read_text(encoding="utf-8", errors="replace").count(
                "\n"
            )
        log.debug(
            "Skipping bibliography tool: no .bib file found, "
            "using pre-compiled {}.bbl ({} lines).",
            jobname,
            bbl_lines,
        )
    else:
        log.warning(
            "No .bib file found for {!r} and no pre-compiled .bbl either; "
            "references will be unresolved.",
            jobname,
        )
        return

    for bib_tool in ("bibtex", "biber"):
        if not shutil.which(bib_tool):
            continue
        log.debug(
            "Running {} for job {!r} (found: {})", bib_tool, jobname, found_bib_files
        )
        bbl_path = work_dir / f"{jobname}.bbl"
        bbl_backup = bbl_path.read_bytes() if bbl_path.exists() else None
        try:
            result = subprocess.run(
                [bib_tool, jobname],
                cwd=work_dir,
                capture_output=True,
                timeout=120,
            )
            if result.returncode != 0:
                log.warning(
                    "{} exited with code {} for job {!r}; "
                    "restoring original .bbl if present.",
                    bib_tool,
                    result.returncode,
                    jobname,
                )
                if bbl_backup is not None:
                    bbl_path.write_bytes(bbl_backup)
        except subprocess.TimeoutExpired:
            log.warning("{} timed out for job {!r}", bib_tool, jobname)
            if bbl_backup is not None:
                bbl_path.write_bytes(bbl_backup)
        except OSError as exc:
            log.warning("{} could not be launched: {}", bib_tool, exc)
        break  # only run one bibliography tool


# ============================================================================
# Compilation
# ============================================================================


def compile_tex(
    tex_path: Path,
    source_dir: Path,
    engine: str,
    max_runs: int = 3,
) -> Optional[CompiledPDF]:
    """
    Compile *tex_path* in an isolated copy of *source_dir*.

    Parameters
    ----------
    tex_path:   Absolute path to the root .tex file inside *source_dir*.
    source_dir: Root of the LaTeX project (mirrored for each compilation).
    engine:     LaTeX engine binary name (e.g. ``"pdflatex"``).
    max_runs:   Maximum number of engine passes (default 3).

    Returns
    -------
    CompiledPDF on success, None on failure.
    """
    rel = tex_path.relative_to(source_dir)
    log.info("Compiling {}  [engine={}, max_runs={}]", rel, engine, max_runs)

    work_root = Path(tempfile.mkdtemp(prefix="latex_build_"))
    work_dir = work_root / "src"

    try:
        shutil.copytree(source_dir, work_dir)
        log.debug("Copied source tree -> {}", work_dir)

        work_tex = work_dir / rel
        jobname = work_tex.stem

        pdf_path = _run_engine_passes(work_tex, work_dir, engine, jobname, max_runs)
        if pdf_path is None:
            return None

        # Lift the PDF above the (soon-to-be-deleted) source copy
        result_pdf = work_root / pdf_path.name
        shutil.copy2(pdf_path, result_pdf)

        pages = count_pdf_pages(result_pdf)
        log.info("  Compiled {} -> {} page(s)", rel, pages)
        return CompiledPDF(pages=pages, path=result_pdf, source_tex=tex_path)

    except Exception as exc:
        log.error("Unexpected error while compiling {}: {}", rel, exc)
        return None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        log.debug("Removed build work dir: {}", work_dir)


def _run_engine_passes(
    work_tex: Path,
    work_dir: Path,
    engine: str,
    jobname: str,
    max_runs: int,
) -> Optional[Path]:
    """
    Execute the LaTeX engine up to *max_runs* times.

    A bibliography pass is inserted after run 1 when a .bib reference is
    detected.  Returns the path to the produced PDF, or None.
    """
    cmd = [
        engine,
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={work_dir}",
        str(work_tex),
    ]
    env = {**os.environ, "TEXMFOUTPUT": str(work_dir)}

    for run_number in range(1, max_runs + 1):
        log.debug("  Engine pass {}/{}", run_number, max_runs)
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                timeout=300,
                env=env,
                # Capture as bytes: LaTeX/BibTeX output may contain Latin-1
                # characters (e.g. accented author names in .bst/.bib files)
                # that are not valid UTF-8.  We decode manually with
                # errors="replace" so a single bad byte never crashes the run.
            )
        except subprocess.TimeoutExpired:
            log.warning("Engine timed out on pass {} for {}", run_number, work_tex.name)
            return None

        _log_engine_result(result, run_number, work_tex.name)

        if run_number == 1:
            run_bibliography_if_needed(work_dir, jobname)

    return _locate_output_pdf(work_dir, work_tex, jobname)


def _log_engine_result(
    result: subprocess.CompletedProcess, run: int, filename: str
) -> None:
    """Emit engine stdout/stderr at the appropriate log level.

    *result.stdout* is raw bytes (no ``text=True`` on the subprocess call),
    decoded here with ``errors="replace"`` to avoid crashes on Latin-1 output
    from .bst/.bib files with accented author names.
    """
    stdout = (result.stdout or b"").decode("utf-8", errors="replace")

    if result.returncode == 0:
        log.debug("  Pass {} succeeded ({})", run, filename)
        return

    log.debug("  Pass {} exit={} ({})", run, result.returncode, filename)
    # Surface LaTeX ! Error lines for easier debugging
    for line in stdout.splitlines():
        if line.startswith("!") or "Error" in line:
            log.debug("    LaTeX: {}", line)


def _locate_output_pdf(work_dir: Path, work_tex: Path, jobname: str) -> Optional[Path]:
    """
    Find the compiled PDF.

    pdflatex writes to *work_dir*; xelatex / lualatex may write next to the
    .tex file when it lives in a subdirectory.
    """
    candidates = [
        work_dir / f"{jobname}.pdf",
        work_tex.parent / f"{jobname}.pdf",
    ]
    for candidate in candidates:
        if candidate.exists():
            log.debug("  Output PDF: {}", candidate)
            return candidate

    log.warning("No PDF produced for {}", work_tex.name)
    return None


# ============================================================================
# Public API
# ============================================================================


def build_latex_to_pdf(
    source_dir_or_path: str | Path,
    output_path: str | Path,
    engine: Optional[str] = None,
    max_compile_runs: int = 3,
    log_level: str = "INFO",
) -> bool:
    """
    Compile a LaTeX project to PDF and write the best result to *output_path*.

    Parameters
    ----------
    source_dir_or_path:
        A directory containing ``.tex`` files **or** a compressed archive
        (``.zip``, ``.tar.gz``, ``.tgz``, ``.tar.bz2``, ``.tar.xz``, ...).
    output_path:
        Destination path for the winning PDF.
    engine:
        LaTeX engine binary name.  When *None*, the first available engine
        from ``LATEX_ENGINES`` is used automatically.
    max_compile_runs:
        Number of engine passes per candidate (default 3).
    log_level:
        Logging verbosity for this run.  Accepts any standard level name:
        ``"DEBUG"``, ``"INFO"`` (default), ``"WARNING"``, ``"ERROR"``,
        ``"CRITICAL"``.

        - DEBUG    – engine output, per-file decisions, temp paths
        - INFO     – progress milestones and final result
        - WARNING  – non-fatal issues (unreadable files, failed candidates)
        - ERROR    – fatal problems (no engine, no root files, no PDF)
        - CRITICAL – reserved for unrecoverable internal errors

    Returns
    -------
    bool
        ``True`` if at least one PDF was produced and saved to *output_path*.
    """
    num_trials = 3

    while num_trials > 0:
        try:

            source = Path(source_dir_or_path)
            output = Path(output_path)

            chosen_engine = _resolve_engine(engine)
            if chosen_engine is None:
                return False

            with extracted_source(source) as source_dir:
                root_files = find_root_tex_files(source_dir)
                if not root_files:
                    log.error(
                        "No root .tex files (with \\documentclass) found in {}",
                        source_dir,
                    )
                    return False

                log.info(
                    "Found {} root .tex file(s): {}",
                    len(root_files),
                    ", ".join(str(f.relative_to(source_dir)) for f in root_files),
                )

                compiled = _compile_all(
                    root_files, source_dir, chosen_engine, max_compile_runs
                )

                return _save_best_pdf(compiled, output)

        except Exception as e:
            log.error(
                f"Error compiling LaTeX: {e}. Retrying... (num_trials left: {num_trials})"
            )
            num_trials -= 1
            if num_trials == 0:
                return False
            continue

    return False


# ---------------------------------------------------------------------------
# Private helpers for build_latex_to_pdf
# ---------------------------------------------------------------------------


def _resolve_engine(requested: Optional[str]) -> Optional[str]:
    """Validate the requested engine or auto-detect one; return None on failure."""
    if requested:
        if not shutil.which(requested):
            log.error("Requested engine {!r} not found on PATH.", requested)
            return None
        log.info("Using LaTeX engine: {}", requested)
        return requested

    detected = find_latex_engine()
    if detected is None:
        log.error(
            "No LaTeX engine found on PATH. "
            "Install TeX Live and ensure one of {} is available.",
            LATEX_ENGINES,
        )
        return None

    log.info("Using LaTeX engine: {} (auto-detected)", detected)
    return detected


def _compile_all(
    root_files: list[Path],
    source_dir: Path,
    engine: str,
    max_runs: int,
) -> list[CompiledPDF]:
    """Compile every root file and return the successful results."""
    results: list[CompiledPDF] = []

    for tex_path in root_files:
        result = compile_tex(tex_path, source_dir, engine, max_runs)
        if result is not None:
            results.append(result)
        else:
            log.warning(
                "Compilation produced no PDF: {}", tex_path.relative_to(source_dir)
            )

    return results


def _save_best_pdf(candidates: list[CompiledPDF], output: Path) -> bool:
    """Copy the PDF with the most pages to *output*; return True on success."""
    if not candidates:
        log.error("No PDF was successfully compiled.")
        return False

    best = max(candidates)  # CompiledPDF is ordered by .pages
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best.path, output)

    log.info(
        "Winner: {}  ({} pages) -> {}",
        best.source_tex.name,
        best.pages,
        output,
    )

    if len(candidates) > 1:
        ranking = "\n".join(
            f"  {'*' if c is best else ' '} {c}"
            for c in sorted(candidates, reverse=True)
        )
        log.debug("All compiled PDFs (best first):\n{}", ranking)

    for candidate in candidates:
        shutil.rmtree(candidate.path.parent, ignore_errors=True)

    return True


# ============================================================================
# CLI
# ============================================================================


def _build_argument_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="latex_to_pdf",
        description=(
            "Compile a LaTeX project to PDF.\n\n"
            "Every root .tex file (containing \\documentclass) is compiled.\n"
            "The PDF with the most pages is written to OUTPUT."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "log levels:\n"
            "  DEBUG    engine output, file decisions, temp paths\n"
            "  INFO     progress milestones and final result  [default]\n"
            "  WARNING  non-fatal issues (unreadable files, failed candidates)\n"
            "  ERROR    fatal problems (no engine, no root files, no PDF)\n"
            "  CRITICAL reserved for unrecoverable internal errors\n\n"
            "examples:\n"
            "  %(prog)s paper/ out/paper.pdf\n"
            "  %(prog)s submission.zip out/paper.pdf --engine xelatex\n"
            "  %(prog)s paper/ out/paper.pdf --log-level DEBUG\n"
            "  %(prog)s paper/ out/paper.pdf --log-level WARNING\n"
        ),
    )

    parser.add_argument(
        "source",
        help=(
            "LaTeX source directory or compressed archive "
            "(.zip / .tar / .tar.gz / .tgz / .tar.bz2 / .tar.xz)."
        ),
    )
    parser.add_argument(
        "output",
        help="Destination path for the output PDF.",
    )
    parser.add_argument(
        "--engine",
        choices=LATEX_ENGINES,
        default=None,
        metavar="ENGINE",
        help=f"LaTeX engine to use. Choices: {', '.join(LATEX_ENGINES)}. "
        "Default: first available engine found on PATH.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        metavar="N",
        help="Number of LaTeX engine passes per candidate (default: 3).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        metavar="LEVEL",
        help="Logging verbosity (default: INFO). See epilog for descriptions.",
    )

    return parser


def main() -> None:
    import sys

    from src.logging import configure_logging

    parser = _build_argument_parser()
    args = parser.parse_args()
    configure_logging(level=args.log_level)

    ok = build_latex_to_pdf(
        source_dir_or_path=args.source,
        output_path=args.output,
        engine=args.engine,
        max_compile_runs=args.runs,
        log_level=args.log_level,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
