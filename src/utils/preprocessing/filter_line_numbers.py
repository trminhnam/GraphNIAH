"""
Filter out the line number column in the text.

python -m src.utils.preprocessing.filter_line_numbers
"""

import re
from collections import Counter


def filter_line_numbers(text: str) -> str:
    """
    Filter out the line number column in the text.

    Handles three types of line numbers from PDF parsing:
    1. Standalone line numbers on their own lines (e.g., "25", "100", "789")
    2. Trailing line numbers appended to content lines (e.g., "...quantum regimes [1]. Classical 27")
    3. Zero-padded, page-resetting line numbers (e.g., "000", "001", ..., "053", "000", ...)

    The function detects line numbers by:
    - First trying to find a monotonically increasing standalone sequence.
    - Falling back to a repeating page-based sequence (zero-padded numbers that
      reset each page, common in multi-column PDF exports).

    It preserves:
    - Numbers within actual content (dates, references, measurements)
    - Numbers that are part of sentences or paragraphs
    - Table numbers, section numbers, equation references
    - Markdown headers, list items, and other structural elements

    Args:
        text: The input text potentially containing line numbers

    Returns:
        Text with line numbers removed
    """
    lines = text.split("\n")

    line_number_set = _detect_line_number_sequence(lines)

    # For trailing number removal, use the full range of detected line numbers.
    line_number_range = (
        (min(line_number_set), max(line_number_set)) if line_number_set else None
    )

    cleaned_lines = []
    for line in lines:
        cleaned = _clean_line(line, line_number_set, line_number_range)
        if cleaned is not None:
            cleaned_lines.append(cleaned)

    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


def _detect_line_number_sequence(lines: list[str]) -> set[int]:
    """
    Detect the set of PDF line numbers present in the document.

    First attempts to find a monotonically increasing sequence (single-column
    PDFs with global line numbers). If that fails, falls back to detecting a
    repeating page-based sequence (zero-padded numbers that reset per page,
    e.g. 000-053 repeated across many pages).
    """
    standalone_numbers = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^\d+$", stripped):
            num = int(stripped)
            standalone_numbers.append(num)

    if len(standalone_numbers) < 5:
        return set()

    # Strategy 1: monotonically increasing global line numbers
    monotonic = _detect_monotonic_sequence(standalone_numbers)
    if monotonic:
        return monotonic

    # Strategy 2: repeating page-based line numbers (e.g. 000–053 per page)
    return _detect_repeating_sequence(standalone_numbers)


def _detect_monotonic_sequence(numbers: list[int]) -> set[int]:
    """
    Build the set of numbers that form monotonically increasing runs.
    Requires at least one run of length >= 5.
    """
    line_numbers: set[int] = set()
    current_run = [numbers[0]]

    for i in range(1, len(numbers)):
        prev = current_run[-1]
        curr = numbers[i]
        # Allow gaps up to 50 (some line numbers may be absent in the text)
        if curr > prev and (curr - prev) <= 50:
            current_run.append(curr)
        else:
            if len(current_run) >= 5:
                line_numbers.update(current_run)
            current_run = [curr]

    if len(current_run) >= 5:
        line_numbers.update(current_run)

    return line_numbers


def _detect_repeating_sequence(numbers: list[int]) -> set[int]:
    """
    Detect page-based line numbers that reset each page (e.g. 000–053 repeating).

    Heuristics:
    - The same integer values appear multiple times (repetition across pages).
    - The repeated values form a contiguous range starting at 0 (or very close).
    - At least 5 distinct values repeat, covering >= 80% of the expected range.

    If the pattern is confirmed, *all* standalone number lines are treated as
    line numbers (they all belong to some page's numbering column).
    """
    counts = Counter(numbers)

    # Values that appear more than once are candidates for repeating line numbers
    repeated = {n for n, c in counts.items() if c >= 2}
    if len(repeated) < 5:
        return set()

    min_val = min(repeated)
    max_val = max(repeated)

    # Page line numbers almost always start at 0 (or 1 at most)
    if min_val > 1:
        return set()

    # The repeated values should form a mostly contiguous range
    expected_range = set(range(min_val, max_val + 1))
    coverage = len(repeated & expected_range) / len(expected_range)
    if coverage < 0.8:
        return set()

    # All standalone numbers belong to the numbering column
    return set(numbers)


def _clean_line(
    line: str, line_number_set: set[int], line_number_range: tuple[int, int] | None
) -> str | None:
    """
    Clean a single line by removing detected line numbers.

    Returns None if the line is a standalone line number (to be removed entirely),
    or the cleaned line with trailing line numbers stripped.
    """
    stripped = line.strip()

    # Remove standalone line numbers
    if re.match(r"^\d{3,}$", stripped):
        return None
    if re.match(r"^\d+$", stripped) and int(stripped) in line_number_set:
        return None

    if _is_protected_line(stripped):
        return line

    # Remove trailing line numbers from content lines.
    match = re.search(r" (\d+)$", stripped)
    if match and line_number_range is not None:
        candidate = int(match.group(1))
        if line_number_range[0] <= candidate <= line_number_range[1]:
            cleaned = stripped[: match.start()]
            leading_space = line[: len(line) - len(line.lstrip())]
            return leading_space + cleaned

    return line


def _is_protected_line(line: str) -> bool:
    """Lines where trailing numbers should never be stripped."""
    if line.startswith("|") or line.startswith("|-"):
        return True
    if line.startswith("$$") or line.startswith("\\"):
        return True
    if not line.strip():
        return True
    if line.startswith("<!--"):
        return True
    return False


if __name__ == "__main__":
    import glob
    import os

    # parsed_pdfs_dir = "./output/parsed_pdfs/docling"
    parsed_pdfs_dir = "./output/parsed_references/docling"
    # print(f"Filtering {parsed_pdfs_dir}")
    for venue_folder in os.listdir(parsed_pdfs_dir):
        venue_path = os.path.join(parsed_pdfs_dir, venue_folder)
        if not os.path.isdir(venue_path):
            continue
        for pdf_path in glob.glob(os.path.join(venue_path, "*.md")):
            try:
                # print(f"Filtering {pdf_path}")
                with open(pdf_path, "r", encoding="utf-8") as f:
                    content = f.read()
                filtered_content = filter_line_numbers(content)
                if content != filtered_content:
                    with open(pdf_path, "w", encoding="utf-8") as f:
                        f.write(filtered_content)
                    print(f"Filtered {pdf_path}")
                else:
                    # print(f"No change {pdf_path}")
                    pass
            except Exception as e:
                print(f"Error filtering {pdf_path}: {e}")
