from __future__ import annotations

import hashlib
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import orjson
import pandas as pd


def resolve_project_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents, Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        if (candidate / "resources" / "llm_models.yaml").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root from the current directory.")


def ensure_project_root_on_sys_path() -> Path:
    project_root = resolve_project_root()
    project_root_text = str(project_root)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)
    return project_root


def load_json(path: Path) -> dict:
    return orjson.loads(path.read_text(encoding="utf-8"))


def json_safe(value: object) -> object:
    """Recursively replace non-finite floats with ``None`` for JSON serialization.

    ``orjson`` rejects NaN/Infinity outright, but derived metrics (e.g. a ratio
    with a zero denominator) can legitimately produce them.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def stable_content_hash(payload: object) -> str:
    """Return a stable hex hash of a JSON-serializable structure.

    Two calls with equal ``payload`` values hash equal regardless of dict key
    order, so this can key a disk cache (e.g. "has this question's content
    changed since it was last tokenized?") without spurious invalidation from
    re-serializing the same underlying data.
    """
    canonical = orjson.dumps(json_safe(payload), option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(canonical).hexdigest()


def read_json_cache(path: Path) -> dict | None:
    """Return a JSON cache document, or ``None`` if missing, empty, or corrupt.

    A cache is a best-effort speedup, never a correctness requirement, so any
    read failure (partial write, hand-edited file, ...) is treated the same as
    "no cache yet" rather than raised.
    """
    if not path.exists():
        return None
    try:
        return load_json(path)
    except (OSError, ValueError):
        return None


def write_json_cache(path: Path, data: dict) -> None:
    """Write ``data`` as a JSON cache document, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(json_safe(data)))


def format_mean_std(values: pd.Series) -> str:
    """Return numeric values formatted as mean plus/minus sample std."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return "N/A"
    standard_deviation = numeric.std(ddof=1) if len(numeric) > 1 else 0.0
    return f"{numeric.mean():.2f} += {standard_deviation:.2f}"


def count_text_tokens(tokenizer: Any, text: str) -> int:
    """Count one text fragment with a prompt-packing tokenizer."""
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            return len(encode(text, add_special_tokens=False))
        except TypeError:
            return len(encode(text))

    try:
        encoded = tokenizer(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer(text)
    if isinstance(encoded, dict):
        return len(encoded["input_ids"])
    return len(encoded.input_ids)


def summarize_prompt_fill_payload(prompts_path: Path) -> dict[str, object] | None:
    """Return context-fill statistics for one bucket's assembled prompts.

    Prefers the ``prompt_fill`` block that ``construct_full_prompts`` records;
    falls back to recomputing from the rendered prompts for older payloads.
    """
    payload = load_json(prompts_path)
    questions = payload.get("questions") or []
    if not questions:
        return None

    recorded = (payload.get("metadata") or {}).get("prompt_fill")
    per_question = recorded.get("per_question") if isinstance(recorded, dict) else None
    if not per_question:
        per_question = []
        for question in questions:
            message = str(question.get("user_message") or "")
            blocks = re.findall(r"^Title: (.+)$", message, flags=re.MULTILINE)
            per_question.append(
                {
                    "prompt_chars": len(message),
                    "paper_blocks": len(blocks),
                    "unique_paper_blocks": len(set(blocks)),
                }
            )

    chars = [int(entry["prompt_chars"]) for entry in per_question]
    blocks = [int(entry["paper_blocks"]) for entry in per_question]
    unique = [int(entry["unique_paper_blocks"]) for entry in per_question]
    median_chars = statistics.median(chars)
    return {
        "Prompts": len(per_question),
        "Median chars": median_chars,
        "Min chars": min(chars),
        "Min/median (%)": 100.0 * min(chars) / median_chars if median_chars else None,
        "Median blocks": statistics.median(blocks),
        "Min blocks": min(blocks),
        "Repeated-block prompts": sum(
            1 for total, uniq in zip(blocks, unique) if total > uniq
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Question-category split (factoid vs. count)
# ─────────────────────────────────────────────────────────────────────────────
#
# Each eval report carries a ``details`` array with one entry per question; the
# ``answer_type`` field distinguishes counting questions (the citation / paper
# count tasks) from every other "factoid" question (titles, author names, k-th
# author, ...). The detailed-results notebooks recompute accuracy per category
# from these per-question records instead of the report-level ``accuracy``.

COUNTING_ANSWER_TYPES: frozenset[str] = frozenset({"COUNT_PAPER_CITATIONS", "COUNT_AUTHOR_PAPERS"})

QUESTION_CATEGORIES: tuple[str, ...] = ("factoid", "count")

QUESTION_CATEGORY_LABELS: dict[str, str] = {
    "factoid": "Factoid",
    "count": "Count",
}


def classify_answer_type(answer_type: object) -> str:
    """Map an eval-detail ``answer_type`` to a coarse question category.

    Count questions are the ``COUNT_*`` answer types (citation count, author
    paper count); everything else (TITLE, AUTHOR_k, AUTHOR_NAME) is a factoid
    question.
    """
    return "count" if str(answer_type or "").startswith("COUNT_") else "factoid"


def count_report_by_category(report: dict) -> dict[str, dict[str, int]]:
    """Tally ``(correct, total)`` per question category from a report.

    Returns a mapping ``{category: {"correct": int, "total": int}}`` covering
    every entry in :data:`QUESTION_CATEGORIES`. Reports without a ``details``
    array yield all-zero tallies.
    """
    tally = {category: {"correct": 0, "total": 0} for category in QUESTION_CATEGORIES}
    for detail in report.get("details", []) or []:
        category = classify_answer_type(detail.get("answer_type"))
        tally[category]["total"] += 1
        if detail.get("correct"):
            tally[category]["correct"] += 1
    return tally


def filter_results_by_subexperiment(
    results: pd.DataFrame,
    sub_experiment: dict[str, list[str]],
) -> pd.DataFrame:
    """Return the rows of ``results`` matching one sub-experiment's axes.

    ``sub_experiment`` maps a column name (e.g. ``question_hops``,
    ``bucket_label``, ``distracting_content_type``) to the list of values
    allowed for that column -- mirroring the corresponding data-creation shell
    script's hyperparameter arrays, where most columns are fixed to one value
    and one or two are swept. Columns absent from ``sub_experiment`` are left
    unfiltered.
    """
    if results.empty:
        return results.copy()
    mask = pd.Series(True, index=results.index)
    for column, allowed_values in sub_experiment.items():
        allowed = {str(value) for value in allowed_values}
        mask &= results[column].astype(str).isin(allowed)
    return results[mask].copy()


def get_model_display_names(model_slugs: list[str]) -> dict[str, str]:
    ensure_project_root_on_sys_path()
    from src.llm.model_info import DEFAULT_LLM_MANAGER

    return {slug: DEFAULT_LLM_MANAGER.get(slug).model for slug in model_slugs}


def get_context_length_token_map() -> dict[str, int]:
    return {
        "1k": 1_000,
        "2k": 2_000,
        "5k": 5_000,
        "10k": 10_000,
        "20k": 20_000,
        "50k": 50_000,
        "100k": 100_000,
        "200k": 200_000,
        "500k": 500_000,
        "1M": 1_000_000,
    }


OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}


QWEN35_MOE_FAMILY_KEY = "qwen3.5-moe"
QWEN35_NON_MOE_FAMILY_KEY = "qwen3.5-non-moe"


MODEL_STYLE: dict[str, dict[str, object]] = {
    # OpenAI GPT-5.6 family - red gradient, ordered Sol, Terra, Luna (darkest first)
    "gpt-5.6-sol": {"color": "#99000D", "marker": "o", "linestyle": "-"},
    "gpt-5.6-terra": {"color": "#CB181D", "marker": "s", "linestyle": "-"},
    "gpt-5.6-luna": {"color": "#FC9272", "marker": "D", "linestyle": "-"},
    # Qwen3.5 MoE family - blue gradient, larger models darker
    "qwen3.5-397b-a17b": {"color": "#08306B", "marker": "^", "linestyle": "-"},
    "qwen3.5-122b-a10b": {"color": "#08519C", "marker": "v", "linestyle": "-"},
    "qwen3.5-35b-a3b": {"color": "#2171B5", "marker": "P", "linestyle": "-"},
    # Qwen3.5 non-MoE family - adjacent cyan/teal gradient, larger models darker
    "qwen3.5-27b": {"color": "#006D8F", "marker": "X", "linestyle": "-"},
    "qwen3.5-9b": {"color": "#1D91C0", "marker": "p", "linestyle": "-"},
    "qwen3.5-4b": {"color": "#4EB3D3", "marker": "*", "linestyle": "-"},
    "qwen3.5-2b": {"color": "#7BCCC4", "marker": "d", "linestyle": "-"},
    "qwen3.5-0.8b": {"color": "#B4E3E8", "marker": "H", "linestyle": "-"},
    # DeepSeek family - green gradient, larger models darker
    "deepseek-v4-pro": {"color": "#00441B", "marker": "h", "linestyle": "-"},
    "deepseek-v4-flash": {"color": "#41AB5D", "marker": "<", "linestyle": "-"},
    # Gemini family - purple gradient, newer version darker
    "gemini-3.8-flash": {"color": "#4A1486", "marker": ">", "linestyle": "-"},
    "gemini-3.7-flash": {"color": "#CE93D8", "marker": "*", "linestyle": "-"},
    # Gemma family - orange gradient, larger models darker
    "gemma-4-31b": {"color": "#7F2704", "marker": ">", "linestyle": "-"},
    "gemma-4-26b-a4b": {"color": "#A63603", "marker": "8", "linestyle": "-"},
    "gemma-4-e4b": {"color": "#E6550D", "marker": "d", "linestyle": "-"},
    "gemma-4-e2b": {"color": "#FDBE85", "marker": "H", "linestyle": "-"},
}


FALLBACK_MODEL_STYLE: dict[str, object] = {
    "color": "#6B7280",
    "marker": "o",
    "linestyle": "-",
}


MODEL_FAMILY_ORDER: tuple[str, ...] = tuple(MODEL_STYLE.keys())

# Fixed samples of Viridis from 0.02 to 0.80; never resample for a model subset.
VIRIDIS_COLORS = ("#46085C", "#433D84", "#31678E", "#228D8D", "#2DB27D", "#7AD151")
PAPER_FAMILY_COLORS = dict(
    zip(
        (
            "gpt",
            QWEN35_MOE_FAMILY_KEY,
            QWEN35_NON_MOE_FAMILY_KEY,
            "deepseek",
            "gemini",
            "gemma",
        ),
        VIRIDIS_COLORS,
    )
)
PAPER_STYLE = 1
ORIGINAL_STYLE_RC: dict[str, object] | None = None


def get_model_style(model_slug: str) -> dict[str, object]:
    style = dict(MODEL_STYLE.get(model_slug, FALLBACK_MODEL_STYLE))
    style.update(markerfacecolor="white", hatch=None)
    if PAPER_STYLE == 2:
        # One Viridis color per family; hatches tell the models of a family apart.
        for family, color in PAPER_FAMILY_COLORS.items():
            if model_slug_matches_family(model_slug, family):
                style["color"] = color
                family_models = [
                    slug
                    for slug in MODEL_FAMILY_ORDER
                    if model_slug_matches_family(slug, family)
                ]
                if model_slug in family_models:
                    style["hatch"] = ("", "//", "\\\\", "xx", "..")[
                        family_models.index(model_slug)
                    ]
                break
    if PAPER_STYLE != 1:
        # Style 3 keeps the style 1 colors (no hatches) but fills markers like style 2.
        style["markerfacecolor"] = style["color"]
    return style


def is_qwen35_moe_model(model_slug: str) -> bool:
    if not model_slug.startswith("qwen3.5"):
        return False
    return any(
        part.startswith("a") and part.endswith("b") and part[1:-1].isdigit()
        for part in model_slug.split("-")
    )


def is_qwen35_non_moe_model(model_slug: str) -> bool:
    return model_slug.startswith("qwen3.5") and not is_qwen35_moe_model(model_slug)


def model_slug_matches_family(model_slug: str, family_key: str) -> bool:
    if family_key == QWEN35_MOE_FAMILY_KEY:
        return is_qwen35_moe_model(model_slug)
    if family_key == QWEN35_NON_MOE_FAMILY_KEY:
        return is_qwen35_non_moe_model(model_slug)
    return model_slug.startswith(family_key)


def sort_model_slugs_by_family(model_slugs: list[str]) -> list[str]:
    canonical_index = {slug: idx for idx, slug in enumerate(MODEL_FAMILY_ORDER)}
    fallback_index = len(MODEL_FAMILY_ORDER)
    return sorted(model_slugs, key=lambda slug: canonical_index.get(slug, fallback_index))


def arrange_legend_row_major(handles: list, labels: list[str], ncol: int) -> tuple[list, list[str]]:
    import math

    n = len(handles)
    if n <= ncol or ncol <= 1:
        return handles, labels
    nrow = math.ceil(n / ncol)
    new_handles: list = []
    new_labels: list[str] = []
    for col in range(ncol):
        for row in range(nrow):
            idx = row * ncol + col
            if idx < n:
                new_handles.append(handles[idx])
                new_labels.append(labels[idx])
    return new_handles, new_labels


# One stroke width (pt) for every line a figure draws, in all three styles and both
# column spans: data lines, error bars and caps, marker and bar borders, hatches,
# axes spines, ticks and grid lines.
LINE_WIDTH = 1.0
STROKE_RC: dict[str, object] = {
    "lines.linewidth": LINE_WIDTH,
    "lines.markeredgewidth": LINE_WIDTH,
    "patch.linewidth": LINE_WIDTH,
    "hatch.linewidth": LINE_WIDTH,
    "axes.linewidth": LINE_WIDTH,
    "grid.linewidth": LINE_WIDTH,
    "xtick.major.width": LINE_WIDTH,
    "ytick.major.width": LINE_WIDTH,
    "xtick.minor.width": LINE_WIDTH,
    "ytick.minor.width": LINE_WIDTH,
}

PUBLICATION_RC: dict[str, object] = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.labelpad": 3.0,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.borderpad": 0.3,
    "legend.columnspacing": 1.2,
    "legend.labelspacing": 0.3,
    "lines.markersize": 4.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

PAPER_RC: dict[str, object] = {
    "font.family": "serif",
    "font.serif": PUBLICATION_RC["font.serif"],
    "mathtext.fontset": "stix",
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.grid.which": "major",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "grid.color": "0.75",
    "grid.alpha": 0.25,
    "lines.markersize": 5,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.facecolor": "white",
    "legend.edgecolor": "0.85",
    "legend.framealpha": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    **STROKE_RC,
}


def set_paper_style(style: int = 2) -> None:
    """Select current (1), Viridis (2) or style-1-colored paper (3) styling.

    Style 3 matches style 2 except that models keep their style 1 colors and
    bars have no hatches. Call after changing the notebook's PLOT_STYLE, then rerun its output-path
    configuration and plotting cells. Switching back restores the Matplotlib
    settings present before the first call; existing figures are not restyled.
    """
    import matplotlib as mpl

    if style not in (1, 2, 3):
        raise ValueError(
            "Plot style must be 1 (current), 2 (Viridis) or 3 (paper style with style 1 colors, no hatches)."
        )
    global PAPER_STYLE, ORIGINAL_STYLE_RC
    if ORIGINAL_STYLE_RC is None:
        ORIGINAL_STYLE_RC = {key: mpl.rcParams[key] for key in PAPER_RC}
    mpl.rcParams.update(ORIGINAL_STYLE_RC if style == 1 else PAPER_RC)
    PAPER_STYLE = style


FIGURE_WIDTHS: dict[str, float] = {
    "single": 3.25,
    "double": 6.75,
}


def get_figure_size(width_key: str, height_ratio: float = 0.62) -> tuple[float, float]:
    width = FIGURE_WIDTHS[width_key]
    return (width, width * height_ratio)


SPAN_WIDTHS: dict[int, float] = {
    1: 5.25,
    2: 6.75,
}


SPAN_FONT_OVERRIDES: dict[int, dict[str, float]] = {
    1: {
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "lines.markersize": 4.5,
    },
    2: {
        # "axes.labelsize": 8,
        # "axes.titlesize": 8,
        # "xtick.labelsize": 7,
        # "ytick.labelsize": 7,
        # "legend.fontsize": 7,
        # "lines.linewidth": 1.2,
        # "lines.markersize": 4.0,
        # "lines.markeredgewidth": 0.9,
        # "axes.labelsize": 10,
        # "axes.titlesize": 11,
        # "xtick.labelsize": 9,
        # "ytick.labelsize": 9,
        # "legend.fontsize": 9,
        # "lines.linewidth": 1.5,
        # "lines.markersize": 6,
        # "lines.markeredgewidth": 1.5,
        "axes.labelsize": 11,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "lines.markersize": 3.5,
    },
}


def rc_for_span(num_columns_span: int) -> dict[str, object]:
    rc = dict(PUBLICATION_RC)
    if PAPER_STYLE != 1:
        rc.update(PAPER_RC)
    rc.update(SPAN_FONT_OVERRIDES[num_columns_span])
    if PAPER_STYLE != 1:
        rc["lines.markersize"] = 5.0 if num_columns_span == 1 else 4.2
    rc.update(STROKE_RC)
    return rc


def get_figure_size_for_span(
    num_columns_span: int, height_ratio: float = 0.62
) -> tuple[float, float]:
    width = SPAN_WIDTHS[num_columns_span]
    return (width, width * height_ratio)


def normalize_num_columns_span(num_columns_span: int | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(num_columns_span, int):
        return (num_columns_span,)
    return tuple(num_columns_span)


# ─────────────────────────────────────────────────────────────────────────────
# Family-grid subplot layout — TUNABLES (edit here)
# ─────────────────────────────────────────────────────────────────────────────
# All spacing / legend-placement knobs for the "family grid" figures (a 2-column
# grid with one line-plot subplot per model family) live here so they can be
# tweaked in a single place. Values are keyed by ``span`` (the column width:
# 1 = single-column, 2 = double-column) so each width is tuned independently.
#
#   * ``shared``      -> one legend beneath the whole grid (shared_legend=True)
#   * ``per_subplot`` -> one legend beneath each subplot (shared_legend=False)
#
# The inner dicts are passed straight to ``fig.subplots_adjust`` (figure
# fractions: left/right/top/bottom margins, hspace/wspace = inter-row/column gap
# as a fraction of the average subplot size).
FAMILY_GRID_LAYOUT: dict[int, dict[str, dict[str, float]]] = {
    1: {
        "shared": {
            "left": 0.10,
            "right": 0.985,
            "top": 0.95,
            "bottom": 0.34,
            "hspace": 0.45,
            "wspace": 0.10,
        },
        "per_subplot": {
            "left": 0.10,
            "right": 0.985,
            "top": 0.92,
            "bottom": 0.20,
            "hspace": 1.05,
            "wspace": 0.10,
        },
    },
    2: {
        "shared": {
            "left": 0.085,
            "right": 0.985,
            "top": 0.965,
            "bottom": 0.30,
            "hspace": 0.45,
            "wspace": 0.08,
        },
        "per_subplot": {
            "left": 0.085,
            "right": 0.985,
            "top": 0.92,
            "bottom": 0.18,
            "hspace": 1.05,
            "wspace": 0.08,
        },
    },
}

# Lower the grid's top edge by this much (figure fraction) when a figure-level
# suptitle is present, so the suptitle has room above the grid.
FAMILY_GRID_SUPTITLE_TOP_DROP: float = 0.065

# Shared legend (shared_legend=True): one horizontally-centered family row per
# line, stacked beneath the grid. ``legend_top``/``legend_step`` are figure
# fractions.
FAMILY_GRID_SHARED_LEGEND: dict[str, float] = {
    "legend_top": 0.205,
    "legend_step": 0.034,
    "handletextpad": 0.5,
    "columnspacing": 1.2,
}

# Per-subplot legend (shared_legend=False): up to two balanced, centered rows
# beneath each subplot.
# ``legend_top``/``legend_step`` are axes fractions (<0.0 is below the plot).
FAMILY_GRID_SUBPLOT_LEGEND: dict[str, float] = {
    "legend_top": -0.36,
    "legend_step": 0.105,
    "handletextpad": 0.25,
    "columnspacing": 1.0,
}


def save_figure(
    fig,
    output_dir: Path,
    stem: str,
    bbox_extra_artists=None,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {f"{fmt}_path": output_dir / f"{stem}.{fmt}" for fmt in formats}
    # ``bbox_extra_artists`` keeps artists drawn outside the axes (e.g. legends
    # placed below a subplot via ``add_artist``) inside the tight bounding box.
    save_kwargs = dict(dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    if bbox_extra_artists is not None:
        save_kwargs["bbox_extra_artists"] = list(bbox_extra_artists)
    for path in paths.values():
        fig.savefig(path, **save_kwargs)
    return paths


def style_publication_axes(ax) -> None:
    ax.set_axisbelow(True)
    if PAPER_STYLE == 1:
        ax.grid(axis="y", which="major", color="#D7DEE8", linewidth=LINE_WIDTH)
        ax.grid(axis="y", which="minor", color="#EDF2F7", linewidth=LINE_WIDTH)
    else:
        ax.grid(False, axis="both", which="both")
        ax.grid(axis="y", which="major", color="0.75", linewidth=LINE_WIDTH, alpha=0.25)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color("#3F3F46")
        ax.spines[spine_name].set_linewidth(LINE_WIDTH)
    ax.tick_params(axis="both", colors="#1F2937", pad=2.5, width=LINE_WIDTH)


def add_figure_title(fig, ax, title: str | None) -> None:
    """Place ``title`` above a single-axes figure whose legend sits above the axes.

    These figures anchor the legend just above the top spine, so a plain
    ``ax.set_title`` would collide with it. We draw the canvas to measure the
    rendered legend's height, then put the title just above it via
    ``fig.suptitle``. ``savefig(bbox_inches="tight")`` expands the saved canvas
    to include the title even when it sits above the figure box. No-op when
    ``title`` is empty.
    """
    if not title:
        return
    fig.canvas.draw()
    legend = ax.get_legend()
    top = 1.0
    if legend is not None:
        try:
            bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())
            top = max(top, bbox.y1)
        except Exception:
            top = 1.0
    fig.suptitle(title, y=top + 0.04, va="bottom")


def wilson_ci_bounds_pct(n_correct, n_total, z: float = 1.96) -> tuple[float, float]:
    """95% (``z=1.96``) Wilson score CI for a proportion, on the percentage scale.

    Each accuracy plotted in these figures is a proportion of correct/incorrect
    (Bernoulli) outcomes over ``n_total`` independent questions, with a single
    eval run per cell. The meaningful uncertainty is therefore the binomial
    sampling error of that proportion, not the per-question spread. The Wilson
    interval is used instead of the Wald interval ``p +/- z * sqrt(p(1-p)/n)``
    because Wald collapses to zero width at 0% or 100% and is too narrow for
    small ``n``. Wilson is not symmetric around the estimate, so this returns
    ``(lower, upper)`` bounds rather than a half-width. Returns ``(nan, nan)``
    when ``n_total`` is missing or non-positive so callers can skip the interval
    at that point.
    """
    if n_total is None or n_correct is None:
        return float("nan"), float("nan")
    n_total = float(n_total)
    if not math.isfinite(n_total) or n_total <= 0:
        return float("nan"), float("nan")
    p = float(n_correct) / n_total
    z_squared_over_n = z * z / n_total
    center = (p + z_squared_over_n / 2.0) / (1.0 + z_squared_over_n)
    margin = (
        z
        / (1.0 + z_squared_over_n)
        * math.sqrt(p * (1.0 - p) / n_total + z_squared_over_n / (4.0 * n_total))
    )
    # Clip only to absorb floating-point error at p = 0 or p = 1.
    return max(0.0, center - margin) * 100.0, min(1.0, center + margin) * 100.0


def mean_ci_bounds_pct(
    estimates_pct: list[float | None], bounds_pct: list[tuple[float, float]]
) -> tuple[float, float]:
    """CI for the unweighted mean of independent proportions, by MOVER.

    MOVER (Zou & Donner, 2008) recovers each estimate's lower and upper margins
    from its own interval (here Wilson) and combines each side in quadrature, so
    the asymmetry of the inputs is kept. With symmetric inputs it equals the
    usual ``sqrt(sum of squared half-widths) / k`` rule. Returns ``(nan, nan)``
    when any estimate or bound is missing.
    """
    values = [*estimates_pct, *(bound for pair in bounds_pct for bound in pair)]
    if not estimates_pct or any(value is None or not math.isfinite(value) for value in values):
        return float("nan"), float("nan")
    count = len(estimates_pct)
    mean_pct = sum(estimates_pct) / count
    lower_margin = math.hypot(
        *(estimate - lower for estimate, (lower, _upper) in zip(estimates_pct, bounds_pct))
    )
    upper_margin = math.hypot(
        *(upper - estimate for estimate, (_lower, upper) in zip(estimates_pct, bounds_pct))
    )
    return mean_pct - lower_margin / count, mean_pct + upper_margin / count


def format_ci_pct(
    estimate_pct: float | None,
    bounds_pct: tuple[float, float] | None,
    decimals: int = 1,
) -> str:
    """Format a percentage estimate and its CI bounds as ``estimate [lower, upper]``."""
    if estimate_pct is None or bounds_pct is None:
        return "N/A"
    lower_pct, upper_pct = bounds_pct
    if not all(math.isfinite(value) for value in (estimate_pct, lower_pct, upper_pct)):
        return "N/A"
    return (
        f"{estimate_pct:.{decimals}f} "
        f"[{lower_pct:.{decimals}f}, {upper_pct:.{decimals}f}]"
    )


def format_bernoulli_ci_pct(
    n_correct, n_total, z: float = 1.96, decimals: int = 1
) -> str:
    """Format a Bernoulli percentage estimate with its Wilson confidence interval."""
    if n_total is None or n_correct is None or float(n_total) <= 0:
        return "N/A"
    estimate_pct = 100.0 * float(n_correct) / float(n_total)
    return format_ci_pct(
        estimate_pct,
        wilson_ci_bounds_pct(n_correct, n_total, z=z),
        decimals=decimals,
    )


def format_accuracy_ci_table(
    table: pd.DataFrame,
    results: pd.DataFrame,
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    column_values: list,
    column_field: str,
) -> pd.DataFrame:
    """Add a 95% Wilson CI to each accuracy cell in a condition matrix."""
    formatted = table.astype(object)
    for model_slug in ordered_model_slugs:
        model_results = results[results["model_slug"] == model_slug]
        for column_value in column_values:
            subset = model_results[model_results[column_field] == column_value]
            if subset.empty:
                continue
            n_correct = int(subset["correct"].sum())
            n_total = int(subset["total"].sum())
            if n_total <= 0:
                continue
            model_name = model_names[model_slug]
            estimate_pct = float(table.loc[model_name, column_value])
            formatted.loc[model_name, column_value] = format_ci_pct(
                estimate_pct,
                wilson_ci_bounds_pct(n_correct, n_total),
            )
    return formatted


OVERALL_COLUMN = "Overall"


def format_accuracy_table_latex(
    summary: pd.DataFrame,
    *,
    models: list[str],
    model_names: dict[str, str],
    columns: dict[Any, str],
    variable: str,
    random_guessing: dict[Any, float],
    caption: str,
    label: str,
) -> str:
    """Booktabs ``table*`` of accuracy (%) with 95% Wilson intervals, for an ACL appendix.

    ``summary`` has one row per (``model``, ``x``) with ``n``, ``accuracy``,
    ``lower`` and ``upper``. ``columns`` maps each x to its LaTeX header, in
    table order; the key ``OVERALL_COLUMN`` sits outside the ``variable``
    header. Each cell shows the accuracy with its interval below it in small
    type. The highest accuracy in each column is bold. A cell with fewer
    examples than the most in its column gets a dagger, is never bold, and is
    listed in the caption; a missing cell is a dash. The caption gets one
    sentence for each of these marks that the table uses, and one that points
    to the manuscript's random guessing appendix for the last row. Model names
    are hyphenated inside ``\\model{...}`` (manuscript rules R17 and R18).
    """
    rows = {(row.model, row.x): row for row in summary.itertuples(index=False)}
    names = {model: rf"\model{{{model_names[model].replace(' ', '-')}}}" for model in models}
    full_n = {
        key: max((rows[(model, key)].n for model in models if (model, key) in rows), default=0)
        for key in columns
    }
    best = {}
    for key in columns:
        complete = [
            round(rows[(model, key)].accuracy, 2)
            for model in models
            if (model, key) in rows and rows[(model, key)].n == full_n[key]
        ]
        best[key] = max(complete) if len(complete) > 1 else None

    body, short, missing, bold = [], {}, False, False
    for model in models:
        cells = [names[model]]
        for key, header in columns.items():
            row = rows.get((model, key))
            if row is None:
                cells.append("--")
                missing = True
                continue
            value = f"{row.accuracy:.2f}"
            if row.n < full_n[key]:
                value += r"$^\dagger$"
                short.setdefault(model, []).append(f"{header} ($n={row.n}$)")
            elif round(row.accuracy, 2) == best[key]:
                value = rf"\textbf{{{value}}}"
                bold = True
            cells.append(rf"\makecell{{{value}\\{{\scriptsize[{row.lower:.2f}, {row.upper:.2f}]}}}}")
        if body:
            # Two-line rows need a little space between them.
            body.append(r"    \addlinespace[2pt]")
        body.append("    " + " & ".join(cells) + r" \\")
    baseline = ["Random guessing", *(f"{random_guessing[key]:.2f}" for key in columns)]

    grouped = [key for key in columns if key != OVERALL_COLUMN]
    trailing = " &" if OVERALL_COLUMN in columns else ""
    notes = r" Below each accuracy is its 95\% Wilson score interval."
    if bold:
        notes += " Bold marks the highest accuracy in each column."
    notes += (
        " Random guessing is the expected accuracy of uniformly random answers"
        r" (Appendix~\ref{appdx:random_guessing_baseline})."
    )
    if short:
        listed = "; ".join(f"{names[model]}: {', '.join(cells)}" for model, cells in short.items())
        notes += rf" $^\dagger$Fewer examples than the other models in the column, so not bold: {listed}."
    if missing:
        notes += " A dash marks a missing result."
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{5pt}",
        rf"  \begin{{tabular}}{{@{{}}l{'c' * len(columns)}@{{}}}}",
        r"    \toprule",
        rf"    & \multicolumn{{{len(grouped)}}}{{c}}{{\textbf{{{variable}}}}}{trailing} \\",
        rf"    \cmidrule(lr){{2-{len(grouped) + 1}}}",
        "    " + " & ".join([r"\textbf{Model}", *(rf"\textbf{{{header}}}" for header in columns.values())]) + r" \\",
        r"    \midrule",
        *body,
        r"    \midrule",
        "    " + " & ".join(baseline) + r" \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        rf"  \caption{{{caption}{notes}}}",
        rf"  \label{{{label}}}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def format_label_accuracy_table_latex(
    summary: pd.DataFrame,
    *,
    models: list[str],
    model_names: dict[str, str],
    labels: dict[str, str],
    columns: dict[Any, str],
    variable: str,
    caption: str,
    label: str,
    random_guessing: float = 50.0,
) -> str:
    """Booktabs ``table*`` of YES/NO accuracy (%) by ground truth answer, for an ACL appendix.

    ``summary`` has one row per (``model``, ``label``, ``x``) with ``n``, ``accuracy``,
    ``lower`` and ``upper``; ``labels`` maps each ground truth answer to its row
    name (for example ``"YES"`` to ``"YES-label"``), in row order, and ``columns``
    is the same mapping as in :func:`format_accuracy_table_latex`. Each model gets
    one row per label. Cells follow :func:`format_accuracy_table_latex` (accuracy
    with its 95% Wilson interval below it, a dagger for fewer examples than the
    most for that label and column, a dash for a missing result), but nothing is
    bold: the highest NO-label accuracy often comes from a model that answers NO
    to almost every question. A block above the random guessing row gives the
    number of examples for each label and column.
    """
    rows = {(row.model, row.label, row.x): row for row in summary.itertuples(index=False)}
    names = {model: rf"\model{{{model_names[model].replace(' ', '-')}}}" for model in models}
    full_n = {
        (answer, key): max(
            (rows[(model, answer, key)].n for model in models if (model, answer, key) in rows), default=0
        )
        for answer in labels
        for key in columns
    }

    def group_row(index: int, first_cell: str, row_name: str, cells: list[str]) -> str:
        # The first cell spans the rows of all labels.
        first = rf"\multirow{{{len(labels)}}}{{*}}{{{first_cell}}}" if index == 0 else ""
        return "    " + " & ".join([first, row_name, *cells]) + r" \\"

    body, short, missing = [], {}, False
    for model in models:
        if body:
            # Two-line rows need a little space between models.
            body.append(r"    \addlinespace[2pt]")
        for index, (answer, row_name) in enumerate(labels.items()):
            cells = []
            for key, header in columns.items():
                row = rows.get((model, answer, key))
                if row is None:
                    cells.append("--")
                    missing = True
                    continue
                value = f"{row.accuracy:.2f}"
                if row.n < full_n[(answer, key)]:
                    value += r"$^\dagger$"
                    short.setdefault(model, []).append(f"{row_name}, {header} ($n={row.n}$)")
                cells.append(rf"\makecell{{{value}\\{{\scriptsize[{row.lower:.2f}, {row.upper:.2f}]}}}}")
            body.append(group_row(index, names[model], row_name, cells))
    counts = [
        group_row(index, r"\makecell[l]{Number of\\examples}", row_name, [str(full_n[(answer, key)]) for key in columns])
        for index, (answer, row_name) in enumerate(labels.items())
    ]
    baseline = ["Random guessing", "", *(f"{random_guessing:.2f}" for _ in columns)]

    grouped = [key for key in columns if key != OVERALL_COLUMN]
    trailing = " &" if OVERALL_COLUMN in columns else ""
    notes = (
        r" Below each accuracy is its 95\% Wilson score interval."
        " Random guessing is the expected accuracy of uniformly random answers"
        r" (Appendix~\ref{appdx:random_guessing_baseline})."
    )
    if short:
        listed = "; ".join(f"{names[model]}: {', '.join(cells)}" for model, cells in short.items())
        notes += rf" $^\dagger$Fewer examples than the other models: {listed}."
    if missing:
        notes += " A dash marks a missing result."
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{5pt}",
        rf"  \begin{{tabular}}{{@{{}}ll{'c' * len(columns)}@{{}}}}",
        r"    \toprule",
        rf"    & & \multicolumn{{{len(grouped)}}}{{c}}{{\textbf{{{variable}}}}}{trailing} \\",
        rf"    \cmidrule(lr){{3-{len(grouped) + 2}}}",
        "    "
        + " & ".join([r"\textbf{Model}", r"\textbf{Examples}", *(rf"\textbf{{{header}}}" for header in columns.values())])
        + r" \\",
        r"    \midrule",
        *body,
        r"    \midrule",
        *counts,
        r"    \midrule",
        "    " + " & ".join(baseline) + r" \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        rf"  \caption{{{caption}{notes}}}",
        rf"  \label{{{label}}}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def draw_bernoulli_ci_band(ax, x_values, y_values, ci_bounds, color) -> None:
    """Draw a 95% CI around an accuracy line on ``ax``.

    ``ci_bounds`` are the per-point ``(lower, upper)`` bounds from
    :func:`wilson_ci_bounds_pct` (or a metric built on it); points with a
    non-finite bound are dropped so the band simply has a gap there. A one-point
    series uses a vertical error bar because a shaded band would have zero width
    and be invisible. Drawn under the line (``zorder=2``).
    """
    import numpy as np

    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    bounds_arr = np.asarray(ci_bounds, dtype=float).reshape(len(x_arr), 2)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    valid = (
        np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(lower) & np.isfinite(upper)
    )
    if not valid.any():
        return
    if valid.sum() == 1:
        valid_y = y_arr[valid]
        # Clip float rounding (e.g. a Wilson upper bound of 99.999... at 100%), which
        # matplotlib rejects as a negative error-bar length.
        yerr = np.clip(np.vstack((valid_y - lower[valid], upper[valid] - valid_y)), 0.0, None)
        ax.errorbar(
            x_arr[valid],
            valid_y,
            yerr=yerr,
            fmt="none",
            ecolor=color,
            elinewidth=LINE_WIDTH,
            capsize=2.5,
            capthick=LINE_WIDTH,
            alpha=0.65,
            zorder=2,
        )
        return
    ax.fill_between(
        x_arr[valid],
        lower[valid],
        upper[valid],
        color=color,
        alpha=0.15,
        linewidth=0.0,
        zorder=2,
    )


def write_context_length_figure_artifacts(
    results: pd.DataFrame,
    output_dir: Path,
    stem: str,
    *,
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    target_hop: int,
    num_columns_span: int | tuple[int, ...] = (1, 2),
    height_ratio: float = 0.62,
    title: str | None = None,
    error_bar: bool = False,
) -> dict[str, object]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MultipleLocator

    if results.empty:
        raise ValueError("Cannot plot context-length results from an empty table.")

    if error_bar and ("correct" not in results.columns or "total" not in results.columns):
        raise ValueError(
            "results must contain 'correct' and 'total' columns to draw the "
            "95% Bernoulli confidence interval (error_bar=True)."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    token_map = get_context_length_token_map()
    observed_set = set(results["context_length"].astype(str))
    observed_labels = [label for label in token_map if label in observed_set]
    observed_tokens = [token_map[label] for label in observed_labels]

    ordered_model_slugs = sort_model_slugs_by_family(list(ordered_model_slugs))

    series: list[tuple[str, list[float], list[float], list[tuple[float, float]] | None, dict[str, object]]] = []
    for model_slug in ordered_model_slugs:
        subset = results[results["model_slug"] == model_slug].copy()
        if subset.empty:
            continue
        subset["context_tokens"] = subset["context_length"].astype(str).map(token_map)
        subset = subset.sort_values("context_tokens")
        x_values = subset["context_tokens"].astype(float).tolist()
        y_values = subset["accuracy_pct"].astype(float).tolist()
        e_values = (
            [
                wilson_ci_bounds_pct(c, t)
                for c, t in zip(subset["correct"], subset["total"])
            ]
            if error_bar
            else None
        )
        series.append((model_slug, x_values, y_values, e_values, get_model_style(model_slug)))

    spans = normalize_num_columns_span(num_columns_span)
    artifacts: dict[str, object] = {"observed_labels": observed_labels, "spans": {}}

    for span in spans:
        with plt.rc_context(rc_for_span(span)):
            fig, ax = plt.subplots(
                figsize=get_figure_size_for_span(span, height_ratio=height_ratio),
                constrained_layout=True,
            )
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            for model_slug, x_values, y_values, e_values, model_style in series:
                if e_values is not None:
                    draw_bernoulli_ci_band(ax, x_values, y_values, e_values, model_style["color"])
                ax.plot(
                    x_values,
                    y_values,
                    color=model_style["color"],
                    linestyle=model_style["linestyle"],
                    marker=model_style["marker"],
                    markerfacecolor=model_style["markerfacecolor"],
                    markeredgecolor=model_style["color"],
                    markeredgewidth=LINE_WIDTH,
                    label=model_names[model_slug],
                    zorder=3,
                )

            ax.set_xscale("log", base=10)
            ax.set_xlim(min(observed_tokens) * 0.82, max(observed_tokens) * 1.18)
            ax.set_xticks(observed_tokens)
            ax.set_xticklabels(observed_labels)
            ax.set_ylim(0, 100)
            ax.set_yticks(range(0, 101, 20))
            ax.yaxis.set_minor_locator(MultipleLocator(10))

            style_publication_axes(ax)

            ax.set_xlabel("Context length (tokens)")
            ax.set_ylabel("Accuracy (%)")

            n_series = len(series)
            ncol = min(n_series, 4 if span == 2 else 2)
            handles, labels = ax.get_legend_handles_labels()
            handles, labels = arrange_legend_row_major(handles, labels, ncol)
            ax.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=ncol,
                handletextpad=0.5,
                columnspacing=1.2,
            )

            add_figure_title(fig, ax, title)

            stem_for_span = f"{stem}_{span}col"
            saved = save_figure(fig, output_dir, stem_for_span)
            plt.close(fig)
            artifacts["spans"][span] = saved

    primary_key = 2 if 2 in spans else spans[0]
    primary = artifacts["spans"][primary_key]
    artifacts["png_path"] = primary["png_path"]
    artifacts["pdf_path"] = primary["pdf_path"]
    return artifacts


def write_num_hops_figure_artifacts(
    results: pd.DataFrame,
    output_dir: Path,
    stem: str,
    *,
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    context_length_label: str,
    num_columns_span: int | tuple[int, ...] = (1, 2),
    title: str | None = None,
    error_bar: bool = False,
) -> dict[str, object]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MultipleLocator

    if results.empty:
        raise ValueError("Cannot plot hop results from an empty table.")

    if error_bar and ("correct" not in results.columns or "total" not in results.columns):
        raise ValueError(
            "results must contain 'correct' and 'total' columns to draw the "
            "95% Bernoulli confidence interval (error_bar=True)."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    observed_steps = sorted((results["hop"].astype(int)).unique())

    ordered_model_slugs = sort_model_slugs_by_family(list(ordered_model_slugs))

    series: list[tuple[str, list[float], list[float], list[tuple[float, float]] | None, dict[str, object]]] = []
    for model_slug in ordered_model_slugs:
        subset = results[results["model_slug"] == model_slug].copy()
        if subset.empty:
            continue
        subset["reasoning_step"] = subset["hop"].astype(int)
        subset = subset.sort_values("reasoning_step")
        x_values = subset["reasoning_step"].astype(float).tolist()
        y_values = subset["accuracy_pct"].astype(float).tolist()
        e_values = (
            [
                wilson_ci_bounds_pct(c, t)
                for c, t in zip(subset["correct"], subset["total"])
            ]
            if error_bar
            else None
        )
        series.append((model_slug, x_values, y_values, e_values, get_model_style(model_slug)))

    spans = normalize_num_columns_span(num_columns_span)
    artifacts: dict[str, object] = {"observed_steps": observed_steps, "spans": {}}

    use_log_x = len(observed_steps) >= 4 and min(observed_steps) >= 1

    for span in spans:
        with plt.rc_context(rc_for_span(span)):
            fig, ax = plt.subplots(
                figsize=get_figure_size_for_span(span, height_ratio=0.62),
                constrained_layout=True,
            )
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            for model_slug, x_values, y_values, e_values, model_style in series:
                if e_values is not None:
                    draw_bernoulli_ci_band(ax, x_values, y_values, e_values, model_style["color"])
                ax.plot(
                    x_values,
                    y_values,
                    color=model_style["color"],
                    linestyle=model_style["linestyle"],
                    marker=model_style["marker"],
                    markerfacecolor=model_style["markerfacecolor"],
                    markeredgecolor=model_style["color"],
                    markeredgewidth=LINE_WIDTH,
                    label=model_names[model_slug],
                    zorder=3,
                )

            if use_log_x:
                ax.set_xscale("log", base=2)
                ax.set_xlim(min(observed_steps) * 0.85, max(observed_steps) * 1.18)
            else:
                ax.set_xlim(min(observed_steps) - 0.6, max(observed_steps) + 0.6)
            ax.set_xticks(observed_steps)
            ax.set_xticklabels([str(int(step)) for step in observed_steps])

            ax.set_ylim(0, 100)
            ax.set_yticks(range(0, 101, 20))
            ax.yaxis.set_minor_locator(MultipleLocator(10))

            style_publication_axes(ax)

            ax.set_xlabel("Reasoning steps (hops)")
            ax.set_ylabel("Accuracy (%)")

            n_series = len(series)
            ncol = min(n_series, 4 if span == 2 else 2)
            handles, labels = ax.get_legend_handles_labels()
            handles, labels = arrange_legend_row_major(handles, labels, ncol)
            ax.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=ncol,
                handletextpad=0.5,
                columnspacing=1.2,
            )

            add_figure_title(fig, ax, title)

            stem_for_span = f"{stem}_{span}col"
            saved = save_figure(fig, output_dir, stem_for_span)
            plt.close(fig)
            artifacts["spans"][span] = saved

    primary_key = 2 if 2 in spans else spans[0]
    primary = artifacts["spans"][primary_key]
    artifacts["png_path"] = primary["png_path"]
    artifacts["pdf_path"] = primary["pdf_path"]
    return artifacts


def get_table_environment_lines(table_environment: str) -> tuple[str, str]:
    if table_environment not in {"table", "table*"}:
        raise ValueError("table_environment must be either 'table' or 'table*'.")
    return (
        rf"\begin{{{table_environment}}}[htbp]",
        rf"\end{{{table_environment}}}",
    )


def print_table_path(artifacts: dict[str, object], prefix: str = "Table") -> None:
    """Print one quoted path for a table written by ``write_*_table_artifacts``.

    The CSV and TeX outputs always share the same directory and stem, so
    printing both (plus the full LaTeX body) is redundant noise; the CSV path
    stands in for the pair.
    """
    print(f'{prefix}: "{artifacts["csv_path"]}"')


def print_figure_path(artifacts: dict[str, object], prefix: str = "Figure") -> None:
    """Print one quoted path for a figure written by ``save_figure``.

    The PNG and PDF outputs always share the same directory and stem.
    """
    print(f'{prefix}: "{artifacts["png_path"]}"')


def write_table_artifacts(
    table: pd.DataFrame,
    output_dir: Path,
    stem: str,
    *,
    caption: str,
    label: str,
    table_environment: str = "table",
    bold_header: bool = False,
    single_line_header: bool = False,
    resize_width: str = r"\textwidth",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{stem}.csv"
    tex_path = output_dir / f"{stem}.tex"

    table.to_csv(csv_path, index=True)

    latex_table = table.reset_index() if single_line_header else table
    if bold_header:
        latex_table = latex_table.rename(
            columns={column: rf"\textbf{{{column}}}" for column in latex_table.columns}
        )

    latex = latex_table.to_latex(
        na_rep="N/A",
        float_format=lambda value: f"{value:.2f}",
        escape=False,
        column_format="l" + ("r" * len(table.columns)),
        index=not single_line_header,
        caption=caption,
        label=label,
    )
    begin_table_line, end_table_line = get_table_environment_lines(table_environment)
    latex = latex.replace(r"\begin{table}", begin_table_line + "\n" + r"\centering", 1)
    latex = latex.replace(r"\end{table}", end_table_line, 1)
    latex = latex.replace(
        r"\begin{tabular}",
        rf"\resizebox{{{resize_width}}}{{!}}{{%" + "\n" + r"\begin{tabular}",
        1,
    )
    latex = latex.replace(r"\end{tabular}", r"\end{tabular}" + "\n" + r"}", 1)
    lines = latex.splitlines()
    caption_line = next((line for line in lines if line.startswith(r"\caption{")), "")
    label_line = next((line for line in lines if line.startswith(r"\label{")), "")
    lines = [
        line
        for line in lines
        if not line.startswith(r"\caption{") and not line.startswith(r"\label{")
    ]
    end_table_index = lines.index(end_table_line)
    if caption_line:
        lines.insert(end_table_index, caption_line)
        end_table_index += 1
    if label_line:
        lines.insert(end_table_index, label_line)
    latex = "\n".join(lines) + "\n"
    tex_path.write_text(latex, encoding="utf-8")

    return {
        "csv_path": csv_path,
        "tex_path": tex_path,
        "latex": latex,
    }


def write_overall_accuracy_summary_artifacts(
    results: pd.DataFrame,
    output_dir: Path,
    stem: str,
    *,
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    caption: str,
    label: str,
    table_environment: str = "table",
) -> dict[str, object]:
    """Per-model overall accuracy with a 95% Wilson Bernoulli CI.

    Pools every row of ``results`` for each model (each sample is 1 when
    correct, 0 otherwise), then reports the sample mean and confidence
    interval as ``estimate [lower, upper]`` on the percentage scale.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if "correct" not in results.columns or "total" not in results.columns:
        raise ValueError(
            "results must contain 'correct' and 'total' columns for the Bernoulli summary."
        )

    rows: list[dict[str, object]] = []
    for slug in sort_model_slugs_by_family(list(ordered_model_slugs)):
        subset = results[results["model_slug"] == slug]
        if subset.empty:
            continue
        n_total = int(subset["total"].sum())
        n_correct = int(subset["correct"].sum())
        if n_total == 0:
            continue
        rows.append(
            {
                "Model": model_names.get(slug, slug),
                "N": n_total,
                "Accuracy, % [95% CI]": format_bernoulli_ci_pct(n_correct, n_total),
            }
        )

    if not rows:
        raise ValueError("No model rows to summarize.")

    summary = pd.DataFrame(rows).set_index("Model")
    csv_path = output_dir / f"{stem}.csv"
    tex_path = output_dir / f"{stem}.tex"
    summary.to_csv(csv_path, index=True)

    begin_table_line, end_table_line = get_table_environment_lines(table_environment)
    body_lines = []
    for model, row in summary.iterrows():
        body_lines.append(
            f"{model} & {int(row['N'])} & {row['Accuracy, % [95% CI]']} \\\\"
        )

    latex_lines = [
        begin_table_line,
        r"\centering",
        r"\begin{tabular}{lrc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{N} & \textbf{Accuracy, \% [95\% CI]} \\",
        r"\midrule",
        *body_lines,
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        end_table_line,
        "",
    ]
    latex = "\n".join(latex_lines)
    tex_path.write_text(latex, encoding="utf-8")

    return {
        "csv_path": csv_path,
        "tex_path": tex_path,
        "summary": summary,
        "latex": latex,
    }


def write_spanner_table_artifacts(
    table: pd.DataFrame,
    output_dir: Path,
    stem: str,
    *,
    caption: str,
    label: str,
    spanner_label: str,
    grouped_columns: list[str],
    index_label: str = "Model",
    trailing_columns: list[str] | None = None,
    table_environment: str = "table",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{stem}.csv"
    tex_path = output_dir / f"{stem}.tex"

    table.to_csv(csv_path, index=True)

    column_names = [str(column) for column in table.columns]
    grouped_column_names = [str(column) for column in grouped_columns]
    trailing_column_names = (
        [str(column) for column in trailing_columns]
        if trailing_columns is not None
        else column_names[len(grouped_column_names) :]
    )
    expected_order = grouped_column_names + trailing_column_names
    if column_names != expected_order:
        raise ValueError("Table columns must match grouped_columns + trailing_columns order.")

    def format_value(value: object) -> str:
        if pd.isna(value):
            return "N/A"
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    column_format = "l" + ("r" * len(column_names))
    begin_table_line, end_table_line = get_table_environment_lines(table_environment)
    lines = [
        begin_table_line,
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
    ]

    top_row_cells = [
        f"\\multirow{{2}}{{*}}{{\\textbf{{{index_label}}}}}",
        f"\\multicolumn{{{len(grouped_column_names)}}}{{c}}{{\\textbf{{{spanner_label}}}}}",
    ]
    top_row_cells.extend(
        f"\\multirow{{2}}{{*}}{{\\textbf{{{column}}}}}" for column in trailing_column_names
    )
    lines.append(" & ".join(top_row_cells) + r" \\")

    if grouped_column_names:
        lines.append(rf"\cmidrule(lr){{2-{len(grouped_column_names)}}}")

    second_row_cells = [
        "",
        *[f"\\textbf{{{column}}}" for column in grouped_column_names],
        *([""] * len(trailing_column_names)),
    ]
    lines.append(" & ".join(second_row_cells) + r" \\")
    lines.append(r"\midrule")

    for index_value, row in table.iterrows():
        row_cells = [str(index_value)]
        row_cells.extend(format_value(row[column]) for column in table.columns)
        lines.append(" & ".join(row_cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            end_table_line,
        ]
    )

    latex = "\n".join(lines) + "\n"
    tex_path.write_text(latex, encoding="utf-8")

    return {
        "csv_path": csv_path,
        "tex_path": tex_path,
        "latex": latex,
    }


def write_ablation_bar_figure_artifacts(
    results: pd.DataFrame,
    output_dir: Path,
    stem: str,
    *,
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    condition_column: str,
    condition_order: list[str],
    condition_labels: dict[str, str],
    x_axis_label: str,
    accuracy_column: str = "accuracy_pct",
    num_columns_span: int | tuple[int, ...] = (1, 2),
    title: str | None = None,
    error_bar: bool = False,
) -> dict[str, object]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MultipleLocator

    if results.empty:
        raise ValueError("Cannot plot ablation results from an empty table.")

    output_dir.mkdir(parents=True, exist_ok=True)

    aggregated = (
        results.groupby(["model_slug", condition_column])[accuracy_column].mean().reset_index()
    )

    # Each bar is an accuracy, i.e. a proportion of correct/incorrect (Bernoulli)
    # outcomes over ``total`` independent questions, with a single eval run per
    # (model, condition) cell. The meaningful uncertainty is therefore the
    # binomial sampling error of that proportion, not the per-question spread.
    # We draw a 95% Wilson score confidence interval on the percentage scale
    # (consistent with write_overall_accuracy_summary_artifacts).
    error_bounds: dict[tuple[str, object], tuple[float, float]] = {}
    if error_bar:
        if "correct" not in results.columns or "total" not in results.columns:
            raise ValueError(
                "results must contain 'correct' and 'total' columns to draw the "
                "95% Bernoulli confidence interval (error_bar=True)."
            )
        pooled = (
            results.groupby(["model_slug", condition_column])[["correct", "total"]].sum()
        )
        for (model_slug_key, condition_key), pooled_row in pooled.iterrows():
            error_bounds[(model_slug_key, condition_key)] = wilson_ci_bounds_pct(
                pooled_row["correct"], pooled_row["total"]
            )

    ordered_model_slugs = sort_model_slugs_by_family(list(ordered_model_slugs))

    n_models = len(ordered_model_slugs)
    n_conditions = len(condition_order)
    group_width = 0.78
    bar_width = group_width / max(n_models, 1)
    x_positions = np.arange(n_conditions)

    spans = normalize_num_columns_span(num_columns_span)
    artifacts: dict[str, object] = {"spans": {}}

    for span in spans:
        with plt.rc_context(rc_for_span(span)):
            fig, ax = plt.subplots(
                figsize=get_figure_size_for_span(span, height_ratio=0.62),
                constrained_layout=True,
            )
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            for model_index, model_slug in enumerate(ordered_model_slugs):
                model_style = get_model_style(model_slug)
                offsets = (model_index - (n_models - 1) / 2) * bar_width
                heights = []
                for condition in condition_order:
                    row = aggregated[
                        (aggregated["model_slug"] == model_slug)
                        & (aggregated[condition_column] == condition)
                    ]
                    heights.append(float(row[accuracy_column].iloc[0]) if not row.empty else np.nan)

                yerr = None
                if error_bar:
                    lower_errors: list[float] = []
                    upper_errors: list[float] = []
                    for condition, height in zip(condition_order, heights):
                        lower, upper = error_bounds.get(
                            (model_slug, condition), (np.nan, np.nan)
                        )
                        if not np.isfinite([lower, upper, height]).all():
                            lower_errors.append(0.0)
                            upper_errors.append(0.0)
                        else:
                            # The bar height is a mean over rows while the bounds use
                            # pooled counts, so guard against a tiny mismatch.
                            lower_errors.append(float(max(0.0, height - lower)))
                            upper_errors.append(float(max(0.0, upper - height)))
                    yerr = np.array([lower_errors, upper_errors])

                ax.bar(
                    x_positions + offsets,
                    heights,
                    width=bar_width * 0.92,
                    color=model_style["color"],
                    # No bar border; edgecolor only colors the hatch lines.
                    edgecolor="#1F2937",
                    hatch=model_style["hatch"],
                    linewidth=0.0,
                    label=model_names[model_slug],
                    yerr=yerr,
                    capsize=2.0,
                    error_kw={
                        "elinewidth": LINE_WIDTH,
                        "capthick": LINE_WIDTH,
                        "ecolor": "#1F2937",
                        "zorder": 4,
                    },
                    zorder=3,
                )

            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                [condition_labels.get(condition, condition) for condition in condition_order],
            )
            ax.set_ylim(0, 100)
            ax.set_yticks(range(0, 101, 20))
            ax.yaxis.set_minor_locator(MultipleLocator(10))

            style_publication_axes(ax)

            ax.set_xlabel(x_axis_label)
            ax.set_ylabel("Accuracy (%)")

            ncol = min(n_models, 4 if span == 2 else 2)
            handles, labels = ax.get_legend_handles_labels()
            handles, labels = arrange_legend_row_major(handles, labels, ncol)
            ax.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=ncol,
                handletextpad=0.4,
                columnspacing=1.0,
            )

            add_figure_title(fig, ax, title)

            stem_for_span = f"{stem}_{span}col"
            saved = save_figure(fig, output_dir, stem_for_span)
            plt.close(fig)
            artifacts["spans"][span] = saved

    primary_key = 2 if 2 in spans else spans[0]
    primary = artifacts["spans"][primary_key]
    artifacts["png_path"] = primary["png_path"]
    artifacts["pdf_path"] = primary["pdf_path"]
    return artifacts


DEFAULT_FAMILY_GRID_SPECS: tuple[tuple[str, str], ...] = (
    ("deepseek", "DeepSeek"),
    ("gemini", "Gemini"),
    (QWEN35_MOE_FAMILY_KEY, "Qwen3.5 MoE"),
    (QWEN35_NON_MOE_FAMILY_KEY, "Qwen3.5 non-MoE"),
    ("gpt-5.6", "GPT-5.6"),
)


def _split_legend_entries_balanced(entries: list) -> list[list]:
    """Split legend entries into up to two balanced, centered rows.

    Up to two items stay on a single line; from three items up the entries are
    split across two rows, the second absorbing the extra item when the count
    is odd, so e.g. ``5 -> [2, 3]``, ``4 -> [2, 2]``, ``3 -> [1, 2]``,
    ``2 -> [2]`` and ``1 -> [1]``. Used for the per-subplot legends when
    ``shared_legend`` is False.
    """
    n = len(entries)
    if n <= 2:
        return [entries] if entries else []
    first = n // 2
    return [entries[:first], entries[first:]]


def _decorate_family_subplot(ax, family_label, row_entries, shared_legend) -> list:
    """Add the subplot title and (when ``shared_legend`` is False) its own legend.

    In shared-legend mode the family name is a plain title and the grid-level
    legend is drawn elsewhere. In per-subplot mode each subplot keeps a
    balanced, centered legend below its x-axis. Returns the created legend
    artists so the caller can pass them to :func:`save_figure` as
    ``bbox_extra_artists``.
    """
    from matplotlib.legend import Legend

    if shared_legend or not row_entries:
        ax.set_title(family_label)
        return []

    cfg = FAMILY_GRID_SUBPLOT_LEGEND
    sub_rows = _split_legend_entries_balanced(row_entries)
    ax.set_title(family_label)

    legends: list = []
    for sub_index, sub_row in enumerate(sub_rows):
        sub_legend = Legend(
            ax,
            [handle for handle, _ in sub_row],
            [label for _, label in sub_row],
            loc="upper center",
            bbox_to_anchor=(0.5, cfg["legend_top"] - sub_index * cfg["legend_step"]),
            bbox_transform=ax.transAxes,
            ncol=len(sub_row),
            handletextpad=cfg["handletextpad"],
            columnspacing=cfg["columnspacing"],
        )
        ax.add_artist(sub_legend)
        legends.append(sub_legend)
    return legends


def _add_shared_family_legend(fig, legend_rows) -> None:
    """Draw the grid-level shared legend: one centered family row per line."""
    cfg = FAMILY_GRID_SHARED_LEGEND
    populated_rows = [row for row in legend_rows if row]
    for index, row in enumerate(populated_rows):
        fig.legend(
            [handle for handle, _ in row],
            [label for _, label in row],
            loc="upper center",
            bbox_to_anchor=(0.5, cfg["legend_top"] - index * cfg["legend_step"]),
            ncol=len(row),
            handletextpad=cfg["handletextpad"],
            columnspacing=cfg["columnspacing"],
        )


def _apply_family_grid_layout(fig, span, shared_legend, has_title) -> None:
    """Apply ``subplots_adjust`` for a family grid from the central tunables."""
    mode = "shared" if shared_legend else "per_subplot"
    layout = dict(FAMILY_GRID_LAYOUT[span][mode])
    if has_title:
        layout["top"] -= FAMILY_GRID_SUPTITLE_TOP_DROP
    fig.subplots_adjust(**layout)


def write_context_length_family_grid_figure_artifacts(
    results: pd.DataFrame,
    output_dir: Path,
    stem: str,
    *,
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    target_hop: int,
    family_specs: tuple[tuple[str, str], ...] = DEFAULT_FAMILY_GRID_SPECS,
    span: int = 2,
    height_ratio: float = 1.2,
    title: str | None = None,
    shared_legend: bool = True,
    error_bar: bool = False,
) -> dict[str, object]:
    """Grid of context-length line plots, one subplot per model family.

    Every subplot shares the same log-scaled context-length x-axis and the
    0-100 accuracy y-axis. The model-family name is the subplot title, and
    models are sorted large -> small within each family following the
    canonical family order in ``MODEL_STYLE``.

    When ``shared_legend`` is True (default) a single legend sits beneath the
    whole grid, with each model family laid out on its own centered row. When
    False, every subplot instead carries its own legend directly beneath it,
    split into two balanced, horizontally-centered rows (see
    :func:`_split_legend_entries_balanced`).
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    if results.empty:
        raise ValueError("Cannot plot context-length results from an empty table.")

    if error_bar and ("correct" not in results.columns or "total" not in results.columns):
        raise ValueError(
            "results must contain 'correct' and 'total' columns to draw the "
            "95% Bernoulli confidence interval (error_bar=True)."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    token_map = get_context_length_token_map()
    observed_set = set(results["context_length"].astype(str))
    observed_labels = [label for label in token_map if label in observed_set]
    observed_tokens = [token_map[label] for label in observed_labels]
    if not observed_tokens:
        raise ValueError("No recognized context-length labels present in results.")

    ordered_model_slugs = sort_model_slugs_by_family(list(ordered_model_slugs))

    def build_family_series(family_key: str):
        series: list[
            tuple[str, list[float], list[float], list[float] | None, dict[str, object]]
        ] = []
        for model_slug in ordered_model_slugs:
            if not model_slug_matches_family(model_slug, family_key):
                continue
            subset = results[results["model_slug"] == model_slug].copy()
            if subset.empty:
                continue
            subset["context_tokens"] = subset["context_length"].astype(str).map(token_map)
            subset = subset.sort_values("context_tokens")
            x_values = subset["context_tokens"].astype(float).tolist()
            y_values = subset["accuracy_pct"].astype(float).tolist()
            e_values = (
                [
                    wilson_ci_bounds_pct(c, t)
                    for c, t in zip(subset["correct"], subset["total"])
                ]
                if error_bar
                else None
            )
            series.append((model_slug, x_values, y_values, e_values, get_model_style(model_slug)))
        return series

    n_families = len(family_specs)
    n_cols = 2
    n_rows = -(-n_families // n_cols)  # ceil division

    with plt.rc_context(rc_for_span(span)):
        fig_width = SPAN_WIDTHS[span]
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_width, fig_width * height_ratio),
            sharex=False,
            sharey=True,
            dpi=300,
        )
        fig.patch.set_facecolor("white")
        axes_flat = list(axes.flat)

        subplot_legends: list[object] = []

        legend_rows: list[list[tuple[object, str]]] = []
        for ax, (family_key, family_label) in zip(axes_flat, family_specs):
            ax.set_facecolor("white")
            series = build_family_series(family_key)

            row_entries: list[tuple[object, str]] = []
            for model_slug, x_values, y_values, e_values, model_style in series:
                if e_values is not None:
                    draw_bernoulli_ci_band(ax, x_values, y_values, e_values, model_style["color"])
                (line,) = ax.plot(
                    x_values,
                    y_values,
                    color=model_style["color"],
                    linestyle=model_style["linestyle"],
                    marker=model_style["marker"],
                    markerfacecolor=model_style["markerfacecolor"],
                    markeredgecolor=model_style["color"],
                    markeredgewidth=LINE_WIDTH,
                    label=model_names.get(model_slug, model_slug),
                    zorder=3,
                )
                row_entries.append((line, model_names.get(model_slug, model_slug)))
            legend_rows.append(row_entries)

            ax.set_xscale("log", base=10)
            ax.set_xlim(min(observed_tokens) * 0.82, max(observed_tokens) * 1.18)
            ax.set_xticks(observed_tokens)
            ax.set_xticklabels(observed_labels)
            ax.set_ylim(0, 100)
            ax.set_yticks(range(0, 101, 20))
            ax.yaxis.set_minor_locator(MultipleLocator(10))
            style_publication_axes(ax)
            ax.set_xlabel("Context length (tokens)")

            if not series:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#9CA3AF",
                    fontstyle="italic",
                )

            # Family-name title plus, when shared_legend is False, the subplot's
            # own balanced/centered legend below the x-axis.
            subplot_legends.extend(
                _decorate_family_subplot(ax, family_label, row_entries, shared_legend)
            )

        # Hide any leftover axes when the family count is not a full grid.
        for ax in axes_flat[n_families:]:
            ax.set_visible(False)

        # Accuracy label only on the left column (shared y-axis).
        for left_ax in axes[:, 0] if n_rows > 1 else [axes[0]]:
            left_ax.set_ylabel("Accuracy (%)")

        _apply_family_grid_layout(fig, span, shared_legend, has_title=title is not None)
        if title:
            fig.suptitle(title, y=0.99, va="top")
        if shared_legend:
            _add_shared_family_legend(fig, legend_rows)

        saved = save_figure(
            fig,
            output_dir,
            f"{stem}_{span}col",
            bbox_extra_artists=subplot_legends or None,
        )
        plt.close(fig)

    return {
        "observed_labels": observed_labels,
        "png_path": saved["png_path"],
        "pdf_path": saved["pdf_path"],
        "spans": {span: saved},
    }


def write_num_hops_family_grid_figure_artifacts(
    results: pd.DataFrame,
    output_dir: Path,
    stem: str,
    *,
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    context_length_label: str,
    family_specs: tuple[tuple[str, str], ...] = DEFAULT_FAMILY_GRID_SPECS,
    span: int = 2,
    height_ratio: float = 1.2,
    title: str | None = None,
    shared_legend: bool = True,
    error_bar: bool = False,
) -> dict[str, object]:
    """Grid of num-hops line plots, one subplot per model family.

    The num-hops analogue of
    :func:`write_context_length_family_grid_figure_artifacts`: accuracy is
    plotted against the number of reasoning hops (log-base-2 x-axis when there
    are enough hop steps) instead of context length. Every subplot shares the
    hop x-axis and the 0-100 accuracy y-axis, the family name is the subplot
    title, and one shared legend per family row sits beneath the grid (models
    sorted large -> small within each row following ``MODEL_STYLE``).
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    if results.empty:
        raise ValueError("Cannot plot hop results from an empty table.")

    if error_bar and ("correct" not in results.columns or "total" not in results.columns):
        raise ValueError(
            "results must contain 'correct' and 'total' columns to draw the "
            "95% Bernoulli confidence interval (error_bar=True)."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    observed_steps = sorted(results["hop"].astype(int).unique())
    if not observed_steps:
        raise ValueError("No hop steps present in results.")
    use_log_x = len(observed_steps) >= 4 and min(observed_steps) >= 1

    ordered_model_slugs = sort_model_slugs_by_family(list(ordered_model_slugs))

    def build_family_series(family_key: str):
        series: list[
            tuple[str, list[float], list[float], list[float] | None, dict[str, object]]
        ] = []
        for model_slug in ordered_model_slugs:
            if not model_slug_matches_family(model_slug, family_key):
                continue
            subset = results[results["model_slug"] == model_slug].copy()
            if subset.empty:
                continue
            subset["reasoning_step"] = subset["hop"].astype(int)
            subset = subset.sort_values("reasoning_step")
            x_values = subset["reasoning_step"].astype(float).tolist()
            y_values = subset["accuracy_pct"].astype(float).tolist()
            e_values = (
                [
                    wilson_ci_bounds_pct(c, t)
                    for c, t in zip(subset["correct"], subset["total"])
                ]
                if error_bar
                else None
            )
            series.append((model_slug, x_values, y_values, e_values, get_model_style(model_slug)))
        return series

    n_families = len(family_specs)
    n_cols = 2
    n_rows = -(-n_families // n_cols)  # ceil division

    with plt.rc_context(rc_for_span(span)):
        fig_width = SPAN_WIDTHS[span]
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_width, fig_width * height_ratio),
            sharex=False,
            sharey=True,
        )
        fig.patch.set_facecolor("white")
        axes_flat = list(axes.flat)

        subplot_legends: list[object] = []
        legend_rows: list[list[tuple[object, str]]] = []
        for ax, (family_key, family_label) in zip(axes_flat, family_specs):
            ax.set_facecolor("white")
            series = build_family_series(family_key)

            row_entries: list[tuple[object, str]] = []
            for model_slug, x_values, y_values, e_values, model_style in series:
                if e_values is not None:
                    draw_bernoulli_ci_band(ax, x_values, y_values, e_values, model_style["color"])
                (line,) = ax.plot(
                    x_values,
                    y_values,
                    color=model_style["color"],
                    linestyle=model_style["linestyle"],
                    marker=model_style["marker"],
                    markerfacecolor=model_style["markerfacecolor"],
                    markeredgecolor=model_style["color"],
                    markeredgewidth=LINE_WIDTH,
                    label=model_names.get(model_slug, model_slug),
                    zorder=3,
                )
                row_entries.append((line, model_names.get(model_slug, model_slug)))
            legend_rows.append(row_entries)

            if use_log_x:
                ax.set_xscale("log", base=2)
                ax.set_xlim(min(observed_steps) * 0.85, max(observed_steps) * 1.18)
            else:
                ax.set_xlim(min(observed_steps) - 0.6, max(observed_steps) + 0.6)
            ax.set_xticks(observed_steps)
            ax.set_xticklabels([str(int(step)) for step in observed_steps])
            ax.set_ylim(0, 100)
            ax.set_yticks(range(0, 101, 20))
            ax.yaxis.set_minor_locator(MultipleLocator(10))
            style_publication_axes(ax)
            ax.set_xlabel("Reasoning steps (hops)")

            if not series:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#9CA3AF",
                    fontstyle="italic",
                )

            # Family-name title plus, when shared_legend is False, the subplot's
            # own balanced/centered legend below the x-axis.
            subplot_legends.extend(
                _decorate_family_subplot(ax, family_label, row_entries, shared_legend)
            )

        # Hide any leftover axes when the family count is not a full grid.
        for ax in axes_flat[n_families:]:
            ax.set_visible(False)

        # Accuracy label only on the left column (shared y-axis).
        for left_ax in axes[:, 0] if n_rows > 1 else [axes[0]]:
            left_ax.set_ylabel("Accuracy (%)")

        _apply_family_grid_layout(fig, span, shared_legend, has_title=title is not None)
        if title:
            fig.suptitle(title, y=0.99, va="top")
        if shared_legend:
            _add_shared_family_legend(fig, legend_rows)

        saved = save_figure(
            fig,
            output_dir,
            f"{stem}_{span}col",
            bbox_extra_artists=subplot_legends or None,
        )
        plt.close(fig)

    return {
        "observed_steps": observed_steps,
        "png_path": saved["png_path"],
        "pdf_path": saved["pdf_path"],
        "spans": {span: saved},
    }


def write_bucket_figure_artifacts(
    results: pd.DataFrame,
    output_dir: Path,
    stem: str,
    *,
    bucket_labels: list[str],
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    num_columns_span: int | tuple[int, ...] = (1, 2),
    title: str | None = None,
    error_bar: bool = False,
    metric_ci: dict[str, Callable[[dict], tuple[float, float]]] | None = None,
    value_column: str = "accuracy_pct",
    overlay_column: str | None = None,
    overlay_styles: tuple[str, str] = ("-", "--"),
    series_names: tuple[str, str] | None = None,
    y_label: str = "Accuracy (%)",
    x_axis_label: str = "Expected search-step bucket",
    baseline_by_bucket: dict[str, float] | None = None,
    baseline_name: str = "Baseline",
) -> dict[str, object]:
    """Plot one metric per bucket, optionally overlaying a second one.

    ``value_column`` and ``overlay_column`` name columns produced by an
    ``add_figure_metric_columns``-style helper. When ``overlay_column`` is set
    both lines use the model's colour and are distinguished by
    ``overlay_styles``, so one model reads as a paired series (e.g. YES/NO
    recall). ``baseline_by_bucket`` draws a flat grey reference line per
    bucket. ``metric_ci`` maps a plotted column name to a function computing
    its 95% CI ``(lower, upper)`` bounds from one row's pooled counts; required
    per plotted column when ``error_bar=True``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    if results.empty:
        raise ValueError("Cannot plot bucket results from an empty table.")
    plotted_columns = [value_column]
    if overlay_column is not None:
        plotted_columns.append(overlay_column)
    missing_columns = [
        column for column in plotted_columns if column not in results.columns
    ]
    if missing_columns:
        raise ValueError(
            "results must contain the plotted column(s) "
            f"{missing_columns} — run add_figure_metric_columns first."
        )
    metric_ci = metric_ci or {}
    if error_bar:
        undefined = [column for column in plotted_columns if column not in metric_ci]
        if undefined:
            raise ValueError(f"No confidence interval is defined for {undefined}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    observed_buckets = [
        label for label in bucket_labels if label in set(results["bucket_label"].astype(str))
    ]
    if not observed_buckets:
        raise ValueError("Cannot plot bucket results without recognized buckets.")

    bucket_positions = {label: index for index, label in enumerate(observed_buckets)}
    ordered_model_slugs = sort_model_slugs_by_family(list(ordered_model_slugs))

    series = []
    for model_slug in ordered_model_slugs:
        subset = results[results["model_slug"] == model_slug].copy()
        subset = subset[subset["bucket_label"].isin(observed_buckets)]
        if subset.empty:
            continue
        subset["bucket_position"] = subset["bucket_label"].map(bucket_positions)
        subset = subset.sort_values("bucket_position")
        x_values = subset["bucket_position"].astype(float).tolist()
        lines = []
        for column, linestyle in zip(plotted_columns, overlay_styles):
            y_values = subset[column].astype(float).tolist()
            e_values = (
                [metric_ci[column](row) for _index, row in subset.iterrows()]
                if error_bar
                else None
            )
            lines.append((linestyle, y_values, e_values))
        series.append((model_slug, x_values, lines, get_model_style(model_slug)))

    if not series:
        raise ValueError("Cannot plot bucket results without model series.")

    spans = normalize_num_columns_span(num_columns_span)
    artifacts = {"observed_buckets": observed_buckets, "spans": {}}

    for span in spans:
        with plt.rc_context(rc_for_span(span)):
            fig, ax = plt.subplots(
                figsize=get_figure_size_for_span(span, height_ratio=0.62),
                constrained_layout=True,
            )
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            for model_slug, x_values, lines, model_style in series:
                for line_index, (linestyle, y_values, e_values) in enumerate(lines):
                    if e_values is not None:
                        draw_bernoulli_ci_band(
                            ax, x_values, y_values, e_values, model_style["color"]
                        )
                    ax.plot(
                        x_values,
                        y_values,
                        color=model_style["color"],
                        linestyle=(
                            linestyle if overlay_column else model_style["linestyle"]
                        ),
                        marker=model_style["marker"],
                        markerfacecolor=model_style["markerfacecolor"],
                        markeredgecolor=model_style["color"],
                        markeredgewidth=LINE_WIDTH,
                        label=(
                            model_names[model_slug]
                            if line_index == 0
                            else "_nolegend_"
                        ),
                        zorder=3,
                    )

            if baseline_by_bucket:
                baseline_values = [
                    baseline_by_bucket.get(label, float("nan"))
                    for label in observed_buckets
                ]
                ax.plot(
                    list(range(len(observed_buckets))),
                    baseline_values,
                    color="#6B7280",
                    linestyle=(0, (2, 2)),
                    linewidth=LINE_WIDTH,
                    label=baseline_name,
                    zorder=2,
                )

            ax.set_xlim(-0.4, len(observed_buckets) - 0.6)
            if len(observed_buckets) == 1:
                ax.set_xlim(-0.4, 0.4)
            ax.set_xticks(list(range(len(observed_buckets))))
            ax.set_xticklabels(observed_buckets)
            ax.set_ylim(0, 100)
            ax.set_yticks(range(0, 101, 20))
            ax.yaxis.set_minor_locator(MultipleLocator(10))
            style_publication_axes(ax)
            ax.set_xlabel(x_axis_label)
            ax.set_ylabel(y_label)

            handles, labels = ax.get_legend_handles_labels()
            if overlay_column is not None and series_names is not None:
                handles = list(handles) + [
                    Line2D([], [], color="#4B5563", linestyle=linestyle)
                    for linestyle in overlay_styles
                ]
                labels = list(labels) + list(series_names)
            ncol = min(len(labels), 4 if span == 2 else 2)
            handles, labels = arrange_legend_row_major(handles, labels, ncol)
            ax.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=ncol,
                handletextpad=0.5,
                columnspacing=1.2,
            )
            add_figure_title(fig, ax, title)

            saved = save_figure(fig, output_dir, f"{stem}_{span}col")
            plt.close(fig)
            artifacts["spans"][span] = saved

    primary_key = 2 if 2 in spans else spans[0]
    primary = artifacts["spans"][primary_key]
    artifacts["png_path"] = primary["png_path"]
    artifacts["pdf_path"] = primary["pdf_path"]
    return artifacts


def write_bucket_family_grid_figure_artifacts(
    results: pd.DataFrame,
    output_dir: Path,
    stem: str,
    *,
    bucket_labels: list[str],
    ordered_model_slugs: list[str],
    model_names: dict[str, str],
    family_specs: tuple[tuple[str, str], ...] = DEFAULT_FAMILY_GRID_SPECS,
    span: int = 2,
    height_ratio: float = 1.2,
    title: str | None = None,
    shared_legend: bool = True,
    error_bar: bool = False,
    x_axis_label: str = "Expected search-step bucket",
) -> dict[str, object]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    if results.empty:
        raise ValueError("Cannot plot bucket results from an empty table.")

    if error_bar and ("correct" not in results.columns or "total" not in results.columns):
        raise ValueError(
            "results must contain 'correct' and 'total' columns to draw the "
            "95% Bernoulli confidence interval (error_bar=True)."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    observed_buckets = [
        label for label in bucket_labels if label in set(results["bucket_label"].astype(str))
    ]
    if not observed_buckets:
        raise ValueError("No recognized bucket labels present in results.")
    bucket_positions = {label: index for index, label in enumerate(observed_buckets)}

    ordered_model_slugs = sort_model_slugs_by_family(list(ordered_model_slugs))

    def build_family_series(family_key: str):
        series: list[
            tuple[str, list[float], list[float], list[float] | None, dict[str, object]]
        ] = []
        for model_slug in ordered_model_slugs:
            if not model_slug_matches_family(model_slug, family_key):
                continue
            subset = results[results["model_slug"] == model_slug].copy()
            subset = subset[subset["bucket_label"].isin(observed_buckets)]
            if subset.empty:
                continue
            subset["bucket_position"] = subset["bucket_label"].map(bucket_positions)
            subset = subset.sort_values("bucket_position")
            x_values = subset["bucket_position"].astype(float).tolist()
            y_values = subset["accuracy_pct"].astype(float).tolist()
            e_values = (
                [
                    wilson_ci_bounds_pct(c, t)
                    for c, t in zip(subset["correct"], subset["total"])
                ]
                if error_bar
                else None
            )
            series.append((model_slug, x_values, y_values, e_values, get_model_style(model_slug)))
        return series

    n_families = len(family_specs)
    n_cols = 2
    n_rows = -(-n_families // n_cols)  # ceil division

    with plt.rc_context(rc_for_span(span)):
        fig_width = SPAN_WIDTHS[span]
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_width, fig_width * height_ratio),
            sharex=False,
            sharey=True,
            dpi=300,
        )
        fig.patch.set_facecolor("white")
        axes_flat = list(axes.flat)

        subplot_legends: list[object] = []
        legend_rows: list[list[tuple[object, str]]] = []
        for ax, (family_key, family_label) in zip(axes_flat, family_specs):
            ax.set_facecolor("white")
            series = build_family_series(family_key)

            row_entries: list[tuple[object, str]] = []
            for model_slug, x_values, y_values, e_values, model_style in series:
                if e_values is not None:
                    draw_bernoulli_ci_band(ax, x_values, y_values, e_values, model_style["color"])
                (line,) = ax.plot(
                    x_values,
                    y_values,
                    color=model_style["color"],
                    linestyle=model_style["linestyle"],
                    marker=model_style["marker"],
                    markerfacecolor=model_style["markerfacecolor"],
                    markeredgecolor=model_style["color"],
                    markeredgewidth=LINE_WIDTH,
                    label=model_names.get(model_slug, model_slug),
                    zorder=3,
                )
                row_entries.append((line, model_names.get(model_slug, model_slug)))
            legend_rows.append(row_entries)

            if len(observed_buckets) == 1:
                ax.set_xlim(-0.4, 0.4)
            else:
                ax.set_xlim(-0.4, len(observed_buckets) - 0.6)
            ax.set_xticks(list(range(len(observed_buckets))))
            ax.set_xticklabels(observed_buckets, rotation=35, ha="right")
            ax.set_ylim(0, 100)
            ax.set_yticks(range(0, 101, 20))
            ax.yaxis.set_minor_locator(MultipleLocator(10))
            style_publication_axes(ax)
            ax.set_xlabel(x_axis_label)

            if not series:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#9CA3AF",
                    fontstyle="italic",
                )

            subplot_legends.extend(
                _decorate_family_subplot(ax, family_label, row_entries, shared_legend)
            )

        for ax in axes_flat[n_families:]:
            ax.set_visible(False)

        for left_ax in axes[:, 0] if n_rows > 1 else [axes[0]]:
            left_ax.set_ylabel("Accuracy (%)")

        _apply_family_grid_layout(fig, span, shared_legend, has_title=title is not None)
        if title:
            fig.suptitle(title, y=0.99, va="top")
        if shared_legend:
            _add_shared_family_legend(fig, legend_rows)

        saved = save_figure(
            fig,
            output_dir,
            f"{stem}_{span}col",
            bbox_extra_artists=subplot_legends or None,
        )
        plt.close(fig)

    return {
        "observed_buckets": observed_buckets,
        "png_path": saved["png_path"],
        "pdf_path": saved["pdf_path"],
        "spans": {span: saved},
    }
