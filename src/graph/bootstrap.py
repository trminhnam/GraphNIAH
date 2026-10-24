from __future__ import annotations

import argparse
import sqlite3
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from src.author_parser import VLMAuthorParser
from src.clients import (
    ArxivBulkClient,
    ArxivFallbackClient,
    ArxivOnlineClient,
    DBLPClient,
    SemanticScholarClient,
)
from src.parsers.pdf_parser import PDFParser
from src.setup.constants import NODE_PAPER, nid_db_id, nid_type, preprocess_paper_title

from .checkpoints import build_graph_stem, extract_base_stem, prune_checkpoints
from .expansion import ARXIV_ID_RE, PaperExpander, apply_fetched_paper, fetch_paper_data
from .loaders import load_graph
from .network import CitationNetwork, GraphLimits
from .persistence import save_graph
from .resolution import (
    infer_output_root_from_graph_path,
    resolve_db_path,
    resolve_output_root,
)
from .storage import DatabaseManager
from src.reference_parser.registry import import_reference_parsers

DEFAULT_VLM_REFERENCE_MODEL = "openai/gpt-5.4-nano"
DEFAULT_VLM_AUTHOR_MODEL = "openai/gpt-5.4-nano"


@dataclass(frozen=True)
class GraphRunPaths:
    out_root: Path
    pdf_dir: Path
    text_dir: Path
    source_dir: Path
    graph_dir: Path
    run_stem: str
    loaded_graph_root: Path | None = None
    loaded_graph_json: Path | None = None
    output_dir_explicit: bool = False


@dataclass(frozen=True)
class GraphComponents:
    arxiv_client: ArxivFallbackClient
    dblp_client: DBLPClient
    semantic_scholar_client: SemanticScholarClient
    pdf_parser: PDFParser
    author_parser: VLMAuthorParser
    reference_parser: Any
    tex_reference_parser: Any


@dataclass(frozen=True)
class GraphBuildContext:
    paths: GraphRunPaths
    components: GraphComponents
    network: CitationNetwork
    expander: PaperExpander


class SeedInputError(ValueError):
    """Raised when the seed-list file argument is present but unusable."""


def add_common_builder_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "paper",
        nargs="?",
        default=None,
        metavar="PAPER",
        help=(
            "arXiv ID or title of the seed paper, or a .txt file with one "
            "seed paper title per line. Required unless --load-graph is given."
        ),
    )
    parser.add_argument(
        "--load-graph",
        metavar="JSON_PATH",
        help="Load an existing graph JSON, a graphs directory, or an output root and expand it further.",
    )
    parser.add_argument(
        "--arxiv-snapshot",
        default="resources/arxiv_metadata_dump/arxiv-metadata-oai-snapshot.json",
        metavar="PATH",
        help="Path to the local arXiv metadata snapshot used by ArxivBulkClient.",
    )
    parser.add_argument(
        "--skip-arxiv-online-client",
        action="store_true",
        default=False,
        help=(
            "Disable ArxivOnlineClient fallback and use only ArxivBulkClient "
            "for arXiv title and metadata lookup."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root directory for PDFs, texts, and graph files. Required unless --load-graph can supply a previous run root.",
    )
    parser.add_argument(
        "--save-graph",
        metavar="JSON_PATH",
        help="Explicit output path for the graph JSON.",
    )
    parser.add_argument(
        "--db-path",
        metavar="DB_PATH",
        help="Explicit path for the SQLite database file.",
    )
    parser.add_argument(
        "--vlm-reference-model",
        default=DEFAULT_VLM_REFERENCE_MODEL,
        help="VLM model string passed to VLMReferenceParser.",
    )
    parser.add_argument(
        "--vlm-author-model",
        default=DEFAULT_VLM_AUTHOR_MODEL,
        help="VLM model string passed to VLMAuthorParser.",
    )
    parser.add_argument(
        "--service-tier",
        default="default",
        choices=["default", "flex", "priority"],
        help="OpenAI service tier passed to the VLM author and reference parsers.",
    )
    parser.add_argument(
        "--vlm-parser-fallback",
        action="store_true",
        default=False,
        help="Fall back to VLMReferenceParser when no grammar parser matches.",
    )
    parser.add_argument(
        "--use-vlm-reference-parser",
        action="store_true",
        default=False,
        help="Use VLMReferenceParser directly instead of UnifiedReferenceParser.",
    )
    parser.add_argument(
        "--parse-reference-from-source",
        action="store_true",
        default=False,
        help=(
            "Try parsing references from the arXiv source archive first. "
            "When disabled, references are parsed directly from extracted PDF "
            "text with VLMReferenceParser."
        ),
    )
    parser.add_argument(
        "--no-skip-et-al",
        dest="resolve_et_al_authors",
        action="store_true",
        default=False,
        help=(
            "Resolve 'et al.' placeholders in author lists via DBLP / Semantic "
            "Scholar. Disabled by default to reduce external API calls."
        ),
    )

    limits = parser.add_argument_group("graph limits")
    limits.add_argument(
        "--max-papers",
        type=nonnegative_int,
        default=None,
        metavar="N",
        help=(
            "Maximum number of papers to fetch and mark explored. "
            "Frontier reference paper rows may exceed this cap. 0 means unlimited."
        ),
    )
    limits.add_argument(
        "--max-authors",
        type=nonnegative_int,
        default=None,
        metavar="N",
    )
    limits.add_argument(
        "--max-nodes",
        type=nonnegative_int,
        default=None,
        metavar="N",
    )
    limits.add_argument(
        "--max-refs",
        type=nonnegative_int,
        default=None,
        metavar="N",
    )
    limits.add_argument(
        "--max-depth",
        type=nonnegative_int,
        default=None,
        metavar="N",
        help="BFS depth for expanding citation references.",
    )
    limits.add_argument(
        "--unbounded-depth",
        action="store_true",
        default=False,
        help="Keep expanding citation references without a max-depth cutoff.",
    )

    parser.add_argument(
        "--max-num-checkpoints",
        type=int,
        default=2,
        metavar="N",
        help="Maximum number of checkpoint sets to keep. 0 keeps all.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        metavar="N",
        help="Save a checkpoint after every N newly explored papers. 0 disables mid-run checkpoints.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--raise-error",
        action="store_true",
        default=False,
        help="Re-raise fetch/parse failures instead of skipping them.",
    )


def add_parallel_builder_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=4,
        metavar="N",
        help="Number of parallel worker threads. Ignored unless --parallel is set.",
    )


def add_visualization_arguments(parser: argparse.ArgumentParser) -> None:
    visualisation = parser.add_argument_group("visualisation")
    visualisation.add_argument(
        "--viz",
        action="store_true",
        default=False,
        help="Render a PNG of the final network. Off by default in both serial and parallel modes.",
    )
    visualisation.add_argument("--viz-show", action="store_true")
    visualisation.add_argument(
        "--viz-layout",
        default="spring",
        choices=["spring", "kamada_kawai", "spectral", "random"],
    )
    visualisation.add_argument("--viz-dpi", type=int, default=180, metavar="N")
    visualisation.add_argument("--viz-output", metavar="PNG_PATH")


def add_strategy_arguments(parser: argparse.ArgumentParser) -> None:
    """Add traversal-strategy and concurrency-mode flags."""
    strategy = parser.add_argument_group("expansion strategy")
    strategy.add_argument(
        "--strategy",
        choices=["bfs", "priority"],
        default="bfs",
        help=(
            "Traversal order. 'bfs' uses a FIFO deque; "
            "'priority' uses a min-heap keyed on BFS depth."
        ),
    )
    strategy.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Enable parallel paper I/O via a thread pool (--workers controls pool size).",
    )


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def positive_int(value: str) -> int:
    parsed = nonnegative_int(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def build_serial_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a citation graph from a seed paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_builder_arguments(parser)
    return parser


def build_parallel_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a citation graph from a seed paper (parallel version).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_builder_arguments(parser)
    add_parallel_builder_arguments(parser)
    add_visualization_arguments(parser)
    return parser


def build_unified_parser() -> argparse.ArgumentParser:
    """Single parser that exposes every traversal strategy and concurrency mode."""
    parser = argparse.ArgumentParser(
        description="Build a citation graph from a seed paper (unified entry point).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_builder_arguments(parser)
    add_strategy_arguments(parser)
    add_parallel_builder_arguments(parser)
    add_visualization_arguments(parser)
    return parser


def build_graph_limits(args: argparse.Namespace) -> GraphLimits:
    return GraphLimits(
        max_papers=args.max_papers or 0,
        max_authors=args.max_authors or 0,
        max_nodes=args.max_nodes or 0,
        max_refs=args.max_refs or 0,
    )


def build_arxiv_client(args: argparse.Namespace) -> ArxivFallbackClient:
    online_client = None
    if not getattr(args, "skip_arxiv_online_client", False):
        online_client = ArxivOnlineClient()
    return ArxivFallbackClient(
        bulk_client=ArxivBulkClient(args.arxiv_snapshot),
        online_client=online_client,
    )


def build_reference_parser(
    model: str,
    service_tier: str,
    vlm_parser_fallback: bool,
    use_vlm_reference_parser: bool,
    *,
    force_vlm_fallback: bool = False,
) -> Any:
    vlm_parser_cls, unified_parser_cls = import_reference_parsers()
    if use_vlm_reference_parser:
        return vlm_parser_cls(model=model, service_tier=service_tier)
    if force_vlm_fallback:
        vlm_parser_fallback = True
    vlm_parser = (
        vlm_parser_cls(model=model, service_tier=service_tier)
        if vlm_parser_fallback
        else None
    )
    return unified_parser_cls(
        vlm_parser=vlm_parser,
        raise_error_without_fallback=not vlm_parser_fallback,
    )


def build_tex_reference_parser() -> Any:
    from src.reference_parser import TexReferenceParser

    return TexReferenceParser()


def build_runtime_reference_parser(args: argparse.Namespace) -> Any:
    parse_reference_from_source = getattr(args, "parse_reference_from_source", False)
    return build_reference_parser(
        model=args.vlm_reference_model,
        service_tier=args.service_tier,
        vlm_parser_fallback=args.vlm_parser_fallback,
        use_vlm_reference_parser=(
            args.use_vlm_reference_parser or not parse_reference_from_source
        ),
        force_vlm_fallback=parse_reference_from_source,
    )


def build_runtime_tex_reference_parser(args: argparse.Namespace) -> Any:
    if (
        hasattr(args, "parse_reference_from_source")
        and not args.parse_reference_from_source
    ):
        return None
    return build_tex_reference_parser()


def describe_reference_parser(args: argparse.Namespace) -> str:
    if not getattr(args, "parse_reference_from_source", False):
        return "VLMReferenceParser"
    if args.use_vlm_reference_parser:
        return "VLMReferenceParser"
    return "UnifiedReferenceParser (+VLM fallback)"


def build_graph_components(args: argparse.Namespace) -> GraphComponents:
    return GraphComponents(
        arxiv_client=build_arxiv_client(args),
        dblp_client=DBLPClient(),
        semantic_scholar_client=SemanticScholarClient(),
        pdf_parser=PDFParser(pipeline_type="standard"),
        author_parser=VLMAuthorParser(
            model=args.vlm_author_model,
            service_tier=args.service_tier,
        ),
        reference_parser=build_runtime_reference_parser(args),
        tex_reference_parser=build_runtime_tex_reference_parser(args),
    )


def prepare_run_paths(args: argparse.Namespace) -> GraphRunPaths:
    original_output_dir = args.output_dir
    out_root, resolved_graph_json = resolve_output_root(
        args.output_dir,
        args.load_graph,
    )
    loaded_graph_root = None
    if resolved_graph_json is not None:
        loaded_graph_root = infer_output_root_from_graph_path(resolved_graph_json)
        if original_output_dir is None:
            logger.info("Inferred output directory from --load-graph: {}", out_root)

    pdf_dir = out_root / "pdfs"
    text_dir = out_root / "texts"
    source_dir = out_root / "source"
    graph_dir = out_root / "graphs"
    for directory in (pdf_dir, text_dir, source_dir, graph_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_stem = build_graph_stem()
    return GraphRunPaths(
        out_root=out_root,
        pdf_dir=pdf_dir,
        text_dir=text_dir,
        source_dir=source_dir,
        graph_dir=graph_dir,
        run_stem=run_stem,
        loaded_graph_root=loaded_graph_root,
        loaded_graph_json=resolved_graph_json,
        output_dir_explicit=original_output_dir is not None,
    )


def same_path(left: Path, right: Path) -> bool:
    """Return whether two paths resolve to the same filesystem location."""
    return left.resolve() == right.resolve()


def should_fork_loaded_graph_db(args: argparse.Namespace, paths: GraphRunPaths) -> bool:
    """Return whether a loaded graph DB should be copied into this run."""
    del paths
    return bool(args.load_graph and not args.db_path)


def fork_loaded_graph_db(graph_json: Path, paths: GraphRunPaths) -> Path:
    """Copy the loaded graph's DB into the current output root and return it."""
    source_db, _resolution = resolve_db_path(
        graph_json,
        allow_any_checkpoint=True,
    )
    if not source_db.exists():
        raise FileNotFoundError(
            f"Database not found at {source_db}. "
            "Pass --db-path explicitly if it has moved."
        )

    target_db = paths.graph_dir / f"{paths.run_stem}.db"
    if same_path(source_db, target_db):
        return target_db

    target_db.parent.mkdir(parents=True, exist_ok=True)
    for stale_path in (
        target_db,
        Path(f"{target_db}-shm"),
        Path(f"{target_db}-wal"),
    ):
        if stale_path.exists():
            stale_path.unlink()

    with sqlite3.connect(str(source_db)) as source, sqlite3.connect(
        str(target_db)
    ) as target:
        source.backup(target)
        target.commit()

    logger.info(
        "Forked loaded graph database for this run: {} -> {}",
        source_db,
        target_db,
    )
    return target_db


def load_or_create_network(
    args: argparse.Namespace,
    paths: GraphRunPaths,
) -> CitationNetwork:
    limits = build_graph_limits(args)
    resolved_graph_json = getattr(paths, "loaded_graph_json", None)
    if resolved_graph_json is None and args.load_graph:
        resolved_graph_json = Path(args.load_graph)

    if resolved_graph_json is not None:
        logger.info("Loading existing graph: {}", resolved_graph_json)
        db_override = Path(args.db_path) if args.db_path else None
        if db_override is None and should_fork_loaded_graph_db(args, paths):
            db_override = fork_loaded_graph_db(resolved_graph_json, paths)
        network = load_graph(resolved_graph_json, limits=None, db_path=db_override)
        for attr in ("max_papers", "max_authors", "max_nodes", "max_refs"):
            value = getattr(args, attr)
            if value is not None:
                setattr(network.limits, attr, value)

        if getattr(args, "unbounded_depth", False):
            network.max_depth = None
            return network

        saved_depth = network.max_depth
        if saved_depth is None:
            if args.max_depth is not None:
                network.db.close()
                raise ValueError(
                    "Cannot run with bounded --max-depth on a graph already "
                    "built with unbounded max_depth."
                )
            return network

        effective_depth = args.max_depth if args.max_depth is not None else saved_depth
        if saved_depth > 0 and effective_depth < saved_depth:
            network.db.close()
            raise ValueError(
                "Cannot run with --max-depth "
                f"{effective_depth} on a graph already built with max_depth={saved_depth}. "
                f"Use --max-depth >= {saved_depth}."
            )
        network.max_depth = max(saved_depth, effective_depth)
        return network

    db_file = (
        Path(args.db_path) if args.db_path else paths.graph_dir / f"{paths.run_stem}.db"
    )
    network = CitationNetwork(db=DatabaseManager(db_file), limits=limits)
    network.max_depth = (
        None
        if getattr(args, "unbounded_depth", False)
        else args.max_depth if args.max_depth is not None else 1
    )
    return network


def build_graph_context(args: argparse.Namespace) -> GraphBuildContext:
    paths = prepare_run_paths(args)
    network = load_or_create_network(args, paths)
    try:
        components = build_graph_components(args)
        expander = PaperExpander(
            network=network,
            arxiv_client=components.arxiv_client,
            pdf_parser=components.pdf_parser,
            author_parser=components.author_parser,
            reference_parser=components.reference_parser,
            tex_reference_parser=components.tex_reference_parser,
            dblp_client=components.dblp_client,
            semantic_scholar_client=components.semantic_scholar_client,
            pdf_dir=paths.pdf_dir,
            text_dir=paths.text_dir,
            source_dir=paths.source_dir,
            resolve_et_al_authors=args.resolve_et_al_authors,
            raise_error=args.raise_error,
        )
    except Exception:
        network.db.close()
        raise
    return GraphBuildContext(
        paths=paths,
        components=components,
        network=network,
        expander=expander,
    )


def log_build_banner(
    title: str,
    args: argparse.Namespace,
    context: GraphBuildContext,
    *,
    workers: int | None = None,
) -> None:
    logger.info("=" * 48)
    logger.info(" {}", title)
    logger.info("  Seed   : {}", args.paper or f"(load) {args.load_graph}")
    logger.info("  Author Model: {}", args.vlm_author_model)
    logger.info("  Ref Model   : {}", args.vlm_reference_model)
    logger.info("  Ref    : {}", describe_reference_parser(args))
    logger.info(
        "  Source Refs: {}",
        "enabled" if args.parse_reference_from_source else "disabled",
    )
    if workers is not None:
        logger.info("  Workers: {}", workers)
    logger.info("  Limits : {}", context.network.limits.summary())
    logger.info("  Output : {}", context.paths.out_root)
    logger.info("  DB     : {}", context.network.db.db_path)
    logger.info("=" * 48)


def expand_seed(expander: PaperExpander, paper: str | None) -> str | None:
    if paper is None:
        return None
    seed = paper.strip()
    if ARXIV_ID_RE.match(seed):
        return expander.expand_by_arxiv_id(seed, depth=0)
    return expander.expand_by_title(preprocess_paper_title(seed), depth=0)


def resolve_seed_inputs(paper: str | None) -> list[str]:
    """Resolve the positional PAPER argument into one or more seed strings."""
    if paper is None:
        return []

    seed = paper.strip()
    if not seed:
        return []

    seed_path = Path(seed).expanduser()
    if seed_path.suffix.lower() != ".txt":
        return [seed]

    if not seed_path.exists():
        raise FileNotFoundError(f"Seed list file not found: {seed_path}")
    if not seed_path.is_file():
        raise SeedInputError(f"Seed list path is not a file: {seed_path}")

    seeds = [
        preprocess_paper_title(line.strip())
        for line in seed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not seeds:
        raise SeedInputError(f"Seed list file is empty: {seed_path}")

    logger.info("Loaded {} seed paper title(s) from {}", len(seeds), seed_path)
    return seeds


def expand_seeds(
    expander: PaperExpander,
    seeds: list[str],
    *,
    graph_dir: Path | None = None,
    run_stem: str = "citation_network",
    checkpoint_every: int = 0,
    max_num_checkpoints: int = 2,
) -> list[str]:
    """Expand seed strings and return successfully-created paper nodes."""
    seed_nids: list[str] = []
    seen: set[str] = set()
    papers_since_checkpoint = 0

    for index, seed in enumerate(seeds, start=1):
        logger.info("START seed {}/{} | {}", index, len(seeds), seed)
        explored_count_before = expander.net.explored_count()
        seed_nid = expand_seed(expander, seed)
        was_new = expander.net.explored_count() > explored_count_before
        if seed_nid is None:
            logger.error(
                "Could not expand seed {}/{}: {}",
                index,
                len(seeds),
                seed,
            )
            continue
        if was_new:
            papers_since_checkpoint += 1
            if (
                checkpoint_every > 0
                and graph_dir is not None
                and papers_since_checkpoint >= checkpoint_every
            ):
                ckpt_stem = build_graph_stem(run_stem)
                logger.info(
                    "Seed checkpoint; {} papers explored, saving {}",
                    expander.net.explored_count(),
                    ckpt_stem,
                )
                save_graph(
                    expander.net,
                    graph_dir,
                    stem=ckpt_stem,
                    write_gexf=False,
                )
                prune_checkpoints(
                    graph_dir,
                    base_stem=extract_base_stem(ckpt_stem),
                    max_keep=max_num_checkpoints,
                )
                papers_since_checkpoint = 0
        if seed_nid in seen:
            logger.debug(
                "Seed {}/{} resolved to duplicate node {}",
                index,
                len(seeds),
                seed_nid,
            )
            continue
        seen.add(seed_nid)
        seed_nids.append(seed_nid)

    return seed_nids


def has_seed_expansion_capacity(
    network: CitationNetwork,
    *,
    pending_count: int = 0,
) -> bool:
    """Return whether another seed expansion may be started."""
    max_papers = network.limits.max_papers
    if not max_papers:
        return True
    return network.explored_count() + pending_count < max_papers


def expand_seeds_parallel(
    context: GraphBuildContext,
    args: argparse.Namespace,
    seeds: list[str],
    *,
    graph_dir: Path | None = None,
    run_stem: str = "citation_network",
    checkpoint_every: int = 0,
    max_num_checkpoints: int = 2,
) -> list[str]:
    """Expand seed strings in parallel and return canonical paper nodes."""
    if not seeds:
        return []

    total = len(seeds)
    logger.info(
        "Starting parallel seed expansion; {} seed(s), workers: {}",
        total,
        args.workers,
    )

    parser_local = threading.local()
    file_locks_mu = threading.Lock()
    file_locks: dict[str, threading.Lock] = {}
    resolved_by_index: dict[int, str] = {}
    papers_since_checkpoint = 0
    next_seed_index = 0

    def get_file_lock(key: str) -> threading.Lock:
        with file_locks_mu:
            if key not in file_locks:
                file_locks[key] = threading.Lock()
            return file_locks[key]

    def get_worker_parsers() -> tuple[Any, Any, Any, Any]:
        parsers = getattr(parser_local, "parsers", None)
        if parsers is None:
            parsers = (
                PDFParser(pipeline_type="standard"),
                VLMAuthorParser(
                    model=args.vlm_author_model,
                    service_tier=args.service_tier,
                ),
                build_runtime_reference_parser(args),
                build_runtime_tex_reference_parser(args),
            )
            parser_local.parsers = parsers
        return parsers

    def fetch_seed_bundle(index: int, seed: str) -> Any:
        logger.info("START seed {}/{} | {}", index + 1, total, seed)
        seed_value = seed.strip()
        arxiv_id = seed_value if ARXIV_ID_RE.match(seed_value) else ""
        title = "" if arxiv_id else preprocess_paper_title(seed_value)
        (
            worker_pdf_parser,
            worker_author_parser,
            worker_ref_parser,
            worker_tex_ref_parser,
        ) = get_worker_parsers()
        return fetch_paper_data(
            arxiv_id=arxiv_id,
            title=title,
            nid=None,
            depth=0,
            arxiv_client=context.components.arxiv_client,
            pdf_parser=worker_pdf_parser,
            author_parser=worker_author_parser,
            ref_parser=worker_ref_parser,
            tex_reference_parser=worker_tex_ref_parser,
            pdf_dir=context.paths.pdf_dir,
            text_dir=context.paths.text_dir,
            source_dir=context.paths.source_dir,
            file_lock_fn=get_file_lock,
            dblp_client=context.components.dblp_client,
            semantic_scholar_client=context.components.semantic_scholar_client,
            raise_error=args.raise_error,
            resolve_et_al_authors=args.resolve_et_al_authors,
        )

    def try_submit_one(
        pool: ThreadPoolExecutor,
        pending: dict[Future, tuple[int, str]],
    ) -> bool:
        nonlocal next_seed_index
        if next_seed_index >= total:
            return False
        if not has_seed_expansion_capacity(
            context.network,
            pending_count=len(pending),
        ):
            return False

        index = next_seed_index
        seed = seeds[index]
        next_seed_index += 1
        future = pool.submit(fetch_seed_bundle, index, seed)
        pending[future] = (index, seed)
        return True

    def fill_pool(
        pool: ThreadPoolExecutor,
        pending: dict[Future, tuple[int, str]],
    ) -> None:
        while len(pending) < args.workers:
            if not try_submit_one(pool, pending):
                break

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending: dict[Future, tuple[int, str]] = {}
        fill_pool(pool, pending)

        while pending:
            done = wait(pending, return_when=FIRST_COMPLETED).done
            for future in done:
                index, seed = pending.pop(future)

                try:
                    bundle = future.result()
                except Exception as exc:
                    logger.warning(
                        "Seed worker for {}/{} raised: {}",
                        index + 1,
                        total,
                        exc,
                    )
                    if args.raise_error:
                        for queued_future in list(pending):
                            queued_future.cancel()
                        raise
                    logger.error(
                        "Could not expand seed {}/{}: {}", index + 1, total, seed
                    )
                    fill_pool(pool, pending)
                    continue

                try:
                    canonical_nid, was_new = apply_fetched_paper(
                        context.network, bundle
                    )
                except Exception as exc:
                    logger.warning(
                        "Seed merge failed for {}/{}: {}",
                        index + 1,
                        total,
                        exc,
                    )
                    if args.raise_error:
                        for queued_future in list(pending):
                            queued_future.cancel()
                        raise
                    logger.error(
                        "Could not expand seed {}/{}: {}", index + 1, total, seed
                    )
                    fill_pool(pool, pending)
                    continue

                if canonical_nid is None:
                    logger.error(
                        "Could not expand seed {}/{}: {}", index + 1, total, seed
                    )
                    fill_pool(pool, pending)
                    continue

                resolved_by_index[index] = canonical_nid

                if was_new:
                    papers_since_checkpoint += 1
                    if (
                        checkpoint_every > 0
                        and graph_dir is not None
                        and papers_since_checkpoint >= checkpoint_every
                    ):
                        ckpt_stem = build_graph_stem(run_stem)
                        logger.info(
                            "Seed checkpoint; {} papers explored, saving {}",
                            context.network.explored_count(),
                            ckpt_stem,
                        )
                        save_graph(
                            context.network,
                            graph_dir,
                            stem=ckpt_stem,
                            write_gexf=False,
                        )
                        prune_checkpoints(
                            graph_dir,
                            base_stem=extract_base_stem(ckpt_stem),
                            max_keep=max_num_checkpoints,
                        )
                        papers_since_checkpoint = 0

                fill_pool(pool, pending)

    if next_seed_index < total and not has_seed_expansion_capacity(context.network):
        logger.info(
            "Explored-paper capacity ({}) reached during seed expansion; {} seed(s) were not scheduled.",
            context.network.limits.max_papers,
            total - next_seed_index,
        )

    seed_nids: list[str] = []
    seen: set[str] = set()
    for index, _seed in enumerate(seeds, start=1):
        seed_nid = resolved_by_index.get(index - 1)
        if seed_nid is None:
            continue
        if seed_nid in seen:
            logger.debug(
                "Seed {}/{} resolved to duplicate node {}",
                index,
                total,
                seed_nid,
            )
            continue
        seen.add(seed_nid)
        seed_nids.append(seed_nid)

    return seed_nids


def collect_unexplored_papers(network: CitationNetwork) -> list[str]:
    return [
        nid
        for nid in network.G.nodes()
        if nid_type(nid) == NODE_PAPER and not network.is_explored(nid_db_id(nid))
    ]


def build_serial_traversal_kwargs(
    context: GraphBuildContext,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Kwargs accepted by ``expand_graph`` / ``expand_graph_priority``."""
    return dict(
        max_depth=context.network.max_depth,
        graph_dir=context.paths.graph_dir,
        run_stem=context.paths.run_stem,
        checkpoint_every=args.checkpoint_every,
        max_num_checkpoints=args.max_num_checkpoints,
    )


def build_parallel_traversal_kwargs(
    context: GraphBuildContext,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Kwargs accepted by ``expand_graph_parallel`` / ``expand_graph_parallel_priority``.

    The thread-local parser factories are bound here so callers do not need to
    re-create them at every call site.
    """

    def build_pdf_parser_for_worker() -> Any:
        return PDFParser(pipeline_type="standard")

    def build_author_parser_for_worker() -> Any:
        return VLMAuthorParser(
            model=args.vlm_author_model,
            service_tier=args.service_tier,
        )

    def build_reference_parser_for_worker() -> Any:
        return build_runtime_reference_parser(args)

    return dict(
        max_depth=context.network.max_depth,
        workers=args.workers,
        arxiv_client=context.components.arxiv_client,
        pdf_parser=context.components.pdf_parser,
        author_parser=context.components.author_parser,
        ref_parser=context.components.reference_parser,
        tex_ref_parser=context.components.tex_reference_parser,
        pdf_parser_factory=build_pdf_parser_for_worker,
        author_parser_factory=build_author_parser_for_worker,
        ref_parser_factory=build_reference_parser_for_worker,
        tex_ref_parser_factory=(
            build_tex_reference_parser if args.parse_reference_from_source else None
        ),
        pdf_dir=context.paths.pdf_dir,
        text_dir=context.paths.text_dir,
        source_dir=context.paths.source_dir,
        graph_dir=context.paths.graph_dir,
        run_stem=context.paths.run_stem,
        checkpoint_every=args.checkpoint_every,
        max_num_checkpoints=args.max_num_checkpoints,
        dblp_client=context.components.dblp_client,
        semantic_scholar_client=context.components.semantic_scholar_client,
        resolve_et_al_authors=args.resolve_et_al_authors,
        raise_error=args.raise_error,
    )


def save_network_output(
    network: CitationNetwork,
    paths: GraphRunPaths,
    save_graph_path: str | None,
    max_num_checkpoints: int,
) -> Path:
    explicit_json = Path(save_graph_path) if save_graph_path else None
    if explicit_json:
        explicit_json.parent.mkdir(parents=True, exist_ok=True)
        saved_json = save_graph(
            network,
            explicit_json.parent,
            stem=explicit_json.stem,
            json_path=explicit_json,
        )
        # The user owns the explicit path, so don't auto-prune it. But mid-run
        # checkpoints under paths.graph_dir would otherwise accumulate forever.
        prune_checkpoints(
            paths.graph_dir,
            base_stem=extract_base_stem(paths.run_stem),
            max_keep=max_num_checkpoints,
        )
        return saved_json

    save_stem = build_graph_stem(paths.run_stem)
    saved_json = save_graph(network, paths.graph_dir, stem=save_stem)
    prune_checkpoints(
        paths.graph_dir,
        base_stem=extract_base_stem(save_stem),
        max_keep=max_num_checkpoints,
    )
    return saved_json
