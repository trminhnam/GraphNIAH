#!/usr/bin/env python3
r"""Create randomized reachability experiments with v8 alpha-based tree growth.

Each component grows to its exact requested node count. A paper with no added
optional children has selection weight alpha; otherwise its weight is its
accepted optional-child count. Protected path edges do not count as children.

One run studies one factor (--vary): each example gets the studied value, and
every other factor is drawn at random. With --vary none, every factor is drawn
at random and --examples-per-value is the total number of examples.

Depth k counts the papers of the gold path (not edges). Each depth has its own
input file of paths with exactly
3k - 2 papers, used whole: a k-paper branch into the source, the gold path,
and a k-paper branch after the target. The paths of one question are drawn
uniformly from precomputed disjoint groups (endpoints/
precompute_disjoint_path_groups.py writes disjoint_path_groups.json next to
each file) and then shuffled. Node counts are per component (100-250) and
include papers that appear only in reference lists. Yes/no and counting
questions use two disjoint components (for YES, source and target share one
of them); multiple choice uses one per option. If the needles of a
multiple-choice example alone exceed the context, a warning is printed and a
new node count is drawn for that example (except when node count is the
studied value).

Arguments:
    --paths PATH [PATH ...]   Required. One papers-only sampled-path file per
                              question depth k (paths of 3k - 2 papers: 10, 22,
                              46, 94), each with its disjoint_path_groups.json.
    --vary FACTOR             Required. Factor to study, with its values:
                                question-depth  4, 8, 16, 32
                                node-count      100, 150, 200, 250
                                question-format yes/no, multiple choice, counting
                                distractor-type code, essays, books,
                                                disjoint-tree, wikipedia
                                none            draw every factor at random
    --examples-per-value N    Examples per studied value (default 200).
                              Total examples when --vary none.
                              Raising it appends to the experiment.
    --pool PATH               Required. Shared JSONL file of complete examples.
    --output PATH             Required. Experiment file (one qid per line);
                              <output>.settings.json is written next to it.
    --context-length N        Token budget per prompt, including the system
                              prompt and chat-template tokens (default 200000).
    --random-seed N           Seed for the plan and constructions (default 1).
    --graph-db PATH           Required. Graph SQLite database.
    --max-attempts N          Construction attempts per slot, each with a new
                              seed and the same config (default 500).
    --max-distractor-items N  Distractor items drawn per attempt before it
                              fails (default 8192).
    --workers N               Processes that build examples in parallel
                              (default 1); each opens its own database and
                              tokenizers.
    --cutoff-date YYYY-MM-DD  Earliest publication date of every paper in a
                              component (default 2026-04-01, the date the input
                              paths were sampled with).
    --alpha X                 Zero-child selection weight > 0 (default 0.1).
    --components-only         For the growth-parameter search: build only the
                              question's components (and check that their
                              needles fit), with no distractors or prompts.
                              The distractor folders and Wikipedia index are
                              then not read and need not exist.
    --code-dir, --essays-dir, --books-dir PATH
                              Distractor folders (default
                              resources/distracting_content/{code,essays,books}).
    --wikipedia-index PATH    Lucene passage index (default: the newest cached
                              BM25 index under
                              resources/distracting_content/wikipedia/bm25).

Example (activate the niah environment first)::

    python create_dataset.v1.py \
        --paths output/data.2026-04-01/hops={10,22,46,94}/papers-only/sampled_paths.json \
        --graph-db resources/semantic_scholar_graph.2026-04-01/semantic_scholar_paths.db \
        --vary question-depth --examples-per-value 200 --pool output/datasets/v1/pool.jsonl \
        --output output/datasets/v1/depth.jsonl --workers 4

The output lists qids in round-robin order over the studied values. Rerun the
same command to resume. A larger --examples-per-value appends slots and keeps
every earlier label; --max-attempts, --max-distractor-items, and --workers can
change freely. Changing --vary, --random-seed, --context-length, --pool, the
input locations, the tokenizers, or GENERATOR_VERSION restarts the experiment
(the old file is kept as *.previous) while reusing pool examples whose built
config matches exactly.
"""

from __future__ import annotations

import argparse
import bisect
import functools
import hashlib
import itertools
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import orjson

import build_full_prompts as packing
import create_distracting_content as file_distractors
from src.questions import reachability_construction as base
from src.questions import reachability_counting as counting
from src.questions import reachability_multiple as multiple
from src.questions import reachability_rendered_pipeline_v8 as growth
from src.questions import reachability_yesno as yesno
from src.logging.console import log_line
from src.questions.dataset_pool import (
    RetryableConstructionError,
    digest,
    held_output,
    portable_path,
    report,
    run_experiment,
    seed_for,
    settings_path_for,
    worker_ready,
)
from src.questions.disjoint_path_groups import (
    PathGroups,
    load_cache,
    load_depth_paths,
    open_graph_checker,
    prepare_path,
)
from src.questions.reachability_level_growth import GrowthBlockers, QuestionGrowthGroup
from src.setup import bootstrap_runtime
from src.setup.constants import NODE_PAPER

QUESTION_DEPTH_VALUES = [4, 8, 16, 32]
COMPONENT_NODE_COUNT_VALUES = [100, 150, 200, 250]
COMPONENT_NODE_COUNT_RANGE = [100, 250]  # Inclusive integer range for controls.
QUESTION_FORMAT_VALUES = [
    "reachability-yesno",
    "reachability-multiple",
    "reachability-counting",
]
FILE_DISTRACTOR_TYPES = ("code", "essays", "books")
DISTRACTOR_TYPE_VALUES = [*FILE_DISTRACTOR_TYPES, "disjoint-tree", "wikipedia"]
CONTEXT_LENGTH_VALUES = [200_000]  # Fixed per run; --context-length overrides it.

# Part of the pool key: bump it whenever a change alters the examples built.
GENERATOR_VERSION = "create_dataset.v1:4"
CONTEXT_ITEM_SEPARATOR = "\n\n"  # Between needles and distractor blocks.
COMPONENT_BUILD_ATTEMPTS = 50  # Path and growth tries inside one slot attempt.
EMPTY_SOURCE_REDRAWS = 100
MAX_NODE_COUNT_RESAMPLES = 20  # New node counts for one oversized MC example.
# What a --components-only record keeps of each component: enough to measure
# paper degrees and identify the tree.
SEARCH_COMPONENT_KEYS = (
    "tree_id",
    "source_node_id",
    "target_node_id",
    "component_size",
    "expanded_paper_ids",
    "references_by_paper",
)
FIXED_CONFIG = {
    "traversal_mode": "papers-only",
    "needle_type": "apa-describe",
    "needle_order": "random",
    "alpha": 0.1,
    "num_children_per_node": -1,
    "num_options": 4,
    "none_option": 0,
    "yes_ratio": 0.5,
    # Earliest publication date of every paper (scripts/path_sampling.new/2026-04-01).
    "cutoff_date": "2026-04-01",
    "only_explored": 1,
    "context_tail_margin": 100,
    "partial_negative_tree_packing": True,
}
DIMENSIONS = {
    "question-depth": ("question_depth", QUESTION_DEPTH_VALUES),
    "node-count": ("component_node_count", COMPONENT_NODE_COUNT_VALUES),
    "question-format": ("question_format", QUESTION_FORMAT_VALUES),
    "distractor-type": ("distractor_type", DISTRACTOR_TYPE_VALUES),
}


def implementation_fingerprint() -> str:
    """Hash the shared construction/rendering code; stored as a note per example."""
    root = Path(__file__).resolve().parent
    paths = [
        Path(__file__),
        Path(packing.__file__),
        Path(file_distractors.__file__),
        root / "create_needles.py",
    ]
    paths.extend(sorted((root / "src").rglob("*.py")))
    return digest(
        [
            [str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest()]
            for path in paths
        ]
    )


def validate_dimensions() -> None:
    if not QUESTION_DEPTH_VALUES or min(QUESTION_DEPTH_VALUES) < 2:
        raise ValueError("Question depths must be at least two path nodes.")
    lower, upper = COMPONENT_NODE_COUNT_RANGE
    minimum = 3 * max(QUESTION_DEPTH_VALUES) - 2
    if lower < minimum or upper < lower:
        raise ValueError(f"Independent sampling requires node counts >= {minimum}.")
    if not COMPONENT_NODE_COUNT_VALUES or any(
        not lower <= value <= upper for value in COMPONENT_NODE_COUNT_VALUES
    ):
        raise ValueError("Studied node counts must lie inside the control range.")
    for name, (_key, values) in DIMENSIONS.items():
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} values must be nonempty and unique.")


def assign_answer_conditions(
    group: list[dict[str, Any]], vary: str, value: Any, seed: int
) -> None:
    """Balance labels in index-order blocks, so a larger group keeps old labels.

    Yes/no examples go in pairs (one YES, one NO) and multiple-choice examples in
    groups of one per option, each block in its own random order.
    """
    for question_format, labels in (
        (yesno.QUESTION_TYPE, ["YES", "NO"]),
        (
            multiple.QUESTION_TYPE,
            list(multiple.option_labels(FIXED_CONFIG["num_options"])),
        ),
    ):
        members = [
            config for config in group if config["question_format"] == question_format
        ]
        for position, config in enumerate(members):
            block = list(labels)
            random.Random(
                seed_for(
                    seed,
                    "answers",
                    vary,
                    value,
                    question_format,
                    position // len(labels),
                )
            ).shuffle(block)
            config["answer_condition"] = block[position % len(labels)]


def experiment_configs(
    vary: str,
    examples_per_value: int,
    context_length: int,
    seed: int,
    *,
    alpha: float = FIXED_CONFIG["alpha"],
    cutoff_date: str = FIXED_CONFIG["cutoff_date"],
    components_only: bool = False,
) -> list[dict[str, Any]]:
    """Draw independent controls; slots go round-robin, so growth only appends.

    alpha and cutoff_date (and components_only, when set) go into every
    config, so the pool never mixes examples built with different ones.
    """
    validate_dimensions()
    growth.validate_alpha(alpha)
    growth.parse_cutoff_date(cutoff_date)
    growth_settings = {"alpha": float(alpha), "cutoff_date": cutoff_date}
    if components_only:
        growth_settings["components_only"] = True
    if examples_per_value <= 0 or context_length <= FIXED_CONFIG["context_tail_margin"]:
        raise ValueError(
            "Positive example counts and context length > tail margin required."
        )
    key, values = (None, [None]) if vary == "none" else DIMENSIONS[vary]
    groups = []
    for value in values:
        group = []
        for index in range(examples_per_value):
            rng = random.Random(seed_for(seed, "config", vary, value, index))
            config = {
                **FIXED_CONFIG,
                **growth_settings,
                "question_depth": rng.choice(QUESTION_DEPTH_VALUES),
                "component_node_count": rng.randint(*COMPONENT_NODE_COUNT_RANGE),
                "question_format": rng.choice(QUESTION_FORMAT_VALUES),
                "distractor_type": rng.choice(DISTRACTOR_TYPE_VALUES),
                "context_length": context_length,
            }
            if key is not None:
                config[key] = value
            group.append(config)
        assign_answer_conditions(group, vary, value, seed)
        groups.append(group)
    return [group[index] for index in range(examples_per_value) for group in groups]


def load_paths(paths: list[Path]) -> dict[int, tuple[Path, list[dict[str, Any]]]]:
    """One papers-only file per question depth k; every path has 3k - 2 papers."""
    by_depth: dict[int, tuple[Path, list[dict[str, Any]]]] = {}
    for path in paths:
        depth, entries = load_depth_paths(path)
        if depth not in QUESTION_DEPTH_VALUES:
            raise ValueError(
                f"{path}: {3 * depth - 2}-paper paths are for depth {depth}, "
                f"not one of {QUESTION_DEPTH_VALUES}."
            )
        if depth in by_depth:
            raise ValueError(f"Two paths files for depth {depth}: {by_depth[depth][0]}, {path}")
        by_depth[depth] = (path, entries)
    missing = [depth for depth in QUESTION_DEPTH_VALUES if depth not in by_depth]
    if missing:
        raise ValueError(
            f"No paths file for depth(s) {missing}; they need paths of "
            f"{[3 * depth - 2 for depth in missing]} papers."
        )
    return by_depth


@dataclass
class PathPool:
    """The input paths of one question depth, with their disjoint groups."""

    entries: dict[str, dict[str, Any]]
    groups: PathGroups


@dataclass
class ComponentBuilder:
    pools: dict[int, PathPool]
    context: Any
    facts: Any
    cutoff: Any
    max_attempts: int = COMPONENT_BUILD_ATTEMPTS
    candidate_cache: dict = field(default_factory=dict)

    def initial_tree(
        self, path_id: str, depth: int, rng: random.Random, failures: Counter
    ) -> tuple[Any, dict[str, Any]] | None:
        """Starting tree of one whole input path (a random orientation), or None."""
        prepared = prepare_path(
            self.pools[depth].entries[path_id],
            depth,
            context=self.context,
            facts=self.facts,
            cutoff=self.cutoff,
            pipeline=growth,
        )
        if isinstance(prepared, str):
            failures[prepared] += 1
            return None
        variants, sliced = prepared
        return rng.choice(variants).state, sliced

    def initial_trees(
        self,
        config: dict[str, Any],
        count: int,
        rng: random.Random,
        blockers: GrowthBlockers | None,
        failures: Counter,
    ) -> tuple[list[Any], list[dict[str, Any]]] | None:
        """Draw ``count`` disjoint input paths and prepare their starting trees.

        A group is drawn uniformly from the precomputed disjoint groups and
        shuffled, so any of its paths may become component 0. A single path (a
        disjoint-tree distractor) must also avoid the already grown trees in
        ``blockers``, which no cache can know, so it is checked here instead.
        """
        depth = config["question_depth"]
        pool = self.pools[depth]
        if count == 1:
            order = list(pool.groups.usable)
            rng.shuffle(order)
            for path_id in order:
                prepared = self.initial_tree(path_id, depth, rng, failures)
                if prepared is None:
                    continue
                if blockers is not None:
                    reason = blockers.conflict_reason(
                        prepared[0], include_author_identity=False
                    )
                    if reason:
                        failures[reason] += 1
                        continue
                return [prepared[0]], [prepared[1]]
            failures["not_enough_compatible_paths"] += 1
            return None
        path_ids = pool.groups.draw(count, rng)
        if path_ids is None:
            failures["no_disjoint_group"] += 1
            return None
        prepared = [self.initial_tree(path_id, depth, rng, failures) for path_id in path_ids]
        if any(item is None for item in prepared):
            return None
        trees = [tree for tree, _sliced in prepared]
        # The cache says these paths are disjoint; check again in case the
        # database changed since it was built.
        reason = QuestionGrowthGroup(
            trees=trees,
            targets=[],
            facts_reader=self.facts,
            include_author_identity=False,
            blockers=blockers,
        ).rebuild_registries()
        if reason:
            failures[f"cached_group_{reason}"] += 1
            return None
        return trees, [sliced for _tree, sliced in prepared]

    def build(
        self,
        config: dict[str, Any],
        count: int,
        rng: random.Random,
        blockers: GrowthBlockers | None = None,
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """Grow ``count`` disjoint components that each reach the node target."""
        failures: Counter = Counter()
        for _attempt in range(self.max_attempts):
            initial = self.initial_trees(config, count, rng, blockers, failures)
            if initial is None:
                continue
            trees, slices = initial
            session = growth.create_preferential_growth_session(
                trees,
                [growth.TARGET_COMPONENT] * count,
                facts_reader=self.facts,
                include_author_identity=False,
                num_children_per_node=config["num_children_per_node"],
                alpha=config["alpha"],
                random_seed=rng.getrandbits(64),
                group_key="dataset-v1",
                candidate_cache=self.candidate_cache,
                paper_candidate_filter=self.cutoff.allowed_paper_ids,
                blockers=blockers,
            )
            if session is None:
                failures["component_conflict"] += 1
                continue
            result = growth.advance_preferential_growth_session(
                session, config["component_node_count"]
            )
            if result.status != growth.GROWTH_REACHED:
                failures[result.status] += 1
                continue
            if not all(
                growth.validate_rendered_growth_tree(
                    tree, facts_reader=self.facts, include_author_identity=False
                )
                for tree in result.trees
            ):
                failures["invalid_final_tree"] += 1
                continue
            # Bound the full-adjacency cache across long dataset runs.
            if len(self.candidate_cache) > 8192:
                self.candidate_cache.clear()
            return list(result.trees), slices
        raise RetryableConstructionError(
            "component_build_failed",
            f"Cannot construct requested config {config}: {dict(failures)}",
        )


def render_tree(
    tree: Any, config: dict[str, Any], db: Any, seed: int
) -> tuple[dict[str, Any], list[str]]:
    payload = growth.serialize_negative_distractor_tree(
        tree,
        target_component_size=config["component_node_count"],
        num_children_per_node=config["num_children_per_node"],
        alpha=config["alpha"],
    )
    blocks = base.render_needles(
        paper_node_ids=payload["paper_block_ids"],
        authors_by_paper={},
        references_by_paper=payload["rendered_references_by_paper"],
        db=db,
        needle_type=config["needle_type"],
        random_seed=seed,
        display_all_database_authors=True,
    )
    return payload, blocks


def build_question(
    trees: list[Any], config: dict[str, Any], db: Any, seed: int
) -> dict[str, Any]:
    """Derive the answer from the final rendered graph, with no DFS-cost matching."""
    rendered = [render_tree(tree, config, db, seed) for tree in trees]
    payloads = [payload for payload, _blocks in rendered]
    blocks = [block for _payload, items in rendered for block in items]
    paper_ids = [paper for payload in payloads for paper in payload["paper_block_ids"]]
    references = {
        paper: refs
        for payload in payloads
        for paper, refs in payload["rendered_references_by_paper"].items()
    }
    visible = yesno.build_visible_graph(
        expanded_paper_ids=paper_ids,
        authors_by_paper={},
        references_by_paper=references,
    )
    components = list(nx.connected_components(visible.graph))
    if len(components) != len(trees) or any(
        len(nodes) != config["component_node_count"] for nodes in components
    ):
        raise RetryableConstructionError(
            "rendered_component_mismatch",
            f"expected {len(trees)} components of {config['component_node_count']} "
            f"nodes, got sizes {sorted(len(nodes) for nodes in components)}",
        )
    source = trees[0].source_node_id
    label = lambda paper: base.node_label(paper, db, {})
    question_format = config["question_format"]
    audit: dict[str, Any] = {"source_node_id": source}
    if question_format == yesno.QUESTION_TYPE:
        target = trees[0 if config["answer_condition"] == "YES" else 1].target_node_id
        answer = "YES" if nx.has_path(visible.graph, source, target) else "NO"
        if answer != config["answer_condition"]:
            raise RetryableConstructionError("yesno_answer_check_failed")
        text = yesno.build_question_text(
            source_label=label(source),
            target_label=label(target),
            include_authorship=False,
        )
        answer_type = yesno.ANSWER_TYPE
        audit["target_node_id"] = target
    elif question_format == multiple.QUESTION_TYPE:
        correct = config["answer_condition"]
        decoys = iter(trees[1:])
        options = [
            (option, (trees[0] if option == correct else next(decoys)).target_node_id)
            for option in multiple.option_labels(config["num_options"])
        ]
        if multiple.reachable_option_labels(
            visible, source_node_id=source, options=options
        ) != (correct,):
            raise RetryableConstructionError("multiple_choice_answer_check_failed")
        text = multiple.build_question_text(
            source_label=label(source),
            options=[(option, label(target)) for option, target in options],
            include_authorship=False,
        )
        answer, answer_type = correct, multiple.ANSWER_TYPE
        audit["options"] = [
            {"label": option, "node_id": target, "text": label(target)}
            for option, target in options
        ]
    else:
        answer = str(len(nx.node_connected_component(visible.graph, source)) - 1)
        text = counting.build_question_text(
            source_label=label(source), source_type=NODE_PAPER, include_authorship=False
        )
        answer_type = counting.ANSWER_TYPE
    return {
        "question_type": question_format,
        "question": text,
        "answer": [answer],
        "answer_type": answer_type,
        "needles": blocks,
        "needle_paper_ids": paper_ids,
        "components": payloads,
        "reachability": {
            **audit,
            "component_node_count": config["component_node_count"],
            "component_sizes": [len(nodes) for nodes in components],
            "visible_graph": yesno.serialize_visible_graph(
                visible, allow_hidden_citations=True
            ),
        },
    }


@dataclass
class DistractorSources:
    files: dict[str, list[Path]]
    wikipedia_index: Path
    wikipedia: Any = None

    def draw(self, kind: str, rng: random.Random) -> dict[str, Any]:
        """Draw one distractor item; an empty file or passage is drawn again."""
        for _draw in range(EMPTY_SOURCE_REDRAWS):
            item = (
                self.draw_wikipedia(rng)
                if kind == "wikipedia"
                else self.draw_file(kind, rng)
            )
            if item["blocks"]:
                return item
        raise RetryableConstructionError(
            f"empty_{kind}_sources",
            f"{EMPTY_SOURCE_REDRAWS} draws in a row had no text",
        )

    def draw_wikipedia(self, rng: random.Random) -> dict[str, Any]:
        if self.wikipedia is None:
            from src.retrieval.prebuilt_index import pyserini_runtime_environment

            with pyserini_runtime_environment(None):
                from pyserini.index.lucene import LuceneIndexReader

                self.wikipedia = LuceneIndexReader(str(self.wikipedia_index))
        reader = self.wikipedia.reader
        if reader.maxDoc() != reader.numDocs():
            raise ValueError("Use a Wikipedia passage index without deleted documents.")
        internal_id = rng.randrange(reader.maxDoc())
        document_id = self.wikipedia.convert_internal_docid_to_collection_docid(
            internal_id
        )
        document = self.wikipedia.doc(document_id)
        if document is None:
            raise ValueError(
                "Wikipedia index has missing documents; use an intact passage index."
            )
        raw = orjson.loads(document.raw())
        content = str(raw.get("contents") or raw.get("text") or "").strip()
        return {
            "source": {"type": "wikipedia", "id": document.docid()},
            "blocks": [content] if content else [],
        }

    def draw_file(self, kind: str, rng: random.Random) -> dict[str, Any]:
        path = rng.choice(self.files[kind])
        blocks = packing.load_path_distracting_content_paragraphs(
            path,
            distracting_content_type=kind,
            path_paragraph_cache=None,
        )
        return {
            "source": {"type": kind, "path": portable_path(path)},
            "blocks": list(blocks),
        }

    def close(self) -> None:
        if self.wikipedia is not None:
            self.wikipedia.reader.close()
            self.wikipedia = None


def open_sources(args: argparse.Namespace) -> DistractorSources:
    files = {}
    for kind in FILE_DISTRACTOR_TYPES:
        directory = getattr(args, f"{kind}_dir")
        extensions = (
            file_distractors.CODE_FILE_EXTS
            if kind == "code"
            else file_distractors.TEXT_FILE_EXTS
        )
        files[kind] = [
            path
            for path in file_distractors.list_distractor_paths(directory, extensions)
            if path.stat().st_size
        ]
        if not files[kind]:
            raise ValueError(f"No nonempty {kind} files in {directory}.")
    index = args.wikipedia_index
    if index is None:
        root = Path(file_distractors.DEFAULT_WIKIPEDIA_DISTRACTING_DIR) / "bm25"
        segments = sorted(
            root.rglob("segments_*"), key=lambda path: path.stat().st_mtime_ns
        )
        if not segments:
            raise ValueError(
                "No cached Wikipedia passage index; provide --wikipedia-index."
            )
        index = segments[-1].parent
    if not list(index.glob("segments_*")):
        raise ValueError(f"Not a Lucene Wikipedia index: {index}")
    return DistractorSources(files, index)


def count_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


@dataclass
class PromptPacker:
    """Mix needles and distractor blocks in random order under the token budget.

    Each block is tokenized once; those counts guide every guess, and each guess
    is checked by tokenizing the whole prompt, so the result never rests on an
    estimate.
    """

    question: dict[str, Any]
    config: dict[str, Any]
    tokenizers: list[tuple[str, Any]]
    seed: int
    blocks: list[str] = field(default_factory=list)
    priorities: list[float] = field(default_factory=list)
    needle_priorities: list[float] = field(init=False)
    separator_tokens: dict[str, int] = field(init=False)
    # Per tokenizer: running totals of block plus separator tokens, from 0.
    cumulative: dict[str, list[int]] = field(init=False)
    baseline: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.needle_priorities = [
            random.Random(seed_for(self.seed, "needle-order", i)).random()
            for i in range(len(self.question["needles"]))
        ]
        self.separator_tokens = {
            name: count_tokens(tokenizer, CONTEXT_ITEM_SEPARATOR)
            for name, tokenizer in self.tokenizers
        }
        self.cumulative = {name: [0] for name, _tokenizer in self.tokenizers}
        self.baseline = self.lengths(0)

    def messages(self, count: int) -> list[dict[str, str]]:
        system, reminder = packing.resolve_question_type_prompts(
            question_type=self.config["question_format"],
            needle_type=self.config["needle_type"],
            traversal_mode="papers-only",
            system_prompt=packing.SYSTEM_PROMPT,
            format_reminder=packing.FORMAT_REMINDER,
        )
        entries = [
            *zip(self.needle_priorities, self.question["needles"]),
            *zip(self.priorities[:count], self.blocks[:count]),
        ]
        text = packing.build_context_text(
            [block for _priority, block in sorted(entries)],
            context_item_prefix="",
            context_item_separator=CONTEXT_ITEM_SEPARATOR,
        )
        user = packing.append_format_reminder(
            packing.build_user_message(self.question["question"], text, False), reminder
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def lengths(self, count: int) -> dict[str, int]:
        messages = self.messages(count)
        return {
            name: packing.measure_tokenized_chat_length(tokenizer, messages)
            for name, tokenizer in self.tokenizers
        }

    def add(self, blocks: list[str]) -> None:
        for block in blocks:
            self.priorities.append(
                random.Random(
                    seed_for(self.seed, "block-order", len(self.blocks))
                ).random()
            )
            self.blocks.append(block)
            for name, tokenizer in self.tokenizers:
                totals = self.cumulative[name]
                totals.append(
                    totals[-1] + count_tokens(tokenizer, block) + self.separator_tokens[name]
                )

    def estimate(self, count: int) -> dict[str, int]:
        return {
            name: self.baseline[name] + totals[count]
            for name, totals in self.cumulative.items()
        }

    def predicted_prefix(self, count: int, lengths: dict[str, int]) -> int:
        """Largest prefix expected to fit, extrapolated from exact lengths at ``count``."""
        budget = self.config["context_length"]
        return min(
            bisect.bisect_right(totals, totals[count] + budget - lengths[name]) - 1
            for name, totals in self.cumulative.items()
        )

    def fitting_prefix(self) -> tuple[int, dict[str, int]]:
        """Keep every needle and the longest prefix of whole distractor blocks.

        Returns that prefix length and its exact token counts. The interval
        between the longest known fit and the shortest known overflow shrinks
        on every check; after a few guesses it falls back to bisection.
        """
        budget = self.config["context_length"]
        low, low_lengths = 0, self.baseline
        high = len(self.blocks) + 1
        count, lengths = low, low_lengths
        for step in itertools.count():
            guess = self.predicted_prefix(count, lengths) if step < 4 else (low + high) // 2
            guess = min(max(guess, low + 1), high - 1)
            if guess <= low:
                return low, low_lengths
            count, lengths = guess, self.lengths(guess)
            if max(lengths.values()) <= budget:
                low, low_lengths = count, lengths
            else:
                high = count


def construct_example(
    config: dict[str, Any],
    seed: int,
    builder: ComponentBuilder,
    sources: DistractorSources | None,
    tokenizers: list[tuple[str, Any]],
    max_distractor_items: int,
) -> dict[str, Any]:
    rng = random.Random(seed_for(seed, "main-components"))
    count = (
        config["num_options"]
        if config["question_format"] == multiple.QUESTION_TYPE
        else 2
    )
    trees, slices = builder.build(config, count, rng)
    question = build_question(trees, config, builder.context.db, seed)
    packer = PromptPacker(question, config, tokenizers, seed)
    budget, margin = config["context_length"], config["context_tail_margin"]
    if max(packer.baseline.values()) > budget:
        raise RetryableConstructionError(
            "needles_over_budget",
            f"Needles exceed the token budget: {max(packer.baseline.values()):,} > "
            f"{budget:,} tokens with {config['component_node_count']} nodes per component",
        )
    blockers = GrowthBlockers()
    for tree in trees:
        if blockers.add_tree(tree, include_author_identity=False):
            raise RetryableConstructionError("main_components_not_disjoint")
    if config.get("components_only"):
        # Growth-parameter search: degrees come from the components alone, so
        # skip distractors and prompt packing (the needle check above still ran).
        return {
            "question_type": question["question_type"],
            "question": question["question"],
            "answer": question["answer"],
            "answer_type": question["answer_type"],
            "needle_paper_ids": question["needle_paper_ids"],
            "components": [
                {key: payload[key] for key in SEARCH_COMPONENT_KEYS}
                for payload in question["components"]
            ],
            "reachability": {
                key: value
                for key, value in question["reachability"].items()
                if key != "visible_graph"
            },
            "config": config,
            "construction_seed": seed,
            "needle_tokens": packer.baseline,
            # There is no prompt, so the qid is built from this instead.
            "content_key": [question["question"], question["needle_paper_ids"]],
        }
    distractors = []
    rng = random.Random(seed_for(seed, "distractors"))
    for _index in range(max_distractor_items):
        if config["distractor_type"] == "disjoint-tree":
            try:
                generated, _slices = builder.build(config, 1, rng, blockers)
            except RetryableConstructionError as error:
                raise RetryableConstructionError(
                    "distractor_tree_build_failed", str(error)
                ) from error
            tree = generated[0]
            if blockers.add_tree(tree, include_author_identity=False):
                raise RetryableConstructionError("distractor_tree_not_disjoint")
            payload, blocks = render_tree(tree, config, builder.context.db, seed)
            item = {
                "source": {"type": "disjoint-tree", "tree_id": payload["tree_id"]},
                "tree": payload,
                "blocks": blocks,
            }
        else:
            item = sources.draw(config["distractor_type"], rng)
        distractors.append(item)
        packer.add(item["blocks"])
        # Tokenize the whole prompt only once the block counts say it is full.
        if max(packer.estimate(len(packer.blocks)).values()) < budget:
            continue
        selected, lengths = packer.fitting_prefix()
        if selected < len(packer.blocks) or max(lengths.values()) >= budget - margin:
            break
    else:
        raise RetryableConstructionError(
            "distractor_limit_reached",
            f"{max_distractor_items} items did not fill the context for config {config}",
        )
    remaining = selected
    included = []
    for item in distractors:
        used = min(remaining, len(item["blocks"]))
        if not used:
            break
        # The text is in user_message; keep only where each item came from.
        included.append(
            {
                **{key: value for key, value in item.items() if key != "blocks"},
                "used_block_count": used,
                "generated_block_count": len(item["blocks"]),
                "partial": used < len(item["blocks"]),
            }
        )
        remaining -= used
    messages = packer.messages(selected)
    return {
        **question,
        "config": config,
        "construction_seed": seed,
        "source_slices": slices,
        "distractors": included,
        "system_message": messages[0]["content"],
        "user_message": messages[1]["content"],
        "token_counts": lengths,
        "packing": {
            "target_tokens": budget,
            "actual_tokens": max(lengths.values()),
            "unused_tokens": budget - max(lengths.values()),
            "needle_blocks": len(question["needles"]),
            # Prompt length with needles only, per tokenizer.
            "needle_tokens": packer.baseline,
            "distractor_blocks": selected,
            "complete_distractor_items": sum(not item["partial"] for item in included),
            "partial_final_item": bool(included and included[-1]["partial"]),
            "stop_reason": "whole_block_boundary"
            if selected < len(packer.blocks)
            else "within_tail_margin",
        },
    }


def construct_fitting_example(
    config: dict[str, Any],
    seed: int,
    builder: ComponentBuilder,
    sources: DistractorSources | None,
    tokenizers: list[tuple[str, Any]],
    max_distractor_items: int,
    *,
    resample_node_count: bool,
) -> dict[str, Any]:
    """Build one example; oversized multiple choice gets a new node count.

    When the needles of a multiple-choice example alone exceed the context, a
    new node count per component is drawn from COMPONENT_NODE_COUNT_RANGE (with
    a new seed) and the rest of the config is kept. ``resample_node_count`` is
    False when node count is the studied value; then the error is raised so the
    slot retries with a new seed and the same config.
    """
    planned = config
    too_large = []
    for resample in range(MAX_NODE_COUNT_RESAMPLES + 1):
        try:
            example = construct_example(
                config,
                seed_for(seed, "node-count-resample", resample) if resample else seed,
                builder,
                sources,
                tokenizers,
                max_distractor_items,
            )
        except RetryableConstructionError as error:
            if (
                error.reason != "needles_over_budget"
                or config["question_format"] != multiple.QUESTION_TYPE
            ):
                raise
            if not resample_node_count:
                raise
            too_large.append(config["component_node_count"])
            count = random.Random(seed_for(seed, "node-count", resample)).randint(
                *COMPONENT_NODE_COUNT_RANGE
            )
            report(
                f"WARNING: {error.detail} (multiple choice); using {count} nodes "
                "per component instead"
            )
            config = {**config, "component_node_count": count}
            continue
        example["planned_config"] = planned
        example["node_counts_over_budget"] = too_large
        return example
    raise RetryableConstructionError(
        "needles_over_budget",
        f"{MAX_NODE_COUNT_RESAMPLES} new node counts did not fit the context",
    )


@dataclass
class Construction:
    """One process's database, distractor sources, and tokenizers."""

    builder: ComponentBuilder
    sources: DistractorSources | None  # None with --components-only
    tokenizers: list[tuple[str, Any]]
    graph_path: Path
    db_path: Path
    max_distractor_items: int
    code_hash: str
    resample_node_count: bool

    def construct(self, config: dict[str, Any], seed: int) -> dict[str, Any]:
        example = construct_fitting_example(
            config,
            seed,
            self.builder,
            self.sources,
            self.tokenizers,
            self.max_distractor_items,
            resample_node_count=self.resample_node_count,
        )
        example["generator_version"] = GENERATOR_VERSION
        # A note for audits only; it is not part of the pool key.
        example["implementation_sha256"] = self.code_hash
        return example

    def close(self) -> None:
        if self.sources is not None:
            self.sources.close()
        self.builder.context.db.close()


def open_pools(
    by_depth: dict[int, tuple[Path, list[dict[str, Any]]]],
    db_path: Path,
    cutoff_date: str,
) -> dict[int, PathPool]:
    """Each depth's paths with its cache, which must match these inputs."""
    pools = {}
    for depth, (paths_file, entries) in sorted(by_depth.items()):
        groups = load_cache(
            paths_file,
            database=portable_path(db_path),
            cutoff_date=cutoff_date,
            only_explored=FIXED_CONFIG["only_explored"],
            paths_name=portable_path(paths_file),
        )
        missing = {2, FIXED_CONFIG["num_options"]} - set(groups.groups)
        if missing:
            raise ValueError(f"{paths_file}: the cache has no groups of {sorted(missing)}.")
        entries_by_id = {entry["path_id"]: entry for entry in entries}
        if any(path_id not in entries_by_id for path_id in groups.usable):
            raise ValueError(f"{paths_file}: the cache names paths the file does not have.")
        pools[depth] = PathPool(entries_by_id, groups)
    return pools


def open_construction(
    args: argparse.Namespace, tokenizer_names: list[str] | None = None
) -> Construction:
    """Open this process's inputs. ``tokenizer_names`` (a worker's case) loads
    only those tokenizers; otherwise every model's is loaded and duplicates are
    collapsed."""
    by_depth = load_paths(args.paths)
    # Components-only examples use no distractors, so skip opening them.
    sources = None if args.components_only else open_sources(args)
    checker = open_graph_checker(
        args.graph_db, args.cutoff_date, FIXED_CONFIG["only_explored"]
    )
    context, graph_path, db_path = checker.context, checker.graph_path, checker.db_path
    try:
        builder = ComponentBuilder(
            open_pools(by_depth, db_path, args.cutoff_date),
            context,
            checker.facts,
            checker.cutoff,
        )
        tokenizers = packing.load_prompt_packing_tokenizers(
            args.context_length, only_slugs=tokenizer_names
        )
        if not tokenizers:
            raise ValueError("At least one prompt tokenizer is required.")
        # These tokenizers only count tokens, so lift the model limit that makes
        # Hugging Face warn about every long text.
        for _name, tokenizer in tokenizers:
            if hasattr(tokenizer, "model_max_length"):
                tokenizer.model_max_length = int(1e30)
    except BaseException:
        if sources is not None:
            sources.close()
        context.db.close()
        raise
    return Construction(
        builder,
        sources,
        tokenizers,
        graph_path,
        db_path,
        args.max_distractor_items,
        implementation_fingerprint(),
        # The studied value of a node-count experiment is never resampled.
        resample_node_count=args.vary != "node-count",
    )


# One Construction per process, opened on first use (the main process sets it).
WORKER_CONSTRUCTION: dict[str, Construction] = {}


def construct_in_worker(
    args: argparse.Namespace,
    tokenizer_names: list[str],
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Build one example, opening this process's database and tokenizers once.

    ``tokenizer_names`` are the distinct tokenizers the main process kept (part
    of the pool key). The worker loads only those, skipping the duplicates, and
    stops the run if it cannot load them all, instead of building examples that
    do not match the key.
    """
    construction = WORKER_CONSTRUCTION.get("current")
    if construction is None:
        # Only worker processes get here. The main process opened the same
        # inputs and printed the same messages, so keep this worker's copies
        # unless something differs.
        with held_output() as startup_lines:
            bootstrap_runtime("WARNING", setup_litellm_runtime=False)
            construction = open_construction(args, tokenizer_names)
        loaded = [name for name, _tokenizer in construction.tokenizers]
        if loaded != tokenizer_names:
            construction.close()
            raise RuntimeError(
                f"Worker {os.getpid()} loaded the tokenizers {loaded}, but the run "
                f"uses {tokenizer_names}. Its startup messages:\n"
                + "\n".join(startup_lines)
            )
        WORKER_CONSTRUCTION["current"] = construction
        worker_ready()
    return construction.construct(config, seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--paths", type=Path, nargs="+", required=True)
    parser.add_argument("--vary", choices=[*DIMENSIONS, "none"], required=True)
    parser.add_argument(
        "--examples-per-value",
        type=int,
        default=200,
        help=(
            "Examples per studied value, or total examples with --vary none. "
            "Raising it appends to the experiment."
        ),
    )
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH_VALUES[0])
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--graph-db", type=Path, required=True)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=500,
        help=(
            "Construction attempts per slot; each uses a new seed with the same "
            "config. Changing it never restarts the experiment."
        ),
    )
    parser.add_argument("--max-distractor-items", type=int, default=8192)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Processes that build examples in parallel. Each opens its own "
            "database connection, caches, and tokenizers, so memory grows with it."
        ),
    )
    parser.add_argument(
        "--cutoff-date",
        default=FIXED_CONFIG["cutoff_date"],
        help="Earliest publication date (YYYY-MM-DD) of every paper in a component.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=FIXED_CONFIG["alpha"],
        help="Selection weight for papers with no accepted optional children; must be positive.",
    )
    parser.add_argument(
        "--components-only",
        action="store_true",
        help=(
            "Growth-parameter search: build only the components (needles must "
            "still fit), with no distractors or prompts. Use a separate pool."
        ),
    )
    for kind in FILE_DISTRACTOR_TYPES:
        parser.add_argument(
            f"--{kind}-dir",
            type=Path,
            default=Path(
                getattr(file_distractors, f"DEFAULT_{kind.upper()}_DISTRACTING_DIR")
            ),
        )
    parser.add_argument(
        "--wikipedia-index",
        type=Path,
        help="Local Lucene passage index; otherwise discover the cached Wikipedia BM25 index.",
    )
    return parser


def experiment_settings(args: argparse.Namespace) -> dict[str, Any]:
    """What identifies an experiment besides its pool key and pool."""
    return {
        "vary": args.vary,
        "random_seed": args.random_seed,
        "context_length": args.context_length,
    }


def pool_compatibility(
    args: argparse.Namespace, construction: Construction
) -> dict[str, Any]:
    """The pool key: locations only, so the same files on another machine share
    the pool."""
    compatibility = {
        "generator_version": GENERATOR_VERSION,
        "paths": [portable_path(path) for path in args.paths],
        "database": portable_path(construction.db_path),
        "tokenizers": [name for name, _tokenizer in construction.tokenizers],
    }
    if construction.sources is not None:
        # Components-only examples never read distractors, so their pool
        # key leaves the distractor locations out.
        compatibility["distractor_sources"] = {
            **{
                kind: portable_path(getattr(args, f"{kind}_dir"))
                for kind in FILE_DISTRACTOR_TYPES
            },
            "wikipedia": portable_path(construction.sources.wikipedia_index),
        }
    return compatibility


def check_destinations(
    args: argparse.Namespace, construction: Construction, destinations: list[Path]
) -> None:
    protected = {
        path.resolve()
        for path in [*args.paths, construction.graph_path, construction.db_path]
    }
    if any(path.resolve() in protected for path in destinations):
        raise ValueError(
            "Dataset outputs must not overwrite input paths or the graph database."
        )


def log_path_groups(construction: Construction) -> None:
    for depth, pool in sorted(construction.builder.pools.items()):
        sizes = ", ".join(
            f"groups of {size}: "
            + (
                f"{info['count']:,} stored"
                if info["method"] == "stored"
                else f"about {info['estimated_count']:,.0f} (draw-and-retry)"
            )
            for size, info in pool.groups.payload["groups"].items()
        )
        log_line(
            f"Depth {depth}: {len(pool.groups.usable)} usable paths of "
            f"{3 * depth - 2} papers; {sizes}"
        )


def main() -> None:
    args = build_parser().parse_args()
    if min(args.max_attempts, args.max_distractor_items, args.workers) <= 0:
        raise ValueError("Attempt, distractor, and worker limits must be positive.")
    configs = experiment_configs(
        args.vary,
        args.examples_per_value,
        args.context_length,
        args.random_seed,
        alpha=args.alpha,
        cutoff_date=args.cutoff_date,
        components_only=args.components_only,
    )
    bootstrap_runtime(setup_litellm_runtime=False)
    log_line(
        "Opening the input paths, graph database, distractor sources, and tokenizers..."
    )
    construction = WORKER_CONSTRUCTION["current"] = open_construction(args)
    try:
        check_destinations(
            args,
            construction,
            [args.pool, args.output, settings_path_for(args.output)],
        )
        compatibility = pool_compatibility(args, construction)
        log_path_groups(construction)
        log_line(
            f"Experiment: vary {args.vary}, {args.examples_per_value} examples "
            + ("total" if args.vary == "none" else "per value")
            + f" ({len(configs)} slots), {args.context_length:,}-token context, "
            f"seed {args.random_seed}, cutoff {args.cutoff_date}, alpha "
            f"{args.alpha:g}, {GENERATOR_VERSION}"
            + (", components only." if args.components_only else ".")
        )
        if args.workers > 1:
            # Each worker opens its own; free this process's copy.
            WORKER_CONSTRUCTION.clear()
            construction.close()
        stats = run_experiment(
            args.pool,
            args.output,
            configs,
            experiment_settings(args),
            compatibility,
            functools.partial(construct_in_worker, args, compatibility["tokenizers"]),
            seed=args.random_seed,
            max_attempts=args.max_attempts,
            workers=args.workers,
        )
        retries = ", ".join(
            f"{reason} x{count}" for reason, count in sorted(stats["retries"].items())
        )
        log_line(
            f"Done: {len(configs)} slots in {args.output} ({stats['created']} new, "
            f"{stats['reused']} reused from the pool, {stats['resumed']} already "
            f"there). Retries: {retries or 'none'}. Pool: {args.pool}"
        )
    finally:
        if WORKER_CONSTRUCTION.pop("current", None) is not None:
            construction.close()


if __name__ == "__main__":
    main()
