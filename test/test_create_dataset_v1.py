"""V8 alpha growth in the node-target dataset pipeline."""

import importlib.util
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import orjson
import pytest

from src.questions import disjoint_path_groups as path_groups
from src.questions import reachability_rendered_pipeline_v8
from src.questions.dataset_pool import ExamplePool
from src.questions.explored import configure_explored_context
from src.setup.constants import EDGE_CITED_BY, EDGE_CITING
from test.test_build_full_prompts import FakeTokenizer
from test.dataset_fixtures import (
    PaperDB,
    check_components_only_needs_no_distractors,
    check_worker_needs_the_main_tokenizers,
    depth_entry,
)
from test.test_reachability_v5_pipeline import (
    make_long_entry,
    make_prefixed_graph,
    write_paths,
)

SCRIPT = Path(__file__).resolve().parents[1] / "create_dataset.v1.py"
SPEC = importlib.util.spec_from_file_location("dataset_v1", SCRIPT)
dataset = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dataset
SPEC.loader.exec_module(dataset)


def make_builder(
    prefixes=tuple("abcdefghijklmnopqrst"), max_attempts=100, graph=None, entries=None, depth=3
):
    """A v1 builder whose one depth pool comes from the shared cache code."""
    context = SimpleNamespace(
        db=PaperDB(),
        graph=graph or make_prefixed_graph(prefixes, add_growth_leaves=True),
    )
    configure_explored_context(context, 1)
    facts = dataset.growth.paper_facts_reader(context, include_author_identity=False)
    cutoff = dataset.growth.PaperDateCutoff(
        context.db, dataset.growth.parse_cutoff_date("2025-09-01"), {}
    )
    entries = entries or [depth_entry(prefix, depth) for prefix in prefixes]
    checks = path_groups.check_paths(
        entries, depth, context=context, facts=facts, cutoff=cutoff
    )
    pool = dataset.PathPool(
        {entry["path_id"]: entry for entry in entries},
        path_groups.PathGroups(path_groups.groups_payload(checks)),
    )
    return dataset.ComponentBuilder({depth: pool}, context, facts, cutoff, max_attempts)


def make_config(question_format="reachability-yesno", answer="YES"):
    return {
        **dataset.FIXED_CONFIG,
        "question_depth": 3,
        "component_node_count": 9,
        "question_format": question_format,
        "distractor_type": "essays",
        "context_length": 4000,
        "answer_condition": answer,
    }


@pytest.mark.parametrize("vary", list(dataset.DIMENSIONS))
def test_sampling_and_extension_preserve_v1_workflow(vary):
    small = dataset.experiment_configs(vary, 7, 200_000, 3)
    large = dataset.experiment_configs(vary, 200, 200_000, 3)
    assert small == large[: len(small)]
    assert all(config["alpha"] == 0.1 for config in small)
    key, values = dataset.DIMENSIONS[vary]
    for value in values:
        group = [config for config in large if config[key] == value]
        assert len(group) == 200
        for question_format in dataset.QUESTION_FORMAT_VALUES[:2]:
            counts = Counter(
                config["answer_condition"]
                for config in group
                if config["question_format"] == question_format
            )
            if counts:
                assert max(counts.values()) - min(counts.values()) <= 1


def test_alpha_cli_and_config():
    args = dataset.build_parser().parse_args(
        [
            "--paths",
            "paths.json",
            "--graph-db",
            "graph.db",
            "--vary",
            "question-depth",
            "--pool",
            "pool.jsonl",
            "--output",
            "depth.jsonl",
        ]
    )
    assert args.alpha == 0.1
    assert args.cutoff_date == "2026-04-01"
    assert dataset.QUESTION_DEPTH_VALUES == [4, 8, 16, 32]
    assert not hasattr(args, "theta") and not hasattr(args, "discount")
    assert dataset.growth is reachability_rendered_pipeline_v8
    configs = dataset.experiment_configs(
        args.vary, 2, 200_000, 1, alpha=0.8, components_only=True
    )
    assert all(c["alpha"] == 0.8 and c["components_only"] for c in configs)


def test_vary_none_draws_all_dimensions_and_preserves_extension():
    configs = dataset.experiment_configs("none", 200, 200_000, 3)
    assert len(configs) == 200
    assert configs == dataset.experiment_configs("none", 200, 200_000, 3)
    assert configs != dataset.experiment_configs("none", 200, 200_000, 4)
    assert dataset.experiment_configs("none", 7, 200_000, 3) == configs[:7]
    for dimension, (key, values) in dataset.DIMENSIONS.items():
        drawn = {config[key] for config in configs}
        if dimension == "node-count":
            assert all(100 <= count <= 250 for count in drawn)
            assert drawn - set(values)  # Controls use the full integer range.
        else:
            assert drawn == set(values)
    assert all(None not in config and "none" not in config for config in configs)
    assert all(config["context_length"] == 200_000 for config in configs)
    for question_format, labels in (
        ("reachability-yesno", {"YES", "NO"}),
        ("reachability-multiple", {"A", "B", "C", "D"}),
    ):
        answers = [
            config["answer_condition"]
            for config in configs
            if config["question_format"] == question_format
        ]
        for start in range(0, len(answers) - len(labels) + 1, len(labels)):
            assert set(answers[start : start + len(labels)]) == labels
    assert all(
        "answer_condition" not in config
        for config in configs
        if config["question_format"] == "reachability-counting"
    )


def test_vary_none_cli_keeps_fixed_settings():
    args = dataset.build_parser().parse_args(
        [
            "--paths",
            "paths.json",
            "--graph-db",
            "graph.db",
            "--vary",
            "none",
            "--examples-per-value",
            "11",
            "--pool",
            "pool.jsonl",
            "--output",
            "random.jsonl",
            "--alpha",
            "0.8",
            "--context-length",
            "100000",
            "--cutoff-date",
            "2025-09-01",
            "--components-only",
        ]
    )
    configs = dataset.experiment_configs(
        args.vary,
        args.examples_per_value,
        args.context_length,
        args.random_seed,
        alpha=args.alpha,
        cutoff_date=args.cutoff_date,
        components_only=args.components_only,
    )
    assert len(configs) == 11
    for config in configs:
        assert config["alpha"] == 0.8
        assert config["context_length"] == 100_000
        assert config["cutoff_date"] == "2025-09-01"
        assert config["components_only"] is True
        assert config["num_options"] == 4


@pytest.mark.parametrize("alpha", [0, -1, float("nan"), float("inf")])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        dataset.experiment_configs("question-depth", 2, 200_000, 1, alpha=alpha)


@pytest.mark.parametrize("alpha", [0.2, 0.8])
@pytest.mark.parametrize(
    "question_format,answer,count",
    [
        ("reachability-yesno", "YES", 2),
        ("reachability-yesno", "NO", 2),
        ("reachability-multiple", "C", 4),
        ("reachability-counting", None, 2),
    ],
)
def test_all_components_reach_exact_nodes_with_valid_answers(
    question_format, answer, count, alpha
):
    builder = make_builder()
    config = {**make_config(question_format, answer), "alpha": alpha}
    trees, slices = builder.build(config, count, random.Random(12))
    # v1 grows its starting trees with the v8 pipeline, from the shared cache.
    assert all(tree.cache_key.endswith("v8") for tree in trees)
    record = dataset.build_question(trees, config, builder.context.db, 12)
    assert record["answer"] == ["8" if answer is None else answer]
    assert record["reachability"]["component_sizes"] == [9] * count
    assert len(slices) == count
    assert "complexity" not in record["reachability"]
    for payload in record["components"]:
        assert payload["constructor_version"] == "v8"
        assert payload["alpha"] == alpha
        assert payload["target_component_size"] == payload["component_size"] == 9
        assert all(payload["rendered_references_by_paper"].values())
        restored = dataset.growth.deserialize_negative_distractor_tree(
            payload, facts_reader=builder.facts, include_author_identity=False
        )
        assert restored is not None and restored.component_size == 9
    assert len(record["needle_paper_ids"]) < count * 9


def test_saturation_below_node_target_is_rejected():
    builder = make_builder(("a",), max_attempts=2)
    config = {**make_config(), "component_node_count": 100}
    with pytest.raises(dataset.RetryableConstructionError, match="saturated"):
        builder.build(config, 1, random.Random(1))
    assert config["component_node_count"] == 100


@pytest.mark.parametrize("depth,nodes", [(4, 100), (32, 100), (32, 300)])
def test_whole_production_paths_reach_exact_node_targets(tmp_path, depth, nodes):
    entry = make_long_entry("a")
    steps, adjacency = [], {}
    for index in range(450):
        paper = f"paper:long-{index}"
        adjacency[paper] = []
        if index:
            previous = f"paper:long-{index - 1}"
            adjacency[previous].append((paper, {"edge_type": EDGE_CITING}))
            adjacency[paper].append((previous, {"edge_type": EDGE_CITED_BY}))
        if index < 3 * depth - 2:
            steps.append(
                {
                    "step": index,
                    "node": {
                        "node_id": paper,
                        "node_type": "paper",
                        "title": f"Title {index}",
                    },
                    "edge_taken": None
                    if not index
                    else {
                        "from": previous,
                        "to": paper,
                        "edge_type": EDGE_CITING,
                    },
                }
            )
    entry["path"] = steps
    builder = make_builder(
        ("a",), graph=SimpleNamespace(adj=adjacency), entries=[entry], depth=depth
    )
    config = {**make_config(), "question_depth": depth, "component_node_count": nodes}
    trees, _slices = builder.build(config, 1, random.Random(42))
    assert trees[0].component_size == nodes
    assert len(trees[0].gold_paper_ids) == 3 * depth - 2
    assert len(trees[0].witness) == depth
    path = tmp_path / "paths.json"
    write_paths(path, [{**entry, "path": steps[:-1]}])  # 3k - 3 papers fit no depth.
    with pytest.raises(ValueError, match="not 3k - 2"):
        dataset.load_paths([path])


@pytest.mark.parametrize("distractor_type", ["essays", "disjoint-tree"])
def test_prompt_packing_and_pool_resume(tmp_path, distractor_type):
    source = tmp_path / "essay.txt"
    source.write_text(("Example distractor paragraph. " * 10 + "\n\n") * 10)
    builder = make_builder()
    config = {**make_config(), "distractor_type": distractor_type}
    tokenizers = [("fake", FakeTokenizer())]
    construction = dataset.Construction(
        builder,
        dataset.DistractorSources({"essays": [source]}, tmp_path),
        tokenizers,
        tmp_path,
        tmp_path,
        30,
        "fixture",
        False,
    )
    pool, output = tmp_path / "pool.jsonl", tmp_path / "experiment.jsonl"
    compatibility = {"generator_version": dataset.GENERATOR_VERSION}
    kwargs = {"seed": 7, "max_attempts": 5}
    dataset.run_experiment(
        pool, output, [config], {}, compatibility, construction.construct, **kwargs
    )
    record = orjson.loads(pool.read_bytes().splitlines()[0])
    assert record["generator_version"] == dataset.GENERATOR_VERSION
    assert record["planned_config"] == record["config"] == config
    assert all(needle in record["user_message"] for needle in record["needles"])
    messages = [
        {"role": role, "content": record[f"{role}_message"]}
        for role in ("system", "user")
    ]
    assert (
        dataset.packing.measure_tokenized_chat_length(tokenizers[0][1], messages)
        == record["packing"]["actual_tokens"]
        <= 4000
    )
    seen = {paper for tree in record["components"] for paper in tree["paper_depth"]}
    for item in record["distractors"]:
        if distractor_type == "disjoint-tree":
            tree = item["tree"]
            assert tree["constructor_version"] == "v8"
            assert tree["alpha"] == 0.1 and tree["component_size"] == 9
            assert not seen.intersection(tree["paper_depth"])
            seen.update(tree["paper_depth"])
    stats = dataset.run_experiment(
        pool, output, [config], {}, compatibility, construction.construct, **kwargs
    )
    assert stats["resumed"] == 1
    with pool.open("rb") as handle:
        indexed = ExamplePool(handle)
        indexed.scan(repair=False)
        assert (
            indexed.choose(config, compatibility, set(), random.Random(1))
            == record["qid"]
        )
        assert (
            indexed.choose(
                config,
                {"generator_version": "incompatible-generator-version"},
                set(),
                random.Random(1),
            )
            is None
        )
        assert (
            indexed.choose(
                {**config, "alpha": 0.8}, compatibility, set(), random.Random(1)
            )
            is None
        )


def test_components_only_preserves_exact_graph_without_distractors():
    builder = make_builder()
    config = {**make_config(), "components_only": True}
    record = dataset.construct_example(
        config,
        7,
        builder,
        None,
        [("fake", FakeTokenizer())],
        30,
    )
    assert record["reachability"]["component_sizes"] == [9, 9]
    assert "user_message" not in record and "distractors" not in record
    for component in record["components"]:
        graph = nx.Graph()
        for owner, references in component["references_by_paper"].items():
            graph.add_edges_from((owner, cited) for cited in references)
        assert nx.is_tree(graph) and len(graph) == 9


def test_vary_none_builds_resumes_and_extends(tmp_path, monkeypatch):
    # Small real components keep this integration test independent of local data.
    monkeypatch.setattr(dataset, "QUESTION_DEPTH_VALUES", [3])
    monkeypatch.setattr(dataset, "COMPONENT_NODE_COUNT_VALUES", [9])
    monkeypatch.setattr(dataset, "COMPONENT_NODE_COUNT_RANGE", [9, 9])
    construction = dataset.Construction(
        make_builder(),
        None,
        [("fake", FakeTokenizer())],
        tmp_path,
        tmp_path,
        30,
        "fixture",
        True,
    )
    configs = dataset.experiment_configs("none", 12, 4000, 3, components_only=True)
    assert {config["question_format"] for config in configs} == set(
        dataset.QUESTION_FORMAT_VALUES
    )
    pool, output = tmp_path / "pool.jsonl", tmp_path / "experiment.jsonl"
    settings = {"vary": "none", "random_seed": 3, "context_length": 4000}
    compatibility = {"generator_version": dataset.GENERATOR_VERSION}
    dataset.run_experiment(
        pool,
        output,
        configs[:5],
        settings,
        compatibility,
        construction.construct,
        seed=3,
        max_attempts=5,
    )
    previous = output.read_bytes()
    stats = dataset.run_experiment(
        pool,
        output,
        configs,
        settings,
        compatibility,
        construction.construct,
        seed=3,
        max_attempts=5,
    )
    assert stats["resumed"] == 5
    assert output.read_bytes().startswith(previous)
    qids = [orjson.loads(line)["qid"] for line in output.read_bytes().splitlines()]
    assert len(qids) == len(set(qids)) == 12
    with pool.open("rb") as handle:
        indexed = ExamplePool(handle)
        indexed.scan(repair=False)
        assert [indexed.get(qid)["config"] for qid in qids] == configs


def test_components_only_needs_no_distractor_folders(monkeypatch, tmp_path):
    check_components_only_needs_no_distractors(dataset, monkeypatch, tmp_path)


def test_worker_needs_the_main_tokenizers(monkeypatch):
    check_worker_needs_the_main_tokenizers(dataset, monkeypatch)
