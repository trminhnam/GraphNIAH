"""run_eval.py scoring and the v1 experiment scripts that call it."""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import orjson
import pytest

import run_eval

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ("v1",)
# Growth parameters of each version: alpha (v8 weights) or theta/discount.
ALPHA_VERSIONS = ("v1", "v3", "v5")


def load_dataset(version: str):
    spec = importlib.util.spec_from_file_location(
        f"dataset_{version}_scripts", ROOT / f"create_dataset.{version}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DATASETS = {version: load_dataset(version) for version in VERSIONS}


def scripts_dir(version: str) -> Path:
    return ROOT / "scripts" / "experiments" / version


def shell_values(command: str, env: dict[str, str] | None = None) -> list[str]:
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()[-1].split("|")


def record(qid, question_format, answer, raw_answer, *, depth=2, nodes=120, distractor="code"):
    answer_type = {
        "reachability-yesno": "REACHABILITY_YESNO",
        "reachability-multiple": "REACHABILITY_MULTIPLE",
        "reachability-counting": "REACHABILITY_COUNTING",
    }[question_format]
    return {
        "qid": qid,
        "question_type": question_format,
        "answer_type": answer_type,
        "answer": [answer],
        "config": {
            "question_format": question_format,
            "question_depth": depth,
            "component_node_count": nodes,
            "distractor_type": distractor,
        },
        "raw_answer": raw_answer,
    }


PREDICTIONS = {
    "schema_version": 1,
    "metadata": {"model": "fake-model"},
    "records": [
        record("a", "reachability-yesno", "YES", "\\answer{YES}", depth=2, nodes=120),
        record("b", "reachability-yesno", "NO", "\\answer{YES}", depth=4, nodes=200),
        record("c", "reachability-multiple", "C", "\\answer{C}", depth=2, nodes=180),
        record("d", "reachability-counting", "119", "\\answer{119}", depth=4, nodes=250),
        record("e", "reachability-counting", "99", None, depth=4, nodes=100),
    ],
}


def test_reports_accuracy_by_studied_value_only():
    report = run_eval.evaluate(PREDICTIONS, "question-depth")
    assert report["overall"]["total"] == 4 and report["overall"]["correct"] == 3
    assert report["overall"]["accuracy"] == 0.75
    assert report["overall"]["ci95"] == pytest.approx(
        [value / 100 for value in run_eval.wilson_ci_bounds_pct(3, 4)]
    )
    assert list(report["by_studied_value"]) == ["2", "4"]
    assert report["by_studied_value"]["2"]["accuracy"] == 1.0
    assert report["by_studied_value"]["4"]["correct"] == 1
    assert report["by_studied_value"]["4"]["accuracy"] == 0.5
    # Factors drawn at random get no breakdown of their own.
    assert "by_factor" not in report
    assert report["by_gold_answer"]["reachability-yesno"]["NO"]["correct"] == 0
    assert report["counts"]["failed_requests"] == 1
    assert report["counts"]["unparseable"] == 0
    assert [item["correct"] for item in report["records"]] == [
        True,
        False,
        True,
        True,
        False,
    ]


def test_studied_node_count_keeps_exact_values():
    report = run_eval.evaluate(PREDICTIONS, "node-count")
    assert list(report["by_studied_value"]) == ["100", "120", "180", "200", "250"]
    assert report["by_studied_value"]["100"] == {
        "total": 0,
        "correct": 0,
        "accuracy": None,
        "ci95": None,
    }


def test_failed_requests_are_excluded_but_unparseable_answers_are_incorrect():
    predictions = {
        **PREDICTIONS,
        "records": [
            record("correct", "reachability-multiple", "C", "\\answer{C}"),
            record("unparseable", "reachability-multiple", "C", "I cannot decide."),
            record("failed", "reachability-multiple", "C", None),
        ],
    }
    report = run_eval.evaluate(predictions, "question-depth")
    for stats in (
        report["overall"],
        report["by_studied_value"]["2"],
        report["by_gold_answer"]["reachability-multiple"]["C"],
    ):
        assert stats["total"] == 2
        assert stats["accuracy"] == 0.5
    assert report["counts"]["failed_requests"] == 1
    assert report["counts"]["unparseable"] == 1
    assert len(report["records"]) == 3


@pytest.mark.parametrize("records", [[], [PREDICTIONS["records"][-1]]])
def test_no_answered_requests_have_no_accuracy_or_interval(records):
    report = run_eval.evaluate({**PREDICTIONS, "records": records}, "none")
    assert report["overall"] == {
        "total": 0,
        "correct": 0,
        "accuracy": None,
        "ci95": None,
    }
    assert report["counts"]["failed_requests"] == len(records)
    assert report["counts"]["unparseable"] == 0
    assert run_eval.format_row("overall", report["overall"]).strip().endswith("n=0")


def test_vary_none_reports_overall_and_gold_answers_without_studied_values():
    report = run_eval.evaluate(PREDICTIONS, "none")
    assert report["vary"] == "none"
    assert report["by_studied_value"] == {}
    studied = run_eval.evaluate(PREDICTIONS, "question-depth")
    for key in ("overall", "by_gold_answer", "counts", "records"):
        assert report[key] == studied[key]


@pytest.mark.parametrize("vary", ["question-format", "none"])
@pytest.mark.parametrize(
    "costs, expected_cost, expected_text",
    [
        ([0.1] * 5, 0.5, "$0.500000 | priced records=5/5"),
        ([0.0] * 5, 0.0, "$0.000000 | priced records=5/5"),
        ([0.1, None, None, None, 0.2], 0.3, "$0.300000 (partial) | priced records=2/5"),
        ([None] * 5, None, "unavailable | priced records=0/5"),
    ],
)
def test_cli_writes_the_report(tmp_path, vary, costs, expected_cost, expected_text):
    predictions = tmp_path / "model=fake.json"
    predictions.write_bytes(
        orjson.dumps(
            {
                **PREDICTIONS,
                "metadata": {
                    **PREDICTIONS["metadata"],
                    "run_accounting": {"total_total_cost": 0.0},
                },
                "records": [
                    {**item, "cost": {"total_cost": cost}, "inference_source": "pool"}
                    for item, cost in zip(PREDICTIONS["records"], costs)
                ],
            }
        )
    )
    output = tmp_path / "eval" / "model=fake.json"
    result = subprocess.run(
        [
            sys.executable,
            "run_eval.py",
            "--predictions",
            str(predictions),
            "--vary",
            vary,
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = orjson.loads(output.read_bytes())
    assert report["model"] == "fake-model"
    assert report["vary"] == vary
    assert report["overall"]["total"] == 4
    assert report["overall"]["accuracy"] == 0.75
    assert report["cost"]["total_cost"] == (
        pytest.approx(expected_cost) if expected_cost is not None else None
    )
    assert report["cost"]["records_without_cost"] == costs.count(None)
    assert report["cost"]["cost_complete"] == (None not in costs)
    assert f"total response cost (USD): {expected_text}" in result.stdout
    assert list(report["by_studied_value"]) == (
        []
        if vary == "none"
        else [
            "reachability-counting",
            "reachability-multiple",
            "reachability-yesno",
        ]
    )


@pytest.mark.parametrize("version", VERSIONS)
def test_every_studied_factor_has_one_create_data_script(version):
    dataset = DATASETS[version]
    varies = {}
    # create_data.0_* prepares inputs; 1 and up each build one experiment.
    for script in scripts_dir(version).glob("create_data.[1-5]_*.sh"):
        if ".none_option." in script.name:
            continue
        match = re.search(r'^VARY="([a-z-]+)"$', script.read_text(), re.MULTILINE)
        varies[script.name] = match.group(1)
    expected = [*dataset.DIMENSIONS, "none"]
    assert sorted(varies.values()) == sorted(expected)
    for name, vary in varies.items():
        experiment = "reasoning_effort" if vary == "none" else vary
        assert name.endswith(f"_{experiment.replace('-', '_')}.sh")


@pytest.mark.parametrize("version", VERSIONS)
def test_config_lists_every_factor_and_run_eval_knows_them(version):
    dataset = DATASETS[version]
    config = (scripts_dir(version) / "config.sh").read_text()
    block = re.search(r"^VARY_VALUES=\(\n(.*?)\n\)", config, re.MULTILINE | re.DOTALL)
    listed = [line.strip() for line in block.group(1).splitlines() if line.strip()]
    assert listed == [*dataset.DIMENSIONS, "reasoning_effort"]
    assert {key: field for key, (field, _values) in dataset.DIMENSIONS.items()} == (
        run_eval.VARY_KEYS
    )


@pytest.mark.parametrize("version", VERSIONS)
def test_config_uses_the_2026_04_01_sampled_paths_and_database(version):
    dataset = DATASETS[version]
    sampling = shell_values(
        "source scripts/path_sampling.new/2026-04-01/path_sampling_settings.sh; "
        'echo "$DATA_DIR|$SHARED_DB_PATH|$CUTOFF_DATE|${HOP_COUNTS[*]}|$PATH_CATEGORY"'
    )
    experiments = shell_values(
        f"source scripts/experiments/{version}/config.sh; "
        'echo "$DATA_DIR|$GRAPH_DB|$CUTOFF_DATE|${HOP_COUNTS[*]}|$PATH_CATEGORY|'
        '${PATHS_FILES[*]}"'
    )
    assert experiments[:5] == sampling
    hop_counts = [int(count) for count in sampling[3].split()]
    # One file per question depth k, with paths of 3k - 2 papers.
    assert hop_counts == [3 * depth - 2 for depth in dataset.QUESTION_DEPTH_VALUES]
    assert experiments[5].split() == [
        f"{sampling[0]}/hops={count}/papers-only/sampled_paths.json"
        for count in hop_counts
    ]
    assert dataset.FIXED_CONFIG["cutoff_date"] == sampling[2]


@pytest.mark.parametrize("version", VERSIONS)
def test_cs_paths_have_their_own_inputs_and_outputs(version):
    """The default all-field data never shares a folder with the cs data."""
    command = (
        f"source scripts/experiments/{version}/config.sh; "
        'echo "$DATA_DIR|$GRAPH_DB|$SETTINGS_DIR|$GROWTH_SETTINGS_DIR"'
    )
    env = {key: value for key, value in os.environ.items() if key != "PATH_CATEGORY"}
    settings = "hops=10-22-46-94/context_length=200k/seed=1"
    root = f"./output/experiments.{version}"
    # The default (all) reads and writes the original folders.
    assert shell_values(command, env=env) == [
        "./output/data.2026-04-01",
        "./resources/semantic_scholar_graph.2026-04-01/semantic_scholar_paths.db",
        f"{root}/cutoff_date=2026-04-01/{settings}",
        f"{root}/growth_search/cutoff_date=2026-04-01/{settings}",
    ]
    assert shell_values(command, env={**env, "PATH_CATEGORY": "cs"}) == [
        "./output/data.2026-04-01.cs",
        "./resources/semantic_scholar_graph.2026-04-01.cs/semantic_scholar_paths.db",
        f"{root}/cutoff_date=2026-04-01/category=cs/{settings}",
        f"{root}/growth_search/cutoff_date=2026-04-01/category=cs/{settings}",
    ]


def test_all_versions_read_the_same_inputs_and_caches():
    fields = (
        'echo "$DATA_DIR|$GRAPH_DB|$CUTOFF_DATE|${PATHS_FILES[*]}|$CONTEXT_LENGTH|'
        '$RANDOM_SEED|$GROUP_LIMIT|${MODEL_SLUGS[*]}|${GLOBAL_DISTRIBUTIONS[*]}"'
    )
    values = [
        shell_values(f"source scripts/experiments/{version}/config.sh; {fields}")
        for version in VERSIONS
    ]
    assert all(value == values[0] for value in values)
    # Each version writes its own outputs.
    roots = [
        shell_values(
            f'source scripts/experiments/{version}/config.sh; '
            'echo "$OUTPUT_ROOT|$GROWTH_SEARCH_ROOT|$INFERENCE_POOL"'
        )
        for version in VERSIONS
    ]
    assert roots == [
        [
            f"./output/experiments.{version}",
            f"./output/experiments.{version}/growth_search",
            f"./output/experiments.{version}/inference_pool.jsonl",
        ]
        for version in VERSIONS
    ]


@pytest.mark.parametrize("version", VERSIONS)
def test_configs_default_to_the_python_growth_settings(version):
    config = DATASETS[version].FIXED_CONFIG
    source = f"source scripts/experiments/{version}/config.sh; "
    if version in ALPHA_VERSIONS:
        alpha, folder = shell_values(source + 'echo "$ALPHA|$(growth_dir)"')
        assert float(alpha) == config["alpha"]
        assert folder.endswith(f"/alpha={alpha}")
    else:
        theta, discount, folder = shell_values(
            source + 'echo "$THETA|$DISCOUNT|$(growth_dir)"'
        )
        assert float(theta) == config["theta"]
        assert float(discount) == config["discount"]
        assert folder.endswith(f"/theta={theta}_discount={discount}")


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize(
    ("job", "expected"),
    [
        pytest.param({}, 4, id="no-slurm"),
        # Memory allows 768 GB / 4 GB = 192 workers; the task has 96 CPUs.
        pytest.param(
            {
                "SLURM_MEM_PER_NODE": "786432",
                "SLURM_CPUS_PER_TASK": "96",
                "SLURM_CPUS_ON_NODE": "99",
            },
            96,
            id="batch-job",
        ),
        # `srun --jobid=... --overlap --pty bash` sets no SLURM_CPUS_PER_TASK.
        pytest.param(
            {"SLURM_MEM_PER_NODE": "786432", "SLURM_CPUS_ON_NODE": "99"},
            99,
            id="attached-shell",
        ),
        pytest.param(
            {"SLURM_MEM_PER_NODE": "40960", "SLURM_CPUS_ON_NODE": "99"},
            10,
            id="memory-bound",
        ),
        pytest.param(
            {"NUM_WORKERS": "7", "SLURM_MEM_PER_NODE": "786432", "SLURM_CPUS_ON_NODE": "99"},
            7,
            id="explicit",
        ),
    ],
)
def test_config_fits_workers_to_the_slurm_job(version, job, expected):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SLURM_")
        and key not in {"NUM_WORKERS", "MEMORY_PER_WORKER_MB"}
    }
    workers = shell_values(
        f'source scripts/experiments/{version}/config.sh; echo "$NUM_WORKERS"',
        env={**env, **job},
    )
    assert workers == [str(expected)]


@pytest.mark.parametrize(
    "script",
    sorted(path for version in VERSIONS for path in scripts_dir(version).glob("*.sh")),
    ids=lambda path: f"{path.parent.name}/{path.name}",
)
def test_scripts_have_valid_bash_syntax(script):
    subprocess.run(["bash", "-n", str(script)], check=True)
