"""Pool reuse, experiment resume/extend/restart, slot retries, and workers."""

import os
import random
from collections import Counter
from pathlib import Path

import orjson
import pytest

from src.questions import dataset_pool
from src.questions.dataset_pool import RetryableConstructionError

CONFIG = {
    "question_format": "reachability-yesno",
    "question_depth": 3,
    "component_node_count": 9,
    "answer_condition": "YES",
}
SETTINGS = {"vary": "question-depth", "random_seed": 1, "context_length": 4000}


def fake_example(config, seed):
    """Picklable stand-in for create_dataset.v2's construction."""
    return {
        "config": config,
        "system_message": "System",
        "user_message": f"Question {seed}",
        "answer": ["YES"],
        "construction_seed": seed,
    }


def example_with_pid(config, seed):
    return {**fake_example(config, seed), "pid": os.getpid()}


def reporting_example(config, seed):
    dataset_pool.report("hello from a worker")
    print("a plain print from a worker")
    return fake_example(config, seed)


# This process's held startup lines (a worker opens its inputs once).
STARTUP = {}


def startup_example(config, seed):
    """Like create_dataset's worker: startup lines are held, then it is ready."""
    if "held" not in STARTUP:
        with dataset_pool.held_output() as held:
            print("loading the same tokenizers as the main process")
        STARTUP["held"] = held
        dataset_pool.worker_ready()
    return {**fake_example(config, seed), "held": STARTUP["held"]}


def fail_on_no(config, seed):
    if config["answer_condition"] == "NO":
        raise ValueError("bug in this slot")
    return fake_example(config, seed)


def lines(path):
    return [orjson.loads(line) for line in path.read_bytes().splitlines()]


def run(
    pool,
    output,
    configs,
    construct=fake_example,
    *,
    settings=SETTINGS,
    compatibility=None,
    seed=1,
    max_attempts=3,
    workers=1,
):
    return dataset_pool.run_experiment(
        pool,
        output,
        configs,
        settings,
        compatibility or {"generator_version": "test"},
        construct,
        seed=seed,
        max_attempts=max_attempts,
        workers=workers,
    )


def test_pool_exact_reuse_without_replacement_and_resume(tmp_path):
    pool, output = tmp_path / "pool.jsonl", tmp_path / "run.jsonl"
    first = run(pool, output, [CONFIG] * 3)
    assert first == {"created": 3, "reused": 0, "resumed": 0, "retries": {}}
    assert len({entry["qid"] for entry in lines(output)}) == 3
    original = pool.read_bytes(), output.read_bytes()
    assert run(pool, output, [CONFIG] * 3)["resumed"] == 3
    assert original == (pool.read_bytes(), output.read_bytes())
    second = run(pool, tmp_path / "run2.jsonl", [CONFIG] * 4, seed=2)
    assert second == {"created": 1, "reused": 3, "resumed": 0, "retries": {}}
    changed = {**CONFIG, "context_length": 500_000}
    assert run(pool, tmp_path / "run3.jsonl", [changed], seed=3)["created"] == 1
    new_key = {"generator_version": "new"}
    fourth = run(pool, tmp_path / "run4.jsonl", [CONFIG], compatibility=new_key)
    assert fourth["created"] == 1


def test_larger_plan_appends_to_the_experiment(tmp_path):
    pool, output = tmp_path / "pool.jsonl", tmp_path / "run.jsonl"
    run(pool, output, [CONFIG] * 2)
    saved = lines(output)
    stats = run(pool, output, [CONFIG] * 5)
    assert stats == {"resumed": 2, "reused": 0, "created": 3, "retries": {}}
    assert lines(output)[:2] == saved
    assert len(lines(output)) == 5
    assert not (tmp_path / "run.jsonl.previous").exists()


def test_attempt_limit_change_does_not_restart(tmp_path):
    pool, output = tmp_path / "pool.jsonl", tmp_path / "run.jsonl"
    run(pool, output, [CONFIG] * 2, max_attempts=3)
    assert run(pool, output, [CONFIG] * 2, max_attempts=50)["resumed"] == 2


def test_changed_settings_restart_and_reuse_the_pool(tmp_path):
    pool, output = tmp_path / "pool.jsonl", tmp_path / "run.jsonl"
    run(pool, output, [CONFIG] * 2)
    old = lines(output)
    stats = run(pool, output, [CONFIG] * 2, settings={**SETTINGS, "random_seed": 2})
    assert stats == {"resumed": 0, "reused": 2, "created": 0, "retries": {}}
    assert lines(tmp_path / "run.jsonl.previous") == old
    assert (tmp_path / "run.jsonl.settings.json.previous").exists()
    assert {entry["qid"] for entry in lines(output)} == {
        entry["qid"] for entry in old
    }


def test_changed_slot_config_restarts(tmp_path):
    pool, output = tmp_path / "pool.jsonl", tmp_path / "run.jsonl"
    run(pool, output, [CONFIG] * 2)
    other = {**CONFIG, "answer_condition": "NO"}
    stats = run(pool, output, [other, CONFIG])
    assert stats == {"resumed": 0, "reused": 1, "created": 1, "retries": {}}
    assert (tmp_path / "run.jsonl.previous").exists()


def test_smaller_plan_stops_without_changing_files(tmp_path):
    pool, output = tmp_path / "pool.jsonl", tmp_path / "run.jsonl"
    run(pool, output, [CONFIG] * 3)
    settings = dataset_pool.settings_path_for(output)
    before = output.read_bytes(), settings.read_bytes()
    with pytest.raises(ValueError, match="more than the 2 planned"):
        run(pool, output, [CONFIG] * 2)
    assert before == (output.read_bytes(), settings.read_bytes())


def test_interruption_after_pool_append_and_incomplete_tail(tmp_path, monkeypatch):
    pool, output = tmp_path / "pool.jsonl", tmp_path / "run.jsonl"
    append = dataset_pool.append_record

    def interrupted_append(handle, record):
        if set(record) == {"qid"}:
            handle.write(b'{"qid":')
            handle.flush()
            raise RuntimeError("interrupted")
        return append(handle, record)

    monkeypatch.setattr(dataset_pool, "append_record", interrupted_append)
    with pytest.raises(RuntimeError, match="interrupted"):
        run(pool, output, [CONFIG])
    assert len(lines(pool)) == 1
    monkeypatch.setattr(dataset_pool, "append_record", append)
    with pool.open("ab") as handle:
        handle.write(b'{"partial":')
    stats = run(pool, output, [CONFIG])
    assert stats["reused"] == 1
    assert len(lines(pool)) == len(lines(output)) == 1


def test_read_only_reader_skips_an_incomplete_tail_without_writing(tmp_path):
    path = tmp_path / "pool.jsonl"
    path.write_bytes(b'{"qid":"a"}\n{"qid":')
    with path.open("rb") as handle:
        records = [record for _offset, record in dataset_pool.read_jsonl(handle, repair=False)]
    assert records == [{"qid": "a"}]
    assert path.read_bytes() == b'{"qid":"a"}\n{"qid":'


def test_existing_pool_choice_is_uniform_and_excludes_used(tmp_path):
    with (tmp_path / "pool").open("w+b") as handle:
        pool = dataset_pool.ExamplePool(handle)
        for qid in ["a", "b", "c"]:
            pool.add({"qid": qid, "config": CONFIG, "compatibility": {}})
        rng = random.Random(123)
        counts = Counter(pool.choose(CONFIG, {}, {"a"}, rng) for _ in range(2000))
        assert set(counts) == {"b", "c"}
        assert 900 < counts["b"] < 1100
        assert pool.choose({**CONFIG, "question_depth": 4}, {}, set(), rng) is None


def test_progress_bar_and_retry_messages_name_their_slot(tmp_path, capsys):
    attempts = []

    def construct(config, seed):
        attempts.append(seed)
        if len(attempts) == 1:
            raise RetryableConstructionError("yesno_answer_check_failed")
        return fake_example(config, seed)

    run(tmp_path / "pool", tmp_path / "output", [CONFIG] * 2, construct)
    captured = capsys.readouterr()
    assert "[slot 1/2] attempt 1/3 failed (yesno_answer_check_failed)" in captured.out
    assert "Examples" in captured.err and "2/2" in captured.err


def test_worker_messages_are_printed_by_the_main_process(tmp_path, capsys):
    run(tmp_path / "pool", tmp_path / "output", [CONFIG] * 2, reporting_example, workers=2)
    out = capsys.readouterr().out
    assert "[slot 1/2] hello from a worker" in out
    assert "[slot 2/2] hello from a worker" in out
    assert out.count("a plain print from a worker") == 2


def test_worker_startup_lines_are_held_and_ready_is_counted(tmp_path, capsys):
    pool = tmp_path / "pool.jsonl"
    run(pool, tmp_path / "output", [CONFIG] * 4, startup_example, workers=2)
    captured = capsys.readouterr()
    assert "Starting 2 worker processes." in captured.out
    assert "loading the same tokenizers" not in captured.out + captured.err
    assert dataset_pool.WORKER_READY not in captured.out + captured.err
    assert {tuple(record["held"]) for record in lines(pool)} == {
        ("loading the same tokenizers as the main process",)
    }


def test_held_output_prints_as_usual_outside_workers(capsys):
    with dataset_pool.held_output() as held:
        print("main process line")
    dataset_pool.worker_ready()
    assert held == [] and capsys.readouterr().out == "main process line\n"


def test_chance_failures_retry_the_same_config(tmp_path):
    attempted = []

    def construct(config, seed):
        attempted.append((dict(config), seed))
        if len(attempted) == 1:
            raise RetryableConstructionError("yesno_answer_check_failed")
        if len(attempted) == 2:
            raise RetryableConstructionError("needles_over_budget", "too long")
        return fake_example(config, seed)

    stats = run(tmp_path / "pool", tmp_path / "output", [CONFIG], construct)
    assert stats["retries"] == {"yesno_answer_check_failed": 1, "needles_over_budget": 1}
    assert [config for config, _seed in attempted] == [CONFIG] * 3
    assert len({seed for _config, seed in attempted}) == 3


def test_exhausted_attempts_stop_without_dropping_the_slot(tmp_path):
    def construct(config, seed):
        raise RetryableConstructionError("component_build_failed")

    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="Could not complete slot 1/1 after 2 attempts"):
        run(tmp_path / "pool", output, [CONFIG], construct, max_attempts=2)
    assert lines(output) == []


def test_unexpected_errors_stop_without_retrying(tmp_path):
    calls = []

    def construct(config, seed):
        calls.append(seed)
        raise ValueError("bug")

    with pytest.raises(ValueError, match="bug"):
        run(tmp_path / "pool", tmp_path / "output", [CONFIG], construct)
    assert len(calls) == 1


def test_duplicate_content_tries_the_next_seed(tmp_path):
    calls = []

    def construct(config, seed):
        calls.append(seed)
        text = "same" if len(calls) <= 2 else f"unique {seed}"
        return {**fake_example(config, seed), "user_message": text}

    output = tmp_path / "output"
    stats = run(tmp_path / "pool", output, [CONFIG] * 2, construct)
    assert stats["created"] == 2
    assert len(calls) == 3
    assert len({entry["qid"] for entry in lines(output)}) == 2


def test_parallel_workers_write_the_same_experiment(tmp_path):
    serial = tmp_path / "serial.jsonl"
    parallel = tmp_path / "parallel.jsonl"
    run(tmp_path / "pool-a.jsonl", serial, [CONFIG] * 5, example_with_pid)
    pool = tmp_path / "pool-b.jsonl"
    stats = run(pool, parallel, [CONFIG] * 5, example_with_pid, workers=2)
    assert stats["created"] == 5
    assert lines(parallel) == lines(serial)
    assert os.getpid() not in {record["pid"] for record in lines(pool)}


def test_worker_failure_keeps_finished_slots_and_resumes(tmp_path):
    pool, output = tmp_path / "pool.jsonl", tmp_path / "run.jsonl"
    bad = {**CONFIG, "answer_condition": "NO"}
    configs = [CONFIG, bad, CONFIG, CONFIG]
    with pytest.raises(ValueError, match="bug in this slot"):
        run(pool, output, configs, fail_on_no, workers=2)
    assert len(lines(output)) == 1
    stats = run(pool, output, configs, fake_example, workers=2)
    assert stats["resumed"] == 1
    assert stats["reused"] + stats["created"] == 3
    assert len(lines(output)) == 4


def resampled_example(config, seed):
    """Like an oversized multiple-choice example: built with fewer nodes."""
    built = {**config, "component_node_count": 4}
    return {**fake_example(built, seed), "planned_config": config}


def test_resampled_examples_resume_in_their_own_slots(tmp_path):
    pool, output = tmp_path / "pool.jsonl", tmp_path / "run.jsonl"
    run(pool, output, [CONFIG] * 2, resampled_example)
    assert run(pool, output, [CONFIG] * 2, resampled_example)["resumed"] == 2
    assert not (tmp_path / "run.jsonl.previous").exists()


def test_reuse_requires_the_config_actually_built(tmp_path):
    pool = tmp_path / "pool.jsonl"
    run(pool, tmp_path / "depth.jsonl", [CONFIG] * 2, resampled_example)
    # A slot that studies 9 nodes must not get the 4-node examples planned as 9.
    studied = run(pool, tmp_path / "nodes.jsonl", [CONFIG] * 2, seed=2)
    assert studied == {"resumed": 0, "reused": 0, "created": 2, "retries": {}}
    with pool.open("rb") as handle:
        built = {
            record["qid"]: record["config"]["component_node_count"]
            for _offset, record in dataset_pool.read_jsonl(handle, repair=False)
        }
    assert {built[entry["qid"]] for entry in lines(tmp_path / "nodes.jsonl")} == {9}
    # A slot that asks for 4 nodes may reuse them.
    small = {**CONFIG, "component_node_count": 4}
    assert run(pool, tmp_path / "small.jsonl", [small] * 2, seed=3)["reused"] == 2


def test_content_key_names_examples_without_a_prompt(tmp_path):
    def components_only(config, seed):
        return {"config": config, "answer": ["YES"], "content_key": [f"graph {seed}"]}

    pool, output = tmp_path / "pool.jsonl", tmp_path / "run.jsonl"
    assert run(pool, output, [CONFIG] * 2, components_only)["created"] == 2
    assert len({entry["qid"] for entry in lines(output)}) == 2


def test_prompt_qids_are_unchanged(tmp_path):
    pool, output = tmp_path / "pool.jsonl", tmp_path / "run.jsonl"
    run(pool, output, [CONFIG], fake_example)
    record = lines(pool)[0]
    assert record["qid"] == dataset_pool.digest(
        [
            {"generator_version": "test"},
            CONFIG,
            record["system_message"],
            record["user_message"],
            record["answer"],
        ]
    )


def test_portable_path_is_repo_relative_or_the_file_name(tmp_path, monkeypatch):
    monkeypatch.chdir(dataset_pool.REPO_ROOT)
    inside = dataset_pool.REPO_ROOT / "src" / "questions" / "dataset_pool.py"
    assert dataset_pool.portable_path(inside) == "src/questions/dataset_pool.py"
    assert dataset_pool.portable_path(tmp_path / "pool.jsonl") == "pool.jsonl"
    link = tmp_path / "graph.db"
    link.symlink_to(tmp_path / "elsewhere" / "real.db")
    assert dataset_pool.portable_path(link) == "graph.db"
    assert dataset_pool.portable_path(Path("src") / "questions") == "src/questions"


def experiment_specs(root, labels=("a", "b"), slots=3):
    """One spec per label, each with its own folder and a different config."""
    return [
        dataset_pool.ExperimentSpec(
            root / label / "pool.jsonl",
            root / label / "run.jsonl",
            [{**CONFIG, "growth": index}] * slots,
            SETTINGS,
            {"generator_version": "test"},
            label=label,
        )
        for index, label in enumerate(labels)
    ]


def run_all(specs, construct=fake_example, workers=1):
    return dataset_pool.run_experiments(
        specs, construct, seed=1, max_attempts=3, workers=workers
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_several_experiments_write_the_same_files_as_separate_runs(tmp_path, workers):
    separate = experiment_specs(tmp_path / "separate")
    for spec in separate:
        run(spec.pool_path, spec.output_path, spec.configs)
    together = experiment_specs(tmp_path / "together")
    stats = run_all(together, workers=workers)
    assert [result["created"] for result in stats] == [3, 3]
    for one, other in zip(separate, together):
        assert one.output_path.read_bytes() == other.output_path.read_bytes()
        settings = dataset_pool.settings_path_for
        assert settings(one.output_path).read_bytes() == settings(other.output_path).read_bytes()
        # Workers may finish in another order, so compare the pool records.
        assert sorted(one.pool_path.read_bytes().splitlines()) == sorted(
            other.pool_path.read_bytes().splitlines()
        )


def test_several_experiments_resume_what_a_single_run_wrote(tmp_path):
    first, second = experiment_specs(tmp_path)
    run(first.pool_path, first.output_path, first.configs)
    stats = run_all([first, second])
    assert stats[0] == {"resumed": 3, "reused": 0, "created": 0, "retries": {}}
    assert stats[1]["created"] == 3
    assert run(second.pool_path, second.output_path, second.configs)["resumed"] == 3


def test_a_locked_pool_stops_before_any_experiment_changes(tmp_path):
    import fcntl

    first, second = experiment_specs(tmp_path)
    second.pool_path.parent.mkdir(parents=True)
    with second.pool_path.open("a+b") as other_process:
        fcntl.flock(other_process, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="Another process is writing this pool"):
            run_all([first, second])
    assert not first.output_path.exists()
    assert not dataset_pool.settings_path_for(first.output_path).exists()


def test_experiments_cannot_share_a_pool(tmp_path):
    first, second = experiment_specs(tmp_path)
    shared = dataset_pool.ExperimentSpec(
        first.pool_path, second.output_path, second.configs, SETTINGS, {}, label="b"
    )
    with pytest.raises(ValueError, match="its own pool"):
        run_all([first, shared])


def test_messages_name_their_experiment(tmp_path, capsys):
    attempts = []

    def construct(config, seed):
        attempts.append(seed)
        if len(attempts) == 1:
            raise RetryableConstructionError("yesno_answer_check_failed")
        return fake_example(config, seed)

    run_all(experiment_specs(tmp_path, slots=2), construct)
    out = capsys.readouterr().out
    assert "Plan: 2 experiments, 4 slots; 0 already written" in out
    assert "[a | slot 1/2] attempt 1/3 failed (yesno_answer_check_failed)" in out
    assert "[a] finished: 2 slots (2 new, 0 reused from the pool, 0 already there)." in out
    assert "[b] finished: 2 slots" in out
