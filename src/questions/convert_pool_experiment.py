"""Create a converted experiment from the multiple-choice subset of a pool."""

from __future__ import annotations

import fcntl
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import orjson

from src.llm.inference_pool import successful_response
from src.questions.dataset_pool import (
    ExamplePool,
    portable_path,
    read_experiment_qids,
    settings_path_for,
)
from src.questions.reachability_multiple import QUESTION_TYPE


def write_bytes_atomically(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def convert_pool_experiment(
    input_path: Path,
    pool_path: Path,
    output_path: Path,
    convert_record: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    predictions_path: Path | None = None,
    check_only: bool = False,
    overwrite: bool = False,
) -> int:
    """Append converted examples once; export only the selected multiple-choice qids.

    The source experiment and inference output are never changed. Conversion
    holds the data pool's lock, as dataset creation does. Run conversions in
    sequence, then run model inference in parallel with separate response pools.
    """
    settings_path = settings_path_for(input_path)
    output_settings_path = settings_path_for(output_path)
    inputs = [input_path, settings_path, pool_path]
    if predictions_path is not None:
        inputs.append(predictions_path)
    for destination in (output_path, output_settings_path):
        for source in inputs:
            if destination.resolve() == source.resolve() or (
                destination.exists()
                and source.exists()
                and destination.samefile(source)
            ):
                raise ValueError(f"Conversion output would overwrite input: {source}")

    source_settings = orjson.loads(settings_path.read_bytes())
    identity = source_settings["identity"]
    if identity.get("pool") != portable_path(pool_path):
        raise ValueError("The experiment settings refer to a different --pool.")
    if identity.get("vary") != "question-format":
        raise ValueError("Expected experiment 3 (--vary question-format).")
    source_qids = read_experiment_qids(input_path, repair=False)
    selected_qids = set(source_qids)
    conversion = {
        "conversion": "add-none-option",
        "mode": "append",
        "version": 1,
        "source_experiment": portable_path(input_path),
    }
    if predictions_path is not None:
        predictions = orjson.loads(predictions_path.read_bytes())
        if predictions.get("schema_version") != 1 or not isinstance(
            predictions.get("records"), list
        ):
            raise ValueError("--predictions must be a run_inference.py output.")
        prediction_qids = [record["qid"] for record in predictions["records"]]
        if len(set(prediction_qids)) != len(prediction_qids):
            raise ValueError("Duplicate qids in --predictions.")
        if not set(prediction_qids) <= selected_qids:
            raise ValueError(
                "--predictions contains qids outside the source experiment."
            )
        selected_qids = {
            record["qid"]
            for record in predictions["records"]
            if successful_response(record)
        }
        conversion["selection"] = {
            "predictions": portable_path(predictions_path),
            "model": predictions["metadata"]["model"],
            "generation_settings": predictions["metadata"]["generation_settings"],
            "policy": "successful-response",
        }
    converted_identity = {**identity, "conversion": conversion}
    # Record-level compatibility is versioned separately from selection metadata.
    converted_identity["compatibility"] = {
        **identity["compatibility"],
        "conversion": {
            key: conversion[key] for key in ("conversion", "mode", "version")
        },
    }
    if output_settings_path.exists() and not overwrite:
        previous = orjson.loads(output_settings_path.read_bytes())
        if previous.get("identity") != converted_identity:
            raise ValueError(
                "Converted experiment settings changed; use a new --output or --overwrite."
            )
    elif output_path.exists() and not overwrite and not output_settings_path.exists():
        raise ValueError(
            "Existing output has no conversion settings; use a new --output."
        )

    converted_qids = []
    with pool_path.open("rb" if check_only else "r+b") as handle:
        try:
            fcntl.flock(
                handle, (fcntl.LOCK_SH if check_only else fcntl.LOCK_EX) | fcntl.LOCK_NB
            )
        except BlockingIOError:
            raise RuntimeError(
                "Another process is writing the data pool; retry after it finishes."
            ) from None
        pool = ExamplePool(handle)
        pool.scan(repair=not check_only)
        for qid in source_qids:
            if qid not in selected_qids:
                continue
            record = pool.get(qid)
            if record["config"]["question_format"] != QUESTION_TYPE:
                continue
            converted = convert_record(record)
            new_qid = converted["qid"]
            if new_qid in pool.offsets:
                if pool.get(new_qid) != converted:
                    raise ValueError(f"Converted pool record differs: {new_qid}")
            elif check_only:
                raise ValueError(
                    f"Converted example is missing from the pool: {new_qid}"
                )
            else:
                pool.add(converted)
            converted_qids.append(new_qid)

        settings = {"identity": converted_identity, "slots": len(converted_qids)}
        if check_only:
            if (
                not output_path.exists()
                or read_experiment_qids(output_path, repair=False) != converted_qids
            ):
                raise ValueError("Converted experiment differs from the expected qids.")
            if orjson.loads(output_settings_path.read_bytes()) != settings:
                raise ValueError("Converted experiment settings differ.")
        else:
            if not overwrite and not set(
                read_experiment_qids(output_path, repair=False)
            ) <= set(converted_qids):
                raise ValueError(
                    "Selection would discard converted examples; use a new --output or --overwrite."
                )
            write_bytes_atomically(
                output_settings_path, orjson.dumps(settings, option=orjson.OPT_INDENT_2)
            )
            write_bytes_atomically(
                output_path,
                b"".join(orjson.dumps({"qid": qid}) + b"\n" for qid in converted_qids),
            )
    return len(converted_qids)
