import orjson
import os
from typing import Any, Type

from pydantic import BaseModel


def save_objects(objects: list[Any], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if path.endswith(".json"):
        with open(path, "w") as f:
            list_of_dicts = [obj.model_dump() for obj in objects]
            f.write(orjson.dumps(list_of_dicts, option=orjson.OPT_INDENT_2).decode("utf-8"))
    elif path.endswith(".jsonl"):
        with open(path, "w") as f:
            for obj in objects:
                f.write(orjson.dumps(obj.model_dump()).decode("utf-8") + "\n")
    else:
        raise ValueError(f"Invalid file extension: {path}")


def load_objects(path: str, schema: Type[BaseModel]) -> list[Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")

    # json or jsonl
    if path.endswith(".json"):
        with open(path, "r") as f:
            # read the json file as a list of dicts, then validate each dict with the schema
            objects = [orjson.loads(line) for line in f.readlines()]
            return [schema.model_validate(obj) for obj in objects]
    elif path.endswith(".jsonl"):
        with open(path, "r") as f:
            return [schema.model_validate(orjson.loads(line)) for line in f.readlines()]
    else:
        raise ValueError(f"Invalid file extension: {path}")
