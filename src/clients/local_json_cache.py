"""Small JSON-backed cache for HTTP client responses."""

from __future__ import annotations

from copy import deepcopy
import orjson
from pathlib import Path
import threading
from typing import Any

from loguru import logger


class LocalJsonCache:
    """Persist request/response entries in a local JSON file."""

    def __init__(self, path: str | Path | None, *, indent: int | None = 2) -> None:
        self.path = Path(path) if path is not None else None
        self.indent = indent
        self.cache: dict[str, Any] | None = None
        self.lock = threading.Lock()

    @classmethod
    def make_key(
        cls,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        body: Any = None,
    ) -> str:
        return orjson.dumps(cls.make_request(method, endpoint, params=params, body=body), option=orjson.OPT_SORT_KEYS).decode("utf-8")

    @classmethod
    def make_request(
        cls,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        body: Any = None,
    ) -> dict[str, Any]:
        return {
            "method": method.upper(),
            "endpoint": endpoint,
            "params": cls.normalize(params or {}),
            "body": cls.normalize(body),
        }

    @classmethod
    def normalize(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): cls.normalize(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [cls.normalize(item) for item in value]
        return value

    def get(self, key: str) -> tuple[bool, Any]:
        if self.path is None:
            return False, None

        with self.lock:
            entry = self.load()["entries"].get(key)
            if entry is None:
                return False, None
            return True, deepcopy(entry.get("response"))

    def store(
        self,
        key: str,
        method: str,
        endpoint: str,
        response: Any,
        *,
        params: dict[str, Any] | None = None,
        body: Any = None,
    ) -> None:
        if self.path is None:
            return

        request = self.make_request(method, endpoint, params=params, body=body)
        with self.lock:
            cache = self.load()
            cache["entries"][key] = {
                "request": request,
                "response": deepcopy(response),
            }
            self.write(cache)

    def load(self) -> dict[str, Any]:
        if self.cache is not None:
            return self.cache

        self.cache = {"version": 1, "entries": {}}
        if self.path is None or not self.path.exists():
            return self.cache

        try:
            with self.path.open(encoding="utf-8") as file:
                data = orjson.loads(file.read())
        except (OSError, orjson.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable cache file {}: {}", self.path, exc)
            return self.cache

        if isinstance(data, dict) and isinstance(data.get("entries"), dict):
            self.cache = data
        else:
            logger.warning("Ignoring cache file with unexpected structure: {}", self.path)
        return self.cache

    def write(self, cache: dict[str, Any]) -> None:
        if self.path is None:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            file.write(orjson.dumps(cache, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS).decode("utf-8"))
            file.write("\n")
        temp_path.replace(self.path)
