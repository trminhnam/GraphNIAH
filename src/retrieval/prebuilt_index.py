"""Helpers for routing Pyserini prebuilt-index downloads into a local cache."""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


DEFAULT_JAVA_HOME_CANDIDATES = (
    Path("/usr/lib/jvm/java-21-openjdk"),
    Path("/usr/lib/jvm/java-21-openjdk-amd64"),
    Path("/usr/lib/jvm/jdk-21"),
    Path("/usr/lib/jvm/temurin-21-jdk"),
    Path("/Library/Java/JavaVirtualMachines/openjdk-21.jdk/Contents/Home"),
    Path("/Library/Java/JavaVirtualMachines/temurin-21.jdk/Contents/Home"),
    Path("/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"),
)


def resolve_cache_dir(cache_dir: str | Path | None) -> Path | None:
    if cache_dir is None:
        return None
    return Path(cache_dir).resolve()


def pyserini_cache_paths(cache_dir: str | Path | None) -> list[Path]:
    resolved = resolve_cache_dir(cache_dir)
    if resolved is None:
        return []
    return [
        resolved / "pyserini" / "indexes",
        resolved / ".cache" / "pyserini" / "indexes",
    ]


def parse_java_major_version(version_text: str) -> int | None:
    normalized = version_text.strip().strip('"').strip("'")
    if not normalized:
        return None

    parts = normalized.split(".")
    major = parts[0]
    if major == "1" and len(parts) > 1:
        major = parts[1]

    try:
        return int(major)
    except ValueError:
        return None


def read_java_major_version(java_home: str | Path | None) -> int | None:
    if java_home is None:
        return None

    release_file = Path(java_home) / "release"
    if not release_file.is_file():
        return None

    for line in release_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("JAVA_VERSION="):
            version_text = line.split("=", maxsplit=1)[1]
            return parse_java_major_version(version_text)
    return None


def resolve_java_home_from_binary(java_binary: str | None) -> Path | None:
    if java_binary is None:
        return None

    java_path = Path(java_binary).expanduser().resolve(strict=False)
    if java_path.name != "java":
        return None

    java_home = java_path.parent.parent
    if not (java_home / "bin" / "java").is_file():
        return None
    return java_home


def find_compatible_java_home(minimum_major: int = 21) -> Path | None:
    candidates: list[Path] = []
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        candidates.append(Path(java_home))

    java_home_from_path = resolve_java_home_from_binary(shutil.which("java"))
    if java_home_from_path is not None:
        candidates.append(java_home_from_path)

    candidates.extend(DEFAULT_JAVA_HOME_CANDIDATES)

    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        resolved_text = str(resolved)
        if resolved_text in seen:
            continue
        seen.add(resolved_text)

        if not (resolved / "bin" / "java").is_file():
            continue

        major_version = read_java_major_version(resolved)
        if major_version is not None and major_version >= minimum_major:
            return resolved

    return None


def prepend_directory_to_path(directory: str | Path) -> str:
    directory_text = str(Path(directory).expanduser().resolve(strict=False))
    current_path = os.environ.get("PATH")
    if not current_path:
        return directory_text

    parts = [part for part in current_path.split(os.pathsep) if part and part != directory_text]
    return os.pathsep.join([directory_text, *parts])


@contextmanager
def pyserini_cache_environment(
    cache_dir: str | Path | None,
) -> Iterator[Path | None]:
    resolved = resolve_cache_dir(cache_dir)
    if resolved is None:
        yield None
        return

    resolved.mkdir(parents=True, exist_ok=True)
    original_home = os.environ.get("HOME")
    original_xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    os.environ["HOME"] = str(resolved)
    os.environ["XDG_CACHE_HOME"] = str(resolved)
    try:
        yield resolved
    finally:
        if original_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = original_home
        if original_xdg_cache_home is None:
            os.environ.pop("XDG_CACHE_HOME", None)
        else:
            os.environ["XDG_CACHE_HOME"] = original_xdg_cache_home


@contextmanager
def pyserini_runtime_environment(
    cache_dir: str | Path | None,
    minimum_java_major: int = 21,
) -> Iterator[Path | None]:
    selected_java_home = find_compatible_java_home(minimum_major=minimum_java_major)
    original_java_home = os.environ.get("JAVA_HOME")
    original_path = os.environ.get("PATH")

    with pyserini_cache_environment(cache_dir) as resolved:
        if selected_java_home is not None:
            os.environ["JAVA_HOME"] = str(selected_java_home)
            os.environ["PATH"] = prepend_directory_to_path(selected_java_home / "bin")

        try:
            yield resolved
        finally:
            if original_java_home is None:
                os.environ.pop("JAVA_HOME", None)
            else:
                os.environ["JAVA_HOME"] = original_java_home

            if original_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = original_path
