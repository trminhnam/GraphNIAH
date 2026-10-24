"""Download Semantic Scholar bulk datasets for a selected release.

This follows ``notebooks2/seamntic_scholar_bulk_api.ipynb``:

1. Use the configured release date unless another release is specified.
2. List datasets for the selected release.
3. Fetch each dataset detail endpoint with ``SEMANTIC_SCHOLAR_API_KEY``.
4. Download every presigned dataset file into its dataset folder.
5. Decompress each ``.gz`` shard and remove the compressed copy.
6. Save the dataset description as ``description.txt``.
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlsplit

import requests

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is in requirements.txt
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://api.semanticscholar.org/datasets/v1/release"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "resources" / "semantic_scholar_bulk" / "raw_datasets"
DEFAULT_RELEASE_ID = "2026-09-09"
DESCRIPTION_FILE_NAME = "description.txt"
CHUNK_SIZE_BYTES = 1024 * 1024
METADATA_TIMEOUT_SECONDS = 30
DOWNLOAD_TIMEOUT_SECONDS = (10, 300)
DEFAULT_NUM_WORKERS = 4
STATUS_LOCK = threading.RLock()
MAX_DISPLAY_NAME_LENGTH = 64

if tqdm is not None:
    tqdm.set_lock(STATUS_LOCK)


def load_dotenv_if_available() -> None:
    """Load .env so SEMANTIC_SCHOLAR_API_KEY is visible when python-dotenv exists."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Semantic Scholar bulk datasets for one release.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--release-id",
        default=DEFAULT_RELEASE_ID,
        help="Semantic Scholar dataset release date to download.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root folder where each dataset subfolder will be created.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files even when the destination file already exists.",
    )
    parser.add_argument(
        "--num-workers",
        type=positive_int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of files to download in parallel within each dataset.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset name(s) to download. Defaults to all datasets in the release.",
    )
    return parser


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return number


def get_api_key() -> str:
    load_dotenv_if_available()
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        raise SystemExit("SEMANTIC_SCHOLAR_API_KEY is required.")
    return api_key


def get_json(session: requests.Session, url: str) -> dict[str, Any]:
    response = session.get(url, timeout=METADATA_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object from {url}, got {type(data).__name__}")
    return data


def fetch_release_ids(session: requests.Session) -> list[str]:
    response = session.get(f"{BASE_URL}/", timeout=METADATA_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError("Expected Semantic Scholar releases endpoint to return a list of strings")
    return data


def resolve_release_id(session: requests.Session, release_id: str | None) -> str:
    if release_id:
        return release_id

    release_ids = fetch_release_ids(session)
    if not release_ids:
        raise ValueError("Semantic Scholar releases endpoint returned no releases")
    return release_ids[-1]


def fetch_release_datasets(
    session: requests.Session,
    release_id: str,
) -> list[dict[str, Any]]:
    data = get_json(session, f"{BASE_URL}/{release_id}")
    datasets = data.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError(f"Release {release_id} did not return a datasets list")
    return datasets


def fetch_dataset_details(
    session: requests.Session,
    release_id: str,
    dataset_name: str,
) -> dict[str, Any]:
    return get_json(session, f"{BASE_URL}/{release_id}/dataset/{dataset_name}")


def safe_dataset_folder_name(dataset_name: str) -> str:
    name = dataset_name.strip()
    if not name or "/" in name or "\\" in name:
        raise ValueError(f"Invalid dataset name: {dataset_name!r}")
    return name


def file_name_from_url(url: str, index: int) -> str:
    path_name = Path(unquote(urlsplit(url).path)).name
    if path_name:
        return path_name
    return f"dataset_file_{index:05d}"


def save_description(dataset_dir: Path, description: str) -> None:
    description_path = dataset_dir / DESCRIPTION_FILE_NAME
    description_path.write_text(description.rstrip() + "\n", encoding="utf-8")


def file_is_complete(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def format_byte_count(byte_count: float) -> str:
    amount = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if amount < 1024 or unit == "GiB":
            if unit == "B":
                if amount.is_integer():
                    return f"{int(amount)} {unit}"
                return f"{amount:.2f} {unit}"
            return f"{amount:.2f} {unit}"
        amount /= 1024
    return f"{amount:.2f} GiB"


def format_download_speed(byte_count: int, elapsed_seconds: float) -> str:
    if elapsed_seconds <= 0:
        elapsed_seconds = 0.001
    return f"{format_byte_count(byte_count / elapsed_seconds)}/s"


def write_status(message: str) -> None:
    with STATUS_LOCK:
        if tqdm is not None:
            tqdm.write(message, file=sys.stdout)
            sys.stdout.flush()
        else:
            print(message, flush=True)


def display_name(path: Path) -> str:
    name = path.name
    if len(name) <= MAX_DISPLAY_NAME_LENGTH:
        return name
    keep_prefix = 24
    keep_suffix = MAX_DISPLAY_NAME_LENGTH - keep_prefix - 3
    return f"{name[:keep_prefix]}...{name[-keep_suffix:]}"


def write_download_summary(
    destination: Path,
    byte_count: int,
    elapsed_seconds: float,
) -> None:
    write_status(
        "downloaded "
        f"{display_name(destination)} | "
        f"{format_byte_count(byte_count)} | "
        f"{format_download_speed(byte_count, elapsed_seconds)} | "
        f"{elapsed_seconds:.1f}s"
    )


def decompressed_path(gzip_path: Path) -> Path:
    if gzip_path.suffix != ".gz":
        raise ValueError(f"Expected a .gz file: {gzip_path}")
    return gzip_path.with_name(gzip_path.name.removesuffix(".gz"))


def temp_decompressed_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.{os.getpid()}.part")


def legacy_temp_decompressed_path(output_path: Path) -> Path:
    return Path(f"{output_path}.part")


def find_gzip_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix == ".gz" else []
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return sorted(item for item in path.rglob("*.gz") if item.is_file())


def decompress_gzip_file(
    gzip_path: Path,
    *,
    remove_compressed: bool = True,
    overwrite: bool = False,
    log_status: bool = True,
) -> Path:
    output_path = decompressed_path(gzip_path)
    stale_temp_path = legacy_temp_decompressed_path(output_path)
    if file_is_complete(output_path) and not overwrite:
        if remove_compressed and gzip_path.exists():
            gzip_path.unlink()
            if log_status:
                write_status(f"removed compressed duplicate: {display_name(gzip_path)}")
        if stale_temp_path.exists():
            stale_temp_path.unlink()
            if log_status:
                write_status(f"removed incomplete decompress: {display_name(stale_temp_path)}")
        return output_path

    temp_path = temp_decompressed_path(output_path)
    for incomplete_path in (temp_path, stale_temp_path):
        if incomplete_path.exists():
            incomplete_path.unlink()
            if log_status:
                write_status(f"removed incomplete decompress: {display_name(incomplete_path)}")

    try:
        with gzip.open(gzip_path, "rb") as compressed:
            with temp_path.open("wb") as output:
                shutil.copyfileobj(compressed, output, length=CHUNK_SIZE_BYTES)
        if file_is_complete(output_path) and not overwrite:
            temp_path.unlink()
        else:
            temp_path.replace(output_path)
    except FileNotFoundError:
        if temp_path.exists():
            temp_path.unlink()
        if file_is_complete(output_path) and not overwrite:
            return output_path
        raise
    except BaseException:
        if temp_path.exists():
            temp_path.unlink()
        raise

    if remove_compressed and gzip_path.exists():
        gzip_path.unlink()
    if log_status:
        write_status(
            "decompressed "
            f"{display_name(gzip_path)} -> {display_name(output_path)}"
        )
    return output_path


def decompress_gzip_files(
    path: Path,
    *,
    remove_compressed: bool = True,
    overwrite: bool = False,
) -> list[Path]:
    decompressed = []
    for gzip_path in find_gzip_files(path):
        decompressed.append(
            decompress_gzip_file(
                gzip_path,
                remove_compressed=remove_compressed,
                overwrite=overwrite,
                log_status=True,
            )
        )
    return decompressed


def temp_download_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.{os.getpid()}.part")


def legacy_temp_download_path(destination: Path) -> Path:
    return Path(f"{destination}.part")


def legacy_raw_dataset_path(path: Path) -> Path | None:
    parts = list(path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "raw_datasets":
            parts[index] = "raw_dataset"
            return Path(*parts)
    return None


def existing_path_candidates(path: Path) -> list[Path]:
    candidates = [path]
    legacy_path = legacy_raw_dataset_path(path)
    if legacy_path is not None and legacy_path != path:
        candidates.append(legacy_path)
    return candidates


def complete_existing_path(path: Path) -> Path | None:
    for candidate in existing_path_candidates(path):
        if file_is_complete(candidate):
            return candidate
    return None


def reuse_existing_file(existing_path: Path, destination: Path) -> Path:
    if existing_path == destination:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_path.replace(destination)
    write_status(
        "reused "
        f"{display_name(existing_path)} -> {display_name(destination)}"
    )
    return destination


def cleanup_existing_download_artifacts(destination: Path) -> None:
    for artifact_path in existing_path_candidates(destination):
        stale_temp_path = legacy_temp_download_path(artifact_path)
        if stale_temp_path.exists():
            stale_temp_path.unlink()
            write_status(f"removed incomplete download: {display_name(stale_temp_path)}")


def remove_existing_compressed_duplicates(destination: Path) -> None:
    for compressed_path in existing_path_candidates(destination):
        if compressed_path.exists():
            compressed_path.unlink()
            write_status(f"removed compressed duplicate: {display_name(compressed_path)}")


def ordered_datasets(
    datasets: list[dict[str, Any]],
    reverse_datasets: bool,
) -> list[dict[str, Any]]:
    if reverse_datasets:
        return list(reversed(datasets))
    return datasets


def filter_requested_datasets(
    datasets: list[dict[str, Any]],
    requested_names: list[str] | None,
) -> list[dict[str, Any]]:
    if requested_names is None:
        return datasets

    datasets_by_name = {}
    for dataset in datasets:
        dataset_name = dataset.get("name")
        if isinstance(dataset_name, str):
            datasets_by_name[dataset_name] = dataset

    missing_names = [name for name in requested_names if name not in datasets_by_name]
    if missing_names:
        raise ValueError(
            "Requested dataset(s) not found in release: "
            + ", ".join(missing_names)
        )

    filtered = []
    seen_names = set()
    for name in requested_names:
        if name in seen_names:
            continue
        filtered.append(datasets_by_name[name])
        seen_names.add(name)
    return filtered


def download_file(
    session: requests.Session,
    url: str,
    destination: Path,
    overwrite: bool,
    show_progress: bool = True,
    progress_callback: Callable[[int], None] | None = None,
) -> Path:
    temp_path = temp_download_path(destination)

    if destination.suffix == ".gz":
        decompressed_destination = decompressed_path(destination)
        existing_decompressed = complete_existing_path(decompressed_destination)
        if existing_decompressed is not None and not overwrite:
            decompressed_destination = reuse_existing_file(
                existing_decompressed,
                decompressed_destination,
            )
            write_status(f"skip existing decompressed: {display_name(decompressed_destination)}")
            remove_existing_compressed_duplicates(destination)
            cleanup_existing_download_artifacts(destination)
            return decompressed_destination

    existing_destination = complete_existing_path(destination)
    if existing_destination is not None and not overwrite:
        destination = reuse_existing_file(existing_destination, destination)
        write_status(f"skip existing file: {display_name(destination)}")
        cleanup_existing_download_artifacts(destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    for candidate in existing_path_candidates(destination):
        for incomplete_path in (
            temp_download_path(candidate),
            legacy_temp_download_path(candidate),
        ):
            if incomplete_path.exists():
                incomplete_path.unlink()
                write_status(f"removed incomplete download: {display_name(incomplete_path)}")

    try:
        with session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0)
            bytes_written = 0
            download_started_at = time.monotonic()
            progress = None
            if show_progress and tqdm is not None:
                progress = tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    desc=display_name(destination),
                    leave=False,
                    position=1,
                    file=sys.stdout,
                    dynamic_ncols=True,
                )

            try:
                with temp_path.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE_BYTES):
                        if not chunk:
                            continue
                        output.write(chunk)
                        bytes_written += len(chunk)
                        if progress_callback is not None:
                            progress_callback(len(chunk))
                        if progress is not None:
                            progress.update(len(chunk))
                if total and bytes_written != total:
                    raise IOError(f"Expected {total} bytes from {url}, wrote {bytes_written} bytes")
                elapsed_seconds = time.monotonic() - download_started_at
            finally:
                if progress is not None:
                    progress.close()
    except BaseException:
        if temp_path.exists():
            temp_path.unlink()
        raise

    if destination.suffix == ".gz" and not overwrite:
        decompressed_destination = decompressed_path(destination)
        existing_decompressed = complete_existing_path(decompressed_destination)
        if existing_decompressed is not None:
            temp_path.unlink()
            decompressed_destination = reuse_existing_file(
                existing_decompressed,
                decompressed_destination,
            )
            write_status(f"skip existing decompressed: {display_name(decompressed_destination)}")
            return decompressed_destination

    existing_destination = complete_existing_path(destination)
    if existing_destination is not None and not overwrite:
        temp_path.unlink()
        destination = reuse_existing_file(existing_destination, destination)
        write_status(f"skip existing file: {display_name(destination)}")
        return destination

    temp_path.replace(destination)
    write_download_summary(destination, bytes_written, elapsed_seconds)
    return destination


def download_dataset_file(
    download_session: requests.Session | None,
    url: str,
    index: int,
    dataset_dir: Path,
    overwrite: bool,
    show_progress: bool,
    progress_callback: Callable[[int], None] | None = None,
) -> Path:
    destination = dataset_dir / file_name_from_url(url, index)
    session = download_session or requests.Session()
    try:
        downloaded = download_file(
            session,
            url,
            destination,
            overwrite,
            show_progress=show_progress,
            progress_callback=progress_callback,
        )
    finally:
        if download_session is None:
            session.close()

    if downloaded.suffix == ".gz":
        return decompress_gzip_file(
            downloaded,
            remove_compressed=True,
            overwrite=overwrite,
            log_status=False,
        )
    return downloaded


def download_dataset(
    api_session: requests.Session,
    download_session: requests.Session,
    release_id: str,
    dataset: dict[str, Any],
    output_root: Path,
    overwrite: bool,
    num_workers: int,
    reverse_files: bool = False,
) -> None:
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")

    dataset_name = dataset.get("name")
    if not isinstance(dataset_name, str):
        raise ValueError(f"Dataset entry is missing a string name: {dataset!r}")

    details = fetch_dataset_details(api_session, release_id, dataset_name)
    files = details.get("files")
    if not isinstance(files, list):
        raise ValueError(f"Dataset {dataset_name} did not return a files list")

    dataset_dir = output_root / safe_dataset_folder_name(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    description = str(details.get("description") or dataset.get("description") or "")
    save_description(dataset_dir, description)

    indexed_files = []
    for index, url in enumerate(files, start=1):
        if not isinstance(url, str):
            raise ValueError(f"Dataset {dataset_name} has a non-string file URL")
        indexed_files.append((index, url))
    if reverse_files:
        indexed_files.reverse()

    write_status(f"Dataset {dataset_name}: {len(files)} file(s)")
    progress = None
    bytes_progress = None
    bytes_progress_lock = threading.Lock()
    if tqdm is not None:
        progress = tqdm(
            total=len(files),
            unit="file",
            desc=f"{dataset_name} files",
            leave=True,
            position=0,
            file=sys.stdout,
            dynamic_ncols=True,
        )
        if num_workers > 1 and len(indexed_files) > 1:
            bytes_progress = tqdm(
                total=None,
                unit="B",
                unit_scale=True,
                desc=f"{dataset_name} bytes",
                leave=False,
                position=1,
                file=sys.stdout,
                dynamic_ncols=True,
            )

    def update_bytes_progress(byte_count: int) -> None:
        if bytes_progress is None:
            return
        with bytes_progress_lock:
            bytes_progress.update(byte_count)

    try:
        if num_workers == 1 or len(indexed_files) <= 1:
            for index, url in indexed_files:
                download_dataset_file(
                    download_session,
                    url,
                    index,
                    dataset_dir,
                    overwrite,
                    show_progress=True,
                )
                if progress is not None:
                    progress.update(1)
            return

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    download_dataset_file,
                    None,
                    url,
                    index,
                    dataset_dir,
                    overwrite,
                    False,
                    update_bytes_progress,
                )
                for index, url in indexed_files
            ]
            for future in as_completed(futures):
                future.result()
                if progress is not None:
                    progress.update(1)
    finally:
        if bytes_progress is not None:
            bytes_progress.close()
        if progress is not None:
            progress.close()


def run_download(
    reverse_datasets: bool = False,
    reverse_files: bool = False,
) -> None:
    args = build_parser().parse_args()
    api_key = get_api_key()

    api_session = requests.Session()
    api_session.headers.update({"x-api-key": api_key})
    download_session = requests.Session()

    release_id = resolve_release_id(api_session, args.release_id)
    datasets = fetch_release_datasets(api_session, release_id)
    write_status(f"Release {release_id}: {len(datasets)} dataset(s)")
    datasets = filter_requested_datasets(datasets, args.datasets)
    if args.datasets is not None:
        write_status(f"Selected {len(datasets)} dataset(s): {', '.join(args.datasets)}")
    if reverse_datasets:
        write_status("Downloading datasets in reverse order")
    if reverse_files:
        write_status("Downloading files within each dataset in reverse order")

    for dataset in ordered_datasets(datasets, reverse_datasets):
        download_dataset(
            api_session=api_session,
            download_session=download_session,
            release_id=release_id,
            dataset=dataset,
            output_root=args.output_root,
            overwrite=args.overwrite,
            num_workers=args.num_workers,
            reverse_files=reverse_files,
        )


def main() -> None:
    run_download()


if __name__ == "__main__":
    main()
